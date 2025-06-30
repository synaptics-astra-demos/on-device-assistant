import os
import sys
import time
from typing import Literal

import numpy as np
import onnxruntime
import soundfile as sf
from synap import Network

from .base import BaseSpeechToTextModel
from ..utils.download import download_from_hf


class MoonshineSynap(BaseSpeechToTextModel):
    def __init__(
        self,
        encoder_model: str | os.PathLike, 
        decoder_uncached_model: str | os.PathLike, 
        decoder_cached_model: str | os.PathLike,
        *,
        hf_repo: str = "UsefulSensors/moonshine",
        model_size: Literal["base", "tiny"] = "tiny",
        rate: int = 16_000,
        max_tok_per_s: int | None = None
    ):
        super().__init__(
            hf_repo,
            f"{hf_repo}-{model_size}",
            rate
        )
        encoder_path = download_from_hf(
            repo_id=hf_repo,
            filename=f"onnx/merged/{model_size}/quantized/encoder_model.onnx",
        )
        # self.encoder = Network(str(encoder_model))
        self.encoder = onnxruntime.InferenceSession(encoder_path, providers=['CPUExecutionProvider'])
        self.decoder_uncached = Network(str(decoder_uncached_model))
        self.decoder_cached = Network(str(decoder_cached_model))
        self.encoder_pad_id: int = 0
        self.max_tok_per_s = max_tok_per_s
        self.cached_decoder_shapes: dict[str, list[int]] = {o.name: list(o.shape) for o in self.decoder_cached.inputs}
        # self.max_inp_len: int = next(inp.shape for inp in self.encoder.get_inputs() if inp.name == "input_values")[-1]
        self.max_inp_len: int = 80000
        self.max_tokens: int = next(inp.shape for inp in self.decoder_cached.inputs if "decoder" in inp.name)[2] # assuming shape [B, H, L, D]
        if isinstance(max_tok_per_s, int) and max_tok_per_s > 0:
            user_max_tokens: int = int(self.max_inp_len / 16_000) * max_tok_per_s
            if user_max_tokens > self.max_tokens:
                raise ValueError(f"Provided max tokens/sec ({max_tok_per_s}) is too high for model (max: {int(self.max_tokens / self.max_inp_len * 16_000)} tokens/sec)")
            self.max_tokens = user_max_tokens

        self.all_cache_tensors = [inp.name for inp in self.decoder_cached.inputs if "past_key_values" in inp.name]
        self.dec_cache_tensors = [k for k in self.all_cache_tensors if "encoder" not in k]
        self.decoder_cache: dict[str, np.ndarray] = {}

        self.transcribe(np.zeros(rate, dtype=np.float32))
    
    def _size_input(self, input: np.ndarray) -> np.ndarray:
        input = input.flatten()
        # input = input * 32768
        if len(input) > self.max_inp_len:
            # print(f"Truncating input from {len(input)} to {self.max_inp_len}")
            input = input[:self.max_inp_len]
        elif len(input) < self.max_inp_len:
            # print(f"Padding input from {len(input)} to {self.max_inp_len}")
            input = np.pad(input, (0, self.max_inp_len - len(input)), constant_values=self.encoder_pad_id)
        return input.reshape((1, self.max_inp_len)).astype(np.float16)

    def _init_cache(self):
        self.decoder_cache.update({
            f"past_key_values.{i}.{a}.{b}": np.zeros(
                (1, self.num_key_value_heads, 1, self.dim_kv), dtype=np.float32
            )
            for i in range(self.decoder_layers)
            for a in ("decoder", "encoder")
            for b in ("key", "value")
        })

    def _pad_cache_tensor(self, cache_tensor: np.ndarray, req_shape: tuple[int]) -> np.ndarray:
        if cache_tensor.shape == req_shape:
            return cache_tensor
        if cache_tensor.ndim != len(req_shape):
            raise ValueError(f"Invalid cache tensor dims: got {cache_tensor.ndim}, expected {len(req_shape)}")
        pad_width = []
        for cache_dim, req_dim in zip(cache_tensor.shape, req_shape):
            if cache_dim > req_dim:
                raise ValueError(f"Unexpected dim for cache tensor: {cache_tensor.shape}, expected: {req_shape}")
            before = 0
            after = req_dim - cache_dim
            pad_width.append((before, after))

        cache_padded = np.pad(cache_tensor, pad_width, mode='constant', constant_values=0)
        return cache_padded.astype(np.float16)

    def _update_cache(self, new_values: list[np.ndarray], *, update_all: bool = False):
        cache_tensors = self.all_cache_tensors if update_all else self.dec_cache_tensors
        if len(cache_tensors) != len(new_values):
            raise RuntimeError(f"Cache tensors mismatch: expected {len(cache_tensors)} new values, got {len(new_values)}")
        for k, v in zip(cache_tensors, new_values):
            self.decoder_cache[k] = self._pad_cache_tensor(v, self.cached_decoder_shapes[k])
    
    def _run_decoder(self, input_tokens: list[int], encoder_out: list[np.ndarray], *, seq_len: int) -> tuple[int, list[np.ndarray]]:
        input_ids = [input_tokens]
        if seq_len == 0:
            decoder_inputs = {
                "input_ids": np.asarray(input_ids, dtype=np.int32),
                "encoder_hidden_states": encoder_out
            }
            logits, *cache = [o.to_numpy() for o in self.decoder_uncached.predict(list(decoder_inputs.values()))]
            # for i, out in enumerate(self.decoder_uncached.outputs):
            #     np.save(f"temp_outputs/decoder_uncached-out-{i}.npy", out.to_numpy())
        else:
            decoder_inputs = {
                "input_ids": np.asarray(input_ids, dtype=np.int32),
                **self.decoder_cache
            }
            decoder_inputs["current_len"] = np.array([[seq_len]], dtype=np.int32)
            logits, *cache = [o.to_numpy() for o in self.decoder_cached.predict(list(decoder_inputs.values()))]
            # for i, out in enumerate(self.decoder_cached.outputs):
            #     np.save(f"temp_outputs/decoder_cached-out-{i}-seq_{seq_len}.npy", out.to_numpy())
        next_token = logits[0, -1].argmax().item()
        return next_token, cache

    def _generate(self, audio: np.ndarray, max_tokens: int | None = None) -> np.ndarray:
        max_tokens = max_tokens if isinstance(max_tokens, int) and max_tokens < self.max_tokens else self.max_tokens
        
        self._init_cache()
        next_token = self.decoder_start_token_id
        tokens = [next_token]
        input = self._size_input(audio).astype(np.float16)
        # np.save("temp_outputs/input.npy", input)

        # encoder_out = self.encoder.predict([input])[0].to_numpy().astype(np.float16)
        encoder_out = self.encoder.run(None, {"input_values": input.astype(np.float32)})[0].astype(np.float16)
        # np.save("temp_outputs/encoder_out.npy", encoder_out)
        # encoder_out = np.load("temp_inputs/encoder_out.npy").astype(np.float16)

        next_token, init_cache = self._run_decoder(tokens, encoder_out, seq_len=0)
        self._update_cache(init_cache, update_all=True)
        tokens.append(next_token)

        for i in range(max_tokens):
            next_token, cache = self._run_decoder([next_token], encoder_out, seq_len=i+1)
            self._update_cache(cache)
            tokens.append(next_token)
            if next_token == self.eos_token_id:
                break
        
        return np.array([tokens])


class MoonshineOnnx(BaseSpeechToTextModel):

    def __init__(
        self,
        *,
        hf_repo: str = "UsefulSensors/moonshine",
        model_size: Literal["base", "tiny"] = "base",
        rate: int = 16_000,
        n_threads: int | None = None
    ):
        super().__init__(
            hf_repo,
            f"{hf_repo}-{model_size}",
            rate
        )
        encoder_path = download_from_hf(
            repo_id=hf_repo,
            filename=f"onnx/merged/{model_size}/quantized/encoder_model.onnx",
        )
        decoder_path = download_from_hf(
            repo_id=hf_repo,
            filename=f"onnx/merged/{model_size}/quantized/decoder_model_merged.onnx",
        )
        opts = onnxruntime.SessionOptions()
        if n_threads is not None:
            opts.intra_op_num_threads = n_threads
            opts.inter_op_num_threads = n_threads
        self.encoder_session = onnxruntime.InferenceSession(encoder_path, opts, providers=['CPUExecutionProvider'])
        self.decoder_session = onnxruntime.InferenceSession(decoder_path, opts, providers=['CPUExecutionProvider'])
        
        self.transcribe(np.zeros(rate, dtype=np.float32))

    def _generate(self, audio: np.ndarray, max_len: int | None = None) -> np.ndarray:
        if max_len is None:
            max_len = min((audio.shape[-1] // self.rate) * 6, self.max_len)
        enc_out = self.encoder_session.run(None, {"input_values": audio})[0]
        batch_size = enc_out.shape[0]
        input_ids = np.array(
            [[self.decoder_start_token_id]] * batch_size, dtype=np.int64
        )
        past_kv = {
            f"past_key_values.{layer}.{mod}.{kv}": np.zeros(
                [batch_size, self.num_key_value_heads, 0, self.dim_kv], dtype=np.float32
            )
            for layer in range(self.decoder_layers)
            for mod in ("decoder", "encoder")
            for kv in ("key", "value")
        }
        gen_tokens = input_ids
        for i in range(max_len):
            use_cache_branch = i > 0
            dec_inputs = {
                "input_ids": gen_tokens[:, -1:],
                "encoder_hidden_states": enc_out,
                "use_cache_branch": [use_cache_branch],
                **past_kv,
            }
            out = self.decoder_session.run(None, dec_inputs)
            logits = out[0]
            present_kv = out[1:]
            next_tokens = logits[:, -1].argmax(axis=-1, keepdims=True)
            for j, key in enumerate(past_kv):
                if not use_cache_branch or "decoder" in key:
                    past_kv[key] = present_kv[j]
            gen_tokens = np.concatenate([gen_tokens, next_tokens], axis=-1)
            if (next_tokens == self.eos_token_id).all():
                break
        return gen_tokens


def main():
    if len(sys.argv) < 2:
        print("Usage: python speech_to_text.py <audio_file>")
        sys.exit(1)
    audio_path = sys.argv[1]
    data, sr = sf.read(audio_path, dtype="float32")
    print("Loading Moonshine model using ONNX runtime ...")
    stt = MoonshineOnnx()
    audio_ms = len(data) / sr * 1000
    print("Transcribing ...")
    start = time.time()
    text = stt.transcribe(data)
    end = time.time()
    transcribe_ms = (end - start) * 1000
    speed_factor = (audio_ms / 1000) / (end - start)
    print(f"audio sample time: {int(audio_ms)}ms")
    print(f"transcribe time:  {int(transcribe_ms)}ms")
    print(f"speed: {speed_factor:.1f}x")
    print(f"result: {text}")


if __name__ == "__main__":
    main()