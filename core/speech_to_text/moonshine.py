import argparse
import logging
import time
from itertools import product
from typing import Literal, Final

import numpy as np
import soundfile as sf

from .base import BaseSpeechToTextModel
from ..inference.runners import OnnxInferenceRunner, SynapInferenceRunner

MODEL_TYPES: Final = ["onnx", "synap"]
MODEL_SIZES: Final = ["tiny", "base"]
QUANT_TYPES: Final = ["float", "quantized"]
MODEL_CHOICES: Final = [f"{t}-{s}-{q}" for (t, s, q) in product(MODEL_TYPES, MODEL_SIZES, QUANT_TYPES)]
logger = logging.getLogger(__name__)


class MoonshineSynap(BaseSpeechToTextModel):

    def __init__(
        self,
        *,
        hf_repo: str = "UsefulSensors/moonshine",
        model_size: Literal["base", "tiny"] = "tiny",
        quant_type: Literal["float", "quantized"] = "float",
        rate: int = 16_000,
        max_tok_per_s: int | None = None,
        n_threads: int | None = None,
        eager_load: bool = True,
    ):
        super().__init__(
            hf_repo,
            f"{hf_repo}-{model_size}",
            quant_type,
            rate
        )
        self.encoder_onnx = OnnxInferenceRunner.from_hf(
            hf_repo=hf_repo,
            filename=f"onnx/merged/{model_size}/float/encoder_model.onnx",
            n_threads=n_threads,
            eager_load=eager_load
        )
        encoder = SynapInferenceRunner.from_uri(
            url=f"https://github.com/spal-synaptics/on-device-assistant/releases/download/models-v1/moonshine_{model_size}_{self.quant_type}_encoder.synap",
            filename=f"models/synap/moonshine/{model_size}/{self.quant_type}/encoder.synap",
            eager_load=eager_load
        )
        self.decoder = SynapInferenceRunner.from_uri(
            url=f"https://github.com/spal-synaptics/on-device-assistant/releases/download/models-v1/moonshine_{model_size}_{self.quant_type}_decoder.synap",
            filename=f"models/synap/moonshine/{model_size}/{self.quant_type}/decoder.synap",
            eager_load=eager_load
        )
        self.decoder_with_past = SynapInferenceRunner.from_uri(
            url=f"https://github.com/spal-synaptics/on-device-assistant/releases/download/models-v1/moonshine_{model_size}_{self.quant_type}_decoder_with_past.synap",
            filename=f"models/synap/moonshine/{model_size}/{self.quant_type}/decoder_with_past.synap",
            eager_load=eager_load
        )
        self.encoder_pad_id: int = 0
        self.max_tok_per_s = max_tok_per_s
        self.cache_shapes: dict[str, list[int]] = {i.name: i.shape for i in self.decoder_with_past.inputs_info}
        self.max_inp_len: int = next(inp.shape for inp in encoder.inputs_info if inp.name == "input_values")[-1]
        encoder.unload()
        self.max_tokens: int = next(inp.shape for inp in self.decoder_with_past.inputs_info if "decoder" in inp.name)[2] # assuming shape [B, H, L, D]
        if isinstance(max_tok_per_s, int) and max_tok_per_s > 0:
            user_max_tokens: int = int(self.max_inp_len / 16_000) * max_tok_per_s
            if user_max_tokens > self.max_tokens:
                raise ValueError(f"Provided max tokens/sec ({max_tok_per_s}) is too high for model (max: {int(self.max_tokens / self.max_inp_len * 16_000)} tokens/sec)")
            self.max_tokens = user_max_tokens

        self.all_cache_tensors = [inp.name for inp in self.decoder_with_past.inputs_info if "past_key_values" in inp.name]
        self.dec_cache_tensors = [k for k in self.all_cache_tensors if "encoder" not in k]
        self.decoder_cache: dict[str, np.ndarray] = {}

        if not eager_load:
            logger.warning("%s: Eager loading disabled, initial inference will be slower", self.__class__.__name__)
    
    def _size_input(self, input: np.ndarray) -> np.ndarray:
        input = input.flatten()
        if len(input) > self.max_inp_len:
            logger.warning("MoonshineSynap: Truncating input from %d to %d samples", len(input), self.max_inp_len)
            input = input[:self.max_inp_len]
        elif len(input) < self.max_inp_len:
            logger.debug("MoonshineSynap: Padding input from %d to %d samples", len(input), self.max_inp_len)
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

    def _pad_cache_tensor(self, cache_tensor: np.ndarray, req_shape: list[int]) -> np.ndarray:
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
            self.decoder_cache[k] = self._pad_cache_tensor(v, self.cache_shapes[k])
    
    def _run_decoder(self, input_tokens: list[int], encoder_out: list[np.ndarray], *, seq_len: int) -> tuple[int, list[np.ndarray]]:
        input_ids = [input_tokens]
        if seq_len == 0:
            decoder_inputs = {
                "input_ids": np.asarray(input_ids, dtype=np.int32),
                "encoder_hidden_states": encoder_out
            }
            logits, *cache = self.decoder.infer(decoder_inputs)
            self._infer_stats["decoder_infer_time_ms"] = self.decoder.infer_time_ms
        else:
            decoder_inputs = {
                "input_ids": np.asarray(input_ids, dtype=np.int32),
                **self.decoder_cache
            }
            decoder_inputs["current_len"] = np.array([[seq_len]], dtype=np.int32)
            logits, *cache = self.decoder_with_past.infer(decoder_inputs)
            if not self._infer_stats.get("decoder_with_past_infer_time_ms"):
                self._infer_stats["decoder_with_past_infer_time_ms"] = 0
            self._infer_stats["decoder_with_past_infer_time_ms"] += self.decoder_with_past.infer_time_ms
        next_token = logits[0, -1].argmax().item()
        return next_token, cache

    def _generate(self, audio: np.ndarray, max_len: int | None = None) -> np.ndarray:
        max_len = max_len if isinstance(max_len, int) and max_len < self.max_tokens else self.max_tokens
        
        self._init_cache()
        next_token = self.decoder_start_token_id
        tokens = [next_token]
        input = self._size_input(audio).astype(np.float16)
        self._infer_stats["input_size"] = input.shape[-1]

        encoder_out = self.encoder_onnx.infer({"input_values": input.astype(np.float32)})[0].astype(np.float16)
        self._infer_stats["encoder_infer_time_ms"] = self.encoder_onnx.infer_time_ms

        next_token, init_cache = self._run_decoder(tokens, encoder_out, seq_len=0)
        self._update_cache(init_cache, update_all=True)
        tokens.append(next_token)

        for i in range(1, max_len):
            next_token, cache = self._run_decoder([next_token], encoder_out, seq_len=i)
            self._update_cache(cache)
            tokens.append(next_token)
            if next_token == self.eos_token_id:
                break
        self._infer_stats["decoder_tokens"] = i
        return np.array([tokens])

    def cleanup(self):
        self.encoder_onnx.unload()
        self.decoder.unload()
        self.decoder_with_past.unload()


class MoonshineOnnx(BaseSpeechToTextModel):

    def __init__(
        self,
        *,
        hf_repo: str = "UsefulSensors/moonshine",
        model_size: Literal["base", "tiny"] = "base",
        quant_type: Literal["float", "quantized"] = "quantized",
        rate: int = 16_000,
        n_threads: int | None = None,
        eager_load: bool = True,
    ):
        super().__init__(
            hf_repo,
            f"{hf_repo}-{model_size}",
            quant_type,
            rate
        )
        self.encoder_session = OnnxInferenceRunner.from_hf(
            hf_repo=hf_repo,
            filename=f"onnx/merged/{model_size}/{self.quant_type}/encoder_model.onnx",
            n_threads=n_threads,
            eager_load=eager_load
        )
        self.decoder_session = OnnxInferenceRunner.from_hf(
            hf_repo=hf_repo,
            filename=f"onnx/merged/{model_size}/{self.quant_type}/decoder_model_merged.onnx",
            n_threads=n_threads,
            eager_load=eager_load
        )
        if eager_load: # warm-up only if eager loading since warm-up loads & keeps model in memory
            self.transcribe(np.zeros(rate, dtype=np.float32))
        else:
            logger.warning("%s: Eager loading disabled, initial inference will be slower", self.__class__.__name__)

    def _generate(self, audio: np.ndarray, max_len: int | None = None) -> np.ndarray:
        if max_len is None:
            max_len = min((audio.shape[-1] // self.rate) * 6, self.max_len)
        self._infer_stats["input_size"] = audio.shape[-1]
        enc_out = self.encoder_session.infer({"input_values": audio})[0]
        self._infer_stats["encoder_infer_time_ms"] = self.encoder_session.infer_time_ms

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

        dec_time = 0
        for i in range(max_len):
            use_cache_branch = i > 0
            dec_inputs = {
                "input_ids": gen_tokens[:, -1:],
                "encoder_hidden_states": enc_out,
                "use_cache_branch": [use_cache_branch],
                **past_kv,
            }
            out = self.decoder_session.infer(dec_inputs)
            dec_time += self.decoder_session.infer_time_ms
            logits = out[0]
            present_kv = out[1:]
            next_tokens = logits[:, -1].argmax(axis=-1, keepdims=True)
            for j, key in enumerate(past_kv):
                if not use_cache_branch or "decoder" in key:
                    past_kv[key] = present_kv[j]
            gen_tokens = np.concatenate([gen_tokens, next_tokens], axis=-1)
            if (next_tokens == self.eos_token_id).all():
                break

        self._infer_stats["decoder_infer_time_ms"] = dec_time
        self._infer_stats["decoder_tokens"] = i
        return gen_tokens

    def cleanup(self):
        self.encoder_session.unload()
        self.decoder_session.unload()


def main():
    import json
    from pathlib import Path

    def _dump_results_json(results: dict, path: Path):
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {path}")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model",
        type=str,
        metavar="MODEL",
        required=True,
        choices=MODEL_CHOICES,
        help="Moonshine model to use for transcription (available:\n%(choices)s)"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        metavar="WAV",
        nargs="+",
        required=True,
        help="Input WAV audio for inference"
    )
    parser.add_argument(
        "-o", "--dump-out",
        type=str,
        metavar="JSON",
        help="Dump inference results to JSON file",
    )
    parser.add_argument(
        "-j", "--threads",
        type=int,
        help="Number of cores to use for CPU execution (default: all)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Display verbose stats"
    )
    args = parser.parse_args()

    model_type, model_size, model_quant = args.model.split("-")
    if model_type == "onnx":
        print("Loading Moonshine model using ONNX runtime ...")
        stt = MoonshineOnnx(
            model_size=model_size, quant_type=model_quant, n_threads=args.threads, eager_load=True
        )
    elif model_type == "synap":
        print("Loading Moonshine model using SyNAP runtime ...")
        stt = MoonshineSynap(
            model_size=model_size, quant_type=model_quant, n_threads=args.threads, eager_load=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}, supported types are {MODEL_TYPES}")

    all_results = {"model": args.model, "results": []}
    for audio_path in args.input:
        print()
        data, _ = sf.read(audio_path, dtype="float32")
        result = stt.transcribe(data)
        print(f"Transcribed ({stt.last_infer_time * 1000:.2f} ms): \"{result}\"")
        curr_results = {
            "input": audio_path,
            "total_infer_time_ms": stt.last_infer_time * 1000,
            "transcription": result,
        }
        if (expected := Path(f"{audio_path}.txt")).exists():
            expected_text = expected.read_text().strip()
            print(f"Expected: \"{expected_text}\"")
            curr_results["expected_transcription"] = expected_text
        if args.verbose:
            print(f"Detailed inference stats:\n{json.dumps(stt.last_infer_stats, indent=2)}")
            curr_results["detailed_infer_stats"] = stt.last_infer_stats
        all_results["results"].append(curr_results)
        print()
    if args.dump_out:
        _dump_results_json(all_results, Path(args.dump_out))


if __name__ == "__main__":
    import logging
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    main()
