import sys
import time
from typing import Literal

import numpy as np
import onnxruntime
import soundfile as sf

from .base import BaseSpeechToTextModel
from ..utils.download import download_from_hf


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

    def transcribe(self, speech: np.ndarray) -> str:
        st = time.time()
        speech = speech.astype(np.float32)[np.newaxis, :]
        tokens = self._generate(speech)
        text = self.tokenizer.decode_batch(tokens, skip_special_tokens=True)[0]
        et = time.time()
        self._transcribe_times.append(et - st)
        return text


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