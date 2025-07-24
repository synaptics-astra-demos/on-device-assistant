import os
import time
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort

from transformers import AutoTokenizer, AutoConfig


class BaseTranslationModel(ABC):

    def __init__(
        self,
        hf_repo: str,
        max_inp_len: int | None,
        max_tokens: int | None
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(hf_repo)
        self.config = AutoConfig.from_pretrained(hf_repo)
        self.max_inp_len: int = max_inp_len
        self.max_tokens: int = max_tokens or self.config.max_length
    
        self._transcribe_times = deque(maxlen=100)
        self._infer_stats = {}

    @property
    def last_infer_stats(self) -> dict:
        return self._infer_stats
    
    @property
    def last_infer_time(self) -> float | None:
        return self._transcribe_times[-1] if self._transcribe_times else None
    
    @property
    def n_infer(self) -> int:
        return len(self._transcribe_times)
    
    @property
    def total_infer_time(self) -> float:
        return sum(self._transcribe_times)
    
    @property
    def avg_infer_time(self) -> float | None:
        return (self.total_infer_time / self.n_infer) if self._transcribe_times else None

    def _tokenize(self, input: str) -> dict[str, np.ndarray]:
        tokenizer_params = {
            "return_tensors": "np"
        }
        if isinstance(self.max_inp_len, int):
            tokenizer_params.update({
                "max_length": self.max_inp_len,
                "padding": "max_length",
                "truncation": True
            })
        inputs = self.tokenizer(
            input,
            **tokenizer_params
        )
        self._infer_stats["input_size"] = inputs["input_ids"].shape[-1]
        return dict(inputs)


class InferenceRunner:

    def __init__(
        self,
        encoder_model: str | os.PathLike | ort.InferenceSession,
        decoder_model: str | os.PathLike | ort.InferenceSession,
        decoder_with_past_model: str | os.PathLike | ort.InferenceSession,
        hf_repo: str,
        *,
        max_inp_len: int | None,
        max_tokens: int | None,
        is_static: bool
    ):
        self.encoder = self._load_infer_sess(encoder_model)
        self.decoder = self._load_infer_sess(decoder_model)
        self.decoder_with_past = self._load_infer_sess(decoder_with_past_model)
        self.max_inp_len = max_inp_len
        self.max_tokens = max_tokens
        self.is_static = is_static
        if self.is_static and not isinstance(self.max_inp_len, int):
            raise ValueError("`max_inp_len` must be a positive integer for static inference")
        if self.is_static and not isinstance(self.max_tokens, int):
            raise ValueError("`max_tokens` must be a positive integer for static inference")
        if isinstance(self.max_inp_len, int) and self.max_inp_len <= 0:
            raise ValueError("`max_inp_len` must be a positive integer or None")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_repo)

        self.config = AutoConfig.from_pretrained(hf_repo)
        self.n_layers: int = int(self.config.num_hidden_layers)
        self.n_kv_heads: int = int(self.config.decoder_attention_heads)
        self.hidden_size: int = int(self.config.d_model)
        self.head_dim: int = self.hidden_size // self.n_kv_heads
        self.start_token_id: int = self.config.pad_token_id
        self.end_token_id: int = self.config.eos_token_id
        self.encoder_pad_id: int = self.config.pad_token_id

        self.max_tokens: int = max_tokens or self.config.max_length
        self.n_tokens_gen: int = 0
        self.infer_times: list[int] = []
        self.decoder_cache: dict[str, np.ndarray] = {}
        self.cache_shapes: dict[str, tuple[int, ...]] = {
            inp.name: inp.shape for inp in self.decoder_with_past.get_inputs() if "past_key_values" in inp.name
        }
        self.all_cache_tensors: list[str] = [
            k for k in self.cache_shapes if "past_key_values" in k
        ]
        self.dec_cache_tensors: list[str] = [
            k for k in self.all_cache_tensors if "encoder" not in k
        ]

        self._transcribe_times = deque(maxlen=100)
        self._infer_stats = {}

    @property
    def avg_infer_time(self) -> float:
        if len(self.infer_times) == 0:
            return 0
        return sum(self.infer_times) / len(self.infer_times)

    @staticmethod
    def _load_infer_sess(model: str | os.PathLike | ort.InferenceSession) -> ort.InferenceSession:
        if isinstance(model, ort.InferenceSession):
            return model
        if not Path(model).exists():
            raise FileNotFoundError(f"Model file not found: {model}")
        return ort.InferenceSession(model)
    
    def _init_cache(self):
        self.decoder_cache.update({
            f"past_key_values.{i}.{a}.{b}": np.zeros(
                (1, self.n_kv_heads, 1, self.head_dim), dtype=np.float32
            )
            for i in range(self.n_layers)
            for a in ("decoder", "encoder")
            for b in ("key", "value")
        })

    def _tokenize(self, input: str) -> dict[str, np.ndarray]:
        tokenizer_params = {
            "return_tensors": "np"
        }
        if isinstance(self.max_inp_len, int):
            tokenizer_params.update({
                "max_length": self.max_inp_len,
                "padding": "max_length",
                "truncation": True
            })
        inputs = self.tokenizer(
            input,
            **tokenizer_params
        )
        return dict(inputs)

    @staticmethod
    def _pad_cache_tensor(cache_tensor: np.ndarray, req_shape: tuple[int, ...]) -> np.ndarray:
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
        return cache_padded

    def _update_cache(self, new_values: list[np.ndarray], *, update_all: bool = False):
        cache_tensors = self.all_cache_tensors if update_all else self.dec_cache_tensors
        if len(cache_tensors) != len(new_values):
            raise ValueError(f"Cache tensors mismatch: expected {len(cache_tensors)} new values, got {len(new_values)}")
        for k, v in zip(cache_tensors, new_values):
            if self.is_static:
                self.decoder_cache[k] = self._pad_cache_tensor(v, self.cache_shapes[k])
            else:
                self.decoder_cache[k] = v

    def _next_token(self, logits: np.ndarray) -> int:
        step_logits = logits[0, -1]
        step_logits[self.encoder_pad_id] = -1e9
        return int(step_logits.argmax())

    def _run_decoder(self, input_tokens: list[int], attn_mask: np.ndarray, encoder_out: np.ndarray, *, seq_len: int) -> tuple[int, list[np.ndarray]]:
        input_ids = [input_tokens]
        decoder_inputs = {
            "input_ids": input_ids,
            "encoder_attention_mask": attn_mask,
        }
        if seq_len == 0:
            decoder_inputs["encoder_hidden_states"] = encoder_out
            logits, *cache = self.decoder.run(None, decoder_inputs)
        else:
            if self.is_static:
                decoder_inputs["current_len"] = np.array([[seq_len]], dtype=np.int64)
            decoder_inputs.update(self.decoder_cache)
            logits, *cache = self.decoder_with_past.run(None, decoder_inputs)
        return self._next_token(logits), cache

    def run(
        self, 
        text: str, 
        max_tokens: int | None = None, 
    ):
        self.n_tokens_gen = 0
        self.infer_times.clear()
        if isinstance(max_tokens, int) and max_tokens < self.max_tokens:
            self.max_tokens = max_tokens

        inputs = self._tokenize(text)
        self._init_cache()
        st = time.time()
        next_token = self.start_token_id
        tokens = [next_token]
        encoder_out = self.encoder.run(None, inputs)[0]

        for i in range(self.max_tokens):
            next_token, cache = self._run_decoder([tokens[-1]], inputs["attention_mask"], encoder_out, seq_len=i)
            self._update_cache(cache, update_all=i==0)

            self.n_tokens_gen += 1
            tokens.append(next_token)
            if next_token == self.end_token_id:
                break

        self.infer_times.append(time.time() - st)
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    @classmethod
    def dynamic_from(cls, model_dir: str | os.PathLike, hf_repo: str, *, max_inp_len: int | None = None):
        return cls(
            encoder_model=Path(model_dir) / "encoder_model.onnx",
            decoder_model=Path(model_dir) / "decoder_model.onnx",
            decoder_with_past_model=Path(model_dir) / "decoder_with_past_model.onnx",
            hf_repo=hf_repo,
            max_inp_len=max_inp_len,
            max_tokens=None,
            is_static=False
        )
    
    @classmethod
    def static_from(cls, model_dir: str | os.PathLike, hf_repo: str):
        encoder_sess = ort.InferenceSession(Path(model_dir) / "encoder_static.onnx")
        decoder_sess = ort.InferenceSession(Path(model_dir) / "decoder_static.onnx")
        decoder_with_past_sess = ort.InferenceSession(Path(model_dir) / "decoder_with_past_static.onnx")
        max_inp_len = next(
            inp.shape for inp in encoder_sess.get_inputs() if inp.name == "input_ids"
        )[-1]
        max_tokens = next(
            inp.shape for inp in decoder_with_past_sess.get_inputs() if "decoder" in inp.name
        )[2] # assuming shape [B, H, L, D]
        return cls(
            encoder_model=encoder_sess,
            decoder_model=decoder_sess,
            decoder_with_past_model=decoder_with_past_sess,
            hf_repo=hf_repo,
            max_inp_len=max_inp_len,
            max_tokens=max_tokens,
            is_static=True
        )


if __name__ == "__main__":
    pass
