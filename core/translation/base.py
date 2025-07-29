from abc import ABC, abstractmethod
from collections import deque
import time

import numpy as np

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

    def _tokenize(self, text: str) -> dict[str, np.ndarray]:
        params = {"return_tensors": "np"}
        if isinstance(self.max_inp_len, int):
            params.update({"max_length": self.max_inp_len, "padding": "max_length", "truncation": True})
        return dict(self.tokenizer(text, **params))

    @abstractmethod
    def _generate(self, inputs: dict[str, np.ndarray], max_len: int | None = None) -> list[int]:
        ...
    
    def translate(self, text: str) -> str:
        self._infer_stats = {}
        st = time.time()
        inputs = self._tokenize(text)
        self._infer_stats["input_size"] = inputs["input_ids"].shape[-1]
        tokens = self._generate(inputs)
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        et = time.time()
        self._transcribe_times.append(et - st)
        return text
