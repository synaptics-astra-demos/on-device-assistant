import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Iterable

import numpy as np


class BaseTranslationModel(ABC):

    def __init__(self):
        self._transcribe_times = deque(maxlen=100)
        self._infer_stats = {}

    @property
    def last_infer_stats(self) -> dict:
        return self._infer_stats
    
    @property
    def last_infer_time(self) -> float:
        return self._transcribe_times[-1] if self._transcribe_times else 0
    
    @property
    def n_infer(self) -> int:
        return len(self._transcribe_times)
    
    @property
    def total_infer_time(self) -> float:
        return sum(self._transcribe_times)
    
    @property
    def avg_infer_time(self) -> float:
        return (self.total_infer_time / self.n_infer) if self._transcribe_times else 0

    @abstractmethod
    def _tokenize(self, text: str) -> dict[str, np.ndarray]:
        ...

    @abstractmethod
    def _generate(self, inputs: dict[str, np.ndarray], max_len: int | None = None) -> list[int]:
        ...

    @abstractmethod
    def _decode_tokens(self, tokens: Iterable[int], skip_special_tokens: bool) -> str:
        ...

    @abstractmethod
    def cleanup(self):
        ...

    def translate(self, text: str) -> str:
        self._infer_stats = {}
        st = time.time()
        inputs = self._tokenize(text)
        self._infer_stats["input_size"] = inputs["input_ids"].shape[-1]
        tokens = self._generate(inputs)
        text = self._decode_tokens(tokens, skip_special_tokens=True)
        et = time.time()
        self._transcribe_times.append(et - st)
        return text
