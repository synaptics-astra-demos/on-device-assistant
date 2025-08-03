from abc import ABC, abstractmethod
from collections import deque


class BaseEmbeddingsModel(ABC):

    def __init__(
        self,
        normalize: bool
    ):
        self.normalize = normalize
        self._infer_times = deque(maxlen=100)
    
    @property
    def last_infer_time(self) -> float | None:
        return self._infer_times[-1] if self._infer_times else None
    
    @property
    def n_infer(self) -> int:
        return len(self._infer_times)
    
    @property
    def total_infer_time(self) -> float:
        return sum(self._infer_times)
    
    @property
    def avg_infer_time(self) -> float | None:
        return (self.total_infer_time / self.n_infer) if self._infer_times else None
    
    @abstractmethod
    def generate(self, text: str) -> list[float]:
        ...
