from abc import ABC, abstractmethod
from collections import deque


class BaseEmbeddingsModel(ABC):

    def __init__(
        self,
        *,
        normalize: bool,
        eager_load: bool
    ):
        self.normalize = normalize
        self.eager_load = eager_load
        self._infer_times = deque(maxlen=100)
    
    @property
    def last_infer_time(self) -> float:
        return self._infer_times[-1] if self._infer_times else 0
    
    @property
    def n_infer(self) -> int:
        return len(self._infer_times)
    
    @property
    def total_infer_time(self) -> float:
        return sum(self._infer_times)
    
    @property
    def avg_infer_time(self) -> float | None:
        return (self.total_infer_time / self.n_infer) if self._infer_times else 0
    
    @abstractmethod
    def generate(self, text: str) -> list[float]:
        ...

    @abstractmethod
    def cleanup(self):
        ...
