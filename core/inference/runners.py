import os
from abc import ABC, abstractmethod
from time import time_ns
from typing import Any

import numpy as np
import onnxruntime as ort
from synap import Network

from ..utils.download import download_from_hf, download_from_url


class InferenceRunner(ABC):

    def __init__(self, model_path: str | os.PathLike):
        self._model_path = str(model_path)
        self._infer_time_ms: float = 0.0

    @property
    def infer_time_ms(self) -> float:
        return self._infer_time_ms

    @property
    @abstractmethod
    def model(self) -> Any:
        ...

    @abstractmethod
    def _infer(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        ...

    def infer(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        st = time_ns()
        results = self._infer(inputs)
        self._infer_time_ms = (time_ns() - st) / 1e6
        return results

    @classmethod
    def from_hf(cls, hf_repo: str, filename: str | os.PathLike, **kwargs):
        model_path = download_from_hf(hf_repo, filename)
        return cls(model_path, **kwargs)

    @classmethod
    def from_uri(cls, url: str, filename: str | os.PathLike, **kwargs):
        model_path = download_from_url(url, filename)
        return cls(model_path, **kwargs)


class OnnxInferenceRunner(InferenceRunner):

    def __init__(
        self,
        model_path: str | os.PathLike,
        n_threads: int | None = None
    ):
        super().__init__(model_path)
        opts = ort.SessionOptions()
        if n_threads is not None:
            opts.intra_op_num_threads = n_threads
            opts.inter_op_num_threads = n_threads
        self._sess = ort.InferenceSession(self._model_path, opts, providers=['CPUExecutionProvider'])

    @property
    def model(self) -> ort.InferenceSession:
        return self._sess

    def _infer(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        return self._sess.run(None, inputs)


class SynapInferenceRunner(InferenceRunner):

    def __init__(
        self,
        model_path: str | os.PathLike
    ):
        super().__init__(model_path)
        self._net = Network(self._model_path)
    
    @property
    def model(self) -> Network:
        return self._net

    def _infer(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        inputs = list(inputs.values())
        for i, inp_tensor in enumerate(self._net.inputs):
            inp_tensor.assign(inputs[i].astype(inp_tensor.data_type.np_type()))
        return [o.to_numpy() for o in self._net.predict()]
