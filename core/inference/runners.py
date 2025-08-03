import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import time_ns
from typing import Any

import numpy as np
import onnxruntime as ort
from synap import Network

from ..utils.download import download_from_hf, download_from_url


@dataclass(frozen=True)
class IOTensorInfo:
    name: str
    shape: list[int | str]
    dtype: Any


class InferenceRunner(ABC):

    def __init__(
        self,
        model_path: str | os.PathLike,
        *,
        eager_load: bool
    ):
        self._model_path = str(model_path)
        self._eager_load = eager_load
        self._infer_time_ms: float = 0.0

    @classmethod
    def from_hf(cls, hf_repo: str, filename: str | os.PathLike, **kwargs):
        model_path = download_from_hf(hf_repo, filename)
        return cls(model_path, **kwargs)

    @classmethod
    def from_uri(cls, url: str, filename: str | os.PathLike, **kwargs):
        model_path = download_from_url(url, filename)
        return cls(model_path, **kwargs)

    @property
    def infer_time_ms(self) -> float:
        return self._infer_time_ms

    @property
    @abstractmethod
    def inputs_info(self) -> list[IOTensorInfo]:
        ...

    @property
    @abstractmethod
    def outputs_info(self) -> list[IOTensorInfo]:
        ...

    @abstractmethod
    def _infer(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        ...

    @abstractmethod
    def unload(self):
        ...

    def infer(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        st = time_ns()
        results = self._infer(inputs)
        self._infer_time_ms = (time_ns() - st) / 1e6
        return results


class OnnxInferenceRunner(InferenceRunner):

    def __init__(
        self,
        model_path: str | os.PathLike,
        *,
        n_threads: int | None = None,
        eager_load: bool = True
    ):
        super().__init__(model_path, eager_load=eager_load)

        self._opts = ort.SessionOptions()
        if n_threads is not None:
            self._opts.intra_op_num_threads = n_threads
            self._opts.inter_op_num_threads = n_threads
        sess = ort.InferenceSession(self._model_path, self._opts, providers=['CPUExecutionProvider'])
        try:
            self._inputs_info = [
                IOTensorInfo(i.name, i.shape, i.type) for i in sess.get_inputs()
            ]
            self._outputs_info = [
                IOTensorInfo(o.name, o.shape, o.type) for o in sess.get_outputs()
            ]
        finally:
            if not self._eager_load:
                sess = None
        self._sess: ort.InferenceSession | None = sess

    @property
    def inputs_info(self) -> list[IOTensorInfo]:
        return self._inputs_info

    @property
    def outputs_info(self) -> list[IOTensorInfo]:
        return self._outputs_info

    def _infer(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        if self._sess is None:
            self._sess = ort.InferenceSession(self._model_path, self._opts, providers=['CPUExecutionProvider'])
        return [np.asarray(o) for o in self._sess.run(None, inputs)]

    def unload(self):
        self._sess = None


class SynapInferenceRunner(InferenceRunner):

    def __init__(
        self,
        model_path: str | os.PathLike,
        *,
        eager_load: bool = True
    ):
        super().__init__(model_path, eager_load=eager_load)

        net = Network(self._model_path)
        try:
            self._inputs_info = [
                IOTensorInfo(i.name, list(i.shape), i.data_type.np_type()) for i in net.inputs
            ]
            self._outputs_info = [
                IOTensorInfo(o.name, list(o.shape), o.data_type.np_type()) for o in net.outputs
            ]
        finally:
            if not self._eager_load:
                net = None
        self._net: Network | None = net

    @property
    def inputs_info(self) -> list[IOTensorInfo]:
        return self._inputs_info

    @property
    def outputs_info(self) -> list[IOTensorInfo]:
        return self._outputs_info

    def _infer(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        if self._net is None:
            self._net = Network(self._model_path)
        for tensor in self._net.inputs:
            tensor.assign(np.asarray(inputs[tensor.name]).astype(tensor.data_type.np_type()))
        self._net.predict()
        return [o.to_numpy() for o in self._net.outputs]

    def unload(self):
        self._net = None
