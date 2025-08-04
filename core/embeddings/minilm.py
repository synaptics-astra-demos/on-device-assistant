import logging
import time
from itertools import product
from typing import Final, Literal

import numpy as np
from llama_cpp import Llama
from tokenizers import Tokenizer, Encoding

from .base import BaseEmbeddingsModel
from ..utils.download import download_from_hf
from ..inference.runners import SynapInferenceRunner

MODEL_TYPES: Final = ["llama", "synap"]
QUANT_TYPES: Final = ["float", "quantized"]
MODEL_CHOICES: Final = [f"{t}-{q}" for (t, q) in product(MODEL_TYPES, QUANT_TYPES)]

logger = logging.getLogger(__name__)


class MiniLMLlama(BaseEmbeddingsModel):

    def __init__(
        self,
        quant_type: Literal["float", "quantized"],
        *,
        eager_load: bool = True,
        normalize: bool = False,
        n_threads: int | None = None
    ):
        super().__init__(normalize=normalize, eager_load=eager_load)
        model_name = "all-MiniLM-L6-v2-Q8_0.gguf" if quant_type == "quantized" else "all-MiniLM-L6-v2-ggml-model-f16.gguf"
        self.model_path = download_from_hf("second-state/All-MiniLM-L6-v2-Embedding-GGUF", model_name)
        self.n_threads = n_threads
        self.model = None

        if self.eager_load:
            self._load_model()
        else:
            logger.warning("%s: Eager loading disabled, initial inference will be slower", self.__class__.__name__)
    
    def _load_model(self):
        if self.model is None:
            self.model = Llama(
                model_path=str(self.model_path),
                n_threads=self.n_threads,
                n_threads_batch=self.n_threads,
                embedding=True,
                verbose=False
            )

    def generate(self, text: str) -> list[float]:
        self._load_model()
        st = time.time()
        embedding = self.model.embed(text, normalize=self.normalize)
        et = time.time()
        self._infer_times.append(et - st)
        if embedding is None:
            raise ValueError("No embedding returned")
        return embedding

    def cleanup(self):
        self.model = None


class MiniLMSynap(BaseEmbeddingsModel):

    def __init__(
        self,
        quant_type: Literal["float", "quantized"],
        *,
        eager_load: bool = True,
        normalize: bool = False,
        hf_repo: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        super().__init__(normalize=normalize, eager_load=eager_load)
        model_name = "model_quantized.synap" if quant_type == "quantized" else "model_float.synap"
        self.model = SynapInferenceRunner.from_uri(
            f"https://github.com/spal-synaptics/on-device-assistant/releases/download/models-v1/all-MiniLM-L6-v2-{quant_type}.synap",
            f"models/synap/all-MiniLM-L6-v2/{model_name}",
            eager_load=eager_load
        )
        self.tokenizer: Tokenizer = Tokenizer.from_file(download_from_hf(repo_id=hf_repo, filename="tokenizer.json"))

        token_dims = sorted([inp.shape[1] for inp in self.model.inputs_info])
        if len(set(token_dims)) > 1:
            logger.warning("Multiple dimensions found for token len, selecting the largest")
        self.token_len = token_dims[-1]
        self.tokenizer.enable_truncation(self.token_len)
        self.tokenizer.enable_padding(length=self.token_len)

        if not self.eager_load:
            logger.warning("%s: Eager loading disabled, initial inference will be slower", self.__class__.__name__)

    @staticmethod
    def mean_pooling(
        token_embeddings: np.ndarray,
        attention_mask: np.ndarray
    ) -> np.ndarray:
        mask_expanded = attention_mask[..., None]
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    @staticmethod
    def normalize_embeds(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, a_min=1e-9, a_max=None)

    def _get_input_tokens(self, input: str) -> dict[str, np.ndarray]:
        enc: Encoding = self.tokenizer.encode(input, add_special_tokens=True)
        return {
            "input_ids": np.asarray(enc.ids, dtype=np.int32)[np.newaxis, :],
            "attention_mask": np.asarray(enc.attention_mask, dtype=np.int32)[np.newaxis, :],
            "token_type_ids": np.asarray(enc.type_ids, dtype=np.int32)[np.newaxis, :]
        }

    def generate(self, text: str) -> list[float]:
        st = time.time()

        tokens = self._get_input_tokens(text)
        attn_mask = tokens["attention_mask"]
        model_outputs = self.model.infer(tokens)
        token_embeddings = model_outputs[0]
        embeddings = self.mean_pooling(token_embeddings, attn_mask)
        if self.normalize:
            embeddings = self.normalize_embeds(embeddings)

        et = time.time()
        self._infer_times.append(et - st)
        return embeddings.squeeze(0).tolist()

    def cleanup(self):
        self.model.unload()


if __name__ == "__main__":
    pass