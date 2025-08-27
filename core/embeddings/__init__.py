import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseEmbeddingsModel
from .minilm import MiniLMLlama, MiniLMSynap, MODEL_CHOICES
from ..utils.device import validate_cpu_only

MODELS_DIR = os.getenv("MODELS", "./models")

logger = logging.getLogger(__name__)


def minilm_factory(
    model_name: str,
    *,
    eager_load: bool = True,
    normalize: bool = False,
    n_threads: int | None = None,
    cpu_only: bool | None = None
) -> MiniLMLlama | MiniLMSynap:
    if model_name not in MODEL_CHOICES:
        raise ValueError(f"Invalid model '{model_name}', please use one of {MODEL_CHOICES}")
    model_type, quant_type = model_name.split("-")
    cpu_only = validate_cpu_only(cpu_only)
    if cpu_only or model_type == "onnx":
        return MiniLMLlama(
            quant_type,
            eager_load=eager_load,
            n_threads=n_threads,
            normalize=normalize
        )
    return MiniLMSynap(
        quant_type,
        eager_load=eager_load,
        normalize=normalize
    )


class TextEmbeddingsAgent:
    def __init__(
        self,
        model_name: str,
        qa_file: str, 
        *,
        cpu_only: bool | None = None,
        eager_load: bool = True,
        normalize: bool = False,
        n_threads: int | None = None,
        cache_root: str | os.PathLike = "./.cache"
    ):
        self.model_name = model_name
        self.qa_file = qa_file
        with open(qa_file, "r") as f:
            self.qa_pairs = json.load(f)
        self.embedding_model = minilm_factory(
            model_name,
            eager_load=eager_load,
            normalize=normalize,
            n_threads=n_threads,
            cpu_only=cpu_only
        )
        self.cache_dir = Path(cache_root) / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.qa_embeddings = self.load_embeddings(self.embedding_model)
        logger.debug("Initialized %s", str(self))

    def __repr__(self):
        return f"TextEmbeddingsAgent@{hex(id(self))}"

    def load_embeddings(self, model: BaseEmbeddingsModel, *, force_regenerate: bool = False) -> np.ndarray:
        import hashlib

        def _cache_checksum() -> str:
            m = hashlib.sha256()
            for text in texts:
                m.update(text.encode('utf-8'))
            m.update((str(self.model_name) + str(self.qa_file)).encode('utf-8'))
            return m.hexdigest()

        if not isinstance(model, BaseEmbeddingsModel):
            raise RuntimeError(f"Unsupported embeddings model: {model}")
        texts: list[str] = [pair["question"] + " " + pair["answer"] for pair in self.qa_pairs]

        cached: Path = self.cache_dir / f"{_cache_checksum()}.npy"
        if cached.exists() and not force_regenerate:
            logger.debug(f"Loaded cached embeddings from '{cached}'")
            return np.load(cached)

        embeddings: list[float] = []
        for text in tqdm(texts, desc=f"Computing embeddings: {self.model_name} @ '{self.qa_file}'"):
            embeddings.append(model.generate(text))
        embeddings_np: np.ndarray = np.array(embeddings)
        np.save(cached, embeddings_np)
        logger.debug(f"Cached embeddings at '{cached}'")
        return embeddings_np
    
    def embed_query(self, query: str, model: BaseEmbeddingsModel) -> tuple[int, np.ndarray]:
        query_emb = model.generate(query)
        sims = cosine_similarity([query_emb], self.qa_embeddings).flatten()
        return np.argmax(sims), sims

    def answer_query(self, query: str) -> dict[str, Any]:
        best_idx, sims = self.embed_query(query, self.embedding_model)
        return {
            "answer": self.qa_pairs[best_idx]["answer"],
            "similarity": float(sims[best_idx]),
            "infer_time": self.embedding_model.last_infer_time,
        }

    def cleanup(self):
        self.embedding_model.cleanup()
        logger.debug("Cleaned up %s", str(self))
