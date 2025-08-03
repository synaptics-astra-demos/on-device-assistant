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

MODELS_DIR = os.getenv("MODELS", "./models")

logger = logging.getLogger(__name__)


def minilm_factory(
    model_name: str,
    *,
    eager_load: bool = True,
    normalize: bool = False,
    n_threads: int | None = None
) -> MiniLMLlama | MiniLMSynap:
    if model_name not in MODEL_CHOICES:
        raise ValueError(f"Invalid model '{model_name}', please use one of {MODEL_CHOICES}")
    model_type, quant_type = model_name.split("-")
    if model_type == "llama":
        return MiniLMLlama(
            quant_type,
            n_threads=n_threads,
            normalize=normalize
        )
    return MiniLMSynap(
        quant_type,
        normalize=normalize
    )


class TextEmbeddingsAgent:
    def __init__(
        self,
        model_name: str,
        qa_file: str, 
        *,
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
            n_threads=n_threads
        )
        self.cache_dir = Path(cache_root) / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.qa_embeddings = self.load_embeddings(self.embedding_model)

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
