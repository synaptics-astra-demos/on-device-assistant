import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseEmbeddingsModel
from .minilm import MiniLMLlama, MiniLMSynap

MODELS_DIR = os.getenv("MODELS", "./models")

logger = logging.getLogger(__name__)


def minilm_factory(cpu_only: bool = False, n_threads: int | None = None) -> MiniLMLlama | MiniLMSynap:
    model_path = Path(f"{MODELS_DIR}/gguf/all-MiniLM-L6-v2-Q8_0.gguf")
    if not cpu_only and (synap_model := Path(f"{MODELS_DIR}/synap/all-MiniLM-L6-v2.synap")).exists():
        model_path = synap_model
    if not model_path.exists():
        raise FileNotFoundError(f"MiniLM model {model_path} doesn't exist")

    if model_path.suffix == ".synap":
        return MiniLMSynap(
            "SyNAP",
            model_path,
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    elif model_path.suffix == ".gguf":
        return MiniLMLlama(
            "Llama (Q8_0)",
            model_path,
            n_threads=n_threads
        )
    else:
        raise ValueError(f"Unsupported model format '{model_path.suffix}' ({model_path})")


class TextEmbeddingsAgent:
    def __init__(
        self, 
        qa_file: str, 
        *, 
        cpu_only: bool = False, 
        cpu_cores: int | None = None, 
        cache_root: str | os.PathLike = "./.cache"
    ):
        with open(qa_file, "r") as f:
            self.qa_pairs = json.load(f)
        self.embedding_models = [minilm_factory(cpu_only, cpu_cores)]
        self.cache_dir = Path(cache_root) / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.qa_embeddings = {model.model_name: self.load_embeddings(model) for model in self.embedding_models}

    def load_embeddings(self, model: BaseEmbeddingsModel, *, force_regenerate: bool = False) -> np.ndarray:
        import hashlib

        def _cache_checksum() -> str:
            m = hashlib.sha256()
            for text in texts:
                m.update(text.encode('utf-8'))
            m.update((model.model_name + str(model.model_path)).encode('utf-8'))
            return m.hexdigest()

        if not isinstance(model, BaseEmbeddingsModel):
            raise RuntimeError(f"Unsupported embeddings model: {model}")
        texts: list[str] = [pair["question"] + " " + pair["answer"] for pair in self.qa_pairs]

        cached: Path = self.cache_dir / f"{_cache_checksum()}.npy"
        if cached.exists() and not force_regenerate:
            logger.debug(f"Loaded cached embeddings from '{cached}'")
            return np.load(cached)

        embeddings: list[float] = []
        for text in tqdm(texts, desc=f"Computing embeddings: {model}"):
            embeddings.append(model.generate(text))
        embeddings_np: np.ndarray = np.array(embeddings)
        np.save(cached, embeddings_np)
        logger.debug(f"Cached embeddings at '{cached}'")
        return embeddings_np
    
    def embed_query(self, query: str, model: BaseEmbeddingsModel) -> tuple[int, np.ndarray]:
        query_emb = model.generate(query)
        sims = cosine_similarity([query_emb], self.qa_embeddings[model.model_name]).flatten()
        return np.argmax(sims), sims

    def answer_query(self, query: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for model in self.embedding_models:
            best_idx, sims = self.embed_query(query, model)
            results.append({
                "model": model.model_name,
                "answer": self.qa_pairs[best_idx]["answer"],
                "similarity": float(sims[best_idx]),
                "infer_time": model.last_infer_time,
            })
        return results
