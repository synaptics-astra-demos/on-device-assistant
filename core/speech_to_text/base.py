import json
from abc import ABC, abstractmethod
from collections import deque

import numpy as np
from tokenizers import Tokenizer

from ..utils.download import download_from_hf

class BaseSpeechToTextModel(ABC):

    def __init__(
        self,
        hf_repo: str,
        config_hf_repo: str,
        rate: int = 16_000
    ):
        self.hf_repo = hf_repo
        self.config_hf_repo = config_hf_repo
        self.rate = rate

        self.config = self._load_config()
        self.tokenizer = self._load_tokenizer()
        self.eos_token_id = self.config["eos_token_id"]
        self.decoder_start_token_id = self.config["decoder_start_token_id"]
        self.num_key_value_heads = self.config["decoder_num_key_value_heads"]
        self.dim_kv = (
            self.config["hidden_size"] // self.config["decoder_num_attention_heads"]
        )
        self.decoder_layers = self.config["decoder_num_hidden_layers"]
        self.max_len = self.config["max_position_embeddings"]

        self._transcribe_times = deque(maxlen=100)
    
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

    def _load_config(self):
        path = download_from_hf(repo_id=self.config_hf_repo, filename="config.json")
        with open(path, "r") as f:
            return json.load(f)

    def _load_tokenizer(self):
        path = download_from_hf(repo_id=self.config_hf_repo, filename="tokenizer.json")
        return Tokenizer.from_file(path)
    
    @abstractmethod
    def _generate(self, audio: np.ndarray, max_len: int | None = None) -> np.ndarray:
        ...

    @abstractmethod
    def transcribe(self, speech: np.ndarray) -> str:
        ...
