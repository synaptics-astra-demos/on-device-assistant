import numpy as np

from .base import BaseTranslationModel
from ..inference.runners import OnnxInferenceRunner, SynapInferenceRunner


class OpusMTBase(BaseTranslationModel):

    def __init__(
        self,
        source_lang: str,
        dest_lang: str,
        encoder: OnnxInferenceRunner | SynapInferenceRunner,
        decoder: OnnxInferenceRunner | SynapInferenceRunner,
        decoder_with_past: OnnxInferenceRunner | SynapInferenceRunner,
        *,
        cache_shapes: dict[str, tuple[int, ...]] | None,
        max_inp_len: int | None,
        max_tokens: int | None,
        is_static: bool
    ):
        super().__init__(f"Helsinki-NLP/opus-mt-{source_lang}-{dest_lang}", max_inp_len, max_tokens)

        self.source_lang = source_lang
        self.dest_lang = dest_lang
        self.is_static = is_static
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_with_past = decoder_with_past
        self.cache_shapes = cache_shapes

        self.n_layers: int = int(self.config.num_hidden_layers)
        self.n_kv_heads: int = int(self.config.decoder_attention_heads)
        self.hidden_size: int = int(self.config.d_model)
        self.head_dim: int = self.hidden_size // self.n_kv_heads
        self.start_token_id: int = self.config.pad_token_id
        self.end_token_id: int = self.config.eos_token_id
        self.encoder_pad_id: int = self.config.pad_token_id
        self.decoder_cache: dict[str, np.ndarray] = {}
        self.all_cache_tensors: list[str] = [
            k for k in self.cache_shapes if "past_key_values" in k
        ]
        self.dec_cache_tensors: list[str] = [
            k for k in self.all_cache_tensors if "encoder" not in k
        ]
    
    def _init_cache(self):
        self.decoder_cache.update({
            f"past_key_values.{i}.{a}.{b}": np.zeros(
                (1, self.n_kv_heads, 1, self.head_dim), dtype=np.float32
            )
            for i in range(self.n_layers)
            for a in ("decoder", "encoder")
            for b in ("key", "value")
        })

    @staticmethod
    def _pad_cache_tensor(cache_tensor: np.ndarray, req_shape: tuple[int, ...]) -> np.ndarray:
        if cache_tensor.shape == req_shape:
            return cache_tensor
        if cache_tensor.ndim != len(req_shape):
            raise ValueError(f"Invalid cache tensor dims: got {cache_tensor.ndim}, expected {len(req_shape)}")
        pad_width = []
        for cache_dim, req_dim in zip(cache_tensor.shape, req_shape):
            if cache_dim > req_dim:
                raise ValueError(f"Unexpected dim for cache tensor: {cache_tensor.shape}, expected: {req_shape}")
            before = 0
            after = req_dim - cache_dim
            pad_width.append((before, after))

        cache_padded = np.pad(cache_tensor, pad_width, mode='constant', constant_values=0)
        return cache_padded

    def _update_cache(self, new_values: list[np.ndarray], *, update_all: bool = False):
        cache_tensors = self.all_cache_tensors if update_all else self.dec_cache_tensors
        if len(cache_tensors) != len(new_values):
            raise ValueError(f"Cache tensors mismatch: expected {len(cache_tensors)} new values, got {len(new_values)}")
        for k, v in zip(cache_tensors, new_values):
            if self.is_static:
                self.decoder_cache[k] = self._pad_cache_tensor(v, self.cache_shapes[k])
            else:
                self.decoder_cache[k] = v

    def _next_token(self, logits: np.ndarray) -> int:
        step_logits = logits[0, -1]
        step_logits[self.encoder_pad_id] = -1e9
        return int(step_logits.argmax())

    def _run_encoder(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        encoder_out = self.encoder.infer(inputs)[0]
        self._infer_stats["encoder_infer_time_ms"] = self.encoder.infer_time_ms
        return encoder_out

    def _run_decoder(self, input_tokens: list[int], attn_mask: np.ndarray, encoder_out: np.ndarray, *, seq_len: int) -> tuple[int, list[np.ndarray]]:
        input_ids = [input_tokens]
        decoder_inputs = {
            "input_ids": np.asarray(input_ids, dtype=np.int64),
            "encoder_attention_mask": attn_mask,
        }
        if seq_len == 0:
            decoder_inputs["encoder_hidden_states"] = encoder_out
            logits, *cache = self.decoder.infer(decoder_inputs)
            self._infer_stats["decoder_infer_time_ms"] = self.decoder.infer_time_ms
        else:
            if self.is_static:
                decoder_inputs["current_len"] = np.array([[seq_len]], dtype=np.int64)
            decoder_inputs.update(self.decoder_cache)
            logits, *cache = self.decoder_with_past.infer(decoder_inputs)
            if not self._infer_stats.get("decoder_with_past_infer_time_ms"):
                self._infer_stats["decoder_with_past_infer_time_ms"] = 0
            self._infer_stats["decoder_with_past_infer_time_ms"] += self.decoder_with_past.infer_time_ms
        return self._next_token(logits), cache

    def _generate(
        self, 
        inputs: dict[str, np.ndarray], 
        max_tokens: int | None = None, 
    ) -> list[int]:
        self._infer_stats["decoder_tokens"] = 0
        if isinstance(max_tokens, int) and max_tokens < self.max_tokens:
            self.max_tokens = max_tokens

        encoder_out = self._run_encoder(inputs)
        self._init_cache()
        next_token = self.start_token_id
        tokens = [next_token]

        for i in range(self.max_tokens):
            next_token, cache = self._run_decoder([tokens[-1]], inputs["attention_mask"], encoder_out, seq_len=i)
            self._infer_stats["decoder_tokens"] += 1
            self._update_cache(cache, update_all=i==0)
            tokens.append(next_token)
            if next_token == self.end_token_id:
                break

        return tokens


class OpusMTSynap(OpusMTBase):

    def __init__(
        self,
        source_lang: str,
        dest_lang: str,
    ):
        encoder: SynapInferenceRunner = SynapInferenceRunner.from_uri(
            url=f"https://github.com/spal-synaptics/on-device-assistant/releases/download/models-v1/opus-mt-{source_lang}-{dest_lang}_encoder.synap",
            filename=f"models/synap/opus-mt/{source_lang}-{dest_lang}/encoder.synap"
        )
        decoder: SynapInferenceRunner = SynapInferenceRunner.from_uri(
            url=f"https://github.com/spal-synaptics/on-device-assistant/releases/download/models-v1/opus-mt-{source_lang}-{dest_lang}_decoder.synap",
            filename=f"models/synap/opus-mt/{source_lang}-{dest_lang}/decoder.synap"
        )
        decoder_with_past: SynapInferenceRunner = SynapInferenceRunner.from_uri(
            url=f"https://github.com/spal-synaptics/on-device-assistant/releases/download/models-v1/opus-mt-{source_lang}-{dest_lang}_decoder_with_past.synap",
            filename=f"models/synap/opus-mt/{source_lang}-{dest_lang}/decoder_with_past.synap"
        )
        cache_shapes: dict[str, tuple[int, ...]] = {
            inp.name: list(inp.shape) for inp in decoder_with_past.model.inputs if "past_key_values" in inp.name
        }
        max_inp_len: int = next(inp.shape for inp in encoder.model.inputs if inp.name == "input_ids")[-1]
        max_tokens: int = next(inp.shape for inp in decoder_with_past.model.inputs if "decoder" in inp.name)[2] # assuming shape [B, H, L, D]

        super().__init__(
            source_lang,
            dest_lang,
            encoder,
            decoder,
            decoder_with_past,
            cache_shapes=cache_shapes,
            max_inp_len=max_inp_len,
            max_tokens=max_tokens,
            is_static=True
        )
