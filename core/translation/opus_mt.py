import json
import logging
import math
from itertools import product
from typing import Any, Final, Iterable, Literal

import numpy as np
import sentencepiece as spm

from .base import BaseTranslationModel
from ..inference.runners import OnnxInferenceRunner, SynapInferenceRunner
from ..utils.download import download_from_hf


MODEL_TYPES: Final = ["onnx", "synap"]
QUANT_TYPES: Final = ["float", "quantized"]
MODEL_CHOICES: Final = [f"{t}-{q}" for (t, q) in product(MODEL_TYPES, QUANT_TYPES)]
NUM_BEAMS: Final = 4

logger = logging.getLogger(__name__)


class OpusMTTokenizer:

    def __init__(
        self,
        hf_repo: str
    ):
        self._encoder = spm.SentencePieceProcessor(download_from_hf(hf_repo, "source.spm"))
        self._decoder = spm.SentencePieceProcessor(download_from_hf(hf_repo, "target.spm"))
        with open(download_from_hf(hf_repo, "config.json"), "r") as f:
            self._config: dict[str, Any] = json.load(f)
        self._max_input_length: int = int(self._config["max_position_embeddings"])
        self._pad_token = int(self._config["pad_token_id"])
        self._eos_token = int(self._config["eos_token_id"])
        self._special_tokens: set[int] = {
            int(tok_id) for tok_name, tok_id in self._config.items() if "token_id" in tok_name
        }
        with open(download_from_hf(hf_repo, "vocab.json"), "r") as f:
            self._vocab: dict[str, int] = json.load(f)
            self._token_map: dict[int, str] = {v: k for k, v in self._vocab.items()}
        self._inp_trunc: int = 0
        self._inp_pad: int = 0

    @property
    def input_truncation_len(self) -> int:
        return self._inp_trunc

    @property
    def input_padding_len(self) -> int:
        return self._inp_pad

    def __call__(
        self,
        text: str,
        max_length: int | None = None
    ) -> dict[str, np.ndarray]:
        return self.encode(text, max_length)

    def _create_attn_mask(
        self,
        tokens: np.ndarray
    ):
        assert tokens.ndim == 1
        return (tokens != self._pad_token).astype(np.int64)

    def _encode_inputs(
        self,
        text: str,
        max_length: int | None,
    ) -> np.ndarray:
        self._inp_trunc = 0
        self._inp_pad = 0
        tokens = [self._vocab[raw_token] for raw_token in self._encoder.encode(text, out_type=str)]
        tokens.append(self._eos_token)
        n_tokens = len(tokens)
        max_length = max_length or n_tokens
        if n_tokens > max_length:
            tokens = tokens[:max_length]
            tokens[-1] = self._eos_token
            self._inp_trunc = n_tokens - max_length
        else:
            diff = max_length - n_tokens
            tokens = tokens + [self._pad_token] * diff
            self._inp_pad = diff
        return np.asarray(tokens, dtype=np.int64)

    def encode(
        self,
        text: str,
        max_length: int | None = None
    ) -> dict[str, np.ndarray]:
        if not isinstance(text, str):
            raise TypeError(f"Input must be str, received type '{type(text)}'")
        if max_length is None:
            max_input_length = None
        else:
            max_input_length = self._max_input_length
            if isinstance(max_length, int) and 0 < max_length < self._max_input_length:
                max_input_length = max_length
        tokens = self._encode_inputs(text, max_input_length)
        return {
            "input_ids": tokens[np.newaxis, :],
            "attention_mask": self._create_attn_mask(tokens)[np.newaxis, :]
        }

    def decode(
        self,
        tokens: Iterable[int],
        skip_special_tokens: bool = False
    ) -> str:
        if isinstance(tokens, np.ndarray):
            tokens = tokens.flatten().tolist()
        if not (isinstance(tokens, Iterable) and all(isinstance(i, int) for i in tokens)):
            raise TypeError(f"Tokens must be an iterable of integers")
        mapped: list[int] = []
        for token in tokens:
            if skip_special_tokens and token in self._special_tokens:
                continue
            mapped.append(self._token_map[token])
        return self._decoder.decode(mapped)


class OpusMTBase(BaseTranslationModel):

    def __init__(
        self,
        source_lang: str,
        dest_lang: str,
        encoder: OnnxInferenceRunner | SynapInferenceRunner,
        decoder: OnnxInferenceRunner | SynapInferenceRunner,
        decoder_with_past: OnnxInferenceRunner | SynapInferenceRunner,
        cache_shapes: dict[str, tuple[int | str, ...]],
        *,
        max_inp_len: int | None = None,
        max_tokens: int | None = None,
        is_static: bool = False,
        num_beams: int | None = None,
        length_penalty: float = 1.0,
    ):
        super().__init__()

        self.tokenizer = OpusMTTokenizer(f"Helsinki-NLP/opus-mt-{source_lang}-{dest_lang}")
        with open(download_from_hf(f"Helsinki-NLP/opus-mt-{source_lang}-{dest_lang}", "config.json"), "r") as f:
            self.config: dict[str, Any] = json.load(f)

        self.source_lang = source_lang
        self.dest_lang = dest_lang
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_with_past = decoder_with_past
        self.cache_shapes = cache_shapes
        self.max_inp_len = max_inp_len
        self.max_tokens = max_tokens
        self.is_static = is_static
        self.num_beams = num_beams if isinstance(num_beams, int) and num_beams > 0 else int(self.config.get("num_beams", NUM_BEAMS))
        self.length_penalty = length_penalty

        self.n_layers: int = int(self.config["num_hidden_layers"])
        self.n_kv_heads: int = int(self.config["decoder_attention_heads"])
        self.hidden_size: int = int(self.config["d_model"])
        self.head_dim: int = self.hidden_size // self.n_kv_heads
        self.start_token_id: int = self.config["decoder_start_token_id"]
        self.end_token_id: int = self.config["eos_token_id"]
        self.encoder_pad_id: int = self.config["pad_token_id"]
        self.decoder_cache: dict[str, np.ndarray] = {}
        self.all_cache_tensors: list[str] = [
            k for k in self.cache_shapes if "past_key_values" in k
        ]
        self.dec_cache_tensors: list[str] = [
            k for k in self.all_cache_tensors if "encoder" not in k
        ]

    @staticmethod
    def _log_softmax(x: np.ndarray) -> np.ndarray:
        m = x.max(axis=-1, keepdims=True)
        y = x - m
        return y - np.log(np.exp(y).sum(axis=-1, keepdims=True))

    @staticmethod
    def _calc_length_penalty(seq_len: int, alpha: float) -> float:
        return math.pow((5.0 + seq_len) / 6.0, alpha) if alpha != 0.0 else 1.0

    @staticmethod
    def _pad_cache_tensor(tensor: np.ndarray, req_shape: tuple[int, ...]) -> np.ndarray:
        if tensor.shape == req_shape:
            return tensor
        pad = [(0, t - s) for s, t in zip(tensor.shape, req_shape)]
        return np.pad(tensor, pad, mode="constant", constant_values=0)

    def _init_empty_cache(self) -> dict[str, np.ndarray]:
        return {
            f"past_key_values.{i}.{a}.{b}": np.zeros((1, self.n_kv_heads, 1, self.head_dim), dtype=np.float32)
            for i in range(self.n_layers)
            for a in ("decoder", "encoder")
            for b in ("key", "value")
        }

    def _tokenize(self, text: str) -> dict[str, np.ndarray]:
        params = {"text": text}
        if isinstance(self.max_inp_len, int):
            params.update({"max_length": self.max_inp_len})
        return dict(self.tokenizer(**params))

    def _decode_tokens(self, tokens: Iterable[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens)

    def _decoder_logits(
        self,
        last_token: int,
        attn_mask: np.ndarray,
        enc_out: np.ndarray,
        cache: dict[str, np.ndarray],
        seq_len: int,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        inp_ids = [[last_token]]
        dec_inp = {
            "input_ids": inp_ids,
            "encoder_attention_mask": attn_mask,
        }
        if seq_len == 0:
            dec_inp["encoder_hidden_states"] = enc_out
            logits, *new_cache = self.decoder.infer(dec_inp)
            self._infer_stats["decoder_infer_time_ms"] = self.decoder.infer_time_ms
            cache_keys = self.all_cache_tensors
        else:
            if self.is_static:
                dec_inp["current_len"] = np.array([[seq_len]], dtype=np.int64)
            dec_inp.update(cache)
            logits, *new_cache = self.decoder_with_past.infer(dec_inp)
            if not self._infer_stats.get("decoder_with_past_infer_time_ms"):
                self._infer_stats["decoder_with_past_infer_time_ms"] = 0
            self._infer_stats["decoder_with_past_infer_time_ms"] += self.decoder_with_past.infer_time_ms
            cache_keys = self.dec_cache_tensors

        updated = cache.copy()
        for k, v in zip(cache_keys, new_cache):
            updated[k] = self._pad_cache_tensor(v, self.cache_shapes[k]) if self.is_static else v
        return logits, updated

    def _run_encoder(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        encoder_out = self.encoder.infer(inputs)[0]
        self._infer_stats["encoder_infer_time_ms"] = self.encoder.infer_time_ms
        return encoder_out

    def _generate(
        self, 
        inputs: dict[str, np.ndarray], 
        max_tokens: int | None = None, 
    ) -> list[int]:
        self._infer_stats["num_beams"] = self.num_beams
        self._infer_stats["length_penalty"] = self.length_penalty
        self._infer_stats["decoder_tokens"] = 0
        if isinstance(max_tokens, int) and max_tokens < self.max_tokens:
            self.max_tokens = max_tokens
        if (trunc_len := self.tokenizer.input_truncation_len) > 0:
            logger.warning("%s: Truncating input from %d to %d tokens", self.__class__.__name__, self.max_inp_len + trunc_len, self.max_inp_len)
        elif (pad_len := self.tokenizer.input_padding_len) > 0:
            logger.debug("%s: Padding input from %d to %d tokens", self.__class__.__name__, self.max_inp_len, self.max_inp_len + pad_len)
        enc_out: np.ndarray = self._run_encoder(inputs)
        attn_mask: np.ndarray = inputs["attention_mask"]
        seqs: list[list[int]] = [[self.start_token_id] for _ in range(self.num_beams)]
        caches: list[dict[str, np.ndarray]] = [self._init_empty_cache() for _ in range(self.num_beams)]
        beam_scores: np.ndarray = np.full(self.num_beams, -np.inf, dtype=np.float32)
        beam_scores[0] = 0.0
        done: list[bool] = [False] * self.num_beams
        vocab_size: int = int(self.config["vocab_size"])

        for step in range(self.max_tokens):
            all_scores, step_caches = [], []
            all_scores = np.full((self.num_beams, vocab_size), -1e9, dtype=np.float32)

            for b in range(self.num_beams):
                if done[b]:
                    all_scores[b][self.end_token_id] = 0.0
                    step_caches.append(caches[b])
                    continue

                logits, new_cache = self._decoder_logits(
                    seqs[b][-1],
                    attn_mask,
                    enc_out,
                    caches[b],
                    step,
                )
                step_caches.append(new_cache)

                logits = logits[0, -1]
                logits[self.encoder_pad_id] = -1e9
                logp = self._log_softmax(logits)
                all_scores[b] = logp

            total = all_scores + beam_scores[:, None]
            flat = total.reshape(-1)

            topk = flat.argsort()[-2 * self.num_beams:][::-1]

            next_scores, next_tokens, next_sources = [], [], []
            for idx in topk:
                src = idx // vocab_size
                tok = idx % vocab_size
                next_scores.append(flat[idx])
                next_tokens.append(int(tok))
                next_sources.append(int(src))
                self._infer_stats["decoder_tokens"] += 1
                if len(next_scores) == self.num_beams:
                    break

            # always guaranteed B elements now
            new_seqs, new_caches, new_done = [], [], []
            for i in range(self.num_beams):
                src = next_sources[i]
                tok = next_tokens[i]

                new_seq = seqs[src] + [tok]
                new_done.append(done[src] or tok == self.end_token_id)
                new_seqs.append(new_seq)
                new_caches.append(step_caches[src])

            seqs, caches, beam_scores, done = (
                new_seqs,
                new_caches,
                np.array(next_scores, dtype=np.float32),
                new_done,
            )
            if all(done):
                break

        final = np.array(
            [
                s / self._calc_length_penalty(len(seq), self.length_penalty)
                for s, seq in zip(beam_scores, seqs)
            ],
            dtype=np.float32,
        )
        best = int(final.argmax())
        return seqs[best]

    def cleanup(self):
        self.encoder.unload()
        self.decoder.unload()
        self.decoder_with_past.unload()


class OpusMTOnnx(OpusMTBase):

    def __init__(
        self,
        source_lang: str,
        dest_lang: str,
        quant_type: Literal["float", "quantized"],
        *,
        num_beams: int | None = None,
        n_threads: int | None = None,
        eager_load: bool = True
    ):
        encoder: OnnxInferenceRunner = OnnxInferenceRunner.from_uri(
            url=f"https://github.com/spal-synaptics/on-device-assistant/releases/download/models-v1/opus-mt-{source_lang}-{dest_lang}-{quant_type}_encoder_model.onnx",
            filename=f"models/Helsinki-NLP/opus-mt-{source_lang}-{dest_lang}/{quant_type}/encoder_model.onnx",
            n_threads=n_threads,
            eager_load=eager_load
        )
        decoder: OnnxInferenceRunner = OnnxInferenceRunner.from_uri(
            url=f"https://github.com/spal-synaptics/on-device-assistant/releases/download/models-v1/opus-mt-{source_lang}-{dest_lang}-{quant_type}_decoder_model.onnx",
            filename=f"models/Helsinki-NLP/opus-mt-{source_lang}-{dest_lang}/{quant_type}/decoder_model.onnx",
            n_threads=n_threads,
            eager_load=eager_load
        )
        decoder_with_past: OnnxInferenceRunner = OnnxInferenceRunner.from_uri(
            url=f"https://github.com/spal-synaptics/on-device-assistant/releases/download/models-v1/opus-mt-{source_lang}-{dest_lang}-{quant_type}_decoder_with_past_model.onnx",
            filename=f"models/Helsinki-NLP/opus-mt-{source_lang}-{dest_lang}/{quant_type}/decoder_with_past_model.onnx",
            n_threads=n_threads,
            eager_load=eager_load
        )
        cache_shapes: dict[str, tuple[int, ...]] = {
            inp.name: inp.shape for inp in decoder_with_past.inputs_info if "past_key_values" in inp.name
        }
        super().__init__(
            source_lang,
            dest_lang,
            encoder,
            decoder,
            decoder_with_past,
            cache_shapes,
            num_beams=num_beams
        )
        if not eager_load:
            logger.warning("%s: Eager loading disabled, initial inference will be slower", self.__class__.__name__)


class OpusMTSynap(OpusMTBase):

    def __init__(
        self,
        source_lang: str,
        dest_lang: str,
        quant_type: Literal["float", "quantized"],
        *,
        num_beams: int | None = None,
        use_onnx_encoder: bool | None = None,
        n_threads: int | None = None,
        eager_load: bool = True
    ):
        encoder: SynapInferenceRunner = SynapInferenceRunner.from_uri(
            url=f"https://github.com/spal-synaptics/on-device-assistant/releases/download/models-v1/opus-mt-{source_lang}-{dest_lang}-{quant_type}_encoder.synap",
            filename=f"models/synap/opus-mt/{source_lang}-{dest_lang}/{quant_type}/encoder.synap",
            eager_load=eager_load
        )
        if use_onnx_encoder:
            encoder_onnx: OnnxInferenceRunner = OnnxInferenceRunner.from_uri(
                url=f"https://github.com/spal-synaptics/on-device-assistant/releases/download/models-v1/opus-mt-{source_lang}-{dest_lang}-{quant_type}_encoder_model.onnx",
                filename=f"models/Helsinki-NLP/opus-mt-{source_lang}-{dest_lang}/{quant_type}/encoder_model.onnx",
                n_threads=n_threads,
                eager_load=eager_load
            )
        decoder: SynapInferenceRunner = SynapInferenceRunner.from_uri(
            url=f"https://github.com/spal-synaptics/on-device-assistant/releases/download/models-v1/opus-mt-{source_lang}-{dest_lang}-{quant_type}_decoder.synap",
            filename=f"models/synap/opus-mt/{source_lang}-{dest_lang}/{quant_type}/decoder.synap",
            eager_load=eager_load
        )
        decoder_with_past: SynapInferenceRunner = SynapInferenceRunner.from_uri(
            url=f"https://github.com/spal-synaptics/on-device-assistant/releases/download/models-v1/opus-mt-{source_lang}-{dest_lang}-{quant_type}_decoder_with_past.synap",
            filename=f"models/synap/opus-mt/{source_lang}-{dest_lang}/{quant_type}/decoder_with_past.synap",
            eager_load=eager_load
        )
        cache_shapes: dict[str, tuple[int, ...]] = {
            inp.name: inp.shape for inp in decoder_with_past.inputs_info if "past_key_values" in inp.name
        }
        max_inp_len: int = next(inp.shape for inp in encoder.inputs_info if inp.name == "input_ids")[-1]
        max_tokens: int = next(inp.shape for inp in decoder_with_past.inputs_info if "decoder" in inp.name)[2] # assuming shape [B, H, L, D]

        super().__init__(
            source_lang,
            dest_lang,
            encoder_onnx if use_onnx_encoder else encoder,
            decoder,
            decoder_with_past,
            cache_shapes,
            max_inp_len=max_inp_len,
            max_tokens=max_tokens,
            is_static=True,
            num_beams=num_beams
        )
        if not eager_load:
            logger.warning("%s: Eager loading disabled, initial inference will be slower", self.__class__.__name__)


def main():
    import argparse
    import json
    from pathlib import Path

    def _dump_results_json(results: dict, path: Path):
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {path}")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "texts",
        metavar="TEXT",
        nargs="+",
        type=str,
        help="Text(s) to translate"
    )
    parser.add_argument(
        "-s", "--source-lang",
        type=str,
        default="en",
        help="Source language code (default: %(default)s)"
    )
    parser.add_argument(
        "-d", "--dest-lang",
        type=str,
        default="zh",
        help="Destination language code (default: %(default)s)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        metavar="MODEL",
        choices=MODEL_CHOICES,
        default="synap-float",
        help="Model type to use for inference (available:\n%(choices)s)"
    )
    parser.add_argument(
        "-b", "--num-beams",
        type=int,
        help="Specify number of beams to use for decoding beam search"        
    )
    parser.add_argument(
        "-o", "--dump-out",
        type=str,
        metavar="JSON",
        help="Dump inference results to JSON file",
    )
    parser.add_argument(
        "-j", "--threads",
        type=int,
        help="Number of cores to use for CPU execution (default: all)"
    )
    parser.add_argument(
        "--use-onnx-encoder",
        action="store_true",
        default=False,
        help="Run NPU based SyNAP encoder instead of CPU based ONNX encoder"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Display verbose stats"
    )
    args = parser.parse_args()

    model_type, quant_type = args.model.split("-")
    if model_type == "onnx":
        translator = OpusMTOnnx(
            args.source_lang,
            args.dest_lang,
            quant_type,
            num_beams=args.num_beams,
            n_threads=args.threads,
            eager_load=True
        )
    else:
        translator = OpusMTSynap(
            args.source_lang,
            args.dest_lang,
            quant_type,
            num_beams=args.num_beams,
            use_onnx_encoder=args.use_onnx_encoder,
            n_threads=args.threads,
            eager_load=True
        )

    all_results = {
        "model": f"opus-mt-{args.source_lang}-{args.dest_lang}",
        "model_type": args.model,
        "source_lang": args.source_lang,
        "dest_lang": args.dest_lang,
        "results": []
    }
    print(f"\nTranslating {args.source_lang} to {args.dest_lang}...\n")
    for text in args.texts:
        print()
        result = translator.translate(text)
        print(f"Translated ({translator.last_infer_time * 1000:.2f} ms): \"{result}\"")
        curr_results = {
            "input": text,
            "translation": result,
            "total_infer_time_ms": translator.last_infer_time * 1000,
        }
        if args.verbose:
            print(f"Detailed inference stats:\n{json.dumps(translator.last_infer_stats, indent=2)}")
            curr_results["detailed_infer_stats"] = translator.last_infer_stats
        all_results["results"].append(curr_results)
        print()
    if args.dump_out:
        _dump_results_json(all_results, Path(args.dump_out))


if __name__ == "__main__":
    import logging
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    main()
