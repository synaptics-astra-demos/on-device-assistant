import math
from itertools import product
from typing import Final
from typing import Literal

import numpy as np

from .base import BaseTranslationModel
from ..inference.runners import OnnxInferenceRunner, SynapInferenceRunner


MODEL_TYPES: Final = ["onnx", "synap"]
QUANT_TYPES: Final = ["float", "quantized"]
MODEL_CHOICES: Final = [f"{t}-{q}" for (t, q) in product(MODEL_TYPES, QUANT_TYPES)]


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
        num_beams: int = 5,
        length_penalty: float = 1.0,
    ):
        super().__init__(f"Helsinki-NLP/opus-mt-{source_lang}-{dest_lang}", max_inp_len, max_tokens)

        self.source_lang = source_lang
        self.dest_lang = dest_lang
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_with_past = decoder_with_past
        self.cache_shapes = cache_shapes
        self.is_static = is_static
        self.num_beams = num_beams
        self.length_penalty = length_penalty

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
        self._infer_stats["decoder_tokens"] = 0
        if isinstance(max_tokens, int) and max_tokens < self.max_tokens:
            self.max_tokens = max_tokens

        enc_out: np.ndarray = self._run_encoder(inputs)
        attn_mask: np.ndarray = inputs["attention_mask"]
        seqs: list[list[int]] = [[self.start_token_id] for _ in range(self.num_beams)]
        caches: list[dict[str, np.ndarray]] = [self._init_empty_cache() for _ in range(self.num_beams)]
        beam_scores: np.ndarray = np.full(self.num_beams, -np.inf, dtype=np.float32)
        beam_scores[0] = 0.0
        done: list[bool] = [False] * self.num_beams
        vocab_size: int = int(self.config.vocab_size)

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


class OpusMTOnnx(OpusMTBase):

    def __init__(
        self,
        source_lang: str,
        dest_lang: str,
        quant_type: Literal["float", "quantized"],
        *,
        num_beams: int = 5,
        n_threads: int | None = None
    ):
        encoder: OnnxInferenceRunner = OnnxInferenceRunner.from_uri(
            url=f"https://github.com/spal-synaptics/on-device-assistant/releases/download/models-v1/opus-mt-{source_lang}-{dest_lang}-{quant_type}_encoder_model.onnx",
            filename=f"models/Helsinki-NLP/opus-mt-{source_lang}-{dest_lang}/{quant_type}/encoder_model.onnx",
            n_threads=n_threads
        )
        decoder: OnnxInferenceRunner = OnnxInferenceRunner.from_uri(
            url=f"https://github.com/spal-synaptics/on-device-assistant/releases/download/models-v1/opus-mt-{source_lang}-{dest_lang}-{quant_type}_decoder_model.onnx",
            filename=f"models/Helsinki-NLP/opus-mt-{source_lang}-{dest_lang}/{quant_type}/decoder_model.onnx",
            n_threads=n_threads
        )
        decoder_with_past: OnnxInferenceRunner = OnnxInferenceRunner.from_uri(
            url=f"https://github.com/spal-synaptics/on-device-assistant/releases/download/models-v1/opus-mt-{source_lang}-{dest_lang}-{quant_type}_decoder_with_past_model.onnx",
            filename=f"models/Helsinki-NLP/opus-mt-{source_lang}-{dest_lang}/{quant_type}/decoder_with_past_model.onnx",
            n_threads=n_threads
        )
        cache_shapes: dict[str, tuple[int, ...]] = {
            inp.name: inp.shape for inp in decoder_with_past.model.get_inputs() if "past_key_values" in inp.name
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


class OpusMTSynap(OpusMTBase):

    def __init__(
        self,
        source_lang: str,
        dest_lang: str,
        quant_type: Literal["float", "quantized"],
        *,
        num_beams: int = 5,
        use_synap_encoder: bool = False,
        n_threads: int | None = None
    ):
        encoder: SynapInferenceRunner = SynapInferenceRunner.from_uri(
            url=f"https://github.com/spal-synaptics/on-device-assistant/releases/download/models-v1/opus-mt-{source_lang}-{dest_lang}-{quant_type}_encoder.synap",
            filename=f"models/synap/opus-mt/{source_lang}-{dest_lang}/{quant_type}/encoder.synap"
        )
        if not use_synap_encoder:
            encoder_onnx: OnnxInferenceRunner = OnnxInferenceRunner.from_uri(
                url=f"https://github.com/spal-synaptics/on-device-assistant/releases/download/models-v1/opus-mt-{source_lang}-{dest_lang}-{quant_type}_encoder_model.onnx",
                filename=f"models/Helsinki-NLP/opus-mt-{source_lang}-{dest_lang}/{quant_type}/encoder_model.onnx",
                n_threads=n_threads
            )
        decoder: SynapInferenceRunner = SynapInferenceRunner.from_uri(
            url=f"https://github.com/spal-synaptics/on-device-assistant/releases/download/models-v1/opus-mt-{source_lang}-{dest_lang}-{quant_type}_decoder.synap",
            filename=f"models/synap/opus-mt/{source_lang}-{dest_lang}/{quant_type}/decoder.synap"
        )
        decoder_with_past: SynapInferenceRunner = SynapInferenceRunner.from_uri(
            url=f"https://github.com/spal-synaptics/on-device-assistant/releases/download/models-v1/opus-mt-{source_lang}-{dest_lang}-{quant_type}_decoder_with_past.synap",
            filename=f"models/synap/opus-mt/{source_lang}-{dest_lang}/{quant_type}/decoder_with_past.synap"
        )
        cache_shapes: dict[str, tuple[int, ...]] = {
            inp.name: list(inp.shape) for inp in decoder_with_past.model.inputs if "past_key_values" in inp.name
        }
        max_inp_len: int = next(inp.shape for inp in encoder.model.inputs if inp.name == "input_ids")[-1]
        max_tokens: int = next(inp.shape for inp in decoder_with_past.model.inputs if "decoder" in inp.name)[2] # assuming shape [B, H, L, D]

        super().__init__(
            source_lang,
            dest_lang,
            encoder if use_synap_encoder else encoder_onnx,
            decoder,
            decoder_with_past,
            cache_shapes,
            max_inp_len=max_inp_len,
            max_tokens=max_tokens,
            is_static=True,
            num_beams=num_beams
        )


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
        default="fr",
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
        default=5,
        help="Number of beams to use for beam search during decoding (default: %(default)s)"        
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
        "--use-synap-encoder",
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
            n_threads=args.threads
        )
    else:
        translator = OpusMTSynap(
            args.source_lang,
            args.dest_lang,
            quant_type,
            num_beams=args.num_beams,
            use_synap_encoder=args.use_synap_encoder,
            n_threads=args.threads
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
