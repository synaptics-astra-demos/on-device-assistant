import logging
from typing import Literal

logger = logging.getLogger(__name__)


def opus_mt_factory(
    source_lang: str,
    dest_lang: str,
    quant_type: Literal["float", "quantized"],
    n_threads: int | None = None,
    cpu_only: bool = False,
    use_onnx_encoder: bool = False
) -> "OpusMTOnnx | OpusMTSynap":
    from .opus_mt import OpusMTOnnx, OpusMTSynap, QUANT_TYPES

    if quant_type not in QUANT_TYPES:
        raise ValueError(f"Invalid quantization type: {quant_type}. Supported types are: {QUANT_TYPES}.")
    if cpu_only:
        return OpusMTOnnx(
            source_lang, dest_lang, quant_type, n_threads=n_threads
        )
    return OpusMTSynap(
        source_lang, dest_lang, quant_type, n_threads=n_threads, use_onnx_encoder=use_onnx_encoder
    )


class TextTranslationAgent:

    def __init__(
        self,
        source_lang: str,
        dest_lang: str,
        quant_type: Literal["float", "quantized"],
        n_threads: int | None = None,
        cpu_only: bool = False
    ):
        self.translator = opus_mt_factory(
            source_lang, dest_lang, quant_type, n_threads, cpu_only
        )

    @property
    def last_infer_time(self) -> float | None:
        return self.translator.last_infer_time

    def translate(self, text: str) -> str:
        return self.translator.translate(text)
