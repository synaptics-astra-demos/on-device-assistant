import logging

logger = logging.getLogger(__name__)


def opus_mt_factory(
    source_lang: str,
    dest_lang: str,
    model_type: str,
    quant_type: str,
    n_beams: int | None = None,
    n_threads: int | None = None,
    use_onnx_encoder: bool = False
) -> "OpusMTOnnx | OpusMTSynap":
    from .opus_mt import OpusMTOnnx, OpusMTSynap, MODEL_TYPES, QUANT_TYPES
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Invalid model type: {model_type}. Supported types are: {MODEL_TYPES}.")
    if quant_type not in QUANT_TYPES:
        raise ValueError(f"Invalid quantization type: {quant_type}. Supported types are: {QUANT_TYPES}.")
    if model_type == "onnx":
        return OpusMTOnnx(
            source_lang, dest_lang, quant_type, num_beams=n_beams, n_threads=n_threads
        )
    return OpusMTSynap(
        source_lang, dest_lang, quant_type, num_beams=n_beams, n_threads=n_threads, use_onnx_encoder=use_onnx_encoder
    )


class TextTranslationAgent:

    def __init__(
        self,
        source_lang: str,
        dest_lang: str,
        model_type: str,
        quant_type: str,
        *,
        n_beams: int | None = None,
        n_threads: int | None = None
    ):
        self.translator = opus_mt_factory(
            source_lang, dest_lang, model_type, quant_type, n_beams=n_beams, n_threads=n_threads
        )

    @property
    def last_infer_time(self) -> float | None:
        return self.translator.last_infer_time

    def translate(self, text: str) -> str:
        return self.translator.translate(text)
