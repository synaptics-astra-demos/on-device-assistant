import logging

logger = logging.getLogger(__name__)


def opus_mt_factory(
    source_lang: str,
    dest_lang: str,
    model_name: str,
    *,
    eager_load: bool = True,
    n_beams: int | None = None,
    n_threads: int | None = None,
    use_onnx_encoder: bool = False
) -> "OpusMTOnnx | OpusMTSynap":
    from .opus_mt import OpusMTOnnx, OpusMTSynap, MODEL_CHOICES

    if model_name not in MODEL_CHOICES:
        raise ValueError(f"Invalid model '{model_name}', please use one of {MODEL_CHOICES}")
    model_type, quant_type = model_name.split("-")
    if model_type == "onnx":
        return OpusMTOnnx(
            source_lang, dest_lang, quant_type,
            num_beams=n_beams,
            n_threads=n_threads,
            eager_load=eager_load
        )
    return OpusMTSynap(
        source_lang, dest_lang, quant_type,
        num_beams=n_beams,
        n_threads=n_threads,
        use_onnx_encoder=use_onnx_encoder,
        eager_load=eager_load
    )


class TextTranslationAgent:

    def __init__(
        self,
        source_lang: str,
        dest_lang: str,
        model_name: str,
        *,
        eager_load: bool = True,
        n_beams: int | None = None,
        n_threads: int | None = None
    ):
        self.translator = opus_mt_factory(
            source_lang, dest_lang, model_name,
            eager_load=eager_load,
            n_beams=n_beams,
            n_threads=n_threads
        )

    @property
    def last_infer_time(self) -> float | None:
        return self.translator.last_infer_time

    def translate(self, text: str) -> str:
        return self.translator.translate(text)
