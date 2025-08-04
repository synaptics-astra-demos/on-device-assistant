import argparse
import logging
from typing import Final

from ._utils import ProfilerBase, add_common_args, configure_logging
from core.embeddings.minilm import MiniLMLlama, MiniLMSynap, MODEL_CHOICES

SAMPLE_INPUT: Final[str] = "Although recent advancements in artificial intelligence have significantly improved natural language understanding, challenges remain in ensuring models grasp contextual nuance, especially when processing complex, multi-clause sentences like this one."


class MiniLMProfiler(ProfilerBase):

    def __init__(
        self,
        model_names: str,
        logger: logging.Logger,
        sample_text: str,
        *,
        run_forever: bool = True,
        n_threads: int | None = None
    ):
        super().__init__(
            model_names, logger,
            run_forever=run_forever,
            n_threads=n_threads
        )

        self._sample_input = sample_text
        self._models: dict[str, MiniLMLlama | MiniLMSynap] = {}
        for model_name in self._model_names:
            model_type, model_quant = model_name.split("-")
            if model_type == "llama":
                self._models[model_name] = MiniLMLlama(
                    model_quant,
                    n_threads=n_threads
                )
            else:
                self._models[model_name] = MiniLMSynap(
                    model_quant
                )

    def _get_inference_time(self, model_name: str):
        model = self._models[model_name]
        model.generate(self._sample_input)
        return model.last_infer_time

    def _cleanup(self, model_name: str):
        self._models[model_name].cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_args(
        parser,
        model_choices=MODEL_CHOICES,
        default_model="synap-quantized",
        default_input=SAMPLE_INPUT,
        input_desc="Input text for inference"
    )
    args = parser.parse_args()

    configure_logging(args.logging)
    logger = logging.getLogger(__name__)
    logger.info("Starting profiling...")

    profiler = MiniLMProfiler(
        args.models, logger, args.input,
        run_forever=args.run_forever,
        n_threads=args.threads
    )
    profiler.profile_models(args.repeat)
