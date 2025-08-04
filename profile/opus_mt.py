import argparse
import logging
from typing import Final

from ._utils import ProfilerBase, ProfilingStat, add_common_args, configure_logging
from core.translation.opus_mt import OpusMTSynap, MODEL_CHOICES

SAMPLE_INPUT: Final = "This is a simple sentence."


class OpusMTProfiler(ProfilerBase):

    def __init__(
        self,
        model_names: str,
        logger: logging.Logger,
        sample_text: str,
        *,
        run_forever: bool = True,
        n_threads: int | None = None,
        n_beams: int | None = None
    ):
        super().__init__(
            model_names, logger,
            run_forever=run_forever,
            n_threads=n_threads
        )

        max_inp_len: int | None = None
        self._sample_input = sample_text
        self._models: dict[str, OpusMTSynap] = {}
        for model_name in self._model_names:
            _, model_quant = model_name.split("-")
            self._models[model_name] = OpusMTSynap(
                "en", "zh", model_quant, num_beams=n_beams, n_threads=n_threads
            )
            if not isinstance(max_inp_len, int) or self._models[model_name].max_inp_len < max_inp_len:
                max_inp_len = self._models[model_name].max_inp_len
        if isinstance(max_inp_len, int):
            self._update_env_param("max_inp_size", ProfilingStat("Max input size", max_inp_len, "tokens"))
            self._sample_input = self._sample_input[:max_inp_len]

    def _get_inference_time(self, model_name: str):
        model = self._models[model_name]
        model.translate(self._sample_input)
        return model.last_infer_time

    def _cleanup(self, model_name: str):
        self._models[model_name].cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_args(
        parser,
        model_choices=[m for m in MODEL_CHOICES if "synap" in m],
        default_model=["synap-float"],
        default_input=SAMPLE_INPUT,
        input_desc="Input text for inference"
    )
    parser.add_argument(
        "-b", "--num-beams",
        type=int,
        help="Specify number of beams to use for decoding beam search"        
    )
    args = parser.parse_args()

    configure_logging(args.logging)
    logger = logging.getLogger(__name__)
    logger.info("Starting profiling...")

    profiler = OpusMTProfiler(
        args.models, logger, args.input,
        run_forever=args.run_forever,
        n_threads=args.threads,
        n_beams=args.num_beams
    )
    profiler.profile_models(args.repeat)

