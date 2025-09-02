import argparse
import logging
from pathlib import Path
from typing import Final

import numpy as np
import soundfile as sf

from ._utils import ProfilerBase, ProfilingStat, add_common_args, configure_logging
from core.speech_to_text.moonshine import MoonshineOnnx, MoonshineSynap, MODEL_CHOICES

SAMPLE_INPUT: Final[str] = "data/audio/shaggy.wav"


class MoonshineProfiler(ProfilerBase):

    def __init__(
        self,
        model_names: str,
        logger: logging.Logger,
        wav_file: str | Path,
        *,
        run_forever: bool = True,
        n_threads: int | None = None
    ):
        super().__init__(
            model_names, logger,
            run_forever=run_forever,
            n_threads=n_threads
        )

        if Path(wav_file).exists():
            self._sample_input, _ = sf.read(str(wav_file), dtype="float32")
        else:
            self._logger.warning("Input WAV file '%s' not found, defaulting to silence input", wav_file)
            self._sample_input = np.zeros(5 * 16000, dtype="float32")

        max_inp_len: int | None = None
        self._models: dict[str, MoonshineOnnx | MoonshineSynap] = {}
        for model_name in self._model_names:
            model_type, model_size, model_quant = model_name.split("-")
            if model_type == "onnx":
                self._models[model_name] = MoonshineOnnx(
                    model_size=model_size, quant_type=model_quant, n_threads=n_threads
                )
            elif model_type == "synap":
                self._models[model_name] = MoonshineSynap(
                    model_size=model_size, quant_type=model_quant, n_threads=n_threads
                )
                if not isinstance(max_inp_len, int) or self._models[model_name].max_inp_len < max_inp_len:
                    max_inp_len = self._models[model_name].max_inp_len
        if isinstance(max_inp_len, int):
            self._update_env_param("max_inp_size", ProfilingStat("Max input size", max_inp_len, "samples"))
            self._sample_input = self._sample_input[:max_inp_len]

    def _get_inference_time(self, model_name: str):
        model = self._models[model_name]
        model.transcribe(self._sample_input)
        return model.last_infer_time

    def _cleanup(self, model_name: str):
        self._models[model_name].cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_args(
        parser,
        model_choices=MODEL_CHOICES,
        default_model=["onnx-tiny-float", "synap-tiny-float"],
        default_input=SAMPLE_INPUT,
        input_desc="Input WAV audio for inference"
    )
    args = parser.parse_args()

    configure_logging(args.logging)
    logger = logging.getLogger(__name__)
    logger.info("Starting profiling...")

    profiler = MoonshineProfiler(
        args.models, logger, args.input,
        run_forever=args.run_forever,
        n_threads=args.threads
    )
    profiler.profile_models(args.repeat)
