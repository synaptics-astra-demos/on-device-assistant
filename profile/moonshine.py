import argparse
import logging
from typing import Final

import soundfile as sf

from core.speech_to_text import MoonshineOnnx, MoonshineSynap

DEFAULT_SIZE: Final = "tiny"
SAMPLE_INPUT: Final = "data/audio/beckett.wav"


def configure_logging(verbosity: str):
    level = getattr(logging, verbosity.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {verbosity}")

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model",
        type=str,
        choices=["tiny", "base"],
        default=DEFAULT_SIZE,
        help="Moonshine model size (default: %(default)s)"
    )
    parser.add_argument(
        "-r", "--repeat",
        type=int,
        default=100,
        help="Number of iterations to repeat inference (default: %(default)s)"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=SAMPLE_INPUT,
        help="Input text for inference (default: \"%(default)s)\""
    )
    parser.add_argument(
        "-j", "--threads",
        type=int,
        help="Number of cores to use for CPU execution (default: all)"
    )
    parser.add_argument(
        "--logging",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity: %(choices)s (default: %(default)s)"
    )
    parser.add_argument(
        "--run-forever",
        action="store_true",
        default=False,
        help="Run profiling forever, alternating between provided models"
    )
    args = parser.parse_args()

    configure_logging(args.logging)
    logger = logging.getLogger(__name__)
    logger.info("Starting profiling...")

    if args.model != "tiny":
        raise ValueError("Only Moonshine tiny is currently available for profiling")

    moonshine_onnx = MoonshineOnnx(
        model_size=args.model, n_threads=args.threads
    )
    moonshine_synap = MoonshineSynap(
        model_size=args.model
    )
    
    audio, sr = sf.read(args.input, dtype="float32")
    audio = audio[:moonshine_synap.max_inp_len]
    
    models: dict[str, MoonshineOnnx | MoonshineSynap] = {
        "ONNX": moonshine_onnx,
        "SyNAP": moonshine_synap
    }
    infer_times: dict[str, dict] = {
        model_name: {"n_iters": 0, "total_infer_time": 0} for model_name in models
    }

    while True:
        try:
            for model_name, model in models.items():
                logger.info(f"Profiling '{model_name}' ({args.repeat} iters)...")
                try:
                    for _ in range(args.repeat):
                        model.transcribe(audio)
                        infer_times[model_name]["n_iters"] += 1
                        infer_times[model_name]["total_infer_time"] += model.last_infer_time or 0
                except Exception as e:
                    logger.warning(f"Stopping inference due to error: {e}")
                    break
        except KeyboardInterrupt:
            break
        
        if not args.run_forever:
            break

    print("\n\nProfiling report")
    print("------------------------------------")
    print("Environment:")
    print(f"\tMax CPU threads: {args.threads}")
    for model_name in models:
        n_iters = infer_times[model_name]["n_iters"]
        total_infer_time = infer_times[model_name]["total_infer_time"]
        print(f"\nStats for '{model_name}' ({n_iters} iters):")
        print(f"\tTotal inference time   : {total_infer_time * 1000:.3f} ms")
        print(f"\tAverage inference time : {(total_infer_time / n_iters) * 1000:.3f} ms")
