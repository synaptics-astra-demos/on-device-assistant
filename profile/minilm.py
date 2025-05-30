import argparse
import logging
from pathlib import Path

from typing import Final

from core.embeddings.minilm import MiniLMLlama, MiniLMSynap

DEFAULT_MODELS: Final = [
    Path("models/gguf/all-MiniLM-L6-v2-Q8_0.gguf"),
    Path("models/synap/all-MiniLM-L6-v2.synap")
]
SAMPLE_INPUT: Final = "Although recent advancements in artificial intelligence have significantly improved natural language understanding, challenges remain in ensuring models grasp contextual nuance, especially when processing complex, multi-clause sentences like this one."


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


def get_model(model_path: str | Path) -> MiniLMLlama | MiniLMSynap:
    model_path = Path(model_path).resolve()
    if model_path.suffix == ".gguf":
        model = MiniLMLlama(
            model_name="Llama",
            model_path=str(model_path), 
            n_threads=args.threads
        )
    elif model_path.suffix == ".synap":
        if not args.hf_repo:
            raise ValueError("Hugging Face model ID required for SyNAP model tokenizer")
        model = MiniLMSynap(
            model_name="SyNAP",
            model_path=str(model_path),
            hf_model=args.hf_repo
        )
    else:
        raise ValueError(f"Invalid model format {model_path.suffix}, must be one of ('.gguf', '.synap')")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--models",
        type=str,
        nargs="+",
        default=DEFAULT_MODELS,
        help="Path to SyNAP or GGUF model"
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
        "--hf-repo",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face model repo (required for SyNAP models) (default: %(default)s)"
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

    models: dict[str, MiniLMLlama | MiniLMSynap] = {
        model_path: get_model(model_path) for model_path in args.models
    }
    infer_times: dict[str, dict] = {
        model_path: {"n_iters": 0, "total_infer_time": 0} for model_path in args.models
    }

    while True:
        try:
            for model_path, model in models.items():
                logger.info(f"Profiling '{model_path}' ({args.repeat} iters)...")
                try:
                    for _ in range(args.repeat):
                        model.generate(args.input)
                        infer_times[model_path]["n_iters"] += 1
                        infer_times[model_path]["total_infer_time"] += model.last_infer_time or 0
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
    for model_path in models:
        n_iters = infer_times[model_path]["n_iters"]
        total_infer_time = infer_times[model_path]["total_infer_time"]
        print(f"\nStats for '{model_path}' ({n_iters} iters):")
        print(f"\tTotal inference time   : {total_infer_time * 1000:.3f} ms")
        print(f"\tAverage inference time : {(total_infer_time / n_iters) * 1000:.3f} ms")
