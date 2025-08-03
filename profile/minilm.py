import argparse
import logging

from typing import Final

from core.embeddings.minilm import MiniLMLlama, MiniLMSynap, MODEL_CHOICES

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--models",
        type=str,
        metavar="MODEL",
        nargs="+",
        choices=MODEL_CHOICES,
        default=["synap-quantized"],
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

    models: dict[str, MiniLMLlama | MiniLMSynap] = {}
    for model_name in args.models:
        model_type, model_quant = model_name.split("-")
        if model_type == "llama":
            models[model_name] = MiniLMLlama(
                model_quant,
                n_threads=args.threads
            )
        else:
            models[model_name] = MiniLMSynap(
                model_quant
            )
    infer_times: dict[str, dict] = {
        model_name: {"n_iters": 0, "total_infer_time": 0} for model_name in models
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
