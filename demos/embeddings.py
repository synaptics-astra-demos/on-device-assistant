import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Final

from core.embeddings import TextEmbeddingsAgent

DEFAULT_QA_FILE: Final = "data/qa_assistant.json"


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


def run_command(command: str):
    try:
        out = subprocess.check_output(command, shell=True).decode().strip()
    except Exception as e:
        out = f"[error: {e}]"
    return out


def replace_tool_tokens(answer: str, tools: dict[str, str]):
    for tool in tools:
        token = tool["token"]
        if token in answer:
            output = run_command(tool["command"])
            answer = answer.replace(token, output)
    return answer


def main():
    GREEN: Final[str] = "\033[32m"
    RESET: Final[str] = "\033[0m"

    tools_path = Path("data/tools.json")
    with open(tools_path, "r") as f:
        tools = json.load(f)

    text_agent = TextEmbeddingsAgent(args.qa_file, cpu_only=args.cpu_only, cpu_cores=args.threads)

    try:
        while True:
            query = input("Query: ")
            result = text_agent.answer_query(query)[0]
            answer, similarity, emb_infer_time = result["answer"], result["similarity"], result["infer_time"]
            answer = replace_tool_tokens(answer, tools)
            print(GREEN + f"Agent: {answer}" + RESET + f" ({emb_infer_time * 1000:.3f} ms, Similarity: {similarity:.6f})")
    except (KeyboardInterrupt, EOFError):
        logger.info("Stopped by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q&A AI Assistant")
    parser.add_argument(
        "--qa-file",
        type=str,
        default=DEFAULT_QA_FILE,
        help="Path to Question-Answer pairs (default: %(default)s)"
    )
    parser.add_argument(
        "--logging",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity: %(choices)s (default: %(default)s)"
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        default=False,
        help="Use CPU only models"
    )
    parser.add_argument(
        "-j", "--threads",
        type=int,
        help="Number of cores to use for CPU execution (default: all)"
    )
    args = parser.parse_args()

    configure_logging(args.logging)
    logger = logging.getLogger(__name__)
    logger.info("Starting demo...")

    main()
