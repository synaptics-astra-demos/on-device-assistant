import argparse
import json
import logging
from pydoc import text
import subprocess
from pathlib import Path
from typing import Final

from core.embeddings import TextEmbeddingsAgent
from core.embeddings.minilm import MODEL_CHOICES

from ._utils import (
    InferenceStat,
    add_common_args,
    configure_logging,
    format_answer
)

DEFAULT_QA_FILE: Final = "data/qa_assistant.json"


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
    tools_path = Path("data/tools.json")
    with open(tools_path, "r") as f:
        tools = json.load(f)

    text_agent = TextEmbeddingsAgent(args.model, args.qa_file, normalize=args.normalize, n_threads=args.threads)

    try:
        while True:
            query = input("Query: ")
            if query.lower() in ("exit", "quit"):
                break
            result = text_agent.answer_query(query)
            answer, similarity, emb_infer_time = result["answer"], result["similarity"], result["infer_time"]
            answer = replace_tool_tokens(answer, tools)
            print(format_answer(answer, emb_infer_time, stats=[InferenceStat("Similarity", f"{similarity:.6f}")]))
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        text_agent.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniLM Demo")
    parser.add_argument(
        "--qa-file",
        type=str,
        default=DEFAULT_QA_FILE,
        help="Path to Question-Answer pairs (default: %(default)s)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        metavar="MODEL",
        choices=MODEL_CHOICES,
        default="synap-quantized",
        help="MiniLM model to use (default: %(default)s), available:\n%(choices)s"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="Normalize embeddings output"
    )
    add_common_args(parser)
    args = parser.parse_args()

    configure_logging(args.logging)
    logger = logging.getLogger(__name__)
    logger.info("Starting demo...")

    main()
