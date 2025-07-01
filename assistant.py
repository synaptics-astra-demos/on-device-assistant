import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Final

from core.embeddings import TextEmbeddingsAgent
from core.speech_to_text import SpeechToTextAgent
from core.text_to_speech import TextToSpeechAgent

DEFAULT_QA_FILE: Final = "data/qa_assistant.json"
DEFAULT_SPEECH_THRESH: Final = 0.5
DEFAULT_SILENCE_DUR_MS: Final = 300 


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
    YELLOW: Final = "\033[93m"
    GREEN: Final = "\033[32m"
    RESET: Final = "\033[0m"

    def handle_speech_input(transcribed_text: str):

        print(YELLOW + f"Query: {transcribed_text}" + RESET + f" ({stt_agent.last_infer_time * 1000:.3f} ms)")
        result = text_agent.answer_query(transcribed_text)[0]
        answer, similarity, emb_infer_time = result["answer"], result["similarity"], result["infer_time"]
        answer = replace_tool_tokens(answer, tools)
        print(GREEN + f"Agent: {answer}" + RESET + f" ({emb_infer_time * 1000:.3f} ms, Similarity: {similarity:.6f})")
        wav_path = tts_agent.synthesize(answer)
        stt_agent.audio_manager.play(wav_path)

    tools_path = Path("data/tools.json")
    with open(tools_path, "r") as f:
        tools = json.load(f)

    text_agent = TextEmbeddingsAgent(args.qa_file, cpu_only=args.cpu_only, cpu_cores=args.threads)
    stt_agent = SpeechToTextAgent(
        "tiny", handle_speech_input, 
        cpu_only=args.cpu_only, 
        n_threads=args.threads,
        threshold=args.threshold,
        min_silence_duration_ms=args.silence_ms
    )
    tts_agent = TextToSpeechAgent()
    stt_agent.run()


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
    stt_args = parser.add_argument_group("speech-to-text options")
    stt_args.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SPEECH_THRESH,
        help="Speech threshold, increase to lower mic capture sensitivity (default: %(default)s)"
    )
    stt_args.add_argument(
        "--silence-ms",
        type=int,
        default=DEFAULT_SILENCE_DUR_MS,
        help="Length of silence that determines end of speech (default: %(default)s ms)"
    )
    args = parser.parse_args()

    configure_logging(args.logging)
    logger = logging.getLogger(__name__)
    logger.info("Starting demo...")

    main()
