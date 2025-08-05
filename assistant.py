import argparse
import json
import logging
import subprocess
import threading
from pathlib import Path
from typing import Final

from core.embeddings import TextEmbeddingsAgent
from core.embeddings.minilm import MODEL_CHOICES as EMB_MODELS
from core.speech_to_text import SpeechToTextAgent
from core.speech_to_text.moonshine import MODEL_CHOICES as STT_MODELS
from core.translation import TextTranslationAgent
from core.translation.opus_mt import MODEL_CHOICES as TT_MODELS
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

    def translate_output(answer: str):

        def _translate():
            translated = tt_agent.translate(answer)
            print(GREEN + f"Agent: {translated}" + RESET + f" ({tt_agent.last_infer_time * 1000:.3f} ms)")

        threading.Thread(target=_translate).start()

    def handle_speech_input(transcribed_text: str):

        print(YELLOW + f"Query: {transcribed_text}" + RESET + f" ({stt_agent.last_infer_time * 1000:.3f} ms)")
        result = text_agent.answer_query(transcribed_text)
        answer, similarity, emb_infer_time = result["answer"], result["similarity"], result["infer_time"]
        answer = replace_tool_tokens(answer, tools)
        print(GREEN + f"Agent: {answer}" + RESET + f" ({emb_infer_time * 1000:.3f} ms, Similarity: {similarity:.6f})")
        translate_output(answer)
        wav_path = tts_agent.synthesize(answer)
        stt_agent.audio_manager.play(wav_path)

    tools_path = Path("data/tools.json")
    with open(tools_path, "r") as f:
        tools = json.load(f)

    text_agent = TextEmbeddingsAgent(
        args.emb_model, args.qa_file,
        n_threads=args.threads,
        eager_load=not args.no_eager_load
    )
    stt_agent = SpeechToTextAgent(
        args.stt_model, handle_speech_input,
        n_threads=args.threads,
        threshold=args.threshold,
        min_silence_duration_ms=args.silence_ms,
        eager_load=not args.no_eager_load
    )
    tt_agent = TextTranslationAgent(
        "en", args.tt_lang, args.tt_model,
        n_threads=args.threads,
        n_beams=args.num_beams,
        eager_load=not args.no_eager_load
    )
    tts_agent = TextToSpeechAgent()
    try:
        stt_agent.run()
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        text_agent.cleanup()
        stt_agent.cleanup()
        tt_agent.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q&A AI Assistant")
    parser.add_argument(
        "--logging",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity: %(choices)s (default: %(default)s)"
    )
    parser.add_argument(
        "-j", "--threads",
        type=int,
        help="Number of cores to use for CPU execution (default: all)"
    )
    parser.add_argument(
        "--no-eager-load",
        action="store_true",
        default=False,
        help="Do not eager load models: less initial memory usage but slower initial inference"
    )
    emb_args = parser.add_argument_group("embeddings options")
    emb_args.add_argument(
        "--qa-file",
        type=str,
        default=DEFAULT_QA_FILE,
        help="Path to Question-Answer pairs (default: %(default)s)"
    )
    emb_args.add_argument(
        "--emb-model",
        type=str,
        choices=EMB_MODELS,
        default="synap-quantized",
        help="Text embeddings model (default: %(default)s)"
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
    stt_args.add_argument(
        "--stt-model",
        type=str,
        choices=STT_MODELS,
        default="synap-tiny-float",
        help="Speech-to-text model (default: %(default)s)"
    )
    tt_args = parser.add_argument_group("text translation options")
    tt_args.add_argument(
        "--tt-lang",
        type=str,
        default="zh",
        help="Target language for translation"
    )
    tt_args.add_argument(
        "--tt-model",
        type=str,
        choices=[m for m in TT_MODELS if "synap" in m],
        default="synap-quantized",
        help="Text translation model (default: %(default)s)"
    )
    tt_args.add_argument(
        "--num-beams",
        type=int,
        help="Specify number of beams to use for decoding beam search"
    )
    args = parser.parse_args()

    configure_logging(args.logging)
    logger = logging.getLogger(__name__)
    logger.info("Initializing assistant...")

    main()
