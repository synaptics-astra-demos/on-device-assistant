import argparse
import json
import logging
import re
import subprocess
import threading
from contextlib import ExitStack
from pathlib import Path
from typing import Callable, Final

from core.embeddings import TextEmbeddingsAgent
from core.embeddings.minilm import MODEL_CHOICES as EMB_MODELS
from core.speech_to_text import SpeechToTextAgent
from core.speech_to_text.moonshine import MODEL_CHOICES as STT_MODELS
from core.translation import TextTranslationAgent
from core.translation.opus_mt import MODEL_CHOICES as TT_MODELS
from core.text_to_speech import TextToSpeechAgent, MODEL_CHOICES as TTS_MODELS
from core.utils.device import validate_cpu_only

DEFAULT_QA_FILE: Final = "data/qa_dishwasher.json"
DEFAULT_SPEECH_THRESH: Final = 0.5
DEFAULT_SILENCE_DUR_MS: Final = 300
DEFAULT_SIMILARITY_THRESHOLD: Final = 0.4

# text colors
YELLOW: Final = "\033[93m"
GREEN: Final = "\033[32m"
RESET: Final = "\033[0m"


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


def get_embeddings(query: str, emb_agent: TextEmbeddingsAgent) -> str:
    result = emb_agent.answer_query(query)
    answer, similarity, emb_infer_time = (
        result["answer"],
        result["similarity"],
        result["infer_time"],
    )

    if similarity < args.sim_threshold:
        answer = "Sorry I don't know the answer"

    answer = replace_tool_tokens(answer, tools)
    print(
        GREEN
        + f"Agent: {answer}"
        + RESET
        + f" ({emb_infer_time * 1000:.3f} ms, Similarity: {similarity:.6f})"
    )
    return answer


def handle_speech_input(
    transcribed_text: str,
    stt_agent: SpeechToTextAgent,
    post_proc_fn: Callable[[str], None] | None = None,
):
    print(
        YELLOW
        + f"Query: {transcribed_text}"
        + RESET
        + f" ({stt_agent.last_infer_time * 1000:.3f} ms)"
    )
    if post_proc_fn:
        post_proc_fn(transcribed_text)


def synthesize_text(text: str, tts_agent: TextToSpeechAgent):
    tts_agent.synthesize(text, play_audio=True)


def translate_output(text: str, tt_agent: TextTranslationAgent):
    def _translate():
        translated = tt_agent.translate(text)
        print(
            GREEN
            + f"Agent: {translated}"
            + RESET
            + f" ({tt_agent.last_infer_time * 1000:.3f} ms)"
        )

    threading.Thread(target=_translate).start()


def handle_input(
    text: str,
    emb_agent: TextEmbeddingsAgent | None,
    tt_agent: TextTranslationAgent | None,
    tts_agent: TextToSpeechAgent | None,
):
    text = re.sub(r'[^a-z0-9\s]', '', text)
    answer = get_embeddings(text, emb_agent) if emb_agent else text
    if tt_agent:
        translate_output(answer, tt_agent)
    if tts_agent:
        synthesize_text(answer, tts_agent)


def main():

    eager_load = not args.no_eager_load
    threads = args.threads
    cpu_only = validate_cpu_only(args.cpu_only)

    emb_agent = None
    if not args.no_emb:
        emb_agent = TextEmbeddingsAgent(
            args.emb_model,
            args.qa_file,
            n_threads=threads,
            eager_load=eager_load,
            cpu_only=cpu_only,
        )

    tt_agent = None
    if args.tt_model:
        tt_agent = TextTranslationAgent(
            "en",
            args.tt_lang,
            args.tt_model,
            n_threads=threads,
            n_beams=args.num_beams,
            eager_load=eager_load,
            cpu_only=cpu_only,
        )

    tts_agent = TextToSpeechAgent(
        args.tts_model,
        args.tts_voice
    )

    stt_agent = None
    if not args.no_stt:
        def _stt_handler(transcribed: str):
            print(
                YELLOW
                + f"Query: {transcribed}"
                + RESET
                + f" ({stt_agent.last_infer_time * 1000:.3f} ms)"
            )
            handle_input(transcribed, emb_agent, tt_agent, tts_agent)

        stt_agent = SpeechToTextAgent(
            args.stt_model,
            _stt_handler,
            audio_manager=tts_agent.audio_manager,
            n_threads=threads,
            threshold=args.threshold,
            min_silence_duration_ms=args.silence_ms,
            eager_load=eager_load,
            cpu_only=cpu_only,
        )

    with ExitStack() as stack:
        # register cleanups in one place
        if emb_agent:
            stack.callback(emb_agent.cleanup)
        if stt_agent:
            stack.callback(stt_agent.cleanup)
        if tt_agent:
            stack.callback(tt_agent.cleanup)

        try:
            if args.cpu_only:
                logger.info("Running all models on CPU")
            if stt_agent:
                stt_agent.run()
            else:
                while True:
                    text = input("Input: ")
                    handle_input(text, emb_agent, tt_agent, tts_agent)
        except KeyboardInterrupt:
            logger.info("Stopped by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q&A AI Assistant")
    parser.add_argument(
        "--logging",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity: %(choices)s (default: %(default)s)",
    )
    parser.add_argument(
        "-j",
        "--threads",
        type=int,
        help="Number of cores to use for CPU execution (default: all)",
    )
    parser.add_argument(
        "--no-eager-load",
        action="store_true",
        default=False,
        help="Do not eager load models: less initial memory usage but slower initial inference",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        default=None,
        help="Run all models on the CPU (default for SL1620)",
    )
    emb_args = parser.add_argument_group("embeddings options")
    emb_args.add_argument(
        "--qa-file",
        type=str,
        default=DEFAULT_QA_FILE,
        help="Path to Question-Answer pairs (default: %(default)s)",
    )
    emb_args.add_argument(
        "--emb-model",
        type=str,
        choices=EMB_MODELS,
        default="synap-quantized",
        help="Text embeddings model (default: %(default)s)",
    )
    emb_args.add_argument(
        "--no-emb",
        action="store_true",
        default=False,
        help="Disable text embeddings"
    )
    emb_args.add_argument(
        "--sim-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="Specify similarity threshold of the answer",
    )
    stt_args = parser.add_argument_group("speech-to-text options")
    stt_args.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SPEECH_THRESH,
        help="Speech threshold, increase to lower mic capture sensitivity (default: %(default)s)",
    )
    stt_args.add_argument(
        "--silence-ms",
        type=int,
        default=DEFAULT_SILENCE_DUR_MS,
        help="Length of silence that determines end of speech (default: %(default)s ms)",
    )
    stt_args.add_argument(
        "--stt-model",
        type=str,
        choices=STT_MODELS,
        default="onnx-base-float",
        help="Speech-to-text model (default: %(default)s)",
    )
    stt_args.add_argument(
        "--no-stt",
        action="store_true",
        default=False,
        help="Disable Speech-To-Text"
    )
    tt_args = parser.add_argument_group("text translation options")
    tt_args.add_argument(
        "--tt-lang", type=str, default="zh", help="Target language for translation"
    )
    tt_args.add_argument(
        "--tt-model",
        type=str,
        choices=[m for m in TT_MODELS if "synap" in m],
        default=None,
        help="Text translation model (default: %(default)s)",
    )
    tt_args.add_argument(
        "--num-beams",
        type=int,
        help="Specify number of beams to use for decoding beam search",
    )
    tts_args = parser.add_argument_group("text-to-speech options")
    tts_args.add_argument(
        "--tts-model",
        type=str,
        choices=TTS_MODELS,
        default="piper",
        help="Text-to-speech model (default: %(default)s)"
    )
    tts_args.add_argument(
        "--tts-voice",
        type=str,
        help="Voice for text-to-speech model"
    )
    args = parser.parse_args()

    configure_logging(args.logging)
    logger = logging.getLogger(__name__)
    logger.info("Initializing assistant...")

    tools_path = Path("data/tools.json")
    with open(tools_path, "r") as f:
        tools = json.load(f)

    main()
