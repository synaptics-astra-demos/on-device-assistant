import argparse
import logging
from typing import Final

from core.speech_to_text import SpeechToTextAgent
from core.speech_to_text.moonshine import MODEL_CHOICES

from ._utils import (
    add_input_args,
    configure_logging,
    format_answer
)

MODEL_SIZES: Final[str] = ["base", "tiny"]
QUANT_TYPES: Final[str] = ["float", "quantized"]
DEFAULT_SPEECH_THRESH: Final[float] = 0.5
DEFAULT_SILENCE_DUR_MS: Final[int] = 300


def main():

    def _speech_output_handler(transcribed_text: str):
        print(format_answer(transcribed_text, agent.last_infer_time, agent_name="Transcribed"))

    model_type, model_size, model_quant = args.model.split("-")

    agent = SpeechToTextAgent(
        model_size, model_quant, _speech_output_handler, 
        cpu_only=model_type=="onnx", 
        n_threads=args.threads,
        threshold=args.threshold,
        min_silence_duration_ms=args.silence_ms
    )

    if args.inputs:
        for wav in args.inputs:
            transcribed = agent.transcribe_wav(wav)
            print(format_answer(transcribed, agent.last_infer_time, agent_name="Transcribed"))
    else:
        agent.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q&A AI Assistant")
    parser.add_argument(
        "-i", "--inputs",
        type=str,
        metavar="WAV",
        nargs="+",
        help="Infer with provied WAV files instead of speech"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        metavar="MODEL",
        choices=MODEL_CHOICES,
        default="synap-tiny-quantized",
        help="Moonshine model to use (default: %(default)s), available:\n%(choices)s"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SPEECH_THRESH,
        help="Speech threshold, increase to lower mic capture sensitivity (default: %(default)s)"
    )
    parser.add_argument(
        "--silence-ms",
        type=int,
        default=DEFAULT_SILENCE_DUR_MS,
        help="Length of silence that determines end of speech (default: %(default)s ms)"
    )
    add_input_args(parser)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    configure_logging(logger, args.logging)
    logger.info("Starting demo...")

    main()

