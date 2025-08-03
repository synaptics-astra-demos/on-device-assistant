import argparse
import logging
from typing import Final

from core.speech_to_text import SpeechToTextAgent
from core.speech_to_text.moonshine import MODEL_CHOICES

from ._utils import (
    add_common_args,
    configure_logging,
    format_answer
)

DEFAULT_SPEECH_THRESH: Final[float] = 0.5
DEFAULT_SILENCE_DUR_MS: Final[int] = 300


def main():

    def _speech_output_handler(transcribed_text: str):
        print(format_answer(transcribed_text, agent.last_infer_time, agent_name="Transcribed"))

    agent = SpeechToTextAgent(
        args.model, _speech_output_handler,
        n_threads=args.threads,
        threshold=args.threshold,
        min_silence_duration_ms=args.silence_ms
    )

    try:
        if args.inputs:
            for wav in args.inputs:
                transcribed = agent.transcribe_wav(wav)
                print(format_answer(transcribed, agent.last_infer_time, agent_name="Transcribed"))
        else:
            agent.run()
    except KeyboardInterrupt:
        agent.cleanup()
        logger.info("Stopped by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Moonshine Demo")
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
    add_common_args(parser)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    configure_logging(logger, args.logging)
    logger.info("Starting demo...")

    main()

