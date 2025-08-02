import argparse
import logging

from core.translation import TextTranslationAgent
from core.translation.opus_mt import MODEL_CHOICES

from ._utils import (
    add_common_args,
    configure_logging,
    format_answer
)


def main():
    agent = TextTranslationAgent(
        args.source_lang, args.dest_lang, args.model,
        n_beams=args.num_beams,
        n_threads=args.threads
    )

    try:
        while True:
            text = input("Text: ")
            translated = agent.translate(text)
            print(format_answer(translated, agent.last_infer_time, agent_name="Translated"))
    except (KeyboardInterrupt, EOFError):
        logger.info("Stopped by user.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q&A AI Assistant")
    parser.add_argument(
        "-s", "--source-lang",
        type=str,
        default="en",
        help="Source language code (default: %(default)s)"
    )
    parser.add_argument(
        "-d", "--dest-lang",
        type=str,
        default="fr",
        help="Destination language code (default: %(default)s)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        metavar="MODEL",
        choices=MODEL_CHOICES,
        default="synap-quantized",
        help="Model type to use for inference (available:\n%(choices)s)"
    )
    parser.add_argument(
        "-b", "--num-beams",
        type=int,
        help="Specify number of beams to use for decoding beam search"        
    )
    add_common_args(parser)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    configure_logging(logger, args.logging)
    logger.info("Starting demo...")

    main()

