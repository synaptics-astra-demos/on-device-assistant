import argparse
import logging
from dataclasses import dataclass
from typing import Any, Final

@dataclass(frozen=True)
class InferenceStat:
    name: str
    value: Any
    unit: str | None = None

    def __repr__(self):
        unit = f" {self.unit}" if self.unit else ""
        return f"{self.name}: {self.value}{unit}"


def add_input_args(parser: argparse.ArgumentParser):
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


def configure_logging(logger: logging.Logger, verbosity: str):
    level = getattr(logging, verbosity.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {verbosity}")

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(handler)


def format_answer(
    answer: str,
    infer_time: float,
    stats: list[InferenceStat] | None = None,
    agent_name: str = "Agent"
) -> str:
    GREEN: Final[str] = "\033[32m"
    RESET: Final[str] = "\033[0m"
    result: str = GREEN + f"{agent_name}: {answer}" + RESET + f" ({infer_time * 1000:.3f} ms, "
    for stat in stats:
        result += str(stat)
    return result + ")"
