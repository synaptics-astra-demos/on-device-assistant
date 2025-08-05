import argparse
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ._sys_usage import SystemProfiler


@dataclass
class ProfilingStat:
    name: str
    value: Any
    unit: str | None = None

    def __repr__(self):
        unit = f" {self.unit}" if self.unit else ""
        return f"{self.name}: {self.value}{unit}"


class ProfilerBase(ABC):

    def __init__(
        self,
        models_names: list[str],
        logger: logging.Logger,
        *,
        run_forever: bool,
        n_threads: int | None
    ):
        self._model_names = models_names
        self._run_forever = run_forever
        self._logger = logger
        self._sys_prof = SystemProfiler(interval_ms=25, hist_length=1000)

        self._all_stats: dict[str, dict[str, ProfilingStat | dict[str, ProfilingStat]]] = {
            "environment": {
                "max_threads": ProfilingStat("Max CPU threads", n_threads)
            },
            "infer_stats": {
                model_name: {
                    "n_iters": ProfilingStat("Num inferences", 0),
                    "tot_infer_time": ProfilingStat("Total inference time", 0, "ms"),
                    "avg_infer_time": ProfilingStat("Average inference time", 0, "ms"),
                    "avg_sys_usage": {}
                } for model_name in self._model_names
            },
        }

    @abstractmethod
    def _get_inference_time(self, model_name: str) -> float:
        ...

    @abstractmethod
    def _cleanup(self, model_name: str):
        ...

    def _update_env_param(self, param: str, value: ProfilingStat):
        self._all_stats["environment"].update({param: value})

    def _update_sys_usage(self, model_name: str):
        sys_usage = self._all_stats["infer_stats"][model_name]["avg_sys_usage"]
        for cpu, usage in self._sys_prof.avg_cpu_usage().items():
            sys_usage[cpu] = ProfilingStat(cpu.upper(), round(100 * usage, 2), "%")
        sys_usage["npu"] = ProfilingStat("NPU", round(100 * self._sys_prof.avg_npu_usage(), 2), "%")
        sys_usage["mem"] = ProfilingStat("RAM", round(self._sys_prof.avg_mem_usage() / 1_048_576, 2), "GB")
        self._sys_prof.reset()

    def profile_models(self, n_iters: int, print_stats: bool = True) -> dict[str, dict]:

        def _cleanup_models():
            for model_name in self._model_names:
                self._cleanup(model_name)

        self._sys_prof.start()
        try:
            while True:
                for model_name in self._model_names:
                    self._logger.info(f"Profiling '{model_name}' ({n_iters} iters)...")
                    infer_stats: dict[str, ProfilingStat] = self._all_stats["infer_stats"][model_name]
                    try:
                        for _ in range(n_iters):
                            infer_stats["tot_infer_time"].value += self._get_inference_time(model_name)
                            infer_stats["n_iters"].value += 1
                        self._update_sys_usage(model_name)
                    except Exception as e:
                        self._logger.warning(f"Stopping inference due to error: {e}")
                        break
                if not self._run_forever:
                    break
        except KeyboardInterrupt:
            print("Stopped by user.")
        finally:
            _cleanup_models()
            self._sys_prof.stop()
    
        for model_name, infer_stats in self._all_stats["infer_stats"].items():
            n_iters = infer_stats["n_iters"].value
            infer_stats["avg_infer_time"].value = round(infer_stats["tot_infer_time"].value / (n_iters or 1)  * 1000, 3)
            infer_stats["tot_infer_time"].value = round(infer_stats["tot_infer_time"].value * 1000, 3)
        if print_stats:
            self.print_stats()
        return self._all_stats

    def print_stats(self):
        SPACER = " " * 4
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        print("\n\nProfiling report")
        print("------------------------------------")
        print("Environment:")
        for env_var in self._all_stats["environment"].values():
            print(SPACER + str(env_var))
        for model_name, infer_stats in self._all_stats["infer_stats"].items():
            print(f"\nStats for '{model_name}':")
            for stat_name, stat in infer_stats.items():
                if isinstance(stat, dict) and stat_name == "avg_sys_usage":
                    print(SPACER + "System usage:")
                    print(2 * SPACER + YELLOW + "NOTE: Measurements are affected by other running processes" + RESET)
                    for sys_stat in stat.values():
                        print(2 * SPACER + str(sys_stat))
                else:
                    print(SPACER + str(stat))
        print()


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    model_choices: list[str],
    default_model: str | list[str],
    default_input: str,
    input_desc: str
):
    if isinstance(default_model, str):
        default_model: list[str] = [default_model]
    parser.add_argument(
        "-m", "--models",
        type=str,
        metavar="MODEL",
        nargs="+",
        choices=model_choices,
        default=default_model,
        help="MiniLM models to profile (default: %(default)s, available: %(choices)s)"
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
        default=default_input,
        help=input_desc + " (default: \"%(default)s\")"
    )
    parser.add_argument(
        "-j", "--threads",
        type=int,
        help="Number of cores to use for CPU execution (default: all)"
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
