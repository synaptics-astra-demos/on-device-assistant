import logging
import subprocess

logger = logging.getLogger(__name__)


def get_SoC() -> str | None:
    try:
        soc = subprocess.check_output(['cat', '/etc/hostname']).decode().strip()
        logger.info("Detected SoC: %s", soc)
        if soc not in ["sl1680", "sl1640", "sl1620"]:
            logger.warning("Unknown SoC: %s", soc)
            return None
        return soc
    except subprocess.CalledProcessError():
        logger.warning("Failed to detect SoC")
        return None


def has_npu() -> bool:
    soc = get_SoC()
    if not soc:
        logger.warning("Invalid SoC, defaulting to CPU execution")
        return False
    if soc == "sl1620":
        logger.info("Detected SoC SL1620, switching to CPU execution")
        return False
    return True


def validate_cpu_only(cpu_only: bool | None) -> bool:
    npu_available = has_npu()
    if cpu_only is None:
        cpu_only = not npu_available
    elif not npu_available and not cpu_only:
        logger.warning("NPU not available, switching to CPU execution")
        cpu_only = True
    return cpu_only


if __name__ == "__main__":
    pass
