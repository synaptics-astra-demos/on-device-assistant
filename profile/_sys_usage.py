import queue
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque

logger = logging.getLogger("__name__")


@dataclass(slots=True)
class StatsSnapshot:
    ts: float
    cpu: dict[str, list[int]]
    npu_time_us: int
    mem_avail_kb: int

    def delta(self, prev: "StatsSnapshot") -> "StatsDelta":
        return StatsDelta.from_snaps(self, prev)

    def __repr__(self) -> str:
        return f"({self.ts}) -> cpu_ticks: {self.cpu}, npu_usage: {self.npu_time_us} us, mem_available: {self.mem_avail_kb} kB"


@dataclass(slots=True)
class StatsDelta:
    d_time: float
    d_cpu_total: dict[str, int]
    d_cpu_idle: dict[str, int]
    d_npu_time_us: int

    @classmethod
    def from_snaps(cls, new_snap: StatsSnapshot, old_snap: StatsSnapshot) -> "StatsDelta":
        d_cpu_total: dict[str, int] = {}
        d_cpu_idle: dict[str, int] = {}
        for proc, new in new_snap.cpu.items():
            old = old_snap.cpu[proc]
            d_cpu_total[proc] = sum(c2 - c1 for c1, c2 in zip(old, new))
            d_cpu_idle[proc] = (new[3] + new[4]) - (old[3] + old[4])
        return cls(
            new_snap.ts - old_snap.ts,
            d_cpu_total,
            d_cpu_idle,
            new_snap.npu_time_us - old_snap.npu_time_us
        )

    def __repr__(self) -> str:
        return (
            f"StatsDelta -> time: {self.d_time}, " 
            f"cpu_ticks_total: {self.d_cpu_total}, "
            f"cpu_ticks_idle: {self.d_cpu_idle}, "
            f"npu_usage: {self.d_npu_time_us} us"
        )


class StatsCollector:

    def __init__(
        self,
        cpu_info: str = "/proc/stat",
        mem_info: str = "/proc/meminfo",
        npu_info: str = "/sys/class/misc/synap/statistics/inference_time"
    ):
        self._cpu_info = Path(cpu_info)
        self._mem_info = Path(mem_info)
        self._npu_info = Path(npu_info)
        self._cpu_names: list[str] = []
        self._mem_total_kb: float | None = None
        with self._cpu_info.open("r") as f:
            for line in f:
                if not line.startswith("cpu"):
                    break
                self._cpu_names.append(line.split()[0])
        if not self._cpu_names:
            raise ValueError(f"Failed to parse cpu names from '{self._cpu_info}'")
        with self._mem_info.open("r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    self._mem_total_kb = int(line.split()[1])
        if self._mem_total_kb is None:
            raise ValueError(f"Failed to parse total available memory from '{self._mem_info}'")

    @property
    def cpu_names(self) -> list[str]:
        return self._cpu_names

    @property
    def mem_total_kb(self) -> int:
        return self._mem_total_kb

    def collect_once(self) -> StatsSnapshot:
        cpu_stats: dict[str, list[int]] = {}
        with self._cpu_info.open() as f:
            for i, line in enumerate(f):
                info = line.split()
                cpu_stats[info[0]] = list(map(int, info[1:]))
                if i >= 4:
                    break
        mem_avail_kb: int = 0
        with self._mem_info.open() as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    mem_avail_kb = int(line.split()[1])
                    break
            else:
                logger.warning(f"Failed to parse available available memory from '{self._mem_info}'")
        npu_time_us = int(self._npu_info.read_text())
        return StatsSnapshot(
            time.monotonic(),
            cpu_stats,
            npu_time_us,
            mem_avail_kb
        )


class SystemProfiler:

    def __init__(
        self,
        *,
        interval_ms: int = 100,
        hist_length: int = 100
    ):
        self._interval_s = interval_ms / 1000
        self._hist_length = hist_length
        
        self._collector = StatsCollector()
        self._total_mem = self._collector.mem_total_kb
        self._cpu_hist: dict[str, Deque[float]] = {
            cpu_name: deque(maxlen=self._hist_length)
            for cpu_name in self._collector.cpu_names
        }
        self._npu_hist: Deque[float] = deque(maxlen=self._hist_length)
        self._mem_hist: Deque[float] = deque(maxlen=self._hist_length)

        self._stats_queue: queue.Queue = queue.Queue(maxsize=8)
        self._prev_stats: StatsSnapshot | None = None
        self._lock: threading.Lock = threading.Lock()
        self._stop_evt: threading.Event = threading.Event()
        self._threads: list[threading.Thread] = []  

    def avg_cpu_usage(self) -> dict[str, float]:
        with self._lock:
            return {
                cpu_name: (sum(hist) / len(hist)) if hist else 0.0
                for cpu_name, hist in self._cpu_hist.items()
            }

    def avg_npu_usage(self) -> float:
        with self._lock:
            return (sum(self._npu_hist) / len(self._npu_hist)) if self._npu_hist else 0.0

    def avg_mem_usage(self) -> float:
        with self._lock:
            return sum(self._mem_hist) / len(self._mem_hist) if self._mem_hist else 0.0

    def start(self):
        t1 = threading.Thread(target=self._poll_loop, name="ProfilerPoll")
        t2 = threading.Thread(target=self._process_loop, name="ProfilerProc")
        t1.start()
        t2.start()
        self._threads.extend((t1, t2))

    def stop(self, *, timeout: float | None = None):
        self._stop_evt.set()
        self._stats_queue.put_nowait(None)
        for t in self._threads:
            t.join(timeout=timeout)

    def reset(self):
        with self._lock:
            for q in self._cpu_hist.values():
                q.clear()
            self._npu_hist.clear()
            self._mem_hist.clear()

    def _poll_loop(self):
        next_t = time.monotonic()
        while not self._stop_evt.is_set():
            now = time.monotonic()
            if now < next_t:
                remaining = next_t - now
                if self._stop_evt.wait(timeout=remaining):
                    break
            stats = self._collector.collect_once()
            try:
                self._stats_queue.put_nowait(stats)
            except queue.Full:
                _ = self._stats_queue.get_nowait()
                self._stats_queue.put_nowait(stats)
            next_t += self._interval_s

    def _process_loop(self):
        while True:
            curr_stats: StatsSnapshot | None = self._stats_queue.get()
            if curr_stats is None:
                break
            if self._prev_stats:
                diff = curr_stats.delta(self._prev_stats)
                d_time_s = diff.d_time
                if d_time_s:  # avoid div-by-zero for <1 ms dt
                    for proc in diff.d_cpu_total:
                        total = diff.d_cpu_total[proc]
                        idle = diff.d_cpu_idle[proc]
                        cpu_usage = 0.0 if total <= 0 else (1.0 - idle / total)
                        with self._lock:
                            self._cpu_hist[proc].append(cpu_usage)
                    npu_usage = (curr_stats.npu_time_us - self._prev_stats.npu_time_us) / (d_time_s * 1e6)
                    npu_usage = min(max(0.0, npu_usage), 1.0)
                    mem_usage = self._total_mem - curr_stats.mem_avail_kb
                    with self._lock:
                        self._npu_hist.append(npu_usage)
                        self._mem_hist.append(mem_usage)
            self._prev_stats = curr_stats
