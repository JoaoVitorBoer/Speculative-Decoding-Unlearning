import json
import time
from pathlib import Path

try:
    import torch
    _CUDA = torch.cuda.is_available()
except ImportError:
    torch = None  # type: ignore
    _CUDA = False


class Profiler:
    """Accumulates wall-clock timings and VRAM peaks."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self.vram_peak_mb: dict[str, float] = {}
        self._t0: float = 0.0
        self._stage: str = ""

    def start(self, stage: str) -> None:
        self._stage = stage
        if _CUDA:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        if _CUDA:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._t0
        self.timings[self._stage] = round(elapsed, 3)
        if _CUDA:
            self.vram_peak_mb[self._stage] = round(
                torch.cuda.max_memory_allocated() / 1e6, 1
            )
            print(
                f"[profiler] {self._stage:<30}  {elapsed:.2f}s  "
                f"VRAM={self.vram_peak_mb[self._stage]:.0f}MB"
            )
        else:
            print(f"[profiler] {self._stage:<30}  {elapsed:.2f}s")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.summary(), indent=2))
        print(f"[profiler] saved → {path}")

    def summary(self) -> dict:
        return {
            "timings_sec": self.timings,
            "vram_peak_mb": self.vram_peak_mb,
        }
