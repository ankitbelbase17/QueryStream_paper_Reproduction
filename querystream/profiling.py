"""Wall-clock profiling for the QueryStream evaluation harness.

Timings are split along the boundaries that matter for a streaming system:

    decode        video demux + 1-FPS sampling            (per frame)
    openclip      the always-on encoder E                 (per frame)
    vision_tower  Qwen's ViT, per 2-frame temporal plane   (per frame)
    qdp_rtar      the pruning + gating logic itself        (per frame)
    llm_prefill   time to first token (TTFT)               (per response)
    llm_decode    remaining generation                     (per response)

`decode`, `openclip` and `vision_tower` are *video*-scoped: StreamingBench asks
several questions of one video, and none of those three stages depends on the
query, so they are paid once per video and shared. `qdp_rtar`, `llm_prefill` and
`llm_decode` are *question*-scoped. The CSV keeps the two scopes in separate
rows rather than silently amortising one into the other.

CUDA is asynchronous, so every timer that wraps GPU work synchronises before
reading the clock. That costs a little throughput and is the only way the
numbers mean anything.
"""

from __future__ import annotations

import csv
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import torch


def _sync(device: Optional[torch.device]) -> None:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


class Stopwatch:
    """Accumulates elapsed milliseconds per named stage."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device
        self.totals: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    @contextmanager
    def stage(self, name: str, n: int = 1) -> Iterator[None]:
        _sync(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync(self.device)
            dt = (time.perf_counter() - t0) * 1e3
            self.totals[name] = self.totals.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + n

    def add(self, name: str, ms: float, n: int = 1) -> None:
        """Record a timing measured elsewhere (e.g. inside the backbone)."""
        self.totals[name] = self.totals.get(name, 0.0) + ms
        self.counts[name] = self.counts.get(name, 0) + n

    def ms(self, name: str) -> float:
        return self.totals.get(name, 0.0)

    def per_unit(self, name: str) -> float:
        c = self.counts.get(name, 0)
        return self.totals.get(name, 0.0) / c if c else 0.0

    def merge(self, other: "Stopwatch") -> None:
        for k, v in other.totals.items():
            self.totals[k] = self.totals.get(k, 0.0) + v
        for k, v in other.counts.items():
            self.counts[k] = self.counts.get(k, 0) + v

    def reset(self) -> None:
        self.totals.clear()
        self.counts.clear()


# The exact column set requested for profile_summary.csv, plus the context
# needed to interpret a row. Order is the file's column order.
PROFILE_COLUMNS: List[str] = [
    "scope",                      # video | question | OVERALL
    "video_id",
    "question_id",
    "task_code",
    "n_frames",
    "n_planes",
    "decode_ms_per_frame",
    "openclip_ms_per_frame",
    "vision_tower_ms_per_frame",
    "qdp_rtar_ms_per_frame",
    "llm_prefill_ms",
    "llm_decode_ms",
    "llm_decode_ms_per_token",
    "n_new_tokens",
    "visual_tokens_dense",
    "visual_tokens_kept",
    "avg_keep_rate",
    "total_wall_clock_sec",
]


@dataclass
class ProfileRow:
    scope: str
    video_id: str = ""
    question_id: str = ""
    task_code: str = ""
    n_frames: int = 0
    n_planes: int = 0
    decode_ms_per_frame: float = 0.0
    openclip_ms_per_frame: float = 0.0
    vision_tower_ms_per_frame: float = 0.0
    qdp_rtar_ms_per_frame: float = 0.0
    llm_prefill_ms: float = 0.0
    llm_decode_ms: float = 0.0
    llm_decode_ms_per_token: float = 0.0
    n_new_tokens: int = 0
    visual_tokens_dense: int = 0
    visual_tokens_kept: int = 0
    avg_keep_rate: float = 0.0
    total_wall_clock_sec: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d


class ProfileWriter:
    """Streams ProfileRows to CSV and keeps running totals for the OVERALL row."""

    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "w", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=PROFILE_COLUMNS)
        self._w.writeheader()
        self._fh.flush()
        self._agg = Stopwatch()
        self._rows = 0
        self._t0 = time.perf_counter()
        self._dense = 0
        self._kept = 0
        self._frames = 0
        self._planes = 0
        self._questions = 0
        self._new_tokens = 0

    def write(self, row: ProfileRow) -> None:
        self._w.writerow(row.as_dict())
        self._rows += 1
        if self._rows % 25 == 0:
            self._fh.flush()

    def accumulate_video(self, sw: Stopwatch, n_frames: int, n_planes: int) -> None:
        self._agg.merge(sw)
        self._frames += n_frames
        self._planes += n_planes

    def accumulate_question(
        self, sw: Stopwatch, dense: int, kept: int, new_tokens: int
    ) -> None:
        self._agg.merge(sw)
        self._questions += 1
        self._dense += dense
        self._kept += kept
        self._new_tokens += new_tokens

    def close(self) -> ProfileRow:
        wall = time.perf_counter() - self._t0
        a = self._agg
        overall = ProfileRow(
            scope="OVERALL",
            video_id=f"{self._rows} rows",
            question_id=f"{self._questions} questions",
            n_frames=self._frames,
            n_planes=self._planes,
            decode_ms_per_frame=a.per_unit("decode"),
            openclip_ms_per_frame=a.per_unit("openclip"),
            vision_tower_ms_per_frame=a.per_unit("vision_tower"),
            qdp_rtar_ms_per_frame=a.per_unit("qdp_rtar"),
            llm_prefill_ms=a.per_unit("llm_prefill"),
            llm_decode_ms=a.per_unit("llm_decode"),
            llm_decode_ms_per_token=(
                a.ms("llm_decode") / self._new_tokens if self._new_tokens else 0.0
            ),
            n_new_tokens=self._new_tokens,
            visual_tokens_dense=self._dense,
            visual_tokens_kept=self._kept,
            avg_keep_rate=(self._kept / self._dense) if self._dense else 0.0,
            total_wall_clock_sec=wall,
        )
        self._w.writerow(overall.as_dict())
        self._fh.flush()
        self._fh.close()
        return overall


__all__ = ["Stopwatch", "ProfileRow", "ProfileWriter", "PROFILE_COLUMNS"]
