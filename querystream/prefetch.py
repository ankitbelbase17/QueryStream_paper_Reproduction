"""Overlap video decoding with GPU work.

The Milestone-3 dry run measured frame decoding at **46 ms/frame** against
15.8 ms for Qwen's vision tower and 0.9-11 ms for OpenCLIP: the GPU spends most
of a run waiting on cv2. Decoding is CPU-bound and the node has 288 cores, so
decoding the NEXT videos while the current one is on the GPU converts a serial
(decode + encode) cost into max(decode / workers, encode).

Threads rather than processes: `cv2.VideoCapture.read()` releases the GIL, so
threads get real parallelism here, and a 1080p video at 1 FPS is a multi-GB uint8
array that would be expensive to pickle across a process boundary.

Order is preserved -- results come out in submission order regardless of which
worker finishes first -- because the evaluation writes one JSONL row per question
and a reordered stream would make resume-by-question_id harder to reason about.

Memory is the real bound, not CPU: one 1440-frame 1080p video is ~8.9 GB of
uint8. `depth` caps how many decoded videos may sit in the queue at once.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple

import torch

from .video_io import VideoMeta, read_video_frames


@dataclass
class DecodedVideo:
    item: Any                       # the caller's manifest entry, passed through
    frames: Optional[torch.Tensor]  # [T, 3, H, W] uint8 RGB, or None on failure
    meta: Optional[VideoMeta]
    decode_ms: float
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.frames is not None


class VideoPrefetcher:
    """Decode videos ahead of the consumer, in order, with a bounded queue."""

    def __init__(
        self,
        items: Sequence[Any],
        *,
        path_fn: Callable[[Any], str],
        max_seconds_fn: Callable[[Any], Optional[int]],
        fps: float = 1.0,
        workers: int = 4,
        depth: int = 3,
    ):
        self.items = list(items)
        self.path_fn = path_fn
        self.max_seconds_fn = max_seconds_fn
        self.fps = fps
        self.workers = max(1, workers)
        # The queue holds `depth` decoded videos; with `workers` more in flight
        # the true ceiling is depth + workers videos resident at once.
        self.depth = max(1, depth)
        self._sem = threading.Semaphore(self.depth)
        self._pool = ThreadPoolExecutor(max_workers=self.workers,
                                        thread_name_prefix="decode")
        self._stop = threading.Event()
        self.total_decode_ms = 0.0
        self.total_frames = 0

    def _decode(self, item: Any) -> DecodedVideo:
        # Block BEFORE doing the work, so an idle consumer cannot let the pool
        # run ahead and exhaust RAM.
        self._sem.acquire()
        if self._stop.is_set():
            return DecodedVideo(item, None, None, 0.0, "stopped")
        t0 = time.perf_counter()
        try:
            ms = self.max_seconds_fn(item)
            frames, meta = read_video_frames(
                self.path_fn(item), target_fps=self.fps,
                max_frames=None if ms is None else int(ms) + 1,
            )
            return DecodedVideo(item, frames, meta, (time.perf_counter() - t0) * 1e3)
        except Exception as exc:
            return DecodedVideo(item, None, None, (time.perf_counter() - t0) * 1e3,
                                f"{type(exc).__name__}: {exc}")

    def __iter__(self) -> Iterator[DecodedVideo]:
        """Yield decoded videos in submission order, holding at most a few at once.

        THE REFERENCE MUST BE DROPPED AFTER EACH YIELD. A `Future` keeps its
        result alive for as long as the Future object exists, so iterating a
        retained `futures` list pins EVERY decoded video in RAM for the whole
        run -- the semaphore bounds concurrent *decodes*, not retained
        *results*. At StreamingBench's 1080p / 504 s average one video is about
        3.1 GB of uint8 frames, so a 125-video shard accumulated ~390 GB and
        the kernel killed three of four workers on the node (job 3178994; the
        survivor was sitting at 332 GB RSS). OmniPro never hit it because
        720p / 189 s is ~0.5 GB per video.

        Clearing the slot before yielding, and deleting the local afterwards,
        caps residency at roughly `depth + workers` videos as intended.
        """
        futures = [self._pool.submit(self._decode, it) for it in self.items]
        try:
            for i in range(len(futures)):
                fut, futures[i] = futures[i], None    # drop the list's reference
                dv = fut.result()
                del fut                               # ... and the Future's
                self._sem.release()                   # frees a slot for the pool
                if dv.ok:
                    self.total_decode_ms += dv.decode_ms
                    self.total_frames += int(dv.frames.shape[0])
                yield dv
                del dv                                # consumer is done with it
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._stop.set()
        # Unblock any worker still parked on the semaphore so shutdown returns.
        for _ in range(self.workers + self.depth):
            self._sem.release()
        self._pool.shutdown(wait=False, cancel_futures=True)

    @property
    def ms_per_frame(self) -> float:
        """Aggregated decode cost. This is WORKER time, not wall clock -- with
        `workers > 1` the wall-clock cost per frame is roughly this over
        `workers`, which is the whole point."""
        return self.total_decode_ms / self.total_frames if self.total_frames else 0.0


__all__ = ["VideoPrefetcher", "DecodedVideo"]
