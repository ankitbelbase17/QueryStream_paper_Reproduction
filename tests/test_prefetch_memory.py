#!/usr/bin/env python3
"""Regression guard: the prefetcher must bound residency, not accumulate it.

    python tests/test_prefetch_memory.py

WHAT THIS CATCHES, AND WHY IT MATTERS
`VideoPrefetcher.__iter__` submits every item to the thread pool up front and
consumes the futures in order. A `Future` keeps its result alive for as long as
the Future object exists, so iterating a retained `futures` list pins EVERY
decoded video in RAM for the whole run. The bounding semaphore does not help:
it limits concurrent *decodes*, not retained *results*.

The failure is invisible on small or low-resolution inputs and lethal on large
ones. OmniPro (720p, 189 s mean -> ~0.5 GB of uint8 frames per video) ran the
whole 2,700-sample benchmark without trouble. StreamingBench (1080p, 504 s mean
-> ~3.1 GB per video, 125 videos per shard -> ~390 GB) killed three of four
workers on the node in job 3178994; the one survivor was sitting at 332 GB RSS
with GPUs 0-2 idle, which looks like a cluster problem rather than a leak.

No video files, no GPU, no network: `read_video_frames` is replaced by a stub
that hands back a fixed-size tensor, so the test measures the retention policy
itself.
"""

from __future__ import annotations

import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import querystream.prefetch as P

N_ITEMS = 20
MB_PER_ITEM = 200                      # 200 MiB per fake "video"
TOTAL_GB = N_ITEMS * MB_PER_ITEM / 1024


def _stub(path, target_fps=1.0, max_frames=None):
    return torch.zeros((MB_PER_ITEM, 1024, 1024), dtype=torch.uint8), object()


def _live_frame_gb() -> float:
    """Bytes held by live uint8 [T, H, W] tensors -- i.e. decoded frames."""
    total = 0
    for o in gc.get_objects():
        try:
            if isinstance(o, torch.Tensor) and o.dtype == torch.uint8 and o.dim() == 3:
                total += o.numel()
        except Exception:                # some objects dislike isinstance
            pass
    return total / 1024 ** 3


def peak_residency(workers: int = 2, depth: int = 2) -> float:
    pf = P.VideoPrefetcher(list(range(N_ITEMS)), path_fn=lambda i: f"v{i}",
                           max_seconds_fn=lambda i: None,
                           workers=workers, depth=depth)
    peak = 0.0
    for _ in pf:
        gc.collect()
        peak = max(peak, _live_frame_gb())
    return peak


def main() -> int:
    P.read_video_frames = _stub

    # A generous ceiling: depth + workers + the one in the consumer's hand,
    # with slack for GC timing. The leak is ~7x this, so the margin cannot
    # mask it.
    ceiling = (depth_plus := 2 + 2 + 1) * MB_PER_ITEM / 1024 * 1.2

    peak = peak_residency()
    print(f"  {N_ITEMS} items x {MB_PER_ITEM / 1024:.2f} GB "
          f"= {TOTAL_GB:.2f} GB if every result were retained")
    print(f"  peak live frame memory: {peak:.2f} GB   (ceiling {ceiling:.2f} GB)")

    ok = peak <= ceiling
    print(f"  {'ok  ' if ok else 'FAIL'} residency is bounded by "
          f"depth+workers, not by the number of videos")

    # Prove the check is meaningful by re-running the pre-fix iterator.
    def leaky(self):
        futures = [self._pool.submit(self._decode, it) for it in self.items]
        try:
            for fut in futures:
                dv = fut.result()
                self._sem.release()
                yield dv
        finally:
            self.shutdown()

    good = P.VideoPrefetcher.__iter__
    P.VideoPrefetcher.__iter__ = leaky
    try:
        leaked = peak_residency()
    finally:
        P.VideoPrefetcher.__iter__ = good
    detects = leaked > ceiling
    print(f"  pre-fix iterator peaks at {leaked:.2f} GB")
    print(f"  {'ok  ' if detects else 'FAIL'} the guard actually detects the leak")

    print(f"\n{int(ok) + int(detects)}/2 passed")
    return 0 if (ok and detects) else 1


if __name__ == "__main__":
    raise SystemExit(main())
