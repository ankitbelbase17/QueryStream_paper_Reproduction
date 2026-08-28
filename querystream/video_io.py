"""Frame sampling for streaming evaluation.

`decord` is not installed on this cluster (and its lmms-eval pin conflicts with
the installed numpy 2.x), so this uses OpenCV with a PyAV fallback. Both return
the same thing: RGB uint8 [T, 3, H, W] sampled at a target FPS.

Sampling policy: nearest source frame to each target timestamp, which is what
1-FPS streaming evaluation means. Frame selection lives here and NOT in the
pipeline, so a benchmark harness can substitute its own sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch


@dataclass
class VideoMeta:
    path: str
    native_fps: float
    n_frames_native: int
    duration_s: float
    height: int
    width: int
    sampled_fps: float
    n_frames_sampled: int

    def summary(self) -> dict:
        return {
            "path": self.path,
            "native_fps": round(self.native_fps, 3),
            "duration_s": round(self.duration_s, 3),
            "resolution": f"{self.width}x{self.height}",
            "sampled_fps": self.sampled_fps,
            "n_frames_sampled": self.n_frames_sampled,
        }


def read_video_frames(
    path: str,
    *,
    target_fps: float = 1.0,
    max_frames: Optional[int] = None,
    backend: str = "auto",
) -> Tuple[torch.Tensor, VideoMeta]:
    """Sample a video at `target_fps`.

    Returns:
        (frames [T, 3, H, W] uint8 RGB, VideoMeta)
    """
    if backend in ("auto", "cv2"):
        try:
            return _read_cv2(path, target_fps, max_frames)
        except Exception:
            if backend == "cv2":
                raise
    return _read_av(path, target_fps, max_frames)


def _read_cv2(path: str, target_fps: float, max_frames: Optional[int]):
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    duration = total / native if native > 0 else 0.0

    step = max(native / target_fps, 1.0)
    want = [int(round(i * step)) for i in range(int(total / step))] if total else []
    if max_frames:
        want = want[:max_frames]
    want_set = set(want)

    out: List[np.ndarray] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want_set:
            out.append(frame[:, :, ::-1].copy())          # BGR -> RGB
            if max_frames and len(out) >= max_frames:
                break
        idx += 1
    cap.release()
    if not out:
        raise RuntimeError(f"decoded 0 frames from {path}")

    arr = torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).contiguous()
    return arr, VideoMeta(path, native, total, duration, h, w, target_fps, arr.shape[0])


def _read_av(path: str, target_fps: float, max_frames: Optional[int]):
    import av

    with av.open(path) as c:
        s = c.streams.video[0]
        native = float(s.average_rate or 30.0)
        h, w = s.height, s.width
        total = s.frames or 0
        duration = float(s.duration * s.time_base) if s.duration else (total / native)
        step = max(native / target_fps, 1.0)
        out, idx, nxt = [], 0, 0.0
        for frame in c.decode(video=0):
            if idx >= nxt:
                out.append(frame.to_ndarray(format="rgb24"))
                nxt += step
                if max_frames and len(out) >= max_frames:
                    break
            idx += 1
    if not out:
        raise RuntimeError(f"decoded 0 frames from {path}")
    arr = torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).contiguous()
    return arr, VideoMeta(path, native, total, duration, h, w, target_fps, arr.shape[0])


__all__ = ["read_video_frames", "VideoMeta"]
