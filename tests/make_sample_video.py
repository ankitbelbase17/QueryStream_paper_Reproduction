#!/usr/bin/env python3
"""Generate a 10 s sample .mp4 with the paper's Appendix A.7 structure.

Timeline at 30 fps (10 s), so 10 frames survive 1 FPS sampling:

    t = 0-3 s   kitchen-ish scene, a GREEN BOTTLE slides left->right
                (the query-relevant object, moving)
    t = 4 s     BLACK FRAMES -- the "transient shock": visually explosive,
                semantically irrelevant. A change-is-important trigger fires
                here; QueryStream must stay silent.
    t = 5-7 s   scene returns, bottle continues
    t = 8-9 s   the bottle's CAP lifts off slowly -- visually subtle but
                query-relevant. Frame-to-frame diffing misses it; the DSH
                accumulates it.

A RED DISTRACTOR square bounces in the lower-right throughout: it moves a lot
and is irrelevant to a bottle query, which is exactly the content M_temp keeps
and M_sem must reject.

    python tests/make_sample_video.py [-o path.mp4] [--seconds 10] [--fps 30]
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def render(seconds: int = 10, fps: int = 30, w: int = 640, h: int = 360) -> np.ndarray:
    """Returns [T, H, W, 3] uint8 BGR frames."""
    T = seconds * fps
    out = np.zeros((T, h, w, 3), dtype=np.uint8)
    rng = np.random.default_rng(0)
    # static-ish backdrop with mild texture so the scene is not perfectly flat
    backdrop = np.zeros((h, w, 3), dtype=np.uint8)
    backdrop[:, :] = (150, 160, 170)
    backdrop[: h // 3, :] = (110, 120, 135)                       # wall
    backdrop += (rng.normal(0, 3, backdrop.shape)).astype(np.int16).clip(-8, 8).astype(np.uint8)

    for t in range(T):
        sec = t / fps
        f = backdrop.copy()
        f += (rng.normal(0, 1.5, f.shape)).astype(np.int16).clip(-4, 4).astype(np.uint8)

        if 4.0 <= sec < 5.0:                                      # SHOCK
            out[t] = 0
            continue

        # ---- relevant object: a green bottle sliding left -> right ----
        prog = min(sec / 9.0, 1.0)
        bx = int(60 + prog * (w - 220))
        by = h // 2
        f[by - 70 : by + 40, bx : bx + 46] = (60, 170, 70)        # body (BGR)
        f[by - 95 : by - 70, bx + 14 : bx + 32] = (60, 170, 70)   # neck

        # ---- subtle relevant event: the cap lifts, slowly, after t=8 ----
        lift = 0 if sec < 8.0 else int((sec - 8.0) * 9)
        cy = by - 103 - lift
        f[cy : cy + 9, bx + 12 : bx + 34] = (40, 130, 200)        # cap

        # ---- irrelevant distractor: a bouncing red square ----
        dx = int(430 + 120 * np.sin(sec * 2.6))
        dy = int(250 + 45 * np.cos(sec * 3.4))
        f[dy : dy + 55, dx : dx + 55] = (55, 55, 205)

        out[t] = f
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="tests/data/sample_bottle.mp4")
    ap.add_argument("--seconds", type=int, default=10)
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()

    import cv2

    frames = render(a.seconds, a.fps)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    h, w = frames.shape[1:3]
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {a.out}")
    for f in frames:
        vw.write(f)
    vw.release()
    print(f"wrote {a.out}  ({a.seconds}s @ {a.fps}fps, {frames.shape[0]} frames, {w}x{h})")
    print(f"  size: {os.path.getsize(a.out) / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
