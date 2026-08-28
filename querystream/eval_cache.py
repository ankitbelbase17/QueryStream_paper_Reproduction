"""Per-video visual pass: everything in the pipeline that does not depend on Q.

StreamingBench asks ~5 questions of each video, each at a different timestamp.
Re-streaming the video once per question would cost 661k frame-encodes over the
500-video split; streaming it once per video costs 252k. The saving is exact,
not approximate, because three of the four heavy stages are query-independent:

    frame decoding          depends on the video only
    OpenCLIP patch features depends on the video only
    Qwen's vision tower     depends on the video only
    DSH temporal novelty    depends on the video only  <-- see below

Only `M_sem = 1[sim(q, v) > mean_j sim(q, v_j)]` involves the query, and that is
one mat-vec per frame.

WHY THE DSH IS QUERY-INDEPENDENT
    v_dsh,t = alpha * v_t + (1 - alpha) * v_dsh,t-1
    M_temp  = 1[ sim(v_t, v_dsh,t-1) < tau_temp ]
Neither line mentions q, so `temp_sim` can be computed once and replayed. This
holds ONLY under `QDPConfig.update_all_patches=True` (the default, and the
literal reading of the paper). With `update_all_patches=False` the EMA is
restricted to patches that survived M_QDP, which depends on M_sem, which depends
on q -- the cache is then invalid and `build_video_pass` refuses to run.

The cache holds normalised patch features, so the semantic similarity replayed
per question is bit-identical to what the streaming pipeline would compute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from .config import QueryStreamConfig
from .dsh_memory import DSHMemory
from .open_clip_wrapper import OpenCLIPWrapper
from .ops import l2_normalize
from .profiling import Stopwatch
from .qwen_wrapper import QwenVLWrapper
from .video_io import VideoMeta, read_video_frames


@dataclass
class PlaneCache:
    """One `temporal_patch_size`-frame Qwen visual-token plane."""

    tokens: torch.Tensor        # [M, hidden]
    mrope: torch.Tensor         # [M, 3] int64, ABSOLUTE on the stream timeline
    plane_idx: int
    frame_ids: Tuple[int, ...]
    timestamp: float            # timestamp of the plane's FIRST frame

    @property
    def num_tokens(self) -> int:
        return self.tokens.shape[0]


@dataclass
class VideoPass:
    """Query-independent state for one video, replayable by any query."""

    video_id: str
    video_path: str
    meta: VideoMeta
    n_frames: int
    clip_grid_hw: Tuple[int, int]
    llm_grid_hw: Optional[Tuple[int, int]]
    time_interval: int
    patch_feats: torch.Tensor       # [T, N, D] L2-normalised
    frame_pooled: torch.Tensor      # [T, D] fp32, mean over patches (\bar v_t)
    temp_sim: torch.Tensor          # [T, N] fp32, cos(v_t, v_dsh_{t-1})
    planes: List[PlaneCache] = field(default_factory=list)
    timings: Stopwatch = field(default_factory=Stopwatch)

    @property
    def tokens_per_plane(self) -> int:
        return self.planes[0].num_tokens if self.planes else 0

    def planes_up_to(self, timestamp_s: float) -> Tuple[List[PlaneCache], int, int]:
        """Planes fully observable at `timestamp_s`, under the streaming rule.

        Frame i carries timestamp i / fps, and a question stamped at t may see
        frames 0..t inclusive. Qwen merges frames in pairs, so a trailing odd
        frame cannot form a plane: it is reported as dropped rather than
        edge-padded, which would change the token count and corrupt the
        keep-rate denominator (Milestone 2 decision 1).

        Returns:
            (planes, n_frames_visible, n_frames_dropped_trailing)
        """
        n_vis = min(self.n_frames, int(timestamp_s) + 1)
        k = 2 if not self.planes else len(self.planes[0].frame_ids)
        n_planes = n_vis // k
        return self.planes[:n_planes], n_vis, n_vis - n_planes * k

    def memory_mb(self) -> float:
        b = self.patch_feats.numel() * self.patch_feats.element_size()
        b += self.temp_sim.numel() * self.temp_sim.element_size()
        b += self.frame_pooled.numel() * self.frame_pooled.element_size()
        for p in self.planes:
            b += p.tokens.numel() * p.tokens.element_size()
            b += p.mrope.numel() * p.mrope.element_size()
        return b / 1024 ** 2


@torch.no_grad()
def build_video_pass(
    video_path: str,
    video_id: str,
    *,
    clip: OpenCLIPWrapper,
    qwen: Optional[QwenVLWrapper],
    cfg: QueryStreamConfig,
    max_seconds: Optional[int] = None,
    clip_batch: int = 1,
    vision_batch_planes: int = 8,
    cache_device: Optional[str] = None,
    frames: Optional[torch.Tensor] = None,
    meta: Optional[VideoMeta] = None,
) -> VideoPass:
    """Decode, encode and pre-compute everything about a video that Q cannot change.

    Args:
        max_seconds: stop after this many 1-FPS frames -- the latest question
                     timestamp on the video. Nothing later is ever visible.
        clip_batch: OpenCLIP minibatch. Defaults to 1 because that is what a
                     1-FPS stream actually pays, and because fp16 GEMMs are not
                     batch-invariant: at batch 32 patch features move by ~6e-5,
                     which flips roughly 0.3% of tokens sitting on the
                     frame-mean threshold. Raise it for throughput when exact
                     agreement with the streaming pipeline is not needed.
        vision_batch_planes: how many 2-frame planes to push through Qwen's
                     vision tower per call. `smart_resize` depends only on
                     (H, W), never on T, so batching cannot change the token
                     grid; the invariant is asserted anyway.
    """
    if not cfg.qdp.update_all_patches:
        raise RuntimeError(
            "the per-video cache assumes a query-independent DSH, which requires "
            "QDPConfig.update_all_patches=True. With update_all_patches=False the "
            "EMA is restricted to M_QDP survivors and therefore depends on the "
            "query; re-run with the streaming pipeline instead."
        )

    sw = Stopwatch(clip.device)
    cache_dev = torch.device(cache_device) if cache_device else clip.device

    # ---- decode -------------------------------------------------------- #
    if frames is None:
        max_frames = None if max_seconds is None else int(max_seconds) + 1
        with sw.stage("decode", n=1):
            frames, meta = read_video_frames(
                video_path, target_fps=cfg.fps, max_frames=max_frames
            )
        sw.counts["decode"] = frames.shape[0]
    assert meta is not None
    t = frames.shape[0]

    # ---- OpenCLIP patch features --------------------------------------- #
    with sw.stage("openclip", n=t):
        feats: List[torch.Tensor] = []
        for i in range(0, t, clip_batch):
            encoded = clip.encode_frames(frames[i : i + clip_batch], batch_size=clip_batch)
            feats.append(torch.stack([e.patch_feats for e in encoded], dim=0))
        patch_feats = torch.cat(feats, dim=0)                    # [T, N, D]
    frame_pooled = patch_feats.to(torch.float32).mean(dim=1)     # [T, D] = \bar v_t

    # ---- DSH temporal novelty (query-independent) ---------------------- #
    with sw.stage("dsh", n=t):
        dsh = DSHMemory(
            alpha=cfg.qdp.alpha, init=cfg.qdp.dsh_init, space=cfg.qdp.dsh_space
        )
        temp_sim = torch.empty((t, patch_feats.shape[1]),
                               dtype=torch.float32, device=patch_feats.device)
        for i in range(t):
            sim = dsh.step(patch_feats[i])
            temp_sim[i] = sim

    # ---- Qwen vision tower, batched over planes ------------------------ #
    planes: List[PlaneCache] = []
    llm_grid_hw: Optional[Tuple[int, int]] = None
    time_interval = 0
    if qwen is not None:
        k = qwen.temporal_patch_size
        n_planes = t // k
        with sw.stage("vision_tower", n=n_planes * k):
            for p0 in range(0, n_planes, vision_batch_planes):
                p1 = min(p0 + vision_batch_planes, n_planes)
                chunk = qwen.encode_chunk(frames[p0 * k : p1 * k], fps=cfg.fps)
                if llm_grid_hw is None:
                    llm_grid_hw, time_interval = chunk.llm_grid_hw, chunk.time_interval
                elif chunk.llm_grid_hw != llm_grid_hw:
                    raise RuntimeError(
                        f"{video_id}: backbone token grid changed mid-video "
                        f"{llm_grid_hw} -> {chunk.llm_grid_hw}; M-ROPE coordinates "
                        f"from different chunks would not be comparable"
                    )
                m = llm_grid_hw[0] * llm_grid_hw[1]
                assert chunk.num_tokens == m * (p1 - p0), (
                    f"{video_id}: vision tower returned {chunk.num_tokens} tokens "
                    f"for {p1 - p0} planes of {m}"
                )
                # build_dense_mrope numbers planes 0..(p1-p0-1) within the batch;
                # shift onto the stream timeline.
                mrope = chunk.mrope.clone()
                mrope[:, 0] += p0 * time_interval
                for j in range(p1 - p0):
                    sl = slice(j * m, (j + 1) * m)
                    pi = p0 + j
                    planes.append(
                        PlaneCache(
                            tokens=chunk.tokens[sl].to(cache_dev),
                            mrope=mrope[sl].to(cache_dev),
                            plane_idx=pi,
                            frame_ids=tuple(range(pi * k, (pi + 1) * k)),
                            timestamp=(pi * k) / cfg.fps,
                        )
                    )

    return VideoPass(
        video_id=video_id,
        video_path=video_path,
        meta=meta,
        n_frames=t,
        clip_grid_hw=clip.grid_hw,
        llm_grid_hw=llm_grid_hw,
        time_interval=time_interval,
        patch_feats=patch_feats.to(cache_dev),
        frame_pooled=frame_pooled.to(cache_dev),
        temp_sim=temp_sim.to(cache_dev),
        planes=planes,
        timings=sw,
    )


__all__ = ["VideoPass", "PlaneCache", "build_video_pass"]
