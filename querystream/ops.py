"""Primitive tensor ops shared by QDP / RTAR / the buffer.

Everything here is pure, stateless and shape-checked. Similarity math is always
promoted to fp32: QDP thresholds sit at tau_temp=0.90 where fp16's ~1e-3
resolution near 1.0 would make the gate decision noise-dominated.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from .config import MaskResampleMode

EPS = 1e-8


# --------------------------------------------------------------------------- #
# Similarity
# --------------------------------------------------------------------------- #

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = EPS) -> torch.Tensor:
    """L2-normalise along `dim`. Zero vectors are left as zero (not NaN)."""
    norm = x.norm(p=2, dim=dim, keepdim=True)
    return x / norm.clamp_min(eps)


def cosine_sim_to_vector(feats: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """cos(feats[i], vec) for every row i.

    Args:
        feats: [N, D]
        vec:   [D] or [1, D]
    Returns:
        [N] float32
    """
    assert feats.dim() == 2, f"feats must be [N, D], got {tuple(feats.shape)}"
    vec = vec.reshape(-1)
    assert vec.numel() == feats.shape[1], (
        f"dim mismatch: feats D={feats.shape[1]} vs vec D={vec.numel()}"
    )
    f32 = torch.float32
    a = l2_normalize(feats.to(f32), dim=-1)
    b = l2_normalize(vec.to(f32), dim=-1)
    return a @ b


def cosine_sim_rowwise(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-aligned cosine: out[i] = cos(a[i], b[i]).

    Args:
        a, b: [N, D]
    Returns:
        [N] float32
    """
    assert a.shape == b.shape, f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"
    assert a.dim() == 2, f"expected [N, D], got {tuple(a.shape)}"
    f32 = torch.float32
    return (l2_normalize(a.to(f32), -1) * l2_normalize(b.to(f32), -1)).sum(-1)


def mean_pool_frame(patch_feats: torch.Tensor) -> torch.Tensor:
    """Holistic frame vector \\bar v_t = mean over patches (paper Sec. 3.3).

    Args:
        patch_feats: [N, D]
    Returns:
        [D] float32, NOT normalised (callers use cosine, which normalises).
    """
    assert patch_feats.dim() == 2, f"expected [N, D], got {tuple(patch_feats.shape)}"
    return patch_feats.to(torch.float32).mean(dim=0)


# --------------------------------------------------------------------------- #
# M-ROPE coordinate construction
# --------------------------------------------------------------------------- #

def build_mrope_coords(
    t_index: int,
    grid_h: int,
    grid_w: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """3D M-ROPE coordinates (t, h, w) for one frame's token grid.

    Row-major over (h, w) -- the raster order Qwen2.5-VL flattens its visual
    tokens in. Returns int64 so coordinates survive slicing without drift.

    Returns:
        [grid_h * grid_w, 3] int64, column order (t, h, w).
    """
    assert grid_h > 0 and grid_w > 0
    hs = torch.arange(grid_h, device=device, dtype=torch.long)
    ws = torch.arange(grid_w, device=device, dtype=torch.long)
    hh, ww = torch.meshgrid(hs, ws, indexing="ij")
    tt = torch.full((grid_h, grid_w), int(t_index), device=device, dtype=torch.long)
    return torch.stack([tt.reshape(-1), hh.reshape(-1), ww.reshape(-1)], dim=1)


# --------------------------------------------------------------------------- #
# Grid resampling: CLIP patch grid -> backbone visual-token grid
# --------------------------------------------------------------------------- #

def resample_patch_mask(
    mask: torch.Tensor,
    src_hw: Tuple[int, int],
    dst_hw: Tuple[int, int],
    mode: MaskResampleMode = "any",
) -> torch.Tensor:
    """Project a boolean patch mask from one 2D grid onto another.

    Needed because QDP scores patches on the OpenCLIP-ViT-L/14 grid (16x16 for a
    224px frame) while the mask must gate Qwen2.5-VL visual tokens, whose grid is
    coarsened by a 2x2 spatial merge (8x8 for the same frame). The paper is
    silent on this step; `mode` is the reproduction knob.

    Modes (for the common integer-downsample case, exact block pooling):
        "any"      : keep a target token if ANY source patch under it is kept
                     (recall-favouring; the safe default -- never drops evidence
                     that QDP wanted to preserve).
        "all"      : keep only if EVERY source patch is kept (precision-favouring,
                     yields much lower keep rates).
        "majority" : keep if >50% of source patches are kept.
        "nearest"  : nearest-neighbour sample (ignores block structure; use only
                     for non-integer ratios).

    Args:
        mask:    [*, Hs*Ws] or [Hs, Ws] bool
        src_hw:  (Hs, Ws)
        dst_hw:  (Hd, Wd)
    Returns:
        bool tensor, flattened row-major to [*, Hd*Wd] (or [Hd*Wd] for 1D input).
    """
    hs, ws = src_hw
    hd, wd = dst_hw
    squeeze_batch = False

    if mask.dim() == 1:
        assert mask.numel() == hs * ws, (
            f"mask has {mask.numel()} elems, src grid is {hs}x{ws}={hs * ws}"
        )
        m = mask.reshape(1, 1, hs, ws)
        squeeze_batch = True
    elif mask.dim() == 2 and mask.shape == (hs, ws):
        m = mask.reshape(1, 1, hs, ws)
        squeeze_batch = True
    elif mask.dim() == 2:
        assert mask.shape[1] == hs * ws, (
            f"mask [{mask.shape[0]}, {mask.shape[1]}] does not match src grid {hs}x{ws}"
        )
        m = mask.reshape(mask.shape[0], 1, hs, ws)
    else:
        raise ValueError(f"unsupported mask shape {tuple(mask.shape)}")

    if (hs, ws) == (hd, wd):
        out = m
    elif mode == "nearest":
        out = F.interpolate(m.float(), size=(hd, wd), mode="nearest") > 0.5
    elif hs % hd == 0 and ws % wd == 0:
        # Exact block pooling -- the case that actually occurs (16x16 -> 8x8).
        kh, kw = hs // hd, ws // wd
        frac = F.avg_pool2d(m.float(), kernel_size=(kh, kw), stride=(kh, kw))
        if mode == "any":
            out = frac > 0.0
        elif mode == "all":
            out = frac >= 1.0 - 1e-6
        elif mode == "majority":
            out = frac > 0.5
        else:
            raise ValueError(f"unknown mode {mode!r}")
    else:
        # Non-integer ratio: fall back to area interpolation + the same threshold.
        frac = F.interpolate(m.float(), size=(hd, wd), mode="area")
        thresh = {"any": 0.0, "majority": 0.5, "all": 1.0 - 1e-6}[mode]
        out = frac > thresh if mode != "all" else frac >= thresh

    out = out.to(torch.bool).reshape(m.shape[0], hd * wd)
    return out[0] if squeeze_batch else out


def expand_mask_to_temporal_merge(
    mask: torch.Tensor,
    temporal_patch_size: int,
    mode: MaskResampleMode = "any",
) -> torch.Tensor:
    """Fold a per-frame mask stack onto Qwen's temporally-merged token grid.

    Qwen2.5-VL merges `temporal_patch_size` (=2) consecutive frames into one
    visual-token plane. Two per-frame masks therefore have to be combined into
    one mask over the merged plane.

    Args:
        mask: [T, P] bool, T frames of P tokens each. T must be divisible by
              `temporal_patch_size` (pad upstream if not).
    Returns:
        [T // temporal_patch_size, P] bool
    """
    assert mask.dim() == 2, f"expected [T, P], got {tuple(mask.shape)}"
    t, p = mask.shape
    k = temporal_patch_size
    assert t % k == 0, f"T={t} not divisible by temporal_patch_size={k}"
    grouped = mask.reshape(t // k, k, p)
    if mode == "any":
        return grouped.any(dim=1)
    if mode == "all":
        return grouped.all(dim=1)
    if mode == "majority":
        return grouped.float().mean(dim=1) > 0.5
    if mode == "nearest":
        return grouped[:, 0, :]
    raise ValueError(f"unknown mode {mode!r}")


def resample_scores(
    scores: torch.Tensor,
    src_hw: Tuple[int, int],
    dst_hw: Tuple[int, int],
    mode: str = "mean",
) -> torch.Tensor:
    """Pool a CONTINUOUS per-patch score map onto a coarser grid.

    The alternative to `resample_patch_mask`. Pooling a binary mask with "any"
    inflates the keep rate badly (4 CLIP patches -> 1 backbone token, so a token
    survives if ANY of them did); pooling the SCORES and thresholding on the
    target grid preserves the threshold's meaning instead.

    Args:
        scores: [Hs*Ws] or [Hs, Ws] float
        mode:   "mean" (default), "max", or "min"
    Returns:
        [Hd*Wd] float32
    """
    hs, ws = src_hw
    hd, wd = dst_hw
    x = scores.reshape(1, 1, hs, ws).to(torch.float32)
    if (hs, ws) == (hd, wd):
        return x.reshape(-1)
    if hs % hd == 0 and ws % wd == 0:
        kh, kw = hs // hd, ws // wd
        if mode == "mean":
            out = F.avg_pool2d(x, (kh, kw), (kh, kw))
        elif mode == "max":
            out = F.max_pool2d(x, (kh, kw), (kh, kw))
        elif mode == "min":
            out = -F.max_pool2d(-x, (kh, kw), (kh, kw))
        else:
            raise ValueError(f"unknown mode {mode!r}")
    else:
        out = F.interpolate(x, size=(hd, wd), mode="area")
    return out.reshape(-1)


def resample_scores_batch(
    scores: torch.Tensor,
    src_hw: Tuple[int, int],
    dst_hw: Tuple[int, int],
    mode: str = "mean",
) -> torch.Tensor:
    """Batched `resample_scores`. [B, Hs*Ws] -> [B, Hd*Wd] float32.

    Identical arithmetic to the single-map version -- the eval harness pools
    hundreds of frames at once and the per-frame Python loop dominated the
    profile otherwise. Kept as a separate symbol so the [N]-shaped contract of
    `resample_scores` (which the streaming pipeline relies on) is untouched.
    """
    assert scores.dim() == 2, f"expected [B, Hs*Ws], got {tuple(scores.shape)}"
    hs, ws = src_hw
    hd, wd = dst_hw
    b = scores.shape[0]
    x = scores.reshape(b, 1, hs, ws).to(torch.float32)
    if (hs, ws) == (hd, wd):
        return x.reshape(b, -1)
    if hs % hd == 0 and ws % wd == 0:
        kh, kw = hs // hd, ws // wd
        if mode == "mean":
            out = F.avg_pool2d(x, (kh, kw), (kh, kw))
        elif mode == "max":
            out = F.max_pool2d(x, (kh, kw), (kh, kw))
        elif mode == "min":
            out = -F.max_pool2d(-x, (kh, kw), (kh, kw))
        else:
            raise ValueError(f"unknown mode {mode!r}")
    else:
        out = F.interpolate(x, size=(hd, wd), mode="area")
    return out.reshape(b, -1)


def keep_rate(mask: torch.Tensor) -> float:
    """Fraction of True entries. The paper's Token Keep Rate, per frame."""
    if mask.numel() == 0:
        return 0.0
    return mask.to(torch.float32).mean().item()


__all__ = [
    "EPS",
    "l2_normalize",
    "cosine_sim_to_vector",
    "cosine_sim_rowwise",
    "mean_pool_frame",
    "build_mrope_coords",
    "resample_patch_mask",
    "resample_scores",
    "resample_scores_batch",
    "expand_mask_to_temporal_merge",
    "keep_rate",
]
