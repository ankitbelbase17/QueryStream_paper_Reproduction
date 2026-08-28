"""Central configuration for QueryStream.

All defaults are the values reported in the paper:
  - Section 4.1 (Implementation Details): alpha = 0.1, OpenCLIP-ViT-L/14, 1 FPS
  - Appendix A.6 (Hyperparameter Selection): tau_temp = 0.90, tau_rel = 0.60, tau_den = 0.15

Anything NOT specified by the paper is marked `[UNSPECIFIED]` and given a
documented, faithful-by-default choice. Do not silently change those without
recording it in the run log -- they are the reproduction's degrees of freedom.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional, Tuple

import torch

# --------------------------------------------------------------------------- #
# Enumerated policy choices
# --------------------------------------------------------------------------- #

# How to seed the DSH buffer on the very first frame of a stream.
#   "self"  : v_dsh_0 = v_0  -> sim(v_0, v_dsh_0) == 1.0, so NOTHING is novel at t=0.
#             Combined with `first_frame_all_novel=True` this is the paper-faithful
#             reading: history does not exist yet, so t=0 cannot be "redundant".
#   "zeros" : v_dsh_0 = 0    -> cosine is undefined (0/0); avoid.
DSHInit = Literal["self", "zeros"]

# Space in which the DSH exponential moving average is accumulated.
#   "normalized" : EMA over L2-normalised patch features (re-normalised at compare time).
#                  Keeps every patch on equal footing regardless of feature norm.
#   "raw"        : EMA over unnormalised features, normalise only at compare time.
# The paper writes the update over `v_t^i` with no statement about normalisation.
# [UNSPECIFIED] -- we default to "normalized" because cosine is the only metric used
# downstream, so norm information is never consumed and only destabilises the EMA.
DSHSpace = Literal["normalized", "raw"]

# Calibration applied to the RTAR relevance signal sim(q, mean_pool(v_t)) before
# it is compared against tau_rel. See the WARNING in rtar.py -- raw OpenCLIP
# image-text cosine similarity lives around 0.15-0.35, so a literal tau_rel=0.60
# gate fires essentially never. [UNSPECIFIED] in the paper.
RelevanceMode = Literal["raw", "logit_softmax", "running_minmax", "running_zscore"]

# How to collapse a fine CLIP patch grid onto the coarser backbone token grid.
MaskResampleMode = Literal["any", "all", "majority", "nearest"]

# WHERE the CLIP -> backbone grid conversion happens.
#   "mask"  : QDP thresholds on the 16x16 CLIP grid, then the BINARY mask is
#             pooled onto the backbone grid. Faithful to a literal reading, but
#             with "any" pooling a backbone token survives if any of the 8
#             contributing (4 spatial x 2 temporal) CLIP patches did, which
#             inflates the keep rate enormously -- measured 41% -> 96% on a real
#             clip. It is very unlikely the paper's 57.2% was produced this way.
#   "score" : the continuous similarity maps are pooled onto the backbone grid
#             FIRST, and the thresholds are applied there. The frame-adaptive
#             mean threshold then means the same thing on the grid that is
#             actually being pruned, and the keep rate lands in the paper's
#             regime. Recommended.
MaskSpace = Literal["mask", "score"]


@dataclass
class EncoderConfig:
    """Lightweight vision-language encoder E (paper: OpenCLIP-ViT-L/14)."""

    model_name: str = "ViT-L-14"
    pretrained: str = "openai"
    image_size: int = 224
    patch_size: int = 14
    # 224 / 14 = 16 -> a 16x16 = 256 patch grid.
    device: str = "cuda"
    # fp16/bf16 are fine for the encoder forward; all QueryStream statistics are
    # promoted to fp32 (see `stats_dtype`) because the DSH EMA is a long
    # recursive accumulation and is genuinely precision-sensitive.
    dtype: torch.dtype = torch.float16
    normalize_features: bool = True

    @property
    def grid_hw(self) -> Tuple[int, int]:
        assert self.image_size % self.patch_size == 0, (
            f"image_size {self.image_size} not divisible by patch_size {self.patch_size}"
        )
        s = self.image_size // self.patch_size
        return (s, s)

    @property
    def num_patches(self) -> int:
        h, w = self.grid_hw
        return h * w


@dataclass
class QDPConfig:
    """Query-Aware Differential Pruning (paper Sec. 3.2)."""

    # v_dsh_t = alpha * v_t + (1 - alpha) * v_dsh_{t-1}   -- paper Sec. 4.1
    alpha: float = 0.1
    # M_temp = 1[ sim(v_t, v_dsh_{t-1}) < tau_temp ]      -- paper Table 7
    tau_temp: float = 0.90

    dsh_init: DSHInit = "self"
    dsh_space: DSHSpace = "normalized"

    # [UNSPECIFIED] The paper's t=0 semantics. With dsh_init="self" the temporal
    # gate would reject every patch of frame 0 (self-similarity == 1.0 > tau_temp),
    # which would throw away the entire opening frame. We instead declare frame 0
    # fully novel: there is no history to be redundant against.
    first_frame_all_novel: bool = True

    # [UNSPECIFIED] The paper updates the DSH "following this check", with no
    # statement about whether pruned patches also update. True = every patch
    # updates every frame (the literal reading of the update equation).
    update_all_patches: bool = True

    # Safety valve, OFF by default for faithfulness. If >0, guarantees at least
    # this many tokens survive per frame (top-k by semantic similarity among the
    # temporally-novel set, then by semantic similarity overall). Prevents a
    # zero-token flush from crashing the backbone at Milestone 2.
    min_keep_per_frame: int = 0

    def validate(self) -> None:
        assert 0.0 < self.alpha <= 1.0, f"alpha must be in (0, 1], got {self.alpha}"
        assert -1.0 <= self.tau_temp <= 1.0, f"tau_temp must be a cosine, got {self.tau_temp}"
        assert self.min_keep_per_frame >= 0


@dataclass
class RTARConfig:
    """Relevance-Triggered Active Response (paper Sec. 3.3)."""

    # R_t = 1[ sim(q, mean_pool(v_t)) > tau_rel ]   -- paper Table 8
    tau_rel: float = 0.60
    # D_t = 1[ keep_rate(t) > tau_den ]             -- paper Table 8
    tau_den: float = 0.15

    relevance_mode: RelevanceMode = "raw"
    # Window for the running_minmax / running_zscore calibrations (frames).
    calibration_window: int = 64
    # Frames observed before a calibrated mode is trusted; until then, no trigger.
    calibration_warmup: int = 8
    # OpenCLIP's learned temperature, used by relevance_mode="logit_softmax".
    logit_scale: float = 100.0

    # Appendix A.4: consecutive trigger timesteps are grouped into event
    # intervals and only the FIRST timestamp of each interval is used as the
    # response point. `event_gap_s` is the silence required to close an interval.
    # [UNSPECIFIED] -- the paper gives no value. 1.0s == one frame at 1 FPS, i.e.
    # strictly-consecutive triggers collapse into one event.
    debounce_enabled: bool = True
    event_gap_s: float = 1.0

    def validate(self) -> None:
        assert 0.0 <= self.tau_den <= 1.0, f"tau_den is a rate, got {self.tau_den}"
        assert self.calibration_window > 1
        assert self.event_gap_s >= 0.0


@dataclass
class BufferConfig:
    """Just-in-time token buffer (paper Sec. 3.1)."""

    # Unbounded streams need a cap. 0 = unbounded (paper's implicit assumption
    # for benchmark-length clips). Eviction is oldest-first.
    max_frames: int = 0
    max_tokens: int = 0
    # Keep the *unpruned* tokens resident (paper: "original, unpruned visual
    # tokens are temporarily held in a memory buffer"). If False, the mask is
    # applied at append time (cheaper, but no longer JIT and not paper-faithful).
    store_unpruned: bool = True
    # Where the buffer lives. "cpu" keeps GPU memory free on long streams at the
    # cost of a H2D copy per flush.
    device: str = "cuda"


@dataclass
class BackboneConfig:
    """Backbone Video-LLM token geometry (Qwen2.5-VL-7B-Instruct).

    Qwen2.5-VL splits a frame into 14x14 pixel patches, then applies a 2x2
    *spatial* merge and a 2-frame *temporal* merge in the vision tower. So the
    visual-token grid the LLM actually sees is COARSER than the CLIP patch grid,
    and M_QDP must be resampled onto it. The paper does not discuss this.
    """

    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    # M-ROPE section layout used by Qwen2.5-VL: (temporal, height, width).
    mrope_sections: Tuple[str, str, str] = ("t", "h", "w")

    def token_grid_hw(self, image_size: int) -> Tuple[int, int]:
        """Merged visual-token grid for a square frame of `image_size` px."""
        eff = self.patch_size * self.spatial_merge_size  # 28 px per output token
        assert image_size % eff == 0, (
            f"image_size {image_size} must be divisible by "
            f"patch_size*spatial_merge_size = {eff}"
        )
        s = image_size // eff
        return (s, s)


@dataclass
class QueryStreamConfig:
    """Top-level config. `QueryStreamConfig()` reproduces the paper's setting."""

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    qdp: QDPConfig = field(default_factory=QDPConfig)
    rtar: RTARConfig = field(default_factory=RTARConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)

    # Paper Sec. 4.1: "our model processes video streams at 1 FPS".
    fps: float = 1.0
    # dtype for every similarity / mask / EMA computation.
    stats_dtype: torch.dtype = torch.float32
    mask_resample_mode: MaskResampleMode = "any"
    mask_space: MaskSpace = "score"
    score_pool_mode: str = "mean"
    seed: int = 0

    def validate(self) -> "QueryStreamConfig":
        self.qdp.validate()
        self.rtar.validate()
        assert self.fps > 0
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        # dtypes are not JSON-serialisable
        d["stats_dtype"] = str(self.stats_dtype)
        d["encoder"]["dtype"] = str(self.encoder.dtype)
        return d


__all__ = [
    "QueryStreamConfig",
    "EncoderConfig",
    "QDPConfig",
    "RTARConfig",
    "BufferConfig",
    "BackboneConfig",
    "DSHInit",
    "DSHSpace",
    "RelevanceMode",
    "MaskResampleMode",
    "MaskSpace",
]
