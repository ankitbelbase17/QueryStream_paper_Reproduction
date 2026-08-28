"""Dynamically Smoothed History (DSH) -- paper Sec. 3.2, Temporal Novelty Filtering.

Maintains one historical feature vector per PATCH LOCATION i:

    v_dsh_{t}^i = alpha * v_t^i + (1 - alpha) * v_dsh_{t-1}^i          (paper eq.)

and exposes the novelty comparison that precedes it:

    sim_temp(t, i) = cos( v_t^i , v_dsh_{t-1}^i )

Order matters and is enforced by `step()`: compare against t-1 FIRST, then
update. Calling `update()` before `similarity()` silently leaks the current
frame into its own history and inflates similarity, which would over-prune.

Why alpha matters (paper Fig. 3, OVO-Bench):
    alpha=1.00 -> naive frame-to-frame (TimeChat-Online's regime), score ~45.8
    alpha=0.50 -> ~48.1
    alpha=0.10 -> 49.4   <-- chosen
    alpha=0.01 -> ~48.4  (memory too long, adapts too slowly)

Numerics: the EMA is a recursion of length T (hundreds of frames at 1 FPS).
State is held in fp32 unconditionally; with alpha=0.1 the effective memory is
~10 frames, so fp16 state would accumulate visible drift near the tau_temp=0.90
decision boundary.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import DSHInit, DSHSpace
from .ops import cosine_sim_rowwise, l2_normalize


class DSHMemory(nn.Module):
    """Per-patch exponentially-smoothed history for one video stream.

    Single-stream by design: streaming evaluation processes one video at a time,
    and a batched variant would need per-stream reset bookkeeping that the
    eval loop does not want. Instantiate one per stream, or call `reset()`
    between videos.

    Shapes:
        state: [N, D]  -- N patch locations, D feature dim. Lazily allocated on
                          the first `step()` / `update()`.
    """

    state: Optional[torch.Tensor]

    def __init__(
        self,
        alpha: float = 0.1,
        *,
        init: DSHInit = "self",
        space: DSHSpace = "normalized",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        assert 0.0 < alpha <= 1.0, f"alpha must be in (0, 1], got {alpha}"
        self.alpha = float(alpha)
        self.init = init
        self.space = space
        self.dtype = dtype
        self.state = None
        self._t = 0  # number of frames absorbed so far

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    @property
    def is_initialized(self) -> bool:
        return self.state is not None

    @property
    def num_frames_seen(self) -> int:
        return self._t

    def reset(self) -> None:
        """Clear all history. MUST be called between videos."""
        self.state = None
        self._t = 0

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _prepare(self, patch_feats: torch.Tensor) -> torch.Tensor:
        """Cast to state dtype and, if configured, project onto the unit sphere."""
        x = patch_feats.to(self.dtype)
        if self.space == "normalized":
            x = l2_normalize(x, dim=-1)
        return x

    def _initialize(self, x: torch.Tensor) -> None:
        if self.init == "self":
            self.state = x.clone()
        elif self.init == "zeros":
            self.state = torch.zeros_like(x)
        else:  # pragma: no cover - guarded by Literal
            raise ValueError(f"unknown dsh init {self.init!r}")

    @staticmethod
    def _check(patch_feats: torch.Tensor) -> None:
        assert patch_feats.dim() == 2, (
            f"DSHMemory expects patch features [N, D], got {tuple(patch_feats.shape)}"
        )
        assert patch_feats.numel() > 0, "empty patch feature tensor"

    def _check_compatible(self, x: torch.Tensor) -> None:
        assert self.state is not None
        if self.state.shape != x.shape:
            raise ValueError(
                f"DSH state shape {tuple(self.state.shape)} does not match incoming "
                f"frame {tuple(x.shape)}. Patch-grid geometry must be constant "
                f"across a stream -- resize frames to a fixed resolution, or call "
                f"reset() at the geometry change."
            )
        if self.state.device != x.device:
            self.state = self.state.to(x.device)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def similarity(self, patch_feats: torch.Tensor) -> torch.Tensor:
        """cos(v_t^i, v_dsh_{t-1}^i) per patch, WITHOUT mutating state.

        On the very first frame there is no v_dsh_{-1}; we return 1.0 for every
        patch (perfect similarity => nothing is novel). `QDPModule` overrides
        this via `first_frame_all_novel`; the raw signal stays honest here.

        Args:
            patch_feats: [N, D]
        Returns:
            [N] float32 cosine similarities in [-1, 1].
        """
        self._check(patch_feats)
        x = self._prepare(patch_feats)
        if self.state is None:
            return torch.ones(x.shape[0], device=x.device, dtype=torch.float32)
        self._check_compatible(x)
        return cosine_sim_rowwise(x, self.state)

    @torch.no_grad()
    def update(
        self,
        patch_feats: torch.Tensor,
        update_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Absorb frame t: v_dsh_t = alpha * v_t + (1 - alpha) * v_dsh_{t-1}.

        Args:
            patch_feats: [N, D]
            update_mask: optional [N] bool. If given, only these patch locations
                are updated; the rest hold their previous history. Used when
                `QDPConfig.update_all_patches=False`.
        """
        self._check(patch_feats)
        x = self._prepare(patch_feats)

        if self.state is None:
            self._initialize(x)
            self._t = 1
            return

        self._check_compatible(x)
        new_state = self.alpha * x + (1.0 - self.alpha) * self.state

        if update_mask is not None:
            assert update_mask.dim() == 1 and update_mask.numel() == x.shape[0], (
                f"update_mask must be [N={x.shape[0]}], got {tuple(update_mask.shape)}"
            )
            m = update_mask.to(x.device).reshape(-1, 1)
            new_state = torch.where(m, new_state, self.state)

        self.state = new_state
        self._t += 1

    @torch.no_grad()
    def step(
        self,
        patch_feats: torch.Tensor,
        update_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compare-then-update, in the paper's order.

        Returns:
            [N] float32 -- cos(v_t^i, v_dsh_{t-1}^i), i.e. the similarity used by
            M_temp, computed BEFORE frame t enters the history.
        """
        sim = self.similarity(patch_feats)
        self.update(patch_feats, update_mask=update_mask)
        return sim

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def effective_memory_frames(self) -> float:
        """1/alpha -- the EMA's characteristic timescale, in frames.

        At alpha=0.1 and 1 FPS this is ~10 frames == ~10 seconds of history,
        which is the "smoothed, medium-term history" the paper argues for.
        """
        return 1.0 / self.alpha

    def state_dict_summary(self) -> dict:
        return {
            "alpha": self.alpha,
            "init": self.init,
            "space": self.space,
            "frames_seen": self._t,
            "initialized": self.is_initialized,
            "state_shape": tuple(self.state.shape) if self.state is not None else None,
            "effective_memory_frames": self.effective_memory_frames,
        }

    def extra_repr(self) -> str:
        return (
            f"alpha={self.alpha}, init={self.init!r}, space={self.space!r}, "
            f"frames_seen={self._t}"
        )


__all__ = ["DSHMemory"]
