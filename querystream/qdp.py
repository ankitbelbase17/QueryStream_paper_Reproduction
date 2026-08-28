"""Query-Aware Differential Pruning (QDP) -- paper Sec. 3.2.

Exact formulation reproduced here:

    Semantic relevance (frame-adaptive threshold = the frame's own mean):
        M_sem(t, i)  = 1[ sim(q, v_t^i) > (1/N) * sum_j sim(q, v_t^j) ]

    Temporal novelty (against the DSH, NOT the previous frame):
        M_temp(t, i) = 1[ sim(v_t^i, v_dsh_{t-1}^i) < tau_temp ]

    Synergistic policy (logical conjunction -- an INTERSECTION, so the keep rate
    is strictly <= min(keep_sem, keep_temp)):
        M_QDP(t, i)  = M_sem(t, i) AND M_temp(t, i)

Both comparisons are STRICT (`>` and `<`) exactly as printed in the paper.
Ties therefore fall on the reject side, which matters at the tau_temp=0.90
boundary where fp16 quantisation would otherwise flip decisions -- hence fp32
statistics throughout.

Sanity anchor from the paper's own ablation (Table 4, StreamingBench):
    temporal only : keep 63.4%, score 74.76
    semantic only : keep 61.7%, score 74.52
    full QDP      : keep 57.2%, score 75.32
Note 57.2 / 61.7 = 0.927 -- the two masks are strongly positively correlated.
If your reproduction yields an intersection keep rate near the independence
product (0.617 * 0.634 = 0.391), the two signals are decorrelated and something
upstream (most likely the patch-feature projection) is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .config import QDPConfig
from .dsh_memory import DSHMemory
from .ops import cosine_sim_to_vector, keep_rate


@dataclass
class QDPOutput:
    """Per-frame QDP result. All mask tensors are [N] bool on the input device."""

    m_sem: torch.Tensor          # [N] semantic relevance mask
    m_temp: torch.Tensor         # [N] temporal novelty mask
    m_qdp: torch.Tensor          # [N] final mask = m_sem AND m_temp
    sem_sim: torch.Tensor        # [N] cos(q, v_t^i)
    temp_sim: torch.Tensor       # [N] cos(v_t^i, v_dsh_{t-1}^i)
    sem_threshold: float         # the frame-adaptive mean threshold
    keep_rate: float             # |M_QDP| / N   <- the paper's Token Keep Rate
    keep_rate_sem: float
    keep_rate_temp: float
    is_first_frame: bool
    forced_keep: int = 0         # tokens revived by min_keep_per_frame, if any

    @property
    def num_kept(self) -> int:
        return int(self.m_qdp.sum().item())

    @property
    def num_patches(self) -> int:
        return int(self.m_qdp.numel())

    def summary(self) -> dict:
        return {
            "keep_rate": round(self.keep_rate, 6),
            "keep_rate_sem": round(self.keep_rate_sem, 6),
            "keep_rate_temp": round(self.keep_rate_temp, 6),
            "sem_threshold": round(self.sem_threshold, 6),
            "num_kept": self.num_kept,
            "num_patches": self.num_patches,
            "first_frame": self.is_first_frame,
            "forced_keep": self.forced_keep,
        }


class QDPModule(nn.Module):
    """Stateful per-stream pruning policy. One instance per video (or `reset()`).

    The module owns a `DSHMemory` so that the compare-then-update ordering can
    never be got wrong by a caller. Inject your own for testing via `dsh=`.
    """

    def __init__(self, cfg: Optional[QDPConfig] = None, *, dsh: Optional[DSHMemory] = None):
        super().__init__()
        self.cfg = cfg or QDPConfig()
        self.cfg.validate()
        self.dsh = dsh if dsh is not None else DSHMemory(
            alpha=self.cfg.alpha,
            init=self.cfg.dsh_init,
            space=self.cfg.dsh_space,
        )
        self._frame_idx = 0

    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Clear stream state. MUST be called between videos."""
        self.dsh.reset()
        self._frame_idx = 0

    @property
    def frame_idx(self) -> int:
        return self._frame_idx

    # ------------------------------------------------------------------ #
    # Individual criteria (exposed separately so the Table-4 ablation is a
    # one-line change rather than a fork of the forward pass)
    # ------------------------------------------------------------------ #

    @staticmethod
    @torch.no_grad()
    def semantic_mask(
        patch_feats: torch.Tensor,
        query_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """M_sem with the frame-adaptive mean threshold.

        Args:
            patch_feats: [N, D] patch features from E, in the shared VL space.
            query_emb:   [D]    query text embedding from E.
        Returns:
            (m_sem [N] bool, sem_sim [N] f32, threshold float)
        """
        sem_sim = cosine_sim_to_vector(patch_feats, query_emb)   # [N]
        threshold = sem_sim.mean()                               # (1/N) sum_j sim(q, v_t^j)
        m_sem = sem_sim > threshold                              # STRICT >
        return m_sem, sem_sim, float(threshold.item())

    @torch.no_grad()
    def temporal_mask(
        self,
        patch_feats: torch.Tensor,
        *,
        advance_history: bool = True,
        update_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """M_temp against the DSH, then advance the history.

        Returns:
            (m_temp [N] bool, temp_sim [N] f32, is_first_frame bool)
        """
        is_first = not self.dsh.is_initialized
        temp_sim = self.dsh.similarity(patch_feats)
        m_temp = temp_sim < self.cfg.tau_temp                    # STRICT <

        if is_first and self.cfg.first_frame_all_novel:
            # No history exists at t=0, so nothing can be redundant. Without this
            # override, dsh_init="self" makes temp_sim == 1.0 everywhere and the
            # entire opening frame is discarded.
            m_temp = torch.ones_like(m_temp)

        if advance_history:
            self.dsh.update(patch_feats, update_mask=update_mask)
        return m_temp, temp_sim, is_first

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def forward(
        self,
        patch_feats: torch.Tensor,
        query_emb: torch.Tensor,
        *,
        advance_history: bool = True,
    ) -> QDPOutput:
        """Run one streaming step.

        Args:
            patch_feats: [N, D] patch-level features of frame t from encoder E.
                         Must be in the SAME embedding space as `query_emb`
                         (i.e. post `ln_post` and post `visual.proj` for
                         OpenCLIP), otherwise `sim(q, v)` is meaningless.
            query_emb:   [D] text embedding of the user query Q.
        Returns:
            QDPOutput
        """
        assert patch_feats.dim() == 2, (
            f"patch_feats must be [N, D], got {tuple(patch_feats.shape)}"
        )
        n, d = patch_feats.shape
        assert query_emb.reshape(-1).numel() == d, (
            f"query_emb dim {query_emb.reshape(-1).numel()} != patch feature dim {d}. "
            "Patch features and the query must share the encoder's projected space."
        )

        # --- criterion 1: semantic relevance to Q ------------------------- #
        m_sem, sem_sim, sem_threshold = self.semantic_mask(patch_feats, query_emb)

        # --- criterion 2: temporal novelty vs the DSH --------------------- #
        # `update_all_patches=False` restricts the EMA to surviving patches; the
        # decision needs M_QDP, which needs M_temp, so we defer the update.
        deferred = not self.cfg.update_all_patches
        m_temp, temp_sim, is_first = self.temporal_mask(
            patch_feats,
            advance_history=advance_history and not deferred,
        )

        # --- synergistic policy: intersection ----------------------------- #
        m_qdp = m_sem & m_temp

        forced = 0
        if self.cfg.min_keep_per_frame > 0:
            m_qdp, forced = self._enforce_min_keep(m_qdp, sem_sim)

        if deferred and advance_history:
            self.dsh.update(patch_feats, update_mask=m_qdp)

        self._frame_idx += 1

        return QDPOutput(
            m_sem=m_sem,
            m_temp=m_temp,
            m_qdp=m_qdp,
            sem_sim=sem_sim,
            temp_sim=temp_sim,
            sem_threshold=sem_threshold,
            keep_rate=keep_rate(m_qdp),
            keep_rate_sem=keep_rate(m_sem),
            keep_rate_temp=keep_rate(m_temp),
            is_first_frame=is_first,
            forced_keep=forced,
        )

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def masks_from_pooled_scores(
        self,
        sem_sim_pooled: torch.Tensor,
        temp_sim_pooled: torch.Tensor,
        *,
        is_first_frame: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply the two QDP criteria on an already-pooled (backbone) grid.

        Used when `mask_space="score"`: the similarity maps are pooled from the
        CLIP grid onto the backbone token grid, and the thresholds are applied
        there. The semantic threshold is recomputed as the mean OVER THE POOLED
        GRID, so "above this frame's average relevance" keeps its meaning on the
        grid actually being pruned.

        Args:
            sem_sim_pooled:  [M] pooled cos(q, v)
            temp_sim_pooled: [M] pooled cos(v_t, v_dsh)
        Returns:
            (m_sem, m_temp, m_qdp, sem_threshold)
        """
        thr = sem_sim_pooled.mean()
        m_sem = sem_sim_pooled > thr                        # STRICT >
        m_temp = temp_sim_pooled < self.cfg.tau_temp        # STRICT <
        if is_first_frame and self.cfg.first_frame_all_novel:
            m_temp = torch.ones_like(m_temp)
        return m_sem, m_temp, m_sem & m_temp, float(thr.item())

    @torch.no_grad()
    def _enforce_min_keep(
        self,
        m_qdp: torch.Tensor,
        sem_sim: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Revive top-scoring patches so a frame never contributes zero tokens.

        Off by default (`min_keep_per_frame=0`) because it is NOT in the paper.
        Enable it only if empty flushes crash the backbone, and log `forced_keep`
        so the deviation stays visible in the results.
        """
        k = min(int(self.cfg.min_keep_per_frame), m_qdp.numel())
        current = int(m_qdp.sum().item())
        if current >= k:
            return m_qdp, 0
        need = k - current
        # Rank the currently-rejected patches by semantic similarity to Q and
        # revive the strongest ones until the floor is met.
        scores = sem_sim.clone().to(torch.float32)
        scores[m_qdp] = float("-inf")
        revive = torch.topk(scores, k=need).indices
        out = m_qdp.clone()
        out[revive] = True
        return out, need


__all__ = ["QDPModule", "QDPOutput"]
