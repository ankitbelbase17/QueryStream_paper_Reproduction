"""Lightweight vision-language encoder E -- paper Sec. 3.1 / 4.1.

The paper uses OpenCLIP-ViT-L/14 (Cherti et al., 2023), chosen in Appendix A.5
as the performance/latency knee (ViT-H/14 scores higher but is "prohibitive for
a real-time system").

THE CRITICAL DETAIL: QDP needs PATCH-LEVEL features that live in the SAME
embedding space as the text query, because M_sem thresholds `sim(q, v_t^i)`.
CLIP's `encode_image` returns only the pooled CLS embedding -- useless here. The
per-patch tokens must be taken from the final transformer block and pushed
through `ln_post` and `visual.proj` themselves, which is what `_forward_patches`
below does. Skipping `visual.proj` leaves the patches in the vision tower's
internal space, where cosine similarity against a text embedding is noise; that
is the single most likely way to silently break this reproduction.

Two implementations:
  OpenCLIPEncoder  -- the real thing, requires `pip install open_clip_torch`.
  SyntheticEncoder -- deterministic fake with controllable query alignment and
                      temporal dynamics, so every downstream module can be
                      tested on this cluster without weights or network access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple

import torch
import torch.nn as nn

from .config import EncoderConfig
from .ops import l2_normalize


@dataclass
class EncodedFrame:
    """Encoder output for one frame."""

    patch_feats: torch.Tensor   # [N, D] projected patch features
    frame_feat: torch.Tensor    # [D]    pooled frame vector \bar v_t
    grid_hw: Tuple[int, int]    # (H, W) patch grid this frame was encoded on

    @property
    def num_patches(self) -> int:
        return self.patch_feats.shape[0]


class VisionLanguageEncoder(Protocol):
    """Interface every encoder must satisfy, so QDP/RTAR never depend on CLIP."""

    embed_dim: int
    grid_hw: Tuple[int, int]

    def encode_text(self, text: str) -> torch.Tensor: ...
    def encode_frame(self, frame: torch.Tensor) -> EncodedFrame: ...


# --------------------------------------------------------------------------- #
# Real encoder
# --------------------------------------------------------------------------- #

class OpenCLIPEncoder(nn.Module):
    """OpenCLIP wrapper exposing projected PATCH tokens, not just the CLS pool."""

    def __init__(self, cfg: Optional[EncoderConfig] = None):
        super().__init__()
        try:
            import open_clip  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "OpenCLIPEncoder needs open_clip_torch:\n"
                "    pip install open_clip_torch\n"
                "Use SyntheticEncoder for tests that must run without it."
            ) from e
        import open_clip

        self.cfg = cfg or EncoderConfig()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.cfg.model_name,
            pretrained=self.cfg.pretrained,
            device=self.cfg.device,
        )
        self.model.eval().requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(self.cfg.model_name)
        self.device = torch.device(self.cfg.device)
        self.embed_dim = int(self.model.text_projection.shape[1]) \
            if hasattr(self.model, "text_projection") and self.model.text_projection is not None \
            else int(self.model.visual.output_dim)
        self.grid_hw = self.cfg.grid_hw

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """Query embedding q. Returned L2-normalised, [D] float32."""
        toks = self.tokenizer([text]).to(self.device)
        emb = self.model.encode_text(toks)
        return l2_normalize(emb.to(torch.float32), dim=-1).squeeze(0)

    @torch.no_grad()
    def _forward_patches(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the vision tower and return PROJECTED patch tokens [B, N, D].

        Mirrors open_clip's VisionTransformer.forward but keeps every token and
        drops the CLS position, then applies ln_post + proj so the tokens land
        in the shared image-text space.
        """
        v = self.model.visual
        x = pixel_values.to(self.device, dtype=self.cfg.dtype)

        x = v.conv1(x)                                        # [B, W, gh, gw]
        b, w, gh, gw = x.shape
        x = x.reshape(b, w, gh * gw).permute(0, 2, 1)         # [B, N, W]

        cls = v.class_embedding.to(x.dtype).reshape(1, 1, -1).expand(b, 1, -1)
        x = torch.cat([cls, x], dim=1)                        # [B, 1+N, W]
        x = x + v.positional_embedding.to(x.dtype)

        if getattr(v, "patch_dropout", None) is not None:
            x = v.patch_dropout(x)
        x = v.ln_pre(x)
        x = v.transformer(x)

        x = x[:, 1:, :]                                       # drop CLS -> [B, N, W]
        x = v.ln_post(x)
        if getattr(v, "proj", None) is not None:
            x = x @ v.proj                                    # -> shared space [B, N, D]
        return x.to(torch.float32), (gh, gw)

    @torch.no_grad()
    def encode_frame(self, frame: torch.Tensor) -> EncodedFrame:
        """Encode one preprocessed frame.

        Args:
            frame: [3, H, W] or [1, 3, H, W] float tensor, already normalised by
                   `self.preprocess`.
        """
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        assert frame.dim() == 4 and frame.shape[0] == 1, (
            f"encode_frame takes one frame, got {tuple(frame.shape)}"
        )
        patches, grid = self._forward_patches(frame)
        patch_feats = patches.squeeze(0)                       # [N, D]
        if self.cfg.normalize_features:
            patch_feats = l2_normalize(patch_feats, dim=-1)
        return EncodedFrame(
            patch_feats=patch_feats,
            frame_feat=patch_feats.mean(dim=0),
            grid_hw=grid,
        )

    @torch.no_grad()
    def encode_frames(self, frames: torch.Tensor) -> List[EncodedFrame]:
        """Batched variant. Args: frames [T, 3, H, W]."""
        assert frames.dim() == 4, f"expected [T, 3, H, W], got {tuple(frames.shape)}"
        patches, grid = self._forward_patches(frames)          # [T, N, D]
        if self.cfg.normalize_features:
            patches = l2_normalize(patches, dim=-1)
        return [
            EncodedFrame(patch_feats=p, frame_feat=p.mean(dim=0), grid_hw=grid)
            for p in patches
        ]


# --------------------------------------------------------------------------- #
# Synthetic encoder (tests)
# --------------------------------------------------------------------------- #

class SyntheticEncoder(nn.Module):
    """Deterministic stand-in for E with controllable structure.

    Generates patch features on the unit sphere where:
      * a configurable spatial REGION of each frame is pushed toward the query
        direction (so M_sem has a ground-truth answer to be checked against);
      * frames drift slowly, with optional injected "shock" frames (the black
        frames of the paper's Appendix A.7 case study) and "subtle event" frames,
        so DSH behaviour can be tested against a known signal.
    """

    def __init__(
        self,
        *,
        embed_dim: int = 768,
        grid_hw: Tuple[int, int] = (16, 16),
        seed: int = 0,
        device: str = "cpu",
        drift: float = 0.02,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_hw = grid_hw
        self.device = torch.device(device)
        self.drift = drift
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        n = grid_hw[0] * grid_hw[1]
        self._base = l2_normalize(
            torch.randn(n, embed_dim, generator=self.gen), dim=-1
        ).to(self.device)
        self._query = l2_normalize(
            torch.randn(embed_dim, generator=self.gen), dim=-1
        ).to(self.device)

    @property
    def num_patches(self) -> int:
        return self.grid_hw[0] * self.grid_hw[1]

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """Ignores `text` -- returns the fixed query direction the synthetic
        frames were built around, so relevance is a known quantity."""
        return self._query.clone()

    @torch.no_grad()
    def encode_frame(
        self,
        frame: Optional[torch.Tensor] = None,
        *,
        t: int = 0,
        relevant_patches: Optional[Sequence[int]] = None,
        relevance_strength: float = 0.8,
        shock: bool = False,
        subtle_delta: float = 0.0,
        motion: float = 0.0,
        distractor_patches: Optional[Sequence[int]] = None,
        distractor_motion: float = 0.0,
    ) -> EncodedFrame:
        """Synthesize frame t's features.

        Args:
            relevant_patches:  indices pushed toward the query direction.
            relevance_strength: how far toward q they are pushed, in [0, 1].
            shock:             replace the frame with an unrelated random field
                               (the "visually salient but irrelevant" case).
            subtle_delta:      small persistent perturbation -- the slow,
                               query-relevant action that frame-to-frame diffing
                               misses but DSH accumulates.
            motion:            per-frame jitter applied to the relevant region
                               AFTER the query mix. Without it the relevant
                               object is perfectly static, M_temp rejects all of
                               it, and the QDP intersection is empty -- which is
                               correct behaviour but a degenerate fixture. Real
                               query-relevant content both matches the query and
                               moves.
            distractor_patches / distractor_motion:
                               a region that MOVES but is NOT query-relevant --
                               the "actors unrelated to the query" of the paper's
                               Sec. 1. M_temp keeps it, M_sem rejects it, so the
                               intersection is strictly smaller than either
                               operand. Without this the two masks coincide
                               exactly and the conjunction is untested.
        """
        n, d = self.num_patches, self.embed_dim
        if shock:
            feats = l2_normalize(torch.randn(n, d, generator=self.gen), dim=-1).to(self.device)
        else:
            noise = torch.randn(n, d, generator=self.gen).to(self.device) * self.drift
            feats = self._base + noise + subtle_delta * t
            feats = l2_normalize(feats, dim=-1)

        feats = feats.clone()

        if distractor_patches is not None and len(distractor_patches) > 0 and distractor_motion > 0:
            didx = torch.tensor(list(distractor_patches), dtype=torch.long, device=self.device)
            jitter = torch.randn(len(didx), d, generator=self.gen).to(self.device) * distractor_motion
            feats[didx] = l2_normalize(feats[didx] + jitter, dim=-1)

        if relevant_patches is not None and len(relevant_patches) > 0:
            idx = torch.tensor(list(relevant_patches), dtype=torch.long, device=self.device)
            mixed = (1.0 - relevance_strength) * feats[idx] + relevance_strength * self._query
            if motion > 0.0:
                jitter = torch.randn(len(idx), d, generator=self.gen).to(self.device) * motion
                mixed = mixed + jitter
            feats[idx] = l2_normalize(mixed, dim=-1)

        return EncodedFrame(
            patch_feats=feats,
            frame_feat=feats.mean(dim=0),
            grid_hw=self.grid_hw,
        )


__all__ = [
    "EncodedFrame",
    "VisionLanguageEncoder",
    "OpenCLIPEncoder",
    "SyntheticEncoder",
]
