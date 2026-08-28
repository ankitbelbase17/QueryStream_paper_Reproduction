"""OpenCLIP-ViT-L/14 feature extraction for QDP/RTAR -- paper Sec. 3.1 / 4.1.

VERIFIED against open_clip_torch 3.3.0 on this cluster:
    ViT-L-14 / openai : grid_size (16, 16) -> 256 patches
                        visual.proj  (1024 -> 768)
                        attn_pool None, final_ln_after_pool False, pool_type 'tok'
                        logit_scale  100.0

THE PROJECTION TRAP
-------------------
open_clip 3.3.0's `VisionTransformer.forward` applies `visual.proj` to the
POOLED CLS embedding ONLY:

        pooled, tokens = self._pool(x)
        if self.proj is not None:
            pooled = pooled @ self.proj          # <- tokens are NOT projected

So `output_tokens=True` hands back per-patch tokens that have had `ln_post`
applied but never `proj`. For ViT-L/14 those are 1024-d while the text embedding
is 768-d, so `sim(q, v_t^i)` cannot even be formed -- and on a checkpoint where
the dimensions happen to coincide it would silently produce noise. QDP's entire
semantic criterion depends on patches and text sharing one space, so this
wrapper applies `@ visual.proj` to the patch tokens itself.

Measured on real weights (see tests): patch-level `sim(q, v)` lands in roughly
[0.12, 0.22] and frame-level `sim(q, mean_pool(v))` around 0.21. That is the
empirical basis for the tau_rel calibration warning in rtar.py -- a literal
tau_rel = 0.60 gate never fires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EncoderConfig
from .encoder import EncodedFrame
from .ops import l2_normalize

# CLIP's channel statistics (identical across OpenAI ViT-L/14 checkpoints).
CLIP_MEAN = (0.48145466, 0.45782750, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _resolve_quickgelu(model_name: str, pretrained: str) -> str:
    """Return the model config whose activation matches `pretrained`'s weights.

    open_clip records `quick_gelu` per pretrained tag. If the tag wants it and
    the named config does not provide it (or the reverse), the built model has
    the wrong activation for its own weights. open_clip issues a UserWarning and
    proceeds; this promotes that warning into a corrected model name, and falls
    back to the original if no matching variant is registered.
    """
    try:
        import open_clip
        from open_clip.pretrained import get_pretrained_cfg
    except Exception:                                   # pragma: no cover
        return model_name
    if not pretrained:
        return model_name
    cfg = get_pretrained_cfg(model_name, pretrained) or {}
    wants = bool(cfg.get("quick_gelu", False))
    has = model_name.endswith("-quickgelu")
    if wants == has:
        return model_name
    candidate = f"{model_name}-quickgelu" if wants else model_name[: -len("-quickgelu")]
    try:
        available = set(open_clip.list_models())
    except Exception:                                   # pragma: no cover
        return model_name
    return candidate if candidate in available else model_name


class OpenCLIPWrapper(nn.Module):
    """Frozen OpenCLIP encoder exposing PROJECTED patch tokens.

    Args:
        cfg: EncoderConfig. Defaults reproduce the paper (ViT-L-14 / openai, 224px).
    """

    def __init__(self, cfg: Optional[EncoderConfig] = None):
        super().__init__()
        try:
            import open_clip
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "OpenCLIPWrapper requires open_clip_torch:  pip install open_clip_torch"
            ) from e

        self.cfg = cfg or EncoderConfig()
        self.device = torch.device(self.cfg.device)

        # The OpenAI CLIP checkpoints were trained with QuickGELU, but open_clip's
        # bare "ViT-L-14" config specifies plain GELU. Pairing them builds a model
        # whose activation does not match its weights -- open_clip only warns, and
        # the result is silently degraded features feeding every threshold in QDP
        # and RTAR. Resolve it to the "-quickgelu" variant instead of running the
        # mismatch.
        self.model_name = _resolve_quickgelu(self.cfg.model_name, self.cfg.pretrained)
        if self.model_name != self.cfg.model_name:
            print(f"[openclip] '{self.cfg.pretrained}' weights need QuickGELU: "
                  f"using '{self.model_name}' instead of '{self.cfg.model_name}'")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.cfg.pretrained, device=self.device
        )
        self.model.eval().requires_grad_(False)
        # create_model_and_transforms honours `device` but NOT dtype -- weights stay
        # fp32. Cast explicitly, then read the dtype back off the parameters so
        # every input cast below is derived from what the model actually is. A
        # cfg-derived cast silently desynchronises the moment anything else
        # re-casts the module.
        if self.cfg.dtype != torch.float32:
            self.model = self.model.to(self.cfg.dtype)
        self.model_dtype = next(self.model.parameters()).dtype
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

        v = self.model.visual
        if getattr(v, "attn_pool", None) is not None:
            raise NotImplementedError(
                f"{self.model_name} uses attn_pool; the patch-extraction path "
                "below assumes the plain CLS+patches layout of ViT-L/14."
            )
        if v.proj is None:
            raise RuntimeError(
                f"{self.model_name}/{self.cfg.pretrained} has no visual.proj, so "
                "patch tokens cannot be mapped into the shared image-text space."
            )

        self.embed_dim = int(v.proj.shape[1])
        gs = getattr(v, "grid_size", None)
        self.grid_hw: Tuple[int, int] = tuple(gs) if gs is not None else self.cfg.grid_hw
        self.image_size = (
            v.image_size if isinstance(v.image_size, (tuple, list)) else (v.image_size, v.image_size)
        )
        self.logit_scale = float(self.model.logit_scale.detach().exp().item())

        self.register_buffer(
            "_mean", torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_std", torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1), persistent=False
        )

    # ------------------------------------------------------------------ #

    @property
    def num_patches(self) -> int:
        return self.grid_hw[0] * self.grid_hw[1]

    def describe(self) -> dict:
        return {
            "model": f"{self.cfg.model_name}/{self.cfg.pretrained}",
            "embed_dim": self.embed_dim,
            "grid_hw": self.grid_hw,
            "num_patches": self.num_patches,
            "image_size": self.image_size,
            "logit_scale": self.logit_scale,
            "dtype": str(self.model_dtype),
            "device": str(self.device),
        }

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def preprocess_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Resize + CLIP-normalise raw RGB frames, on-device and batched.

        Avoids a PIL round-trip per frame, which dominates wall-clock at 1 FPS
        over long videos.

        Args:
            frames: [T, 3, H, W] uint8 in [0, 255] or float in [0, 1].
        Returns:
            [T, 3, 224, 224] in the encoder's dtype.
        """
        assert frames.dim() == 4 and frames.shape[1] == 3, (
            f"expected [T, 3, H, W] RGB, got {tuple(frames.shape)}"
        )
        x = frames.to(self.device)
        x = x.float() / 255.0 if x.dtype == torch.uint8 else x.float()
        if tuple(x.shape[-2:]) != tuple(self.image_size):
            x = F.interpolate(
                x, size=self.image_size, mode="bicubic", align_corners=False, antialias=True
            ).clamp_(0, 1)
        return ((x - self._mean) / self._std).to(self.model_dtype)

    # ------------------------------------------------------------------ #
    # Text
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def encode_text(self, text: Union[str, Sequence[str]]) -> torch.Tensor:
        """Query embedding q, L2-normalised float32. [D] for one string, [B, D] for many."""
        single = isinstance(text, str)
        toks = self.tokenizer([text] if single else list(text)).to(self.device)
        emb = self.model.encode_text(toks)
        emb = l2_normalize(emb.to(torch.float32), dim=-1)   # normalise in fp32
        return emb.squeeze(0) if single else emb

    # ------------------------------------------------------------------ #
    # Vision -- projected patch tokens
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _patch_tokens(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vision tower -> (projected patch tokens [B, N, D], pooled CLS [B, D]).

        Uses open_clip's own `_embeds` so patch-embedding/pos-embedding/patch-dropout
        stay in sync with the installed version, then reproduces `_pool`'s
        `ln_post` and applies `proj` to the patches, which `_pool` does not.
        """
        v = self.model.visual
        x = pixel_values.to(self.device, dtype=self.model_dtype)

        if hasattr(v, "_embeds"):
            x = v._embeds(x)                       # open_clip >= 2.24
        else:                                      # pragma: no cover - legacy path
            from open_clip.transformer import _expand_token
            x = v.conv1(x)
            b, w = x.shape[0], x.shape[1]
            x = x.reshape(b, w, -1).permute(0, 2, 1)
            x = torch.cat([_expand_token(v.class_embedding, b).to(x.dtype), x], dim=1)
            x = x + v.positional_embedding.to(x.dtype)
            x = v.patch_dropout(x)
            x = v.ln_pre(x)

        x = v.transformer(x)
        x = v.ln_post(x)                           # `_pool` applies ln_post to ALL tokens
        cls, patches = x[:, 0], x[:, 1:]           # pool_type 'tok' -> CLS is index 0

        patches = patches @ v.proj                 # <- the step open_clip omits
        cls = cls @ v.proj
        return patches.to(torch.float32), cls.to(torch.float32)

    @torch.no_grad()
    def encode_frames(
        self,
        frames: torch.Tensor,
        *,
        preprocessed: bool = False,
        batch_size: int = 32,
    ) -> List[EncodedFrame]:
        """Encode a stack of frames into QDP-ready patch features.

        Args:
            frames: [T, 3, H, W]. Raw RGB unless `preprocessed=True`.
            batch_size: encoder minibatch; bounds VRAM on long streams.
        Returns:
            List of `EncodedFrame`, one per input frame, patch_feats [N, D] float32.
        """
        px = frames if preprocessed else self.preprocess_frames(frames)
        outs: List[EncodedFrame] = []
        for i in range(0, px.shape[0], batch_size):
            patches, _ = self._patch_tokens(px[i : i + batch_size])
            if self.cfg.normalize_features:
                patches = l2_normalize(patches, dim=-1)
            for p in patches:
                # \bar v_t is the mean over patches (paper Sec. 3.3).
                outs.append(
                    EncodedFrame(patch_feats=p, frame_feat=p.mean(dim=0), grid_hw=self.grid_hw)
                )
        return outs

    @torch.no_grad()
    def encode_frame(self, frame: torch.Tensor, *, preprocessed: bool = False) -> EncodedFrame:
        """Single-frame convenience wrapper. Args: frame [3, H, W] or [1, 3, H, W]."""
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        return self.encode_frames(frame, preprocessed=preprocessed)[0]


__all__ = ["OpenCLIPWrapper", "CLIP_MEAN", "CLIP_STD"]
