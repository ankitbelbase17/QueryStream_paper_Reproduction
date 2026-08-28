"""Pick the backbone wrapper that matches a model id.

Qwen2.5-VL and Qwen3-VL present the same public surface to the runners
(`encode_chunk` / `generate_from_pruned`) but their token geometry differs
enough to need separate implementations -- see the `qwen3_wrapper` module
docstring. Selecting on the model id keeps every eval driver a one-line change
and makes an unrecognised backbone fail loudly at load rather than at the first
generation.
"""

from __future__ import annotations

from typing import Optional

import torch


def is_qwen3(model_id: str) -> bool:
    m = model_id.lower()
    return "qwen3-vl" in m or "qwen3_vl" in m


def make_backbone(cfg, *, device: str = "cuda",
                  max_pixels: Optional[int] = None,
                  dtype: torch.dtype = torch.bfloat16):
    """Construct the wrapper for `cfg.model_id`."""
    mid = cfg.model_id
    if is_qwen3(mid):
        from .qwen3_wrapper import Qwen3VLWrapper
        return Qwen3VLWrapper(cfg, device=device, dtype=dtype, max_pixels=max_pixels)
    if "qwen2.5-vl" in mid.lower() or "qwen2_5_vl" in mid.lower() \
            or "timechatonline" in mid.lower().replace("-", "") \
            or "timechat-online" in mid.lower():
        from .qwen_wrapper import QwenVLWrapper
        return QwenVLWrapper(cfg, device=device, dtype=dtype, max_pixels=max_pixels)
    raise ValueError(
        f"no backbone wrapper registered for {mid!r}. Qwen2.5-VL derivatives "
        f"(including TimeChat-Online) use QwenVLWrapper; Qwen3-VL uses "
        f"Qwen3VLWrapper. Add an explicit branch rather than guessing -- the "
        f"two families differ in M-ROPE layout and deepstack injection, and a "
        f"wrong guess produces plausible but positionally corrupt output."
    )


__all__ = ["make_backbone", "is_qwen3"]
