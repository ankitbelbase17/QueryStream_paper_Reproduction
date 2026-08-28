"""Qwen2.5-VL-7B-Instruct backbone with just-in-time pruned-token delivery.

Verified against transformers 5.9.0 / Qwen2.5-VL-7B-Instruct on this cluster:
    class              Qwen2_5_VLForConditionalGeneration   (underscored)
    hidden_size        3584
    patch_size         14
    spatial_merge_size 2      -> 2x2 spatial merge
    temporal_patch_size 2     -> 2 frames per temporal grid plane
    tokens_per_second  2

M-ROPE, EXACTLY AS transformers COMPUTES IT
-------------------------------------------
From `Qwen2_5_VLModel.get_vision_position_ids`, for one video block:

    llm_grid_t, llm_grid_h, llm_grid_w = T, H // merge, W // merge
    time_interval = tokens_per_second * int(second_per_grid_t)
    t = (arange(llm_grid_t) * time_interval).repeat_interleave(llm_grid_h*llm_grid_w) + start
    h = (arange(llm_grid_h) + start).repeat_interleave(llm_grid_w).repeat(llm_grid_t)
    w = (arange(llm_grid_w) + start).repeat(llm_grid_h * llm_grid_t)
    position_ids = stack([t, h, w])                       # [3, N], raster (t, h, w)

Three consequences that a naive pruning implementation gets wrong:

1. ALL THREE axes carry the `start_position` offset, not just the temporal one.
2. At 1 FPS, `second_per_grid_t = temporal_patch_size / fps = 2`, so
   `time_interval = 2 * 2 = 4`. Consecutive temporal planes are 4 apart, NOT 1.
   Using the frame index directly compresses the model's sense of elapsed time.
3. After a video block the running text position advances by
   `max(llm_grid_h, llm_grid_w)` computed from the DENSE grid. Pruning must not
   change it -- otherwise the text following the video shifts position depending
   on how many patches survived, and the pruned run stops being comparable to
   the 100%-keep control.

This module therefore builds the DENSE coordinates for a chunk, hands them to
the TokenBuffer alongside the tokens, and lets the buffer slice both with one
index set. The surviving coordinates are used verbatim -- never renumbered.

GENERATION
----------
`Qwen2_5_VLModel.forward` uses `position_ids` verbatim when supplied. During
decoding (`past_key_values_length > 0`) it falls back to
`cumsum(attention_mask) - 1 + self.rope_deltas`, so a custom prefill only stays
consistent if `model.model.rope_deltas` is set to match. `generate_from_pruned`
sets it to `max(prefill_positions) + 1 - prefill_len`, the same definition
`get_rope_index` uses.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .config import BackboneConfig


@dataclass
class VideoChunk:
    """One `temporal_patch_size`-frame plane, encoded to backbone visual tokens."""

    tokens: torch.Tensor            # [N, hidden] visual tokens (post spatial merge)
    mrope: torch.Tensor             # [N, 3] int64 RELATIVE (t*interval, h, w)
    grid_thw: torch.Tensor          # [3] dense (T, H, W) in pre-merge patch units
    llm_grid_hw: Tuple[int, int]    # merged (H, W) -- the grid QDP masks must match
    num_frames: int
    time_interval: int

    @property
    def num_tokens(self) -> int:
        return self.tokens.shape[0]


@dataclass
class GenerationResult:
    text: str
    num_visual_tokens: int
    num_prefill_tokens: int
    prefill_ms: float          # time to first token (TTFT)
    decode_ms: float           # first token -> last token
    peak_vram_bytes: int
    rope_delta: int
    num_new_tokens: int = 0

    @property
    def total_ms(self) -> float:
        return self.prefill_ms + self.decode_ms


class _FirstTokenClock:
    """Stamps the wall clock the first time generate() produces logits.

    `generate()` reports one elapsed time for prefill + decode, but the two have
    completely different cost structures and the paper's efficiency claim is
    about prefill (it is what shrinks when tokens are pruned). A logits
    processor is invoked once per generated token, and its FIRST invocation
    happens immediately after the prefill forward pass -- so that timestamp is
    time-to-first-token. It synchronises once, on that first call only, because
    CUDA queues would otherwise make the reading meaningless.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.first: Optional[float] = None
        self.n_calls = 0

    def __call__(self, input_ids, scores):
        self.n_calls += 1
        if self.first is None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            import time as _t
            self.first = _t.perf_counter()
        return scores


class QwenVLWrapper(nn.Module):
    """Frozen Qwen2.5-VL backbone that accepts an externally pruned token set."""

    def __init__(
        self,
        cfg: Optional[BackboneConfig] = None,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
        max_pixels: Optional[int] = None,
    ):
        super().__init__()
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.cfg = cfg or BackboneConfig()
        self.device = torch.device(device)
        self.dtype = dtype

        proc_kwargs: Dict[str, Any] = {}
        if max_pixels is not None:
            # Bounds tokens/frame. Qwen's smart_resize otherwise scales with the
            # source resolution and the token count (hence the keep-rate
            # denominator) drifts between videos.
            proc_kwargs["max_pixels"] = max_pixels
        self.processor = AutoProcessor.from_pretrained(self.cfg.model_id, **proc_kwargs)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.cfg.model_id, dtype=dtype, attn_implementation=attn_implementation
        ).to(self.device)
        self.model.eval().requires_grad_(False)

        vc = self.model.config.vision_config
        self.patch_size = int(vc.patch_size)
        self.spatial_merge_size = int(vc.spatial_merge_size)
        self.temporal_patch_size = int(vc.temporal_patch_size)
        self.tokens_per_second = int(vc.tokens_per_second)
        self.hidden_size = int(self.model.config.text_config.hidden_size)

        tok = self.processor.tokenizer
        self.video_pad_id = tok.convert_tokens_to_ids("<|video_pad|>")
        self.vision_start_id = tok.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_id = tok.convert_tokens_to_ids("<|vision_end|>")
        assert None not in (self.video_pad_id, self.vision_start_id, self.vision_end_id)

    # ------------------------------------------------------------------ #

    def describe(self) -> dict:
        return {
            "model_id": self.cfg.model_id,
            "hidden_size": self.hidden_size,
            "patch_size": self.patch_size,
            "spatial_merge_size": self.spatial_merge_size,
            "temporal_patch_size": self.temporal_patch_size,
            "tokens_per_second": self.tokens_per_second,
            "dtype": str(self.dtype),
            "device": str(self.device),
        }

    def second_per_grid_t(self, fps: float) -> float:
        """Seconds of wall-clock covered by one temporal grid plane."""
        return self.temporal_patch_size / float(fps)

    def time_interval(self, fps: float) -> int:
        """`tokens_per_second * int(second_per_grid_t)` -- transformers' own formula."""
        return self.tokens_per_second * int(self.second_per_grid_t(fps))

    # ------------------------------------------------------------------ #
    # Vision path
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def preprocess_video(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run Qwen's own video preprocessing (incl. smart_resize).

        Args:
            frames: [T, 3, H, W] uint8 RGB. T must be a multiple of
                    `temporal_patch_size` (the caller chunks the stream).
        Returns:
            (pixel_values_videos, video_grid_thw [1, 3] dense pre-merge (T, H, W))
        """
        assert frames.dim() == 4 and frames.shape[1] == 3, (
            f"expected [T, 3, H, W] uint8 RGB, got {tuple(frames.shape)}"
        )
        assert frames.shape[0] % self.temporal_patch_size == 0, (
            f"chunk of {frames.shape[0]} frames is not a multiple of "
            f"temporal_patch_size={self.temporal_patch_size}; Qwen pairs frames "
            f"into temporal planes and a partial plane gets edge-padded, which "
            f"silently changes the token count"
        )
        vp = getattr(self.processor, "video_processor", None)
        arr = frames.permute(0, 2, 3, 1).cpu().numpy()   # [T, H, W, 3] uint8
        if vp is not None:
            out = vp(videos=[arr], return_tensors="pt")
        else:  # pragma: no cover - older processor layout
            out = self.processor.image_processor(images=None, videos=[arr], return_tensors="pt")
        return (
            out["pixel_values_videos"].to(self.device, self.dtype),
            out["video_grid_thw"].to(self.device),
        )

    @torch.no_grad()
    def encode_chunk(self, frames: torch.Tensor, *, fps: float = 1.0) -> VideoChunk:
        """Preprocess + run the vision tower on one chunk, and build dense M-ROPE.

        The vision tower runs per chunk so the buffer holds *visual tokens*, which
        is what the paper specifies ("the original, unpruned visual tokens are
        temporarily held in a memory buffer"). Buffering pixels instead and
        deferring the tower to trigger time would be cheaper per non-trigger
        frame but is not what Sec. 3.1 describes.
        """
        pvv, grid_thw = self.preprocess_video(frames)
        tokens = self._extract_video_features(pvv, grid_thw)

        g = grid_thw[0]
        ti = self.time_interval(fps)
        mrope = self.build_dense_mrope(g, time_interval=ti, start_position=0)
        assert mrope.shape[0] == tokens.shape[0], (
            f"M-ROPE rows ({mrope.shape[0]}) != visual tokens ({tokens.shape[0]}); "
            f"grid {g.tolist()} with merge {self.spatial_merge_size}"
        )
        m = self.spatial_merge_size
        return VideoChunk(
            tokens=tokens,
            mrope=mrope,
            grid_thw=g,
            llm_grid_hw=(int(g[1]) // m, int(g[2]) // m),
            num_frames=frames.shape[0],
            time_interval=ti,
        )

    @torch.no_grad()
    def _extract_video_features(
        self, pixel_values_videos: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """Run the vision tower and normalise its return shape to [N, hidden].

        transformers 5.9.0 returns a `BaseModelOutputWithPooling` whose
        `pooler_output` is a TUPLE of per-video tensors (split by
        `grid_thw.prod(-1) // merge**2`). Older versions returned a bare tensor.
        Both are handled so a transformers upgrade cannot silently change what
        gets buffered.
        """
        out = self.model.model.get_video_features(pixel_values_videos, grid_thw)

        if torch.is_tensor(out):
            feats = out
        elif isinstance(out, (list, tuple)) and out and torch.is_tensor(out[0]):
            feats = torch.cat([t.reshape(-1, self.hidden_size) for t in out], dim=0)
        else:
            pooled = getattr(out, "pooler_output", None)
            if pooled is None:
                pooled = getattr(out, "last_hidden_state", None)
            if pooled is None:
                raise RuntimeError(
                    f"cannot locate visual features on {type(out).__name__}; "
                    f"attributes: {list(getattr(out, 'keys', lambda: [])()) or dir(out)[:20]}"
                )
            feats = (
                torch.cat([t.reshape(-1, self.hidden_size) for t in pooled], dim=0)
                if isinstance(pooled, (list, tuple)) else pooled
            )

        feats = feats.reshape(-1, self.hidden_size)
        expected = int((grid_thw.prod(-1) // (self.spatial_merge_size ** 2)).sum())
        assert feats.shape[0] == expected, (
            f"vision tower produced {feats.shape[0]} tokens, geometry predicts "
            f"{expected} for grid {grid_thw.tolist()} at merge {self.spatial_merge_size}"
        )
        return feats

    def build_dense_mrope(
        self,
        grid_thw: torch.Tensor,
        *,
        time_interval: int,
        start_position: int = 0,
    ) -> torch.Tensor:
        """Dense (t, h, w) coordinates for one video block, [N, 3] int64.

        Byte-for-byte the transpose of `get_vision_position_ids`. Verified against
        it in `tests/test_integration.py` -- if transformers changes the raster
        order, that assertion fails loudly rather than corrupting positions.
        """
        m = self.spatial_merge_size
        t_g, h_g, w_g = int(grid_thw[0]), int(grid_thw[1]) // m, int(grid_thw[2]) // m
        dev = self.device

        pos_t = (torch.arange(t_g, device=dev) * time_interval).repeat_interleave(h_g * w_g)
        pos_h = torch.arange(h_g, device=dev).repeat_interleave(w_g).repeat(t_g)
        pos_w = torch.arange(w_g, device=dev).repeat(h_g * t_g)
        return (torch.stack([pos_t, pos_h, pos_w], dim=1) + start_position).to(torch.long)

    # ------------------------------------------------------------------ #
    # Prompt assembly
    # ------------------------------------------------------------------ #

    def build_prompt_ids(self, query: str, num_visual_tokens: int,
                         system: Optional[str] = None) -> torch.Tensor:
        """Chat-template ids whose video placeholder is exactly K pad tokens.

        The processor normally expands `<|video_pad|>` to the DENSE token count.
        We expand it to the PRUNED count instead, which is the whole point.
        """
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        messages.append({
            "role": "user",
            "content": [{"type": "video"}, {"type": "text", "text": query}],
        })
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if "<|video_pad|>" not in text:
            raise RuntimeError(
                "chat template produced no <|video_pad|> placeholder; the video "
                "content block was not rendered"
            )
        text = text.replace("<|video_pad|>", "<|video_pad|>" * max(num_visual_tokens, 1), 1)
        ids = self.processor.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        n_pad = int((ids[0] == self.video_pad_id).sum())
        assert n_pad == num_visual_tokens, (
            f"placeholder expansion produced {n_pad} pad tokens, expected "
            f"{num_visual_tokens}"
        )
        return ids

    @torch.no_grad()
    def build_position_ids(
        self,
        input_ids: torch.Tensor,
        mrope: torch.Tensor,
        dense_llm_grid_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, int]:
        """3D M-ROPE position ids for a prefill containing one pruned video block.

        Args:
            input_ids:          [1, L]
            mrope:              [K, 3] SURVIVING relative coords from the buffer.
            dense_llm_grid_hw:  merged (H, W) of the UNPRUNED grid. The text after
                                the video advances by max(H, W) of this, so text
                                positions are identical to the 100%-keep control.
        Returns:
            (position_ids [3, 1, L] int64, rope_delta int)
        """
        assert input_ids.shape[0] == 1, "batch size 1 only (streaming)"
        ids = input_ids[0]
        L = ids.numel()
        is_vid = ids == self.video_pad_id
        vid_idx = is_vid.nonzero(as_tuple=True)[0]
        assert vid_idx.numel() == mrope.shape[0], (
            f"{vid_idx.numel()} video pad tokens but {mrope.shape[0]} M-ROPE rows"
        )
        if vid_idx.numel() == 0:
            pos = torch.arange(L, device=ids.device).view(1, 1, -1).expand(3, 1, -1)
            return pos.contiguous().to(torch.long), 0
        # single contiguous video block (streaming assembles exactly one)
        assert int(vid_idx[-1] - vid_idx[0]) == vid_idx.numel() - 1, (
            "video pad tokens are not contiguous"
        )
        a, b = int(vid_idx[0]), int(vid_idx[-1]) + 1

        pos = torch.zeros(3, L, dtype=torch.long, device=ids.device)
        # leading text
        pos[:, :a] = torch.arange(a, device=ids.device).view(1, -1).expand(3, -1)
        start = a
        # video block: surviving coords, offset on ALL THREE axes
        pos[:, a:b] = (mrope.to(ids.device) + start).T
        # trailing text advances past the DENSE spatial extent, not the pruned one
        h_g, w_g = dense_llm_grid_hw
        nxt = start + max(h_g, w_g)
        tail = L - b
        if tail > 0:
            pos[:, b:] = (torch.arange(tail, device=ids.device) + nxt).view(1, -1).expand(3, -1)

        rope_delta = int(pos.max().item()) + 1 - L
        return pos.unsqueeze(1).contiguous(), rope_delta

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate_from_pruned(
        self,
        *,
        query: str,
        visual_tokens: torch.Tensor,
        mrope: torch.Tensor,
        dense_llm_grid_hw: Tuple[int, int],
        system: Optional[str] = None,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 1.0,
        profile_vram: bool = True,
    ) -> GenerationResult:
        """Answer `query` from an externally pruned visual token set.

        Args:
            visual_tokens: [K, hidden] surviving tokens, buffer order.
            mrope:         [K, 3] their relative (t, h, w) -- SAME order.
        """
        import time as _time

        assert visual_tokens.dim() == 2, f"[K, hidden] expected, got {tuple(visual_tokens.shape)}"
        assert visual_tokens.shape[0] == mrope.shape[0], (
            f"visual_tokens ({visual_tokens.shape[0]}) and mrope ({mrope.shape[0]}) "
            "disagree -- token/coordinate pairing was broken upstream"
        )
        assert visual_tokens.shape[1] == self.hidden_size, (
            f"token dim {visual_tokens.shape[1]} != backbone hidden {self.hidden_size}"
        )
        k = visual_tokens.shape[0]

        input_ids = self.build_prompt_ids(query, k, system=system)
        position_ids, rope_delta = self.build_position_ids(input_ids, mrope, dense_llm_grid_hw)

        embed = self.model.get_input_embeddings()
        inputs_embeds = embed(input_ids).clone()
        vid_mask = (input_ids[0] == self.video_pad_id)
        inputs_embeds[0, vid_mask] = visual_tokens.to(inputs_embeds.dtype)

        attention_mask = torch.ones_like(input_ids)

        # Decoding falls back to cumsum + rope_deltas; keep it consistent with our
        # custom prefill or generated tokens land at the wrong positions.
        self.model.model.rope_deltas = torch.tensor(
            [[rope_delta]], device=self.device, dtype=torch.long
        )

        if profile_vram and self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(self.device)

        ttft = _FirstTokenClock(self.device)
        t0 = _time.perf_counter()
        out = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=None if not do_sample else 0.9,
            pad_token_id=self.processor.tokenizer.pad_token_id
            or self.processor.tokenizer.eos_token_id,
            logits_processor=[ttft],
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_end = _time.perf_counter()
        prefill_ms = ((ttft.first or t_end) - t0) * 1e3
        decode_ms = (t_end - (ttft.first or t_end)) * 1e3

        peak = (
            int(torch.cuda.max_memory_allocated(self.device))
            if profile_vram and self.device.type == "cuda" else 0
        )
        # With inputs_embeds, generate() returns ONLY the new tokens.
        text = self.processor.tokenizer.decode(out[0], skip_special_tokens=True).strip()

        return GenerationResult(
            text=text,
            num_visual_tokens=k,
            num_prefill_tokens=int(input_ids.shape[1]),
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            peak_vram_bytes=peak,
            rope_delta=rope_delta,
            num_new_tokens=int(out.shape[1]),
        )


__all__ = ["QwenVLWrapper", "VideoChunk", "GenerationResult"]


class StubBackbone(nn.Module):
    """Weightless stand-in with Qwen2.5-VL's EXACT token geometry.

    Same constants (patch 14, spatial merge 2, temporal patch 2,
    tokens_per_second 2, hidden 3584) and the same `encode_chunk` contract, but
    no weights -- so the resample -> temporal-fold -> buffer -> M-ROPE path can be
    tested on a machine with no free GPU, and in CI.

    Its tokens ENCODE THEIR OWN (t, h, w) in the first three channels, which
    turns the paper's spatio-temporal-integrity guarantee into a structural
    assertion: after any pruning, `tokens[:, :3] == mrope` must still hold.
    """

    def __init__(
        self,
        cfg: Optional[BackboneConfig] = None,
        *,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        grid_hw: Tuple[int, int] = (16, 16),   # pre-merge patch grid -> 8x8 merged
        hidden_size: int = 3584,
    ):
        super().__init__()
        self.cfg = cfg or BackboneConfig()
        self.device = torch.device(device)
        self.dtype = dtype
        self.patch_size = self.cfg.patch_size
        self.spatial_merge_size = self.cfg.spatial_merge_size
        self.temporal_patch_size = self.cfg.temporal_patch_size
        self.tokens_per_second = 2
        self.hidden_size = hidden_size
        self._grid_hw = grid_hw

    describe = QwenVLWrapper.describe
    second_per_grid_t = QwenVLWrapper.second_per_grid_t
    time_interval = QwenVLWrapper.time_interval
    build_dense_mrope = QwenVLWrapper.build_dense_mrope

    @torch.no_grad()
    def encode_chunk(self, frames: torch.Tensor, *, fps: float = 1.0) -> VideoChunk:
        assert frames.shape[0] % self.temporal_patch_size == 0, (
            f"chunk of {frames.shape[0]} frames is not a multiple of "
            f"temporal_patch_size={self.temporal_patch_size}"
        )
        t_planes = frames.shape[0] // self.temporal_patch_size
        h, w = self._grid_hw
        grid = torch.tensor([t_planes, h, w], device=self.device)
        ti = self.time_interval(fps)
        mrope = self.build_dense_mrope(grid, time_interval=ti, start_position=0)

        tokens = torch.randn(mrope.shape[0], self.hidden_size,
                             device=self.device, dtype=self.dtype)
        tokens[:, :3] = mrope.to(self.dtype)      # self-describing coordinates
        m = self.spatial_merge_size
        return VideoChunk(
            tokens=tokens, mrope=mrope, grid_thw=grid,
            llm_grid_hw=(h // m, w // m), num_frames=frames.shape[0], time_interval=ti,
        )


__all__ = ["QwenVLWrapper", "StubBackbone", "VideoChunk", "GenerationResult"]
