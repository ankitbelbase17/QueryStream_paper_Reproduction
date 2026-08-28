"""Qwen3-VL-8B-Instruct as a QueryStream backbone.

Qwen3-VL is NOT a drop-in for Qwen2.5-VL. Four differences matter here, and
each one changes something this file has to do:

1. **Per-frame vision blocks, not one video block.** Qwen2.5-VL emits a single
   `<|vision_start|> ...all planes... <|vision_end|>` span whose M-ROPE
   temporal axis advances by `tokens_per_second * second_per_grid_t`. Qwen3-VL
   interleaves a timestamp with every temporal plane:

       <0.5 seconds><|vision_start|>plane0<|vision_end|>
       <2.5 seconds><|vision_start|>plane1<|vision_end|> ...

   and `get_rope_index` splits `video_grid_thw` per plane, giving each block
   `T=1` and `time_interval=1`. So a plane's three M-ROPE axes all start from
   the running position, and the temporal axis is CONSTANT within a plane.
   There is no `tokens_per_second` in the config at all.

2. **DeepStack.** The vision tower returns three extra feature maps (config
   `deepstack_visual_indexes = [8, 16, 24]`) that `Qwen3VLTextModel` ADDS into
   the hidden states at those decoder layers, at the visual positions only:
   `hidden_states[visual_pos_masks] += visual_embeds`. They are 1:1 with the
   merged visual tokens, so a pruned token set MUST carry a pruned deepstack in
   exactly the same order -- otherwise every surviving token gets another
   token's mid-level features added to it. They are carried through the
   existing `TokenBuffer` by concatenating them onto the token feature
   dimension (see `encode_chunk`), so the buffer's row slicing prunes them for
   free and no pairing can drift.

3. **`generate()` cannot be used.** `Qwen3VLModel.forward` computes the
   deepstack features itself from `pixel_values` and passes them positionally
   to the text model; there is no kwarg path that accepts precomputed ones. So
   prefill and greedy decode are written out by hand here.

4. **`max_pixels` is a budget over the whole chunk, not per frame.** Qwen3-VL's
   video `smart_resize` bounds `t_bar * h_bar * w_bar`. `max_pixels` is
   therefore multiplied by `temporal_patch_size` before being handed over, so
   the argument keeps the same per-frame meaning it has in the Qwen2.5 arm
   (§12.6 decision 4: a fixed cap so `smart_resize` cannot make the token
   count -- and hence the keep-rate denominator -- drift between videos).

Public surface is deliberately identical to `QwenVLWrapper` so the runners,
the buffer and the eval drivers are untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import BackboneConfig
from .qwen_wrapper import GenerationResult, VideoChunk

DEFAULT_QWEN3_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


@dataclass
class Qwen3Geometry:
    """What one call to the vision tower produced, in grid terms."""

    grid_thw: torch.Tensor          # [3] dense pre-merge (T, H, W)
    llm_grid_hw: Tuple[int, int]    # merged (H, W) of ONE plane
    tokens_per_plane: int
    n_planes: int


class Qwen3VLWrapper(nn.Module):
    """Frozen Qwen3-VL backbone that accepts an externally pruned token set."""

    def __init__(
        self,
        cfg: Optional[BackboneConfig] = None,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
        max_pixels: Optional[int] = None,
        fps: float = 1.0,
    ):
        super().__init__()
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.cfg = cfg or BackboneConfig(model_id=DEFAULT_QWEN3_MODEL)
        self.device = torch.device(device)
        self.dtype = dtype
        self.fps = float(fps)
        self.max_pixels_per_frame = max_pixels

        self.processor = AutoProcessor.from_pretrained(self.cfg.model_id)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.cfg.model_id, dtype=dtype, attn_implementation=attn_implementation
        ).to(self.device)
        self.model.eval().requires_grad_(False)

        vc = self.model.config.vision_config
        self.patch_size = int(vc.patch_size)
        self.spatial_merge_size = int(vc.spatial_merge_size)
        self.temporal_patch_size = int(vc.temporal_patch_size)
        self.hidden_size = int(self.model.config.text_config.hidden_size)
        self.deepstack_indexes = list(getattr(vc, "deepstack_visual_indexes", []) or [])
        self.n_deepstack = len(self.deepstack_indexes)
        # Qwen3-VL has no tokens_per_second; every plane block carries T=1 and
        # transformers passes time_interval=1. Kept only so callers that log
        # `time_interval` keep working.
        self.tokens_per_second = 1

        tok = self.processor.tokenizer
        self.video_token = getattr(self.processor, "video_token", "<|video_pad|>")
        self.vision_start_token = getattr(
            self.processor, "vision_start_token", "<|vision_start|>")
        self.vision_end_token = getattr(
            self.processor, "vision_end_token", "<|vision_end|>")
        self.video_pad_id = tok.convert_tokens_to_ids(self.video_token)
        self.vision_start_id = tok.convert_tokens_to_ids(self.vision_start_token)
        self.vision_end_id = tok.convert_tokens_to_ids(self.vision_end_token)
        assert None not in (self.video_pad_id, self.vision_start_id, self.vision_end_id)

        # Width of one buffered row: the token itself plus its deepstack maps.
        self.row_width = self.hidden_size * (1 + self.n_deepstack)

    # ------------------------------------------------------------------ #

    def describe(self) -> dict:
        return {
            "model_id": self.cfg.model_id,
            "family": "qwen3_vl",
            "hidden_size": self.hidden_size,
            "patch_size": self.patch_size,
            "spatial_merge_size": self.spatial_merge_size,
            "temporal_patch_size": self.temporal_patch_size,
            "deepstack_visual_indexes": self.deepstack_indexes,
            "row_width": self.row_width,
            "max_pixels_per_frame": self.max_pixels_per_frame,
            "dtype": str(self.dtype),
            "device": str(self.device),
        }

    def second_per_grid_t(self, fps: float) -> float:
        return self.temporal_patch_size / float(fps)

    def time_interval(self, fps: float) -> int:
        """Always 1 for Qwen3-VL: each plane is its own T=1 vision block."""
        return 1

    def plane_timestamp(self, plane_idx: int, fps: Optional[float] = None) -> float:
        """Seconds label for a plane, matching `Qwen3VLProcessor._calculate_timestamps`.

        Upstream averages the timestamps of the frames inside one temporal
        patch: `(t[i] + t[i + merge - 1]) / 2`. At `fps` with plane j covering
        frames `j*k .. j*k+k-1` that is `(j*k + (k-1)/2) / fps`.
        """
        f = float(fps or self.fps)
        k = self.temporal_patch_size
        return (plane_idx * k + (k - 1) / 2.0) / f

    # ------------------------------------------------------------------ #
    # Vision path
    # ------------------------------------------------------------------ #

    def _video_size_kwarg(self, num_frames: int) -> Dict[str, Any]:
        """Translate a per-frame pixel cap into Qwen3-VL's whole-chunk budget."""
        if self.max_pixels_per_frame is None:
            return {}
        t_bar = -(-num_frames // self.temporal_patch_size) * self.temporal_patch_size
        longest = int(self.max_pixels_per_frame) * t_bar
        vp = getattr(self.processor, "video_processor", None)
        shortest = 4096
        if vp is not None and getattr(vp, "size", None) is not None:
            got = vp.size.get("shortest_edge") if isinstance(vp.size, dict) \
                else getattr(vp.size, "shortest_edge", None)
            if got:
                shortest = min(int(got), longest)
        return {"size": {"shortest_edge": shortest, "longest_edge": longest}}

    @torch.no_grad()
    def preprocess_video(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run Qwen3-VL's own video preprocessing (incl. its `smart_resize`)."""
        assert frames.dim() == 4 and frames.shape[1] == 3, (
            f"expected [T, 3, H, W] uint8 RGB, got {tuple(frames.shape)}"
        )
        assert frames.shape[0] % self.temporal_patch_size == 0, (
            f"chunk of {frames.shape[0]} frames is not a multiple of "
            f"temporal_patch_size={self.temporal_patch_size}; a partial plane "
            f"gets edge-padded, silently changing the token count"
        )
        vp = self.processor.video_processor
        arr = frames.permute(0, 2, 3, 1).cpu().numpy()   # [T, H, W, 3] uint8
        # do_sample_frames DEFAULTS TO TRUE on Qwen3VLVideoProcessor, and with no
        # video_metadata it falls back to resampling at fps=24. We hand it a
        # chunk that has ALREADY been sampled at the stream's 1 FPS, so letting
        # it resample would silently drop or duplicate frames -- and the token
        # count would still look plausible. Turn it off explicitly.
        out = vp(videos=[arr], return_tensors="pt", do_sample_frames=False,
                 **self._video_size_kwarg(frames.shape[0]))
        return (
            out["pixel_values_videos"].to(self.device, self.dtype),
            out["video_grid_thw"].to(self.device),
        )

    @torch.no_grad()
    def encode_chunk(self, frames: torch.Tensor, *, fps: float = 1.0) -> VideoChunk:
        """Preprocess + vision tower on one chunk; returns tokens WITH deepstack.

        `VideoChunk.tokens` is `[N, hidden * (1 + n_deepstack)]`: the merged
        visual token followed by its three deepstack feature maps. They travel
        together through `TokenBuffer` so that row slicing prunes all four in
        lockstep -- there is no way for the pairing to drift, which is the one
        failure mode that would silently corrupt every surviving token.

        `VideoChunk.mrope[:, 0]` is the PLANE INDEX (not `t * time_interval` as
        in the Qwen2.5 wrapper). Qwen3-VL gives each plane its own vision block
        whose absolute start depends on the timestamp text before it, so the
        absolute temporal coordinate cannot be known until prefill; the plane
        index is what `build_position_ids` needs to reconstruct it.
        """
        pvv, grid_thw = self.preprocess_video(frames)
        tokens, deepstack = self._extract_video_features(pvv, grid_thw)

        g = grid_thw[0]
        m = self.spatial_merge_size
        n_planes, h_g, w_g = int(g[0]), int(g[1]) // m, int(g[2]) // m
        per_plane = h_g * w_g
        assert tokens.shape[0] == n_planes * per_plane, (
            f"{tokens.shape[0]} tokens for {n_planes} planes of {per_plane}"
        )

        pos_t = torch.arange(n_planes, device=self.device).repeat_interleave(per_plane)
        pos_h = torch.arange(h_g, device=self.device).repeat_interleave(w_g).repeat(n_planes)
        pos_w = torch.arange(w_g, device=self.device).repeat(h_g * n_planes)
        mrope = torch.stack([pos_t, pos_h, pos_w], dim=1).to(torch.long)

        rows = [tokens] + list(deepstack)
        wide = torch.cat(rows, dim=1) if self.n_deepstack else tokens
        assert wide.shape[1] == self.row_width, (
            f"row width {wide.shape[1]} != expected {self.row_width}"
        )
        return VideoChunk(
            tokens=wide,
            mrope=mrope,
            grid_thw=g,
            llm_grid_hw=(h_g, w_g),
            num_frames=frames.shape[0],
            time_interval=1,
        )

    @torch.no_grad()
    def _extract_video_features(
        self, pixel_values_videos: torch.Tensor, grid_thw: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Merged visual tokens `[N, hidden]` plus the deepstack maps.

        The merged tokens live on `.pooler_output` (a TUPLE split per video);
        `.last_hidden_state` is the PRE-merge sequence and is 4x longer with a
        different width -- pairing it with the deepstack maps would raise here
        rather than corrupt the run, which is how the mistake was caught.
        """
        out = self.model.model.get_video_features(pixel_values_videos, grid_thw)

        pooled = getattr(out, "pooler_output", None)
        if pooled is None and torch.is_tensor(out):
            pooled = out
        if pooled is None:
            raise RuntimeError(
                f"cannot locate merged visual features on {type(out).__name__}"
            )
        feats = (
            torch.cat([t.reshape(-1, self.hidden_size) for t in pooled], dim=0)
            if isinstance(pooled, (list, tuple)) else pooled.reshape(-1, self.hidden_size)
        )

        ds_raw = getattr(out, "deepstack_features", None) or []
        deepstack: List[torch.Tensor] = []
        for d in ds_raw:
            d = torch.cat([t.reshape(-1, self.hidden_size) for t in d], dim=0) \
                if isinstance(d, (list, tuple)) else d.reshape(-1, self.hidden_size)
            deepstack.append(d)

        expected = int((grid_thw.prod(-1) // (self.spatial_merge_size ** 2)).sum())
        assert feats.shape[0] == expected, (
            f"vision tower produced {feats.shape[0]} merged tokens, geometry "
            f"predicts {expected} for grid {grid_thw.tolist()}"
        )
        assert len(deepstack) == self.n_deepstack, (
            f"{len(deepstack)} deepstack maps, config declares {self.n_deepstack}"
        )
        for i, d in enumerate(deepstack):
            assert d.shape == feats.shape, (
                f"deepstack[{i}] is {tuple(d.shape)} but visual tokens are "
                f"{tuple(feats.shape)}; they must be 1:1 or the additive "
                f"injection lands on the wrong tokens"
            )
        return feats, deepstack

    def split_row(self, wide: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Inverse of the concat in `encode_chunk`."""
        assert wide.shape[1] == self.row_width, (
            f"row width {wide.shape[1]} != expected {self.row_width}"
        )
        h = self.hidden_size
        tokens = wide[:, :h]
        deepstack = [wide[:, h * (i + 1): h * (i + 2)] for i in range(self.n_deepstack)]
        return tokens, deepstack

    # ------------------------------------------------------------------ #
    # Prompt assembly
    # ------------------------------------------------------------------ #

    def build_prompt_ids(
        self,
        query: str,
        plane_counts: List[Tuple[int, int]],
        system: Optional[str] = None,
        fps: Optional[float] = None,
    ) -> torch.Tensor:
        """Chat-template ids with one timestamped vision block per SURVIVING plane.

        Args:
            plane_counts: `[(plane_idx, n_surviving_tokens), ...]`, ascending.
                          Planes pruned to zero tokens are not passed at all --
                          an empty `<|vision_start|><|vision_end|>` pair is not
                          a construction the model ever sees in training. Their
                          positional span is still skipped over in
                          `build_position_ids`, so survivors in later planes
                          keep their absolute coordinates.
        """
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system",
                             "content": [{"type": "text", "text": system}]})
        messages.append({
            "role": "user",
            "content": [{"type": "video"}, {"type": "text", "text": query}],
        })
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        blocks = "".join(
            f"<{self.plane_timestamp(j, fps):.1f} seconds>"
            f"{self.vision_start_token}{self.video_token * n}{self.vision_end_token}"
            for j, n in plane_counts if n > 0
        )
        triple = f"{self.vision_start_token}{self.video_token}{self.vision_end_token}"
        if triple in text:
            text = text.replace(triple, blocks, 1)
        elif self.video_token in text:
            text = text.replace(self.video_token, blocks, 1)
        else:
            raise RuntimeError(
                "chat template produced no video placeholder to expand"
            )

        ids = self.processor.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        want = sum(n for _, n in plane_counts if n > 0)
        got = int((ids[0] == self.video_pad_id).sum())
        assert got == want, f"placeholder expansion gave {got} pads, expected {want}"
        return ids

    @torch.no_grad()
    def build_position_ids(
        self,
        input_ids: torch.Tensor,
        mrope: torch.Tensor,
        dense_llm_grid_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, int]:
        """Interleaved 3D M-ROPE for a prefill of several pruned plane blocks.

        Walks the sequence exactly as `Qwen3VLModel.get_rope_index` does: text
        runs advance the running position by their length; a vision run takes
        `(t, h, w) = (pos, pos + h_i, pos + w_i)` and advances the position by
        `max(H, W)` of the DENSE merged grid -- never by how many tokens
        survived. So a pruned prefill puts every surviving token at exactly the
        coordinate it would have had at 100% keep.
        """
        assert input_ids.shape[0] == 1, "batch size 1 only (streaming)"
        ids = input_ids[0]
        L = ids.numel()
        dev = ids.device
        h_dense, w_dense = dense_llm_grid_hw
        advance = max(int(h_dense), int(w_dense))

        is_vid = (ids == self.video_pad_id)
        pos = torch.zeros(3, L, dtype=torch.long, device=dev)

        planes = mrope[:, 0].tolist()
        assert all(planes[i] <= planes[i + 1] for i in range(len(planes) - 1)), (
            "buffered tokens are not in ascending plane order; the vision "
            "blocks would be emitted out of temporal order"
        )

        cur = 0        # running M-ROPE position
        i = 0          # cursor into the token sequence
        taken = 0      # cursor into `mrope`
        while i < L:
            j = i
            vid = bool(is_vid[i])
            while j < L and bool(is_vid[j]) == vid:
                j += 1
            n = j - i
            if not vid:
                pos[:, i:j] = torch.arange(n, device=dev).view(1, -1).expand(3, -1) + cur
                cur += n
            else:
                blk = mrope[taken: taken + n]
                assert blk.shape[0] == n, (
                    f"vision block of {n} pads but only {blk.shape[0]} M-ROPE rows left"
                )
                assert int(blk[:, 0].min()) == int(blk[:, 0].max()), (
                    "one vision block spans more than one plane index"
                )
                pos[0, i:j] = cur
                pos[1, i:j] = blk[:, 1].to(dev) + cur
                pos[2, i:j] = blk[:, 2].to(dev) + cur
                taken += n
                cur += advance
            i = j
        assert taken == mrope.shape[0], (
            f"consumed {taken} of {mrope.shape[0]} M-ROPE rows"
        )

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
        fps: Optional[float] = None,
    ) -> GenerationResult:
        """Answer `query` from an externally pruned token set.

        Hand-rolled prefill + greedy decode: `generate()` cannot carry
        precomputed deepstack features (see the module docstring).
        """
        import time as _time
        from transformers import DynamicCache

        assert visual_tokens.dim() == 2, f"[K, W] expected, got {tuple(visual_tokens.shape)}"
        assert visual_tokens.shape[0] == mrope.shape[0], (
            f"visual_tokens ({visual_tokens.shape[0]}) and mrope "
            f"({mrope.shape[0]}) disagree -- token/coordinate pairing broke upstream"
        )
        tokens, deepstack = self.split_row(visual_tokens)
        k = tokens.shape[0]

        plane_counts: List[Tuple[int, int]] = []
        for p in mrope[:, 0].tolist():
            if plane_counts and plane_counts[-1][0] == p:
                plane_counts[-1] = (p, plane_counts[-1][1] + 1)
            else:
                plane_counts.append((int(p), 1))

        input_ids = self.build_prompt_ids(query, plane_counts, system=system, fps=fps)
        position_ids, rope_delta = self.build_position_ids(
            input_ids, mrope, dense_llm_grid_hw)

        embed = self.model.get_input_embeddings()
        inputs_embeds = embed(input_ids).clone()
        vid_mask = (input_ids[0] == self.video_pad_id)
        inputs_embeds[0, vid_mask] = tokens.to(inputs_embeds.dtype)
        visual_pos_masks = vid_mask.unsqueeze(0)
        ds = [d.to(inputs_embeds.dtype) for d in deepstack] or None

        L = int(input_ids.shape[1])
        attention_mask = torch.ones(1, L, dtype=torch.long, device=self.device)

        if profile_vram and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

        lm = self.model.model.language_model
        cache = DynamicCache()
        t0 = _time.perf_counter()
        out = lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=ds,
        )
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        logits = self.model.lm_head(hidden[:, -1:, :])
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t_first = _time.perf_counter()

        eos_ids = self._eos_ids()
        next_pos = int(position_ids.max().item()) + 1
        new_ids: List[int] = []
        for _ in range(max_new_tokens):
            if do_sample and temperature > 0:
                probs = torch.softmax(logits[0, -1].float() / temperature, dim=-1)
                nxt = int(torch.multinomial(probs, 1).item())
            else:
                nxt = int(logits[0, -1].argmax().item())
            new_ids.append(nxt)
            if nxt in eos_ids:
                break
            step_ids = torch.tensor([[nxt]], device=self.device)
            attention_mask = torch.cat(
                [attention_mask, torch.ones(1, 1, dtype=torch.long, device=self.device)],
                dim=1)
            step_pos = torch.full((3, 1, 1), next_pos, dtype=torch.long, device=self.device)
            next_pos += 1
            out = lm(
                inputs_embeds=embed(step_ids),
                attention_mask=attention_mask,
                position_ids=step_pos,
                past_key_values=cache,
                use_cache=True,
            )
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            logits = self.model.lm_head(hidden[:, -1:, :])

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t_end = _time.perf_counter()

        peak = (
            int(torch.cuda.max_memory_allocated(self.device))
            if profile_vram and self.device.type == "cuda" else 0
        )
        text = self.processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return GenerationResult(
            text=text,
            num_visual_tokens=k,
            num_prefill_tokens=L,
            prefill_ms=(t_first - t0) * 1e3,
            decode_ms=(t_end - t_first) * 1e3,
            peak_vram_bytes=peak,
            rope_delta=rope_delta,
            num_new_tokens=len(new_ids),
        )

    def _eos_ids(self) -> set:
        tok = self.processor.tokenizer
        ids = set()
        gc = getattr(self.model, "generation_config", None)
        e = getattr(gc, "eos_token_id", None) if gc is not None else None
        if isinstance(e, (list, tuple)):
            ids.update(int(x) for x in e)
        elif e is not None:
            ids.add(int(e))
        if tok.eos_token_id is not None:
            ids.add(int(tok.eos_token_id))
        for s in ("<|im_end|>", "<|endoftext|>"):
            i = tok.convert_tokens_to_ids(s)
            if i is not None and i >= 0:
                ids.add(int(i))
        return ids


__all__ = ["Qwen3VLWrapper", "Qwen3Geometry", "DEFAULT_QWEN3_MODEL"]
