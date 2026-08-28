"""End-to-end QueryStream controller -- paper Sec. 3.1.

Per frame, at 1 FPS:
    1. OpenCLIP encodes the frame            -> patch features [256, 768]
    2. QDP                                   -> M_QDP on the 16x16 CLIP grid
    3. RTAR                                  -> R_t, D_t, T_t
    4. every `temporal_patch_size` frames, Qwen's vision tower encodes the plane;
       the frames' CLIP-grid masks are resampled to Qwen's merged grid, folded
       across the temporal merge, and buffered WITH the plane's M-ROPE coords
    5. on a fired trigger, the buffer applies the accumulated masks just-in-time
       and the sparse token set goes to Qwen for generation

Two geometry conversions the paper does not mention, both handled here:

  SPATIAL   OpenCLIP ViT-L/14 gives a 16x16 patch grid at 224px. Qwen2.5-VL's
            visual tokens sit on a grid coarsened by its 2x2 spatial merge, and
            Qwen's `smart_resize` picks the resolution, so the target grid is
            only known once a chunk has been preprocessed. Masks are therefore
            resampled at chunk-assembly time, not at QDP time.

  TEMPORAL  Qwen merges `temporal_patch_size` (=2) frames into one token plane,
            so two per-frame masks must be folded into one. Both conversions use
            `mask_resample_mode` (default "any", recall-favouring).

WHICH KEEP RATE FEEDS D_t?
Paper Sec. 3.3 defines the density gate on "the token keep rate from our QDP
mechanism", i.e. the mask QDP itself produced -- the CLIP-grid rate. That is
what gates D_t here. The post-resample backbone-grid rate is logged separately
as `tok_keep_rate` because it, not the QDP rate, is the number comparable to the
paper's reported Token Keep Rate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import torch

from .buffer import FlushResult, TokenBuffer
from .config import QueryStreamConfig
from .open_clip_wrapper import OpenCLIPWrapper
from .ops import expand_mask_to_temporal_merge, resample_patch_mask, resample_scores
from .qdp import QDPModule, QDPOutput
from .qwen_wrapper import GenerationResult, QwenVLWrapper
from .rtar import RTARModule, RTAROutput


@dataclass
class FrameRecord:
    """Everything observable about one streamed frame. One .jsonl row at M3."""

    frame_idx: int
    timestamp: float
    qdp_keep_rate: float          # on the CLIP 16x16 grid -- gates D_t
    qdp_keep_rate_sem: float
    qdp_keep_rate_temp: float
    sem_threshold: float
    tok_keep_rate: Optional[float]  # after resampling to the backbone grid
    relevance: float
    relevance_raw: float
    r_t: bool
    d_t: bool
    trigger: bool
    fired: bool
    event_id: Optional[int]
    calibrating: bool
    clip_ms: float
    qdp_ms: float
    vision_ms: float
    buffer_tokens: int
    buffer_mb: float

    def to_json(self) -> dict:
        d = dict(self.__dict__)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 6)
        return d


@dataclass
class StreamResult:
    """Outcome of streaming one (video, query) pair."""

    query: str
    frames: List[FrameRecord] = field(default_factory=list)
    responses: List[Dict[str, Any]] = field(default_factory=list)
    total_ms: float = 0.0
    n_frames: int = 0
    n_chunks: int = 0
    dropped_trailing_frames: int = 0

    @property
    def mean_qdp_keep_rate(self) -> float:
        v = [f.qdp_keep_rate for f in self.frames]
        return sum(v) / len(v) if v else 0.0

    @property
    def mean_tok_keep_rate(self) -> float:
        v = [f.tok_keep_rate for f in self.frames if f.tok_keep_rate is not None]
        return sum(v) / len(v) if v else 0.0

    @property
    def n_triggers(self) -> int:
        return sum(f.trigger for f in self.frames)

    @property
    def n_fired(self) -> int:
        return sum(f.fired for f in self.frames)

    def summary(self) -> dict:
        return {
            "query": self.query,
            "n_frames": self.n_frames,
            "n_chunks": self.n_chunks,
            "mean_qdp_keep_rate": round(self.mean_qdp_keep_rate, 6),
            "mean_tok_keep_rate": round(self.mean_tok_keep_rate, 6),
            "n_triggers": self.n_triggers,
            "n_fired": self.n_fired,
            "n_responses": len(self.responses),
            "total_ms": round(self.total_ms, 2),
            "ms_per_frame": round(self.total_ms / max(self.n_frames, 1), 2),
            "dropped_trailing_frames": self.dropped_trailing_frames,
        }


class JITStreamingPipeline:
    """The QueryStream gateway wired to a real backbone."""

    def __init__(
        self,
        clip: OpenCLIPWrapper,
        qwen: Optional[QwenVLWrapper],
        cfg: Optional[QueryStreamConfig] = None,
        *,
        gate_mode: str = "full",
    ):
        self.clip = clip
        self.qwen = qwen
        self.cfg = (cfg or QueryStreamConfig()).validate()
        self.qdp = QDPModule(self.cfg.qdp)
        self.rtar = RTARModule(self.cfg.rtar, gate_mode=gate_mode)
        self.buffer = TokenBuffer(self.cfg.buffer)
        self.chunk_frames = qwen.temporal_patch_size if qwen is not None else 2
        self.reset()

    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Clear ALL per-stream state. Must run between videos."""
        self.qdp.reset()
        self.rtar.reset()
        self.buffer.clear()
        self._pending_frames: List[torch.Tensor] = []
        self._pending_masks: List[torch.Tensor] = []
        self._pending_sem: List[torch.Tensor] = []
        self._pending_temp: List[torch.Tensor] = []
        self._pending_first: List[bool] = []
        self._pending_idx: List[int] = []
        self._pending_ts: List[float] = []
        self._plane_idx = 0
        self._dense_llm_grid_hw: Optional[Tuple[int, int]] = None
        self._dense_tokens_seen = 0

    @property
    def dense_llm_grid_hw(self) -> Optional[Tuple[int, int]]:
        """Backbone merged grid, known only after the first chunk is preprocessed."""
        return self._dense_llm_grid_hw

    # ------------------------------------------------------------------ #
    # Chunk assembly
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _flush_pending_chunk(self) -> float:
        """Encode the pending frames into one backbone token plane and buffer it.

        Returns the vision-tower wall-clock in ms (0.0 if nothing was pending).
        """
        if self.qwen is None or len(self._pending_frames) < self.chunk_frames:
            return 0.0

        t0 = time.perf_counter()
        frames = torch.stack(self._pending_frames[: self.chunk_frames], dim=0)
        chunk = self.qwen.encode_chunk(frames, fps=self.cfg.fps)
        if self.qwen.device.type == "cuda":
            torch.cuda.synchronize()
        vision_ms = (time.perf_counter() - t0) * 1e3

        llm_hw = chunk.llm_grid_hw
        if self._dense_llm_grid_hw is None:
            self._dense_llm_grid_hw = llm_hw
        elif self._dense_llm_grid_hw != llm_hw:
            raise RuntimeError(
                f"backbone token grid changed mid-stream: {self._dense_llm_grid_hw} "
                f"-> {llm_hw}. Frame resolution must be constant, or M-ROPE "
                f"coordinates from different chunks are not comparable."
            )

        src_hw = self.clip.grid_hw
        n = self.chunk_frames

        if self.cfg.mask_space == "score":
            # Pool the CONTINUOUS similarities onto the backbone grid, average
            # across the temporally merged frames, then threshold once on the
            # grid that is actually pruned. Keeps the frame-adaptive mean
            # threshold meaningful; see config.MaskSpace.
            pm = self.cfg.score_pool_mode
            sem = torch.stack([resample_scores(s, src_hw, llm_hw, pm)
                               for s in self._pending_sem[:n]], 0).mean(0)
            tmp = torch.stack([resample_scores(s, src_hw, llm_hw, pm)
                               for s in self._pending_temp[:n]], 0).mean(0)
            _, _, plane_mask, _ = self.qdp.masks_from_pooled_scores(
                sem, tmp, is_first_frame=self._pending_first[0])
        else:
            # Pool the BINARY masks. With "any" this inflates the keep rate.
            per_frame = torch.stack(
                [
                    resample_patch_mask(m, src_hw, llm_hw, self.cfg.mask_resample_mode)
                    for m in self._pending_masks[:n]
                ],
                dim=0,
            )                                             # [chunk_frames, H*W]
            plane_mask = expand_mask_to_temporal_merge(
                per_frame, n, self.cfg.mask_resample_mode
            ).reshape(-1)                                 # [N] for this plane

        assert plane_mask.numel() == chunk.num_tokens, (
            f"plane mask has {plane_mask.numel()} entries but the vision tower "
            f"produced {chunk.num_tokens} tokens for grid {chunk.grid_thw.tolist()}"
        )

        # M-ROPE coords are RELATIVE; shift this plane onto the stream timeline.
        mrope = chunk.mrope.clone()
        mrope[:, 0] += self._plane_idx * chunk.time_interval

        self.buffer.append(
            frame_idx=self._pending_idx[0],
            timestamp=self._pending_ts[0],
            tokens=chunk.tokens,
            mrope=mrope,
            mask=plane_mask,
            meta={"plane_idx": self._plane_idx, "frames": list(self._pending_idx)},
        )
        self._dense_tokens_seen += chunk.num_tokens
        self._plane_idx += 1

        del self._pending_frames[:n]
        del self._pending_masks[:n]
        del self._pending_sem[:n]
        del self._pending_temp[:n]
        del self._pending_first[:n]
        del self._pending_idx[:n]
        del self._pending_ts[:n]
        return vision_ms

    # ------------------------------------------------------------------ #
    # Streaming step
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def step(
        self,
        frame: torch.Tensor,
        query_emb: torch.Tensor,
        *,
        frame_idx: int,
        timestamp: float,
    ) -> Tuple[FrameRecord, Optional[FlushResult]]:
        """Process one frame. Returns its record and a FlushResult if it fired.

        Args:
            frame: [3, H, W] uint8 RGB.
        """
        t0 = time.perf_counter()
        ef = self.clip.encode_frame(frame)
        if self.clip.device.type == "cuda":
            torch.cuda.synchronize()
        clip_ms = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        out: QDPOutput = self.qdp(ef.patch_feats, query_emb)
        qdp_ms = (time.perf_counter() - t0) * 1e3

        # D_t uses the QDP mask's own keep rate (paper Sec. 3.3).
        dec: RTAROutput = self.rtar(
            keep_rate=out.keep_rate,
            frame_feat=ef.patch_feats,
            query_emb=query_emb,
            timestamp=timestamp,
        )

        self._pending_frames.append(frame)
        self._pending_masks.append(out.m_qdp)
        self._pending_sem.append(out.sem_sim)
        self._pending_temp.append(out.temp_sim)
        self._pending_first.append(out.is_first_frame)
        self._pending_idx.append(frame_idx)
        self._pending_ts.append(timestamp)

        vision_ms = 0.0
        tok_keep = None
        if len(self._pending_frames) >= self.chunk_frames:
            vision_ms = self._flush_pending_chunk()
            if self.buffer.num_frames:
                tok_keep = self.buffer._frames[-1].keep_rate

        rec = FrameRecord(
            frame_idx=frame_idx,
            timestamp=timestamp,
            qdp_keep_rate=out.keep_rate,
            qdp_keep_rate_sem=out.keep_rate_sem,
            qdp_keep_rate_temp=out.keep_rate_temp,
            sem_threshold=out.sem_threshold,
            tok_keep_rate=tok_keep,
            relevance=dec.relevance,
            relevance_raw=dec.relevance_raw,
            r_t=dec.r_t,
            d_t=dec.d_t,
            trigger=dec.trigger,
            fired=dec.fired,
            event_id=dec.event_id,
            calibrating=dec.calibrating,
            clip_ms=clip_ms,
            qdp_ms=qdp_ms,
            vision_ms=vision_ms,
            buffer_tokens=self.buffer.num_tokens,
            buffer_mb=self.buffer.memory_bytes() / 1024**2,
        )

        flush = None
        if dec.fired and self.buffer.num_frames > 0:
            flush = self.buffer.flush(
                apply_mask=True, flush_frame_idx=frame_idx, flush_timestamp=timestamp
            )
        return rec, flush

    # ------------------------------------------------------------------ #
    # Whole-stream driver
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def run(
        self,
        frames: Iterable[torch.Tensor],
        query: str,
        *,
        system: Optional[str] = None,
        max_new_tokens: int = 64,
        generate: bool = True,
        answer_at_end: bool = True,
        on_frame: Optional[Callable[[FrameRecord], None]] = None,
    ) -> StreamResult:
        """Stream a whole video and answer `query`.

        Args:
            frames:        iterable of [3, H, W] uint8 RGB, already sampled at
                           `cfg.fps` (the sampler owns frame selection).
            answer_at_end: force one final answer even if RTAR never fired. Needed
                           for StreamingBench-style MCQ, where a response is
                           always required; proactive benchmarks set it False.
        """
        self.reset()
        res = StreamResult(query=query)
        q = self.clip.encode_text(query)
        t_start = time.perf_counter()

        for i, frame in enumerate(frames):
            ts = i / self.cfg.fps
            rec, flush = self.step(frame, q, frame_idx=i, timestamp=ts)
            res.frames.append(rec)
            if on_frame is not None:
                on_frame(rec)
            if flush is not None and generate and self.qwen is not None:
                res.responses.append(self._answer(flush, query, system, max_new_tokens, ts))

        # Trailing frames that never completed a temporal plane are NOT encoded:
        # Qwen edge-pads a partial plane, which would change the token count and
        # make the keep-rate denominator inconsistent with every other chunk.
        # Recorded so the loss is visible rather than silent.
        res.dropped_trailing_frames = len(self._pending_frames)

        if answer_at_end and generate and self.qwen is not None and self.buffer.num_frames:
            already = res.responses and res.responses[-1].get("timestamp") == (
                (len(res.frames) - 1) / self.cfg.fps
            )
            if not already:
                flush = self.buffer.flush(apply_mask=True)
                res.responses.append(
                    self._answer(flush, query, system, max_new_tokens,
                                 (len(res.frames) - 1) / self.cfg.fps, final=True)
                )

        res.total_ms = (time.perf_counter() - t_start) * 1e3
        res.n_frames = len(res.frames)
        res.n_chunks = self._plane_idx
        return res

    def _answer(
        self,
        flush: FlushResult,
        query: str,
        system: Optional[str],
        max_new_tokens: int,
        timestamp: float,
        final: bool = False,
    ) -> Dict[str, Any]:
        assert self.qwen is not None and self._dense_llm_grid_hw is not None
        gen: GenerationResult = self.qwen.generate_from_pruned(
            query=query,
            visual_tokens=flush.tokens,
            mrope=flush.mrope,
            dense_llm_grid_hw=self._dense_llm_grid_hw,
            system=system,
            max_new_tokens=max_new_tokens,
        )
        return {
            "timestamp": timestamp,
            "final": final,
            "text": gen.text,
            "visual_tokens": gen.num_visual_tokens,
            "tokens_before": flush.num_tokens_before,
            "keep_rate": round(flush.keep_rate, 6),
            "prefill_tokens": gen.num_prefill_tokens,
            "decode_ms": round(gen.decode_ms, 2),
            "peak_vram_mb": round(gen.peak_vram_bytes / 1024**2, 1),
            "rope_delta": gen.rope_delta,
        }


__all__ = ["JITStreamingPipeline", "FrameRecord", "StreamResult"]
