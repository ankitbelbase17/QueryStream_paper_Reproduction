"""Just-in-time token buffer -- paper Sec. 3.1.

    "The original, unpruned visual tokens are temporarily held in a memory
     buffer. Upon receiving a trigger signal from RTAR, the accumulated pruning
     masks are applied to this buffer of tokens in a just-in-time manner."

So: tokens and masks accumulate SEPARATELY, and the mask is applied only at
flush time. This matters -- a mask computed at frame t can in principle be
revised before the flush, and keeping the tokens intact is what makes the
buffer replayable at different keep rates without re-encoding the video.

The critical invariant this module exists to enforce, from Sec. 3.2:

    "For each preserved patch, both its feature vector and its corresponding
     Multi-modal Rotary Position Embedding (M-ROPE) are retained. By excising
     the positional embeddings of discarded patches, we ensure the remaining
     tokens retain their original and correct {temporal, height, width}
     coordinates."

i.e. tokens and their (t, h, w) coordinates are sliced by the SAME index set, in
the SAME order. Coordinates are absolute and never renumbered after pruning --
renumbering would silently tell the LLM that the surviving patches are spatially
adjacent when they are not. `smoke_test.py` asserts this correspondence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from .config import BufferConfig


@dataclass
class BufferedFrame:
    """One frame's residency in the buffer."""

    frame_idx: int
    timestamp: float
    tokens: torch.Tensor        # [M, Dv] unpruned backbone visual tokens
    mrope: torch.Tensor         # [M, 3]  int64 (t, h, w)
    mask: torch.Tensor          # [M]     bool, the QDP decision for these tokens
    keep_rate: float
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        m = self.tokens.shape[0]
        assert self.tokens.dim() == 2, (
            f"tokens must be [M, Dv], got {tuple(self.tokens.shape)}"
        )
        assert self.mrope.shape == (m, 3), (
            f"mrope must be [M={m}, 3], got {tuple(self.mrope.shape)} -- M-ROPE "
            f"coordinates must be 1:1 with visual tokens"
        )
        assert self.mask.shape == (m,), (
            f"mask must be [M={m}], got {tuple(self.mask.shape)}"
        )
        assert self.mask.dtype == torch.bool, f"mask must be bool, got {self.mask.dtype}"
        assert self.mrope.dtype in (torch.long, torch.int32, torch.int64), (
            f"mrope must be integer, got {self.mrope.dtype}"
        )

    @property
    def num_tokens(self) -> int:
        return self.tokens.shape[0]

    @property
    def num_kept(self) -> int:
        return int(self.mask.sum().item())


@dataclass
class FlushResult:
    """The sparse token set handed to the backbone Video-LLM."""

    tokens: torch.Tensor            # [K, Dv]
    mrope: torch.Tensor             # [K, 3] int64, ABSOLUTE (t, h, w)
    frame_ids: torch.Tensor         # [K] int64, source frame index per token
    timestamps: torch.Tensor        # [K] float32, source timestamp per token
    num_frames: int
    num_tokens_before: int
    num_tokens_after: int
    per_frame_keep_rate: List[float]
    flush_frame_idx: int
    flush_timestamp: float

    @property
    def keep_rate(self) -> float:
        """Aggregate Token Keep Rate over everything flushed -- the paper's metric."""
        if self.num_tokens_before == 0:
            return 0.0
        return self.num_tokens_after / self.num_tokens_before

    def summary(self) -> dict:
        return {
            "flush_frame_idx": self.flush_frame_idx,
            "flush_t": round(self.flush_timestamp, 3),
            "num_frames": self.num_frames,
            "tokens_before": self.num_tokens_before,
            "tokens_after": self.num_tokens_after,
            "keep_rate": round(self.keep_rate, 6),
        }


class TokenBuffer:
    """Holds unpruned visual tokens + M-ROPE coords, prunes at flush time.

    Not an nn.Module: it holds no parameters and we never want it captured by
    `.to()` / `state_dict()` sweeps over the model.
    """

    def __init__(self, cfg: Optional[BufferConfig] = None):
        self.cfg = cfg or BufferConfig()
        self._frames: List[BufferedFrame] = []
        self._evicted_frames = 0
        self._evicted_tokens = 0

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def num_frames(self) -> int:
        return len(self._frames)

    @property
    def num_tokens(self) -> int:
        return sum(f.num_tokens for f in self._frames)

    @property
    def num_kept_tokens(self) -> int:
        return sum(f.num_kept for f in self._frames)

    @property
    def evicted(self) -> Tuple[int, int]:
        """(frames, tokens) dropped by the capacity policy. Non-zero => truncated
        context; log it, because it silently changes what the LLM can see."""
        return (self._evicted_frames, self._evicted_tokens)

    def memory_bytes(self) -> int:
        return sum(
            f.tokens.element_size() * f.tokens.nelement()
            + f.mrope.element_size() * f.mrope.nelement()
            + f.mask.nelement()
            for f in self._frames
        )

    def clear(self) -> None:
        """Drop everything. Call between videos AND after a flush that consumed
        the context (see `flush(consume=True)`)."""
        self._frames.clear()
        self._evicted_frames = 0
        self._evicted_tokens = 0

    reset = clear

    # ------------------------------------------------------------------ #
    # Ingest
    # ------------------------------------------------------------------ #

    def append(
        self,
        *,
        frame_idx: int,
        timestamp: float,
        tokens: torch.Tensor,
        mrope: torch.Tensor,
        mask: torch.Tensor,
        keep_rate: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> BufferedFrame:
        """Buffer one frame's UNPRUNED tokens together with its QDP mask.

        Args:
            tokens: [M, Dv] backbone visual tokens for this frame.
            mrope:  [M, 3]  integer (t, h, w) coordinates, 1:1 with `tokens`.
            mask:   [M]     bool M_QDP already resampled onto the backbone grid
                            (see ops.resample_patch_mask -- the CLIP grid and the
                            backbone grid differ).
        """
        dev = torch.device(self.cfg.device)
        tokens = tokens.to(dev, non_blocking=True)
        mrope = mrope.to(dev, non_blocking=True).to(torch.long)
        mask = mask.to(dev, non_blocking=True).to(torch.bool)

        if not self.cfg.store_unpruned:
            # Non-paper mode: prune eagerly. Saves memory, forfeits JIT replay.
            keep = mask.nonzero(as_tuple=True)[0]
            tokens, mrope, mask = tokens[keep], mrope[keep], torch.ones_like(mask[keep])

        frame = BufferedFrame(
            frame_idx=frame_idx,
            timestamp=float(timestamp),
            tokens=tokens,
            mrope=mrope,
            mask=mask,
            keep_rate=float(keep_rate) if keep_rate is not None
            else float(mask.to(torch.float32).mean().item()),
            meta=meta or {},
        )
        self._frames.append(frame)
        self._enforce_capacity()
        return frame

    def _enforce_capacity(self) -> None:
        """Oldest-first eviction under the configured caps (0 == unbounded)."""
        if self.cfg.max_frames > 0:
            while len(self._frames) > self.cfg.max_frames:
                old = self._frames.pop(0)
                self._evicted_frames += 1
                self._evicted_tokens += old.num_tokens
        if self.cfg.max_tokens > 0:
            while self._frames and self.num_tokens > self.cfg.max_tokens:
                old = self._frames.pop(0)
                self._evicted_frames += 1
                self._evicted_tokens += old.num_tokens

    def update_mask(self, frame_idx: int, mask: torch.Tensor) -> None:
        """Revise a buffered frame's mask before flush.

        This is the whole point of deferring pruning: masks stay mutable while
        the tokens sit intact. Used by ablations that re-threshold a stream
        without re-encoding it.
        """
        for f in self._frames:
            if f.frame_idx == frame_idx:
                assert mask.shape == (f.num_tokens,), (
                    f"mask {tuple(mask.shape)} != frame token count {f.num_tokens}"
                )
                f.mask = mask.to(f.tokens.device).to(torch.bool)
                f.keep_rate = float(f.mask.to(torch.float32).mean().item())
                return
        raise KeyError(f"frame_idx {frame_idx} is not in the buffer")

    # ------------------------------------------------------------------ #
    # Flush -- the just-in-time pruning step
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def flush(
        self,
        *,
        apply_mask: bool = True,
        consume: bool = False,
        flush_frame_idx: Optional[int] = None,
        flush_timestamp: Optional[float] = None,
        device: Optional[str | torch.device] = None,
    ) -> FlushResult:
        """Apply accumulated masks and emit the sparse token set.

        Tokens and M-ROPE coordinates are indexed by the SAME boolean mask, in
        buffer order (frame-major, then the frame's raster order), so the k-th
        output token's coordinate is the k-th output M-ROPE row. Coordinates are
        emitted ABSOLUTE and unrenumbered.

        Args:
            apply_mask: False returns the full unpruned context -- the 100%-keep
                        control row in every table.
            consume:    True clears the buffer afterwards. The paper implies the
                        context persists across responses within a stream, so
                        this defaults to False.
        """
        out_dev = torch.device(device) if device is not None else torch.device(self.cfg.device)

        if not self._frames:
            empty_t = torch.empty((0, 0), device=out_dev)
            return FlushResult(
                tokens=empty_t,
                mrope=torch.empty((0, 3), dtype=torch.long, device=out_dev),
                frame_ids=torch.empty((0,), dtype=torch.long, device=out_dev),
                timestamps=torch.empty((0,), dtype=torch.float32, device=out_dev),
                num_frames=0,
                num_tokens_before=0,
                num_tokens_after=0,
                per_frame_keep_rate=[],
                flush_frame_idx=flush_frame_idx if flush_frame_idx is not None else -1,
                flush_timestamp=flush_timestamp if flush_timestamp is not None else -1.0,
            )

        tok_chunks: List[torch.Tensor] = []
        rope_chunks: List[torch.Tensor] = []
        fid_chunks: List[torch.Tensor] = []
        ts_chunks: List[torch.Tensor] = []
        n_before = 0
        per_frame: List[float] = []

        for f in self._frames:
            n_before += f.num_tokens
            per_frame.append(f.keep_rate)

            if apply_mask:
                idx = f.mask.nonzero(as_tuple=True)[0]     # [K_f] -- ONE index set
            else:
                idx = torch.arange(f.num_tokens, device=f.tokens.device)

            if idx.numel() == 0:
                continue

            # ---- the invariant: identical index applied to BOTH tensors ---- #
            tok_chunks.append(f.tokens.index_select(0, idx))
            rope_chunks.append(f.mrope.index_select(0, idx))
            # ---------------------------------------------------------------- #

            fid_chunks.append(
                torch.full((idx.numel(),), f.frame_idx, dtype=torch.long, device=f.tokens.device)
            )
            ts_chunks.append(
                torch.full((idx.numel(),), f.timestamp, dtype=torch.float32, device=f.tokens.device)
            )

        if tok_chunks:
            tokens = torch.cat(tok_chunks, dim=0).to(out_dev)
            mrope = torch.cat(rope_chunks, dim=0).to(out_dev)
            frame_ids = torch.cat(fid_chunks, dim=0).to(out_dev)
            timestamps = torch.cat(ts_chunks, dim=0).to(out_dev)
        else:
            dv = self._frames[0].tokens.shape[1]
            tokens = torch.empty((0, dv), dtype=self._frames[0].tokens.dtype, device=out_dev)
            mrope = torch.empty((0, 3), dtype=torch.long, device=out_dev)
            frame_ids = torch.empty((0,), dtype=torch.long, device=out_dev)
            timestamps = torch.empty((0,), dtype=torch.float32, device=out_dev)

        assert tokens.shape[0] == mrope.shape[0] == frame_ids.shape[0] == timestamps.shape[0], (
            "token / M-ROPE / provenance length mismatch after flush -- the JIT "
            "slicing invariant is broken"
        )

        result = FlushResult(
            tokens=tokens,
            mrope=mrope,
            frame_ids=frame_ids,
            timestamps=timestamps,
            num_frames=len(self._frames),
            num_tokens_before=n_before,
            num_tokens_after=tokens.shape[0],
            per_frame_keep_rate=per_frame,
            flush_frame_idx=(
                flush_frame_idx if flush_frame_idx is not None else self._frames[-1].frame_idx
            ),
            flush_timestamp=(
                flush_timestamp if flush_timestamp is not None else self._frames[-1].timestamp
            ),
        )

        if consume:
            self.clear()
        return result

    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        ev_f, ev_t = self.evicted
        return {
            "frames": self.num_frames,
            "tokens_unpruned": self.num_tokens,
            "tokens_kept": self.num_kept_tokens,
            "keep_rate": (self.num_kept_tokens / self.num_tokens) if self.num_tokens else 0.0,
            "memory_mb": round(self.memory_bytes() / 1024**2, 3),
            "evicted_frames": ev_f,
            "evicted_tokens": ev_t,
        }


__all__ = ["TokenBuffer", "BufferedFrame", "FlushResult"]
