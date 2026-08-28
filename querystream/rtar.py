"""Relevance-Triggered Active Response (RTAR) -- paper Sec. 3.3.

    R_t = 1[ sim(q, \\bar v_t) > tau_rel ]      \\bar v_t = mean-pooled frame feature
    D_t = 1[ (1/N) sum_i M_QDP(t, i) > tau_den ]
    T_t = R_t AND D_t

Both comparisons are STRICT (`>`), as printed.

Paper ablation this reproduces (Table 5, OVO-Bench Forward Active Responding):
    density-only  (= TimeChat-Online)  Acc 36.8  Score 29.5
    relevance-only                     Acc 40.3  Score 30.2
    full RTAR (R AND D)                Acc 40.2  Score 34.6
Set `gate_mode` to reproduce each row without forking the pipeline.

================================================================================
CALIBRATION WARNING -- read before burning GPU hours on tau_rel
================================================================================
`R_t` compares a raw CLIP image-text cosine similarity against tau_rel = 0.60.
Raw OpenCLIP ViT-L/14 image-text cosine similarity for a MATCHING pair sits
around 0.25-0.35; ~0.60 is essentially unreachable. Taken literally, R_t is
False for every frame of every video and the model never speaks.

The paper does not state any normalisation, so this is a genuine reproduction
gap, not a transcription error. `relevance_mode` exposes the candidate readings:

  "raw"            -- literal paper text. Expect ~0 triggers. Run it first
                      anyway: the trigger count is the diagnostic that tells you
                      calibration is required.
  "logit_softmax"  -- softmax over the stream's similarities scaled by CLIP's
                      learned logit_scale (100.0). Yields a [0, 1] relative
                      score; 0.60 then means "clearly above the field".
  "running_minmax" -- min-max normalise against a trailing window. Makes 0.60
                      mean "in the top 40% of the recent stream". Cheap,
                      stream-friendly, and the most likely intended semantics.
  "running_zscore" -- z-score against a trailing window, squashed by a sigmoid.

Log `relevance_raw` alongside `relevance` in every run so the distribution can
be inspected offline and the mode chosen from evidence, not from guessing.
================================================================================
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Literal, Optional

import torch
import torch.nn as nn

from .config import RTARConfig, RelevanceMode
from .ops import cosine_sim_to_vector, mean_pool_frame

GateMode = Literal["full", "relevance_only", "density_only"]


@dataclass
class RTAROutput:
    """Per-frame RTAR decision."""

    r_t: bool                   # relevance gate
    d_t: bool                   # density gate
    trigger: bool               # T_t = R_t AND D_t (before debouncing)
    fired: bool                 # trigger AND survived the A.4 event debouncer
    relevance: float            # calibrated sim(q, \bar v_t) actually thresholded
    relevance_raw: float        # uncalibrated cosine -- always log this
    keep_rate: float            # the density signal
    timestamp: float            # seconds into the stream
    frame_idx: int
    event_id: Optional[int] = None   # index of the event this frame opened
    calibrating: bool = False        # True while the calibration window warms up

    def summary(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "t": round(self.timestamp, 3),
            "R_t": self.r_t,
            "D_t": self.d_t,
            "T_t": self.trigger,
            "fired": self.fired,
            "relevance": round(self.relevance, 6),
            "relevance_raw": round(self.relevance_raw, 6),
            "keep_rate": round(self.keep_rate, 6),
            "event_id": self.event_id,
            "calibrating": self.calibrating,
        }


class RelevanceCalibrator:
    """Maps a raw cosine similarity onto the scale tau_rel is expressed in."""

    def __init__(self, mode: RelevanceMode, window: int, warmup: int, logit_scale: float):
        self.mode = mode
        self.window = window
        self.warmup = warmup
        self.logit_scale = logit_scale
        self._hist: Deque[float] = deque(maxlen=window)

    def reset(self) -> None:
        self._hist.clear()

    @property
    def ready(self) -> bool:
        return self.mode == "raw" or len(self._hist) >= self.warmup

    def __call__(self, raw: float) -> float:
        self._hist.append(float(raw))

        if self.mode == "raw":
            return float(raw)

        if self.mode == "logit_softmax":
            # Relative standing of this frame among the observed stream, at
            # CLIP's own temperature. Bounded in (0, 1).
            xs = torch.tensor(list(self._hist), dtype=torch.float32) * self.logit_scale
            probs = torch.softmax(xs, dim=0)
            return float(probs[-1].item() * len(xs))  # 1.0 == exactly average

        if not self.ready:
            return 0.0

        xs = torch.tensor(list(self._hist), dtype=torch.float64)
        if self.mode == "running_minmax":
            lo, hi = float(xs.min()), float(xs.max())
            if hi - lo < 1e-6:
                return 0.0
            # Clamp: `raw` is a float64 Python value while the window statistics
            # round-trip through a tensor, so raw == hi can yield 1.0 + 3e-8 and
            # push the "calibrated relevance is a [0, 1] score" contract.
            return min(1.0, max(0.0, (float(raw) - lo) / (hi - lo)))

        if self.mode == "running_zscore":
            mu, sd = float(xs.mean()), float(xs.std(correction=0))
            if sd < 1e-6:
                return 0.0
            return float(torch.sigmoid(torch.tensor((float(raw) - mu) / sd)).item())

        raise ValueError(f"unknown relevance_mode {self.mode!r}")


class EventDebouncer:
    """Appendix A.4 trigger grouping.

    "From the sequence of trigger timestamps generated by each model, we
     identify distinct event intervals. For each interval, we select the first
     timestamp as the definitive response point t'_m."

    A new event opens when the gap since the last trigger exceeds `event_gap_s`.
    Only the frame that opens an event fires.
    """

    def __init__(self, event_gap_s: float):
        self.event_gap_s = event_gap_s
        self._last_trigger_ts: Optional[float] = None
        self._event_count = 0

    def reset(self) -> None:
        self._last_trigger_ts = None
        self._event_count = 0

    @property
    def num_events(self) -> int:
        return self._event_count

    def __call__(self, trigger: bool, timestamp: float) -> tuple[bool, Optional[int]]:
        if not trigger:
            return False, None
        is_new_event = (
            self._last_trigger_ts is None
            or (timestamp - self._last_trigger_ts) > self.event_gap_s
        )
        self._last_trigger_ts = timestamp
        if is_new_event:
            eid = self._event_count
            self._event_count += 1
            return True, eid
        return False, self._event_count - 1


class RTARModule(nn.Module):
    """Stateful per-stream response scheduler. One instance per video (or `reset()`)."""

    def __init__(self, cfg: Optional[RTARConfig] = None, *, gate_mode: GateMode = "full"):
        super().__init__()
        self.cfg = cfg or RTARConfig()
        self.cfg.validate()
        self.gate_mode = gate_mode
        self.calibrator = RelevanceCalibrator(
            self.cfg.relevance_mode,
            self.cfg.calibration_window,
            self.cfg.calibration_warmup,
            self.cfg.logit_scale,
        )
        self.debouncer = EventDebouncer(self.cfg.event_gap_s)
        self._frame_idx = 0
        self.history: List[RTAROutput] = []

    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Clear stream state. MUST be called between videos."""
        self.calibrator.reset()
        self.debouncer.reset()
        self._frame_idx = 0
        self.history = []

    @property
    def frame_idx(self) -> int:
        return self._frame_idx

    @property
    def num_events(self) -> int:
        return self.debouncer.num_events

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def forward(
        self,
        keep_rate: float,
        *,
        frame_feat: Optional[torch.Tensor] = None,
        query_emb: Optional[torch.Tensor] = None,
        relevance_raw: Optional[float] = None,
        timestamp: Optional[float] = None,
        record: bool = True,
    ) -> RTAROutput:
        """Evaluate both gates for frame t.

        Supply EITHER (`frame_feat`, `query_emb`) -- in which case the holistic
        relevance is computed here -- OR a precomputed `relevance_raw`.

        Args:
            keep_rate:     (1/N) sum_i M_QDP(t, i), from QDPOutput.keep_rate.
            frame_feat:    [N, D] patch features (mean-pooled internally) or [D]
                           an already-pooled frame vector.
            query_emb:     [D] query embedding.
            relevance_raw: precomputed cosine, bypasses the two args above.
            timestamp:     seconds into the stream. Defaults to frame_idx / fps
                           assuming 1 FPS if omitted.
        """
        assert 0.0 <= keep_rate <= 1.0, f"keep_rate must be a rate, got {keep_rate}"

        if relevance_raw is None:
            assert frame_feat is not None and query_emb is not None, (
                "provide either relevance_raw, or both frame_feat and query_emb"
            )
            if frame_feat.dim() == 2:
                pooled = mean_pool_frame(frame_feat)          # \bar v_t
            elif frame_feat.dim() == 1:
                pooled = frame_feat.to(torch.float32)
            else:
                raise ValueError(
                    f"frame_feat must be [N, D] or [D], got {tuple(frame_feat.shape)}"
                )
            relevance_raw = float(
                cosine_sim_to_vector(pooled.unsqueeze(0), query_emb).item()
            )

        ts = float(timestamp) if timestamp is not None else float(self._frame_idx)
        relevance = self.calibrator(relevance_raw)
        calibrating = not self.calibrator.ready

        r_t = bool(relevance > self.cfg.tau_rel) and not calibrating   # STRICT >
        d_t = bool(keep_rate > self.cfg.tau_den)                        # STRICT >

        if self.gate_mode == "full":
            trigger = r_t and d_t
        elif self.gate_mode == "relevance_only":
            trigger = r_t
        elif self.gate_mode == "density_only":
            trigger = d_t
        else:  # pragma: no cover
            raise ValueError(f"unknown gate_mode {self.gate_mode!r}")

        if self.cfg.debounce_enabled:
            fired, event_id = self.debouncer(trigger, ts)
        else:
            fired, event_id = trigger, (self.debouncer.num_events if trigger else None)

        out = RTAROutput(
            r_t=r_t,
            d_t=d_t,
            trigger=trigger,
            fired=fired,
            relevance=float(relevance),
            relevance_raw=float(relevance_raw),
            keep_rate=float(keep_rate),
            timestamp=ts,
            frame_idx=self._frame_idx,
            event_id=event_id,
            calibrating=calibrating,
        )
        self._frame_idx += 1
        if record:
            self.history.append(out)
        return out

    # ------------------------------------------------------------------ #

    def diagnostics(self) -> dict:
        """Stream-level trigger statistics -- the first thing to check on a new run.

        `n_trigger == 0` with relevance_mode="raw" is the expected symptom of the
        tau_rel calibration gap documented at the top of this module.
        """
        if not self.history:
            return {"frames": 0}
        raws = torch.tensor([h.relevance_raw for h in self.history])
        cals = torch.tensor([h.relevance for h in self.history])
        keeps = torch.tensor([h.keep_rate for h in self.history])
        return {
            "frames": len(self.history),
            "n_R": sum(h.r_t for h in self.history),
            "n_D": sum(h.d_t for h in self.history),
            "n_trigger": sum(h.trigger for h in self.history),
            "n_fired": sum(h.fired for h in self.history),
            "n_events": self.num_events,
            "relevance_raw": {
                "min": float(raws.min()), "mean": float(raws.mean()),
                "max": float(raws.max()), "std": float(raws.std(unbiased=False)),
            },
            "relevance_cal": {
                "min": float(cals.min()), "mean": float(cals.mean()),
                "max": float(cals.max()),
            },
            "keep_rate": {
                "min": float(keeps.min()), "mean": float(keeps.mean()),
                "max": float(keeps.max()),
            },
            "gate_mode": self.gate_mode,
            "relevance_mode": self.cfg.relevance_mode,
            "tau_rel": self.cfg.tau_rel,
            "tau_den": self.cfg.tau_den,
        }


__all__ = ["RTARModule", "RTAROutput", "RelevanceCalibrator", "EventDebouncer", "GateMode"]
