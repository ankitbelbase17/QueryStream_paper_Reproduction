"""OmniPro Online-mode runner: QDP + RTAR replay, with RTAR driving generation.

This is the OmniPro counterpart to `querystream.eval_runner.StreamingBenchRunner`
and it differs in exactly one respect, which is the whole point of running this
benchmark: `response_mode` is not a choice here. StreamingBench asks a question
at a fixed timestamp and wants one answer, so there RTAR is computed and logged
but does not drive anything. OmniPro Online mode issues the instruction once at
t=0 and then scores *when* the model speaks, so here `T_t = R_t AND D_t` decides
every generation. Nothing else about the architecture changes: same OpenCLIP
encoder, same DSH, same score-space QDP, same thresholds, same JIT token buffer,
same backbone.

THE LOOP
    for each frame t in the stream (1 FPS):
        M_sem  = 1[ sim(q, v_t^i) > mean_j sim(q, v_t^j) ]        (query-dependent)
        M_temp = 1[ sim(v_t^i, v_dsh,t-1^i) < tau_temp ]          (cached per video)
        D_t    = 1[ keep_rate(M_QDP) > tau_den ]
        R_t    = 1[ calibrated sim(q, mean_pool v_t) > tau_rel ]
        if T_t fires and survives the event debouncer:
            flush the buffer through the accumulated masks and generate

Everything except `M_sem` and the relevance cosine is query-independent and
arrives pre-computed in the `VideoPass`, so a video shared by several samples is
decoded and encoded once (see querystream/eval_cache.py).

THE DEBOUNCER IS A SCORING-RELEVANT KNOB, SO IT IS LOGGED IN FULL
QueryStream's Appendix A.4 groups consecutive triggers into event intervals and
takes the first timestamp of each as the response point. Its gap parameter is
unspecified in the paper; the StreamingBench harness uses 1.0 s (one frame at
1 FPS). OmniPro matches responses with a +-3 s tolerance, so at a 1 s gap a
single real event can emit three responses of which at most one can match --
the other two are pure false positives. `event_gap_s` therefore defaults to the
matching tolerance for this benchmark, and every raw `T_t` timestamp is written
to the JSONL so any other gap can be re-scored offline for the two time-only
tasks without touching a GPU.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from querystream.buffer import TokenBuffer
from querystream.config import QueryStreamConfig
from querystream.eval_cache import VideoPass
from querystream.eval_runner import _sem_similarity
from querystream.open_clip_wrapper import OpenCLIPWrapper
from querystream.ops import (expand_mask_to_temporal_merge, l2_normalize,
                             resample_patch_mask, resample_scores_batch)
from querystream.profiling import Stopwatch
from querystream.qwen_wrapper import QwenVLWrapper
from querystream.rtar import RTARModule

from . import prompts
from .dataset import Sample


@dataclass
class SampleOutcome:
    """One JSONL row plus the counters the profiler and the driver need."""

    record: Dict[str, Any]
    stopwatch: Stopwatch
    n_emits: int = 0
    dense_tokens: int = 0
    kept_tokens: int = 0
    new_tokens: int = 0
    frames_streamed: int = 0


class OmniProRunner:
    """Streams one sample through QDP + RTAR and emits at every RTAR event."""

    def __init__(
        self,
        clip: OpenCLIPWrapper,
        qwen: Optional[QwenVLWrapper],
        cfg: QueryStreamConfig,
        *,
        gate_mode: str = "full",
        keep_mode: str = "qdp",
        max_emits: int = 64,
        emit_time_offset: float = 0.0,
        keep_frame_traces: bool = True,
    ):
        self.clip = clip
        self.qwen = qwen
        self.cfg = cfg
        self.gate_mode = gate_mode
        self.keep_mode = keep_mode
        self.max_emits = max_emits
        self.emit_time_offset = emit_time_offset
        self.keep_frame_traces = keep_frame_traces
        assert keep_mode in ("qdp", "dense"), keep_mode

    # ------------------------------------------------------------------ #
    # masks
    # ------------------------------------------------------------------ #

    def _plane_masks(self, sem, temp, clip_hw, llm_hw, n_planes: int, k: int
                     ) -> torch.Tensor:
        """QDP masks on the BACKBONE token grid, one row per 2-frame plane.

        Computed once for the whole video: plane j's mask depends only on
        plane j's own similarity maps -- the frame-adaptive threshold is a mean
        over that plane's tokens -- so the mask for a prefix of the stream is
        the corresponding prefix of this tensor, and an emission at t simply
        takes the first ceil(t) planes.

        `mask_space="score"` pools the continuous maps onto the backbone grid
        and thresholds there. Thresholding on CLIP's 16x16 grid and pooling the
        binary result is kept only for diagnostics: 8 CLIP patches back one
        backbone token, so "any" pooling lets a token survive if any of its 8
        contributors did, measured at 41% -> 96% keep on real video
        (CLAUDE.md 12.2).
        """
        n = n_planes * k
        qcfg = self.cfg.qdp

        if self.keep_mode == "dense":
            return torch.ones((n_planes, llm_hw[0] * llm_hw[1]),
                              dtype=torch.bool, device=sem.device)

        if self.cfg.mask_space == "score":
            pm = self.cfg.score_pool_mode
            sem_pl = resample_scores_batch(sem[:n], clip_hw, llm_hw, pm
                                           ).reshape(n_planes, k, -1).mean(dim=1)
            tmp_pl = resample_scores_batch(temp[:n], clip_hw, llm_hw, pm
                                           ).reshape(n_planes, k, -1).mean(dim=1)
            m_sem = sem_pl > sem_pl.mean(dim=-1, keepdim=True)      # STRICT >
            m_temp = tmp_pl < qcfg.tau_temp                          # STRICT <
            if qcfg.first_frame_all_novel and n_planes > 0:
                m_temp[0] = True
            return m_sem & m_temp

        m_clip = (sem[:n] > sem[:n].mean(dim=-1, keepdim=True)) & (temp[:n] < qcfg.tau_temp)
        if qcfg.first_frame_all_novel and n > 0:
            m_clip[0] = True
        per_frame = resample_patch_mask(m_clip, clip_hw, llm_hw, self.cfg.mask_resample_mode)
        return expand_mask_to_temporal_merge(per_frame, k, self.cfg.mask_resample_mode)

    # ------------------------------------------------------------------ #
    # one sample
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def run(self, vp: VideoPass, s: Sample) -> SampleOutcome:
        sw = Stopwatch(self.clip.device)
        t0 = time.perf_counter()
        rec: Dict[str, Any] = {
            "sample_id": s.sample_id,
            "task": s.task,
            "video_id": s.video_id,
            "audio_dependency": s.audio_dependency,
            "duration": s.duration,
            "question": s.question,
            "event": s.event,
            "query_text": s.query_text(),
            "ground_truth": s.ground_truth,
            "n_gt": len(s.ground_truth),
        }
        try:
            return self._run_inner(vp, s, rec, sw, t0)
        except Exception as exc:            # never abort a multi-hundred-sample run
            rec.update(emits=[], n_emits=0, error=f"{type(exc).__name__}: {exc}",
                       traceback=traceback.format_exc(limit=6),
                       wall_clock_sec=time.perf_counter() - t0)
            return SampleOutcome(rec, sw)

    def _run_inner(self, vp: VideoPass, s: Sample, rec, sw, t0) -> SampleOutcome:
        cfg, qcfg = self.cfg, self.cfg.qdp
        n_frames = vp.n_frames
        k = len(vp.planes[0].frame_ids) if vp.planes else 2
        n_planes_all = len(vp.planes)

        rec.update(n_frames_streamed=n_frames, n_planes=n_planes_all,
                   stream_seconds=round((n_frames - 1) / cfg.fps, 3) if n_frames else 0.0)

        if n_planes_all == 0:
            rec.update(emits=[], n_emits=0,
                       error="video yielded no complete 2-frame visual plane",
                       wall_clock_sec=time.perf_counter() - t0)
            return SampleOutcome(rec, sw)

        # ---- query-dependent signals -------------------------------- #
        with sw.stage("qdp_rtar", n=n_frames):
            q_emb = self.clip.encode_text(s.query_text())
            sem = _sem_similarity(vp.patch_feats[:n_frames], q_emb)      # [T, Nclip]
            temp = vp.temp_sim[:n_frames].to(torch.float32)
            rel_raw = (
                l2_normalize(vp.frame_pooled[:n_frames].to(torch.float32), dim=-1)
                @ l2_normalize(q_emb.reshape(-1).to(torch.float32), dim=-1)
            )                                                             # [T]

            # D_t gates on the CLIP-grid keep rate -- paper Sec. 3.3, "the token
            # keep rate from our QDP mechanism". The backbone-grid rate is what
            # the efficiency tables quote and is logged separately below.
            m_temp_c = temp < qcfg.tau_temp
            if qcfg.first_frame_all_novel:
                m_temp_c[0] = True
            clip_keep = ((sem > sem.mean(dim=-1, keepdim=True)) & m_temp_c
                         ).to(torch.float32).mean(dim=-1)                 # [T]

            plane_mask = self._plane_masks(sem, temp, vp.clip_grid_hw,
                                           vp.llm_grid_hw, n_planes_all, k)

        # ---- RTAR, frame by frame; generate on every fired event ---- #
        rtar = RTARModule(cfg.rtar, gate_mode=self.gate_mode)
        kr = clip_keep.tolist()
        rr = rel_raw.tolist()

        buf = TokenBuffer(cfg.buffer)
        next_plane = 0
        emits: List[Dict[str, Any]] = []
        trig_ts: List[float] = []          # raw T_t, BEFORE debouncing
        rel_cal: List[float] = []
        r_flags: List[bool] = []
        d_flags: List[bool] = []
        dense_tok = kept_tok = new_tok = 0
        capped = False

        for i in range(n_frames):
            t_sec = i / cfg.fps

            # Admit every plane that is complete as of this frame. Frame i is
            # visible at t_sec, and Qwen folds frames in pairs, so a trailing odd
            # frame cannot form a plane and simply waits for its partner.
            while next_plane < n_planes_all and (next_plane + 1) * k - 1 <= i:
                pl = vp.planes[next_plane]
                buf.append(frame_idx=pl.frame_ids[0], timestamp=pl.timestamp,
                           tokens=pl.tokens, mrope=pl.mrope,
                           mask=plane_mask[next_plane],
                           meta={"plane_idx": pl.plane_idx})
                next_plane += 1

            # No Stopwatch around this: RTAR consumes Python floats, so timing
            # it per frame would force two CUDA syncs per frame purely to
            # measure ~10 us of CPU work, serialising the whole loop. It is
            # covered by the vectorised `qdp_rtar` stage above.
            o = rtar(kr[i], relevance_raw=rr[i], timestamp=t_sec)
            rel_cal.append(o.relevance)
            r_flags.append(o.r_t)
            d_flags.append(o.d_t)
            if o.trigger:
                trig_ts.append(t_sec)
            if not o.fired or next_plane == 0:
                continue
            if len(emits) >= self.max_emits:
                capped = True
                continue

            flush = buf.flush(apply_mask=True, flush_frame_idx=i, flush_timestamp=t_sec)
            dense_tok += flush.num_tokens_before
            kept_tok += flush.num_tokens_after
            if flush.num_tokens_after == 0:
                # Nothing survived anywhere in the stream so far. Revive the
                # single most query-relevant backbone token rather than handing
                # the model an empty visual context, and flag it: a run with many
                # of these is over-pruned, not merely unlucky.
                rec["forced_min_keep"] = rec.get("forced_min_keep", 0) + 1
                pooled = resample_scores_batch(
                    sem[: next_plane * k], vp.clip_grid_hw, vp.llm_grid_hw,
                    cfg.score_pool_mode).reshape(next_plane, k, -1).mean(dim=1)
                pj, pmx = divmod(int(pooled.argmax().item()), pooled.shape[1])
                plane_mask[pj, pmx] = True
                buf.update_mask(vp.planes[pj].frame_ids[0], plane_mask[pj])
                flush = buf.flush(apply_mask=True, flush_frame_idx=i, flush_timestamp=t_sec)
                kept_tok += flush.num_tokens_after

            text = ""
            if self.qwen is not None:
                gen = self.qwen.generate_from_pruned(
                    query=prompts.build_prompt(s, t_sec),
                    system=prompts.system_prompt(),
                    visual_tokens=flush.tokens,
                    mrope=flush.mrope,
                    dense_llm_grid_hw=vp.llm_grid_hw,
                    max_new_tokens=prompts.max_new_tokens_for(s.task),
                )
                text = gen.text.strip()
                new_tok += gen.num_new_tokens
                sw.add("llm_prefill", gen.prefill_ms)
                sw.add("llm_decode", gen.decode_ms, n=max(gen.num_new_tokens, 1))

            emits.append({
                "t_sec": round(t_sec + self.emit_time_offset, 3),
                "text": text,
                "event_id": o.event_id,
                "frame_idx": i,
                "n_planes": next_plane,
                "tokens_dense": int(flush.num_tokens_before),
                "tokens_kept": int(flush.num_tokens_after),
                "keep_rate": round(flush.keep_rate, 6),
                "relevance": round(o.relevance, 6),
                "relevance_raw": round(o.relevance_raw, 6),
                "clip_keep_rate": round(o.keep_rate, 6),
            })

        # Backbone-grid keep rate over the WHOLE stream -- the efficiency number
        # comparable to the paper's Token Keep Rate column. Computed from the
        # plane masks directly, not from the flushes, because flushes overlap
        # (each is a prefix of the next) and averaging them would double-count
        # early planes.
        tok_dense_all = int(plane_mask.numel())
        tok_kept_all = int(plane_mask.sum().item())

        rec.update(
            emits=emits,
            n_emits=len(emits),
            emit_cap_hit=capped,
            max_emits=self.max_emits,
            n_raw_triggers=len(trig_ts),
            raw_trigger_ts=[round(t, 3) for t in trig_ts],
            event_gap_s=cfg.rtar.event_gap_s,
            tokens_dense_stream=tok_dense_all,
            tokens_kept_stream=tok_kept_all,
            token_keep_rate=round(tok_kept_all / tok_dense_all, 6) if tok_dense_all else 0.0,
            clip_keep_rate_mean=round(float(clip_keep.mean().item()), 6),
            relevance_raw_min=round(float(rel_raw.min().item()), 6),
            relevance_raw_max=round(float(rel_raw.max().item()), 6),
            relevance_raw_mean=round(float(rel_raw.mean().item()), 6),
            n_R_t=int(sum(r_flags)),
            n_D_t=int(sum(d_flags)),
            wall_clock_sec=round(time.perf_counter() - t0, 4),
        )
        if self.keep_frame_traces:
            # Per-second traces, rounded to 6 dp. ~200 floats x 4 lists per
            # sample: large enough to matter over a 1,000-sample run, small
            # enough that being able to re-derive every gate offline is worth it.
            rec["per_frame_keep_rates"] = [round(x, 6) for x in kr]
            rec["relevance_raw_scores"] = [round(x, 6) for x in rr]
            rec["relevance_normalized_scores"] = [round(x, 6) for x in rel_cal]
            rec["R_t"] = [int(x) for x in r_flags]
            rec["D_t"] = [int(x) for x in d_flags]

        return SampleOutcome(rec, sw, n_emits=len(emits), dense_tokens=dense_tok,
                             kept_tokens=kept_tok, new_tokens=new_tok,
                             frames_streamed=n_frames)
