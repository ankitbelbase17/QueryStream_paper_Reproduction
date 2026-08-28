"""Per-question replay of QDP + RTAR over a cached video pass.

This is the evaluation-time counterpart to `JITStreamingPipeline`. The pipeline
streams one query through one video frame by frame; the runner replays many
queries over the *same* pre-computed visual state (see `eval_cache`). Both apply
identical arithmetic -- same thresholds, same strictness, same score-space
resampling, same JIT flush -- and `tests/test_eval_equivalence.py` asserts they
agree token-for-token on a real clip.

RESPONSE POLICY ON STREAMINGBENCH
The paper evaluates RTAR on OVO-Bench's Forward Active Responding split, where
*when* to speak is the thing being measured. StreamingBench asks a question at a
fixed timestamp and always requires exactly one answer, so triggering is not
what is being scored there: Table 1 is a pruning result. The runner therefore
answers once, at the question's timestamp, from everything visible up to it
(`response_mode="query_ts"`). R_t / D_t / T_t are still evaluated and logged on
every frame, so the trigger behaviour is on record and OVO-Bench can consume it
later; `response_mode="rtar"` makes them drive generation instead.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from .benchmarks import streamingbench as sb
from .buffer import TokenBuffer
from .config import QueryStreamConfig
from .eval_cache import VideoPass
from .open_clip_wrapper import OpenCLIPWrapper
from .ops import expand_mask_to_temporal_merge, l2_normalize, resample_patch_mask, resample_scores_batch
from .profiling import Stopwatch
from .qwen_wrapper import QwenVLWrapper
from .rtar import RTARModule


def _sem_similarity(feats: torch.Tensor, q: torch.Tensor, chunk: int = 128) -> torch.Tensor:
    """cos(q, v_t^i) for every frame and patch. [T, N, D] x [D] -> [T, N] fp32.

    Chunked over frames: a 1440-frame video promoted to fp32 in one go is 1.1 GB
    of transient allocation for no benefit. Re-normalising in fp32 (rather than
    trusting the fp16 unit norm from the encoder) keeps this bit-identical to
    `ops.cosine_sim_to_vector`, which is what the streaming pipeline calls.
    """
    qn = l2_normalize(q.reshape(-1).to(torch.float32), dim=-1)
    out = torch.empty(feats.shape[:2], dtype=torch.float32, device=feats.device)
    for i in range(0, feats.shape[0], chunk):
        blk = l2_normalize(feats[i : i + chunk].to(torch.float32), dim=-1)
        out[i : i + chunk] = blk @ qn
    return out


@dataclass
class AnswerOutcome:
    """One JSONL row plus the profiling fields the CSV needs."""

    record: Dict[str, Any]
    stopwatch: Stopwatch
    dense_tokens: int
    kept_tokens: int
    new_tokens: int


class StreamingBenchRunner:
    """Answers StreamingBench questions from a `VideoPass`."""

    def __init__(
        self,
        clip: OpenCLIPWrapper,
        qwen: Optional[QwenVLWrapper],
        cfg: QueryStreamConfig,
        *,
        gate_mode: str = "full",
        max_new_tokens: int = 16,
        response_mode: str = "query_ts",
        keep_mode: str = "qdp",
        judge=None,
        keep_raw_response: bool = True,
    ):
        self.clip = clip
        self.qwen = qwen
        self.cfg = cfg
        self.gate_mode = gate_mode
        self.max_new_tokens = max_new_tokens
        self.response_mode = response_mode
        self.keep_mode = keep_mode
        self.judge = judge
        self.keep_raw_response = keep_raw_response
        assert response_mode in ("query_ts", "rtar"), response_mode
        assert keep_mode in ("qdp", "dense"), keep_mode

    # ------------------------------------------------------------------ #
    # Mask construction
    # ------------------------------------------------------------------ #

    def _plane_masks(
        self,
        sem: torch.Tensor,          # [T, Nclip] fp32
        temp: torch.Tensor,         # [T, Nclip] fp32
        clip_hw: Tuple[int, int],
        llm_hw: Tuple[int, int],
        n_planes: int,
        k: int,
    ) -> torch.Tensor:
        """QDP masks on the backbone token grid, one row per plane. [P, M] bool.

        `mask_space="score"` pools the continuous similarity maps onto the
        backbone grid and thresholds THERE, so the frame-adaptive mean threshold
        keeps its meaning on the grid actually being pruned. `mask_space="mask"`
        thresholds on the CLIP grid first and pools the binary result, which is
        kept only for diagnostics: with "any" pooling a backbone token survives
        if any of its 8 contributing CLIP patches did, which measured 41% -> 96%
        on real video (Milestone 2, CLAUDE.md 12.2).
        """
        n = n_planes * k
        qcfg = self.cfg.qdp

        if self.keep_mode == "dense":
            # 100%-keep control row: pruning off, everything else identical.
            return torch.ones((n_planes, llm_hw[0] * llm_hw[1]),
                              dtype=torch.bool, device=sem.device)

        if self.cfg.mask_space == "score":
            pm = self.cfg.score_pool_mode
            sem_p = resample_scores_batch(sem[:n], clip_hw, llm_hw, pm)     # [n, M]
            tmp_p = resample_scores_batch(temp[:n], clip_hw, llm_hw, pm)
            sem_pl = sem_p.reshape(n_planes, k, -1).mean(dim=1)             # [P, M]
            tmp_pl = tmp_p.reshape(n_planes, k, -1).mean(dim=1)
            m_sem = sem_pl > sem_pl.mean(dim=-1, keepdim=True)              # STRICT >
            m_temp = tmp_pl < qcfg.tau_temp                                 # STRICT <
            if qcfg.first_frame_all_novel and n_planes > 0:
                m_temp[0] = True          # plane 0 contains frame 0: no history yet
            return m_sem & m_temp

        m_sem = sem[:n] > sem[:n].mean(dim=-1, keepdim=True)
        m_temp = temp[:n] < qcfg.tau_temp
        if qcfg.first_frame_all_novel and n > 0:
            m_temp[0] = True
        m_clip = m_sem & m_temp
        per_frame = resample_patch_mask(m_clip, clip_hw, llm_hw, self.cfg.mask_resample_mode)
        return expand_mask_to_temporal_merge(per_frame, k, self.cfg.mask_resample_mode)

    # ------------------------------------------------------------------ #
    # One question
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def answer(self, vp: VideoPass, q: sb.Question) -> AnswerOutcome:
        sw = Stopwatch(self.clip.device)
        t_q0 = time.perf_counter()
        rec: Dict[str, Any] = {
            "video_id": q.video_id,
            "question_id": q.question_id,
            "task_type": q.task_type,
            "task_code": q.task_code,
            "timestamp_s": q.timestamp_s,
            "query_text": q.query_text(),
            "question": q.question,
            "options": q.options,
            "ground_truth": q.answer,
        }
        try:
            return self._answer_inner(vp, q, rec, sw, t_q0)
        except Exception as exc:                       # never abort a 2500-item run
            rec.update(
                predicted_response="",
                predicted_letter=None,
                is_correct=False,
                parse_failed=True,
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(limit=6),
                wall_clock_sec=time.perf_counter() - t_q0,
            )
            return AnswerOutcome(rec, sw, 0, 0, 0)

    def _answer_inner(self, vp, q, rec, sw, t_q0) -> AnswerOutcome:
        cfg, qcfg = self.cfg, self.cfg.qdp
        planes, n_vis, dropped = vp.planes_up_to(q.timestamp_s)
        k = len(planes[0].frame_ids) if planes else 2
        n_planes = len(planes)
        n = n_planes * k

        rec.update(n_frames_visible=n_vis, n_planes=n_planes,
                   dropped_trailing_frames=dropped)

        if n_planes == 0:
            # Question stamped before a full 2-frame plane exists. Cannot answer
            # from video; recorded rather than silently scored against.
            rec.update(predicted_response="", predicted_letter=None, is_correct=False,
                       parse_failed=True, error="no complete visual plane before timestamp",
                       wall_clock_sec=time.perf_counter() - t_q0)
            return AnswerOutcome(rec, sw, 0, 0, 0)

        # ---- query-dependent signals ---------------------------------- #
        with sw.stage("qdp_rtar", n=n_vis):
            q_emb = self.clip.encode_text(q.query_text())
            sem = _sem_similarity(vp.patch_feats[:n_vis], q_emb)          # [n_vis, Nclip]
            temp = vp.temp_sim[:n_vis].to(torch.float32)
            rel_raw = (
                l2_normalize(vp.frame_pooled[:n_vis].to(torch.float32), dim=-1)
                @ l2_normalize(q_emb.reshape(-1).to(torch.float32), dim=-1)
            )                                                              # [n_vis]

            # CLIP-grid QDP keep rate -- the density signal D_t gates on
            # (paper Sec. 3.3: "the token keep rate from our QDP mechanism").
            m_sem_c = sem > sem.mean(dim=-1, keepdim=True)
            m_temp_c = temp < qcfg.tau_temp
            if qcfg.first_frame_all_novel:
                m_temp_c[0] = True
            clip_keep = (m_sem_c & m_temp_c).to(torch.float32).mean(dim=-1)   # [n_vis]

            # ---- RTAR, frame by frame (the calibrator is sequential) ---- #
            rtar = RTARModule(cfg.rtar, gate_mode=self.gate_mode)
            kr = clip_keep.tolist()
            rr = rel_raw.tolist()
            rel_cal: List[float] = []
            r_flags: List[bool] = []
            d_flags: List[bool] = []
            t_flags: List[bool] = []
            fire_ts: List[float] = []
            for i in range(n_vis):
                o = rtar(kr[i], relevance_raw=rr[i], timestamp=i / cfg.fps)
                rel_cal.append(o.relevance)
                r_flags.append(o.r_t)
                d_flags.append(o.d_t)
                t_flags.append(o.trigger)
                if o.fired:
                    fire_ts.append(o.timestamp)

            # ---- plane masks on the backbone grid ---------------------- #
            plane_mask = self._plane_masks(
                sem, temp, vp.clip_grid_hw, vp.llm_grid_hw, n_planes, k
            )                                                              # [P, M]

            # ---- JIT buffer ------------------------------------------- #
            buf = TokenBuffer(cfg.buffer)
            for j, pl in enumerate(planes):
                buf.append(
                    frame_idx=pl.frame_ids[0], timestamp=pl.timestamp,
                    tokens=pl.tokens, mrope=pl.mrope, mask=plane_mask[j],
                    meta={"plane_idx": pl.plane_idx},
                )
            flush = buf.flush(apply_mask=True, flush_frame_idx=n_vis - 1,
                              flush_timestamp=q.timestamp_s)

        dense = flush.num_tokens_before
        kept = flush.num_tokens_after
        forced = False
        if kept == 0:
            # Degenerate: nothing survived anywhere in the stream. Revive the
            # single most query-relevant token so the backbone still receives a
            # visual context, and flag it -- a run with many of these is
            # over-pruned, not merely unlucky.
            forced = True
            pooled = resample_scores_batch(
                sem[:n], vp.clip_grid_hw, vp.llm_grid_hw, cfg.score_pool_mode
            ).reshape(n_planes, k, -1).mean(dim=1)
            best = int(pooled.argmax().item())
            pj, pm = divmod(best, pooled.shape[1])
            plane_mask[pj, pm] = True
            buf = TokenBuffer(cfg.buffer)
            for j, pl in enumerate(planes):
                buf.append(frame_idx=pl.frame_ids[0], timestamp=pl.timestamp,
                           tokens=pl.tokens, mrope=pl.mrope, mask=plane_mask[j])
            flush = buf.flush(apply_mask=True, flush_frame_idx=n_vis - 1,
                              flush_timestamp=q.timestamp_s)
            kept = flush.num_tokens_after

        # ---- generate --------------------------------------------------- #
        text, gen = "", None
        if self.qwen is not None:
            gen = self.qwen.generate_from_pruned(
                query=q.prompt(),
                visual_tokens=flush.tokens,
                mrope=flush.mrope,
                dense_llm_grid_hw=vp.llm_grid_hw,
                max_new_tokens=self.max_new_tokens,
            )
            text = gen.text
            sw.add("llm_prefill", gen.prefill_ms)
            sw.add("llm_decode", gen.decode_ms, n=max(gen.num_new_tokens, 1))

        # ---- score ------------------------------------------------------ #
        letter, method = sb.parse_mcq(text, return_method=True)
        judged = False
        if letter is None:
            letter = sb.match_option_text(text, q.options)
            if letter is not None:
                method = "option_text"
        if letter is None and self.judge is not None:
            jr = self.judge(q.question, q.options, text)
            if jr.letter:
                letter, method, judged = jr.letter, "judge", True

        rec.update(
            predicted_response=(text if self.keep_raw_response else text[:200]),
            predicted_letter=letter,
            parse_method=method,
            parse_failed=letter is None,
            judged=judged,
            is_correct=bool(letter is not None and letter == q.answer),

            per_frame_keep_rates=[round(v, 6) for v in kr],
            per_plane_token_keep_rates=[round(v, 6) for v in flush.per_frame_keep_rate],
            avg_keep_rate=(kept / dense) if dense else 0.0,
            avg_qdp_keep_rate_clip=(sum(kr) / len(kr)) if kr else 0.0,
            visual_tokens_dense=dense,
            visual_tokens_kept=kept,
            forced_min_keep=forced,

            relevance_raw_scores=[round(v, 6) for v in rr],
            relevance_normalized_scores=[round(v, 6) for v in rel_cal],
            r_t_flags=[int(v) for v in r_flags],
            d_t_flags=[int(v) for v in d_flags],
            trigger_flags=[int(v) for v in t_flags],
            trigger_timestamps=[round(v, 3) for v in fire_ts],
            n_triggers=int(sum(t_flags)),
            n_fired=len(fire_ts),

            prefill_tokens=(gen.num_prefill_tokens if gen else 0),
            new_tokens=(gen.num_new_tokens if gen else 0),
            llm_prefill_ms=round(gen.prefill_ms, 3) if gen else 0.0,
            llm_decode_ms=round(gen.decode_ms, 3) if gen else 0.0,
            peak_vram_mb=round(gen.peak_vram_bytes / 1024 ** 2, 1) if gen else 0.0,
            wall_clock_sec=time.perf_counter() - t_q0,
            error=None,
        )
        return AnswerOutcome(rec, sw, dense, kept, gen.num_new_tokens if gen else 0)


__all__ = ["StreamingBenchRunner", "AnswerOutcome"]
