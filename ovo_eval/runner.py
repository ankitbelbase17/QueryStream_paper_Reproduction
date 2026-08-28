"""OVO-Bench runner: QDP over one pre-trimmed clip, one answer at its end.

OVO-Bench ships every question as a clip already cut to the query's `realtime`,
so the streaming rule is enforced by the dataset: "everything visible up to the
question" is simply the whole clip. That makes this the simplest of the three
arms -- decode, encode, prune, answer once -- and it is the same response policy
as StreamingBench (`response_mode="query_ts"`).

RTAR IS COMPUTED AND LOGGED BUT DOES NOT GATE GENERATION.
Table 2 scores accuracy at fixed probe points; nothing there measures *when* the
model speaks. The paper evaluates RTAR separately, in Table 5, on the Forward
split under the Appendix A.4 simulated protocol -- which needs the untrimmed
source video to place a response at a self-chosen timestamp, and the released
clips are pre-cut to the probe times. So `R_t / D_t / T_t` are recorded per
frame here for analysis and Table 5 is out of reach with this data; see
OVOBENCH.md.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
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
from .dataset import Call
from .scoring import score_call, strict_letter


@dataclass
class CallOutcome:
    record: Dict[str, Any]
    stopwatch: Stopwatch
    dense_tokens: int = 0
    kept_tokens: int = 0
    new_tokens: int = 0
    frames: int = 0


class OVORunner:
    """Answers one OVO-Bench Call from a `VideoPass` over its clip."""

    def __init__(
        self,
        clip: OpenCLIPWrapper,
        qwen: Optional[QwenVLWrapper],
        cfg: QueryStreamConfig,
        *,
        keep_mode: str = "qdp",
        gate_mode: str = "full",
        log_frame_traces: bool = False,
    ):
        self.clip = clip
        self.qwen = qwen
        self.cfg = cfg
        self.keep_mode = keep_mode
        self.gate_mode = gate_mode
        self.log_frame_traces = log_frame_traces
        assert keep_mode in ("qdp", "dense"), keep_mode

    # ------------------------------------------------------------------ #

    def _plane_masks(self, sem, temp, clip_hw, llm_hw, n_planes: int, k: int
                     ) -> torch.Tensor:
        """QDP masks on the BACKBONE token grid, one row per 2-frame plane.

        Identical arithmetic to the StreamingBench and OmniPro arms:
        `mask_space="score"` pools the continuous similarity maps onto the
        backbone grid and thresholds there, because 8 CLIP patches back one
        backbone token and pooling the binary mask with "any" inflates the keep
        rate from 41% to 96% (CLAUDE.md 12.2).
        """
        n = n_planes * k
        q = self.cfg.qdp

        if self.keep_mode == "dense":
            return torch.ones((n_planes, llm_hw[0] * llm_hw[1]),
                              dtype=torch.bool, device=sem.device)

        if self.cfg.mask_space == "score":
            pm = self.cfg.score_pool_mode
            sem_pl = resample_scores_batch(sem[:n], clip_hw, llm_hw, pm
                                           ).reshape(n_planes, k, -1).mean(dim=1)
            tmp_pl = resample_scores_batch(temp[:n], clip_hw, llm_hw, pm
                                           ).reshape(n_planes, k, -1).mean(dim=1)
            m_sem = sem_pl > sem_pl.mean(dim=-1, keepdim=True)     # STRICT >
            m_temp = tmp_pl < q.tau_temp                            # STRICT <
            if q.first_frame_all_novel and n_planes > 0:
                m_temp[0] = True
            return m_sem & m_temp

        m_clip = (sem[:n] > sem[:n].mean(dim=-1, keepdim=True)) & (temp[:n] < q.tau_temp)
        if q.first_frame_all_novel and n > 0:
            m_clip[0] = True
        per_frame = resample_patch_mask(m_clip, clip_hw, llm_hw, self.cfg.mask_resample_mode)
        return expand_mask_to_temporal_merge(per_frame, k, self.cfg.mask_resample_mode)

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def answer(self, vp: VideoPass, c: Call) -> CallOutcome:
        sw = Stopwatch(self.clip.device)
        t0 = time.perf_counter()
        rec: Dict[str, Any] = {
            "call_id": c.call_id, "sample_id": c.sample_id, "task": c.task,
            "category": c.category, "probe_index": c.probe_index,
            "realtime": c.realtime, "question": c.question,
            "options": c.options, "ground_truth": c.ground_truth,
            "probe_type": c.probe_type, "query_text": prompts.query_text(c),
        }
        try:
            return self._inner(vp, c, rec, sw, t0)
        except Exception as exc:              # never abort a long run
            rec.update(response=None, score=0, error=f"{type(exc).__name__}: {exc}",
                       traceback=traceback.format_exc(limit=6),
                       wall_clock_sec=round(time.perf_counter() - t0, 4))
            return CallOutcome(rec, sw)

    def _inner(self, vp: VideoPass, c: Call, rec, sw, t0) -> CallOutcome:
        cfg, q = self.cfg, self.cfg.qdp
        n_frames = vp.n_frames
        planes = vp.planes
        n_planes = len(planes)
        k = len(planes[0].frame_ids) if planes else 2
        rec.update(n_frames=n_frames, n_planes=n_planes,
                   dropped_trailing_frames=n_frames - n_planes * k)

        if n_planes == 0:
            rec.update(response=None, score=0,
                       error="clip yielded no complete 2-frame visual plane",
                       wall_clock_sec=round(time.perf_counter() - t0, 4))
            return CallOutcome(rec, sw)

        with sw.stage("qdp_rtar", n=n_frames):
            q_emb = self.clip.encode_text(prompts.query_text(c))
            sem = _sem_similarity(vp.patch_feats[:n_frames], q_emb)
            temp = vp.temp_sim[:n_frames].to(torch.float32)
            rel_raw = (l2_normalize(vp.frame_pooled[:n_frames].to(torch.float32), dim=-1)
                       @ l2_normalize(q_emb.reshape(-1).to(torch.float32), dim=-1))

            m_temp_c = temp < q.tau_temp
            if q.first_frame_all_novel:
                m_temp_c[0] = True
            clip_keep = ((sem > sem.mean(dim=-1, keepdim=True)) & m_temp_c
                         ).to(torch.float32).mean(dim=-1)

            plane_mask = self._plane_masks(sem, temp, vp.clip_grid_hw,
                                           vp.llm_grid_hw, n_planes, k)

            # RTAR is evaluated for the record; it does not gate here.
            rtar = RTARModule(cfg.rtar, gate_mode=self.gate_mode)
            kr, rr = clip_keep.tolist(), rel_raw.tolist()
            n_r = n_d = n_t = 0
            rel_cal: List[float] = []
            for i in range(n_frames):
                o = rtar(kr[i], relevance_raw=rr[i], timestamp=i / cfg.fps)
                n_r += o.r_t; n_d += o.d_t; n_t += o.trigger
                if self.log_frame_traces:
                    rel_cal.append(round(o.relevance, 6))

            buf = TokenBuffer(cfg.buffer)
            for j, pl in enumerate(planes):
                buf.append(frame_idx=pl.frame_ids[0], timestamp=pl.timestamp,
                           tokens=pl.tokens, mrope=pl.mrope, mask=plane_mask[j],
                           meta={"plane_idx": pl.plane_idx})
            flush = buf.flush(apply_mask=True, flush_frame_idx=n_frames - 1,
                              flush_timestamp=c.realtime)

        dense, kept = flush.num_tokens_before, flush.num_tokens_after
        forced = False
        if kept == 0:
            # Nothing survived anywhere: revive the single most query-relevant
            # token rather than hand the backbone an empty visual context.
            forced = True
            pooled = resample_scores_batch(
                sem[: n_planes * k], vp.clip_grid_hw, vp.llm_grid_hw,
                cfg.score_pool_mode).reshape(n_planes, k, -1).mean(dim=1)
            pj, pm = divmod(int(pooled.argmax().item()), pooled.shape[1])
            plane_mask[pj, pm] = True
            buf.update_mask(planes[pj].frame_ids[0], plane_mask[pj])
            flush = buf.flush(apply_mask=True, flush_frame_idx=n_frames - 1,
                              flush_timestamp=c.realtime)
            kept = flush.num_tokens_after

        text: Optional[str] = None
        new_tok = 0
        if self.qwen is not None:
            gen = self.qwen.generate_from_pruned(
                query=prompts.build_prompt(c),
                visual_tokens=flush.tokens, mrope=flush.mrope,
                dense_llm_grid_hw=vp.llm_grid_hw,
                max_new_tokens=prompts.max_new_tokens_for(c.prompt_kind),
            )
            text = gen.text.strip()
            new_tok = gen.num_new_tokens
            sw.add("llm_prefill", gen.prefill_ms)
            sw.add("llm_decode", gen.decode_ms, n=max(new_tok, 1))

        score = score_call(c.task, text, c.ground_truth)
        rec.update(
            response=text, score=int(score),
            strict_letter=strict_letter(text) if c.prompt_kind == "br" else None,
            tokens_dense=int(dense), tokens_kept=int(kept),
            keep_rate=round(kept / dense, 6) if dense else 0.0,
            forced_min_keep=forced,
            clip_keep_rate_mean=round(float(clip_keep.mean().item()), 6),
            relevance_raw_min=round(float(rel_raw.min().item()), 6),
            relevance_raw_max=round(float(rel_raw.max().item()), 6),
            n_R_t=int(n_r), n_D_t=int(n_d), n_T_t=int(n_t),
            wall_clock_sec=round(time.perf_counter() - t0, 4),
        )
        if self.log_frame_traces:
            rec["per_frame_keep_rates"] = [round(x, 6) for x in kr]
            rec["relevance_raw_scores"] = [round(x, 6) for x in rr]
            rec["relevance_normalized_scores"] = rel_cal
        return CallOutcome(rec, sw, dense_tokens=dense, kept_tokens=kept,
                           new_tokens=new_tok, frames=n_frames)
