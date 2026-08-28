"""OmniPro Online-mode metrics: Time F1, Content Accuracy, Joint F1.

DEFINITIONS (OmniPro Sec. 3.2.1, verbatim):
    "Model responses are matched to ground-truth triggers via greedy temporal
     alignment with a tolerance of +-3 s. A match is considered valid only if the
     response is also content-correct. Precision is the fraction of model
     responses that are validly matched, recall is the fraction of ground-truth
     triggers that are validly matched, and F1 is their harmonic mean."

That single paragraph defines the headline metric (Joint F1) but folds two
separable failure modes into one number: speaking at the wrong moment, and
speaking the wrong thing. This module reports both halves and the product:

    TIME F1      timing alone. A response is a true positive if it lands within
                 +-tolerance of an unused ground-truth trigger, whatever it says.
                     time_P = tp_time / n_emits      (n_emits = tp_time + fp)
                     time_R = tp_time / n_gt         (n_gt     = tp_time + fn)

    CONTENT ACC  of the responses that DID land on a trigger, the fraction whose
                 text is also correct.
                     content_acc = tp_content / tp_time
                 Conditioned on a time match by construction -- content is only
                 ever adjudicated against the trigger a response was matched to,
                 because there is no ground-truth text for a response that
                 matched nothing.

    JOINT F1     the benchmark's headline. Denominators are the FULL response
                 and trigger counts, not the content-filtered ones:
                     joint_P = tp_content / n_emits
                     joint_R = tp_content / n_gt
                 Writing joint_P as tp_content / (tp_content + fp) instead is a
                 real and tempting bug: it quietly drops every response that
                 matched in time but failed on content, instead of counting it
                 as a false positive. That inflates Joint F1 by up to ~18%, and
                 inflates it most exactly where content is worst.

    Identity worth keeping in mind: joint_F1 is NOT time_F1 x content_acc, but
    when precision and recall are close the two track each other tightly, and a
    large divergence between them is a signal that emissions and triggers are
    badly imbalanced.

MICRO VS MACRO
`overall` pools tp/fp/fn across every sample, so a task contributes in
proportion to how many responses and triggers it happens to carry. The paper's
Table 2 "Mean" column is instead an equal-weight average over the nine
sub-tasks. Both are computed; `macro_*` is the number comparable to the paper
and `overall` is the micro pool. They can differ substantially on an uneven
subset, which is why neither is reported alone.

UNJUDGED IS NOT ZERO
The two free-text tasks need an LLM judge. If a verdict is missing, the match is
counted in neither the correct nor the incorrect column: `tp_content` becomes a
lower bound, and `content_acc` / `joint_f1` are withheld (None) in favour of
`content_acc_lb` / `joint_f1_lb`, named so they cannot be mistaken for the real
thing. Timing metrics never involve the judge and are always exact.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .content import structured_correct
from .dataset import (ALL_TASKS, AUDIO_CLASSES, FULL_BENCHMARK_AUDIO_SHARE,
                      PAPER_ORDER, TASK_CONTENT_KIND)

DEFAULT_TOLERANCE = 3.0     # seconds; OmniPro default, Appendix A.1


# --------------------------------------------------------------------------- #
# temporal matching
# --------------------------------------------------------------------------- #

def match_emits_to_gt(
    emit_times: Sequence[float],
    gt_times: Sequence[float],
    tolerance: float = DEFAULT_TOLERANCE,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Greedy 1-to-1 alignment, closest pair first.

    Every (emit, trigger) pair within `tolerance` is a candidate; pairs are
    consumed in order of increasing |dt|, and each emit and each trigger is used
    at most once. Greedy-closest is what the benchmark specifies. It is not
    guaranteed optimal -- a maximum-cardinality matching could occasionally pair
    up one more emit -- but matching the benchmark matters more than matching an
    optimum it does not use.

    Returns:
        (matches, unmatched_emit_idx, unmatched_gt_idx) where each match is
        (emit_index, gt_index, |dt|).
    """
    pairs = []
    for i, et in enumerate(emit_times):
        for j, gt in enumerate(gt_times):
            dt = abs(float(et) - float(gt))
            if dt <= tolerance:
                pairs.append((dt, i, j))
    pairs.sort(key=lambda x: (x[0], x[1], x[2]))     # deterministic tie-break

    used_e, used_g, matches = set(), set(), []
    for dt, i, j in pairs:
        if i in used_e or j in used_g:
            continue
        used_e.add(i)
        used_g.add(j)
        matches.append((i, j, dt))
    matches.sort(key=lambda m: m[0])
    return (matches,
            [i for i in range(len(emit_times)) if i not in used_e],
            [j for j in range(len(gt_times)) if j not in used_g])


# --------------------------------------------------------------------------- #
# per-sample scoring
# --------------------------------------------------------------------------- #

@dataclass
class SampleScore:
    """Counts for one sample, plus the per-match detail the report needs."""

    sample_id: str
    task: str
    audio_dependency: str
    tp_time: int
    tp_content: int
    n_unjudged: int
    fp: int
    fn: int
    n_emits: int
    n_gt: int
    matches: List[dict]              # one per time-match, with dt and verdict
    unmatched_gt: List[dict]         # ground truth the model never spoke to

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


JudgeFn = Callable[[str, str, str], Optional[bool]]
"""(question, ground_truth_response, model_reply) -> True / False / None(unjudged)."""


def score_sample(
    record: dict,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    judge: Optional[JudgeFn] = None,
) -> SampleScore:
    """Score one prediction record produced by runner.py.

    `record` carries `emits` = [{"t_sec": float, "text": str}, ...] and the
    sample's `ground_truth`. The record is not mutated; per-match verdicts are
    returned inside the `SampleScore` so they can be written to the JSONL and
    re-scored later without re-running the GPU.
    """
    task = record["task"]
    kind = TASK_CONTENT_KIND.get(task, "gpt_judge")
    emits = record.get("emits", [])
    gts = record["ground_truth"]

    emit_times = [float(e["t_sec"]) for e in emits]
    gt_times = [float(g["t_sec"]) for g in gts]
    matched, un_e, un_g = match_emits_to_gt(emit_times, gt_times, tolerance)

    tp_content = 0
    n_unjudged = 0
    detail: List[dict] = []
    for ei, gj, dt in matched:
        reply = emits[ei].get("text", "")
        gt_item = gts[gj]
        if kind == "time_only":
            ok: Optional[bool] = True          # content is not part of the metric
        elif kind == "gpt_judge":
            if judge is None:
                ok = None
            else:
                # A CachedJudge keys its table by (sample_id, emit_idx); a plain
                # callable ignores the binding. Doing it here rather than
                # widening the JudgeFn signature keeps a bare function usable.
                if hasattr(judge, "bind"):
                    judge.bind(record["sample_id"], ei)
                ok = judge(record.get("question", ""),
                           gt_item.get("response", ""), reply)
        else:
            ok = structured_correct(kind, reply, gt_item)

        if ok is None:
            n_unjudged += 1
        elif ok:
            tp_content += 1
        detail.append({
            "emit_idx": ei, "gt_idx": gj, "dt": round(dt, 3),
            "emit_t": emit_times[ei], "gt_t": gt_times[gj],
            "content_kind": kind, "content_ok": ok,
            "trigger_type": gt_item.get("trigger_type", "unknown"),
        })

    return SampleScore(
        sample_id=record["sample_id"], task=task,
        audio_dependency=record.get("audio_dependency", "unknown"),
        tp_time=len(matched), tp_content=tp_content, n_unjudged=n_unjudged,
        fp=len(un_e), fn=len(un_g), n_emits=len(emits), n_gt=len(gt_times),
        matches=detail,
        unmatched_gt=[{"gt_idx": j, "gt_t": gt_times[j],
                       "trigger_type": gts[j].get("trigger_type", "unknown")}
                      for j in un_g],
    )


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #

def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) else 0.0


def block(counts: Dict[str, float]) -> dict:
    """Turn pooled tp/fp/fn counts into the full metric block."""
    tpt = counts.get("tp_time", 0.0)
    tpc = counts.get("tp_content", 0.0)
    fp = counts.get("fp", 0.0)
    fn = counts.get("fn", 0.0)
    unj = counts.get("n_unjudged", 0.0)
    n_emits = tpt + fp
    n_gt = tpt + fn

    time_p = tpt / n_emits if n_emits else 0.0
    time_r = tpt / n_gt if n_gt else 0.0
    joint_p = tpc / n_emits if n_emits else 0.0
    joint_r = tpc / n_gt if n_gt else 0.0

    complete = (unj == 0)
    out = {
        "n_samples": int(counts.get("n", 0)),
        "n_emits": int(n_emits),
        "n_gt": int(n_gt),
        "n_matched": int(tpt),
        "n_content_correct": int(tpc),
        "n_judged": int(tpt - unj),
        "n_unjudged": int(unj),
        "content_coverage": round((tpt - unj) / tpt, 4) if tpt else 1.0,
        "content_complete": bool(complete),
        "time_precision": round(time_p, 4),
        "time_recall": round(time_r, 4),
        "time_f1": round(_f1(time_p, time_r), 4),
        # Lower bounds are always available: they treat every unjudged match as
        # wrong, which is the worst case, and are named so they cannot pass for
        # the real number.
        "content_acc_lb": round(tpc / tpt, 4) if tpt else 0.0,
        "joint_precision_lb": round(joint_p, 4),
        "joint_recall_lb": round(joint_r, 4),
        "joint_f1_lb": round(_f1(joint_p, joint_r), 4),
        # Emission behaviour -- not a benchmark metric, but the thing that
        # explains a low precision, so it is reported next to it.
        "emits_per_sample": round(n_emits / counts["n"], 3) if counts.get("n") else 0.0,
        "gt_per_sample": round(n_gt / counts["n"], 3) if counts.get("n") else 0.0,
    }
    if complete:
        out.update({
            "content_acc": round(tpc / tpt, 4) if tpt else 0.0,
            "joint_precision": round(joint_p, 4),
            "joint_recall": round(joint_r, 4),
            "joint_f1": round(_f1(joint_p, joint_r), 4),
        })
    else:
        out.update({"content_acc": None, "joint_precision": None,
                    "joint_recall": None, "joint_f1": None})
    return out


_COUNT_KEYS = ("tp_time", "tp_content", "n_unjudged", "fp", "fn")


def _pool(scores: Sequence[SampleScore]) -> Dict[str, float]:
    c: Dict[str, float] = defaultdict(float)
    for s in scores:
        for k in _COUNT_KEYS:
            c[k] += getattr(s, k)
        c["n"] += 1
    return dict(c)


def macro_over_tasks(per_task: Dict[str, dict], key: str) -> Tuple[Optional[float], int]:
    """Equal-weight mean over sub-tasks -- the paper's Table 2 "Mean" column.

    Tasks whose metric is withheld (unjudged) are excluded and the surviving
    count is returned alongside, so a partial average can never be mistaken for
    a full one.
    """
    vals = [b[key] for b in per_task.values() if b.get(key) is not None]
    return (round(sum(vals) / len(vals), 4) if vals else None), len(vals)


def aggregate(scores: Sequence[SampleScore]) -> dict:
    """Micro overall + per-task blocks + the macro means over sub-tasks."""
    by_task: Dict[str, List[SampleScore]] = defaultdict(list)
    for s in scores:
        by_task[s.task].append(s)

    per_task = {t: block(_pool(v)) for t, v in by_task.items()}
    ordered = {t: per_task[t] for t in PAPER_ORDER if t in per_task}
    ordered.update({t: b for t, b in per_task.items() if t not in ordered})

    ov = block(_pool(scores))
    for key, out_key in (("time_f1", "macro_time_f1"),
                         ("content_acc", "macro_content_acc"),
                         ("joint_f1", "macro_joint_f1"),
                         ("content_acc_lb", "macro_content_acc_lb"),
                         ("joint_f1_lb", "macro_joint_f1_lb")):
        v, n = macro_over_tasks(ordered, key)
        ov[out_key] = v
        ov[out_key + "_n_tasks"] = n
    ov["n_tasks"] = len(ordered)
    return {"overall": ov, "per_task": ordered}


# --------------------------------------------------------------------------- #
# slices
# --------------------------------------------------------------------------- #

def slice_scores(scores: Sequence[SampleScore], predicate) -> List[SampleScore]:
    return [s for s in scores if predicate(
        {"task": s.task, "audio_dependency": s.audio_dependency,
         "sample_id": s.sample_id})]


def modality_block(scores: Sequence[SampleScore], modality: str,
                   ) -> dict:
    """Metrics restricted to ground-truth triggers of one modality (Figure 4).

    This one cannot be pooled from sample-level counts: modality is a property
    of a *trigger*, not of a sample, and a single sample routinely mixes them.
    So the counts are rebuilt trigger by trigger.

    A false positive has no modality -- it matched no trigger -- so precision
    here has no honest denominator, and only RECALL-side quantities are defined.
    That is exactly what the paper's Figure 4 reports as a per-modality score,
    and reporting a fabricated precision alongside would be worse than
    reporting less.
    """
    n_gt = n_matched = n_correct = n_unj = 0
    for s in scores:
        for m in s.matches:
            if m["trigger_type"] != modality:
                continue
            n_gt += 1
            n_matched += 1
            if m["content_ok"] is None:
                n_unj += 1
            elif m["content_ok"]:
                n_correct += 1
        for u in s.unmatched_gt:
            if u["trigger_type"] == modality:
                n_gt += 1
    complete = (n_unj == 0)
    return {
        "modality": modality,
        "n_gt": n_gt,
        "n_matched": n_matched,
        "n_content_correct": n_correct,
        "n_unjudged": n_unj,
        "time_recall": round(n_matched / n_gt, 4) if n_gt else 0.0,
        "joint_recall_lb": round(n_correct / n_gt, 4) if n_gt else 0.0,
        "joint_recall": (round(n_correct / n_gt, 4) if (n_gt and complete) else None),
        "content_acc": (round(n_correct / n_matched, 4)
                        if (n_matched and complete) else None),
        "note": "recall-side only: a false positive matches no trigger and so "
                "carries no modality, leaving precision undefined here.",
    }


def slice_report(scores: Sequence[SampleScore], slices) -> Dict[str, dict]:
    """Run every registered slice and return {slice_name: {...}}."""
    out: Dict[str, dict] = {}
    for sl in slices:
        if sl.kind == "modality":
            m = sl.name[len("modality_"):].replace("_", "+")
            b = modality_block(scores, m)
            b.update(title=sl.title, description=sl.description, kind=sl.kind)
            out[sl.name] = b
            continue
        sub = slice_scores(scores, sl.predicate)
        agg = aggregate(sub) if sub else {"overall": block({"n": 0}), "per_task": {}}
        b = dict(agg["overall"])
        b.update(title=sl.title, description=sl.description, kind=sl.kind,
                 per_task={t: v["joint_f1"] for t, v in agg["per_task"].items()},
                 per_task_time_f1={t: v["time_f1"] for t, v in agg["per_task"].items()},
                 per_task_full=agg["per_task"])
        out[sl.name] = b
    return out


# --------------------------------------------------------------------------- #
# the slice-weighted headline
# --------------------------------------------------------------------------- #

def weighted_joint(
    slice_blocks: Dict[str, dict],
    *,
    slice_names: Sequence[str] = ("audio_none", "audio_helpful", "audio_required"),
    weights: Optional[Dict[str, float]] = None,
    metric: str = "macro_joint_f1",
) -> dict:
    """One number over all the data, weighting the audio slices explicitly.

    Two weightings are reported because they answer different questions:

      benchmark_weighted   weights each audio slice by its share of the FULL
                           2,700-sample OmniPro benchmark (required 65.5%,
                           helpful 18.5%, none 16.0%). This is the number that
                           estimates what the system would score on the whole
                           benchmark, and it is dominated by the audio-required
                           slice that a vision-only system cannot do. It is the
                           honest headline, and it is meant to look bad.

      subset_weighted      weights each slice by its share of the samples
                           actually evaluated. Differs from the micro pool only
                           through the macro-over-tasks step inside each slice.

      equal_weighted       one third each. Reported because it is the reading
                           that treats "how well does it handle each modality
                           condition" as the question, rather than "what would
                           it score on the benchmark as distributed".

    Whichever is quoted, the weights are carried in the output next to the
    number so it can never be reported without them.
    """
    present = [n for n in slice_names if slice_blocks.get(n, {}).get(metric) is not None]
    missing = [n for n in slice_names if n not in present]
    if not present:
        return {"metric": metric, "value": None, "reason": "no slice has a complete metric",
                "missing": missing}

    def wmean(w: Dict[str, float]) -> Optional[float]:
        tot = sum(w.get(n, 0.0) for n in present)
        if tot <= 0:
            return None
        return round(sum(slice_blocks[n][metric] * w.get(n, 0.0)
                         for n in present) / tot, 4)

    bench_w = {f"audio_{a}": FULL_BENCHMARK_AUDIO_SHARE[a] for a in AUDIO_CLASSES}
    subset_w = {n: float(slice_blocks[n]["n_samples"]) for n in present}
    equal_w = {n: 1.0 for n in present}

    if weights:
        bench_w = dict(weights)

    return {
        "metric": metric,
        "slices": list(present),
        "missing_slices": missing,
        "benchmark_weighted": wmean(bench_w),
        "subset_weighted": wmean(subset_w),
        "equal_weighted": wmean(equal_w),
        "benchmark_weights": {n: round(bench_w.get(n, 0.0), 4) for n in present},
        "subset_weights": {n: round(subset_w[n] / sum(subset_w.values()), 4)
                           for n in present},
        "per_slice": {n: slice_blocks[n][metric] for n in present},
    }


# --------------------------------------------------------------------------- #
# debounce-gap sensitivity, computed offline at zero GPU cost
# --------------------------------------------------------------------------- #

def _debounce(trigger_ts: Sequence[float], gap: float) -> List[float]:
    """Appendix A.4 event grouping: the first timestamp of each interval.

    A new event opens when the gap since the *previous trigger* (not since the
    previous event) exceeds `gap`, so an interval is a maximal run of triggers
    each within `gap` of its predecessor.
    """
    out: List[float] = []
    last: Optional[float] = None
    for t in trigger_ts:
        if last is None or (t - last) > gap:
            out.append(t)
        last = t
    return out


def regate(record: dict, gap: float) -> dict:
    """A copy of `record` holding the responses a LARGER debounce gap would give.

    WHY THIS IS EXACT AND NOT AN APPROXIMATION. For gaps g1 < g2, every
    g2-interval is a union of consecutive g1-intervals, so the first timestamp
    of a g2-interval is the first timestamp of its leading g1-interval. The
    g2 response set is therefore a strict SUBSET of the g1 response set --
    verified on the run's own data -- and since every g1 response's generated
    text is already in the JSONL, a larger gap can be scored completely,
    content included, without a GPU.

    The converse does not hold: a SMALLER gap would open events at timestamps
    where nothing was ever generated, so it is refused rather than guessed at.
    """
    base = float(record.get("event_gap_s", DEFAULT_TOLERANCE))
    if gap < base - 1e-9:
        raise ValueError(
            f"cannot re-gate to {gap}s from a run recorded at {base}s: a smaller "
            f"gap opens events at timestamps where no response was generated. "
            f"Re-run the stream with --event-gap-s {gap}."
        )
    keep = set(_debounce(record.get("raw_trigger_ts", []), gap))
    out = dict(record)
    out["emits"] = [e for e in record.get("emits", []) if e["t_sec"] in keep]
    out["n_emits"] = len(out["emits"])
    out["event_gap_s"] = gap
    return out


def gap_sweep(
    records: Sequence[dict],
    gaps: Sequence[float],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    judge: Optional[JudgeFn] = None,
) -> List[dict]:
    """Score the run again at each debounce gap. One row per gap.

    This turns "the Appendix A.4 gap is an unspecified reproduction choice" into
    a measured curve rather than an assumption. It is a SENSITIVITY ANALYSIS,
    not a search: the shipped gap stays the headline, because picking the gap
    that maximises the metric would be tuning against the benchmark.
    """
    rows = []
    for g in gaps:
        try:
            regated = [regate(r, g) for r in records]
        except ValueError as exc:
            rows.append({"gap_s": g, "error": str(exc)})
            continue
        agg = aggregate([score_sample(r, tolerance=tolerance, judge=judge)
                         for r in regated])
        ov = agg["overall"]
        rows.append({
            "gap_s": g,
            "n_emits": ov["n_emits"], "n_gt": ov["n_gt"],
            "emits_per_gt": round(ov["n_emits"] / ov["n_gt"], 3) if ov["n_gt"] else None,
            "time_precision": ov["time_precision"],
            "time_recall": ov["time_recall"],
            "time_f1": ov["time_f1"],
            "macro_time_f1": ov["macro_time_f1"],
            "content_acc": ov.get("content_acc"),
            "content_acc_lb": ov["content_acc_lb"],
            "joint_f1": ov.get("joint_f1"),
            "joint_f1_lb": ov["joint_f1_lb"],
            "macro_joint_f1": ov.get("macro_joint_f1"),
            "macro_joint_f1_lb": ov.get("macro_joint_f1_lb"),
        })
    return rows


# --------------------------------------------------------------------------- #
# RTAR gate ablation, timing-only, computed offline
# --------------------------------------------------------------------------- #

GATE_MODES = ("full", "relevance_only", "density_only")


def gate_ablation(
    records: Sequence[dict],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    gap: Optional[float] = None,
) -> List[dict]:
    """Reproduce the paper's Table 5 gate ablation on the TIMING dimension.

    QueryStream Table 5 compares three triggering policies on OVO-Bench's
    Forward Active Responding split:

        density-only   (== TimeChat-Online's query-agnostic heuristic)
        relevance-only
        full RTAR      R_t AND D_t

    `R_t` and `D_t` are logged per frame, so all three trigger *timelines* can
    be rebuilt exactly from the JSONL and scored for timing without a GPU.

    ONLY TIMING. The full gate is the intersection, so both single-gate policies
    fire on a SUPERSET of its frames -- responses exist for the full gate's
    frames alone, and the extra ones have no generated text. Content, and
    therefore Joint F1, cannot be computed for the single-gate rows and is not
    guessed at. Timing is exactly where the paper's ablation is interesting
    anyway: its claim is that relevance-only over-fires on static relevant
    scenes and density-only over-fires on irrelevant motion.
    """
    rows = []
    for mode in GATE_MODES:
        tp = fp = fn = n_emits = n_gt = 0
        per_sample = 0
        for r in records:
            rt, dt = r.get("R_t"), r.get("D_t")
            if not rt or not dt:
                continue
            g = float(gap if gap is not None else r.get("event_gap_s",
                                                        DEFAULT_TOLERANCE))
            fps = 1.0
            if mode == "full":
                fire = [i for i, (a, b) in enumerate(zip(rt, dt)) if a and b]
            elif mode == "relevance_only":
                fire = [i for i, a in enumerate(rt) if a]
            else:
                fire = [i for i, b in enumerate(dt) if b]
            ts = _debounce([i / fps for i in fire], g)
            gts = [float(x["t_sec"]) for x in r.get("ground_truth", [])]
            m, ue, ug = match_emits_to_gt(ts, gts, tolerance)
            tp += len(m)
            fp += len(ue)
            fn += len(ug)
            n_emits += len(ts)
            n_gt += len(gts)
            per_sample += 1
        p = tp / n_emits if n_emits else 0.0
        rec = tp / n_gt if n_gt else 0.0
        rows.append({
            "gate_mode": mode,
            "n_samples": per_sample,
            "n_emits": n_emits,
            "n_gt": n_gt,
            "emits_per_gt": round(n_emits / n_gt, 3) if n_gt else None,
            "time_precision": round(p, 4),
            "time_recall": round(rec, 4),
            "time_f1": round(_f1(p, rec), 4),
            "note": ("timing only -- the single-gate policies fire on a superset "
                     "of the full gate's frames, so no response text exists for "
                     "their extra firings and content cannot be scored"),
        })
    return rows
