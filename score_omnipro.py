#!/usr/bin/env python3
"""Offline scorer for an OmniPro run: merge shards, apply judge verdicts, aggregate.

Separated from `eval_omnipro.py` on purpose. Every GPU-side quantity is already
in the JSONL -- emission timestamps, response text, gate traces, token counts --
so scoring is a pure function of that file plus the judge verdicts. Changing the
matching tolerance, the debounce gap for the time-only tasks, or the judge's
pass threshold therefore costs nothing and touches no GPU.

    python score_omnipro.py runs/op_subset                     # timing + exact match
    python score_omnipro.py runs/op_subset --verdicts runs/op_subset/judge_output.jsonl
    python score_omnipro.py runs/op_subset --tolerance 5       # Appendix A.1 sweep
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omnipro_eval import dataset as ds
from omnipro_eval import judge as J
from omnipro_eval import metrics as M


def read_records(run_dir: str) -> List[dict]:
    """Union every shard's JSONL; last write wins per sample_id.

    Shards are disjoint by video, so overlap only happens when a shard was
    re-run; taking the last row means a resumed shard's fresh rows supersede
    the ones it replaced rather than being double-counted.
    """
    paths = sorted(glob.glob(os.path.join(run_dir, "omnipro_raw_logs*.jsonl")))
    by_id: Dict[str, dict] = {}
    n_lines = 0
    for p in paths:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                n_lines += 1
                by_id[r["sample_id"]] = r
    print(f"[read] {len(paths)} shard file(s), {n_lines} lines, "
          f"{len(by_id)} unique samples", flush=True)
    return list(by_id.values())


def load_verdicts(path: Optional[str]) -> Dict[str, dict]:
    """Judge verdicts from a Batch API output file, or {}.

    Accepts the raw `judge_output.jsonl` the Batch API returns; a malformed or
    schema-violating row is skipped, leaving that match UNJUDGED rather than
    inventing a verdict for it.
    """
    if not path or not os.path.exists(path):
        return {}
    out: Dict[str, dict] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if "response" in row:      # raw Batch API output
                    txt = row["response"]["body"]["choices"][0]["message"]["content"]
                    obj = json.loads(txt)
                    out[row["custom_id"]] = {"score": int(obj["score"]),
                                             "explanation": obj.get("explanation", "")}
                else:                      # already-normalised {custom_id: {...}}
                    out[row["custom_id"]] = {"score": int(row["score"]),
                                             "explanation": row.get("explanation", "")}
            except Exception:
                continue
    print(f"[judge] {len(out)} verdicts loaded from {path}", flush=True)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--verdicts", default=None,
                   help="judge_output.jsonl; defaults to the one in run_dir")
    p.add_argument("--tolerance", type=float, default=M.DEFAULT_TOLERANCE)
    p.add_argument("--pass-score", type=int, default=J.PASS_SCORE)
    p.add_argument("--gap-sweep", default="3,4,5,6,8,10,15,20",
                   help="comma-separated debounce gaps (seconds) to re-score "
                        "offline. Only gaps >= the one the run was recorded at "
                        "are possible: a larger gap yields a strict subset of "
                        "the responses already generated, a smaller one would "
                        "need responses that were never produced. Empty string "
                        "disables the sweep.")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    records = read_records(a.run_dir)
    ok = [r for r in records if not r.get("error")]
    errored = [r for r in records if r.get("error")]
    if errored:
        print(f"[warn] {len(errored)} samples errored and are EXCLUDED from the "
              f"metrics; they are listed in the output under `errors`.", flush=True)

    vpath = a.verdicts or os.path.join(a.run_dir, "judge_output.jsonl")
    verdicts = load_verdicts(vpath)
    if verdicts:
        attach = J.attach_verdicts(ok, verdicts)
        print(f"[judge] attached {attach['attached']} verdicts, "
              f"score distribution {attach['score_distribution']}, "
              f"pass rate {attach['pass_rate']}", flush=True)
    else:
        attach = {"attached": 0}

    judge_fn = J.CachedJudge(verdicts, pass_score=a.pass_score) if verdicts else None
    scores = [M.score_sample(r, tolerance=a.tolerance, judge=judge_fn) for r in ok]

    agg = M.aggregate(scores)
    slices = M.slice_report(scores, ds.build_slices())
    wj = M.weighted_joint(slices, metric="macro_joint_f1")
    wj_time = M.weighted_joint(slices, metric="macro_time_f1")
    wj_lb = M.weighted_joint(slices, metric="macro_joint_f1_lb")

    gaps = [float(x) for x in a.gap_sweep.split(",") if x.strip()]
    sweep = M.gap_sweep(ok, gaps, tolerance=a.tolerance, judge=judge_fn) if gaps else []

    gate_ab = M.gate_ablation(ok, tolerance=a.tolerance)

    # Gate firing rates -- the diagnosis behind the ablation.
    nf = sum(r.get("n_frames_streamed", 0) for r in ok) or 1
    gate_rates = {
        "frames": nf,
        "R_t_true_pct": round(100 * sum(r.get("n_R_t", 0) for r in ok) / nf, 2),
        "D_t_true_pct": round(100 * sum(r.get("n_D_t", 0) for r in ok) / nf, 2),
        "T_t_true_pct": round(100 * sum(r.get("n_raw_triggers", 0) for r in ok) / nf, 2),
        "clip_keep_rate_mean": round(
            sum(r.get("clip_keep_rate_mean", 0.0) for r in ok) / len(ok), 4) if ok else None,
        "relevance_raw_max": round(max((r.get("relevance_raw_max", 0.0) for r in ok),
                                       default=0.0), 4),
    }

    keep_num = sum(r.get("tokens_kept_stream", 0) for r in ok)
    keep_den = sum(r.get("tokens_dense_stream", 0) for r in ok)

    out = {
        "run_dir": a.run_dir,
        "tolerance_s": a.tolerance,
        "judge": {"verdicts_loaded": len(verdicts), "pass_score": a.pass_score,
                  **attach,
                  "hits": getattr(judge_fn, "n_hit", 0),
                  "misses": getattr(judge_fn, "n_miss", 0),
                  "model_note": "OpenAI substituted for the paper's "
                                "Gemini-3-Flash; rubric and >=3 threshold "
                                "unchanged. Judged scores from different judges "
                                "are not strictly comparable."},
        "n_samples_scored": len(scores),
        "n_samples_errored": len(errored),
        "errors": [{"sample_id": r.get("sample_id"), "error": r.get("error")}
                   for r in errored][:50],
        "token_keep_rate": round(keep_num / keep_den, 6) if keep_den else 0.0,
        "tokens_kept": keep_num, "tokens_dense": keep_den,
        "metrics": agg,
        "slices": slices,
        "weighted_joint_f1": wj,
        "weighted_joint_f1_lb": wj_lb,
        "weighted_time_f1": wj_time,
        "gate_ablation": gate_ab,
        "gate_rates": gate_rates,
        "gap_sweep": sweep,
        "gap_sweep_note": (
            "Sensitivity analysis, not a search. The QueryStream paper leaves "
            "the Appendix A.4 event-grouping gap unspecified; this run was "
            "recorded at the gap in `config.event_gap_s` and that row is the "
            "headline. Larger gaps are scored EXACTLY -- they select a strict "
            "subset of the responses already generated, whose text is in the "
            "JSONL -- so no GPU is involved and content is fully included. "
            "Quoting the best row as the result would be tuning against the "
            "benchmark."),
        "paper_reference": {
            "online_audio_visual": ds.PAPER_ONLINE,
            "online_video_only_minicpm_o_4_5": ds.PAPER_ONLINE_VIDEO_ONLY,
            "online_audio_only_minicpm_o_4_5_mean": ds.PAPER_ONLINE_AUDIO_ONLY_MEAN,
            "tolerance_s": ds.PAPER_TOLERANCE_S,
        },
    }
    dest = a.out or os.path.join(a.run_dir, "omnipro_metrics.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2, default=str)

    _print_tables(out)
    print(f"\nwrote {dest}")
    return 0


def _print_tables(out: dict) -> None:
    ov = out["metrics"]["overall"]
    print("\n" + "=" * 78)
    print(f"OMNIPRO ONLINE MODE  |  n={out['n_samples_scored']}  "
          f"tol=+-{out['tolerance_s']}s  keep={100 * out['token_keep_rate']:.1f}%")
    print("=" * 78)
    print(f"  responses emitted {ov['n_emits']}   ground-truth triggers {ov['n_gt']}   "
          f"time-matched {ov['n_matched']}")
    print(f"  TIME   P {ov['time_precision']:.4f}  R {ov['time_recall']:.4f}  "
          f"F1 {ov['time_f1']:.4f}   macro {ov['macro_time_f1']}")
    ca = ov.get("content_acc")
    print(f"  CONTENT ACC  {ca if ca is not None else str(ov['content_acc_lb']) + ' (lower bound)'}"
          f"   macro {ov.get('macro_content_acc') or ov.get('macro_content_acc_lb')}")
    jf = ov.get("joint_f1")
    print(f"  JOINT  F1 {jf if jf is not None else str(ov['joint_f1_lb']) + ' (lower bound)'}"
          f"   macro {ov.get('macro_joint_f1') or ov.get('macro_joint_f1_lb')}")

    print("\n  per sub-task (joint F1 x100, paper Table 2 order):")
    hdr = f"    {'task':<30}{'n':>5}{'emits':>7}{'gt':>6}{'timeF1':>9}{'cAcc':>8}{'jointF1':>9}"
    print(hdr)
    for t, b in out["metrics"]["per_task"].items():
        ca = b.get("content_acc")
        jf = b.get("joint_f1")
        ca_s = f"{100 * ca:.1f}" if ca is not None else f"<{100 * b['content_acc_lb']:.1f}"
        jf_s = f"{100 * jf:.1f}" if jf is not None else f"<{100 * b['joint_f1_lb']:.1f}"
        print(f"    {ds.SHORT_NAME.get(t, t):<30}{b['n_samples']:>5}{b['n_emits']:>7}"
              f"{b['n_gt']:>6}{100 * b['time_f1']:>9.1f}{ca_s:>8}{jf_s:>9}")

    print("\n  headline slices:")
    for name in ds.HEADLINE_SLICES:
        b = out["slices"].get(name, {})
        if not b:
            continue
        jf = b.get("macro_joint_f1")
        jf_s = f"{100 * jf:.1f}" if jf is not None else \
               f"<{100 * (b.get('macro_joint_f1_lb') or 0):.1f}"
        print(f"    {b['title']:<34} n={b['n_samples']:<5} "
              f"timeF1 {100 * (b.get('macro_time_f1') or 0):.1f}  jointF1 {jf_s}")

    gr = out.get("gate_rates", {})
    if gr:
        print(f"\n  gate firing rates over {gr['frames']:,} frames:")
        print(f"    R_t true {gr['R_t_true_pct']:.1f}%   "
              f"D_t true {gr['D_t_true_pct']:.1f}%   "
              f"T_t true {gr['T_t_true_pct']:.1f}%")
        print(f"    CLIP-grid keep rate (the D_t signal) mean "
              f"{gr['clip_keep_rate_mean']:.4f} vs tau_den -- D_t is near-vacuous "
              f"when the keep rate sits far above the threshold")
        print(f"    max raw cos(q, mean-pool v) anywhere: {gr['relevance_raw_max']:.4f} "
              f"vs tau_rel = 0.60")
    if out.get("gate_ablation"):
        print("\n  RTAR gate ablation (paper Table 5 structure; TIMING only):")
        print(f"    {'gate':<16}{'emits':>9}{'em/GT':>8}{'timeP':>8}{'timeR':>8}{'timeF1':>9}")
        for row in out["gate_ablation"]:
            print(f"    {row['gate_mode']:<16}{row['n_emits']:>9}"
                  f"{row['emits_per_gt']:>8.2f}{100 * row['time_precision']:>8.1f}"
                  f"{100 * row['time_recall']:>8.1f}{100 * row['time_f1']:>9.1f}")
        print("    (single-gate rows fire on a superset of the full gate, so no "
              "response text exists for their extra firings: timing only)")

    if out.get("gap_sweep"):
        print("\n  debounce-gap sensitivity (offline, exact for gaps >= the "
              "recorded one):")
        print(f"    {'gap':>6}{'emits':>9}{'em/GT':>8}{'timeP':>8}{'timeR':>8}"
              f"{'timeF1':>9}{'jointF1':>9}")
        for row in out["gap_sweep"]:
            if row.get("error"):
                print(f"    {row['gap_s']:>6.1f}   {row['error'][:60]}")
                continue
            jf = row.get("joint_f1")
            jf_s = f"{100 * jf:.1f}" if jf is not None else \
                   f"<{100 * row['joint_f1_lb']:.1f}"
            print(f"    {row['gap_s']:>6.1f}{row['n_emits']:>9}"
                  f"{row['emits_per_gt']:>8.2f}{100 * row['time_precision']:>8.1f}"
                  f"{100 * row['time_recall']:>8.1f}{100 * row['time_f1']:>9.1f}"
                  f"{jf_s:>9}")
        print("    (the recorded gap is the headline; the rest is sensitivity)")

    wj = out["weighted_joint_f1"]
    val = wj.get("benchmark_weighted")
    if val is None:
        wj = out["weighted_joint_f1_lb"]
        val = wj.get("benchmark_weighted")
        label = "benchmark-weighted macro joint F1 (LOWER BOUND, judge pending)"
    else:
        label = "benchmark-weighted macro joint F1"
    print(f"\n  {label}: "
          f"{100 * val:.2f}" if val is not None else f"\n  {label}: n/a")
    print(f"    weights {wj.get('benchmark_weights')}")
    print(f"    equal-weighted {wj.get('equal_weighted')}   "
          f"subset-weighted {wj.get('subset_weighted')}")
    print(f"  paper reference, MiniCPM-o 4.5 VIDEO-ONLY online mean F1: "
          f"{ds.PAPER_ONLINE_VIDEO_ONLY['Mean']}  "
          f"(same model with audio: {ds.PAPER_ONLINE['MiniCPM-o-4.5-9B']['Mean']})")
    print("=" * 78)


if __name__ == "__main__":
    raise SystemExit(main())
