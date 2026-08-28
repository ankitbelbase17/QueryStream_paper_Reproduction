#!/usr/bin/env python
"""Rebuild summary.json for a run from its JSONL, without re-running anything.

A long evaluation can die mid-flight -- a wall-clock kill, an OOM, a decoder that
wedges on a bad file -- after it has already written thousands of good rows.
`eval_raw_logs.jsonl` is appended and flushed per video, so it survives; only the
end-of-run aggregation is lost. This reconstructs that aggregation, so a crashed
run is still a reportable run.

    python finalize_run.py runs/sb_subset
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from querystream.benchmarks import streamingbench as sb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing summary.json")
    a = ap.parse_args()

    jsonl = os.path.join(a.run_dir, "eval_raw_logs.jsonl")
    out = os.path.join(a.run_dir, "summary.json")
    if os.path.exists(out) and not a.force:
        print(f"{out} already exists; pass --force to overwrite")
        return 0

    records = [json.loads(l) for l in open(jsonl) if l.strip()]
    summary = sb.aggregate(records)

    # Profile rows are flushed every 25 writes, so the OVERALL row is usually
    # missing on a crashed run. Re-derive the per-stage means from what landed.
    prof, agg = {}, {}
    pcsv = os.path.join(a.run_dir, "profile_summary.csv")
    if os.path.exists(pcsv):
        vids, qs = [], []
        for row in csv.DictReader(open(pcsv)):
            (vids if row["scope"] == "video" else
             qs if row["scope"] == "question" else []).append(row)

        def wmean(rows, col, wcol):
            num = den = 0.0
            for r in rows:
                try:
                    w = float(r[wcol] or 0); num += float(r[col] or 0) * w; den += w
                except (ValueError, KeyError):
                    continue
            return num / den if den else 0.0

        agg = {
            "decode_ms_per_frame": wmean(vids, "decode_ms_per_frame", "n_frames"),
            "openclip_ms_per_frame": wmean(vids, "openclip_ms_per_frame", "n_frames"),
            "vision_tower_ms_per_frame": wmean(vids, "vision_tower_ms_per_frame", "n_frames"),
            "qdp_rtar_ms_per_frame": wmean(qs, "qdp_rtar_ms_per_frame", "n_frames"),
            "llm_prefill_ms": sum(float(r["llm_prefill_ms"] or 0) for r in qs) / max(len(qs), 1),
            "llm_decode_ms": sum(float(r["llm_decode_ms"] or 0) for r in qs) / max(len(qs), 1),
            "n_frames": sum(int(float(r["n_frames"] or 0)) for r in vids),
            "n_planes": sum(int(float(r["n_planes"] or 0)) for r in vids),
            "total_wall_clock_sec": sum(float(r["total_wall_clock_sec"] or 0)
                                        for r in vids + qs),
            "scope": "OVERALL (rebuilt)",
        }
        prof = agg

    cfg = {}
    cpath = os.path.join(a.run_dir, "config.json")
    if os.path.exists(cpath):
        cfg = json.load(open(cpath)).get("args", {})

    summary["profile"] = prof
    summary["run"] = {
        "videos_completed": len({r["video_id"] for r in records}),
        "questions_evaluated": len(records),
        "rebuilt_from_jsonl": True,
        "wall_clock_sec": agg.get("total_wall_clock_sec", 0.0),
        "clip_batch": cfg.get("clip_batch"),
        "decode_workers": cfg.get("decode_workers"),
        "time_budget_sec": cfg.get("time_budget_sec"),
        "mask_space": cfg.get("mask_space"), "pool": cfg.get("pool"),
        "relevance_mode": cfg.get("relevance_mode"), "keep_mode": cfg.get("keep_mode"),
        "tau_temp": cfg.get("tau_temp"), "tau_rel": cfg.get("tau_rel"),
        "tau_den": cfg.get("tau_den"), "alpha": cfg.get("alpha"),
        "fps": cfg.get("fps"), "max_pixels": cfg.get("max_pixels"),
    }
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(sb.format_table(summary))
    print(f"\n  rebuilt {out} from {len(records)} records "
          f"over {summary['run']['videos_completed']} videos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
