#!/usr/bin/env python
"""Merge sharded StreamingBench runs into one result set.

Each SLURM worker writes its own `shard<i>/eval_raw_logs.jsonl`. This
concatenates them (de-duplicating by question_id, last write wins), recomputes
the paper's Table 1 layout over the union, and reports coverage against the
2500-question total so a silently short run cannot be mistaken for a complete
one.

    python scripts/merge_shards.py runs/sb_full --expected 2500
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from querystream.benchmarks import streamingbench as sb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--pattern", default="shard*/eval_raw_logs.jsonl")
    ap.add_argument("--expected", type=int, default=None,
                    help="total questions the split should contain")
    ap.add_argument("--out", default=None, help="default: <run_dir>/merged")
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.run_dir, a.pattern)))
    if not paths:
        print(f"no shard logs under {a.run_dir}/{a.pattern}", file=sys.stderr)
        return 2

    by_id, bad = {}, 0
    for p in paths:
        n0 = len(by_id)
        with open(p) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    bad += 1
                    continue
                by_id[r["question_id"]] = r
        print(f"  {os.path.relpath(p, a.run_dir):<40} +{len(by_id) - n0}")

    records = list(by_id.values())
    out_dir = a.out or os.path.join(a.run_dir, "merged")
    os.makedirs(out_dir, exist_ok=True)
    merged = os.path.join(out_dir, "eval_raw_logs.jsonl")
    with open(merged, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    summary = sb.aggregate(records)
    summary["run"] = {"shards": len(paths), "malformed_lines": bad,
                      "questions": len(records)}
    if a.expected:
        summary["run"]["expected"] = a.expected
        summary["run"]["coverage_pct"] = 100.0 * len(records) / a.expected
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print()
    print(sb.format_table(summary))
    if a.expected and len(records) < a.expected:
        print(f"\n  INCOMPLETE: {len(records)}/{a.expected} questions "
              f"({100.0 * len(records) / a.expected:.1f}%) -- rerun with --resume")
    print(f"\n  merged  {merged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
