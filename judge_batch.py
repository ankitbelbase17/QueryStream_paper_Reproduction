#!/usr/bin/env python
"""Run the LLM-as-a-judge pass over a finished evaluation, via the OpenAI Batch API.

    export OPENAI_API_KEY=...
    python judge_batch.py runs/sb_subset --submit --wait 900
    python judge_batch.py runs/sb_subset --collect          # later, if it was slow

Writes `judge_summary.json` and rewrites `eval_raw_logs.jsonl` with
`judge_letter` / `judge_is_correct` added. The exact-match numbers in
`summary.json` are never overwritten -- the judge is reported alongside them,
not in place of them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from querystream.benchmarks import batch_judge as bj
from querystream.benchmarks import streamingbench as sb


def load_records(run_dir: str):
    path = os.path.join(run_dir, "eval_raw_logs.jsonl")
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()], path


def save_records(records, path: str) -> None:
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--submit", action="store_true", help="build and submit the batch")
    ap.add_argument("--collect", action="store_true", help="fetch an already-submitted batch")
    ap.add_argument("--batch-id", default=None)
    ap.add_argument("--wait", type=float, default=0.0, help="seconds to poll after submit")
    ap.add_argument("--completion-window", default="24h")
    a = ap.parse_args()

    records, jsonl_path = load_records(a.run_dir)
    print(f"[judge] {len(records)} records in {jsonl_path}")

    meta_path = os.path.join(a.run_dir, "batch_meta.json")
    batch_id = a.batch_id
    if a.submit:
        reqs = bj.build_requests(records, model=a.model)
        print(f"[judge] submitting {len(reqs)} requests to the Batch API "
              f"(model={a.model}, window={a.completion_window})")
        meta = bj.submit(reqs, a.run_dir, completion_window=a.completion_window)
        batch_id = meta["batch_id"]
        print(f"[judge] batch_id={batch_id} status={meta['status']}")
    elif batch_id is None and os.path.exists(meta_path):
        batch_id = json.load(open(meta_path))["batch_id"]
        print(f"[judge] resuming batch_id={batch_id}")

    if batch_id is None:
        print("[judge] nothing to do: pass --submit, or --batch-id", file=sys.stderr)
        return 2

    if a.wait > 0 or a.collect:
        batch = bj.poll(batch_id, timeout_s=max(a.wait, 1.0))
        if batch.status != "completed":
            print(f"[judge] batch is '{batch.status}' -- not collecting yet.\n"
                  f"        Re-run later:  python judge_batch.py {a.run_dir} --collect")
            return 1
        letters = bj.fetch(batch, a.run_dir)
        print(f"[judge] fetched {len(letters)} mappings")
        stats = bj.apply_to_records(records, letters)
        save_records(records, jsonl_path)

        exact = sb.aggregate(records)
        judged = [dict(r, is_correct=r.get("judge_is_correct", r.get("is_correct")))
                  for r in records if "judge_letter" in r]
        stats["exact_match_summary"] = {
            "acc": exact["overall"]["acc"], "n": exact["overall"]["total"]}
        stats["judge_summary_table"] = sb.aggregate(judged) if judged else None
        stats["batch_id"] = batch_id
        stats["model"] = a.model
        with open(os.path.join(a.run_dir, "judge_summary.json"), "w") as fh:
            json.dump(stats, fh, indent=2)

        print()
        print(f"  judged questions          {stats['judged']}")
        print(f"  exact-match accuracy      {stats['exact_acc_on_judged']:.2f}%")
        print(f"  judge accuracy            {stats['judge_acc']:.2f}%")
        print(f"  parser/judge agreement    {stats['parser_judge_agreement_pct']:.2f}% "
              f"({stats['n_disagreements']} disagreements)")
        print(f"  parse failures recovered  {stats['recovered_parse_failures']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
