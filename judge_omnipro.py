#!/usr/bin/env python3
"""Submit / poll / collect the OmniPro content judge over the OpenAI Batch API.

Only Event-Narration and Step-Instruction reach the judge, and only their
time-matched responses do (see omnipro_eval/judge.py for why). Everything else
is exact match and is already scored by `score_omnipro.py`.

    export OPENAI_API_KEY=...            # environment only; never written to disk
    python judge_omnipro.py runs/op_subset --submit
    python judge_omnipro.py runs/op_subset --collect --wait 1800

Submit and collect are separate steps because `completion_window="24h"` is the
SLA, not the latency. Nothing in the timing metrics or the seven exact-match
sub-tasks waits on the batch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omnipro_eval import judge as J
from omnipro_eval import metrics as M
from score_omnipro import read_records


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--collect", action="store_true")
    p.add_argument("--batch-id", default=None)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--tolerance", type=float, default=M.DEFAULT_TOLERANCE)
    p.add_argument("--wait", type=float, default=900.0,
                   help="seconds to poll before giving up (not an error)")
    a = p.parse_args()

    if not (a.submit or a.collect):
        p.error("pass --submit and/or --collect")

    meta_path = os.path.join(a.run_dir, "judge_batch_meta.json")

    if a.submit:
        records = [r for r in read_records(a.run_dir) if not r.get("error")]
        # Only matched responses need a verdict, so the matching runs first.
        scores = {s.sample_id: s for s in
                  (M.score_sample(r, tolerance=a.tolerance, judge=None)
                   for r in records)}
        reqs = J.build_requests(records, scores, model=a.model)
        if not reqs:
            print("[judge] nothing to judge: no time-matched response on either "
                  "open-ended task. The exact-match tasks are already complete.")
            return 0
        meta = J.submit(reqs, a.run_dir)
        print(f"[judge] submitted {meta['n_requests']} requests as "
              f"{meta['batch_id']} (status {meta['status']})")
        print(f"[judge] collect with: python judge_omnipro.py {a.run_dir} --collect")

    if a.collect:
        bid = a.batch_id
        if bid is None and os.path.exists(meta_path):
            bid = json.load(open(meta_path))["batch_id"]
        if bid is None:
            print("[judge] no batch id: pass --batch-id or run --submit first")
            return 1
        batch = J.poll(bid, timeout_s=a.wait)
        if batch.status != "completed":
            print(f"[judge] batch is {batch.status} after {a.wait:.0f}s. "
                  f"Nothing is lost -- re-run --collect later. Metrics for the "
                  f"two judged tasks stay withheld until it lands.")
            return 2
        verdicts = J.fetch(batch, a.run_dir)
        print(f"[judge] {len(verdicts)} verdicts written to "
              f"{os.path.join(a.run_dir, 'judge_output.jsonl')}")
        print(f"[judge] now run: python score_omnipro.py {a.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
