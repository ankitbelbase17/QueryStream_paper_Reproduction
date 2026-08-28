#!/usr/bin/env python3
"""Run an already-built OmniPro judge request file synchronously, concurrently.

WHY THIS EXISTS
`judge_omnipro.py --submit` uses the Batch API, which is half price but whose
`completion_window="24h"` is an SLA, not a latency -- the op_full batch took
about nine hours. When the verdicts are needed now, this replays exactly the
same `judge_requests.jsonl` against the synchronous endpoint and writes a
`judge_output.jsonl` in the **Batch API's own output shape**, so
`score_omnipro.py` cannot tell the difference and no scoring code changes.

Identical prompts, identical model, identical rubric and >=3 threshold, so the
numbers stay comparable with any arm judged through the batch path.

A request that never succeeds is written as NO ROW AT ALL, which the scorer
treats as UNJUDGED. It is never written as a default or zero verdict -- that
would silently convert an API failure into a model failure.

    OPENAI_API_KEY=... python3 scripts/judge_sync.py runs/op_qwen3 --concurrency 16
"""
from __future__ import annotations

import argparse, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--requests", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("OPENAI_API_KEY not set (environment only; never a file)", file=sys.stderr)
        return 2

    req_path = a.requests or os.path.join(a.run_dir, "judge_requests.jsonl")
    out_path = a.out or os.path.join(a.run_dir, "judge_output.jsonl")
    reqs = [json.loads(l) for l in open(req_path) if l.strip()]
    print(f"[judge-sync] {len(reqs)} requests from {req_path}", flush=True)

    from openai import OpenAI
    client = OpenAI(api_key=key)

    lock = threading.Lock()
    done = {"n": 0, "fail": 0}

    def one(r):
        body = r["body"]
        for attempt in range(a.max_retries):
            try:
                resp = client.chat.completions.create(**body)
                txt = resp.choices[0].message.content
                row = {
                    "id": f"sync_{r['custom_id']}",
                    "custom_id": r["custom_id"],
                    "response": {"status_code": 200, "request_id": resp.id,
                                 "body": {"id": resp.id, "model": resp.model,
                                          "choices": [{"index": 0,
                                                       "message": {"role": "assistant",
                                                                   "content": txt}}]}},
                    "error": None,
                }
                with lock:
                    done["n"] += 1
                    if done["n"] % 100 == 0:
                        print(f"[judge-sync] {done['n']}/{len(reqs)}", flush=True)
                return row
            except Exception as e:                       # 429 / 5xx / transient
                if attempt == a.max_retries - 1:
                    with lock:
                        done["fail"] += 1
                    print(f"[judge-sync] FAILED {r['custom_id']}: "
                          f"{type(e).__name__}: {str(e)[:120]}", flush=True)
                    return None                          # -> unjudged, not zero
                time.sleep((2 ** attempt) + random.random())
        return None

    t0 = time.perf_counter()
    rows = []
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        for fut in as_completed([ex.submit(one, r) for r in reqs]):
            row = fut.result()
            if row is not None:
                rows.append(row)
    dt = time.perf_counter() - t0

    order = {r["custom_id"]: i for i, r in enumerate(reqs)}
    rows.sort(key=lambda x: order.get(x["custom_id"], 1 << 30))
    with open(out_path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    print(f"[judge-sync] wrote {len(rows)}/{len(reqs)} verdicts to {out_path} "
          f"in {dt:.0f}s ({done['fail']} unrecoverable -> left UNJUDGED)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
