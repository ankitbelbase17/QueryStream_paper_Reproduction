#!/usr/bin/env python3
"""Offline scorer for an OVO-Bench run: merge shards, re-score, compare to the paper.

Every GPU-side quantity is in the JSONL, so scoring is a pure function of it and
can be run at any point, including mid-run.

    python score_ovo.py runs/ovo_20pct
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ovo_eval import dataset as ds
from ovo_eval import scoring as sc


def read_records(run_dir: str) -> List[dict]:
    """Union every shard JSONL; last write wins per call_id.

    Shards are disjoint by call, so an overlap means a shard was re-run and the
    later row supersedes rather than double-counting. A truncated final line
    from a killed worker is skipped.
    """
    by: Dict[str, dict] = {}
    paths = sorted(glob.glob(os.path.join(run_dir, "ovo_raw_logs*.jsonl")))
    n = 0
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
                n += 1
                by[r["call_id"]] = r
    print(f"[read] {len(paths)} shard file(s), {n} lines, {len(by)} unique calls",
          flush=True)
    return list(by.values())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    recs = read_records(a.run_dir)
    ok = [r for r in recs if not r.get("error")]
    errored = [r for r in recs if r.get("error")]
    agg = sc.aggregate(recs)

    kn = sum(r.get("tokens_kept", 0) for r in ok)
    kd = sum(r.get("tokens_dense", 0) for r in ok)

    cfg = {}
    for sp in sorted(glob.glob(os.path.join(a.run_dir, "summary*.json"))):
        try:
            cfg = json.load(open(sp)).get("config", {}) or cfg
        except Exception:
            pass

    out = {
        "run_dir": a.run_dir,
        "backbone": cfg.get("backbone"),
        "config": cfg,
        "n_calls": len(recs), "n_scored": agg["n_calls"], "n_errored": len(errored),
        "errors": [{"call_id": r.get("call_id"), "error": r.get("error")}
                   for r in errored][:50],
        "token_keep_rate": round(kn / kd, 6) if kd else 0.0,
        "tokens_kept": kn, "tokens_dense": kd,
        "metrics": agg,
        "paper": ds.PAPER,
    }
    dest = a.out or os.path.join(a.run_dir, "ovo_metrics.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2, default=str)
    _print(out)
    print(f"\nwrote {dest}")
    return 0


def _print(out: dict) -> None:
    m = out["metrics"]
    print("\n" + "=" * 92)
    print(f"OVO-BENCH  |  backbone {out.get('backbone')}  |  "
          f"calls {m['n_calls']}  keep {100*out['token_keep_rate']:.2f}%  "
          f"errors {out['n_errored']}")
    print("=" * 92)
    hdr = f"{'Model':<34}{'Keep':>7}"
    for t in ds.ALL_TASKS:
        hdr += f"{t:>6}"
    hdr += f"{'RT':>7}{'BW':>7}{'FW':>7}{'All':>7}"
    print(hdr)
    print("-" * len(hdr))

    row = f"{'QueryStream (ours, TimeChat)':<34}{100*out['token_keep_rate']:>6.1f}%"
    for t in ds.ALL_TASKS:
        v = m["per_task"].get(t, {}).get("acc")
        row += f"{v:>6.1f}" if v is not None else f"{'-':>6}"
    for c in ds.CATEGORY_ORDER:
        v = m["category"][c]
        row += f"{v:>7.1f}" if v is not None else f"{'-':>7}"
    row += f"{m['overall']:>7.1f}" if m["overall"] is not None else f"{'-':>7}"
    print(row)

    for name in ("TimeChat-Online-7B (100%)", "TimeChat-Online-7B (55.4%)",
                 "QueryStream-7B (52.9%)", "QueryStream-7B (20.0%)"):
        pr = ds.PAPER[name]
        r = f"  paper: {name:<27}{'':>7}"
        for t in ds.ALL_TASKS:
            r += f"{pr[t]:>6.1f}"
        for c in ds.CATEGORY_ORDER:
            r += f"{pr[c]:>7.1f}"
        r += f"{pr['overall']:>7.1f}"
        print(r)

    print()
    for t in ds.ALL_TASKS:
        d = m["per_task"].get(t)
        if not d:
            continue
        extra = (f"   strict-letter {d['strict_letter_acc']:.1f}"
                 if "strict_letter_acc" in d else "")
        print(f"  {t:<5} {ds.TASK_NAME[t]:<32} n={d['n']:>4} "
              f"acc={d['acc']:>6.2f}{extra}")
    if m["overall_is_partial"]:
        print("\n  Overall withheld: not all three categories are present.")
    print("=" * 92)


if __name__ == "__main__":
    raise SystemExit(main())
