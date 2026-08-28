#!/usr/bin/env python3
"""Sequential, resumable chain runner: eval -> score -> PDF, for a list of stages.

WHY THIS EXISTS
Every arm of this reproduction has the same three-step tail (stream, score,
report) and the same two operational hazards: a worker dies and the run must
resume without redoing finished work, and a run finishes but the PDF never gets
built because the shell went away. This chains the steps so a completed eval
always produces its PDF, and makes the whole chain restartable from any point.

RESUMABILITY IS AT TWO LEVELS
  * within a stage -- every worker passes --resume and skips ids already in its
    shard JSONL; a stage that is already complete costs one manifest read and
    exits without loading 20 GB of weights;
  * across stages -- a stage whose JSONL already holds `expect` rows is skipped
    entirely, so re-running the chain after a crash picks up at the first
    incomplete stage.

    python run_chain.py --stages ovo_laion            # one stage
    python run_chain.py --list                        # what is defined
    python run_chain.py --stages ovo_laion,omnipro_qwen3 --gpus 0,1,2,3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)


@dataclass
class Stage:
    name: str
    out: str                       # run directory, relative to REPO
    eval_script: str               # eval_ovo.py | eval_omnipro.py
    eval_args: List[str]
    expect: int                    # rows in the merged JSONL when complete
    jsonl_glob: str                # how to find the shard files
    id_field: str                  # dedupe key
    score_cmd: List[str]
    report_cmd: List[str]
    note: str = ""
    env: Dict[str, str] = field(default_factory=dict)


def _rows(run_dir: str, pattern: str, id_field: str) -> int:
    ids = set()
    for p in glob.glob(os.path.join(run_dir, pattern)):
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(json.loads(line)[id_field])
                except Exception:
                    pass
    return len(ids)


def build_stages() -> Dict[str, Stage]:
    ovo_common = [
        "--fraction", "1.0", "--calls-from", "runs/ovo_20pct",
        "--backbone", "wyccccc/TimeChatOnline-7B",
        "--decode-workers", "1", "--prefetch-depth", "1",
        "--clip-batch", "8", "--progress-every", "20",
    ]
    s: Dict[str, Stage] = {}

    # ---- OVO-Bench with the paper's cited LAION OpenCLIP weights -------- #
    s["ovo_laion"] = Stage(
        name="ovo_laion",
        out="runs/ovo_laion",
        eval_script="eval_ovo.py",
        eval_args=ovo_common + ["--clip-pretrained", "laion2b_s32b_b82k"],
        expect=584,
        jsonl_glob="ovo_raw_logs*.jsonl",
        id_field="call_id",
        score_cmd=["score_ovo.py", "runs/ovo_laion"],
        report_cmd=["make_ovo_report.py", "runs/ovo_laion"],
        note=("The paper cites 'OpenCLIP-ViT-L/14 (Cherti et al. 2023)' -- the "
              "LAION paper, whose ViT-L/14 tag is laion2b_s32b_b82k. Every "
              "earlier run loaded the OpenAI weights through open_clip. QDP's "
              "M_sem and RTAR's relevance both come from this encoder, so it "
              "decides WHICH tokens survive; paired to runs/ovo_20pct on the "
              "same 584 calls so the encoder is the only variable."),
    )

    # ---- OmniPro, full benchmark, on Qwen3-VL-8B-Instruct --------------- #
    s["omnipro_qwen3"] = Stage(
        name="omnipro_qwen3",
        out="runs/op_qwen3",
        eval_script="eval_omnipro.py",
        eval_args=[
            "--backbone", "Qwen/Qwen3-VL-8B-Instruct",
            # Everything else is pinned to runs/op_full's configuration so the
            # BACKBONE is the only variable between the two OmniPro arms:
            # same encoder weights, same thresholds, same 448-px cap, same
            # 3.0 s debounce, no --max-duration (all 2,700 samples evaluated).
            "--clip-model", "ViT-L-14", "--clip-pretrained", "openai",
            "--max-pixels-side", "448",
            "--decode-workers", "2", "--prefetch-depth", "1",
            "--clip-batch", "8",
        ],
        expect=2700,
        jsonl_glob="omnipro_raw_logs*.jsonl",
        id_field="sample_id",
        score_cmd=["score_omnipro.py", "runs/op_qwen3"],
        report_cmd=["make_omnipro_report.py", "runs/op_qwen3"],
        note=("Full 2,700-sample OmniPro on a NEWER backbone family. Qwen3-VL "
              "is not a drop-in for Qwen2.5-VL: patch_size 16 vs 14, text "
              "hidden 4096 vs 3584, interleaved M-ROPE, and deepstack visual "
              "features injected at 3 LLM layers. Still vision-only by "
              "instruction -- no audio encoder -- so audio_dependency=required "
              "samples remain a floor, reported not dropped."),
    )
    return s


def run_stage(st: Stage, gpus: List[str], *, budget: Optional[float],
              dry: bool = False) -> bool:
    out_dir = os.path.join(REPO, st.out)
    os.makedirs(out_dir, exist_ok=True)
    have = _rows(out_dir, st.jsonl_glob, st.id_field)
    print(f"\n{'='*74}\n[stage {st.name}] {have}/{st.expect} rows present")
    if st.note:
        print(f"  {st.note}")
    if have >= st.expect:
        print(f"[stage {st.name}] already complete; scoring + reporting only")
    elif dry:
        print(f"[stage {st.name}] DRY RUN -- would launch {len(gpus)} workers")
        return True
    else:
        env = dict(os.environ)
        env.update({
            "HF_HOME": env.get("HF_HOME", "/iopsstor/scratch/cscs/dbartaula/hf_cache"),
            "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "OPENCV_FFMPEG_CAPTURE_OPTIONS": "timeout;15000000",
            "PYTHONPATH": REPO + ":" + env.get("PYTHONPATH", ""),
        })
        env.update(st.env)
        procs = []
        for i, g in enumerate(gpus):
            cmd = [sys.executable, os.path.join(REPO, st.eval_script),
                   "--out", st.out, "--run-name", st.name, "--device", "cuda:0",
                   "--num-shards", str(len(gpus)), "--shard-id", str(i),
                   "--resume"] + st.eval_args
            if budget:
                cmd += ["--time-budget-sec", str(budget)]
            e = dict(env, CUDA_VISIBLE_DEVICES=g)
            log = open(os.path.join(out_dir, f"worker{i:02d}.log"), "a")
            procs.append((subprocess.Popen(cmd, env=e, stdout=log, stderr=log,
                                           cwd=REPO), log, i, g))
            print(f"  worker {i} -> gpu {g}")
        t0 = time.time()
        for p, log, i, g in procs:
            rc = p.wait()
            log.close()
            print(f"  worker {i} (gpu {g}) exit={rc} after {time.time()-t0:.0f}s")
        have = _rows(out_dir, st.jsonl_glob, st.id_field)
        print(f"[stage {st.name}] {have}/{st.expect} rows after the pass")

    # ---- score, then report; both are pure functions of the JSONL ------ #
    for label, cmd in (("score", st.score_cmd), ("report", st.report_cmd)):
        full = [sys.executable, os.path.join(REPO, cmd[0])] + cmd[1:]
        print(f"[stage {st.name}] {label}: {' '.join(cmd)}")
        r = subprocess.run(full, cwd=REPO, capture_output=True, text=True)
        tail = (r.stdout or "").strip().splitlines()[-3:]
        for ln in tail:
            print("   " + ln)
        if r.returncode != 0:
            print(f"   !! {label} failed rc={r.returncode}")
            print((r.stderr or "")[-600:])
            if label == "score":
                return False
    return have >= st.expect


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="", help="comma-separated stage names, in order")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--time-budget-sec", type=float, default=0.0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--progress", action="store_true",
                    help="print one integer -- total deduped rows across the "
                         "requested stages -- and exit. The supervisor loop "
                         "uses it to tell a resumed round that made progress "
                         "from one that stalled.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--continue-on-incomplete", action="store_true",
                    help="move to the next stage even if this one did not reach "
                         "`expect` rows (default: stop, so a truncated stage is "
                         "obvious rather than silently followed by the next)")
    a = ap.parse_args()

    stages = build_stages()
    if a.list or not a.stages:
        print("stages:")
        for k, v in stages.items():
            have = _rows(os.path.join(REPO, v.out), v.jsonl_glob, v.id_field)
            print(f"  {k:<16} {have}/{v.expect}  -> {v.out}")
        return 0

    want = [x for x in a.stages.split(",") if x]
    if a.progress:
        total = 0
        for name in want:
            v = stages.get(name)
            if v is not None:
                total += _rows(os.path.join(REPO, v.out), v.jsonl_glob, v.id_field)
        print(total)
        return 0

    gpus = [g for g in a.gpus.split(",") if g != ""]
    ok_all = True
    for name in want:
        if name not in stages:
            print(f"unknown stage {name!r}; --list to see them")
            return 2
        ok = run_stage(stages[name], gpus, budget=a.time_budget_sec or None,
                       dry=a.dry_run)
        ok_all &= ok
        if not ok and not a.continue_on_incomplete:
            print(f"\n[chain] stage {name} incomplete; stopping. Re-run the same "
                  f"command to resume it.")
            return 1
    print(f"\n[chain] done; all stages complete = {ok_all}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
