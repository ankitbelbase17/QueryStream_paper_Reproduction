#!/usr/bin/env python3
"""OVO-Bench evaluation for QueryStream on TimeChat-Online-7B (paper Table 2).

This is the arm that closes the backbone gap. Milestones 3-5 ran QDP+RTAR on
stock Qwen2.5-VL-7B-Instruct; the paper instead swaps QueryStream into
TimeChat-Online-7B, replacing its query-agnostic DTD module with QDP+RTAR.
TimeChat-Online is architecturally `Qwen2_5_VLForConditionalGeneration`, so the
existing wrapper loads it unchanged and only `--backbone` differs.

    # smoke, login node
    python eval_ovo.py --limit 12 --out runs/ovo_smoke --device cuda:1

    # 20% stratified subset, sharded
    python eval_ovo.py --fraction 0.2 --num-shards 12 --shard-id 3 \
        --out runs/ovo_20pct --device cuda:0 --time-budget-sec 1500

Sharding is strided over CALLS, not samples: every OVO-Bench clip is used by
exactly one call, so there is no per-video cache to preserve and striding just
balances the very uneven clip durations (CRR averages 753 s, REC 64 s).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from ovo_eval import dataset as ds
from ovo_eval import scoring
from ovo_eval.runner import OVORunner
from querystream.config import QueryStreamConfig
from querystream.eval_cache import build_video_pass
from querystream.prefetch import VideoPrefetcher
from querystream.profiling import ProfileRow, ProfileWriter

TIMECHAT = "wyccccc/TimeChatOnline-7B"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    d = p.add_argument_group("dataset")
    d.add_argument("--anno", default=ds.DEFAULT_ANNO)
    d.add_argument("--video-dir", default=ds.DEFAULT_VIDEOS)
    d.add_argument("--tasks", default="", help="comma-separated task filter")
    d.add_argument("--fraction", type=float, default=1.0,
                   help="fraction of SAMPLES per task to keep (probes stay together)")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--limit", type=int, default=0, help="hard cap on calls, after selection")
    d.add_argument("--calls-from", default=None,
                   help="restrict to the call_ids present in another run's "
                        "JSONL. The only reliable way to make a control EXACTLY "
                        "paired with a baseline: it takes the sample set from "
                        "the data rather than re-deriving it from a seed, so "
                        "changes to the selection logic cannot unpair them.")

    s = p.add_argument_group("sharding / resume")
    s.add_argument("--num-shards", type=int, default=1)
    s.add_argument("--shard-id", type=int, default=0)
    s.add_argument("--resume", action="store_true")
    s.add_argument("--time-budget-sec", type=float, default=0.0)

    m = p.add_argument_group("model / pruning")
    m.add_argument("--device", default="cuda:0")
    m.add_argument("--backbone", default=TIMECHAT,
                   help="default is the paper's TimeChat-Online-7B")
    m.add_argument("--clip-model", default="ViT-L-14")
    m.add_argument("--clip-pretrained", default="openai")
    m.add_argument("--no-backbone", action="store_true")
    m.add_argument("--mask-space", default="score", choices=["score", "mask"])
    m.add_argument("--pool", default="mean", choices=["mean", "max", "min"])
    m.add_argument("--keep-mode", default="qdp", choices=["qdp", "dense"])
    m.add_argument("--gate-mode", default="full",
                   choices=["full", "relevance_only", "density_only"])
    m.add_argument("--relevance-mode", default="running_minmax",
                   choices=["raw", "running_minmax", "running_zscore", "logit_softmax"])
    m.add_argument("--tau-temp", type=float, default=0.90)
    m.add_argument("--tau-rel", type=float, default=0.60)
    m.add_argument("--tau-den", type=float, default=0.15)
    m.add_argument("--alpha", type=float, default=0.1)
    m.add_argument("--max-pixels-side", type=int, default=448)

    r = p.add_argument_group("runtime")
    r.add_argument("--clip-batch", type=int, default=8)
    r.add_argument("--vision-batch-planes", type=int, default=8)
    # Decode residency is the thing that OOM-killed StreamingBench workers, and
    # OVO clips run to 1920x1440. One decode thread and depth 1 caps residency
    # at ~2-3 clips; raise only on short-clip task subsets.
    r.add_argument("--decode-workers", type=int, default=1)
    r.add_argument("--prefetch-depth", type=int, default=1)
    r.add_argument("--max-clip-seconds", type=float, default=0.0,
                   help="skip clips longer than this many seconds (0 = no cap). "
                        "Recorded in the manifest stats; excludes whole calls "
                        "rather than truncating, which would change what the "
                        "model saw relative to the benchmark's own clip.")
    r.add_argument("--frame-traces", action="store_true")
    r.add_argument("--progress-every", type=int, default=10)
    r.add_argument("--out", default="runs/ovo")
    r.add_argument("--run-name", default="")
    return p


def make_config(a) -> QueryStreamConfig:
    cfg = QueryStreamConfig()
    cfg.fps = 1.0
    cfg.mask_space = a.mask_space
    cfg.score_pool_mode = a.pool
    cfg.qdp.alpha = a.alpha
    cfg.qdp.tau_temp = a.tau_temp
    cfg.rtar.tau_rel = a.tau_rel
    cfg.rtar.tau_den = a.tau_den
    cfg.rtar.relevance_mode = a.relevance_mode
    cfg.encoder.model_name = a.clip_model
    cfg.encoder.pretrained = a.clip_pretrained
    cfg.encoder.device = a.device
    cfg.backbone.model_id = a.backbone
    cfg.buffer.device = a.device
    return cfg


def main() -> int:
    a = build_argparser().parse_args()
    t0 = time.perf_counter()
    os.makedirs(a.out, exist_ok=True)
    tag = "" if a.num_shards == 1 else f".shard{a.shard_id:02d}"
    jsonl_path = os.path.join(a.out, f"ovo_raw_logs{tag}.jsonl")
    summ_path = os.path.join(a.out, f"summary{tag}.json")
    prof_path = os.path.join(a.out, f"profile_summary{tag}.csv")
    if a.resume:
        n = 1
        while os.path.exists(prof_path):
            prof_path = os.path.join(a.out, f"profile_summary{tag}.pass{n}.csv")
            n += 1

    calls, st = ds.load_manifest(anno=a.anno, video_dir=a.video_dir,
                                 tasks=[t for t in a.tasks.split(",") if t] or None)
    print(f"[manifest] {st.n_calls} calls / {st.n_samples_kept} samples, "
          f"{st.video_seconds/3600:.1f} h, missing_video={st.dropped_missing_video}",
          flush=True)

    sel = {"mode": "full", "n_calls": len(calls)}
    if a.fraction < 1.0:
        calls, sel = ds.subset(calls, a.fraction, seed=a.seed)
        print(f"[subset] {sel['n_calls']} calls / {sel['n_samples']} samples, "
              f"{sel['video_seconds']/3600:.1f} h", flush=True)
    if a.max_clip_seconds:
        before = len(calls)
        calls = [c for c in calls if c.realtime <= a.max_clip_seconds]
        sel["dropped_long_clips"] = before - len(calls)
        print(f"[filter] dropped {before-len(calls)} clips over "
              f"{a.max_clip_seconds:.0f}s", flush=True)
    if a.calls_from:
        import glob as _glob
        want = set()
        for p_ in _glob.glob(os.path.join(a.calls_from, "ovo_raw_logs*.jsonl")):
            with open(p_) as fh_:
                for line in fh_:
                    line = line.strip()
                    if line:
                        try:
                            want.add(json.loads(line)["call_id"])
                        except Exception:
                            pass
        before = len(calls)
        calls = [c for c in calls if c.call_id in want]
        sel = dict(sel, paired_with=a.calls_from, n_calls=len(calls),
                   dropped_unpaired=before - len(calls))
        print(f"[paired] {len(calls)} calls matched against {a.calls_from} "
              f"({len(want)} ids there)", flush=True)
    if a.limit:
        calls = calls[: a.limit]

    if a.num_shards > 1:
        calls = calls[a.shard_id :: a.num_shards]
    print(f"[shard] {a.shard_id+1}/{a.num_shards}: {len(calls)} calls", flush=True)

    done = set()
    if a.resume and os.path.exists(jsonl_path):
        with open(jsonl_path) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["call_id"])
                except Exception:
                    pass
        print(f"[resume] {len(done)} calls already logged", flush=True)

    todo = [c for c in calls if c.call_id not in done]
    print(f"[todo] {len(todo)} calls remaining", flush=True)

    cfg = make_config(a)
    clip = qwen = runner = None
    if todo:
        from querystream.open_clip_wrapper import OpenCLIPWrapper
        tl = time.perf_counter()
        clip = OpenCLIPWrapper(cfg.encoder)
        if not a.no_backbone:
            from querystream.backbone_factory import make_backbone
            qwen = make_backbone(cfg.backbone, device=a.device,
                                 max_pixels=a.max_pixels_side ** 2)
            print(f"[load] backbone = {a.backbone}", flush=True)
        print(f"[load] ready in {time.perf_counter()-tl:.1f}s", flush=True)
        runner = OVORunner(clip, qwen, cfg, keep_mode=a.keep_mode,
                           gate_mode=a.gate_mode, log_frame_traces=a.frame_traces)
    else:
        print("[load] nothing to do; skipping model load", flush=True)

    prof = ProfileWriter(prof_path)
    records: List[dict] = []
    fh = open(jsonl_path, "a")
    n_err = 0
    budget_hit = False
    pf = VideoPrefetcher(todo, path_fn=lambda c: c.video_path,
                         max_seconds_fn=lambda c: None, fps=cfg.fps,
                         workers=a.decode_workers, depth=a.prefetch_depth) if todo else None
    try:
        for i, dv in enumerate(pf or ()):
            c = dv.item
            if a.time_budget_sec and (time.perf_counter() - t0) > a.time_budget_sec:
                budget_hit = True
                print(f"[budget] stopping after {i} calls", flush=True)
                pf.shutdown()
                break
            try:
                if not dv.ok:
                    raise RuntimeError(dv.error or "decode failed")
                vp = build_video_pass(c.video_path, c.call_id, clip=clip, qwen=qwen,
                                      cfg=cfg, max_seconds=None,
                                      clip_batch=a.clip_batch,
                                      vision_batch_planes=a.vision_batch_planes,
                                      frames=dv.frames, meta=dv.meta)
                vp.timings.add("decode", dv.decode_ms, n=int(dv.frames.shape[0]))
                del dv
            except Exception as exc:
                print(f"  !! {c.call_id} ({c.task}): {type(exc).__name__}: {exc}",
                      flush=True)
                row = {"call_id": c.call_id, "sample_id": c.sample_id, "task": c.task,
                       "category": c.category, "ground_truth": c.ground_truth,
                       "response": None, "score": 0,
                       "error": f"video pass failed: {type(exc).__name__}: {exc}"}
                fh.write(json.dumps(row) + "\n"); fh.flush()
                records.append(row); n_err += 1
                continue

            out = runner.answer(vp, c)
            fh.write(json.dumps(out.record) + "\n")
            records.append(out.record)
            if out.record.get("error"):
                n_err += 1
            prof.accumulate_video(vp.timings, vp.n_frames, len(vp.planes))
            prof.accumulate_question(out.stopwatch, out.dense_tokens,
                                     out.kept_tokens, out.new_tokens)
            prof.write(ProfileRow(
                scope="call", video_id=c.call_id, question_id=c.call_id,
                task_code=c.task, n_frames=out.frames,
                n_planes=out.record.get("n_planes", 0),
                decode_ms_per_frame=vp.timings.per_unit("decode"),
                openclip_ms_per_frame=vp.timings.per_unit("openclip"),
                vision_tower_ms_per_frame=vp.timings.per_unit("vision_tower"),
                qdp_rtar_ms_per_frame=out.stopwatch.per_unit("qdp_rtar"),
                llm_prefill_ms=out.stopwatch.ms("llm_prefill"),
                llm_decode_ms=out.stopwatch.ms("llm_decode"),
                n_new_tokens=out.new_tokens,
                visual_tokens_dense=out.dense_tokens,
                visual_tokens_kept=out.kept_tokens,
                avg_keep_rate=out.record.get("keep_rate", 0.0),
                total_wall_clock_sec=out.record.get("wall_clock_sec", 0.0)))
            del vp
            if (i + 1) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Flush EVERY call. Buffering the JSONL until the loop ends means a
            # killed worker loses everything it computed, which is exactly the
            # failure --resume exists to absorb -- and OOM kills have taken
            # workers in this project more than once (CLAUDE.md 16). One
            # fsync-free flush per call is far cheaper than recomputing a shard.
            fh.flush()

            if (i + 1) % a.progress_every == 0:
                agg = scoring.aggregate(records)
                cats = " ".join(f"{k}={v}" for k, v in agg["category"].items() if v is not None)
                print(f"[{i+1}/{len(todo)}] {cats}  "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
    finally:
        fh.close()
        if pf is not None:
            pf.shutdown()
    overall_row = prof.close()

    all_rows = _read(jsonl_path)
    agg = scoring.aggregate(all_rows)
    kn = sum(r.get("tokens_kept", 0) for r in all_rows)
    kd = sum(r.get("tokens_dense", 0) for r in all_rows)
    summary = {
        "run_name": a.run_name or os.path.basename(a.out),
        "shard": {"id": a.shard_id, "n": a.num_shards},
        "backbone": a.backbone,
        "calls_this_pass": len(records), "calls_in_jsonl": len(all_rows),
        "errors": n_err, "budget_hit": budget_hit,
        "wall_clock_sec": round(time.perf_counter() - t0, 1),
        "manifest": st.as_dict(), "selection": sel,
        "token_keep_rate": round(kn / kd, 6) if kd else 0.0,
        "metrics": agg,
        "config": {
            "fps": cfg.fps, "encoder": f"{a.clip_model}/{a.clip_pretrained}",
            "backbone": a.backbone, "mask_space": a.mask_space, "pool": a.pool,
            "keep_mode": a.keep_mode, "gate_mode": a.gate_mode,
            "relevance_mode": a.relevance_mode, "alpha": a.alpha,
            "tau_temp": a.tau_temp, "tau_rel": a.tau_rel, "tau_den": a.tau_den,
            "max_pixels_side": a.max_pixels_side, "clip_batch": a.clip_batch,
            "decode_workers": a.decode_workers,
        },
        "profile": overall_row.as_dict(),
    }
    with open(summ_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"backbone {a.backbone}")
    print(f"calls {agg['n_calls']}  keep {100*summary['token_keep_rate']:.2f}%  errors {n_err}")
    for c, v in agg["category"].items():
        if v is not None:
            print(f"  {ds.CATEGORY_TITLE[c]:<30} {v:.2f}")
    print(f"  {'Overall':<30} {agg['overall'] if agg['overall'] is not None else 'n/a'}")
    print(f"wrote {jsonl_path}\n      {summ_path}")
    print("=" * 70, flush=True)
    return 0


def _read(path: str) -> List[dict]:
    seen: Dict[str, dict] = {}
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                seen[r["call_id"]] = r
            except Exception:
                pass
    return list(seen.values())


if __name__ == "__main__":
    raise SystemExit(main())
