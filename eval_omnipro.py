#!/usr/bin/env python3
"""OmniPro Online-mode evaluation for QueryStream (QDP + RTAR, vision only).

Streams each sample's video at 1 FPS through the unchanged QueryStream stack --
OpenCLIP-ViT-L/14 -> QDP -> RTAR -> Qwen2.5-VL-7B-Instruct -- and lets
`T_t = R_t AND D_t` decide when to speak. Writes one self-contained JSONL row
per sample plus a wall-clock profile, then scores everything it can offline and
leaves the two judged sub-tasks to `judge_omnipro.py`.

    # smoke test on a handful of samples
    python eval_omnipro.py --limit 8 --out runs/op_smoke --device cuda:0

    # a shard of a stratified subset
    python eval_omnipro.py --n-samples 600 --num-shards 16 --shard-id 3 \
        --out runs/op_subset --device cuda:0 --time-budget-sec 1500

Sharding is BY VIDEO and strided, so every worker holds a whole video's cache
(one decode+encode pass serves all its samples) and neighbouring videos -- which
have correlated durations -- do not pile onto one worker.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from omnipro_eval import dataset as ds
from omnipro_eval import metrics as M
from omnipro_eval.runner import OmniProRunner
from querystream.config import QueryStreamConfig
from querystream.eval_cache import build_video_pass
from querystream.prefetch import VideoPrefetcher
from querystream.profiling import ProfileRow, ProfileWriter


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    d = p.add_argument_group("dataset")
    d.add_argument("--benchmark-json", default=ds.DEFAULT_BENCHMARK_JSON)
    d.add_argument("--dataset-dir", default=ds.DEFAULT_DATASET_DIR)
    d.add_argument("--tasks", default="", help="comma-separated task filter")
    d.add_argument("--audio", default="none,helpful,required",
                   help="audio_dependency classes to keep")
    d.add_argument("--max-duration", type=float, default=None,
                   help="drop samples whose video is longer than this many "
                        "seconds. Bounds cost WITHOUT truncating a stream: a "
                        "truncated stream would drop its tail false positives "
                        "and inflate precision, so samples are excluded whole.")
    d.add_argument("--n-samples", type=int, default=0,
                   help="stratified subset size (0 = everything on disk)")
    d.add_argument("--strata", default="task_audio",
                   choices=["task_audio", "task", "audio"])
    d.add_argument("--prefer-short", type=float, default=0.0,
                   help="0..1 bias toward shorter videos within each stratum. "
                        "Buys sample count but skews toward early triggers; "
                        "recorded in the run metadata when non-zero.")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--limit", type=int, default=0, help="hard cap after selection")

    s = p.add_argument_group("sharding / resume")
    s.add_argument("--num-shards", type=int, default=1)
    s.add_argument("--shard-id", type=int, default=0)
    s.add_argument("--resume", action="store_true")
    s.add_argument("--time-budget-sec", type=float, default=0.0,
                   help="stop cleanly before the wall clock; 0 = no budget")

    m = p.add_argument_group("model / pruning")
    m.add_argument("--device", default="cuda:0")
    m.add_argument("--clip-model", default="ViT-L-14")
    m.add_argument("--clip-pretrained", default="openai")
    m.add_argument("--backbone", default="Qwen/Qwen2.5-VL-7B-Instruct")
    m.add_argument("--no-backbone", action="store_true",
                   help="gate-only dry run: RTAR timings, no generation")
    m.add_argument("--mask-space", default="score", choices=["score", "mask"])
    m.add_argument("--pool", default="mean", choices=["mean", "max", "min"])
    m.add_argument("--mask-resample-mode", default="any",
                   choices=["any", "all", "majority"])
    m.add_argument("--keep-mode", default="qdp", choices=["qdp", "dense"])
    m.add_argument("--gate-mode", default="full",
                   choices=["full", "relevance_only", "density_only"])
    m.add_argument("--relevance-mode", default="running_minmax",
                   choices=["raw", "running_minmax", "running_zscore", "logit_softmax"])
    m.add_argument("--tau-temp", type=float, default=0.90)
    m.add_argument("--tau-rel", type=float, default=0.60)
    m.add_argument("--tau-den", type=float, default=0.15)
    m.add_argument("--alpha", type=float, default=0.1)
    m.add_argument("--event-gap-s", type=float, default=M.DEFAULT_TOLERANCE,
                   help="Appendix A.4 event grouping gap. Defaults to OmniPro's "
                        "+-3 s matching tolerance: at a smaller gap one real "
                        "event emits several responses of which at most one can "
                        "match, and the rest are pure false positives.")
    m.add_argument("--max-emits", type=int, default=64)
    m.add_argument("--max-pixels-side", type=int, default=448)

    r = p.add_argument_group("runtime")
    r.add_argument("--clip-batch", type=int, default=8)
    r.add_argument("--vision-batch-planes", type=int, default=8)
    r.add_argument("--decode-workers", type=int, default=3)
    r.add_argument("--prefetch-depth", type=int, default=2)
    r.add_argument("--no-frame-traces", action="store_true",
                   help="omit the per-second gate traces from the JSONL")
    r.add_argument("--tolerance", type=float, default=M.DEFAULT_TOLERANCE)
    r.add_argument("--out", default="runs/omnipro")
    r.add_argument("--run-name", default="")
    return p


def make_config(a) -> QueryStreamConfig:
    cfg = QueryStreamConfig()
    cfg.fps = 1.0
    cfg.mask_space = a.mask_space
    cfg.score_pool_mode = a.pool
    cfg.mask_resample_mode = a.mask_resample_mode
    cfg.qdp.alpha = a.alpha
    cfg.qdp.tau_temp = a.tau_temp
    cfg.rtar.tau_rel = a.tau_rel
    cfg.rtar.tau_den = a.tau_den
    cfg.rtar.relevance_mode = a.relevance_mode
    cfg.rtar.event_gap_s = a.event_gap_s
    cfg.encoder.model_name = a.clip_model
    cfg.encoder.pretrained = a.clip_pretrained
    cfg.encoder.device = a.device
    cfg.backbone.model_id = a.backbone
    cfg.buffer.device = a.device
    return cfg


def main() -> int:
    a = build_argparser().parse_args()
    t_start = time.perf_counter()
    os.makedirs(a.out, exist_ok=True)

    shard_tag = "" if a.num_shards == 1 else f".shard{a.shard_id:02d}"
    jsonl_path = os.path.join(a.out, f"omnipro_raw_logs{shard_tag}.jsonl")
    summ_path = os.path.join(a.out, f"summary{shard_tag}.json")

    # ProfileWriter truncates its file, so a resumed pass would erase the
    # previous pass's timings. Each pass gets its own CSV instead; the report
    # globs `profile_summary*.csv` and frame-weights the OVERALL rows, so a run
    # finished across three resubmissions still reports one coherent profile.
    prof_path = os.path.join(a.out, f"profile_summary{shard_tag}.csv")
    if a.resume:
        n_pass = 1
        while os.path.exists(prof_path):
            prof_path = os.path.join(
                a.out, f"profile_summary{shard_tag}.pass{n_pass}.csv")
            n_pass += 1

    # ---- manifest ---------------------------------------------------- #
    samples, stats = ds.load_manifest(
        benchmark_json=a.benchmark_json, dataset_dir=a.dataset_dir,
        tasks=[t for t in a.tasks.split(",") if t] or None,
        audio=[x for x in a.audio.split(",") if x] or None,
        max_duration=a.max_duration,
    )
    print(f"[manifest] {stats.n_kept}/{stats.n_total} samples runnable "
          f"({stats.n_videos} videos, {stats.video_seconds / 3600:.1f} h, "
          f"{stats.gt_triggers} GT triggers)", flush=True)
    print(f"[manifest] dropped: missing_video={stats.dropped_missing_video} "
          f"task={stats.dropped_task} audio={stats.dropped_audio} "
          f"duration={stats.dropped_duration}", flush=True)

    select_info = {"mode": "all", "n": len(samples)}
    if a.n_samples and a.n_samples < len(samples):
        samples, select_info = ds.stratified_subset(
            samples, n_target=a.n_samples, seed=a.seed, strata=a.strata,
            prefer_short=a.prefer_short)
        print(f"[subset] {select_info['n_selected']} samples, "
              f"strata={a.strata}, seed={a.seed}, "
              f"prefer_short={a.prefer_short}", flush=True)
    if a.limit:
        samples = samples[: a.limit]

    # Shard BY VIDEO (strided) so a video's cache is never rebuilt on two
    # workers, and so correlated durations do not stack onto one worker.
    groups = ds.group_by_video(samples)
    if a.num_shards > 1:
        groups = groups[a.shard_id :: a.num_shards]
    n_shard_samples = sum(len(g) for _, g in groups)
    print(f"[shard] {a.shard_id + 1}/{a.num_shards}: {len(groups)} videos, "
          f"{n_shard_samples} samples", flush=True)

    done: set = set()
    if a.resume and os.path.exists(jsonl_path):
        with open(jsonl_path) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["sample_id"])
                except Exception:
                    pass
        print(f"[resume] {len(done)} samples already in {jsonl_path}", flush=True)

    # Work out what is left BEFORE loading 24 GB of weights. On a resubmitted
    # job most shards are already finished, and a worker with nothing to do
    # should exit in seconds rather than spend 90 s loading models to find out.
    todo = [(vid, ss) for vid, ss in groups
            if any(smp.sample_id not in done for smp in ss)]
    print(f"[todo] {len(todo)} videos still to run "
          f"({sum(len(g) for _, g in todo)} samples)", flush=True)

    # ---- models ------------------------------------------------------ #
    cfg = make_config(a)
    clip = qwen = runner = None
    if todo:
        from querystream.open_clip_wrapper import OpenCLIPWrapper
        t_load = time.perf_counter()
        clip = OpenCLIPWrapper(cfg.encoder)
        if not a.no_backbone:
            from querystream.backbone_factory import make_backbone
            qwen = make_backbone(cfg.backbone, device=a.device,
                                 max_pixels=a.max_pixels_side ** 2)
        print(f"[load] models ready in {time.perf_counter() - t_load:.1f}s",
              flush=True)
        runner = OmniProRunner(clip, qwen, cfg, gate_mode=a.gate_mode,
                               keep_mode=a.keep_mode, max_emits=a.max_emits,
                               keep_frame_traces=not a.no_frame_traces)
    else:
        print("[load] nothing left in this shard; skipping model load", flush=True)
    prof = ProfileWriter(prof_path)

    # ---- stream ------------------------------------------------------ #
    # Full duration, never truncated: cutting a stream at its last ground-truth
    # trigger would remove the tail's false positives and inflate precision.
    prefetch = VideoPrefetcher(
        todo,
        path_fn=lambda it: it[1][0].video_path,
        max_seconds_fn=lambda it: None,
        fps=cfg.fps, workers=a.decode_workers, depth=a.prefetch_depth,
    ) if todo else None

    records: List[dict] = []
    fh = open(jsonl_path, "a")
    n_done = n_err = 0
    budget_hit = False
    stream = iter(prefetch) if prefetch is not None else iter(())
    try:
        for vi, dec in enumerate(stream):
            vid, ss = dec.item
            if a.time_budget_sec and (time.perf_counter() - t_start) > a.time_budget_sec:
                budget_hit = True
                print(f"[budget] stopping at video {vi}/{len(todo)}", flush=True)
                break
            try:
                if not dec.ok:
                    raise RuntimeError(f"decode failed: {dec.error}")
                vp = build_video_pass(
                    ss[0].video_path, vid, clip=clip, qwen=qwen, cfg=cfg,
                    max_seconds=None, clip_batch=a.clip_batch,
                    vision_batch_planes=a.vision_batch_planes,
                    frames=dec.frames, meta=dec.meta,
                )
            except Exception as exc:
                print(f"  !! video {vid}: {type(exc).__name__}: {exc}", flush=True)
                for smp in ss:
                    if smp.sample_id in done:
                        continue
                    row = {"sample_id": smp.sample_id, "task": smp.task,
                           "video_id": vid, "audio_dependency": smp.audio_dependency,
                           "question": smp.question, "ground_truth": smp.ground_truth,
                           "emits": [], "n_emits": 0,
                           "error": f"video pass failed: {type(exc).__name__}: {exc}"}
                    fh.write(json.dumps(row) + "\n")
                    records.append(row)
                    n_err += 1
                fh.flush()
                continue

            prof.accumulate_video(vp.timings, vp.n_frames, len(vp.planes))
            prof.write(ProfileRow(
                scope="video", video_id=vid, n_frames=vp.n_frames,
                n_planes=len(vp.planes),
                decode_ms_per_frame=dec.decode_ms / max(vp.n_frames, 1),
                openclip_ms_per_frame=vp.timings.per_unit("openclip"),
                vision_tower_ms_per_frame=vp.timings.per_unit("vision_tower"),
            ))
            for smp in ss:
                if smp.sample_id in done:
                    continue
                out = runner.run(vp, smp)
                fh.write(json.dumps(out.record) + "\n")
                records.append(out.record)
                prof.accumulate_question(out.stopwatch, out.dense_tokens,
                                         out.kept_tokens, out.new_tokens)
                prof.write(ProfileRow(
                    scope="sample", video_id=vid, question_id=smp.sample_id,
                    task_code=smp.task, n_frames=out.frames_streamed,
                    n_planes=out.record.get("n_planes", 0),
                    qdp_rtar_ms_per_frame=out.stopwatch.per_unit("qdp_rtar"),
                    llm_prefill_ms=out.stopwatch.per_unit("llm_prefill"),
                    llm_decode_ms=out.stopwatch.per_unit("llm_decode"),
                    llm_decode_ms_per_token=(
                        out.stopwatch.ms("llm_decode") / out.new_tokens
                        if out.new_tokens else 0.0),
                    n_new_tokens=out.new_tokens,
                    visual_tokens_dense=out.record.get("tokens_dense_stream", 0),
                    visual_tokens_kept=out.record.get("tokens_kept_stream", 0),
                    avg_keep_rate=out.record.get("token_keep_rate", 0.0),
                    total_wall_clock_sec=out.record.get("wall_clock_sec", 0.0),
                ))
                n_done += 1
                if out.record.get("error"):
                    n_err += 1
            fh.flush()
            del vp
            if vi % 10 == 0:
                torch.cuda.empty_cache()
            print(f"[{vi + 1}/{len(todo)}] {vid}: {len(ss)} samples, "
                  f"{n_done} done, {time.perf_counter() - t_start:.0f}s",
                  flush=True)
    finally:
        fh.close()
        if prefetch is not None:
            prefetch.shutdown()

    overall_row = prof.close()

    # ---- offline scoring (everything except the two judged tasks) ---- #
    all_records = _read_jsonl(jsonl_path)
    scores = [M.score_sample(r, tolerance=a.tolerance, judge=None)
              for r in all_records if not r.get("error")]
    agg = M.aggregate(scores) if scores else {"overall": {}, "per_task": {}}
    slices = M.slice_report(scores, ds.build_slices()) if scores else {}

    summary = {
        "run_name": a.run_name or os.path.basename(a.out),
        "shard": {"id": a.shard_id, "n": a.num_shards},
        "samples_evaluated_this_pass": n_done,
        "samples_in_jsonl": len(all_records),
        "errors": n_err,
        "budget_hit": budget_hit,
        "wall_clock_sec": round(time.perf_counter() - t_start, 1),
        "manifest": stats.as_dict(),
        "selection": select_info,
        "tolerance_s": a.tolerance,
        "config": _config_dict(a, cfg),
        "metrics_unjudged": agg,
        "slices_unjudged": slices,
        "profile": overall_row.as_dict(),
        "decode_ms_per_frame_worker": round(prefetch.ms_per_frame, 3) if prefetch is not None else None,
        "decode_workers": a.decode_workers,
        "note": ("Event-Narration and Step-Instruction content is UNJUDGED in "
                 "this file: their content_acc / joint_f1 are withheld (None) "
                 "and only the *_lb lower bounds are populated. Run "
                 "judge_omnipro.py, then score_omnipro.py, for the complete "
                 "numbers. Timing metrics are exact here."),
    }
    with open(summ_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    ov = agg.get("overall", {})
    print("\n" + "=" * 72)
    print(f"samples {len(scores)}  emits {ov.get('n_emits', 0)}  "
          f"GT triggers {ov.get('n_gt', 0)}")
    print(f"TIME   P {ov.get('time_precision')}  R {ov.get('time_recall')}  "
          f"F1 {ov.get('time_f1')}   (macro {ov.get('macro_time_f1')})")
    print(f"JOINT  F1 lower bound {ov.get('joint_f1_lb')}   "
          f"(macro {ov.get('macro_joint_f1_lb')})  -- judge pending")
    print(f"token keep rate {_keep_rate(all_records):.4f}")
    print(f"wrote {jsonl_path}\n      {summ_path}\n      {prof_path}")
    print("=" * 72, flush=True)
    return 0


def _read_jsonl(path: str) -> List[dict]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    # last write wins, so a --resume rerun cannot double-count a sample
    seen: Dict[str, dict] = {}
    for r in out:
        seen[r.get("sample_id", id(r))] = r
    return list(seen.values())


def _keep_rate(records: List[dict]) -> float:
    k = sum(r.get("tokens_kept_stream", 0) for r in records)
    d = sum(r.get("tokens_dense_stream", 0) for r in records)
    return k / d if d else 0.0


def _config_dict(a, cfg) -> dict:
    return {
        "fps": cfg.fps, "encoder": f"{a.clip_model}/{a.clip_pretrained}",
        "backbone": a.backbone, "mask_space": a.mask_space, "pool": a.pool,
        "keep_mode": a.keep_mode, "gate_mode": a.gate_mode,
        "relevance_mode": a.relevance_mode,
        "alpha": a.alpha, "tau_temp": a.tau_temp, "tau_rel": a.tau_rel,
        "tau_den": a.tau_den, "event_gap_s": a.event_gap_s,
        "max_emits": a.max_emits, "max_pixels_side": a.max_pixels_side,
        "clip_batch": a.clip_batch, "decode_workers": a.decode_workers,
        "audio_encoder": None,
        "audio_note": ("QueryStream has no audio pathway. OmniPro samples "
                       "labelled audio_dependency=required are unsolvable by "
                       "construction and are reported as their own slice."),
    }


if __name__ == "__main__":
    raise SystemExit(main())
