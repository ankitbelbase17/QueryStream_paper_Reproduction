#!/usr/bin/env python
"""QueryStream evaluation harness -- StreamingBench (paper Table 1).

    python eval_streaming.py \
        --anno  .../streamingbench/questions_real.json \
        --video-dir .../streamingbench/videos \
        --output-dir runs/sb_full \
        --mask-space score --pool mean --relevance-mode running_minmax

Outputs, all under --output-dir:
    eval_raw_logs.jsonl   one row per question: query, ground truth, prediction,
                          per-frame keep rates, raw + calibrated relevance,
                          trigger timestamps, correctness
    profile_summary.csv   millisecond timings per video and per question
    summary.json          per-subtask accuracy in the paper's Table 1 layout
    config.json           the exact configuration the run used

Per-video work is done ONCE and shared by that video's questions -- decoding,
OpenCLIP and Qwen's vision tower do not depend on the query. See
`querystream/eval_cache.py` for why the DSH is cacheable too.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set

import torch

from querystream.benchmarks import streamingbench as sb
from querystream.benchmarks.judge import build_judge
from querystream.config import QueryStreamConfig
from querystream.eval_cache import build_video_pass
from querystream.eval_runner import StreamingBenchRunner
from querystream.prefetch import VideoPrefetcher
from querystream.profiling import ProfileRow, ProfileWriter

DEFAULT_ANNO = (
    "/iopsstor/scratch/cscs/dbartaula/eval_7_10_8_19/simplestreamingeval/"
    "data/streamingbench/questions_real.json"
)
DEFAULT_VIDEOS = (
    "/iopsstor/scratch/cscs/dbartaula/eval_7_10_8_19/simplestreamingeval/"
    "data/streamingbench/videos"
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    d = p.add_argument_group("data")
    d.add_argument("--anno", default=DEFAULT_ANNO)
    d.add_argument("--video-dir", default=DEFAULT_VIDEOS)
    d.add_argument("--output-dir", required=True)
    d.add_argument("--limit-videos", type=int, default=None)
    d.add_argument("--limit-questions", type=int, default=None)
    d.add_argument("--tasks", default=None,
                   help="comma-separated subtask codes, e.g. OP,CR,CT")
    d.add_argument("--shuffle-seed", type=int, default=None,
                   help="shuffle videos before --limit-videos, for an unbiased subset")
    d.add_argument("--resume", action="store_true",
                   help="skip question_ids already present in eval_raw_logs.jsonl")
    d.add_argument("--num-shards", type=int, default=1,
                   help="split the video list across N workers (one GPU each)")
    d.add_argument("--shard-id", type=int, default=0,
                   help="0-based index of this worker's shard")

    m = p.add_argument_group("model")
    m.add_argument("--device", default="cuda:0")
    m.add_argument("--cache-device", default=None,
                   help="where the per-video visual cache lives (default: --device)")
    m.add_argument("--clip-model", default="ViT-L-14")
    m.add_argument("--clip-pretrained", default="openai")
    m.add_argument("--backbone", default="Qwen/Qwen2.5-VL-7B-Instruct")
    m.add_argument("--max-pixels", type=int, default=448 * 448,
                   help="cap on Qwen smart_resize; fixes the keep-rate denominator")
    m.add_argument("--max-new-tokens", type=int, default=16)
    m.add_argument("--no-qwen", action="store_true",
                   help="pruning statistics only, no generation (geometry smoke run)")

    q = p.add_argument_group("QueryStream")
    q.add_argument("--mask-space", choices=["score", "mask"], default="score")
    q.add_argument("--pool", choices=["mean", "max", "min"], default="mean",
                   help="score-space pooling; mean ~48%% keep, max ~30%% keep")
    q.add_argument("--mask-resample-mode", choices=["any", "all", "majority", "nearest"],
                   default="any", help="only used with --mask-space mask")
    q.add_argument("--relevance-mode",
                   choices=["raw", "running_minmax", "logit_softmax", "running_zscore"],
                   default="running_minmax")
    q.add_argument("--gate-mode", choices=["full", "relevance_only", "density_only"],
                   default="full")
    q.add_argument("--alpha", type=float, default=0.1)
    q.add_argument("--tau-temp", type=float, default=0.90)
    q.add_argument("--tau-rel", type=float, default=0.60)
    q.add_argument("--tau-den", type=float, default=0.15)
    q.add_argument("--fps", type=float, default=1.0)
    q.add_argument("--keep-mode", choices=["qdp", "dense"], default="qdp",
                   help="'dense' disables pruning entirely -- the 100%% control row")

    r = p.add_argument_group("runtime")
    r.add_argument("--clip-batch", type=int, default=1,
                   help="OpenCLIP minibatch. 1 == what a 1-FPS stream actually pays "
                        "(11.1 ms/frame) and reproduces the streaming pipeline's masks "
                        "bitwise; 32 is 12.7x faster (0.88 ms/frame) but flips ~0.3%% of "
                        "mask decisions and makes openclip_ms_per_frame unrepresentative")
    r.add_argument("--vision-batch-planes", type=int, default=8)
    r.add_argument("--decode-workers", type=int, default=1,
                   help="threads decoding UPCOMING videos while the GPU works. "
                        "Decode is 46 ms/frame vs ~17-27 ms of GPU, so >1 hides it")
    r.add_argument("--prefetch-depth", type=int, default=2,
                   help="decoded videos allowed to queue; one 1440-frame 1080p "
                        "video is ~8.9 GB of uint8, so this bounds RAM not CPU")
    r.add_argument("--time-budget-sec", type=float, default=None,
                   help="stop starting new videos after this many seconds and "
                        "write a complete summary of what finished")
    r.add_argument("--judge", choices=["none", "openai"], default="none",
                   help="LLM fallback for UNPARSEABLE responses only; needs OPENAI_API_KEY")
    r.add_argument("--judge-model", default="gpt-4o-mini")
    r.add_argument("--progress-every", type=int, default=5, help="videos between summaries")
    r.add_argument("--dry-run-assert-keep", default=None,
                   help="LO,HI -- exit nonzero unless the final keep rate lands inside")
    return p


def make_config(a: argparse.Namespace) -> QueryStreamConfig:
    cfg = QueryStreamConfig()
    cfg.fps = a.fps
    cfg.encoder.model_name = a.clip_model
    cfg.encoder.pretrained = a.clip_pretrained
    cfg.encoder.device = a.device
    cfg.backbone.model_id = a.backbone
    cfg.qdp.alpha = a.alpha
    cfg.qdp.tau_temp = a.tau_temp
    cfg.rtar.tau_rel = a.tau_rel
    cfg.rtar.tau_den = a.tau_den
    cfg.rtar.relevance_mode = a.relevance_mode
    cfg.buffer.device = a.cache_device or a.device
    cfg.mask_space = a.mask_space
    cfg.score_pool_mode = a.pool
    cfg.mask_resample_mode = a.mask_resample_mode
    return cfg.validate()


def load_done(path: str) -> Set[str]:
    done: Set[str] = set()
    if not os.path.exists(path):
        return done
    with open(path) as fh:
        for line in fh:
            try:
                done.add(json.loads(line)["question_id"])
            except Exception:
                continue
    return done


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    os.makedirs(a.output_dir, exist_ok=True)
    jsonl_path = os.path.join(a.output_dir, "eval_raw_logs.jsonl")
    csv_path = os.path.join(a.output_dir, "profile_summary.csv")

    cfg = make_config(a)
    done = load_done(jsonl_path) if a.resume else set()
    if done:
        print(f"[resume] {len(done)} questions already logged; they will be skipped")

    with open(os.path.join(a.output_dir, "config.json"), "w") as fh:
        json.dump({"argv": sys.argv, "args": vars(a), "config": cfg.to_dict()},
                  fh, indent=2, default=str)

    # ---- data ---------------------------------------------------------- #
    items = sb.load_streamingbench(
        a.anno, a.video_dir,
        limit_videos=a.limit_videos,
        limit_questions=a.limit_questions,
        task_filter=a.tasks.split(",") if a.tasks else None,
        shuffle_seed=a.shuffle_seed,
    )
    if a.num_shards > 1:
        # Stride-based, not block-based: consecutive videos have similar lengths
        # in this manifest, so contiguous blocks give badly unbalanced shards.
        total = len(items)
        items = items[a.shard_id :: a.num_shards]
        print(f"[shard] {a.shard_id + 1}/{a.num_shards}: "
              f"{len(items)} of {total} videos")
    n_q = sum(len(i.questions) for i in items)
    print(f"[data] {len(items)} videos / {n_q} questions from {a.anno}")
    if not items:
        print("[data] nothing to evaluate", file=sys.stderr)
        return 2

    # ---- models -------------------------------------------------------- #
    from querystream.open_clip_wrapper import OpenCLIPWrapper

    t0 = time.perf_counter()
    print(f"[load] OpenCLIP {a.clip_model}/{a.clip_pretrained} on {a.device} ...", flush=True)
    clip = OpenCLIPWrapper(cfg.encoder)
    print(f"[load] OpenCLIP ready in {time.perf_counter() - t0:.1f}s, grid {clip.grid_hw}")

    qwen = None
    if not a.no_qwen:
        from querystream.qwen_wrapper import QwenVLWrapper

        t0 = time.perf_counter()
        print(f"[load] {a.backbone} on {a.device} ...", flush=True)
        qwen = QwenVLWrapper(cfg.backbone, device=a.device, max_pixels=a.max_pixels)
        print(f"[load] backbone ready in {time.perf_counter() - t0:.1f}s")

    judge = build_judge(a.judge, a.judge_model)
    runner = StreamingBenchRunner(
        clip, qwen, cfg,
        gate_mode=a.gate_mode,
        max_new_tokens=a.max_new_tokens,
        keep_mode=a.keep_mode,
        judge=judge,
    )
    if a.keep_mode == "dense":
        print("[mode] keep-mode=dense -- pruning disabled (100% control row)")

    # ---- loop ---------------------------------------------------------- #
    prof = ProfileWriter(csv_path)
    records: List[Dict[str, Any]] = []
    if a.resume and os.path.exists(jsonl_path):
        with open(jsonl_path) as fh:
            for line in fh:
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue

    fh_jsonl = open(jsonl_path, "a" if a.resume else "w")
    t_run = time.perf_counter()
    n_done = 0

    pending = [it for it in items
               if any(q.question_id not in done for q in it.questions)]
    prefetch = VideoPrefetcher(
        pending,
        path_fn=lambda it: it.video_path,
        max_seconds_fn=lambda it: max(
            (q.timestamp_s for q in it.questions if q.question_id not in done),
            default=0),
        fps=cfg.fps, workers=a.decode_workers, depth=a.prefetch_depth,
    )
    print(f"[decode] {a.decode_workers} prefetch threads, queue depth {a.prefetch_depth}")
    if a.time_budget_sec:
        print(f"[budget] will stop starting new videos after {a.time_budget_sec:.0f}s")
    budget_hit = False
    vi = -1

    try:
        for vi, dv in enumerate(prefetch):
            item = dv.item
            todo = [q for q in item.questions if q.question_id not in done]
            if not todo:
                continue
            if a.time_budget_sec and (time.perf_counter() - t_run) > a.time_budget_sec:
                budget_hit = True
                print(f"[budget] {a.time_budget_sec:.0f}s reached after {vi} videos; "
                      f"stopping cleanly")
                prefetch.shutdown()
                break
            t_vid = time.perf_counter()
            try:
                if not dv.ok:
                    raise RuntimeError(dv.error or "decode failed")
                vp = build_video_pass(
                    item.video_path, item.video_id,
                    clip=clip, qwen=qwen, cfg=cfg,
                    max_seconds=max(q.timestamp_s for q in todo),
                    clip_batch=a.clip_batch,
                    vision_batch_planes=a.vision_batch_planes,
                    cache_device=a.cache_device,
                    frames=dv.frames, meta=dv.meta,
                )
                vp.timings.add("decode", dv.decode_ms, n=int(dv.frames.shape[0]))
                del dv
            except Exception as exc:
                print(f"[video {vi}] {item.video_id}: FAILED {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                for q in todo:
                    rec = {
                        "video_id": q.video_id, "question_id": q.question_id,
                        "task_type": q.task_type, "task_code": q.task_code,
                        "ground_truth": q.answer, "predicted_response": "",
                        "predicted_letter": None, "is_correct": False,
                        "parse_failed": True,
                        "error": f"video pass failed: {type(exc).__name__}: {exc}",
                    }
                    fh_jsonl.write(json.dumps(rec) + "\n")
                    records.append(rec)
                fh_jsonl.flush()
                continue

            prof.accumulate_video(vp.timings, vp.n_frames, len(vp.planes))
            prof.write(ProfileRow(
                scope="video", video_id=item.video_id,
                n_frames=vp.n_frames, n_planes=len(vp.planes),
                decode_ms_per_frame=vp.timings.per_unit("decode"),
                openclip_ms_per_frame=vp.timings.per_unit("openclip"),
                vision_tower_ms_per_frame=vp.timings.per_unit("vision_tower"),
                total_wall_clock_sec=time.perf_counter() - t_vid,
            ))

            for q in todo:
                out = runner.answer(vp, q)
                fh_jsonl.write(json.dumps(out.record) + "\n")
                records.append(out.record)
                n_done += 1
                prof.accumulate_question(out.stopwatch, out.dense_tokens,
                                         out.kept_tokens, out.new_tokens)
                prof.write(ProfileRow(
                    scope="question", video_id=q.video_id, question_id=q.question_id,
                    task_code=q.task_code,
                    n_frames=out.record.get("n_frames_visible", 0),
                    n_planes=out.record.get("n_planes", 0),
                    qdp_rtar_ms_per_frame=out.stopwatch.per_unit("qdp_rtar"),
                    llm_prefill_ms=out.stopwatch.ms("llm_prefill"),
                    llm_decode_ms=out.stopwatch.ms("llm_decode"),
                    llm_decode_ms_per_token=out.stopwatch.per_unit("llm_decode"),
                    n_new_tokens=out.new_tokens,
                    visual_tokens_dense=out.dense_tokens,
                    visual_tokens_kept=out.kept_tokens,
                    avg_keep_rate=out.record.get("avg_keep_rate", 0.0),
                    total_wall_clock_sec=out.record.get("wall_clock_sec", 0.0),
                ))
            fh_jsonl.flush()

            del vp
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (vi + 1) % a.progress_every == 0 or vi == len(pending) - 1:
                s = sb.aggregate(records)
                el = time.perf_counter() - t_run
                print(f"[{vi + 1}/{len(items)} videos | {n_done} q | {el / 60:.1f} min] "
                      f"acc {s['overall']['acc']:.2f}%  "
                      f"keep {s['token_keep_rate_pct']:.2f}%  "
                      f"unparsed {s['unparsed']}  errors {s['errors']}", flush=True)
    finally:
        prefetch.shutdown()
        fh_jsonl.close()
        overall = prof.close()

    # ---- report -------------------------------------------------------- #
    summary = sb.aggregate(records)
    summary["profile"] = overall.as_dict()
    summary["run"] = {
        "videos_in_split": len(items), "videos_completed": vi + 1,
        "questions_evaluated": n_done,
        "stopped_on_time_budget": budget_hit,
        "time_budget_sec": a.time_budget_sec,
        "clip_batch": a.clip_batch, "decode_workers": a.decode_workers,
        "wall_clock_sec": time.perf_counter() - t_run,
        "mask_space": a.mask_space, "pool": a.pool,
        "relevance_mode": a.relevance_mode, "keep_mode": a.keep_mode,
        "tau_temp": a.tau_temp, "tau_rel": a.tau_rel, "tau_den": a.tau_den,
        "alpha": a.alpha, "fps": a.fps, "max_pixels": a.max_pixels,
    }
    if judge is not None:
        summary["run"].update(judge.stats())
    with open(os.path.join(a.output_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print()
    print(sb.format_table(summary))
    print()
    print(f"  wall clock        {summary['run']['wall_clock_sec'] / 60:.1f} min")
    print(f"  openclip          {overall.openclip_ms_per_frame:.2f} ms/frame")
    print(f"  vision tower      {overall.vision_tower_ms_per_frame:.2f} ms/frame")
    print(f"  qdp+rtar          {overall.qdp_rtar_ms_per_frame:.3f} ms/frame")
    print(f"  llm prefill/TTFT  {overall.llm_prefill_ms:.1f} ms/question")
    print(f"  llm decode        {overall.llm_decode_ms:.1f} ms/question "
          f"({overall.llm_decode_ms_per_token:.1f} ms/token)")
    print()
    print(f"  jsonl   {jsonl_path}")
    print(f"  csv     {csv_path}")
    print(f"  summary {os.path.join(a.output_dir, 'summary.json')}")

    if a.dry_run_assert_keep:
        lo, hi = (float(x) for x in a.dry_run_assert_keep.split(","))
        kr = summary["token_keep_rate_pct"]
        if not (lo <= kr <= hi):
            print(f"\nGATE FAILED: keep rate {kr:.2f}% outside [{lo}, {hi}]",
                  file=sys.stderr)
            return 3
        print(f"\nGATE PASSED: keep rate {kr:.2f}% inside [{lo}, {hi}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
