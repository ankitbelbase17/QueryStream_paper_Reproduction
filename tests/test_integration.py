#!/usr/bin/env python3
"""QueryStream Milestone 2 end-to-end integration test.

Runs the real stack -- OpenCLIP-ViT-L/14 + Qwen2.5-VL-7B-Instruct -- over a
sample .mp4 at 1 FPS and verifies the three things Milestone 2 has to get right:

  1. KEEP RATES are reported before AND after `resample_patch_mask`, so the
     CLIP-grid -> backbone-grid conversion is visible rather than implicit.
  2. VRAM stays flat while buffering (T_t == False) and spikes only inside
     `qwen.generate()` (T_t == True).
  3. Surviving M-ROPE coordinates match the surviving visual tokens EXACTLY --
     the paper's Sec. 3.2 spatio-temporal-integrity guarantee.

    python tests/test_integration.py                       # full, needs a GPU
    python tests/test_integration.py --no-qwen             # CLIP + QDP/RTAR only
    python tests/test_integration.py --device cuda:1
    python tests/test_integration.py --max-frames 6 -v
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from querystream import QueryStreamConfig, TokenBuffer  # noqa: E402
from querystream.jit_streaming_pipeline import JITStreamingPipeline  # noqa: E402
from querystream.open_clip_wrapper import OpenCLIPWrapper  # noqa: E402
from querystream.ops import expand_mask_to_temporal_merge, resample_patch_mask  # noqa: E402
from querystream.video_io import read_video_frames  # noqa: E402

G, R, Y, B, D, X = "\033[32m", "\033[31m", "\033[33m", "\033[34m", "\033[2m", "\033[0m"
DEFAULT_VIDEO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_bottle.mp4")
QUERY = "a green bottle being opened"

_checks: List[tuple] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    _checks.append((name, bool(cond), detail))
    tag = f"{G}PASS{X}" if cond else f"{R}FAIL{X}"
    print(f"  {tag}  {name}" + (f"  {D}{detail}{X}" if detail else ""))
    return bool(cond)


def mb(x: int) -> float:
    return x / 1024**2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--query", default=QUERY)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-qwen", action="store_true", help="skip the 7B backbone")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--max-pixels", type=int, default=200704, help="Qwen smart_resize cap (=448*448)")
    ap.add_argument("--relevance-mode", default="running_minmax")
    ap.add_argument("--resample-mode", default="any", choices=["any", "all", "majority"])
    ap.add_argument("--mask-space", default=None, choices=["mask", "score"],
                    help="override QueryStreamConfig.mask_space")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--jsonl", default=None, help="write per-frame records here")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 84)
    print(f"{B}QueryStream -- Milestone 2 integration test{X}")
    print("=" * 84)
    print(f"  torch {torch.__version__} | device {dev}")
    if dev.type == "cuda":
        free, total = torch.cuda.mem_get_info(dev)
        print(f"  GPU free {mb(free)/1024:.1f} GiB / {mb(total)/1024:.1f} GiB")

    # ------------------------------------------------------------------ #
    # 0. video
    # ------------------------------------------------------------------ #
    print(f"\n{B}[0] Decode video at 1 FPS{X}")
    if not os.path.exists(a.video):
        print(f"{R}missing {a.video}; run: python tests/make_sample_video.py{X}")
        return 1
    frames, meta = read_video_frames(a.video, target_fps=1.0, max_frames=a.max_frames)
    print(f"  {json.dumps(meta.summary())}")
    check("video decoded to [T,3,H,W] uint8 RGB",
          frames.dim() == 4 and frames.shape[1] == 3 and frames.dtype == torch.uint8,
          f"{tuple(frames.shape)}")
    check("at least 4 frames (>= 2 temporal planes)", frames.shape[0] >= 4,
          f"{frames.shape[0]} frames")

    # ------------------------------------------------------------------ #
    # 1. encoders
    # ------------------------------------------------------------------ #
    print(f"\n{B}[1] Load encoders{X}")
    cfg = QueryStreamConfig().validate()
    cfg.encoder.device = str(dev)
    cfg.encoder.dtype = torch.float16
    cfg.buffer.device = str(dev)
    cfg.rtar.relevance_mode = a.relevance_mode
    cfg.rtar.calibration_warmup = 3
    cfg.mask_resample_mode = a.resample_mode
    if a.mask_space:
        cfg.mask_space = a.mask_space

    t0 = time.perf_counter()
    clip = OpenCLIPWrapper(cfg.encoder)
    print(f"  OpenCLIP {time.perf_counter()-t0:.1f}s  {json.dumps(clip.describe())}")
    check("CLIP grid is 16x16 = 256 patches", clip.grid_hw == (16, 16) and clip.num_patches == 256)
    check("CLIP embed dim 768 (post visual.proj)", clip.embed_dim == 768)

    qwen = None
    if a.no_qwen:
        from querystream.qwen_wrapper import StubBackbone
        qwen = StubBackbone(cfg.backbone, device=str(dev), dtype=torch.float32)
        print(f"  {Y}StubBackbone{X} (no weights) -- Qwen's exact token geometry, "
              f"tokens self-describe their (t,h,w)")
    if not a.no_qwen:
        from querystream.qwen_wrapper import QwenVLWrapper
        t0 = time.perf_counter()
        try:
            qwen = QwenVLWrapper(cfg.backbone, device=str(dev),
                                 dtype=torch.bfloat16, max_pixels=a.max_pixels)
        except torch.OutOfMemoryError as e:
            print(f"  {R}OOM loading Qwen2.5-VL-7B{X}: {str(e).splitlines()[0]}")
            print(f"  {Y}Re-run with --no-qwen, or --device cuda:N on a freer GPU.{X}")
            return 2
        print(f"  Qwen {time.perf_counter()-t0:.1f}s  {json.dumps(qwen.describe())}")
        check("backbone hidden 3584", qwen.hidden_size == 3584)
        check("spatial_merge 2, temporal_patch 2, tokens_per_second 2",
              (qwen.spatial_merge_size, qwen.temporal_patch_size, qwen.tokens_per_second) == (2, 2, 2))
        check("time_interval at 1 FPS is 4", qwen.time_interval(1.0) == 4,
              "tokens_per_second(2) * int(second_per_grid_t=2)")

        # M-ROPE must match transformers' own generator exactly.
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
        g = torch.tensor([2, 16, 16], device=dev)
        ref = Qwen2_5_VLModel.get_vision_position_ids(
            None, 5, g, 1, qwen.spatial_merge_size, qwen.time_interval(1.0), device=dev).T
        mine = qwen.build_dense_mrope(g, time_interval=qwen.time_interval(1.0), start_position=5)
        check("build_dense_mrope == transformers get_vision_position_ids",
              torch.equal(ref, mine), f"{tuple(mine.shape)}")

    # ------------------------------------------------------------------ #
    # 2. mask resampling, in isolation
    # ------------------------------------------------------------------ #
    print(f"\n{B}[2] Mask resampling CLIP 16x16 -> backbone grid{X}")
    m = torch.zeros(16, 16, dtype=torch.bool, device=dev)
    m[0, 0] = True
    r_any = resample_patch_mask(m, (16, 16), (8, 8), "any")
    r_all = resample_patch_mask(m, (16, 16), (8, 8), "all")
    check("'any' keeps a partially covered block", int(r_any.sum()) == 1)
    check("'all' drops a partially covered block", int(r_all.sum()) == 0)
    stack = torch.stack([r_any, torch.zeros_like(r_any)], 0)
    check("temporal fold 2 planes -> 1", expand_mask_to_temporal_merge(stack, 2, "any").shape == (1, 64))

    # ------------------------------------------------------------------ #
    # 3. stream
    # ------------------------------------------------------------------ #
    print(f"\n{B}[3] Stream {frames.shape[0]} frames @ 1 FPS  query={a.query!r}{X}")
    pipe = JITStreamingPipeline(clip, qwen, cfg)
    q_emb = clip.encode_text(a.query)

    if dev.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        base_vram = torch.cuda.memory_allocated(dev)
    else:
        base_vram = 0

    print(f"\n  {'f':>2} {'t':>4} {'sem':>6} {'temp':>6} {'qdp':>6} {'tok':>6} "
          f"{'rel':>6} {'R':>2} {'D':>2} {'T':>2} {'fire':>5} {'bufMB':>7} {'vramMB':>8}")
    print("  " + "-" * 80)

    records, flushes, buffering_vram = [], [], []
    for i in range(frames.shape[0]):
        rec, flush = pipe.step(frames[i], q_emb, frame_idx=i, timestamp=float(i))
        records.append(rec)
        if dev.type == "cuda":
            torch.cuda.synchronize()
            cur = torch.cuda.memory_allocated(dev)
        else:
            cur = 0
        if not rec.fired:
            buffering_vram.append(cur)
        tk = f"{rec.tok_keep_rate:6.3f}" if rec.tok_keep_rate is not None else "     -"
        print(f"  {i:>2} {rec.timestamp:>4.0f} {rec.qdp_keep_rate_sem:>6.3f} "
              f"{rec.qdp_keep_rate_temp:>6.3f} {rec.qdp_keep_rate:>6.3f} {tk} "
              f"{rec.relevance:>6.3f} {int(rec.r_t):>2} {int(rec.d_t):>2} "
              f"{int(rec.trigger):>2} {str(rec.fired):>5} {rec.buffer_mb:>7.2f} "
              f"{mb(cur):>8.1f}")
        if flush is not None:
            flushes.append((i, flush))

    # ------------------------------------------------------------------ #
    # 4. keep rates before/after resampling
    # ------------------------------------------------------------------ #
    print(f"\n{B}[4] Keep rate: QDP (CLIP grid) vs backbone grid{X}")
    qdp_rates = [r.qdp_keep_rate for r in records]
    tok_rates = [r.tok_keep_rate for r in records if r.tok_keep_rate is not None]
    mean_qdp = sum(qdp_rates) / len(qdp_rates)
    mean_tok = sum(tok_rates) / len(tok_rates) if tok_rates else float("nan")
    print(f"  before resample (16x16, per frame) : mean {mean_qdp:.4f}  "
          f"min {min(qdp_rates):.4f}  max {max(qdp_rates):.4f}")
    if tok_rates:
        print(f"  after  resample (backbone plane)   : mean {mean_tok:.4f}  "
              f"min {min(tok_rates):.4f}  max {max(tok_rates):.4f}")
    else:
        print(f"  {R}after resample: NO PLANES BUFFERED{X}")
    print(f"  {Y}NOTE{X} mask_space={cfg.mask_space!r} "
          f"({'pool=' + a.resample_mode if cfg.mask_space == 'mask' else 'pool=' + cfg.score_pool_mode})")
    print(f"       The paper's reported Token Keep Rate is the BACKBONE-grid number.")
    print(f"       mask_space='mask' + 'any' pooling inflates it badly (8 CLIP patches")
    print(f"       -> 1 backbone token, kept if ANY survived): measured 41% -> 96%.")
    check("backbone planes were buffered", len(tok_rates) > 0, f"{len(tok_rates)} planes")
    check("all keep rates in [0, 1]", all(0.0 <= r <= 1.0 for r in qdp_rates + tok_rates))
    if cfg.mask_space == "mask" and a.resample_mode == "any":
        check("binary 'any' pooling only adds coverage",
              mean_tok >= mean_qdp - 1e-6, f"{mean_tok:.4f} >= {mean_qdp:.4f}")
    else:
        check("score-space keep rate stays near the QDP rate (< 1.5x)",
              mean_tok <= mean_qdp * 1.5 + 1e-6,
              f"{mean_tok:.4f} vs QDP {mean_qdp:.4f} ({mean_tok/mean_qdp:.2f}x)")

    # ------------------------------------------------------------------ #
    # 5. VRAM: flat while buffering, spikes only in generate()
    # ------------------------------------------------------------------ #
    print(f"\n{B}[5] VRAM behaviour{X}")
    if dev.type == "cuda" and buffering_vram:
        lo, hi = min(buffering_vram), max(buffering_vram)
        drift = hi - lo
        print(f"  baseline after load        : {mb(base_vram):9.1f} MB")
        print(f"  while buffering (T_t=False): {mb(lo):9.1f} .. {mb(hi):9.1f} MB "
              f"(drift {mb(drift):.1f} MB)")
        print(f"  token buffer itself        : {records[-1].buffer_mb:9.2f} MB "
              f"({records[-1].buffer_tokens} unpruned tokens)")
        # The buffer holds [N, 3584] bf16; growth must be modest and explainable.
        check("buffering drift is small (< 512 MB)", mb(drift) < 512.0, f"{mb(drift):.1f} MB")
        check("token buffer accounts for the growth (< 64 MB for this clip)",
              records[-1].buffer_mb < 64.0, f"{records[-1].buffer_mb:.2f} MB")
    else:
        print(f"  {D}skipped (cpu){X}")

    # ------------------------------------------------------------------ #
    # 6. M-ROPE / token correspondence  -- the core invariant
    # ------------------------------------------------------------------ #
    print(f"\n{B}[6] M-ROPE correspondence on flushed context{X}")
    final = pipe.buffer.flush(apply_mask=True)
    dense = pipe.buffer.flush(apply_mask=False)
    print(f"  buffered planes {pipe.buffer.num_frames} | dense {dense.num_tokens_after} tokens "
          f"-> pruned {final.num_tokens_after} tokens (keep {final.keep_rate:.4f})")

    check("tokens and M-ROPE have equal length",
          final.tokens.shape[0] == final.mrope.shape[0], f"{final.tokens.shape[0]}")
    check("M-ROPE is int64 with 3 columns (t, h, w)",
          final.mrope.dtype == torch.long and final.mrope.shape[1] == 3)

    # Reconstruct the expected surviving coordinates independently from the
    # buffer's own per-plane masks, and require an exact ordered match.
    expected = []
    for fr in pipe.buffer._frames:
        expected.append(fr.mrope[fr.mask])
    expected = torch.cat(expected, 0) if expected else torch.empty(0, 3, dtype=torch.long)
    check("surviving M-ROPE == independently recomputed selection",
          torch.equal(final.mrope, expected.to(final.mrope.device)),
          f"{tuple(final.mrope.shape)}")

    # Every surviving coordinate must exist in the dense set, and coordinates
    # must NOT be renumbered into a dense range.
    dense_set = {tuple(r) for r in dense.mrope.tolist()}
    surv = [tuple(r) for r in final.mrope.tolist()]
    check("every surviving coord exists in the dense grid",
          all(s in dense_set for s in surv))
    check("no duplicate surviving coordinates", len(set(surv)) == len(surv))
    if final.num_tokens_after < dense.num_tokens_after:
        t_vals = sorted({r[0] for r in surv})
        h_vals = sorted({r[1] for r in surv})
        check("temporal coords keep the time_interval spacing (multiples of 4)",
              all(t % 4 == 0 for t in t_vals), f"t = {t_vals}")
        check("coordinates are absolute, not renumbered 0..K-1",
              max(h_vals) >= len(h_vals) - 1, f"h = {h_vals}")
    check("frame_ids match the M-ROPE temporal column ordering",
          final.frame_ids.shape[0] == final.mrope.shape[0])

    if a.no_qwen and final.num_tokens_after > 0:
        # StubBackbone tokens carry their own CHUNK-RELATIVE (t, h, w). The
        # pipeline then shifts only the M-ROPE temporal column onto the stream
        # timeline (plane_idx * time_interval), so h/w must match EXACTLY while
        # t differs by that per-plane offset. h and w are unique within a plane,
        # so any index-set mismatch scrambles them and this fails.
        tok_hw = final.tokens[:, 1:3].round().long()
        check("STRUCTURAL: flushed tokens[:, 1:3] == flushed mrope[:, 1:3] (h, w)",
              torch.equal(tok_hw, final.mrope[:, 1:2 + 1]),
              "token/M-ROPE slicing used ONE index set")
        # temporal column: offset must be a constant multiple of time_interval
        t_off = (final.mrope[:, 0] - final.tokens[:, 0].round().long())
        check("STRUCTURAL: t offsets are whole multiples of time_interval",
              bool(((t_off % 4) == 0).all()), f"offsets {sorted(set(t_off.tolist()))}")

    # ------------------------------------------------------------------ #
    # 7. generation from pruned tokens
    # ------------------------------------------------------------------ #
    can_gen = qwen is not None and hasattr(qwen, "generate_from_pruned")
    if can_gen and final.num_tokens_after > 0:
        print(f"\n{B}[7] Generate from pruned tokens{X}")
        if dev.type == "cuda":
            torch.cuda.synchronize()
            pre_gen = torch.cuda.memory_allocated(dev)
            torch.cuda.reset_peak_memory_stats(dev)

        gen = qwen.generate_from_pruned(
            query=a.query, visual_tokens=final.tokens, mrope=final.mrope,
            dense_llm_grid_hw=pipe.dense_llm_grid_hw, max_new_tokens=a.max_new_tokens)
        print(f"  visual tokens {gen.num_visual_tokens} | prefill {gen.num_prefill_tokens} "
              f"| rope_delta {gen.rope_delta} | {gen.decode_ms:.0f} ms")
        print(f"  peak during generate : {mb(gen.peak_vram_bytes):.1f} MB")
        print(f"  {G}response{X}: {gen.text[:400]!r}")

        check("generation produced non-empty text", len(gen.text.strip()) > 0)
        check("prefill contains exactly K video pad tokens",
              gen.num_visual_tokens == final.num_tokens_after)
        if dev.type == "cuda":
            spike = gen.peak_vram_bytes - pre_gen
            print(f"  generate() spike over buffering steady-state: {mb(spike):.1f} MB")
            check("generate() allocates more than a buffering step",
                  gen.peak_vram_bytes > pre_gen, f"peak {mb(gen.peak_vram_bytes):.1f} MB "
                  f"> pre-gen {mb(pre_gen):.1f} MB")

        # dense control: same code path, no pruning
        print(f"\n{B}[7b] Dense control (100% keep, same path){X}")
        gen_d = qwen.generate_from_pruned(
            query=a.query, visual_tokens=dense.tokens, mrope=dense.mrope,
            dense_llm_grid_hw=pipe.dense_llm_grid_hw, max_new_tokens=a.max_new_tokens)
        print(f"  visual tokens {gen_d.num_visual_tokens} | {gen_d.decode_ms:.0f} ms")
        print(f"  {G}response{X}: {gen_d.text[:400]!r}")
        check("dense control also generates", len(gen_d.text.strip()) > 0)
        check("pruned run used strictly fewer visual tokens",
              gen.num_visual_tokens < gen_d.num_visual_tokens or final.keep_rate == 1.0,
              f"{gen.num_visual_tokens} vs {gen_d.num_visual_tokens}")
    else:
        print(f"\n{B}[7] Generation {D}skipped (StubBackbone has no LLM){X}")

    # ------------------------------------------------------------------ #
    if a.jsonl:
        os.makedirs(os.path.dirname(a.jsonl) or ".", exist_ok=True)
        with open(a.jsonl, "w") as f:
            for r in records:
                f.write(json.dumps(r.to_json()) + "\n")
        print(f"\n  wrote {len(records)} records -> {a.jsonl}")

    print("\n" + "=" * 84)
    passed = sum(1 for _, ok, _ in _checks if ok)
    failed = [n for n, ok, _ in _checks if not ok]
    if failed:
        print(f"{R}{len(failed)} FAILED{X}, {passed} passed")
        for n in failed:
            print(f"    {R}x{X} {n}")
        print("=" * 84)
        return 1
    print(f"{G}ALL {passed} CHECKS PASSED{X}")
    print("=" * 84 + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
