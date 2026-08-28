#!/usr/bin/env python
"""Prove the cached evaluation path == the streaming pipeline, token for token.

`eval_runner.StreamingBenchRunner` replays QDP/RTAR over a pre-computed
`VideoPass` instead of streaming frame by frame. That is a real optimisation
(661k frame-encodes -> 252k on the 500-video split) but it duplicates decision
logic, so it needs a proof rather than an argument. This test runs both paths
over the same clip and asserts they agree on:

    1. the per-frame CLIP-grid QDP keep rate   (the signal D_t gates on)
    2. the raw and calibrated relevance        (the signal R_t gates on)
    3. R_t / D_t / T_t per frame
    4. the per-plane backbone masks            (what actually prunes)
    5. the flushed M-ROPE coordinate set       (what the backbone receives)

Run:
    python tests/test_eval_equivalence.py --device cuda:0            # StubBackbone
    python tests/test_eval_equivalence.py --device cuda:0 --real-qwen
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from querystream.benchmarks.streamingbench import Question
from querystream.config import QueryStreamConfig
from querystream.eval_cache import build_video_pass
from querystream.eval_runner import StreamingBenchRunner
from querystream.jit_streaming_pipeline import JITStreamingPipeline
from querystream.open_clip_wrapper import OpenCLIPWrapper
from querystream.video_io import read_video_frames

SAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_bottle.mp4")
QUERY = "a person opening a green bottle"

_checks = [0, 0]


def check(name: str, ok: bool, detail: str = "") -> None:
    _checks[0] += 1
    if ok:
        _checks[1] += 1
        print(f"  PASS  {name}" + (f"   [{detail}]" if detail else ""))
    else:
        print(f"  FAIL  {name}   {detail}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--real-qwen", action="store_true")
    ap.add_argument("--video", default=SAMPLE)
    ap.add_argument("--mask-space", default="score")
    ap.add_argument("--pool", default="mean")
    ap.add_argument("--relevance-mode", default="running_minmax")
    ap.add_argument("--clip-batch", type=int, default=1,
                    help="1 reproduces the pipeline's one-frame-at-a-time encoder "
                         "calls exactly; >1 is the fast path the harness uses")
    a = ap.parse_args()

    if not os.path.exists(a.video):
        print(f"missing {a.video} -- run: python tests/make_sample_video.py", file=sys.stderr)
        return 2

    cfg = QueryStreamConfig()
    cfg.encoder.device = a.device
    cfg.buffer.device = a.device
    cfg.mask_space = a.mask_space
    cfg.score_pool_mode = a.pool
    cfg.rtar.relevance_mode = a.relevance_mode
    cfg.validate()

    print(f"[load] OpenCLIP on {a.device} ...", flush=True)
    clip = OpenCLIPWrapper(cfg.encoder)

    if a.real_qwen:
        from querystream.qwen_wrapper import QwenVLWrapper
        print("[load] Qwen2.5-VL ...", flush=True)
        qwen = QwenVLWrapper(cfg.backbone, device=a.device, max_pixels=448 * 448)
    else:
        from querystream.qwen_wrapper import StubBackbone
        qwen = StubBackbone(device=a.device)

    frames, meta = read_video_frames(a.video, target_fps=cfg.fps)
    n = frames.shape[0] - (frames.shape[0] % 2)          # whole planes only
    frames = frames[:n]
    print(f"[data] {a.video}: {n} frames at {cfg.fps} fps, {meta.width}x{meta.height}")

    # ---- path A: the streaming pipeline ------------------------------- #
    pipe = JITStreamingPipeline(clip, qwen, cfg)
    res = pipe.run(frames, QUERY, generate=False, answer_at_end=False)
    a_keep = [f.qdp_keep_rate for f in res.frames]
    a_relraw = [f.relevance_raw for f in res.frames]
    a_relcal = [f.relevance for f in res.frames]
    a_flags = [(f.r_t, f.d_t, f.trigger) for f in res.frames]
    a_masks = torch.stack([bf.mask for bf in pipe.buffer._frames], dim=0)
    a_flush = pipe.buffer.flush(apply_mask=True)

    # ---- path B: the cached evaluation runner -------------------------- #
    vp = build_video_pass(
        a.video, "equiv", clip=clip, qwen=qwen, cfg=cfg,
        frames=frames, meta=meta, vision_batch_planes=4, clip_batch=a.clip_batch,
    )
    runner = StreamingBenchRunner(clip, None, cfg)      # None => no generation
    q = Question(
        video_id="equiv", question_id="equiv_q0", index_in_video=0,
        timestamp_s=n - 1, question=QUERY, options=[], answer="A",
        task_type="Object Perception",
    )
    out = runner.answer(vp, q)
    r = out.record
    assert r.get("error") is None, r.get("error")

    planes, n_vis, dropped = vp.planes_up_to(q.timestamp_s)
    b_masks = runner._plane_masks(
        _sem(vp, q, clip),
        vp.temp_sim[:n_vis].to(torch.float32),
        vp.clip_grid_hw, vp.llm_grid_hw, len(planes), 2,
    )

    print(f"\n[equivalence] {len(a_keep)} frames, {a_masks.shape[0]} planes, "
          f"{a_flush.num_tokens_after}/{a_flush.num_tokens_before} tokens kept "
          f"({100 * a_flush.keep_rate:.2f}%)")

    # ---- assertions ----------------------------------------------------- #
    b_keep = r["per_frame_keep_rates"]
    check("frame count matches", len(a_keep) == len(b_keep),
          f"{len(a_keep)} vs {len(b_keep)}")
    # With clip_batch=1 the encoder is driven exactly as the pipeline drives it,
    # so these must be bit-identical. With clip_batch>1 the ViT runs a different
    # GEMM shape and fp16 reduction order, which perturbs patch features at the
    # 1e-5 level; that moves a couple of patches across the frame-mean threshold
    # and therefore the keep rate. The masks that actually prune are unaffected
    # (they are asserted bitwise below) because score-space pooling averages 8
    # patches per backbone token.
    # The JSONL rounds every per-frame list to 6 decimals, so 5e-7 is the floor
    # on any comparison here, and running_minmax divides by the observed range
    # (~0.05 on real video), amplifying that floor by ~20x. 1e-5 is therefore
    # "identical up to logging precision", not a slack tolerance.
    exact = a.clip_batch == 1
    tol_sim = 1e-5 if exact else 1e-3
    tol_keep = 1e-5 if exact else 2e-2
    tag = "to logging precision" if exact else f"within {tol_keep:g}"
    tag_sim = "to logging precision" if exact else f"within {tol_sim:g}"

    d = max((abs(x - y) for x, y in zip(a_keep, b_keep)), default=0.0)
    check(f"CLIP-grid QDP keep rate matches {tag}", d <= tol_keep, f"max |diff| = {d:.2e}")

    d = max((abs(x - y) for x, y in zip(a_relraw, r["relevance_raw_scores"])), default=0.0)
    check(f"relevance_raw matches {tag_sim}", d <= tol_sim, f"max |diff| = {d:.2e}")

    d = max((abs(x - y) for x, y in zip(a_relcal, r["relevance_normalized_scores"])), default=0.0)
    check(f"calibrated relevance matches {tag}", d <= tol_keep, f"max |diff| = {d:.2e}")

    b_flags = list(zip([bool(x) for x in r["r_t_flags"]],
                       [bool(x) for x in r["d_t_flags"]],
                       [bool(x) for x in r["trigger_flags"]]))
    check("R_t / D_t / T_t identical per frame", a_flags == b_flags,
          f"{sum(f[2] for f in a_flags)} triggers")

    check("plane mask shape matches", tuple(a_masks.shape) == tuple(b_masks.shape),
          f"{tuple(a_masks.shape)} vs {tuple(b_masks.shape)}")
    same = bool(torch.equal(a_masks.cpu(), b_masks.cpu()))
    check("backbone plane masks bitwise identical", same,
          f"{int((a_masks.cpu() ^ b_masks.cpu()).sum())} differing entries")

    check("kept token count matches", a_flush.num_tokens_after == r["visual_tokens_kept"],
          f"{a_flush.num_tokens_after} vs {r['visual_tokens_kept']}")
    check("dense token count matches", a_flush.num_tokens_before == r["visual_tokens_dense"],
          f"{a_flush.num_tokens_before} vs {r['visual_tokens_dense']}")

    # The M-ROPE set is the real contract: same coordinates, same order.
    buf2 = _rebuild_buffer(cfg, planes, b_masks)
    b_flush = buf2.flush(apply_mask=True)
    check("flushed M-ROPE coordinates identical",
          bool(torch.equal(a_flush.mrope.cpu(), b_flush.mrope.cpu())),
          f"{tuple(a_flush.mrope.shape)}")

    print(f"\n{'ALL' if _checks[1] == _checks[0] else 'SOME'} "
          f"{_checks[1]}/{_checks[0]} EQUIVALENCE CHECKS PASSED")
    return 0 if _checks[1] == _checks[0] else 1


def _sem(vp, q, clip):
    from querystream.eval_runner import _sem_similarity
    n_vis = min(vp.n_frames, q.timestamp_s + 1)
    return _sem_similarity(vp.patch_feats[:n_vis], clip.encode_text(q.query_text()))


def _rebuild_buffer(cfg, planes, masks):
    from querystream.buffer import TokenBuffer
    buf = TokenBuffer(cfg.buffer)
    for j, pl in enumerate(planes):
        buf.append(frame_idx=pl.frame_ids[0], timestamp=pl.timestamp,
                   tokens=pl.tokens, mrope=pl.mrope, mask=masks[j])
    return buf


if __name__ == "__main__":
    raise SystemExit(main())
