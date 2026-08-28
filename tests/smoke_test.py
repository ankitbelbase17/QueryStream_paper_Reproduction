#!/usr/bin/env python3
"""QueryStream Milestone 1 smoke test.

Runs without pytest, without open_clip, without network access. Every paper
formula is checked against an independently hand-computed reference, not
against the implementation's own output.

    python tests/smoke_test.py            # cpu
    python tests/smoke_test.py --device cuda
    python tests/smoke_test.py -v         # print per-frame traces

Synthetic stream: 5 frames at 224x224, OpenCLIP-ViT-L/14 geometry
(16x16 = 256 patches, D=768), Qwen2.5-VL merged token grid (8x8 = 64 tokens).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from typing import Callable, List, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from querystream import (  # noqa: E402
    BackboneConfig, BufferConfig, DSHMemory, EncoderConfig, QDPConfig, QDPModule,
    QueryStreamConfig, RTARConfig, RTARModule, SyntheticEncoder, TokenBuffer,
    build_mrope_coords, cosine_sim_rowwise, cosine_sim_to_vector,
    expand_mask_to_temporal_merge, l2_normalize, mean_pool_frame, resample_patch_mask,
)

# --------------------------------------------------------------------------- #
# Test harness
# --------------------------------------------------------------------------- #

GREEN, RED, YELLOW, BLUE, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[34m", "\033[2m", "\033[0m"
)

_REGISTRY: List[Tuple[str, Callable]] = []
VERBOSE = False
DEVICE = torch.device("cpu")


def test(name: str):
    def deco(fn):
        _REGISTRY.append((name, fn))
        return fn
    return deco


def vprint(*a):
    if VERBOSE:
        print(f"{DIM}      ", *a, RESET, sep="")


def approx(a, b, tol=1e-5, msg=""):
    a_ = a.item() if torch.is_tensor(a) else float(a)
    b_ = b.item() if torch.is_tensor(b) else float(b)
    assert abs(a_ - b_) <= tol, f"{msg} | {a_!r} != {b_!r} (tol {tol})"


def tensors_equal(a: torch.Tensor, b: torch.Tensor, msg=""):
    assert a.shape == b.shape, f"{msg} | shape {tuple(a.shape)} != {tuple(b.shape)}"
    assert torch.equal(a, b), (
        f"{msg} | {int((a != b).sum())} of {a.numel()} elements differ"
    )


def shape_is(t: torch.Tensor, expected: tuple, name: str):
    assert tuple(t.shape) == expected, (
        f"{name}: expected shape {expected}, got {tuple(t.shape)}"
    )
    vprint(f"{name:<28} {tuple(t.shape)}  {t.dtype}")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

FRAMES = 5
IMG = 224
CLIP_PATCH = 14
CLIP_GRID = (IMG // CLIP_PATCH, IMG // CLIP_PATCH)      # (16, 16) -> 256 patches
N_PATCH = CLIP_GRID[0] * CLIP_GRID[1]
EMB = 768
QWEN_GRID = BackboneConfig().token_grid_hw(IMG)          # (8, 8) -> 64 tokens
N_TOKEN = QWEN_GRID[0] * QWEN_GRID[1]
DV = 3584                                                # Qwen2.5-VL-7B hidden size


def make_encoder(seed=0) -> SyntheticEncoder:
    return SyntheticEncoder(
        embed_dim=EMB, grid_hw=CLIP_GRID, seed=seed, device=str(DEVICE), drift=0.01
    )


# Two 8x8 regions of the 16x16 grid, 25% of the frame each:
#   RELEVANT   (top-left)     -- matches the query AND moves  -> M_sem & M_temp
#   DISTRACTOR (bottom-right) -- moves but is irrelevant      -> M_temp only
# The remaining 50% is near-static background                 -> neither.
# This makes M_QDP a STRICT subset of M_temp, so the conjunction is exercised
# rather than coinciding with one operand.
RELEVANT = [r * CLIP_GRID[1] + c for r in range(8) for c in range(8)]
DISTRACTOR = [r * CLIP_GRID[1] + c for r in range(8, 16) for c in range(8, 16)]


def make_stream(enc: SyntheticEncoder):
    """5-frame scripted stream exercising the paper's Appendix A.7 failure modes.

        f0: baseline, a relevant object occupies the top-left quadrant
        f1: near-static (tests DSH suppression of redundancy)
        f2: SHOCK -- unrelated content (the inserted black frames): visually
            explosive, semantically irrelevant. Must NOT trigger.
        f3: back to baseline
        f4: SUBTLE relevant event: small persistent change. Frame-to-frame
            diffing misses it; DSH accumulates it. Must trigger.
    """
    K = dict(relevant_patches=RELEVANT, distractor_patches=DISTRACTOR,
             distractor_motion=0.05, motion=0.03, shock=False)
    return [
        dict(t=0, relevance_strength=0.80, subtle_delta=0.0000, **K),
        dict(t=1, relevance_strength=0.80, subtle_delta=0.0000, **K),
        dict(t=2, relevant_patches=None, distractor_patches=None, distractor_motion=0.0,
             relevance_strength=0.0, subtle_delta=0.0, motion=0.0, shock=True),
        dict(t=3, relevance_strength=0.80, subtle_delta=0.0000, **K),
        dict(t=4, relevance_strength=0.90, subtle_delta=0.0006, **K),
    ]


# =========================================================================== #
# 1. ops
# =========================================================================== #

@test("ops: l2_normalize / cosine primitives")
def t_ops_basic():
    x = torch.randn(37, EMB, device=DEVICE)
    xn = l2_normalize(x)
    approx(xn.norm(dim=-1).max(), 1.0, 1e-5, "rows not unit norm")
    approx(xn.norm(dim=-1).min(), 1.0, 1e-5, "rows not unit norm")

    z = torch.zeros(1, EMB, device=DEVICE)
    assert torch.isfinite(l2_normalize(z)).all(), "zero vector produced NaN/Inf"

    v = torch.randn(EMB, device=DEVICE)
    s = cosine_sim_to_vector(v.unsqueeze(0), v)
    approx(s[0], 1.0, 1e-5, "self-similarity != 1")
    s = cosine_sim_to_vector((-v).unsqueeze(0), v)
    approx(s[0], -1.0, 1e-5, "anti-similarity != -1")

    a = torch.randn(11, EMB, device=DEVICE)
    ref = torch.stack([
        torch.nn.functional.cosine_similarity(a[i], v, dim=0) for i in range(11)
    ])
    got = cosine_sim_to_vector(a, v)
    assert torch.allclose(got, ref, atol=1e-5), "cosine_sim_to_vector != torch reference"

    b = torch.randn(11, EMB, device=DEVICE)
    ref_rw = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    assert torch.allclose(cosine_sim_rowwise(a, b), ref_rw, atol=1e-5), "rowwise mismatch"

    # fp16 input must still be compared in fp32
    assert cosine_sim_to_vector(a.half(), v.half()).dtype == torch.float32


@test("ops: mean_pool_frame == \\bar v_t")
def t_mean_pool():
    x = torch.randn(N_PATCH, EMB, device=DEVICE)
    pooled = mean_pool_frame(x)
    shape_is(pooled, (EMB,), "frame_feat")
    assert torch.allclose(pooled, x.float().mean(0), atol=1e-6)


@test("ops: build_mrope_coords -> absolute (t, h, w), raster order")
def t_mrope_build():
    co = build_mrope_coords(t_index=7, grid_h=QWEN_GRID[0], grid_w=QWEN_GRID[1], device=DEVICE)
    shape_is(co, (N_TOKEN, 3), "mrope")
    assert co.dtype == torch.long, f"mrope must be int64, got {co.dtype}"
    assert (co[:, 0] == 7).all(), "temporal coordinate not constant within a frame"
    # raster (row-major) order over (h, w)
    assert co[0].tolist() == [7, 0, 0]
    assert co[1].tolist() == [7, 0, 1]
    assert co[QWEN_GRID[1]].tolist() == [7, 1, 0]
    assert co[-1].tolist() == [7, QWEN_GRID[0] - 1, QWEN_GRID[1] - 1]
    assert co[:, 1].max().item() == QWEN_GRID[0] - 1
    assert co[:, 2].max().item() == QWEN_GRID[1] - 1
    # every (h, w) appears exactly once
    assert len({tuple(r) for r in co[:, 1:].tolist()}) == N_TOKEN


@test("ops: resample CLIP 16x16 grid -> Qwen 8x8 grid")
def t_resample():
    m = torch.zeros(CLIP_GRID, dtype=torch.bool, device=DEVICE)
    m[0, 0] = True                      # one patch in the top-left 2x2 block

    any_ = resample_patch_mask(m, CLIP_GRID, QWEN_GRID, "any")
    shape_is(any_, (N_TOKEN,), "resampled[any]")
    assert any_.sum().item() == 1 and any_[0].item(), "any-pooling lost the patch"

    all_ = resample_patch_mask(m, CLIP_GRID, QWEN_GRID, "all")
    assert all_.sum().item() == 0, "all-pooling kept a partially-covered block"

    m[0, 1], m[1, 0], m[1, 1] = True, True, True   # complete the 2x2 block
    assert resample_patch_mask(m, CLIP_GRID, QWEN_GRID, "all").sum().item() == 1
    assert resample_patch_mask(m, CLIP_GRID, QWEN_GRID, "majority").sum().item() == 1

    # majority needs > 50%: exactly 2 of 4 must fail
    m2 = torch.zeros(CLIP_GRID, dtype=torch.bool, device=DEVICE)
    m2[0, 0], m2[0, 1] = True, True
    assert resample_patch_mask(m2, CLIP_GRID, QWEN_GRID, "majority").sum().item() == 0
    assert resample_patch_mask(m2, CLIP_GRID, QWEN_GRID, "any").sum().item() == 1

    # flat input and identity resample
    flat = m.reshape(-1)
    tensors_equal(
        resample_patch_mask(flat, CLIP_GRID, QWEN_GRID, "any"),
        resample_patch_mask(m, CLIP_GRID, QWEN_GRID, "any"),
        "flat vs 2D input disagree",
    )
    ident = resample_patch_mask(flat, CLIP_GRID, CLIP_GRID, "any")
    tensors_equal(ident, flat, "identity resample changed the mask")

    # batched
    batch = torch.stack([flat, ~flat], 0)
    shape_is(resample_patch_mask(batch, CLIP_GRID, QWEN_GRID, "any"), (2, N_TOKEN), "batched")

    # ordering: block (r, c) of the target must come from source rows 2r..2r+1
    m3 = torch.zeros(CLIP_GRID, dtype=torch.bool, device=DEVICE)
    m3[5, 9] = True                                     # -> target (2, 4) -> index 2*8+4
    out = resample_patch_mask(m3, CLIP_GRID, QWEN_GRID, "any")
    assert out.nonzero().item() == 2 * QWEN_GRID[1] + 4, "block ordering is wrong"


@test("ops: temporal merge folding (Qwen 2-frame merge)")
def t_temporal_merge():
    m = torch.zeros(4, N_TOKEN, dtype=torch.bool, device=DEVICE)
    m[0, 0] = True
    m[3, 5] = True
    any_ = expand_mask_to_temporal_merge(m, 2, "any")
    shape_is(any_, (2, N_TOKEN), "merged[any]")
    assert any_[0, 0].item() and any_[1, 5].item()
    assert expand_mask_to_temporal_merge(m, 2, "all").sum().item() == 0
    try:
        expand_mask_to_temporal_merge(torch.zeros(3, N_TOKEN, dtype=torch.bool), 2)
        raise AssertionError("odd frame count should have been rejected")
    except AssertionError as e:
        if "should have been rejected" in str(e):
            raise


# =========================================================================== #
# 2. DSHMemory
# =========================================================================== #

@test("DSH: EMA matches the closed-form recursion exactly")
def t_dsh_ema():
    alpha = 0.1
    dsh = DSHMemory(alpha=alpha, init="self", space="raw")
    feats = [torch.randn(N_PATCH, EMB, device=DEVICE) for _ in range(6)]

    # independent reference: v_dsh_0 = v_0 ; v_dsh_t = a*v_t + (1-a)*v_dsh_{t-1}
    ref = feats[0].clone().float()
    dsh.update(feats[0])
    assert torch.allclose(dsh.state, ref, atol=1e-6), "init state != v_0"

    for k in range(1, 6):
        ref = alpha * feats[k].float() + (1 - alpha) * ref
        dsh.update(feats[k])
        assert torch.allclose(dsh.state, ref, atol=1e-5), f"EMA diverged at frame {k}"
    vprint(f"6-frame EMA max abs err = {(dsh.state - ref).abs().max():.3e}")
    assert dsh.num_frames_seen == 6
    approx(dsh.effective_memory_frames, 10.0, 1e-9, "1/alpha")


@test("DSH: compare-then-update ordering (t vs t-1, never t vs t)")
def t_dsh_ordering():
    dsh = DSHMemory(alpha=0.1)
    v0 = l2_normalize(torch.randn(N_PATCH, EMB, device=DEVICE))
    v1 = l2_normalize(torch.randn(N_PATCH, EMB, device=DEVICE))

    dsh.update(v0)
    state_before = dsh.state.clone()

    sim_peek = dsh.similarity(v1)                       # must not mutate
    assert torch.equal(dsh.state, state_before), "similarity() mutated the DSH state"

    sim_step = dsh.step(v1)                             # compare, then update
    assert torch.allclose(sim_peek, sim_step, atol=1e-6), "step() != similarity() then update()"
    assert not torch.equal(dsh.state, state_before), "step() failed to advance the history"

    # the returned similarity is against t-1, so it must equal cos(v1, v0)
    ref = cosine_sim_rowwise(l2_normalize(v1), l2_normalize(v0))
    assert torch.allclose(sim_step, ref, atol=1e-5), (
        "similarity was computed against the POST-update state -- current frame "
        "leaked into its own history"
    )
    vprint(f"sim(v1, dsh_0) mean = {sim_step.mean():.4f}")


@test("DSH: first frame, reset, geometry guard, masked update")
def t_dsh_edges():
    dsh = DSHMemory(alpha=0.1)
    assert not dsh.is_initialized
    v0 = l2_normalize(torch.randn(N_PATCH, EMB, device=DEVICE))
    sim = dsh.similarity(v0)
    shape_is(sim, (N_PATCH,), "first-frame sim")
    assert torch.allclose(sim, torch.ones_like(sim)), "first frame should report sim == 1"

    dsh.step(v0)
    assert dsh.is_initialized
    dsh.reset()
    assert not dsh.is_initialized and dsh.num_frames_seen == 0, "reset() left state behind"

    dsh.update(v0)
    try:
        dsh.update(torch.randn(N_PATCH + 1, EMB, device=DEVICE))
        raise AssertionError("geometry change should have raised")
    except ValueError:
        pass

    # masked update: unmasked locations must be byte-identical
    dsh.reset()
    dsh.update(v0)
    prev = dsh.state.clone()
    mask = torch.zeros(N_PATCH, dtype=torch.bool, device=DEVICE)
    mask[:10] = True
    dsh.update(l2_normalize(torch.randn(N_PATCH, EMB, device=DEVICE)), update_mask=mask)
    assert torch.equal(dsh.state[10:], prev[10:]), "masked-out patches were updated"
    assert not torch.equal(dsh.state[:10], prev[:10]), "masked-in patches were not updated"


@test("DSH: 600-frame fp32 stability (no drift at the tau_temp boundary)")
def t_dsh_stability():
    dsh = DSHMemory(alpha=0.1)
    base = l2_normalize(torch.randn(N_PATCH, EMB, device=DEVICE))
    for _ in range(600):
        dsh.step(l2_normalize(base + 0.001 * torch.randn_like(base)))
    assert torch.isfinite(dsh.state).all(), "DSH state went non-finite over a long stream"
    sim = dsh.similarity(base)
    assert sim.min() > 0.9, f"DSH drifted off a static scene: min sim {sim.min():.4f}"
    vprint(f"after 600 frames: sim to base in [{sim.min():.5f}, {sim.max():.5f}]")


# =========================================================================== #
# 3. QDP
# =========================================================================== #

@test("QDP: M_sem uses the frame-adaptive MEAN threshold, strictly")
def t_qdp_sem():
    feats = l2_normalize(torch.randn(N_PATCH, EMB, device=DEVICE))
    q = l2_normalize(torch.randn(EMB, device=DEVICE))

    m_sem, sem_sim, thr = QDPModule.semantic_mask(feats, q)
    shape_is(m_sem, (N_PATCH,), "M_sem")
    assert m_sem.dtype == torch.bool

    # independent recomputation of sim(q, v^i) and of (1/N) sum_j sim(q, v^j)
    ref_sim = torch.stack([
        torch.nn.functional.cosine_similarity(feats[i], q, dim=0) for i in range(N_PATCH)
    ])
    assert torch.allclose(sem_sim, ref_sim, atol=1e-5), "sem_sim != reference cosine"
    approx(thr, ref_sim.mean(), 1e-6, "threshold is not the frame mean")
    tensors_equal(m_sem, ref_sim > ref_sim.mean(), "M_sem != (sim > mean)")

    # strictness: a constant frame has sim == mean everywhere -> nothing kept
    const = feats[0:1].expand(N_PATCH, EMB).contiguous()
    m_c, _, _ = QDPModule.semantic_mask(const, q)
    assert m_c.sum().item() == 0, "'>' is not strict: ties were kept"
    vprint(f"M_sem keep = {m_sem.float().mean():.4f} (mean-threshold -> ~0.5 expected)")
    assert 0.3 < m_sem.float().mean().item() < 0.7


@test("QDP: M_temp = 1[sim < tau_temp], strictly, vs the DSH")
def t_qdp_temp():
    cfg = QDPConfig(alpha=0.1, tau_temp=0.90, first_frame_all_novel=True)
    qdp = QDPModule(cfg)

    v0 = l2_normalize(torch.randn(N_PATCH, EMB, device=DEVICE))
    m0, sim0, first = qdp.temporal_mask(v0)
    assert first and m0.all(), "frame 0 must be fully novel (no history exists)"
    assert torch.allclose(sim0, torch.ones_like(sim0)), "frame-0 raw sim should be 1.0"

    # identical frame -> similarity ~1 -> nothing novel
    m1, sim1, _ = qdp.temporal_mask(v0)
    assert sim1.min() > 0.99, f"static frame reported novelty: min sim {sim1.min():.4f}"
    assert m1.sum().item() == 0, "static frame was flagged novel"

    # fully new content -> novel
    m2, sim2, _ = qdp.temporal_mask(l2_normalize(torch.randn(N_PATCH, EMB, device=DEVICE)))
    assert m2.float().mean().item() > 0.95, f"new content not flagged novel ({m2.float().mean():.3f})"
    tensors_equal(m2, sim2 < cfg.tau_temp, "M_temp != (sim < tau_temp)")

    # Exact boundary: sim == tau_temp must be REJECTED (strict <).
    # Self-cosine is only ~1.0 to fp tolerance, so anchoring tau_temp at 1.0
    # would test float noise. Instead: observe the actual similarities, then set
    # tau_temp to one of those exact values and check that patch flips off.
    a = l2_normalize(torch.randn(N_PATCH, EMB, device=DEVICE))
    b = l2_normalize(torch.randn(N_PATCH, EMB, device=DEVICE))
    probe = QDPModule(QDPConfig(tau_temp=1.0, first_frame_all_novel=False))
    probe.temporal_mask(a)
    _, sims, _ = probe.temporal_mask(b)

    j = int(sims.argsort()[len(sims) // 2])          # a mid-ranked patch
    tau_exact = float(sims[j])
    exact = QDPModule(QDPConfig(tau_temp=tau_exact, first_frame_all_novel=False))
    exact.temporal_mask(a)
    m_eq, sim_eq, _ = exact.temporal_mask(b)
    approx(sim_eq[j], tau_exact, 1e-7, "probe run was not reproducible")
    assert not m_eq[j].item(), "'<' is not strict: sim == tau_temp was kept"
    lower = int(sims.argmin())
    assert m_eq[lower].item(), "a strictly-lower similarity was rejected"
    vprint(f"boundary: sim[{j}] == tau_temp == {tau_exact:.8f} -> rejected")


@test("QDP: M_QDP == M_sem AND M_temp (an intersection)")
def t_qdp_conjunction():
    enc = make_encoder(seed=1)
    q = enc.encode_text("what is being poured")
    qdp = QDPModule(QDPConfig(alpha=0.1, tau_temp=0.90))

    for spec in make_stream(enc):
        out = qdp(enc.encode_frame(**spec).patch_feats, q)
        shape_is(out.m_qdp, (N_PATCH,), f"M_QDP[f{spec['t']}]")
        tensors_equal(out.m_qdp, out.m_sem & out.m_temp, "M_QDP != M_sem AND M_temp")
        # an intersection can never exceed either operand
        assert out.keep_rate <= out.keep_rate_sem + 1e-9
        assert out.keep_rate <= out.keep_rate_temp + 1e-9
        # The moving distractor is kept by M_temp and killed by M_sem, so on the
        # settled frames the intersection must be STRICTLY smaller than M_temp.
        # If these ever coincide, the semantic filter is a no-op.
        if spec["t"] in (1, 3):
            assert out.keep_rate < out.keep_rate_temp - 1e-6, (
                f"f{spec['t']}: M_QDP ({out.keep_rate:.4f}) did not shrink M_temp "
                f"({out.keep_rate_temp:.4f}) -- the semantic filter is inert"
            )
            kept_idx = set(out.m_qdp.nonzero().flatten().tolist())
            assert kept_idx <= set(RELEVANT), (
                "QDP kept tokens outside the query-relevant region"
            )
            assert not (kept_idx & set(DISTRACTOR)), (
                "QDP kept the moving-but-irrelevant distractor -- this is exactly "
                "the TimeChat-Online failure mode the paper targets"
            )
        approx(out.keep_rate, out.m_qdp.float().mean(), 1e-6, "keep_rate != mean(M_QDP)")
        assert out.num_kept == int(out.m_qdp.sum())
        vprint(
            f"f{spec['t']}: sem={out.keep_rate_sem:.3f} temp={out.keep_rate_temp:.3f} "
            f"qdp={out.keep_rate:.3f} thr={out.sem_threshold:+.4f}"
        )


@test("QDP: reset() fully clears stream state")
def t_qdp_reset():
    enc = make_encoder(seed=2)
    q = enc.encode_text("q")
    qdp = QDPModule(QDPConfig())
    specs = make_stream(enc)

    first_run = [qdp(enc.encode_frame(**s).patch_feats, q).keep_rate for s in specs]
    assert qdp.frame_idx == FRAMES

    qdp.reset()
    assert qdp.frame_idx == 0 and not qdp.dsh.is_initialized, "reset() incomplete"

    enc2 = make_encoder(seed=2)          # same seed -> identical stream
    second_run = [qdp(enc2.encode_frame(**s).patch_feats, enc2.encode_text("q")).keep_rate
                  for s in specs]
    for a, b in zip(first_run, second_run):
        approx(a, b, 1e-6, "run not reproducible after reset()")


@test("QDP: DSH beats frame-to-frame on the subtle-event case (paper A.7)")
def t_qdp_subtle_event():
    """alpha=1.0 IS frame-to-frame diffing -- TimeChat-Online's regime, and the
    worst point of the paper's Fig. 3 sweep. A slow cumulative change should be
    invisible to it and visible to the alpha=0.1 DSH."""
    n, d = 64, 128
    base = l2_normalize(torch.randn(n, d, device=DEVICE))
    step = l2_normalize(torch.randn(n, d, device=DEVICE)) * 0.02   # tiny per-frame delta

    def novelty_trace(alpha: float) -> List[float]:
        dsh = DSHMemory(alpha=alpha)
        out = []
        cur = base.clone()
        for k in range(12):
            cur = l2_normalize(base + step * k)
            out.append(float(dsh.step(cur).mean()))
        return out

    f2f = novelty_trace(1.0)      # naive frame-to-frame
    smoothed = novelty_trace(0.1)  # DSH
    # lower similarity == more novelty detected
    assert smoothed[-1] < f2f[-1], (
        f"DSH ({smoothed[-1]:.5f}) did not detect more cumulative drift than "
        f"frame-to-frame ({f2f[-1]:.5f})"
    )
    vprint(f"frame 11 novelty-sim: f2f={f2f[-1]:.5f}  dsh(a=0.1)={smoothed[-1]:.5f}")


@test("QDP: min_keep_per_frame safety valve (non-paper, off by default)")
def t_qdp_min_keep():
    assert QDPConfig().min_keep_per_frame == 0, "safety valve must default OFF"
    qdp = QDPModule(QDPConfig(tau_temp=-1.0, min_keep_per_frame=8))   # temp gate rejects all
    feats = l2_normalize(torch.randn(N_PATCH, EMB, device=DEVICE))
    q = l2_normalize(torch.randn(EMB, device=DEVICE))
    qdp(feats, q)                                                    # frame 0 (all novel)
    out = qdp(feats, q)                                              # frame 1: would be empty
    assert out.num_kept == 8, f"floor not enforced: kept {out.num_kept}"
    assert out.forced_keep == 8, "forced_keep not reported"
    # revived tokens must be the top-8 by semantic similarity
    top8 = set(torch.topk(out.sem_sim, 8).indices.tolist())
    assert set(out.m_qdp.nonzero().flatten().tolist()) == top8, "wrong tokens revived"


# =========================================================================== #
# 4. RTAR
# =========================================================================== #

@test("RTAR: R_t, D_t and T_t = R_t AND D_t, strictly")
def t_rtar_gates():
    cfg = RTARConfig(tau_rel=0.60, tau_den=0.15, relevance_mode="raw", debounce_enabled=False)
    rt = RTARModule(cfg)

    cases = [
        # (relevance, keep_rate) -> (R, D, T)
        (0.90, 0.50, True,  True,  True),    # both gates pass
        (0.90, 0.05, True,  False, False),   # relevant but no new information
        (0.10, 0.50, False, True,  False),   # dynamic but irrelevant  <- the shock case
        (0.10, 0.05, False, False, False),
        (0.60, 0.50, False, True,  False),   # exactly tau_rel -> rejected (strict >)
        (0.90, 0.15, True,  False, False),   # exactly tau_den -> rejected (strict >)
    ]
    for rel, kr, exp_r, exp_d, exp_t in cases:
        o = rt(keep_rate=kr, relevance_raw=rel, timestamp=float(rt.frame_idx))
        assert (o.r_t, o.d_t, o.trigger) == (exp_r, exp_d, exp_t), (
            f"rel={rel} keep={kr}: got R={o.r_t} D={o.d_t} T={o.trigger}, "
            f"expected R={exp_r} D={exp_d} T={exp_t}"
        )
    vprint("all 6 gate truth-table rows correct (including both boundaries)")


@test("RTAR: relevance computed from mean-pooled frame feature")
def t_rtar_relevance_source():
    rt = RTARModule(RTARConfig(relevance_mode="raw", debounce_enabled=False))
    feats = l2_normalize(torch.randn(N_PATCH, EMB, device=DEVICE))
    q = l2_normalize(torch.randn(EMB, device=DEVICE))

    o = rt(keep_rate=0.5, frame_feat=feats, query_emb=q)
    ref = torch.nn.functional.cosine_similarity(mean_pool_frame(feats), q, dim=0)
    approx(o.relevance_raw, ref, 1e-5, "R_t not computed from \\bar v_t = mean-pool")

    # passing the already-pooled vector must agree
    o2 = rt(keep_rate=0.5, frame_feat=mean_pool_frame(feats), query_emb=q)
    approx(o2.relevance_raw, o.relevance_raw, 1e-6, "[N,D] and [D] paths disagree")


@test("RTAR: gate_mode reproduces the three Table-5 ablation rows")
def t_rtar_gate_modes():
    seq = [(0.9, 0.02), (0.1, 0.9), (0.9, 0.9), (0.1, 0.02)]
    got = {}
    for mode in ("full", "relevance_only", "density_only"):
        rt = RTARModule(RTARConfig(relevance_mode="raw", debounce_enabled=False),
                        gate_mode=mode)
        got[mode] = [rt(keep_rate=k, relevance_raw=r, timestamp=float(i)).trigger
                     for i, (r, k) in enumerate(seq)]
    assert got["full"] == [False, False, True, False]
    assert got["relevance_only"] == [True, False, True, False]
    assert got["density_only"] == [False, True, True, False]
    # density-only fires on the visually-loud, semantically-empty frame; full does not
    assert got["density_only"][1] and not got["full"][1], (
        "the shock frame must be suppressed by the relevance gate"
    )


@test("RTAR: Appendix A.4 event debouncing (first trigger per interval)")
def t_rtar_debounce():
    rt = RTARModule(RTARConfig(relevance_mode="raw", debounce_enabled=True, event_gap_s=1.0))
    # triggers at t = 0,1,2 (one event) then 10,11 (a second event)
    schedule = [(0.0, True), (1.0, True), (2.0, True),
                (3.0, False), (4.0, False),
                (10.0, True), (11.0, True)]
    fired = []
    for ts, want in schedule:
        o = rt(keep_rate=0.9 if want else 0.0, relevance_raw=0.9 if want else 0.0, timestamp=ts)
        fired.append(o.fired)
    assert fired == [True, False, False, False, False, True, False], f"got {fired}"
    assert rt.num_events == 2, f"expected 2 events, got {rt.num_events}"
    vprint("7 frames -> 5 raw triggers -> 2 response points")


@test("RTAR: relevance calibration modes + the tau_rel warning is real")
def t_rtar_calibration():
    # A realistic OpenCLIP image-text cosine band. tau_rel=0.60 is unreachable.
    rels = [0.18, 0.22, 0.31, 0.27, 0.20, 0.35, 0.19, 0.24, 0.29, 0.33, 0.21, 0.26]

    raw = RTARModule(RTARConfig(relevance_mode="raw", debounce_enabled=False))
    for i, r in enumerate(rels):
        raw(keep_rate=0.9, relevance_raw=r, timestamp=float(i))
    d = raw.diagnostics()
    assert d["n_R"] == 0, "sanity check broken: raw cosines should never exceed 0.60"
    assert d["n_trigger"] == 0
    vprint(f"raw mode: 0/{len(rels)} triggers -- calibration IS required, as documented")

    mm = RTARModule(RTARConfig(relevance_mode="running_minmax", calibration_warmup=4,
                               debounce_enabled=False))
    for i, r in enumerate(rels):
        mm(keep_rate=0.9, relevance_raw=r, timestamp=float(i))
    dm = mm.diagnostics()
    assert dm["n_trigger"] > 0, "running_minmax produced no triggers either"
    assert all(0.0 <= h.relevance <= 1.0 for h in mm.history), "calibrated value out of [0,1]"
    # frames inside the warmup window must never fire
    assert all(not h.trigger for h in mm.history if h.calibrating), "fired while calibrating"
    vprint(f"running_minmax: {dm['n_trigger']}/{len(rels)} triggers")

    for mode in ("logit_softmax", "running_zscore"):
        m = RTARModule(RTARConfig(relevance_mode=mode, calibration_warmup=4,
                                  debounce_enabled=False))
        for i, r in enumerate(rels):
            o = m(keep_rate=0.9, relevance_raw=r, timestamp=float(i))
            assert math.isfinite(o.relevance), f"{mode} produced a non-finite score"


@test("RTAR: reset() clears calibrator, debouncer and history")
def t_rtar_reset():
    rt = RTARModule(RTARConfig(relevance_mode="running_minmax"))
    for i in range(20):
        rt(keep_rate=0.9, relevance_raw=0.2 + 0.01 * i, timestamp=float(i))
    assert rt.frame_idx == 20 and len(rt.history) == 20
    rt.reset()
    assert rt.frame_idx == 0 and rt.history == [] and rt.num_events == 0, "reset() incomplete"
    assert rt.diagnostics() == {"frames": 0}


# =========================================================================== #
# 5. TokenBuffer -- the M-ROPE invariant
# =========================================================================== #

def _signed_tokens(frame_idx: int, mrope: torch.Tensor, dv: int) -> torch.Tensor:
    """Visual tokens whose first three channels ENCODE their own (t, h, w).

    This is the trick that makes the M-ROPE correspondence checkable: after any
    slicing, tokens[:, :3] must still equal mrope exactly. If the buffer ever
    slices tokens and coordinates with different index sets, this breaks loudly.
    """
    m = mrope.shape[0]
    tok = torch.randn(m, dv, device=DEVICE)
    tok[:, :3] = mrope.to(tok.dtype)
    tok[:, 3] = frame_idx
    return tok


@test("Buffer: append / capacity / stats bookkeeping")
def t_buffer_basic():
    buf = TokenBuffer(BufferConfig(device=str(DEVICE)))
    assert len(buf) == 0 and buf.num_tokens == 0

    for i in range(FRAMES):
        rope = build_mrope_coords(i, *QWEN_GRID, device=DEVICE)
        mask = torch.zeros(N_TOKEN, dtype=torch.bool, device=DEVICE)
        mask[: (i + 1) * 5] = True
        buf.append(frame_idx=i, timestamp=float(i), tokens=_signed_tokens(i, rope, DV),
                   mrope=rope, mask=mask)

    assert len(buf) == FRAMES
    assert buf.num_tokens == FRAMES * N_TOKEN
    assert buf.num_kept_tokens == sum((i + 1) * 5 for i in range(FRAMES))
    s = buf.stats()
    approx(s["keep_rate"], buf.num_kept_tokens / buf.num_tokens, 1e-9, "stats keep_rate")
    vprint(f"buffer: {s['frames']} frames, {s['tokens_unpruned']} tokens, {s['memory_mb']} MB")

    buf.clear()
    assert len(buf) == 0 and buf.num_tokens == 0

    # capped buffer evicts oldest-first
    cap = TokenBuffer(BufferConfig(device=str(DEVICE), max_frames=3))
    for i in range(6):
        rope = build_mrope_coords(i, *QWEN_GRID, device=DEVICE)
        cap.append(frame_idx=i, timestamp=float(i), tokens=_signed_tokens(i, rope, 8),
                   mrope=rope, mask=torch.ones(N_TOKEN, dtype=torch.bool, device=DEVICE))
    assert len(cap) == 3, f"max_frames not enforced ({len(cap)})"
    assert cap.evicted == (3, 3 * N_TOKEN), f"eviction miscounted: {cap.evicted}"
    kept_ids = [f.frame_idx for f in cap._frames]
    assert kept_ids == [3, 4, 5], f"eviction is not oldest-first: {kept_ids}"


@test("Buffer: JIT flush slices tokens and M-ROPE with ONE index set")
def t_buffer_mrope_invariant():
    """The paper's Sec. 3.2 guarantee: surviving tokens keep their ORIGINAL
    {t, h, w}. Verified by construction -- each token carries its own coordinate
    in its first three channels."""
    buf = TokenBuffer(BufferConfig(device=str(DEVICE)))
    g = torch.Generator(device="cpu").manual_seed(7)
    expected: List[Tuple[int, int, int]] = []

    for i in range(FRAMES):
        rope = build_mrope_coords(i, *QWEN_GRID, device=DEVICE)
        mask = (torch.rand(N_TOKEN, generator=g) > 0.45).to(DEVICE)
        buf.append(frame_idx=i, timestamp=float(i) / 1.0,
                   tokens=_signed_tokens(i, rope, DV), mrope=rope, mask=mask)
        expected += [tuple(r) for r in rope[mask].tolist()]

    res = buf.flush(apply_mask=True, flush_frame_idx=FRAMES - 1,
                    flush_timestamp=float(FRAMES - 1))
    k = res.num_tokens_after

    shape_is(res.tokens, (k, DV), "flushed tokens")
    shape_is(res.mrope, (k, 3), "flushed mrope")
    shape_is(res.frame_ids, (k,), "frame_ids")
    shape_is(res.timestamps, (k,), "timestamps")
    assert res.num_tokens_before == FRAMES * N_TOKEN

    # ---- THE core assertion of Milestone 1 -------------------------------- #
    assert torch.equal(res.tokens[:, :3].round().long(), res.mrope), (
        "M-ROPE slicing does NOT match visual-token slicing -- tokens and their "
        "positional coordinates were indexed differently"
    )
    # ----------------------------------------------------------------------- #

    assert [tuple(r) for r in res.mrope.tolist()] == expected, (
        "flush order does not follow (frame-major, then raster) buffer order"
    )
    assert torch.equal(res.frame_ids, res.mrope[:, 0]), "frame_ids != M-ROPE t-coordinate"
    assert torch.equal(res.tokens[:, 3].round().long(), res.frame_ids), "provenance mismatch"

    # coordinates must stay ABSOLUTE (not renumbered to be dense)
    ts = res.mrope[:, 0].unique().tolist()
    assert ts == list(range(FRAMES)), f"temporal coords renumbered: {ts}"
    assert res.mrope[:, 1].max().item() <= QWEN_GRID[0] - 1

    # Targeted gap test: prune every token in rows h=0..2 of a single frame and
    # confirm the survivors still report h >= 3. A pipeline that renumbered
    # coordinates after pruning would report h starting at 0 and silently tell
    # the LLM those patches are at the top of the frame.
    gap = TokenBuffer(BufferConfig(device=str(DEVICE)))
    rope = build_mrope_coords(9, *QWEN_GRID, device=DEVICE)
    keep_rows = rope[:, 1] >= 3
    gap.append(frame_idx=9, timestamp=9.0, tokens=_signed_tokens(9, rope, 8),
               mrope=rope, mask=keep_rows)
    gres = gap.flush(apply_mask=True)
    got_h = sorted(set(gres.mrope[:, 1].tolist()))
    assert got_h == [3, 4, 5, 6, 7], f"h coordinates were renumbered: {got_h}"
    assert (gres.mrope[:, 0] == 9).all(), "t coordinate was renumbered"
    assert torch.equal(gres.tokens[:, :3].round().long(), gres.mrope)
    vprint(f"pruned rows 0-2 -> surviving h = {got_h} (absolute, gap preserved)")
    approx(res.keep_rate, k / (FRAMES * N_TOKEN), 1e-9, "aggregate keep rate")
    vprint(f"flush: {res.num_tokens_before} -> {k} tokens (keep {res.keep_rate:.4f})")


@test("Buffer: apply_mask=False gives the 100%-keep control")
def t_buffer_no_mask():
    buf = TokenBuffer(BufferConfig(device=str(DEVICE)))
    for i in range(FRAMES):
        rope = build_mrope_coords(i, *QWEN_GRID, device=DEVICE)
        buf.append(frame_idx=i, timestamp=float(i), tokens=_signed_tokens(i, rope, 16),
                   mrope=rope, mask=torch.zeros(N_TOKEN, dtype=torch.bool, device=DEVICE))
    res = buf.flush(apply_mask=False)
    assert res.num_tokens_after == FRAMES * N_TOKEN, "unmasked flush lost tokens"
    approx(res.keep_rate, 1.0, 1e-9, "control keep rate != 1.0")
    assert torch.equal(res.tokens[:, :3].round().long(), res.mrope)

    # and the all-reject case must not crash
    empty = buf.flush(apply_mask=True)
    assert empty.num_tokens_after == 0
    shape_is(empty.tokens, (0, 16), "empty flush tokens")
    shape_is(empty.mrope, (0, 3), "empty flush mrope")
    approx(empty.keep_rate, 0.0, 1e-9, "empty keep rate")


@test("Buffer: masks stay mutable until flush (JIT, not eager)")
def t_buffer_jit_semantics():
    buf = TokenBuffer(BufferConfig(device=str(DEVICE), store_unpruned=True))
    rope = build_mrope_coords(0, *QWEN_GRID, device=DEVICE)
    m = torch.zeros(N_TOKEN, dtype=torch.bool, device=DEVICE)
    m[:4] = True
    buf.append(frame_idx=0, timestamp=0.0, tokens=_signed_tokens(0, rope, 8), mrope=rope, mask=m)

    assert buf.num_tokens == N_TOKEN, "tokens were pruned at append time -- not JIT"
    assert buf.flush().num_tokens_after == 4

    m2 = torch.zeros(N_TOKEN, dtype=torch.bool, device=DEVICE)
    m2[:9] = True
    buf.update_mask(0, m2)
    assert buf.flush().num_tokens_after == 9, "mask revision had no effect"

    # eager mode (non-paper) prunes on append
    eager = TokenBuffer(BufferConfig(device=str(DEVICE), store_unpruned=False))
    eager.append(frame_idx=0, timestamp=0.0, tokens=_signed_tokens(0, rope, 8), mrope=rope, mask=m)
    assert eager.num_tokens == 4, "store_unpruned=False did not prune eagerly"
    assert torch.equal(eager.flush().tokens[:, :3].round().long(), eager.flush().mrope)

    # consume=True empties the buffer
    buf.flush(consume=True)
    assert len(buf) == 0


# =========================================================================== #
# 6. Integration
# =========================================================================== #

@test("INTEGRATION: 5-frame stream, encoder -> QDP -> resample -> RTAR -> buffer")
def t_integration():
    cfg = QueryStreamConfig().validate()
    cfg.rtar.relevance_mode = "running_minmax"     # raw would never trigger (see rtar.py)
    cfg.rtar.calibration_warmup = 2
    cfg.buffer.device = str(DEVICE)

    enc = make_encoder(seed=11)
    q = enc.encode_text("how to properly open a bottle of champagne")
    qdp = QDPModule(cfg.qdp)
    rtar = RTARModule(cfg.rtar)
    buf = TokenBuffer(cfg.buffer)

    flushes = []
    print()
    print(f"      {'f':>2} {'t(s)':>5} {'sem':>6} {'temp':>6} {'qdp':>6} "
          f"{'tok':>6} {'rel':>6} {'R':>2} {'D':>2} {'T':>2} {'fire':>5}")
    print(f"      {'-'*2} {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} "
          f"{'-'*2} {'-'*2} {'-'*2} {'-'*5}")

    for spec in make_stream(enc):
        i = spec["t"]
        ts = i / cfg.fps                                    # 1 FPS
        ef = enc.encode_frame(**spec)

        # --- QDP on the CLIP patch grid --------------------------------- #
        out = qdp(ef.patch_feats, q)
        shape_is(out.m_qdp, (N_PATCH,), f"M_QDP f{i}") if VERBOSE else None

        # --- resample onto the Qwen visual-token grid -------------------- #
        tok_mask = resample_patch_mask(out.m_qdp, CLIP_GRID, QWEN_GRID, cfg.mask_resample_mode)
        assert tok_mask.shape == (N_TOKEN,)
        # "any" pooling can only ever add coverage
        assert tok_mask.float().mean() >= out.keep_rate - 1e-6, (
            "any-pooling reduced coverage, which is impossible"
        )

        # --- RTAR ------------------------------------------------------- #
        dec = rtar(keep_rate=out.keep_rate, frame_feat=ef.patch_feats,
                   query_emb=q, timestamp=ts)

        # --- buffer the UNPRUNED tokens + the mask ----------------------- #
        rope = build_mrope_coords(i, *QWEN_GRID, device=DEVICE)
        buf.append(frame_idx=i, timestamp=ts, tokens=_signed_tokens(i, rope, DV),
                   mrope=rope, mask=tok_mask,
                   keep_rate=float(tok_mask.float().mean()))

        print(f"      {i:>2} {ts:>5.1f} {out.keep_rate_sem:>6.3f} {out.keep_rate_temp:>6.3f} "
              f"{out.keep_rate:>6.3f} {float(tok_mask.float().mean()):>6.3f} "
              f"{dec.relevance:>6.3f} {int(dec.r_t):>2} {int(dec.d_t):>2} "
              f"{int(dec.trigger):>2} {str(dec.fired):>5}")

        # --- JIT flush on trigger ---------------------------------------- #
        if dec.fired:
            res = buf.flush(apply_mask=True, flush_frame_idx=i, flush_timestamp=ts)
            assert torch.equal(res.tokens[:, :3].round().long(), res.mrope), (
                f"M-ROPE invariant broken at flush on frame {i}"
            )
            assert res.mrope[:, 0].max().item() <= i, "flush leaked a FUTURE frame"
            flushes.append(res)

    print()
    assert buf.num_frames == FRAMES, "buffer did not retain the full stream"
    assert qdp.frame_idx == FRAMES and rtar.frame_idx == FRAMES

    # frame 2 is the semantically-irrelevant shock: it must not open an event
    shock = rtar.history[2]
    assert not shock.fired, (
        "the shock frame triggered a response -- the relevance gate is not "
        "suppressing visually-salient but query-irrelevant content"
    )

    final = buf.flush(apply_mask=True)
    assert torch.equal(final.tokens[:, :3].round().long(), final.mrope)
    assert final.num_tokens_before == FRAMES * N_TOKEN
    vprint(f"final context: {final.num_tokens_after}/{final.num_tokens_before} tokens "
           f"({final.keep_rate:.1%}); {len(flushes)} mid-stream flush(es); "
           f"{rtar.num_events} event(s)")
    d = rtar.diagnostics()
    assert d["frames"] == FRAMES


@test("INTEGRATION: full pipeline is deterministic across identical runs")
def t_determinism():
    def run():
        enc = make_encoder(seed=99)
        q = enc.encode_text("q")
        qdp = QDPModule(QDPConfig())
        rtar = RTARModule(RTARConfig(relevance_mode="running_minmax", calibration_warmup=2))
        rows = []
        for spec in make_stream(enc):
            ef = enc.encode_frame(**spec)
            o = qdp(ef.patch_feats, q)
            d = rtar(keep_rate=o.keep_rate, frame_feat=ef.patch_feats, query_emb=q,
                     timestamp=float(spec["t"]))
            rows.append((round(o.keep_rate, 8), d.trigger, d.fired))
        return rows

    a, b = run(), run()
    assert a == b, f"pipeline is non-deterministic:\n  {a}\n  {b}"
    vprint(f"identical across 2 runs: {a}")


@test("INTEGRATION: keep rate is monotone in tau_temp (the paper's Table-7 knob)")
def t_keep_rate_monotone():
    """Table 7: tau_temp 0.75 -> 21.6% keep, 0.90 -> 52.9%, 0.98 -> 78.5%.
    The direction must reproduce even if the absolute values are data-dependent."""
    enc = make_encoder(seed=5)
    q = enc.encode_text("q")
    specs = make_stream(enc)
    rates = []
    for tau in (0.75, 0.85, 0.90, 0.95, 0.98):
        qdp = QDPModule(QDPConfig(tau_temp=tau))
        e = make_encoder(seed=5)
        rs = [qdp(e.encode_frame(**s).patch_feats, q).keep_rate for s in specs]
        rates.append(sum(rs) / len(rs))
        vprint(f"tau_temp={tau:.2f} -> mean keep rate {rates[-1]:.4f}")
    for lo, hi in zip(rates, rates[1:]):
        assert hi >= lo - 1e-6, f"keep rate not monotone in tau_temp: {rates}"


@test("INTEGRATION: dtype / device robustness")
def t_dtype_device():
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        feats = l2_normalize(torch.randn(N_PATCH, EMB, device=DEVICE)).to(dt)
        q = l2_normalize(torch.randn(EMB, device=DEVICE)).to(dt)
        qdp = QDPModule(QDPConfig())
        o = qdp(feats, q)
        assert o.m_qdp.dtype == torch.bool
        assert o.sem_sim.dtype == torch.float32, (
            f"{dt}: statistics must be promoted to fp32, got {o.sem_sim.dtype}"
        )
        assert torch.isfinite(o.sem_sim).all() and math.isfinite(o.keep_rate)
        assert qdp.dsh.state.dtype == torch.float32, "DSH state must be fp32"
    vprint("fp32 / fp16 / bf16 all produce fp32 statistics and a bool mask")

    # dim mismatch must be caught, not silently broadcast
    try:
        QDPModule(QDPConfig())(torch.randn(N_PATCH, EMB, device=DEVICE),
                               torch.randn(EMB + 1, device=DEVICE))
        raise AssertionError("dim mismatch should have raised")
    except AssertionError as e:
        if "should have raised" in str(e):
            raise


# =========================================================================== #
# Runner
# =========================================================================== #

def main() -> int:
    global VERBOSE, DEVICE
    ap = argparse.ArgumentParser(description="QueryStream Milestone 1 smoke test")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-k", "--filter", default=None, help="substring filter on test names")
    args = ap.parse_args()

    VERBOSE = args.verbose
    if args.device == "cuda" and not torch.cuda.is_available():
        print(f"{YELLOW}CUDA unavailable, falling back to CPU{RESET}")
        args.device = "cpu"
    DEVICE = torch.device(args.device)
    torch.manual_seed(0)

    tests = [(n, f) for n, f in _REGISTRY if not args.filter or args.filter in n]

    print()
    print("=" * 78)
    print(f"{BLUE}QueryStream -- Milestone 1 smoke test{RESET}")
    print("=" * 78)
    print(f"  torch {torch.__version__} | device {DEVICE} | {len(tests)} tests")
    print(f"  stream      : {FRAMES} frames @ {IMG}x{IMG}, 1 FPS")
    print(f"  CLIP grid   : {CLIP_GRID[0]}x{CLIP_GRID[1]} = {N_PATCH} patches, D={EMB}")
    print(f"  Qwen grid   : {QWEN_GRID[0]}x{QWEN_GRID[1]} = {N_TOKEN} tokens, Dv={DV}")
    c = QueryStreamConfig()
    print(f"  paper params: alpha={c.qdp.alpha} tau_temp={c.qdp.tau_temp} "
          f"tau_rel={c.rtar.tau_rel} tau_den={c.rtar.tau_den}")
    print("=" * 78)

    passed, failed = 0, []
    t_all = time.perf_counter()
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            dt = (time.perf_counter() - t0) * 1e3
            print(f"  {GREEN}PASS{RESET}  {name:<62} {dt:7.1f} ms")
            passed += 1
        except Exception as e:
            dt = (time.perf_counter() - t0) * 1e3
            print(f"  {RED}FAIL{RESET}  {name:<62} {dt:7.1f} ms")
            print(f"        {RED}{type(e).__name__}: {e}{RESET}")
            if VERBOSE:
                traceback.print_exc()
            failed.append(name)

    total = (time.perf_counter() - t_all)
    print("=" * 78)
    if failed:
        print(f"{RED}{len(failed)} FAILED{RESET}, {passed} passed in {total:.2f}s")
        for n in failed:
            print(f"    {RED}x{RESET} {n}")
        print("=" * 78)
        return 1
    print(f"{GREEN}ALL {passed} TESTS PASSED{RESET} in {total:.2f}s")
    print("=" * 78)
    print(f"{DIM}Milestone 1 verified. Ready for Milestone 2 (Qwen2.5-VL wrapper).{RESET}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
