"""Qwen3-VL geometry + pruned-generation probe.

The load-bearing assertion is `test_positions_match_get_rope_index`: our
interleaved M-ROPE builder is compared, element for element, against the
model's OWN `get_rope_index` on a dense prompt. Qwen3-VL chains one vision
block per temporal plane with timestamp text between them, so a builder that
is even slightly wrong about how the running position advances puts every
plane after the first at the wrong coordinate -- and nothing downstream would
notice. This is the Qwen3 counterpart of the `build_dense_mrope` check in
tests/test_integration.py (CLAUDE.md 12.1).

    python3 tests/test_qwen3_geometry.py --device cuda:0
"""
from __future__ import annotations

import argparse, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from querystream.config import BackboneConfig
from querystream.qwen3_wrapper import Qwen3VLWrapper, DEFAULT_QWEN3_MODEL

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok   {name}")
    else:
        FAIL += 1; print(f"  FAIL {name}  {detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-pixels-side", type=int, default=448)
    ap.add_argument("--planes", type=int, default=4)
    a = ap.parse_args()

    torch.manual_seed(0)
    t0 = time.perf_counter()
    w = Qwen3VLWrapper(BackboneConfig(model_id=DEFAULT_QWEN3_MODEL),
                       device=a.device, max_pixels=a.max_pixels_side ** 2)
    print(f"[load] {time.perf_counter()-t0:.1f}s")
    print(" ", w.describe())

    k = w.temporal_patch_size
    H = W = 360
    # ---- 1. encode one chunk ---------------------------------------- #
    frames = torch.randint(0, 255, (k, 3, H, W), dtype=torch.uint8)
    c0 = w.encode_chunk(frames, fps=1.0)
    h_g, w_g = c0.llm_grid_hw
    per_plane = h_g * w_g
    print(f"\n[geometry] grid_thw={c0.grid_thw.tolist()} merged={c0.llm_grid_hw} "
          f"tokens/plane={per_plane} row_width={c0.tokens.shape[1]}")
    check("one chunk -> one plane", int(c0.grid_thw[0]) == 1, c0.grid_thw.tolist())
    check("tokens == h*w", c0.tokens.shape[0] == per_plane, c0.tokens.shape)
    check("row carries token + deepstack",
          c0.tokens.shape[1] == w.hidden_size * (1 + w.n_deepstack), c0.tokens.shape)
    check("mrope rows == tokens", c0.mrope.shape[0] == c0.tokens.shape[0])
    check("plane index is 0", int(c0.mrope[:, 0].max()) == 0)
    check("h in [0,h_g)", int(c0.mrope[:, 1].max()) == h_g - 1)
    check("w in [0,w_g)", int(c0.mrope[:, 2].max()) == w_g - 1)

    tok, ds = w.split_row(c0.tokens)
    check("split round-trips tokens", tok.shape == (per_plane, w.hidden_size))
    check("split gives n_deepstack maps", len(ds) == w.n_deepstack)

    # ---- 2. many chunks, buffer-style concat ------------------------ #
    chunks = [c0]
    for j in range(1, a.planes):
        f = torch.randint(0, 255, (k, 3, H, W), dtype=torch.uint8)
        c = w.encode_chunk(f, fps=1.0)
        c.mrope[:, 0] += j
        chunks.append(c)
    all_tok = torch.cat([c.tokens for c in chunks], 0)
    all_mrope = torch.cat([c.mrope for c in chunks], 0)
    check("stacked planes", all_tok.shape[0] == a.planes * per_plane)
    check("grids identical across chunks",
          all(c.llm_grid_hw == c0.llm_grid_hw for c in chunks),
          [c.llm_grid_hw for c in chunks])

    # ---- 3. THE CHECK: dense positions == get_rope_index ------------ #
    print("\n[positions] dense prompt vs model.get_rope_index")
    counts = [(j, per_plane) for j in range(a.planes)]
    ids = w.build_prompt_ids("Describe the video.", counts, system="You are helpful.")
    pos_ours, delta_ours = w.build_position_ids(ids, all_mrope, c0.llm_grid_hw)

    is_vid = (ids[0] == w.video_pad_id)
    mm = torch.zeros_like(ids)
    mm[0, is_vid] = 2                      # 0 = text, 1 = image, 2 = video
    grid = torch.tensor([[a.planes, int(c0.grid_thw[1]), int(c0.grid_thw[2])]],
                        device=ids.device)
    pos_hf, delta_hf = w.model.model.get_rope_index(
        ids, mm_token_type_ids=mm, video_grid_thw=grid,
        attention_mask=torch.ones_like(ids))
    same = torch.equal(pos_ours.to(pos_hf.dtype), pos_hf)
    if not same:
        d = (pos_ours.to(pos_hf.dtype) != pos_hf)
        bad = d.any(0).nonzero().flatten()[:8].tolist()
        print(f"    first differing seq idx: {bad}")
        for i in bad[:4]:
            print(f"      i={i} ours={pos_ours[:,0,i].tolist()} hf={pos_hf[:,0,i].tolist()}")
    check("position_ids identical to get_rope_index", same)
    check("rope_delta identical", int(delta_hf[0, 0]) == delta_ours,
          f"ours={delta_ours} hf={int(delta_hf[0,0])}")

    # ---- 4. pruned prefill keeps survivor coordinates --------------- #
    g = torch.Generator().manual_seed(7)
    keep = torch.rand(all_tok.shape[0], generator=g) < 0.5
    keep[:2] = True                                    # never empty
    pruned_tok, pruned_mrope = all_tok[keep], all_mrope[keep]
    pc = []
    for p in pruned_mrope[:, 0].tolist():
        if pc and pc[-1][0] == p: pc[-1] = (p, pc[-1][1] + 1)
        else: pc.append((int(p), 1))
    ids_p = w.build_prompt_ids("Describe the video.", pc, system="You are helpful.")
    pos_p, _ = w.build_position_ids(ids_p, pruned_mrope, c0.llm_grid_hw)
    vid_p = (ids_p[0] == w.video_pad_id)
    check("pruned pads == survivors", int(vid_p.sum()) == int(keep.sum()))

    dense_vis = pos_hf[:, 0, is_vid]                   # [3, N] dense coords
    sel = dense_vis[:, keep.to(dense_vis.device)]
    check("survivor coordinates unchanged by pruning",
          torch.equal(pos_p[:, 0, vid_p].to(sel.dtype), sel))

    # ---- 4b. the processor must not resample our frames ------------- #
    # Qwen3VLVideoProcessor sets do_sample_frames=True by default and, absent
    # video_metadata, resamples at fps=24. Our chunks are already sampled at the
    # stream's 1 FPS, so any resampling silently changes WHICH frames the
    # backbone sees while leaving the token count looking perfectly reasonable.
    print("\n[frames] no subsampling")
    for n in (2, 4, 8, 16):
        f = torch.randint(0, 255, (n, 3, H, W), dtype=torch.uint8)
        c = w.encode_chunk(f, fps=1.0)
        check(f"{n} frames -> {n // k} planes", int(c.grid_thw[0]) == n // k,
              f"got {int(c.grid_thw[0])}")
    # content check: frames 2,3 differ between the two clips, frames 0,1 do not,
    # so plane 0 must match and plane 1 must not. A processor that resampled
    # would blur that distinction.
    base = torch.zeros(4, 3, H, W, dtype=torch.uint8)
    base[:2] = 40
    va, vb = base.clone(), base.clone()
    va[2:] = 90
    vb[2:] = 200
    ca, cb = w.encode_chunk(va, fps=1.0), w.encode_chunk(vb, fps=1.0)
    n_pp = ca.tokens.shape[0] // 2
    p0_same = torch.allclose(ca.tokens[:n_pp].float(), cb.tokens[:n_pp].float(),
                             atol=1e-2)
    p1_diff = not torch.allclose(ca.tokens[n_pp:].float(), cb.tokens[n_pp:].float(),
                                 atol=1e-2)
    check("plane 0 identical when frames 0-1 identical", p0_same)
    check("plane 1 differs when frames 2-3 differ", p1_diff)

    # ---- 5. generation actually runs -------------------------------- #
    print("\n[generate] pruned")
    r = w.generate_from_pruned(
        query="What do you see?", visual_tokens=pruned_tok, mrope=pruned_mrope,
        dense_llm_grid_hw=c0.llm_grid_hw, system="You are a helpful assistant.",
        max_new_tokens=24)
    print(f"    tokens {int(keep.sum())}/{all_tok.shape[0]}  prefill "
          f"{r.num_prefill_tokens}  TTFT {r.prefill_ms:.0f}ms  "
          f"decode {r.decode_ms:.0f}ms  new {r.num_new_tokens}")
    print(f"    text: {r.text[:160]!r}")
    check("generation returned text", len(r.text) > 0)
    check("generation stopped or hit cap", 0 < r.num_new_tokens <= 24)

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
