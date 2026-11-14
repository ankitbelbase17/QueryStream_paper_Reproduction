"""Verify the findings claimed in the PR. Writes validation/findings_checks.json.

1. trigger_times (vectorized) == contiguous-run-start definition (paper A.4).
2. Supplied reproduction's Qwen3VLWrapper: M-RoPE positions of surviving tokens
   vs. upstream dense positions (dense / partial pruning / whole-plane drop),
   and the audited replacement used in this experiment.
3. How often whole frames are pruned to zero tokens in this run's QDP masks,
   and how many held-out generation calls had such a frame in context.
4. Raw relevance range (is the paper's tau_rel = 0.60 reachable?).
5. Question times (online loop must start at t = 0 for every sample).
6. Silence: how many generation calls produced an empty (standby) output.
"""
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

import torch

ROOT = Path('/iopsstor/scratch/cscs/dbartaula/querystream_omnipro_20261114')
sys.path.insert(0, str(ROOT))
from experiment import read, write, key, feature_path, trigger_times, get_qwen, setup_torch  # noqa: E402
from querystream.qwen3_wrapper import Qwen3VLWrapper  # noqa: E402
from utils.online_parser import is_standby  # noqa: E402

out = {}

# ---- 1. trigger_times: only the run-start extraction changed; flags identical
rng = random.Random(0)
levels = [0.1, 0.2, 0.25, 0.3]
for _ in range(20000):
    n = rng.randint(1, 60)
    sig = {'relevance': torch.tensor([rng.choice(levels) for _ in range(n)]),
           'density': torch.tensor([rng.choice(levels) for _ in range(n)])}
    r, d = rng.choice(levels), rng.choice(levels)
    flags = ((sig['relevance'] > r) & (sig['density'] > d)).tolist()
    ref = [i for i in range(n) if flags[i] and (i == 0 or not flags[i - 1])]
    assert trigger_times(sig, r, d) == ref
out['trigger_times_vectorized_equals_reference'] = {'random_cases': 20000, 'passed': True}

# ---- 4/5. relevance range + question times (all 932 subset samples)
subset = read(ROOT / 'data/subset.json')
out['question_time_all_zero'] = all(
    float(s.get('question_time_sec', 0)) == 0.0 and s.get('question_time', '00:00') == '00:00'
    for s in subset)
rel_all = []
den_all = []
zero_frames = total_frames = 0
disagree = mask_entries_checked = samples_with_disagreement = 0
masks = {}
by_vid = defaultdict(list)
for s in subset:
    by_vid[s['video_id']].append(s['id'])
for vid, sids in by_vid.items():
    data = torch.load(feature_path(vid), map_location='cpu', weights_only=False, mmap=True)
    for sid in sids:
        sig = data['signals'][sid]
        rel_all.append(sig['relevance'].float())
        den_all.append(sig['density'].float())
        mk = sig['mask']
        masks[sid] = (mk.sum(1) == 0)
        zero_frames += int(masks[sid].sum())
        total_frames += mk.shape[0]
        if 'vectorized_reference_disagreements' in sig:
            dd = int(sig['vectorized_reference_disagreements'])
            disagree += dd
            mask_entries_checked += mk.numel()
            samples_with_disagreement += dd > 0
rel = torch.cat(rel_all)
out['relevance_raw_cosine'] = {
    'frames': int(rel.numel()), 'min': float(rel.min()), 'max': float(rel.max()),
    'p50': float(rel.quantile(.5)), 'p99': float(rel.quantile(.99)),
    'frames_above_paper_tau_rel_0.60': int((rel > 0.6).sum())}
den = torch.cat(den_all)
out['density_keep_rate'] = {
    'frames': int(den.numel()), 'mean': float(den.mean()),
    'frames_above_paper_tau_den_0.15': float((den > 0.15).float().mean()),
    'frames_above_0.4': float((den > 0.4).float().mean()),
    'frames_above_0.5': float((den > 0.5).float().mean()),
    'frames_above_0.6': float((den > 0.6).float().mean())}
out['fully_pruned_frames'] = {
    'frames_with_zero_kept_tokens': zero_frames, 'frames_total': total_frames,
    'fraction': zero_frames / total_frames}
out['vectorized_vs_official_qdp'] = {
    'note': 'recorded only for videos processed after the official function became the '
            'source of truth; earlier videos asserted bit-identical masks',
    'mask_entries_checked': mask_entries_checked, 'entries_differing': disagree,
    'samples_with_any_difference': samples_with_disagreement}

# ---- 3/6. held-out generation calls: fully-pruned frame in context; silence
calls_total = calls_zero_ctx = standby = none_literal = 0
for j in read(ROOT / 'jobs_test.json'):
    s = j['sample']
    p = read(ROOT / 'predictions/test' / f"{key('test|' + j['policy'] + '|' + s['id'])}.json")
    cz = torch.cumsum(masks[s['id']].int(), 0)
    for c in p['generation_calls']:
        calls_total += 1
        calls_zero_ctx += int(cz[c['t_sec']] > 0)
        standby += is_standby(c['raw'])
        none_literal += c['raw'].strip().strip('.').lower() in ('(none)', 'none')
out['heldout_generation_calls'] = {
    'total': calls_total, 'with_fully_pruned_frame_in_context': calls_zero_ctx,
    'fraction_with_fully_pruned_frame': calls_zero_ctx / calls_total,
    'standby_empty_outputs': standby, 'literal_none_outputs': none_literal}

# ---- 2. M-RoPE check on the GPU
setup_torch()
q = get_qwen()
hw = tuple(torch.load(next((ROOT / 'features').glob('*.pt')), map_location='cpu',
                      weights_only=False, mmap=True)['hw'])
H, W = hw
m = H * W
n = 4
query = 'Count how many people are visible.'


def coords(k):
    return torch.stack([torch.arange(k).repeat_interleave(m),
                        torch.arange(H).repeat_interleave(W).repeat(k),
                        torch.arange(W).repeat(H * k)], -1).to(q.device)


def upstream(ids, k):
    types = torch.zeros_like(ids)
    types[ids == q.video_pad_id] = 2
    grids = torch.tensor([[1, 2 * H, 2 * W]] * k, device=ids.device)
    pos, _ = q.model.model.get_rope_index(ids, mm_token_type_ids=types, video_grid_thw=grids)
    return pos


ids_d = Qwen3VLWrapper.build_prompt_ids(q, query, [(j, m) for j in range(n)], fps=1.)
pos_orig_d, _ = Qwen3VLWrapper.build_position_ids(q, ids_d, coords(n), hw)
pos_up_d = upstream(ids_d, n)
vis_d = ids_d[0] == q.video_pad_id
last_end_d = int((ids_d[0] == q.vision_end_id).nonzero()[-1])


def original_sparse(keep):
    counts = [(j, int(keep[j].sum())) for j in range(n) if int(keep[j].sum()) > 0]
    ids_s = Qwen3VLWrapper.build_prompt_ids(q, query, counts, fps=1.)
    pos_s, _ = Qwen3VLWrapper.build_position_ids(q, ids_s, coords(n)[keep.flatten().to(q.device)], hw)
    ref_vis = pos_up_d[:, 0, vis_d][:, keep.flatten().to(q.device)]
    got_vis = pos_s[:, 0, ids_s[0] == q.video_pad_id]
    last_end_s = int((ids_s[0] == q.vision_end_id).nonzero()[-1])
    tail_ref, tail_got = pos_up_d[:, 0, last_end_d + 1:], pos_s[:, 0, last_end_s + 1:]
    assert tail_ref.shape == tail_got.shape
    return {'surviving_visual_positions_equal_dense': bool(torch.equal(ref_vis, got_vis)),
            'max_visual_position_shift': int((ref_vis - got_vis).abs().max()),
            'question_text_positions_equal_dense': bool(torch.equal(tail_ref, tail_got)),
            'question_text_position_shift': int((tail_ref - tail_got).abs().max())}


g = torch.Generator().manual_seed(0)
partial = torch.rand(n, m, generator=g) < .5
partial[:, 0] = True                      # every plane keeps >= 1 token
whole = partial.clone()
whole[1] = False                          # plane 1 pruned to zero tokens
q.prepare_context(n, hw, whole)
q.build_prompt_ids(query, [(j, int(whole[j].sum())) for j in range(n) if whole[j].any()])
out['qwen3_mrope'] = {
    'grid_hw': list(hw), 'planes': n,
    'original_wrapper_dense_equals_upstream': bool(torch.equal(pos_orig_d, pos_up_d)),
    'original_wrapper_partial_pruning': original_sparse(partial),
    'original_wrapper_whole_plane_dropped': original_sparse(whole),
    'audited_wrapper_whole_plane_dropped_equals_upstream_dense_sliced':
        bool(torch.equal(q.sparse_pos, q.last_dense_pos[:, :, q.last_keep])),
}
write(ROOT / 'validation/findings_checks.json', out)
print(json.dumps(out, indent=2))
