# Preregistered run protocol and audit

## Experiment

Original QueryStream QDP + RTAR, adapted to `Qwen/Qwen3-VL-8B-Instruct`.
OmniPro **Online** on samples with `audio_dependency in {none, helpful}`;
video only. This is an adapted implementation, not the paper's reported result
or an untouched official executable. No training of model weights.

Sources are pinned in `data/split_manifest.json`. Model snapshot:
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.

## Split and fitting

932 samples, 699 videos. Within each task, sort IDs, independently shuffle
with Python `random.Random(42)`, take `floor(0.2*N)` for calibration and rest
for test. This gives 182 calibration and 750 held-out samples. It is a
**sample-disjoint, not video-disjoint split** (66 videos occur in both), as
requested. No trained weights; no threshold selection on test labels.

Fit raw-cosine `tau_rel` and density `tau_den` separately per task. Keep
`alpha=0.1` and `tau_temp=0.90` fixed (paper A.6). Search relevance at 17
calibration-only quantiles plus paper values 0.5/0.6/0.7; density grid:
0, .05, .10, .15, .20, .25, .30, .40, .50, .60.

For computational feasibility, shortlist four distinct trigger policies per
task by calibration timing F1, then run **full causal inference with separate
response history for every shortlisted policy**. Select by calibration joint
F1 (timing + content), using the same judge as evaluation; ties by policy ID
(timing-rank order). This is a two-stage search, not exhaustive joint-F1 search.
Freeze all selected thresholds before held-out inference/scoring. Do not
post-filter one policy's responses to pretend they came from another policy.

## Paper correctness

QDP: `semantic = cosine(patch,query) > frame mean`; `novelty =
cosine(patch,previous_DSH) < .90`; keep intersection. Update DSH **after**
comparison: `.1*current + .9*history`. Feature normalization and overlap grid
alignment follow the public QDP demo's `_make_qdp_masks_from_aligned_features`
and `_pool_clip_features_to_qwen_grid`. These exact pure functions are imported
via AST, avoiding incompatible Qwen2.5 imports. Every sample's masks and
relevance are checked against the official functions, not merely a rewritten
equation. The official per-frame function's outputs are the source of truth;
vectorized GEMM disagreed at a few strict floating-point threshold boundaries,
so those vectorized masks are diagnostic only. First-frame full keep follows
the official demo.

RTAR: raw cosine of query and normalized mean pooled aligned frame features,
strict `> tau_rel`, AND strict `keep_rate > tau_den`. No running-minmax,
softmax or zscore normalization. Paper A.4 emits on the start of each distinct
contiguous trigger interval. At 1fps we define contiguous as adjacent true
ticks; no metric-dependent three-second debouncer.

## Necessary backbone integration choices

Qwen3-VL temporal patches require two images. To consume **one new causal
observation every official 1fps tick**, duplicate that same image inside its
temporal patch. No future frame is admitted. Each observation has one block
labelled with its real tick. This temporal-packing choice differs from the
official Qwen2.5 offline demo's two distinct consecutive-frame packing and
must be disclosed. Per-frame pixel budget: `128*32*32=131072`, mapping the
official QDP demo's 128-token budget to Qwen3's 32-pixel merged grid. Neither
paper defines a universal per-frame pixel budget for this adapted backbone.

Use released CLIP PIL preprocessing. OpenAI ViT-L/14 weights use QuickGELU,
so select the matching OpenCLIP config rather than the official demo's
GELU/QuickGELU warning path. Project patch tokens into the CLIP text space.
CLIP forward fp32; Qwen3 bf16/SDPA; frozen weights. Cache per-video causal
vision outputs in batches of 8 observations; no temporal attention crosses
observation blocks. Measured batch-vs-single numerical tolerance is recorded.

The supplied reproduction's sparse M-ROPE builder failed to account for whole
missing frame blocks. The adapter instead creates the dense prompt and uses
**upstream Qwen3 `get_rope_index`, then applies the pruning mask to positions
and token IDs**. Deepstack maps are pruned in lockstep with visual features.
Unpruned tokens stay in the buffer/cache and are selected just in time.

## Official OmniPro behavior retained

Use the released `render_observe_prompt`, with complete own-response history,
at RTAR triggers only. No GT trigger polling. State-Monitor receives the
official state vocabulary from `_extract_states`; this is part of the
released benchmark input. All other reference answers and GT times are kept
out of inference. Greedy decoding, max 512 new tokens. Official `is_standby`
and `_merge_continuation_fragments(gap_sec=2)` determine final emissions.
Full video from tick 0 through `round(duration)` inclusive, matching official
frame index rounding/clamping. OpenCV replaces decord on ARM; same indices.
No sample exclusions for length and no emission cap.

Scoring calls the unmodified official `score_sample`: global-closest greedy
one-to-one time matching within +/-3 sec; official structured parser; alert
tasks time-only; narration and step instruction LLM judged. Report both time
and joint F1, per task and equal-weight task macro. Joint precision denominator
includes all emitted responses and recall includes all ground-truth triggers.
Missing predictions, model exceptions or outstanding judgments prevent final
completion; they are not silently treated as negatives or dropped.

Judge is the user-approved cheaper local OpenCode model
`github-copilot/gemini-3.8-flash`. Exact official `_JUDGE_PROMPT`, score 1–5,
pass >=3. SHA256 prompt-keyed verdict cache, model/session IDs and raw outputs
recorded. Only time-matched open-ended pairs are judged. No Gemini credentials
or private authentication data transferred to the cluster.

## Artifacts

`validation/`: dense logit and greedy decoding parity, sparse position and
deepstack checks, causal feature check, hand-calculated scorer tests.
`features/`: server-only video-token caches and per-sample gate traces.
`predictions/{calibration,test}/`: resumable sample/policy outputs.
`results/calibration_search.json`: full timing grid and four shortlists.
`results/frozen_thresholds.json`: selected policies with calibration scores.
`judge_requests/`, `judge_results/`: audit trail.
`results/FINAL_RESULTS.json` and `results/RESULTS.md`: only written when complete.
