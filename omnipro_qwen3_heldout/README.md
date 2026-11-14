# QueryStream (adapted) + Qwen3-VL-8B-Instruct on OmniPro Online, held-out evaluation

This is OmniPro Online mode restricted to samples whose audio dependency is
`none` or `helpful`. The model gets video only. RTAR thresholds were fit per task on
a seed-42 20% calibration split and frozen, then evaluated once on the remaining 80%.
The run finished. `results/FINAL_AUDIT.json` independently recomputes every number
below from the saved predictions and judge verdicts (`passed: true`).

## Headline: 750 held-out samples, never used for fitting

| Metric | Value |
|---|---:|
| **Macro joint F1** (OmniPro headline, equal weight over 9 tasks) | **7.36%** |
| 95% bootstrap CI (2,000 reps, seed 42, samples resampled within task) | 5.92–8.82% |
| Macro time F1 (timing only, content ignored) | 17.69% |
| Micro joint F1 (pooled counts, for reference) | 8.06% |
| Mean per-sample QDP visual-token keep rate | 42.1% |
| Generation calls / judged open-ended pairs (pass rate) | 11,348 / 445 (24.7%) |

| Task | N | Emits / GT | Joint P | Joint R | Time F1 | **Joint F1** | Calib. joint F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Event-Alert | 31 | 92 / 42 | 6.5 | 14.3 | 8.96 | **8.96** | 25.00 |
| Target-Ground | 10 | 18 / 13 | 0.0 | 0.0 | 6.45 | **0.00** | 0.00 |
| State-Monitor | 114 | 986 / 506 | 5.3 | 10.3 | 18.63 | **6.97** | 14.50 |
| Snap.-Count | 59 | 243 / 59 | 1.6 | 6.8 | 9.27 | **2.65** | 6.25 |
| Cond.-Alert | 31 | 34 / 100 | 35.3 | 12.0 | 17.91 | **17.91** | 25.00 |
| Cum.-Count | 82 | 1,442 / 373 | 4.5 | 17.4 | 21.16 | **7.16** | 8.10 |
| Event-Narr. | 77 | 871 / 351 | 3.0 | 7.4 | 16.69 | **4.26** | 8.93 |
| Dedup.-Count | 227 | 4,863 / 929 | 5.4 | 28.5 | 22.76 | **9.15** | 9.24 |
| Step-Inst. | 119 | 1,187 / 649 | 7.1 | 12.9 | 37.36 | **9.15** | 12.71 |
| **Macro** | **750** | | | | **17.69** | **7.36** | 12.19 |

### Frozen thresholds (fit on the calibration split only)

| Task | τ_rel (raw cosine) | τ_den |
|---|---:|---:|
| Event-Alert | 0.313182 | 0.5 |
| Target-Ground | 0.234903 | 0.6 |
| State-Monitor | 0.257871 | 0.5 |
| Snap.-Count | 0.260982 | 0.5 |
| Cond.-Alert | 0.303875 | 0.5 |
| Cum.-Count | 0.217945 | 0.5 |
| Event-Narr. | 0.220876 | 0.4 |
| Dedup.-Count | 0.192371 | 0.5 |
| Step-Inst. | 0.188515 | 0.5 |

α = 0.1 and τ_temp = 0.90 are fixed to the paper's values.

## Findings

Evidence for each finding is in `validation/findings_checks.json` unless noted.

1. **The released QueryStream code covers QDP only.** `Zhangkr2003/QueryStreamPlus`
   @ `b793153` ships a single-query, offline QDP demo (`querystream/demo/infer.py`).
   It has no RTAR response loop, no Qwen3-VL support and no OmniPro adapter. This run
   is therefore an *adapted* implementation. QDP masks and frame relevance come from
   the release's own pure functions (`_pool_clip_features_to_qwen_grid` and
   `_make_qdp_masks_from_aligned_features`, imported through the Python AST). RTAR
   follows the paper's equations.
2. **The paper's τ_rel = 0.60 can never fire on raw OpenCLIP cosine.** Across 178,022
   frames, relevance ranges from 0.088 to 0.350 (median 0.241, p99 0.312). No frame
   exceeds 0.60. The fitted τ_rel values are 0.189–0.313. The fitted τ_den values are
   0.4–0.6, against the paper's 0.15. The paper's 0.15 density gate would pass 88% of
   frames; τ_den = 0.5 passes 36%.
3. **`querystream/qwen3_wrapper.py` shifts M-RoPE positions when a whole plane is
   pruned.** Its positions are exact for dense input and for partial pruning. When a
   plane has zero surviving tokens, its block is removed from the prompt. Every later
   visual token and the question text then shift by 23 positions on a 15×8 grid,
   which contradicts the wrapper's docstring. In this run, 1.8% of frames were fully
   pruned, and **18.1% (2,052 of 11,348) of held-out generation calls** had such a
   frame in context.

   The fix used here is `AuditedQwen` in `experiment.py`. It builds the dense prompt,
   takes positions from upstream `get_rope_index`, and then drops pruned visual tokens
   together with their positions and deepstack rows. This is the same compute-dense,
   slice-by-keep-mask scheme the official Qwen2.5 QueryStream model uses.
   `validation/qwen3_equivalence.json` checks it against upstream `transformers`:
   identical dense logits (max error 0.0) and identical greedy decoding. This PR does
   **not** change the library file, because that would alter `runs/op_qwen3`.
4. **The existing `omnipro_eval` harness departs from OmniPro's released online
   protocol.** Its prompt tells the model the event "has just occurred". It gives no
   own-response history and no option to stay silent, whereas the official
   `render_observe_prompt` includes both. It caps generation per task at 8–80
   tokens; the paper uses 512. It omits the State-Monitor state list, which the
   official prompt supplies through `_extract_states`. It also adds a system prompt.
   This run calls the official prompt, `is_standby`, `_merge_continuation_fragments`
   and `score_sample` functions directly.
5. **Qwen3-VL never stays silent when RTAR calls it.** 0 of 11,348 calls returned an
   empty output (27 replied a literal "(none)" or "none"). Every trigger becomes an
   emission, so precision is the bottleneck: joint precision is 0–7.1% on 8 of 9 tasks.
6. **Fitting on small calibration sets is optimistic.** Macro joint F1 falls from
   12.19 on calibration to 7.36 on held-out. Calibration sets hold 2–56 samples per
   task; Target-Ground has 2 calibration and 10 test samples.
7. **Context only, not a controlled comparison.** The repo's existing run
   `runs/op_qwen3`, scored with the repo's own scorer and OpenAI judge on exactly
   these 750 IDs, gets **8.50%** macro joint F1 and 18.87% time F1. That is inside
   this run's CI (`results/context_existing_op_qwen3_on_heldout.json`). Prompts,
   judge, gating, thresholds, frame packing and M-RoPE handling all differ, so the
   gap cannot be attributed to any one factor. The judge differs most: the existing
   OpenAI judge passed 48% of its judged matches across all 2,700 samples, while
   Gemini-3.8-Flash passed 25% here. Per task, this run is better on State-Monitor
   (6.97 vs 2.55) and Dedup.-Count (9.15 vs 5.97). It is worse on Event-Narr. (4.26
   vs 10.36), Cum.-Count, Target-Ground, Step-Inst. and Cond.-Alert.

## Protocol in brief

The full specification is in `PROTOCOL.md`.

- **Data:** OmniPro `benchmark.json` (SHA-256 in `splits/split_manifest.json`),
  filtered to `audio_dependency in {none, helpful}`. That gives 932 samples from 699
  videos. Within each task, IDs are sorted, shuffled with `random.Random(42)`, and
  `floor(20%)` go to calibration: 182 calibration and 750 test samples. The split is
  by sample, so 66 videos appear on both sides.
- **Online loop:** official ticks run from 0 s to `round(duration)` s at 1 fps, with
  the official frame-index rounding. No audio, no emission cap and no sample
  exclusions. Every sample's question time is 0 s.
- **Prompting and decoding:** the official `render_observe_prompt` with full
  own-response history, called only at RTAR triggers. Greedy decoding, at most 512
  new tokens.
- **Response timing:** RTAR fires on `relevance > τ_rel AND keep_rate > τ_den`
  (strict). A response is emitted at the first tick of each contiguous trigger run,
  per the paper's Appendix A.4. Pruned tokens and their positions are removed just in
  time from a buffer of unpruned tokens.
- **Backbone:** `Qwen/Qwen3-VL-8B-Instruct` (snapshot `0c351dd0…`), bf16, SDPA
  attention, 131,072 pixels per frame. Each causal 1-fps frame is duplicated inside
  Qwen3's two-frame temporal patch, so no future frame is ever used.
- **Scoring:** the unmodified official `score_sample`, with greedy one-to-one
  matching, ±3 s tolerance and the official parsers. Open-ended tasks use the exact
  `_JUDGE_PROMPT`, scored 1–5 with pass at ≥ 3.
- **Judge:** `github-copilot/gemini-3.8-flash` via local OpenCode (user-approved
  cheaper judge). Verdicts are cached by the SHA-256 of the prompt.
- **Threshold fitting:** for each task, a timing-F1 grid on calibration (17
  calibration quantiles plus 0.5/0.6/0.7, against 10 density values) produces 4
  distinct policies. Each policy gets full causal inference and judging on
  calibration. The best calibration joint F1 wins, and the thresholds are frozen
  before any test inference.

## Caveats when quoting the number

- This is an adapted implementation (finding 1), with an adapted backbone. It is not
  the paper's Qwen2.5-VL or TimeChat-Online result.
- It is not comparable to OmniPro Table 2 online numbers (for example, MiniCPM-o 4.5
  scores 20.9 with audio and video, and 13.8 with video only). Those use all 2,700
  samples, including audio-required ones, and have no held-out split.
- The judge differs from the paper's Gemini-3-Flash API. This affects only
  Event-Narr. and Step-Inst. content.
- Per-task numbers for Target-Ground (10 test samples) and the two alert tasks (31
  each) are very noisy.
- 5 of 14.6M QDP mask entries differed between a vectorized recomputation and the
  official per-frame function. The official outputs were used. For videos processed
  before that switch, stored relevance came from the vectorized computation, which
  was asserted to be within 2e-6 of the official function, and the masks were
  asserted bit-identical.

## Reproducing

The scripts expect the run's scratch layout. `experiment.py` honours `QS_ROOT`; the
other scripts hard-code `ROOT=/iopsstor/scratch/cscs/dbartaula/querystream_omnipro_20261114`,
so edit `ROOT` to run elsewhere. That root holds copies of this directory's scripts
plus three clones:

```
$ROOT/reproduction     this repo @ 87391e5
$ROOT/official_query   Zhangkr2003/QueryStreamPlus @ b793153
$ROOT/official_omni    RuixiangZhao/OmniPro @ 7c72e81
```

```bash
python3 setup_assets.py                          # benchmark, seed-42 split, videos, Qwen3 snapshot
bash worker.sh 0 validation                      # Qwen3 dense-logit / greedy / M-RoPE / deepstack checks
python3 metric_checks.py                         # hand-computed scorer + split checks
bash worker.sh <gpu> features                    # one per GPU; per-video features + QDP/RTAR signals
python3 experiment.py shortlist                  # calibration timing grid -> 4 policies per task
bash worker.sh <gpu> calibration                 # calibration inference
python3 experiment.py score --stage calibration  # needs verdicts; freezes thresholds, writes jobs_test.json
bash worker.sh <gpu> test
python3 experiment.py score --stage test         # writes results/FINAL_RESULTS.json
python3 final_audit.py && python3 verify_findings.py
```

`laptop/orchestrate.py` drives the run from the machine that has OpenCode. Over
SSH it polls `control.py`, launches GPU workers on `clariden-ln00{1..4}`, judges
pending requests with `opencode run --pure --model github-copilot/gemini-3.8-flash`,
and writes verdicts back. Feature caches (~479 GB) and model weights stay on scratch
and are not committed.

## Files

| Path | Contents |
|---|---|
| `results/FINAL_RESULTS.json` | Official-scorer per-task metrics, macro, split, thresholds |
| `results/FINAL_AUDIT.json` | Independent recomputation, bootstrap CI, provenance hashes |
| `results/predictions_test.jsonl` | 750 held-out rows: generation calls, emissions, thresholds, keep rates |
| `results/predictions_calibration.jsonl` | 728 calibration rows (182 samples × 4 candidate policies) |
| `results/judge_verdicts.jsonl` | 797 judge prompts and verdicts (calibration, test, 1 plumbing smoke) |
| `results/calibration_search.json` | Full timing grid and shortlists |
| `results/frozen_thresholds.json` | Selected policies and calibration metrics |
| `results/context_existing_op_qwen3_on_heldout.json` | Finding 7 |
| `splits/` | Split manifest (pinned commits, benchmark SHA-256), calibration/test ID lists |
| `validation/` | Qwen3 equivalence, scorer checks, inference smoke, findings checks |
| `logs/` | Laptop orchestrator log, filtered worker events, the 2 recovered feature errors |
