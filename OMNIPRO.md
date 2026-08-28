# OmniPro evaluation — deep reference

Everything about the OmniPro arm of this reproduction: what the benchmark is,
what we changed (nothing in the architecture), what we had to decide that
neither paper specifies, and how to run and re-score it. `HANDOFF.md` §10 is the
short version; this file is the *why*.

Benchmark paper: `omnipro1.pdf` — *OmniPro: A Comprehensive Benchmark for
Omni-Proactive Streaming Video Understanding* (arXiv 2605.18577), Ruixiang Zhao
et al., RUC + WeChat Vision.

---

## 1. Why OmniPro, and what it measures that StreamingBench cannot

StreamingBench asks a question stamped at a fixed timestamp and always wants
exactly one answer. RTAR — QueryStream's whole "when to speak" half — is
therefore **not scored** there; our StreamingBench harness computes `R_t / D_t /
T_t` on every frame and logs them, but answers at the query timestamp
(`response_mode="query_ts"`). Table 1 of the QueryStream paper is a *pruning*
result.

OmniPro's **Online mode** hands the model one standing instruction at t=0 and
then scores *when* it chooses to speak and *what* it says, with no further
prompting. That is exactly `T_t = R_t ∧ D_t`. **This is the first benchmark in
the reproduction where RTAR drives generation rather than being logged.**

The paper's own numbers for online-mode models (Table 2, metric F1, mean over 9
sub-tasks):

| Model | Params | Mean F1 |
|---|---|---|
| LiveStar | 8B | 3.6 |
| MMDuet2 | 3B | 11.3 |
| MiniCPM-o 4.5 | 9B | 20.9 |

## 2. The audio problem, stated plainly

84% of OmniPro samples depend on audio. Every sample carries
`audio_dependency ∈ {required, helpful, none}`:

| class | meaning | share of 2,700 |
|---|---|---|
| `required` | trigger cannot be perceived without speech or sound | 1768 (65.5%) |
| `helpful` | audio assists, trigger is visually recoverable | 500 (18.5%) |
| `none` | trigger is purely visual | 432 (16.0%) |

**QueryStream has no audio pathway** — OpenCLIP image tower → Qwen2.5-VL. The
user's instruction was explicitly to evaluate the existing architecture, not to
add an audio encoder. So `required` samples are unsolvable by construction.

They are **not dropped**. Dropping them would convert a known limitation into a
flattering number. They are reported as their own slice, and the report always
places them next to the one published yardstick that fits:

> **OmniPro Table 3, MiniCPM-o 4.5 under input ablation** — the only published
> online-mode row produced from video *without* audio.
> `A+V 20.9  |  V-only 13.8  |  A-only 10.9`

A 7.1-point drop is what losing audio costs a purpose-built omni model on this
benchmark. That is the size of the handicap, from the benchmark's own authors.

## 3. Metrics — Time F1, Content Accuracy, Joint F1

OmniPro Sec. 3.2.1 defines one metric and folds two failure modes into it:

> "Model responses are matched to ground-truth triggers via greedy temporal
> alignment with a tolerance of ±3 s. A match is considered valid only if the
> response is also content-correct. Precision is the fraction of model responses
> that are validly matched, recall is the fraction of ground-truth triggers that
> are validly matched, and F1 is their harmonic mean."

We report all three levels (`omnipro_eval/metrics.py`):

```
TIME F1      timing alone; a response is a TP if it lands within ±tol of an
             unused GT trigger, whatever it says.
                 time_P = tp_time / n_emits          n_emits = tp_time + fp
                 time_R = tp_time / n_gt             n_gt    = tp_time + fn

CONTENT ACC  of the responses that DID land on a trigger, the fraction whose
             text is also correct.
                 content_acc = tp_content / tp_time
             Conditioned on a time match by construction — a response that
             matched nothing has no reference text to be scored against.

JOINT F1     the benchmark's headline. DENOMINATORS ARE THE FULL COUNTS:
                 joint_P = tp_content / n_emits
                 joint_R = tp_content / n_gt
```

**The bug worth naming.** Writing `joint_P = tp_content / (tp_content + fp)`
silently drops every response that matched in time but failed on content,
instead of counting it as a false positive. That inflates Joint F1 by up to
~18%, and inflates it *most* exactly where content is worst. The correct
denominators are asserted in `tests/test_omnipro_metrics.py`.

Joint F1 is **not** `time_F1 × content_acc`, though the two track each other
when precision and recall are close; a large divergence signals that emissions
and triggers are badly imbalanced.

### Micro vs macro
`overall` pools tp/fp/fn across every sample — a task contributes in proportion
to how many responses and triggers it happens to carry. The paper's Table 2
"Mean" column is an **equal-weight average over the nine sub-tasks**. Both are
computed; `macro_*` is the one comparable to the paper. Neither is reported
alone.

### Unjudged is not zero
The two free-text tasks need an LLM judge. A missing verdict makes the match
**UNJUDGED**: counted as neither correct nor incorrect. `content_acc` and
`joint_f1` are then withheld (`None`) and only `content_acc_lb` / `joint_f1_lb`
are populated — named so they cannot pass for the real number. Timing metrics
never involve the judge and are always exact.

## 4. Content scoring per sub-task (paper Sec. 3.2.1)

| sub-task | level | content kind | how it is decided |
|---|---|---|---|
| `instant_event_alert` | Perception | `time_only` | content not scored at all |
| `semantic_condition_alert` | Comprehension | `time_only` | content not scored at all |
| `explicit_target_grounding` | Perception | `position` | exact match, 9 hyphenated grid cells |
| `snapshot_counting` | Perception | `count` | first integer after clock times stripped |
| `cumulative_counting` | Comprehension | `count` | ditto |
| `dedup_counting` | Reasoning | `count` | ditto |
| `realtime_state_monitor` | Perception | `state` | WHOLE reply lowercased == `state_to` |
| `event_narration` | Comprehension | `gpt_judge` | LLM, 1–5, correct if ≥3 |
| `sequential_step_instruction` | Reasoning | `gpt_judge` | LLM, 1–5, correct if ≥3 |

Extractor rules are transcribed from the OmniPro reference online parser
(`omnipro_eval/content.py`), including the parts that are *stricter* than they
first look — each such rule carries a comment saying what a lenient variant
would have inflated:

- **Clock strip before the integer.** `"At 01:23, 5 people"` yields `1` without
  it. Only affects counting.
- **Only hyphenated position labels.** `"bottom center"` matches the bare
  `center` — the wrong cell — which is what upstream does. Accepting the
  space-separated form would inflate Target-Ground. `"top left"` returns `None`
  (there is no bare `left` cell). `_POSITIONS` is sorted longest-first so
  `center-right` cannot be swallowed by `center`.
- **State is whole-reply exact equality**, not a substring search. A rambling
  reply that happens to contain the target word must not score correct. The one
  liberty taken is stripping a trailing `.`/`,`, since the prompt asks for a
  bare noun phrase.
- **An unparseable reply is WRONG, not unscorable.** Failing to produce the
  required output format is part of what Online mode tests.

State is an **open vocabulary** — 611 distinct `state_to` values across the
benchmark — and the prompt deliberately does **not** leak candidate states. The
model is told the shape of the answer, never its content.

## 5. Slices

The three headline slices, in `omnipro_eval/dataset.py::build_slices()`:

1. **`audio_required`** — the 65.5% whose triggers need audio. Unsolvable here;
   reported so the cost of the missing modality is visible.
2. **`audio_helpful_none`** — the vision-fair slice. **This is what QueryStream
   can legitimately be judged on.**
3. **`event_narration_required`** — Event-Narration restricted to its
   audio-required samples: a single-task probe of the free-text path under the
   hardest modality condition, and the one place the LLM judge decides content.

Diagnostic slices below those: each audio class separately, each of the nine
sub-tasks, and a trigger-modality split mirroring the paper's Figure 4.

**Modality slices are recall-side only.** Modality is a property of a *trigger*,
not of a sample, and a false positive matched no trigger — so it carries no
modality and precision has no honest denominator there. Reporting a fabricated
precision alongside would be worse than reporting less.

### The slice-weighted headline
`metrics.weighted_joint()` reports three weightings and carries the weights next
to the number so it cannot be quoted without them:

| weighting | what it estimates |
|---|---|
| **benchmark_weighted** | the full 2,700 distribution (required 65.5 / helpful 18.5 / none 16.0). The honest headline. **It is meant to look bad** — it is dominated by what audio would solve. |
| subset_weighted | the samples actually evaluated |
| equal_weighted | one third per audio condition — "how well does it handle each modality condition", ignoring how common each is |

Benchmark weighting matters because the evaluated set can be skewed. Before the
missing videos were fetched, on-disk coverage was 1,705/2,700 and biased *toward*
`none`/`helpful` (46.9% required vs the true 65.5%) — i.e. biased toward the
samples we can do. Benchmark weighting corrects exactly that.

## 6. What we had to decide (neither paper specifies these)

1. **Event debounce gap = the matching tolerance (3.0 s), not 1.0 s.**
   QueryStream Appendix A.4 groups consecutive triggers into event intervals and
   takes the first timestamp of each; the gap is unspecified, and the
   StreamingBench harness uses 1.0 s (one frame at 1 FPS). At 1 s, one real
   event can emit three responses inside OmniPro's ±3 s window, of which at most
   one can match — the other two are **pure false positives**. Every raw `T_t`
   timestamp is written to the JSONL (`raw_trigger_ts`), so any other gap can be
   re-scored offline for the two `time_only` tasks without touching a GPU.
2. **Streams are never truncated.** Cutting a video at its last ground-truth
   trigger would remove the tail's false positives and inflate precision.
   `--max-duration` therefore excludes *whole samples* rather than truncating
   them, and the exclusion is recorded in `ManifestStats`.
3. **Judge model = OpenAI, not the paper's Gemini-3-Flash.** Rubric, 1–5 scale
   and ≥3 threshold unchanged. A different judge is a different instrument;
   judged scores across judges are not strictly comparable. Stated in the report
   rather than assumed away.
4. **The judge sees the ground truth here** — the opposite of the StreamingBench
   judge policy, deliberately. There the judge maps free text onto one of four
   options and must not see the key. Here the benchmark's metric *is* "how well
   does this response match the reference", so the reference is the input.
5. **Only time-matched responses on the two judged tasks are sent.** A response
   that matched no trigger has no reference text and is already a false
   positive. Keeps the batch small.
6. **Per-task `max_new_tokens`** (`prompts.max_new_tokens_for`): 8 for counts,
   80 for the narration tasks. Letting a counting task ramble only creates text
   for the extractor to trip over, and decode is charged per token.

## 7. Architecture — unchanged

Nothing in `querystream/` was modified for OmniPro. The runner
(`omnipro_eval/runner.py`) is a sibling of
`querystream/eval_runner.StreamingBenchRunner` and reuses the same masks, the
same thresholds, the same score-space resampling, the same JIT `TokenBuffer`,
and `_sem_similarity` imported directly from it. The single difference is that
RTAR drives generation.

The loop, per 1-FPS frame:

```
M_sem  = 1[ sim(q, v_t^i) > mean_j sim(q, v_t^j) ]     query-dependent
M_temp = 1[ sim(v_t^i, v_dsh,t-1^i) < tau_temp ]       cached per video
D_t    = 1[ keep_rate(M_QDP) > tau_den ]
R_t    = 1[ calibrated sim(q, mean_pool v_t) > tau_rel ]
if T_t fires and survives the debouncer:
    flush the buffer through the accumulated masks -> generate
```

Plane masks are computed **once per (video, query)**: plane *j*'s mask depends
only on plane *j*'s own similarity maps (the frame-adaptive threshold is a mean
over that plane's tokens), so the mask for a prefix of the stream is the prefix
of that tensor. An emission at *t* takes the first ⌈t⌉ planes. The `TokenBuffer`
is appended to as time advances and flushed non-destructively at each event.

Per-video caching (`querystream/eval_cache.py`) is unchanged and still exact:
decode, OpenCLIP, the Qwen vision tower **and the whole DSH recursion** are
query-independent, so a video shared by several samples is decoded and encoded
once. Only `M_sem` and the relevance cosine re-run per sample.

## 8. Layout

```
omnipro_eval/
  dataset.py    manifest, taxonomy, slices, stratified subset, paper baselines
  prompts.py    per-task backbone prompts + output-format constraints
  content.py    count / position / state extractors (deterministic, offline)
  metrics.py    matching, Time/Content/Joint, micro+macro, slices, weighting
  judge.py      OpenAI Batch API judge for the two open-ended tasks
  runner.py     OmniProRunner — QDP+RTAR replay, RTAR drives generation
  report.py     matplotlib figures (palette slots 1-3)
eval_omnipro.py           CLI driver: shard, stream, log, profile, pre-score
score_omnipro.py          offline scorer: merge shards, fold verdicts, aggregate
judge_omnipro.py          judge batch submit / poll / collect
make_omnipro_report.py    -> QueryStream_OmniPro_report.pdf
scripts/submit_omnipro.sh SLURM: 4 nodes x 4 GPUs, strided video shards
tests/test_omnipro_metrics.py
```

## 9. Data

```
/iopsstor/scratch/cscs/dbartaula/omnipro_data/
    benchmark.json        2,700 samples, 9 tasks, 1,262 videos
    raw_videos/*.mp4      1,262 files, 1280x720 @ ~30 fps, ~32 GB
```

HF dataset: `RuixiangZhao/OmniPro` (`repo_type="dataset"`). The local copy
originally held 678 of 1,262 videos (1,705 of 2,700 samples); the remaining 584
were fetched with `hf_hub_download(..., local_dir=<dataset_dir>)`, 12 threads,
584/584 in 136 s. **Coverage is now complete**: 2,700/2,700 samples,
66.2 h of video, 9,051 ground-truth triggers.

Reference implementation consulted for the scorer:
`/iopsstor/scratch/cscs/dbartaula/streamagent/omnipro/` (a different model's
harness — used only for the metric definitions and the dataset layout, per the
user's instruction).

## 10. Running it

```bash
export HF_HOME=/iopsstor/scratch/cscs/dbartaula/hf_cache
export OPENCV_FFMPEG_CAPTURE_OPTIONS="timeout;15000000"

# smoke, login node, ~2 min once the models are warm
python eval_omnipro.py --limit 8 --n-samples 40 --prefer-short 1.0 \
    --out runs/op_smoke --device cuda:1

# full benchmark, 3 nodes x 4 GPUs on debug (~30 min)
sbatch scripts/submit_omnipro.sh
# submit it TWICE: the second job resumes whatever the first does not finish
sbatch scripts/submit_omnipro.sh

# variants
RUN_NAME=op_dense    KEEP_MODE=dense           sbatch scripts/submit_omnipro.sh
RUN_NAME=op_relonly  GATE_MODE=relevance_only  sbatch scripts/submit_omnipro.sh
RUN_NAME=op_denonly  GATE_MODE=density_only    sbatch scripts/submit_omnipro.sh

# judge (never writes the key to disk)
export OPENAI_API_KEY=...
python judge_omnipro.py runs/op_full --submit
python judge_omnipro.py runs/op_full --collect --wait 1800

# re-score and report — pure functions of the JSONL, no GPU
python score_omnipro.py runs/op_full
python score_omnipro.py runs/op_full --tolerance 5      # Appendix A.1 sweep
python make_omnipro_report.py runs/op_full
```

Every worker passes `--resume` and `--time-budget-sec`, so a job that runs out
of wall clock stops cleanly *between* videos and can be resubmitted unchanged.
A worker with nothing left to do exits in seconds without loading the models.

**Three nodes, not four.** The `debug` QOS caps a job at `node=90` TRES-minutes,
so 4 nodes x 30 min = 120 is rejected and the job sits `PENDING` forever with
`Reason=QOSMaxNodeMinutesPerJob` — which is indistinguishable from a busy queue
unless you look. `MaxNodes=4` and `MaxTime=1:30:00` are not the binding limits.
Workable shapes: 3x30 (shipped), 4x22, 2x45. `MaxJobsPU=1` / `MaxSubmitPU=2`
means one job runs at a time and two can be queued.

### Reporting a partial run
`score_omnipro.py` and `make_omnipro_report.py` are pure functions of the shard
JSONLs and work at any point, including while a job is still running. The PDF
opens with a **Run completeness** section giving samples in the JSONL, how many
were scored, how many errored, how many shard files were found, and how many
workers stopped on their time budget — so a partial report can never be mistaken
for a full one. A truncated final line from a killed worker is skipped rather
than aborting the read. Each pass writes its own `profile_summary*.pass*.csv`
(the writer truncates, so a shared file would lose the earlier pass); the report
frame-weights every `OVERALL` row it finds into one profile.

## 11. Emission rate is the binding constraint on precision

Full run: **24,915 responses against 9,051 ground-truth triggers** (2.75x).
Matching is one-to-one, so joint precision is capped at roughly
`n_gt / n_emits` ≈ 36% *no matter what the model says*. Recall and content
accuracy could both be perfect and the headline would still not exceed it.
Measured time precision is 12.5%.

This is a property of two reproduction choices, not of the method:

- **the relevance calibration** (`running_minmax`), forced by τ_rel = 0.60 being
  unreachable on raw CLIP cosine. Min-max against a trailing window makes 0.60
  mean "in the top 40% of the recent stream", which by construction fires on a
  substantial fraction of frames;
- **the debounce gap**, which decides how many of those frames collapse into one
  response.

Both are logged per frame (`relevance_raw_scores`,
`relevance_normalized_scores`, `raw_trigger_ts`) precisely so the trade-off can
be re-examined without re-running anything. **Do not tune either against the
metric** — the point of logging the raw signal is to be able to choose from the
observed distribution rather than from the score.

For reference, the same over-triggering pattern appeared on StreamingBench
(§14.4: 30,524 `T_t` firings debounced to 7,931 response events over 73k
frame-query pairs), where it was invisible because nothing scored it.

## 12. Results and the two offline ablations

Full numbers, the gate diagnosis and the gap sweep live in **CLAUDE.md
§15.8–15.11** so they sit next to the milestone log; the PDF at
`runs/op_full/QueryStream_OmniPro_report.pdf` is the readable version. The two
things not to re-derive:

**`D_t` is near-vacuous.** Over 512,540 frames it fires on 96.4% of them,
because the CLIP-grid QDP keep rate averages 0.467 against `tau_den = 0.15`.
So `T_t = R ∧ D` is arithmetically almost identical to `R_t` alone, and the gate
ablation shows full RTAR 18.3 vs relevance-only 18.2 vs density-only 4.7 time
F1. **The paper's central RTAR claim cannot be tested in this configuration.**

**Two ablations are free.** Both are computed from the JSONL with no GPU:

- `metrics.gate_ablation()` rebuilds the three trigger timelines from the logged
  per-frame `R_t` / `D_t`. *Timing only* — the single-gate policies fire on a
  superset of the full gate's frames, so no response text exists for their extra
  firings and content is not guessed at.
- `metrics.gap_sweep()` re-scores at any gap ≥ the recorded one, *content
  included*, because a larger gap selects a strict subset of the responses
  already generated. Verified on the run's own data.

Both are wired into `score_omnipro.py` (`--gap-sweep`) and the PDF.

## 13. Gotchas

- **Sharding is by video, strided.** Blocked shards finish at very different
  times because neighbouring videos have correlated durations. All shards write
  into the *same* run directory as `omnipro_raw_logs.shardNN.jsonl`;
  `score_omnipro.py` unions every match it finds, last write wins per
  `sample_id`, so a resumed shard supersedes rather than double-counts.
- **`summary.shardNN.json` carries `metrics_unjudged`.** Those `joint_f1` /
  `content_acc` fields are `None` by design and only the `_lb` bounds are
  filled. The real numbers come from `score_omnipro.py` after the judge lands.
- **No Stopwatch around the per-frame RTAR call.** It consumes Python floats;
  timing it per frame forces two CUDA syncs per frame to measure ~10 µs of CPU
  work and serialises the loop. It is covered by the vectorised `qdp_rtar` stage.
- **One video filename begins with `-`** (`-powI0eeLbw.mp4`), which breaks bare
  `ls *.mp4`. Use `find . -name "*.mp4"`.
- **τ_rel = 0.60 is unreachable**, now confirmed on a sixth dataset. Raw
  `cos(q, mean-pool v)` never approaches it. `relevance_mode="running_minmax"`
  is the default and `relevance_raw` is logged per frame.

---

## Second backbone arm — Qwen3-VL-8B-Instruct (`runs/op_qwen3`)

Same benchmark, same 2,700 samples, same everything except the backbone. See
CLAUDE.md §19.3 for why Qwen3-VL needed a new wrapper rather than a
`--backbone` flag: per-plane vision blocks separated by timestamp text,
interleaved M-ROPE, and deepstack features that must be pruned in lockstep with
the tokens they belong to.

### What is held fixed

| | `runs/op_full` | `runs/op_qwen3` |
|---|---|---|
| backbone | Qwen2.5-VL-7B-Instruct | **Qwen3-VL-8B-Instruct** |
| encoder | OpenCLIP ViT-L-14 / openai | same |
| α, τ_temp, τ_rel, τ_den | 0.1, 0.90, 0.60, 0.15 | same |
| relevance mode | running_minmax | same |
| debounce gap | 3.0 s | same |
| max pixels | 448² per frame | same |
| samples | all 2,700, no `--max-duration` | same |

### The one asymmetry, stated up front

At the same 448 px cap, Qwen3-VL produces **~180 visual tokens per plane**
against Qwen2.5-VL's **~252** — patch 16 tiles a frame more coarsely than patch
14. Equalising it would require giving one arm a different pixel budget, which
is a worse confound than the one it removes. Token **keep rate** is a ratio and
stays comparable; absolute visual capacity does not.

### Running it

```bash
STAGES=omnipro_qwen3 GPUS=0,1,2,3 nohup setsid \
  bash scripts/run_chain_supervised.sh > runs/chain_qwen3.log 2>&1 &

python3 run_chain.py --list                    # rows/expect
python3 run_chain.py --stages omnipro_qwen3 --progress   # one integer
```

Score and report are pure functions of the JSONL, so a partial run always
yields a PDF:

```bash
python3 score_omnipro.py runs/op_qwen3 && python3 make_omnipro_report.py runs/op_qwen3
```

### Judge

The two open-ended sub-tasks (Event-Narration, Step-Instruction) need the
OpenAI batch judge, exactly as `op_full` did. Until it lands, `content_acc` and
`joint_f1` are `None` for those tasks and only `*_lb` lower bounds are
published — never silently zeroed. Timing metrics never involve the judge.

```bash
export OPENAI_API_KEY=...        # environment only; never written to a file
python3 judge_omnipro.py runs/op_qwen3 --submit
python3 judge_omnipro.py runs/op_qwen3 --collect
python3 score_omnipro.py runs/op_qwen3 && python3 make_omnipro_report.py runs/op_qwen3
```
