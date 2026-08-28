# OVO-Bench evaluation — deep reference

The arm that closes the **backbone gap**. Milestones 3–5 ran QDP+RTAR on stock
Qwen2.5-VL-7B-Instruct; the QueryStream paper's headline instead swaps the
modules into **TimeChat-Online-7B**, replacing its query-agnostic DTD with
QDP+RTAR. Every earlier number therefore carried an unquantified difference from
the paper. This arm removes it and scores the paper's own Table 2 benchmark.

`HANDOFF.md` §11 is the short version; this is the *why*.

---

## 1. The backbone

`wyccccc/TimeChatOnline-7B` — 16 GB, already in `$HF_HOME`. Downloaded in 38 s.

```
architectures  ["Qwen2_5_VLForConditionalGeneration"]
model_type     qwen2_5_vl
vision_config  spatial_patch_size 14, tokens_per_second 2, bfloat16
```

**It loads through the existing `QwenVLWrapper` unchanged** — same class, same
vision geometry, same M-ROPE arithmetic. Only `--backbone` differs, which is
exactly the paper's setup: TimeChat-Online is a streaming fine-tune of
Qwen2.5-VL-7B, and QueryStream swaps its DTD module for QDP+RTAR.

DTD is not disabled so much as **bypassed**: QDP does the pruning upstream and
the backbone receives an already-sparse token set through
`generate_from_pruned`, never seeing the dense stream DTD would have filtered.

## 2. The benchmark, and how it is laid out on disk

OVO-Bench (Li et al., CVPR 2025): 1,640 samples, 12 sub-tasks, 3 categories.

| category | tasks | samples | calls |
|---|---|---|---|
| Real-Time Visual Perception | OCR ACR ATR STU FPD OJR | 837 | 837 |
| Backward Tracing | EPM ASI HLD | 631 | 631 |
| Forward Active Responding | REC SSR CRR | 172 | **1,567** |

```
data/ovo_bench/
  ovo_bench_new.json         1,640 samples
  chunked_videos/{id}.mp4    the 9 single-timestamp tasks, PRE-TRIMMED to `realtime`
  chunked_videos/{id}_{k}.mp4 forward tasks, one clip per probe, 0-INDEXED
```

**3,035 inference calls, 178 hours of video.** The forward tasks carry multiple
probes per sample (`test_info`), each with its own clip, which is why calls far
exceed samples. All 3,035 clips are present — verified, 0 missing.

The clips being pre-trimmed matters: "stream up to the query time" is simply
"decode the whole clip", so the streaming rule is enforced by the dataset and
there is no truncation decision for us to make or get wrong.

> A first pass concluded the forward videos were absent, because the 172 forward
> samples have no `{id}.mp4`. They are there under `{id}_{k}.mp4` — 1,567 files,
> exactly the number of probes.

## 3. Metrics — OVO-Bench's own rules, transcribed

From `utils/OVOBenchScore.py`. Reproduced in `ovo_eval/scoring.py` **including
the parts that look wrong**, because Table 2 was produced by these rules.

```
9 single-timestamp tasks   score = int(ground_truth_letter in response)
REC                        digits extracted, JOINED, compared as a string
SSR / CRR                  "Y"/"N" shorthands accepted, then int("Yes"/"No" in response)

per-task accuracy -> category average = MACRO mean over that category's tasks
                  -> Overall          = mean of the three category averages
```

Two quirks pinned by tests rather than silently fixed:

1. **The letter test is a substring test.** `"A photo of a dog"` scores when the
   answer is A. `strict_letter_acc` is reported alongside as a diagnostic — a
   reply that is *only* the letter — so it is visible how much the headline
   leans on the loose rule. The headline still uses the official rule.
2. **REC concatenates digits.** `re.findall(r'\d+')` then `"".join`, so
   `"1 or 2"` becomes `"12"`, not 1. Hedged answers are punished hard.

The macro-not-micro rule is load-bearing: OCR has 149 calls and FPD 101, and a
micro pool would quietly reweight every published comparison.
`test_category_average_is_macro_over_tasks_not_micro` pins it with a case where
the two differ by 49 points.

**Overall is withheld unless all three categories are present.** A two-category
mean is not the published quantity.

## 4. Prompts

`ovo_eval/prompts.py` carries the four official templates from `constant.py`
**verbatim**, including the upstream typos (`repetively`, `You're task is`,
`sequential of steps`). The prompts are part of the measuring instrument: the
scorer substring-matches a letter, a bare Yes/No, or digits, and all of that
depends on the model having been asked in exactly this way. A test asserts the
typos are still there, so a well-meaning cleanup fails loudly.

`constant.py` also holds two commented-out earlier REC/SSR variants; those are
not what the published numbers used and are not reproduced.

`query_text()` strips the boilerplate for OpenCLIP, which truncates at 77 BPE
tokens — every single-timestamp prompt shares the same ~30-token instruction
tail, which would otherwise crowd out the actual question. Same rule as the
other two arms; the backbone still gets the full official prompt.

## 5. What this arm can and cannot measure

**Table 2: yes.** Per-task accuracy, three category averages, Overall.

**Table 5 (the RTAR ablation): no.** The paper's Appendix A.4 protocol collects
trigger timestamps, groups them into events, and generates a response at a
timestamp *the model chooses*. The released clips are pre-cut to the benchmark's
own probe times, so there is no video beyond the probe to respond at. Table 5
would need the untrimmed sources. `R_t / D_t / T_t` are computed and logged per
frame regardless, so the trigger behaviour is on record.

Independently, **the timeliness exponent `p`** in `Score = Σ F·2^(−|t'−t|·p)` is
never specified in the paper, so Table 5's Score column is unreproducible even
with full video. This was already recorded as a blocker in CLAUDE.md §8.

## 6. Scale and subsetting

Full benchmark: 3,035 calls, 640,828 frames at 1 FPS — 2.5× StreamingBench.
Too large for the one usable partition (see §7), hence a **20% stratified
subset**: 584 calls, 328 samples, 37.6 h.

Sampling is **per task, at the sample level**, keeping every forward sample's
probes together. Both properties matter:

- per task, so small tasks (SSR has 42 samples) keep a defined accuracy and the
  macro averages stay meaningful;
- at the sample level, so a forward sample's probes are never split across the
  in/out boundary, which would score a partial trajectory.

`test_subset_keeps_a_samples_probes_together` pins this.

## 7. Cluster constraints

- **`normal` is unusable for account `a168`** — every association carries the
  `stop` QOS (`MaxJobsPU=0`). `debug` works only because the partition overrides
  with `debug-qos`.
- **`debug` caps a job at `node=90` TRES-minutes**, so 3 nodes × 30 min. Submit
  twice: `MaxJobsPU=1 / MaxSubmitPU=2`, and the second job resumes the first.
- **Decode residency is the failure mode.** OVO clips reach 1920×1440 and CRR
  averages 753 s, so one decoded clip can exceed 6 GB. `DECODE_WORKERS=1 /
  PREFETCH_DEPTH=1` caps residency at ~2–3 clips per worker. This is the same
  failure that OOM-killed StreamingBench workers (CLAUDE.md §16) and it is worse
  here.

## 8. Layout

```
ovo_eval/
  dataset.py    taxonomy, manifest -> flat Calls, subset, paper Table 2 rows
  prompts.py    the four official templates, verbatim
  scoring.py    official scorers + macro aggregation
  runner.py     QDP over one clip, one answer at its end
  report.py     figures
eval_ovo.py            driver (shard, resume, time budget)
score_ovo.py           offline scorer, works mid-run
make_ovo_report.py  -> QueryStream_OVOBench_report.pdf (landscape, Table-2 layout)
scripts/submit_ovo.sh  SLURM: 3 nodes x 4 GPUs on debug
tests/test_ovo_scoring.py   15 checks, no GPU/network
```

## 9. Running it

```bash
python tests/test_ovo_scoring.py -v          # 15 checks, run first

# smoke
python eval_ovo.py --limit 12 --fraction 0.2 --out runs/ovo_smoke --device cuda:1

# 20% subset, 3 nodes x 4 GPUs; submit twice, the second resumes
FRACTION=0.2 sbatch scripts/submit_ovo.sh
FRACTION=0.2 sbatch scripts/submit_ovo.sh

# score + report (pure functions of the JSONL, no GPU, work mid-run)
python score_ovo.py runs/ovo_20pct
python make_ovo_report.py runs/ovo_20pct

# controls
RUN_NAME=ovo_dense KEEP_MODE=dense sbatch scripts/submit_ovo.sh          # 100% keep
BACKBONE=Qwen/Qwen2.5-VL-7B-Instruct RUN_NAME=ovo_qwen sbatch scripts/submit_ovo.sh
```

The `ovo_dense` control is the one that makes the result interpretable: it holds
the backbone fixed and varies only the pruning, which is the comparison the
paper's "TimeChat-Online-7B (100%)" row provides.
