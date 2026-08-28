# QueryStream reproduction — START HERE

Orientation for a session with no prior conversation context. Read this first;
`CLAUDE.md` is the deep reference (paper notes §1–10, milestone log §11–14) and
is where you go for *why* a decision was made.

**Goal.** Reproduce ICLR 2026 *QueryStream: Advancing Streaming Video
Understanding with Query-Aware Pruning and Proactive Response* — two
training-free modules (QDP pruning, RTAR triggering) bolted in front of a frozen
Qwen2.5-VL-7B-Instruct, targeting ~75.32 on StreamingBench at ~57% token keep
rate.

**Working dir.** `/iopsstor/scratch/cscs/dbartaula/querystream` (not a git repo).

---

## 1. State in one table

| Milestone | What | Status |
|---|---|---|
| M1 | QDP / RTAR / DSH / TokenBuffer + 29 unit tests | ✅ 29/29 |
| M2 | OpenCLIP + Qwen2.5-VL wrappers, JIT pipeline, 29 integration checks | ✅ 29/29 |
| M3 | StreamingBench harness, JSONL logging, profiler, SLURM, 10 equivalence checks | ✅ 10/10 |
| M3b | First real numbers on a 270-question subset + PDF report | ✅ done |
| — | **Full 2500-question StreamingBench run** | ✅ **done 2026-08-25 — 69.40% @ 49.4% keep** |
| **M5** | **OmniPro Online mode — RTAR actually scored. See [`OMNIPRO.md`](OMNIPRO.md)** | ✅ **full 2700/2700 run done**, 26/26 tests |
| **M6** | **OVO-Bench on TimeChat-Online-7B — the paper's own backbone. See [`OVOBENCH.md`](OVOBENCH.md)** | ✅ **20% subset done, 584/584** — Overall 44.3 vs paper 49.4 |
| M6b | OVO controls: dense (`runs/ovo_dense`), 768-px detail (`runs/ovo_detail768`) | ✅ done — dense 46.0 **beats** pruned; 768 px *hurt* (−5.6, p=0.42) |
| **M7a** | **LAION OpenCLIP encoder control (`runs/ovo_laion`)** | ✅ **584/584** — 45.4 vs 44.3, **p = 0.551, not significant** |
| — | OmniPro judge batch (965 requests) | ✅ **landed 2026-08-25** — joint F1 **6.70**, no longer a bound |
| **M7b** | **OmniPro full on Qwen3-VL-8B-Instruct (`runs/op_qwen3`)** | 🔄 wrapper done, 23/23 geometry checks; smoke then full |

**Standing user rule: build incrementally and WAIT for a signal after each
milestone. Do not start the full StreamingBench run or M4 unsolicited.**
(The OmniPro full run in M5 *was* explicitly requested — see §10.)

### OmniPro headline (`runs/op_full/`, all 2,700 samples)

```
responses 24,915   GT triggers 9,051   time-matched 3,110   0 errors
TIME    P 12.5   R 34.4   F1 18.3     macro 17.7
CONTENT acc 36.6                     macro 41.3   (judge landed; 0 unjudged)
JOINT   F1 6.70                      macro 6.70
token keep rate 48.0%                benchmark-weighted macro joint F1 6.49
reference: MiniCPM-o 4.5 VIDEO-ONLY 13.8 mean joint F1 (with audio: 20.9)
```

> **The central finding is not the score.** Over 512,540 frames `D_t` fires on
> **96.4%** of them — the CLIP-grid keep rate averages 0.467 against
> `tau_den = 0.15` — so `T_t = R ∧ D` is arithmetically almost identical to
> `R_t` alone (42.5% vs 43.8% of frames). The gate ablation confirms it:
> full RTAR time F1 **18.3** vs relevance-only **18.2** vs density-only 4.7.
> **The paper's central RTAR claim cannot be tested in this configuration.**
> That is a statement about the reproduction gap, not about the method — see
> CLAUDE.md §15.9 and §15.11 for the three experiments that would resolve it.

Also worth not re-deriving: the debounce gap was swept exactly and offline
(larger gaps select a strict subset of already-generated responses).
**Time precision is flat at 12.5–13.7% across every gap** while recall falls, so
the timing error is in *where* the gate fires, not how often, and tuning the gap
cannot help. CLAUDE.md §15.10.

### OVO-Bench, 20% subset on the paper's backbone (`runs/ovo_20pct/`)

```
backbone wyccccc/TimeChatOnline-7B   584/584 calls   1 error   keep 50.78%
                      RT     BW     FW    Overall
ours                 53.8   44.0   35.0    44.3
paper QueryStream    63.0   44.9   40.2    49.4   (52.9% keep)
paper TimeChat 100%  61.9   41.7   36.7    46.8
```

> **Backward Tracing essentially reproduces (44.0 vs 44.9; EPM 55.9 vs 56.9).**
> The overall gap is bimodal and systematic, not noise: we are **ahead** on
> ACR +14.1, ASI +7.8, FPD +7.7 (semantic / temporal) and **behind** on
> ATR −30.7, STU −19.4, OCR −18.5 (fine detail). Prime suspect is our own
> `max_pixels = 448²` cap, which the paper never specifies — a 768² control on
> those three tasks is in `runs/ovo_detail768/`. HLD (2.7 vs 12.4) is a separate,
> genuine refusal-calibration deficit that resolution cannot explain.
> **Table 5 is unreproducible from this data** — clips are pre-trimmed to their
> probe timestamps, so RTAR has no video to choose a response time in.

### StreamingBench FULL SPLIT (`runs/sb_score_mean/`, all 2,500 questions)

```
accuracy 69.40%   1735/2500   95% CI 67.56-71.18   0 unparsed   0 errors
token keep rate 49.40%   41,058,754 / 83,110,428
```

| | OP | CR | CS | ATP | EU | TR | PR | SU | ACP | CT | All |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **ours @49.4% keep** | 72.6 | 78.9 | 77.0 | 73.1 | 73.0 | 69.5 | 75.9 | 59.4 | 63.4 | **55.3** | **69.40** |
| paper QueryStream-7B @57.2% | 82.4 | 84.4 | 79.2 | 82.4 | 78.0 | 81.3 | 78.7 | 65.0 | 69.3 | 47.3 | 75.32 |
| paper Qwen2.5-VL-7B, full tokens | 78.3 | 80.5 | 78.9 | 80.5 | 76.7 | 78.5 | 79.6 | 63.4 | 66.2 | 53.2 | 73.68 |

**Read it against the Qwen2.5-VL row (73.68), not the QueryStream row.** The
paper's headline swaps QDP+RTAR into TimeChat-Online-7B, a streaming
fine-tune; this reproduction uses stock Qwen2.5-VL-7B-Instruct. We are 4.3
below that baseline while keeping 49.4% of tokens, and 5.9 below the paper's
QueryStream row — which also prunes less (57.2%).

**CT (Counting) is the standout: 55.3 vs the paper's 47.3.** Counting is the
one subtask the paper reports degrading badly under pruning, and it is the one
place this reproduction is clearly ahead — consistent across the 270-question
subset (73.3, n=15) and now the full split (n=188).

The subset run's 71.11% (CI 65.4-76.2) contains 69.40%, so the two are
consistent; the full split is the number to quote.

Report: `runs/sb_score_mean/QueryStream_StreamingBench_full_report.pdf`

### StreamingBench subset (`runs/sb_subset/`, superseded by the above)

```
270 / 2500 questions (10.8%), 31 videos, uniform random by video (--shuffle-seed 0)
accuracy (exact match)   71.11%   192/270   95% CI 65.4–76.2   (paper 75.32)
token keep rate          50.02%   4,590,808 / 9,178,263        (paper 57.2%)
unparsed 0   errors 0   zero-survivor guard fired 0
```

The keep rate reproduces (9.2M tokens, tight). **The accuracy does not yet mean
anything precise** — the CI spans the paper's 75.32%, so this rules out a gross
failure but cannot resolve a one-point gap. Only the full 2500 can. Do not quote
71.11% as "below the paper".

Per subtask (correct/n): OP 30/45, CR 9/12, CS 26/35, ATP 33/40, EU 16/22,
TR 17/27, PR 5/6, SU 10/16, ACP 35/52, **CT 11/15 = 73.3%**.

> **CT is the interesting one.** The paper's worst subtask (47.34%) is this run's
> best. n=15, so it is not a finding yet — but check it first at full scale.

---

## 2. What is running right now

| | |
|---|---|
| GPU work | Qwen3-VL OmniPro smoke on login GPU 1, then the full 2,700 via `run_chain.py` |
| SLURM | **nothing of ours.** `debug` is the only partition open to `a168` and both of its per-user slots are held by the user's own `sa_full` jobs — not ours to cancel (§5). Everything runs on the four login GH200s. |
| OmniPro judge batch | ✅ **collected 2026-08-25.** 965 verdicts in `runs/op_full/judge_output.jsonl`; metrics + PDF regenerated. |
| StreamingBench judge batch | `batch_6a8c3abb08748190a52e2eea67cc4e1b`, 270 requests — never collected. It **cannot change** the headline (270/270 parsed from a bare letter); collect only if someone wants the parser-dependence bound. |

### Resuming the Qwen3-VL arm after any interruption

```bash
cd /iopsstor/scratch/cscs/dbartaula/querystream
python3 run_chain.py --list                       # rows/expect per stage
STAGES=omnipro_qwen3 GPUS=0,1,2,3 nohup setsid \
  bash scripts/run_chain_supervised.sh > runs/chain_qwen3.log 2>&1 &
```

Resumability is at three levels — per sample (`--resume` + per-row JSONL
flush), per stage (`run_chain.py` skips a complete stage without loading
weights, and always runs score + report), and per chain (the supervisor
re-enters until complete, bounded by `MAX_ROUNDS`, stopping after two rounds
with no progress). Score and report are pure functions of the JSONL and need
no GPU, so a partial run can always be turned into a PDF:

```bash
python3 score_omnipro.py runs/op_qwen3 && python3 make_omnipro_report.py runs/op_qwen3
```

---

## 3. Layout

```
CLAUDE.md                 deep reference: paper notes + milestone log
HANDOFF.md                this file
OMNIPRO.md                deep reference for the OmniPro arm (M5)
OVOBENCH.md               deep reference for the OVO-Bench arm (M6, TimeChat-Online)
15489_QueryStream_*.pdf   the QueryStream paper
omnipro1.pdf              the OmniPro benchmark paper

querystream/
  config.py               every hyperparameter; [UNSPECIFIED] marks each open choice
  ops.py                  cosine/normalise, M-ROPE builder, grid + score resampling
  dsh_memory.py           per-patch EMA, compare-then-update ordering
  qdp.py                  M_sem, M_temp, M_QDP = AND
  rtar.py                 R_t, D_t, T_t, relevance calibration, A.4 debouncer
  buffer.py               TokenBuffer — JIT flush, paired token/M-ROPE slicing
  encoder.py              legacy + SyntheticEncoder (tests)
  open_clip_wrapper.py    OpenCLIP w/ PROJECTED patch tokens + QuickGELU fix
  qwen_wrapper.py         Qwen2.5-VL + StubBackbone (weightless geometry twin)
  jit_streaming_pipeline.py   the streaming controller
  video_io.py             1-FPS sampling (cv2, PyAV fallback)
  prefetch.py             threaded decode-ahead (decode is the bottleneck)
  eval_cache.py           VideoPass — per-video, query-independent precompute
  eval_runner.py          per-question QDP/RTAR replay over a VideoPass
  profiling.py            Stopwatch / ProfileWriter -> profile_summary.csv
  report.py               matplotlib figures for the PDF
  benchmarks/
    streamingbench.py     manifest, prompt, MCQ parsing, Table-1 aggregation
    judge.py              synchronous judge (parse failures only, off by default)
    batch_judge.py        OpenAI Batch API judge

eval_streaming.py         main CLI (sharding, resume, time budget, keep gate)
finalize_run.py           rebuild summary.json from JSONL after a crash
judge_batch.py            submit / poll / collect the judge batch
make_report.py            -> 4-page PDF
scripts/
  submit_cluster_eval.sh  SLURM: one task per GPU, strided shards, auto-merge
  merge_shards.py         union shard JSONLs + coverage check
  collect_judge_and_report.sh
omnipro_eval/             OmniPro arm — nothing in querystream/ was changed for it
  dataset.py              manifest, 9-task taxonomy, slices, stratified subset
  prompts.py              per-task backbone prompts + output-format constraints
  content.py              count / position / state extractors (deterministic)
  metrics.py              Time F1 / Content Acc / Joint F1, micro+macro, slices
  judge.py                OpenAI Batch judge for the 2 open-ended sub-tasks
  runner.py               OmniProRunner — RTAR DRIVES generation here
  report.py               matplotlib figures

ovo_eval/                 OVO-Bench arm — QDP on the paper's TimeChat-Online-7B
  dataset.py              12-task taxonomy, manifest -> Calls, subset, Table 2 rows
  prompts.py              the 4 official templates, VERBATIM (typos included)
  scoring.py              official scorers + macro-over-tasks aggregation
  runner.py               QDP over one pre-trimmed clip, one answer at its end
  report.py               figures
eval_ovo.py               OVO driver;  score_ovo.py;  make_ovo_report.py
scripts/submit_ovo.sh     SLURM: 3 nodes x 4 GPUs on debug

eval_omnipro.py           OmniPro driver (shard by video, resume, time budget)
score_omnipro.py          offline scorer: merge shards, fold verdicts, aggregate
judge_omnipro.py          judge batch submit / poll / collect
make_omnipro_report.py    -> QueryStream_OmniPro_report.pdf
scripts/submit_omnipro.sh SLURM: 3 nodes x 4 GPUs on debug (90 node-min cap)
scripts/collect_omnipro_judge.sh  poll the judge batch, re-score, rebuild the PDF

tests/
  smoke_test.py           M1, 29 tests, no pytest/open_clip/network needed
  test_integration.py     M2, 29 checks, real stack
  test_eval_equivalence.py M3, 10 checks: cached path == streaming pipeline
  test_omnipro_metrics.py M5, 26 checks, no GPU/network — run this first
  test_prefetch_memory.py prefetcher residency guard, no GPU/network
  test_ovo_scoring.py     M6, 15 checks, no GPU/network — run this first
  make_sample_video.py    generates tests/data/sample_bottle.mp4
```

---

## 4. Commands that work

```bash
# regression (run these before trusting anything)
python tests/smoke_test.py --device cuda              # NOTE: "cuda", not "cuda:0"
python tests/test_integration.py --device cuda:0      # needs ~17 GB
python tests/test_eval_equivalence.py --device cuda:0 --clip-batch 1

# subset eval (what produced runs/sb_subset)
OPENCV_FFMPEG_CAPTURE_OPTIONS="timeout;15000000" python eval_streaming.py \
  --output-dir runs/sb_subset --device cuda:0 --resume --shuffle-seed 0 \
  --decode-workers 3 --prefetch-depth 2 --clip-batch 32 \
  --time-budget-sec 1200 --mask-space score --pool mean \
  --relevance-mode running_minmax --judge none --progress-every 10

# recover a crashed run, then report
python finalize_run.py runs/<name> --force
python make_report.py runs/<name>

# FULL SPLIT — approved and launched 2026-08-25 as jobs 3178994 / 3178995.
# `normal` is unusable (see §5), so it runs on `debug`: 1 node x 4 GPUs x 90 min,
# TIME_BUDGET stops workers cleanly, --resume finishes the rest. Submit twice.
TIME_BUDGET=4900 RUN_NAME=sb_score_mean \
  sbatch --partition=debug --time=01:30:00 --nodes=1 scripts/submit_cluster_eval.sh
RUN_NAME=sb_score_max POOL=max sbatch scripts/submit_cluster_eval.sh   # paper's 29.6% row
RUN_NAME=sb_dense KEEP_MODE=dense sbatch scripts/submit_cluster_eval.sh # 100% control
python scripts/merge_shards.py runs/sb_score_mean --expected 2500

# ---- OmniPro (M5) ----------------------------------------------------------
python tests/test_omnipro_metrics.py -v          # 26 checks, no GPU, no network

# smoke, login node, ~2 min once the models are warm
python eval_omnipro.py --limit 8 --n-samples 40 --prefer-short 1.0 \
    --out runs/op_smoke --device cuda:1

# full benchmark: 4 nodes x 4 GPUs on debug, ~20 min
RUN_NAME=op_full sbatch scripts/submit_omnipro.sh
# same command again resumes it -- workers skip sample_ids already in their shard
RUN_NAME=op_dense   KEEP_MODE=dense          sbatch scripts/submit_omnipro.sh
RUN_NAME=op_relonly GATE_MODE=relevance_only sbatch scripts/submit_omnipro.sh

# score / judge / report -- all pure functions of the JSONL, NO GPU
python score_omnipro.py runs/op_full                 # works mid-run
python make_omnipro_report.py runs/op_full           # works mid-run
export OPENAI_API_KEY=...                            # env only, never a file
python judge_omnipro.py runs/op_full --submit
python judge_omnipro.py runs/op_full --collect --wait 1800
python score_omnipro.py runs/op_full && python make_omnipro_report.py runs/op_full

# or leave it polling in the background (dies with the login session)
nohup bash scripts/collect_omnipro_judge.sh runs/op_full > runs/op_judge.log 2>&1 &
```

---

## 5. Environment gotchas that cost time

- **Dataset is already on disk**, do not re-download:
  `/iopsstor/scratch/cscs/dbartaula/eval_7_10_8_19/simplestreamingeval/data/streamingbench/`
  (`questions_real.json`, 500 videos / 2500 questions, 123 GB of 1080p @ 25 fps).
  OVO-Bench is in the same `data/` dir (286 GB) for M4.
- **Login node `clariden-ln001` has 4 GH200s**, usually busy with the user's own
  `rekv_offline_vqa.py` jobs. Check free memory; Qwen needs ~17 GB. Never kill them.
- **OmniPro is on disk and COMPLETE**: `/iopsstor/scratch/cscs/dbartaula/omnipro_data/`
  — `benchmark.json` (2,700 samples, 9 tasks) + `raw_videos/` (1,262 mp4,
  1280x720 @ ~30 fps, ~32 GB, 66.2 h). It started at 678 videos; the missing 584
  were fetched from `RuixiangZhao/OmniPro` (`repo_type="dataset"`) in 136 s.
  **Do not re-download.** One filename begins with `-`, so `ls *.mp4` fails —
  use `find . -name "*.mp4"`.
- **THE `a168` ACCOUNT CANNOT RUN ON `normal` OR `low`.** Every association in
  it is assigned the **`stop` QOS**, which has `MaxJobsPU=0`, so a job
  submitted to `normal` is admitted, silently given `QOS=stop`, and sits
  `PENDING` forever with `Reason=QOSMaxJobsPerUserLimit`. `--qos=normal` is
  rejected outright (`Invalid qos specification`). Other accounts on the
  cluster (a135, a144) are in the same state, so this is an allocation/hold
  condition, not something wrong with our submission.
  ```
  sacctmgr -n show assoc where account=a168 format=Account,User,QOS   # -> stop
  sacctmgr -n show qos stop format=Name,MaxJobsPU                     # -> 0
  ```
  **`debug` is the only usable partition**, because the partition itself
  overrides with `QoS=debug-qos`. Everything must therefore fit the 90
  node-minute cap and be resumable.
- **The `debug` QOS caps a job at `node=90` TRES-minutes.** `MaxNodes=4` and
  `MaxTime=1:30:00` are *not* the binding limits — node-minutes are. 4 nodes x
  30 min = 120 is rejected outright and the job sits `PENDING` forever with
  `Reason=QOSMaxNodeMinutesPerJob`, which looks exactly like a busy queue.
  Check it with `squeue -j <id> -o "%R"` before assuming contention.
  Workable shapes: **3 x 30 min** (12 GPUs, what `submit_omnipro.sh` ships),
  4 x 22 min, 2 x 45 min. `MaxJobsPU=1` / `MaxSubmitPU=2`: one job runs at a
  time and at most two can be queued — so submit a second identical job to
  resume whatever the first does not finish.
- **`$0` in a batch script is SLURM's per-node spool copy**
  (`/var/spool/slurmd/job<id>/slurm_script`), which exists ONLY on the node
  running the batch step. The fan-out idiom `srun bash "$0"` therefore succeeds
  on that one node and dies with `exit code 127` on every other — it looks like
  a partial hardware failure, not a missing file. Re-exec the copy on shared
  scratch instead (`SELF=$REPO/scripts/...`). **`submit_cluster_eval.sh` still
  has this bug** and is only safe at `--nodes=1`; `submit_omnipro.sh` is fixed.
- **Use the absolute interpreter** `/users/dbartaula/miniforge3/bin/python3` in
  SLURM scripts rather than the bare name.
- **`sbatch` snapshots the script at submit time.** Fixing a bug in the file does
  not fix an already-queued job — cancel and resubmit both.
- **Filesystem is slow**: OpenCLIP load ~135 s, Qwen ~20–40 s. Background every
  long command; the Bash tool's 120 s timeout will kill them otherwise.
- **`decord` is not installed** and conflicts with numpy 2.4.6. Use cv2/PyAV.
- `HF_HOME=/iopsstor/scratch/cscs/dbartaula/hf_cache`; weights already cached.
- **Decode is the bottleneck, and Lustre bites.** Six concurrent cv2 readers hit
  `_opencv_ffmpeg_interrupt_callback Stream timeout` at 42–60 s and killed a run.
  Three workers + `OPENCV_FFMPEG_CAPTURE_OPTIONS="timeout;15000000"` is stable.
  Some videos legitimately log `partial file` / `Invalid NAL unit size`.

---

## 6. Findings that shape everything — do not re-derive

### 6.1 Binary mask pooling cannot reproduce the paper (M2)
QDP scores on CLIP's 16×16 grid; Qwen's tokens sit on 8×8 after a 2×2 spatial
merge, plus a 2-frame temporal merge — **8 CLIP patches per backbone token**.
Measured from a 41.4% CLIP-grid keep rate:

| space | pool | backbone keep |
|---|---|---|
| mask | any | **95.6%** ← no pruning at all |
| mask | majority | 9.1% |
| mask | all | 0.3% |
| **score** | **mean** | **47.8%** ← default |
| score | max | 30.3% (≈ paper's aggressive 29.6%) |

`any` survives if *any* of 8 patches did: `1 − 0.586⁸ ≈ 96%`. Combinatorial, not
data-dependent. Hence `mask_space="score"` is the default. The paper never
mentions this conversion.

### 6.2 τ_rel = 0.60 is unreachable — confirmed six times
Raw `cos(q, mean_pool(v))` on real OpenCLIP: **0.0526–0.3462** over 73k
StreamingBench frame-query pairs, and **0.162–0.315** on OmniPro. Frames above
0.60, on either benchmark: **zero**. Taken literally, R_t is false on
every frame and the model never speaks. Default is `relevance_mode="running_minmax"`;
`relevance_raw` is logged per frame so the mode can be re-chosen post hoc.

### 6.3 The QuickGELU bug (fixed in M3)
`create_model_and_transforms("ViT-L-14", pretrained="openai")` builds **plain
GELU**, but the OpenAI checkpoints were trained with **QuickGELU**. open_clip
only warns. `_resolve_quickgelu()` now corrects it to `ViT-L-14-quickgelu`.
**M1 and M2 ran with the mismatch**; anything measured then is suspect.

### 6.4 fp16 GEMMs are not batch-invariant
`--clip-batch 32` perturbs patch features ~6e-5, flipping ~0.3% of tokens at the
frame-mean threshold. `clip_batch=1` is the shipped default (exact agreement with
the streaming pipeline, and 11.14 ms/frame is what a real 1-FPS stream pays);
the subset run used 32 for throughput, which is logged in its `config.json`.

### 6.5 The per-video cache is exact, not approximate
Three of four heavy stages never see the query — decode, OpenCLIP, Qwen's vision
tower — **and so does the DSH recursion** (`v_dsh` and `M_temp` mention only
frames). Only `M_sem` is query-dependent, and it is one mat-vec. 661k
frame-encodes → 252k. Valid ONLY under `update_all_patches=True`;
`build_video_pass` refuses otherwise. Proven equal to the streaming pipeline by
`tests/test_eval_equivalence.py` (masks bitwise identical, M-ROPE identical).

### 6.6 The prefetcher leaked every decoded video (fixed 2026-08-25)
`VideoPrefetcher.__iter__` built `futures = [pool.submit(...) for ...]` and
iterated it. A `Future` keeps its result alive as long as the Future exists, so
the retained list pinned **every** decoded video in RAM for the whole run. The
bounding semaphore does not help — it limits concurrent *decodes*, not retained
*results*.

Invisible at 720p, lethal at 1080p:

| benchmark | per video | per 125-video shard |
|---|---|---|
| OmniPro (720p, 189 s) | ~0.5 GB | ~55 GB — survived |
| StreamingBench (1080p, 504 s) | ~3.1 GB | **~390 GB — OOM** |

Job 3178994 lost **3 of its 4 workers** to the OOM killer; the survivor sat at
**332 GB RSS** with GPUs 0–2 idle. The symptom is three shards that simply stop
writing with *no traceback and no error rows* — it reads as a cluster or Lustre
problem, which is how the M3b "stream timeout" diagnosis may also have been a
symptom rather than the cause. **Diagnose a stalled shard with
`srun --jobid=<id> --overlap -N1 -n1 nvidia-smi` — idle GPUs mean dead workers.**

Fixed by dropping the list slot and the local after each yield. Guarded by
`tests/test_prefetch_memory.py`, which also re-runs the pre-fix iterator to
prove the check detects the leak (3.91 GB vs 0.59 GB on the same input).

### 6.7 The A.4 debounce gap is scoring-relevant, and 1.0 s is wrong for OmniPro
QueryStream leaves the event-grouping gap unspecified; the StreamingBench
harness uses 1.0 s (one frame at 1 FPS). OmniPro matches responses within ±3 s,
so at a 1 s gap **one real event emits up to three responses of which at most
one can match** — the rest are pure false positives. `event_gap_s` defaults to
the matching tolerance on OmniPro. Every raw `T_t` timestamp is in the JSONL
(`raw_trigger_ts`) so any other gap can be re-scored offline for the two
`time_only` sub-tasks without a GPU.

### 6.8 SLURM sets `SLURM_PROCID=0` in the batch step
Branching on it made the launcher run itself as worker 0 and never fan out — a
4-task job silently evaluating a quarter of the split. Fixed with an explicit
`QS_WORKER=1` sentinel through `srun --export=ALL,QS_WORKER=1`.

---

## 7. Open decisions the user has NOT ruled on

1. **OpenCLIP weights.** The paper cites "OpenCLIP-ViT-L/14 (Cherti et al. 2023)"
   — the LAION paper, whose ViT-L/14 tag is `laion2b_s32b_b82k`. We load **OpenAI**
   CLIP weights through open_clip. This is a real fork that moves the numbers;
   laion2b is not cached and has never been measured. **Ask before the full run.**
2. **Which full-split rows to run.** score/mean (~50–54% keep, the paper's 57.2%
   row), score/max (~30%, the 29.6% row), and/or the dense 100% control. ~1.5 h
   each on 4 GPUs.
3. **Backbone.** The paper's headline swaps QDP+RTAR into TimeChat-Online-7B; we
   use stock Qwen2.5-VL-7B. The honest comparison for our config is the paper's
   Qwen2.5-VL-7B row (73.68), not the QueryStream row (75.32).
4. **Timeliness exponent `p`** in `Score = Σ F·2^(−|t'−t|·p)` is never specified
   — blocks exact reproduction of Table 5. Only matters at M4.
5. **Whether M4 (OVO-Bench) is still wanted.** OmniPro now scores RTAR directly,
   on 9 sub-tasks with an explicit proactive protocol, which is what M4 was for.
   OVO-Bench would add the paper's own Table 2/5 comparison but needs the
   unspecified `p`.
6. **Whether to also run the OmniPro gate ablations** (`GATE_MODE=relevance_only`
   / `density_only`), which are the OmniPro analogue of the paper's Table 5.
   Each is another ~20 min on 4 nodes.

---

## 8. Gotchas in our own code

- `summary.json`'s `run.questions_evaluated` counts only the **current** pass;
  with `--resume` the true total is `overall.total`. (270 total = 120 + 150.)
- `smoke_test.py` takes `--device cuda`, **not** `cuda:0` (argparse choices).
- `--no-qwen` cannot produce backbone planes, so every question errors with
  "no complete visual plane before timestamp". It is a geometry smoke path only.
- A question stamped before 2 s has no complete temporal plane and is logged as
  an error rather than silently scored 0.
- Trailing frames that never complete a 2-frame plane are **not** encoded (Qwen
  edge-pads, which would corrupt the keep-rate denominator); counted in
  `dropped_trailing_frames`.
- The API key is read from `OPENAI_API_KEY` only and must never be written to a
  file, log line, or JSONL.
- **OmniPro `summary.shardNN.json` carries `metrics_unjudged`.** Its `joint_f1`
  and `content_acc` are `None` *by design* — only `*_lb` bounds are filled until
  the judge lands. The real numbers come from `score_omnipro.py`.
- **An OmniPro macro mean can cover fewer than 9 sub-tasks.** A task whose
  content metric is withheld is excluded, not zeroed; `macro_*_n_tasks` records
  how many were covered and the PDF says so explicitly. A partial macro is not
  comparable to the paper's Mean column.
- **OmniPro shards all write into the same run directory** as
  `omnipro_raw_logs.shardNN.jsonl`. `score_omnipro.py` unions every match, last
  write wins per `sample_id`, so a resumed shard supersedes rather than
  double-counts. A truncated final line from a killed worker is skipped.

---

## 9. Suggested next steps

1. **OmniPro:** when job 3175772 lands, `score_omnipro.py` runs automatically
   from the batch script. Then submit the judge batch, collect it, re-score and
   rebuild the PDF. If the job hit its 30-min wall clock, resubmit the *same*
   command — it resumes.
2. Collect the StreamingBench judge batch when it lands; regenerate that PDF.
3. Resolve §7.1 (openai vs laion2b weights) with the user.
4. Get approval, then launch the full 2500-question StreamingBench run(s) on
   `normal`.
5. Decide §7.5 / §7.6 — whether OVO-Bench is still wanted, and whether to run
   the OmniPro gate ablations.

---

## 10. OmniPro (M5) in one page

Full detail in [`OMNIPRO.md`](OMNIPRO.md). The parts you need to not get wrong:

**Why it exists.** StreamingBench stamps each question with a timestamp, so RTAR
is computed and logged there but does not drive anything — Table 1 is a *pruning*
result. OmniPro's Online mode gives one standing instruction at t=0 and scores
*when* the model speaks. **This is the first benchmark where `T_t = R_t ∧ D_t`
drives generation.**

**The architecture did not change.** Nothing in `querystream/` was touched.
`omnipro_eval/runner.py` is a sibling of `eval_runner.StreamingBenchRunner`,
reusing the same masks, thresholds, score-space resampling and JIT buffer.

**The audio caveat, which must always travel with the numbers.** 84% of OmniPro
depends on audio; QueryStream has no audio pathway and the user said not to add
one. Samples are labelled `audio_dependency ∈ {required 65.5%, helpful 18.5%,
none 16.0%}`. `required` is unsolvable by construction. **Those samples are
reported, never dropped** — dropping them would turn a known limitation into a
flattering number. The yardstick is OmniPro's own Table 3:

> **MiniCPM-o 4.5, the only published online row run without audio:
> A+V 20.9 | V-only 13.8 | A-only 10.9.**

**Metrics.** Greedy closest-first matching at ±3 s, then:
`time_P = tp_time/n_emits`, `time_R = tp_time/n_gt`,
`content_acc = tp_content/tp_time`,
**`joint_P = tp_content/n_emits`, `joint_R = tp_content/n_gt`**.
The last pair is the trap: using `tp_content/(tp_content+fp)` drops every
response that matched in time but failed on content, inflating Joint F1 by up to
~18% and worst where content is worst. Asserted in
`tests/test_omnipro_metrics.py::test_joint_precision_denominator_is_all_responses`.

**Three headline slices**, then diagnostics: `audio_required`,
`audio_helpful_none` (the vision-fair slice — the one QueryStream can fairly be
judged on), `event_narration_required`. Plus per-audio-class, per-sub-task, and a
recall-side-only trigger-modality split (a false positive matched no trigger, so
it has no modality and precision is undefined there).

**The weighted headline.** `weighted_joint()` reports benchmark-weighted (the
real 2,700 distribution — the honest number, and it is *meant* to look bad),
subset-weighted, and equal-weighted, always with the weights attached.

**Everything is resumable and reportable mid-run.** Workers pass `--resume` and
`--time-budget-sec`, stop cleanly between videos, and append to their own shard
JSONL. `score_omnipro.py` and `make_omnipro_report.py` are pure functions of
those files and work at any point — the PDF carries a **Run completeness**
section stating coverage and whether any worker hit its budget.
