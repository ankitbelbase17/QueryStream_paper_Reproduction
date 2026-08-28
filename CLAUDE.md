# QueryStream (ICLR 2026) — Working Reference

> **New session? Read [`HANDOFF.md`](HANDOFF.md) first.** It carries the current
> state, what is running, the commands that work, the environment gotchas, and
> the decisions still open. This file is the *deep* reference: paper notes
> (§1–10) and the milestone log (§11–15) — go here for **why** something was
> decided, not for **where we are**.
>
> The reproduction now has two benchmark arms. StreamingBench (§13–14) measures
> **pruning**; OmniPro (§15, and [`OMNIPRO.md`](OMNIPRO.md)) measures
> **proactive triggering**, and is the first place `T_t = R_t ∧ D_t` actually
> drives generation rather than being logged.
>
> Standing rule from the user: build incrementally and **wait for a signal after
> each milestone**. The full 2500-question StreamingBench run has never been
> executed and needs explicit approval. (The OmniPro full run was explicitly
> requested and is a different matter.)

Source: `15489_QueryStream_Advancing_St.pdf` (16 pages, camera-ready).
**Title:** QueryStream: Advancing Streaming Video Understanding with Query-Aware Pruning and Proactive Response
**Authors:** Kairui Zhang (ShanghaiTech), Zhenyu Yang (MAIS/CASIA + UCAS), Bing Wang (TJUT), Shengsheng Qian (corresp., shengsheng.qian@nlpr.ia.ac.cn), Changsheng Xu (CASIA / Peng Cheng Lab)
**Code:** https://github.com/Zhangkr2003/QueryStream

---

## 1. One-paragraph gist

Streaming Video-LLMs prune tokens and decide when to speak using a **query-agnostic "change-is-important"** heuristic (the DTD module of TimeChat-Online). That conflates *visual dynamics* with *semantic relevance*: it fires on scene cuts / irrelevant actors, and misses brief or visually subtle but query-relevant events. QueryStream injects **query-awareness** into both halves of the loop with two **training-free** modules bolted in front of a frozen Video-LLM:

1. **QDP (Query-Aware Differential Pruning)** — keep a patch token only if it is *both* semantically relevant to the query *and* temporally novel vs. a **dynamically smoothed history (DSH)**, not vs. the previous frame.
2. **RTAR (Relevance-Triggered Active Response)** — fire the decoder only when *both* frame-level query relevance *and* information density (the QDP keep rate) are high.

Result: SOTA on StreamingBench and OVO-Bench among online models, matching / beating full-token baselines while dropping >70% of visual tokens. QDP also transfers to **offline** long-video QA as a context-denoiser (VideoMME, LongVideoBench).

---

## 2. Method — exact formulation

Notation: stream `V = {f_1..f_T}`, query `Q`. A lightweight VL encoder `E` (OpenCLIP) gives patch features `{v_t^1..v_t^N}` per frame `f_t` and a query embedding `q`. `sim(·,·)` = cosine similarity. `I(·)` = indicator.

### 2.1 QDP

**Semantic relevance mask** — threshold is the *per-frame mean* similarity (frame-adaptive, no fixed constant, robust to scene complexity):

```
M_sem(t,i) = I( sim(q, v_t^i) > (1/N) * Σ_{j=1..N} sim(q, v_t^j) )
```

**Temporal novelty mask** — compared against the DSH vector `v̄_dsh` maintained *per patch location i*:

```
M_temp(t,i) = I( sim(v_t^i, v̄_dsh,t-1^i) < τ_temp )
```

**DSH update** (EMA, run every frame regardless of trigger):

```
v̄_dsh,t^i = α * v_t^i + (1-α) * v̄_dsh,t-1^i
```

**Final mask** — logical AND (intersection, so the keep rate is *lower* than either filter alone):

```
M_QDP(t,i) = M_sem(t,i) ∧ M_temp(t,i)
```

**M-ROPE handling (important implementation detail):** the mask selects *complete* visual tokens. For each preserved patch, both its feature vector **and its Multi-modal Rotary Position Embedding (M-ROPE)** are retained. Positional embeddings of *discarded* patches are excised, so surviving tokens keep their original, correct `{temporal, height, width}` coordinates. Output = positionally coherent sparse token stream.

### 2.2 RTAR

```
R_t = I( sim(q, v̄_t) > τ_rel )                       # v̄_t = mean-pooled frame feature (holistic relevance)
D_t = I( (1/N) Σ_i M_QDP(t,i) > τ_den )               # keep rate as proxy for info density
T_t = R_t ∧ D_t                                       # trigger the Video-LLM decoder
```

- Relevance gate alone → redundant responses on static-but-relevant scenes (good Acc, bad timeliness).
- Density gate alone (≈ TimeChat-Online) → false fires on irrelevant motion.
- AND of both → timely + contextually appropriate.

### 2.3 Runtime architecture (Fig. 2)

- QueryStream is a **pre-processing gateway** between raw stream and a frozen backbone Video-LLM.
- Two **parallel paths** off one lightweight OpenCLIP encoder: (a) patch-level → QDP mask; (b) frame-level (mean-pooled) → RTAR signals.
- **Original unpruned visual tokens sit in a memory buffer.** On an RTAR trigger, the *accumulated* per-frame masks are applied to that buffer **just-in-time**; the resulting sparse token set + query goes to the Video-LLM.
- Consequence: the expensive LLM is invoked sparingly; CLIP runs every frame, LLM runs only on triggers.

---

## 3. Implementation details / hyperparameters

| Item | Value |
|---|---|
| Feature encoder `E` | **OpenCLIP-ViT-L/14** (Cherti et al. 2023) |
| Backbones | **TimeChat-Online-7B** (main; QueryStream = swap out its query-agnostic DTD for QDP+RTAR) and **Qwen2.5-VL-7B** (zero-shot QDP plug-in) |
| DSH smoothing `α` | **0.1** |
| Temporal novelty `τ_temp` | **0.90** |
| Relevance `τ_rel` | **0.60** |
| Density `τ_den` | **0.15** |
| Frame rate | **1 FPS** (unless stated) |
| Training | **None** — fully zero-shot / plug-and-play |
| Hardware | single **NVIDIA A800 80GB** |
| Efficiency metric | **Token Keep Rate (%)** = % visual tokens retained |
| Tuning set | ~10% random sample per task category of OVO-Bench (~250 samples) |

Note: TimeChat-Online is itself built on Qwen2.5-VL-7B — chosen deliberately so gains are attributable to QDP/RTAR, not backbone differences.

---

## 4. StreamingBench results (Table 1) — the numbers to remember

Sub-tasks: OP (Object Perception), CR (Causal Reasoning), CS (Clips Summarization), ATP (Attribute Perception), EU (Event Understanding), TR (Text-Rich Understanding), PR (Prospective Reasoning), SU (Spatial Understanding), ACP (Action Perception), CT (Counting).

### Headline comparison (all 1 fps, 7B)

| Model | Keep Rate | **All** |
|---|---|---|
| Human | – | 91.46 |
| Gemini 1.5 Pro | – | 75.69 |
| GPT-4o (64f) | – | 73.28 |
| Claude 3.5 Sonnet (20f) | – | 72.44 |
| Qwen2.5-VL-7B (full) | – | 73.68 |
| LLaVA-OneVision-7B (32f) | – | 71.12 |
| Dispider-7B | – | 67.63 |
| VideoLLM-online-8B (2 fps) | – | 35.99 |
| Flash-VStream-7B | – | 23.23 |
| TimeChat-Online-7B | 100% | 75.36 |
| TimeChat-Online-7B | 55.8% | 74.32 |
| TimeChat-Online-7B | 33.0% | 72.96 |
| **QueryStream-7B** | **57.2%** | **75.32** |
| **QueryStream-7B** | **29.6%** | **74.04** |

Key deltas:
- **+1.00** over TimeChat-Online at comparable keep rate (75.32 @ 57.2% vs 74.32 @ 55.8%).
- **−0.04** vs full-token TimeChat-Online (75.32 vs 75.36) while dropping ~43% of tokens — effectively lossless.
- **+1.08** under aggressive pruning (74.04 @ **29.6%** vs 72.96 @ 33.0%) — *better score with fewer tokens*.
- Beats Gemini 1.5 Pro / GPT-4o / Claude 3.5 Sonnet on this benchmark.
- Reasoning-heavy subtasks at the ~60% keep level: **CR +0.79** (84.38 vs 83.59), **TR +0.94** (81.31 vs 80.37).

### Full per-subtask row for QueryStream and its direct baselines

| Model | Keep | OP | CR | CS | ATP | EU | TR | PR | SU | ACP | CT | All |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B | – | 78.32 | 80.47 | 78.86 | 80.45 | 76.73 | 78.50 | 79.63 | 63.41 | 66.19 | 53.19 | 73.68 |
| TimeChat-Online-7B | 100% | 80.22 | 82.03 | 79.50 | 83.33 | 76.10 | 78.50 | 78.70 | 64.63 | 69.60 | 57.98 | 75.36 |
| TimeChat-Online-7B | 55.8% | 81.03 | 83.59 | 78.55 | 81.09 | 76.73 | 80.37 | 75.93 | 63.82 | 68.47 | 47.87 | 74.32 |
| TimeChat-Online-7B | 33.0% | 81.03 | 82.03 | 77.60 | 82.37 | 73.58 | 79.13 | 77.78 | 62.20 | 66.48 | 39.89 | 72.96 |
| **QueryStream-7B** | **57.2%** | 82.38 | 84.38 | 79.18 | 82.37 | 77.99 | 81.31 | 78.70 | 65.04 | 69.32 | 47.34 | **75.32** |
| **QueryStream-7B** | **29.6%** | 82.11 | 83.59 | 78.23 | 82.69 | 75.47 | 80.06 | 79.63 | 63.01 | 67.90 | 42.55 | **74.04** |

Caveat worth remembering: **CT (Counting) regresses hard under any pruning** — 57.98 (full) → 47.34 (57.2%) → 42.55 (29.6%). Counting is the one subtask where dropping tokens genuinely hurts; QueryStream does not fix this.

---

## 5. OVO-Bench results (Table 2)

Categories: Real-Time Visual Perception (OCR, ACR, ATR, STU, FPD, OJR), Backward Tracing (EPM, ASI, HLD), Forward Active Responding (REC, SSR, CRR).

| Model | Keep | RT Perception Avg | Backward Avg | Forward Avg | **Overall** |
|---|---|---|---|---|---|
| Human | – | 93.2 | 92.3 | 92.9 | 92.8 |
| Gemini 1.5 Pro | – | 70.8 | 62.3 | 57.2 | 65.3 |
| GPT-4o (64f) | – | 63.6 | 58.7 | 53.4 | 58.6 |
| Qwen2-VL-7B (64f) | – | 60.7 | 48.6 | 48.9 | 52.7 |
| TimeChat-Online-7B | 100% | 61.9 | 41.7 | 36.7 | 46.7 |
| TimeChat-Online-7B | 55.4% | 61.4 | 44.5 | 36.8 | 47.6 |
| TimeChat-Online-7B | 15.2% | 58.6 | 42.0 | 36.4 | 45.6 |
| **QueryStream-7B** | **52.9%** | **63.1** | **44.9** | **40.2** | **49.4** |
| **QueryStream-7B** | **20.0%** | 61.4 | 42.1 | 39.0 | **47.5** |

- **New SOTA among online models: 49.4** — +2.7 over full-token TimeChat-Online (46.7), +1.8 over its best pruned config (47.6).
- Even at **20.0% keep**, 47.5 still beats every TimeChat-Online variant.
- Biggest gains in **Backward Tracing** and **Forward Active Responding** (the hard temporal-reasoning halves). Forward Avg 36.8 → 40.2 (+3.4).
- Still well below Gemini 1.5 Pro (65.3) — OVO-Bench is far from saturated for 7B online models.

---

## 6. Offline long-video results (Table 3)

| Model | Keep | LongVideoBench | VideoMME overall | VideoMME long |
|---|---|---|---|---|
| Qwen2.5-VL-7B | 100% | 61.5 | 63.2 | 50.4 |
| Qwen2.5-VL-7B **w/ DTD** (query-agnostic) | 53.8% | 61.6 | 63.4 | 51.9 |
| Qwen2.5-VL-7B **w/ QDP** (ours, zero-shot plug-in) | 52.4% | **61.9** | **63.6** | **52.6** |
| TimeChat-Online-7B | 100% | 55.4 | 62.4 | 48.4 |
| TimeChat-Online-7B | 53.7% | 57.1 | 63.3 | 52.4 |
| TimeChat-Online-7B | 15.0% | 57.7 | 62.5 | 49.2 |
| **QueryStream-7B** | 52.4% | 57.3 | **63.8** | **52.9** |
| **QueryStream-7B** | 16.6% | **58.0** | 63.2 | 49.8 |

Takeaways:
- **Pruning > full tokens** on offline long video — QDP is a *context denoiser*, not just a speedup. Qwen2.5-VL: 63.2 → 63.6 with half the tokens; long subset **+2.2** (50.4 → 52.6).
- On LongVideoBench, the *most aggressive* setting (16.6% keep) is the *best* (58.0) — long videos are dominated by redundancy.
- VideoMME reported **w/o subtitles**.

---

## 7. Ablations

### 7.1 QDP components — StreamingBench (Table 4)

| Pruning method | Keep (%) | Score (All) |
|---|---|---|
| No pruning (baseline) | 100.0 | 75.36 |
| + Visual/Temporal pruning only | 63.4 | 74.76 |
| + Semantic pruning only | 61.7 | 74.52 |
| **Full QDP (intersection)** | **57.2** | **75.32** |

Either filter alone *loses* accuracy; their AND both prunes hardest **and** restores accuracy to the full-token level. That's the paper's central empirical claim about synergy.

### 7.2 RTAR triggering policy — OVO-Bench Forward Active Responding (Table 5)

| Trigger | Acc. (Avg) | Score (Avg) |
|---|---|---|
| Density-only (≈ TimeChat-Online) | 36.8 | 29.5 |
| Relevance-only | **40.3** | 30.2 |
| **Full RTAR (R ∧ D)** | 40.2 | **34.6** |

Relevance-only gets peak accuracy but is punished on timeliness (redundant fires on static relevant scenes). Full RTAR is **+4.4 Score** over the next best.

### 7.3 DSH smoothing factor α — OVO-Bench (Fig. 3)

| α | Overall Score | Token Keep Rate |
|---|---|---|
| 1.00 (= naive frame-to-frame) | ~45.8 | 31.5% |
| 0.50 | ~48.1 | 24.8% |
| **0.10 (chosen)** | **49.4** | 20.0% |
| 0.01 | ~48.4 | 13.7% |

α=1.0 (pure frame-diff, i.e. DTD's regime) is the *worst* — direct evidence that smoothed history beats frame-to-frame. α too small → memory too long, slow adaptation.

### 7.4 τ_temp sweep — OVO-Bench val subset (Table 7)

| τ_temp | Keep Rate | Overall Score |
|---|---|---|
| 0.75 | 21.6 | 46.5 |
| 0.85 | 35.1 | 48.2 |
| **0.90** | **52.9** | **49.4** |
| 0.95 | 68.3 | 49.1 |
| 0.98 | 78.5 | 49.0 |

τ_temp is the main knob controlling keep rate.

### 7.5 RTAR threshold grid — OVO-Bench Forward val subset (Table 8)

| τ_rel | τ_den | Avg Score |
|---|---|---|
| 0.50 | 0.15 | 33.1 |
| 0.70 | 0.15 | 33.8 |
| 0.60 | 0.10 | 32.5 |
| 0.60 | 0.20 | 33.2 |
| **0.60** | **0.15** | **34.6** |

### 7.6 Encoder × backbone selection — OVO-Bench val subset (Table 6)

| Feature encoder | Qwen2.5-VL-7B | InternVL2.5-8B | InternVideo2.5-8B |
|---|---|---|---|
| OpenCLIP-ViT-B/32 | 47.2 | 46.9 | 47.8 |
| **OpenCLIP-ViT-L/14** | **48.5** | 48.2 | 49.0 |
| OpenCLIP-ViT-H/14 | 48.9 | 48.4 | 49.2 |

Bigger CLIP → monotonically better, but ViT-H latency is prohibitive for real time; ViT-L/14 is the perf/latency knee. Qwen2.5-VL-7B chosen for fair comparison with TimeChat-Online. QDP improves *all three* backbones → genuinely plug-and-play.

---

## 8. Evaluation protocol caveat (Appendix A.4) — read before reproducing

OVO-Bench's official **online** eval code and TimeChat-Online's real-time inference code were **not public** at the time, so the RTAR ablation (Table 5) uses a **simulated protocol** the authors devised:

1. Run the whole stream to collect all trigger timestamps per model (QueryStream: `T_t` fires; TimeChat-Online: keep rate significantly above a baseline threshold).
2. Group triggers into distinct event intervals; take the **first** timestamp of each interval as the response point `t'_m` (prevents duplicate scoring of one continuous event).
3. Generate the response feeding **only frames up to `t'_m`** — response generation is strictly causal.
4. Metrics:
   - `Acc = (1/N) Σ F(R_{m'}, A_m)`
   - `Score = Σ F(R_{m'}, A_m) · 2^(−|t'_m − t_m|·p)` — penalizes **both early and late** responses.

The trigger-point identification step sees the full stream (an offline simplification); only the answer generation is causal. Numbers are therefore an approximation of true online behavior, and `p` is not specified in the paper.

Minor inconsistency to be aware of: Table 2 reports the 49.4 OVO-Bench score at **52.9% keep rate**, while Fig. 3 plots the α=0.1 optimum (score 49.4) at a **20.0% keep rate**. Table 7 ties 49.4 to 52.9% keep. Treat 52.9% as the Table-2 config.

---

## 9. Limitations (Appendix A.8)

1. **Bound by OpenCLIP's representational quality** — fine-grained detail and abstract relations are weak spots; subtle critical events can still be pruned away.
2. **Single static query only** — no multi-turn / evolving user intent. A continuously updated query state or dialogue history is named as the crucial next step.
3. **Fixed hyperparameters** (α, τ_temp, τ_rel, τ_den) — static thresholds are unlikely to be universally optimal; adaptive thresholding or learned soft gating is proposed future work.
4. (Observed, not stated by authors) **Counting degrades sharply** under pruning on StreamingBench.

---

## 10. Reproduction checklist

- Backbone: TimeChat-Online-7B (main results) — replace its DTD module with QDP; also test QDP on stock Qwen2.5-VL-7B for the offline table.
- Encoder: OpenCLIP-ViT-L/14, patch features for QDP, mean-pooled frame feature for `R_t`.
- Set α=0.1, τ_temp=0.90, τ_rel=0.60, τ_den=0.15, 1 FPS.
- Keep an unpruned token buffer + accumulated masks; apply masks just-in-time on trigger.
- Preserve M-ROPE coordinates of surviving tokens; drop those of pruned ones.
- Benchmarks: StreamingBench, OVO-Bench, VideoMME (no subtitles), LongVideoBench. Metric: accuracy + Token Keep Rate %.
- Target keep rates in the paper: ~57.2% / ~29.6% (StreamingBench), ~52.9% / ~20.0% (OVO-Bench), ~52.4% / ~16.6% (offline).

---

# 11. Reproduction — implementation status

Repo layout (`/iopsstor/scratch/cscs/dbartaula/querystream`):

```
querystream/
  config.py       QueryStreamConfig; every paper hyperparameter + [UNSPECIFIED] markers
  ops.py          cosine/normalise primitives, M-ROPE builder, grid resampling
  encoder.py      OpenCLIPEncoder (projected PATCH tokens) + SyntheticEncoder (tests)
  dsh_memory.py   DSHMemory — per-patch EMA, compare-then-update ordering
  qdp.py          QDPModule — M_sem, M_temp, M_QDP = AND
  rtar.py         RTARModule — R_t, D_t, T_t, relevance calibration, A.4 debouncer
  buffer.py       TokenBuffer — JIT flush, paired token/M-ROPE slicing
tests/
  smoke_test.py   29 tests, no pytest/open_clip/network needed
```

Status: **Milestone 1 complete.** 29/29 pass on CPU (12.6 s) and CUDA (1.6 s),
identical results. Env: GH200 120GB, torch 2.12.1+cu126, transformers 5.9.0.
`open_clip_torch` and `pytest` are NOT installed.

Run: `python tests/smoke_test.py [--device cuda] [-v] [-k filter]`

Milestone 2 = `model_wrapper.py` (Qwen2.5-VL-7B-Instruct + JIT delivery).
Milestone 3 = `eval_streaming.py` (StreamingBench first, then OVO-Bench) with
`.jsonl` logging and wall-clock profiling.

## 11.1 Gaps the paper leaves open (decided in code, all reversible)

1. **CLIP grid != Qwen grid.** QDP scores patches on OpenCLIP-ViT-L/14's 16x16
   grid (224 px); Qwen2.5-VL's visual tokens sit on an 8x8 grid after its 2x2
   spatial merge, plus a 2-frame temporal merge. The mask must be resampled.
   The paper never mentions this. → `ops.resample_patch_mask`, default `"any"`
   pooling (recall-favouring); `"all"`/`"majority"` also implemented.
   `mask_resample_mode` is a real keep-rate knob and a likely source of
   divergence from the reported 57.2%.
2. **`tau_rel = 0.60` is not reachable by raw CLIP cosine.** OpenCLIP ViT-L/14
   image-text cosine for a matching pair sits at ~0.25-0.35. Taken literally,
   `R_t` is False on every frame and the model never speaks — confirmed by a
   smoke test that asserts 0 triggers in `"raw"` mode. Some normalisation is
   implied but unstated. → `RTARConfig.relevance_mode` offers `raw` (literal),
   `running_minmax` (most likely intent), `logit_softmax`, `running_zscore`.
   **Always log `relevance_raw` so the mode can be chosen from the observed
   distribution rather than guessed.**
3. **Frame 0 temporal semantics.** With `v_dsh_0 = v_0`, self-similarity is 1.0
   and the temporal gate would discard the entire opening frame. →
   `first_frame_all_novel=True`: no history exists, so nothing can be redundant.
4. **DSH update scope.** The paper updates "following this check" without saying
   whether pruned patches also update. → `update_all_patches=True` (literal
   reading); `False` restricts the EMA to survivors.
5. **DSH normalisation space.** Unstated. → EMA over L2-normalised features
   (`dsh_space="normalized"`); only cosine is consumed downstream.
6. **Appendix A.4 `event_gap_s`.** No value given. → 1.0 s (= one frame at
   1 FPS), so strictly-consecutive triggers collapse into one response point.
7. **Timeliness exponent `p`** in `Score = Σ F·2^(−|t'−t|·p)` is never
   specified. Blocks exact reproduction of Table 5's Score column.

## 11.2 Correctness anchors baked into the tests

- DSH EMA checked against an independently computed closed-form recursion.
- `similarity()` proven non-mutating; `step()` proven to compare against `t-1`
  (a compare-after-update bug inflates similarity and silently over-prunes).
- Both thresholds proven **strict**: `sim == tau_temp` is rejected, `relevance ==
  tau_rel` and `keep_rate == tau_den` are rejected.
- `M_sem` threshold verified to equal the frame mean, recomputed independently.
- **M-ROPE invariant**: buffered tokens encode their own `(t,h,w)` in their first
  three channels, so `flush().tokens[:, :3] == flush().mrope` is asserted after
  every slice. A separate test prunes whole rows and confirms surviving `h`
  values keep their absolute values (gaps preserved, never renumbered).
- All statistics promoted to fp32 under fp16/bf16 inputs; 600-frame EMA
  stability check at the `tau_temp=0.90` boundary.
- Synthetic fixture has a **moving distractor** region, so `M_QDP` is a strict
  subset of `M_temp` and the conjunction is genuinely exercised. A fixture where
  the two masks coincide would pass a broken semantic filter.

## 11.3 Integration trace (the paper's Appendix A.7 story, reproduced)

```
 f  t(s)    sem   temp    qdp    tok    rel  R  D  T  fire
 0   0.0  0.250  1.000  0.250  0.250  0.000  0  1  0 False   warmup
 1   1.0  0.250  0.500  0.250  0.250  1.000  1  1  1  True   event 1
 2   2.0  0.496  1.000  0.496  0.953  0.000  0  1  0 False   SHOCK suppressed
 3   3.0  0.250  0.500  0.250  0.250  1.000  1  1  1  True   event 2
 4   4.0  0.250  0.500  0.250  0.250  0.978  1  1  1 False   A.4 debounced
```

Frame 2 is the key row: the shock drives the token keep rate to 0.953 and `D_t`
fires, but `R_t = 0` suppresses the response. That is precisely the false-trigger
failure mode QueryStream is built to fix.


---

# 12. Milestone 2 — backbone integration (COMPLETE)

New files:

```
querystream/
  open_clip_wrapper.py        OpenCLIPWrapper — PROJECTED patch tokens
  qwen_wrapper.py             QwenVLWrapper + StubBackbone (weightless geometry twin)
  jit_streaming_pipeline.py   JITStreamingPipeline — the controller
  video_io.py                 1-FPS sampling (cv2, PyAV fallback; decord absent)
tests/
  make_sample_video.py        10 s .mp4 with the Appendix A.7 structure
  test_integration.py         end-to-end, 29 checks
```

Status: **29/29 integration checks pass on the real stack** (OpenCLIP-ViT-L/14 +
Qwen2.5-VL-7B-Instruct, bf16, cuda:0). Milestone 1 still 29/29 — no regression.

```
python tests/make_sample_video.py
python tests/test_integration.py --device cuda:0        # full stack, ~17 GB
python tests/test_integration.py --no-qwen              # StubBackbone, no weights
```

## 12.1 Verified API facts (transformers 5.9.0 / open_clip 3.3.0)

| Fact | Value |
|---|---|
| Backbone class | `Qwen2_5_VLForConditionalGeneration` (underscored, **not** `Qwen25VL…`) |
| `get_video_features` returns | `BaseModelOutputWithPooling`; features are in `.pooler_output`, itself a **tuple** split per video |
| M-ROPE order | `[t, h, w]`, all three offset by `start_position` |
| `time_interval` | `tokens_per_second * int(second_per_grid_t)` = `2 * 2` = **4** at 1 FPS |
| Text after video advances by | `max(llm_grid_h, llm_grid_w)` of the **dense** grid |
| Decode-step positions | `cumsum(attention_mask) - 1 + model.model.rope_deltas` |
| OpenCLIP `visual.proj` | applied to **pooled CLS only** — patch tokens are NOT projected |
| ViT-L/14 geometry | grid 16×16=256, proj 1024→768, `logit_scale` 100.0 |

`build_dense_mrope` is asserted **exactly equal** to
`Qwen2_5_VLModel.get_vision_position_ids` across 30 configs (5 grids × 2 fps ×
3 start offsets). If transformers changes the raster order, that fails loudly.

## 12.2 THE BIG FINDING — binary mask pooling is unusable

QDP scores on CLIP's 16×16 grid; Qwen's tokens are 8×8 after its 2×2 spatial
merge, and 2 frames fold into one temporal plane. So **8 CLIP patches collapse
into 1 backbone token**. Pooling the *binary mask* is catastrophic. Measured on
the sample clip (QDP keep 41.4% on the CLIP grid):

| `mask_space` | pool | backbone keep | inflation |
|---|---|---|---|
| mask | any | **95.6%** | 2.31× |
| mask | majority | 9.1% | 0.22× |
| mask | all | 0.3% | 0.01× |
| **score** | **mean** | **47.8%** | 1.15× |
| score | max | **30.3%** | 0.73× |
| score | min | 40.6% | 0.98× |

With `any`, a token survives if **any** of its 8 contributing patches did:
`1 - 0.586^8 ≈ 96%` — i.e. no pruning at all. This is combinatorial, not
data-dependent. **The paper's 57.2% cannot be produced by binary-mask pooling.**

Fix: `mask_space="score"` (now the default) pools the *continuous* similarity
maps onto the backbone grid and applies the thresholds **there**, so the
frame-adaptive mean threshold keeps its meaning on the grid actually pruned.
`score/mean` → 47.8% (paper: 57.2%); `score/max` → 30.3% (paper aggressive: 29.6%).
τ_temp remains the knob to close the gap.

## 12.3 τ_rel confirmed unreachable on real weights

Measured `sim(q, mean_pool(v))` on real OpenCLIP-ViT-L/14: **0.1776–0.2178**.
τ_rel = 0.60 is never reached, so `relevance_mode="raw"` yields zero triggers.
The Milestone-1 prediction (0.25–0.35) was already generous. Default stays
`running_minmax`; `relevance_raw` is logged on every frame.

## 12.4 VRAM profile (measured, cuda:0, bf16)

```
baseline after model load        16740.5 MB
buffering (T_t == False)         16741.3 .. 16755.8 MB   (drift 14.5 MB)
token buffer itself                  8.64 MB  (1260 unpruned tokens)
peak inside generate()           16889.0 MB   (spike 120.6 MB)
```

Buffering is flat; the spike is confined to `generate()`, as the design requires.

## 12.5 Qualitative result worth following up

On the sample clip (n=1, synthetic — do not over-read), 578 pruned vs 1260 dense
tokens:

- **Pruned (45.9% keep)**: *"a green bottle with a red cap … in the process of
  being opened. The cap is partially removed"* — identifies the bottle and the
  subtle cap-lift event.
- **Dense (100%)**: *"two red squares and one green bottle … a yellow cap"* —
  distracted by the irrelevant red distractor, and wrong about the cap colour.

The dense model attends to the distractor; the pruned model does not. That is
exactly the context-denoising claim of Sec. 4.3, and it is worth checking on
real benchmark data at Milestone 3.

## 12.6 Design decisions taken in Milestone 2

1. **Chunk granularity = `temporal_patch_size` (2 frames).** QDP/RTAR run per
   frame; the vision tower runs per 2-frame plane, matching Qwen's geometry.
   Trailing frames that never complete a plane are **not** encoded (Qwen
   edge-pads a partial plane, which would change the token count and corrupt the
   keep-rate denominator). Counted in `StreamResult.dropped_trailing_frames`.
2. **D_t uses the CLIP-grid QDP keep rate**, per Sec. 3.3 ("the token keep rate
   from our QDP mechanism"). The backbone-grid rate is logged separately as
   `tok_keep_rate` — that is the number comparable to the paper's table.
3. **Vision tower runs per chunk**, so the buffer holds *visual tokens* as
   Sec. 3.1 specifies, not pixels.
4. **`max_pixels` is capped** (default 448²) so Qwen's `smart_resize` cannot make
   the token count — and hence the keep-rate denominator — drift between videos.
5. **`generate()` is used, not a hand-rolled decode loop**: `model.model.rope_deltas`
   is set to `max(prefill_positions) + 1 - prefill_len` so decode steps continue
   the custom prefill correctly.

## 12.7 Environment notes

- `decord` is **not** installed and its lmms-eval pin conflicts with numpy 2.4.6;
  `video_io.py` uses cv2 with a PyAV fallback. Both are present.
- `lmms-eval 0.3.0` is installed but has many unmet/conflicting pins — relevant
  for Milestone 3 if we reuse its StreamingBench harness.
- HF cache is redirected: `HF_HOME=/iopsstor/scratch/cscs/dbartaula/hf_cache`.
  Qwen2.5-VL-7B-Instruct (16 GB) is already there.
- Filesystem is slow: OpenCLIP load ~120 s, Qwen ~31 s. Long-running commands
  must go to the background.
- All 4 GH200s are usually busy with the user's `rekv_offline_vqa.py` MLVU jobs.
  Check `nvidia-smi` for free memory and pass `--device cuda:N`.

---

# 13. Milestone 3 — StreamingBench evaluation harness (COMPLETE, full run PENDING approval)

New files:

```
querystream/
  profiling.py                Stopwatch / ProfileRow / ProfileWriter -> profile_summary.csv
  eval_cache.py               VideoPass + build_video_pass (per-video, query-independent)
  eval_runner.py              StreamingBenchRunner -- per-question QDP/RTAR replay
  benchmarks/
    __init__.py
    streamingbench.py         manifest, prompt, MCQ parsing, Table-1 aggregation
    judge.py                  optional OpenAI fallback for UNPARSEABLE replies only
eval_streaming.py             CLI driver (sharding, resume, keep-rate gate)
scripts/
  submit_cluster_eval.sh      SLURM: one task per GPU, strided video shards, auto-merge
  merge_shards.py             union shard JSONLs -> merged summary + coverage check
tests/
  test_eval_equivalence.py    cached eval path == streaming pipeline, 10 checks
```

## 13.1 Dataset — StreamingBench is already on disk

```
/iopsstor/scratch/cscs/dbartaula/eval_7_10_8_19/simplestreamingeval/data/streamingbench/
    questions_real.json     500 videos, 2500 questions, 246 GB of .mp4
    videos/sample_*.mp4     1920x1080 @ 25 fps, 188-1440 s
```

Confirmed to be the exact split the paper scored: the per-task question counts
reproduce Table 1's percentages to the digit — Counting 188 questions and 47.34%
is **89/188**; Causal Reasoning 128 and 84.38% is **108/128**; overall 75.32% of
2500 is **1883**. Task-type -> paper code mapping is in `streamingbench.TASK_CODE`.

Per-video max question timestamp: mean 504 s, p99 972 s, max 1440 s.

## 13.2 The per-video cache — 661k frame-encodes down to 252k

Streaming each question separately costs 661k frame-encodes; streaming each
*video* once costs 252k. The saving is exact, not approximate, because
**three of the four heavy stages never see the query**:

| stage | depends on Q? |
|---|---|
| frame decode | no |
| OpenCLIP patch features | no |
| Qwen vision tower | no |
| **DSH temporal novelty `temp_sim`** | **no** |
| `M_sem = 1[sim(q,v) > mean_j sim(q,v_j)]` | yes (one mat-vec) |

`v_dsh,t = α v_t + (1-α) v_dsh,t-1` and `M_temp = 1[sim(v_t, v_dsh,t-1) < τ_temp]`
mention only frames, so the whole DSH recursion is cacheable. This holds ONLY
under `update_all_patches=True`; with `False` the EMA is restricted to M_QDP
survivors, which depend on q, and `build_video_pass` refuses to run.

Vision-tower batching is safe: `smart_resize` is a function of (H, W, factor,
min_pixels, max_pixels) only — never of T — so encoding 16 planes at once yields
the same grid as encoding one. The invariant is asserted per batch anyway.

## 13.3 Equivalence proof (tests/test_eval_equivalence.py)

The runner duplicates decision logic, so it is proven rather than argued. Both
paths over the same clip, 10 checks, **10/10 pass at the default `clip_batch=1`**:

- per-frame CLIP-grid QDP keep rate (the D_t signal)
- raw and calibrated relevance (the R_t signal)
- R_t / D_t / T_t per frame
- **backbone plane masks bitwise identical** (0 differing entries)
- kept / dense token counts
- **flushed M-ROPE coordinate set identical**

With `--clip-batch 1` (**now the default**) the encoder is driven exactly as the
pipeline drives it and all ten checks are exact to logging precision (5e-7, the
JSONL's 6-dp rounding).

**fp16 GEMMs are not batch-invariant.** At `--clip-batch 32` the ViT runs a
different GEMM shape and reduction order, perturbing patch features by ~6e-5.
That moves 1-2 of 256 patches across the frame-mean threshold, so the CLIP-grid
keep rate shifts by up to 4e-3 — and on the corrected (QuickGELU) encoder it also
flipped **1 of 320 backbone tokens**, breaking the bitwise mask check. An earlier
note here claimed pooling absorbed the noise entirely; that was true only of the
mismatched-GELU encoder on that one clip, and it was luck, not an invariant.

Hence `clip_batch=1` by default, at three costs and one benefit:
  - 11.14 ms/frame vs 0.88 ms/frame (measured, GH200) -- 12.7x slower
  - ~16% on total pipeline wall clock (63 -> 73 ms/frame)
  - full split 5.1 h -> 5.8 h on one GPU, 1.3 h -> 1.5 h across four
  - `openclip_ms_per_frame` becomes the number a real 1-FPS stream actually pays.
    A batch-32 figure cannot be achieved by a system that sees one frame per
    second, so reporting it as streaming overhead would be misleading.
Raise `--clip-batch` for throughput when exact pipeline agreement is not needed.

## 13.4 THE QUICKGELU BUG — fixed, and it was silently degrading every feature

`open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")` builds
the model with **plain GELU**, but the OpenAI checkpoints were trained with
**QuickGELU**. open_clip only emits a `UserWarning` and proceeds, so every patch
feature feeding QDP and RTAR came from an activation that does not match its own
weights. `_resolve_quickgelu()` now promotes that warning into a corrected model
name (`ViT-L-14-quickgelu`), in both directions, falling back if no variant is
registered. Milestones 1-2 ran with the mismatch.

**Open question for the paper's setting.** The paper cites "OpenCLIP-ViT-L/14
(Cherti et al. 2023)" — the LAION OpenCLIP paper, whose ViT-L/14 tag is
`laion2b_s32b_b82k`, not `openai`. We currently load OpenAI CLIP weights through
open_clip. Measured on the sample clip, related-vs-unrelated query separation:
`openai` 0.2175 vs 0.1485 (Δ 0.069); `laion2b` not yet measured (not cached).
This is a real fork that moves the numbers and is unresolved.

## 13.5 Dry run — 5 videos / 25 questions (SLURM job 3162894, debug, 1x GH200)

```
GATE PASSED: token keep rate 53.89%  (187806/348516)   target 45-58%, paper 57.2%
accuracy 68.00% (17/25)   unparsed 0   errors 0   forced_min_keep 0
```

Per-question keep rate is tight: 46.9% - 58.7%, aggregate 53.89%. Accuracy at
n=25 is noise (EU is 0/1) and must not be read as a result.

Wall-clock profile (`profile_summary.csv`, OVERALL row):

| stage | cost |
|---|---|
| frame decode | **46.01 ms/frame**  <- dominates |
| Qwen vision tower | 15.81 ms/frame |
| OpenCLIP | 1.62 ms/frame (dry run used batch 32; batch 1 costs 11.14) |
| QDP + RTAR | **0.277 ms/frame** |
| LLM prefill (TTFT) | 317.7 ms/question (149 ms @ 504 tokens -> 1592 ms @ 55.9k) |
| LLM decode | 17.2 ms/question (2 new tokens) |

Projected full split at the shipped `clip_batch=1`: ~73 ms/frame x 252k frames
+ 2500 prefills ~= **5.8 h on one GPU, ~1.5 h across 4**. Use `--partition=normal`; debug's 90 node-minute QOS
cap cannot hold it.

## 13.6 tau_rel = 0.60 is unreachable — fourth confirmation, now on benchmark video

Over **2779 real StreamingBench frames**, raw `sim(q, mean_pool(v))` spans
**0.1203 - 0.3392** (mean 0.2503). `relevance_mode="raw"` would fire R_t on zero
frames of the entire benchmark. `running_minmax` is the default and
`relevance_raw` is logged per frame in the JSONL so the mode can be re-chosen
post-hoc from the observed distribution.

RTAR under `running_minmax`: 290 fired events over 2779 frames (~10%). Questions
shorter than `calibration_warmup` (8 frames) never fire, by design — the
calibrator suppresses triggers until its window is trusted.

## 13.7 Design decisions taken in Milestone 3

1. **StreamingBench answers at the question timestamp, not on an RTAR trigger.**
   Table 1 is a pruning result; RTAR is evaluated on OVO-Bench's Forward Active
   Responding split (Table 5), where *when* to speak is what is scored.
   `response_mode="query_ts"` is the default; R_t/D_t/T_t are still computed and
   logged on every frame. `response_mode="rtar"` switches the behaviour.
2. **`query_text` for QDP/RTAR is question + option bodies, not the full MCQ
   prompt.** OpenCLIP's text encoder truncates at 77 tokens; the instruction
   boilerplate would push the actual question out of the window. The backbone
   still receives the complete official prompt.
3. **Exact-match letter parsing is the scorer.** Four ordered patterns, then a
   loose isolated-letter fallback that skips the English article "A ...". The
   OpenAI judge runs ONLY on parse failures, is off by default, and every judged
   item is flagged `judged=True`. 0/25 needed it in the dry run.
4. **Zero-survivor guard.** If no token survives anywhere in a stream, the single
   highest-scoring token is revived and `forced_min_keep=True` is logged, rather
   than handing the backbone an empty visual context.
5. **Strided, not blocked, shards.** Neighbouring videos in the manifest have
   similar durations, so contiguous blocks finish at very different times.
6. **`--keep-mode dense`** gives the 100%-keep control row with everything else
   (RTAR logging, prompting, timing) identical.
7. **The API key is never written to a file.** `judge.py` reads
   `OPENAI_API_KEY` from the environment only; the SLURM script forwards it if
   exported and otherwise runs `--judge none`.

## 13.8 SLURM launcher bug found by the dry run

SLURM sets `SLURM_PROCID=0` in the **batch** step as well as in srun tasks, so
branching on it made the batch script run itself as worker 0 and never fan out —
a 4-task job would have silently evaluated one quarter of the split and skipped
the merge. Job 3162894 did exactly that (1 shard, no merge line in the log).
Fixed with an explicit `QS_WORKER=1` sentinel passed through
`srun --export=ALL,QS_WORKER=1`.

## 13.9 Running it

```bash
# dry run (debug partition, 1 GPU, ~4 min)
LIMIT_VIDEOS=5 ASSERT_KEEP=45,58 RUN_NAME=sb_dryrun EXPECTED_Q=25 \
  sbatch --partition=debug --time=00:30:00 --nodes=1 --ntasks-per-node=1 \
         --gres=gpu:1 scripts/submit_cluster_eval.sh

# full split, 4 GPUs on one node (~1.3 h)
RUN_NAME=sb_score_mean sbatch scripts/submit_cluster_eval.sh

# aggressive-pruning row (paper 29.6%)
RUN_NAME=sb_score_max POOL=max sbatch scripts/submit_cluster_eval.sh

# 100%-keep control row
RUN_NAME=sb_dense KEEP_MODE=dense sbatch scripts/submit_cluster_eval.sh

python scripts/merge_shards.py runs/sb_score_mean --expected 2500
```

Every worker passes `--resume`, so a job that hits the wall clock can be
resubmitted unchanged.

---

# 14. Milestone 3b — first real StreamingBench numbers (subset)

`runs/sb_subset/` — 270 of 2500 questions (10.8%) over 31 videos, uniform random
by video (`--shuffle-seed 0`), two wall-clock-budgeted passes on one GH200
login-node GPU.

```
accuracy (exact match)   71.11%   192/270    95% CI 65.4 - 76.2   (paper 75.32)
token keep rate          50.02%   4,590,808 / 9,178,263           (paper 57.2%)
unparsed 0   errors 0   zero-survivor guard fired 0
```

The CI spans the paper's 75.32%, so this rules out a gross reproduction failure
but cannot resolve a one-point difference. The keep rate, averaged over 9.2M
tokens, is already tight and lands in the paper's regime.

| | OP | CR | CS | ATP | EU | TR | PR | SU | ACP | CT | All |
|---|---|---|---|---|---|---|---|---|---|---|---|
| this run (n=270) | 66.7 | 75.0 | 74.3 | 82.5 | 72.7 | 63.0 | 83.3 | 62.5 | 67.3 | **73.3** | 71.11 |
| n per subtask | 45 | 12 | 35 | 40 | 22 | 27 | 6 | 16 | 52 | 15 | 270 |
| paper QueryStream @57.2% | 82.4 | 84.4 | 79.2 | 82.4 | 78.0 | 81.3 | 78.7 | 65.0 | 69.3 | 47.3 | 75.32 |

Note **CT 73.3% vs the paper's 47.3%** — the paper's one bad subtask is this
run's best. n=15, so it is not yet a finding, but it is the first thing to check
on the full split.

## 14.1 New harness pieces

```
querystream/prefetch.py     VideoPrefetcher -- threaded decode ahead of the GPU
querystream/report.py       matplotlib figures (validated palette, slots 1-3)
querystream/benchmarks/batch_judge.py   OpenAI Batch API judge
finalize_run.py             rebuild summary.json from JSONL after a crash
judge_batch.py              submit / poll / collect the judge batch
make_report.py              -> QueryStream_StreamingBench_report.pdf (4 pages)
scripts/collect_judge_and_report.sh     poll the batch, regenerate the PDF
```

## 14.2 Measured profile (270 questions, 16,222 unique frames)

| stage | cost | scope |
|---|---|---|
| frame decode, worker time | 116.36 ms | per frame |
| frame decode, effective (3 threads) | 38.79 ms | hidden behind GPU |
| Qwen vision tower | 22.56 ms | per frame |
| OpenCLIP (at `--clip-batch 32`) | 1.21 ms | per frame |
| **QDP + RTAR** | **0.277 ms** | per frame — 82x cheaper than the tower it gates |
| LLM prefill / TTFT | 939.6 ms | per question |
| LLM decode | 27.2 ms | per question |

## 14.3 Decode is the bottleneck, and it bites

The first pass **died** after 24 videos: six concurrent cv2 readers on Lustre hit
`_opencv_ffmpeg_interrupt_callback Stream timeout` at 42 s and 60 s, and several
videos log `partial file` / `Invalid NAL unit size`. Three workers plus
`OPENCV_FFMPEG_CAPTURE_OPTIONS="timeout;15000000"` ran clean.

Because the crash lost only the end-of-run aggregation and not the rows,
`finalize_run.py` rebuilds `summary.json` (and the profile means) from the JSONL.
**Always run it before concluding a run is lost.**

## 14.4 tau_rel — fifth confirmation, now over 73k frame-query pairs

raw `cos(q, mean_pool(v))` spans **0.0526 - 0.3462**; frames above 0.60: **0**.
30,524 T_t triggers, 7,931 response events after A.4 debouncing.

## 14.5 The judge is a cross-check, not the scorer

Exact-match parsing resolved **270/270** replies via `pattern0` (a bare letter) —
the judge cannot change the headline number, only bound how parser-dependent it
is. It is submitted over the **Batch API** (`batch_6a8c3abb08748190a52e2eea67cc4e1b`,
270 requests) and was still queued server-side at report time; the collector in
`scripts/collect_judge_and_report.sh` folds it in and regenerates the PDF when it
lands. The judge is never shown the answer key — it only maps free text to a
letter, and scoring happens locally.

## 14.6 What the paper does NOT specify (asked, and answered honestly)

The paper gives the encoder, four thresholds, 1 FPS, zero training and one A800.
It states **no** batch size, **no** dataloader design and **no** mixed-precision
policy. Everything of that kind here is this reproduction's choice: bf16 backbone,
fp16 encoder, **fp32 for every QueryStream statistic** (tau_temp=0.90 sits where
fp16 resolution near 1.0 would make the gate noise-dominated), `max_pixels` capped
at 448^2, 3 decode threads, `--clip-batch 32`.

---

# 15. Milestone 5 — OmniPro Online mode (RTAR finally scored)

Full detail lives in **[`OMNIPRO.md`](OMNIPRO.md)**; this section records *why*
each decision was taken, in the same register as §11–14.

Benchmark paper: `omnipro1.pdf` — *OmniPro: A Comprehensive Benchmark for
Omni-Proactive Streaming Video Understanding*, arXiv 2605.18577 (Ruixiang Zhao,
Jie Yang, Zijie Xin, Tianyi Wang, Fengyun Rao, Jing Lyu, Xirong Li; RUC + WeChat
Vision). 2,700 human-verified samples, 9 sub-tasks, 3 cognitive levels, 1,262
videos, 84% audio-dependent.

## 15.1 Why this benchmark, and what it fixes about §13.7 decision 1

Milestone 3 decision 1 recorded that StreamingBench answers at the question
timestamp because *when* to speak is not what Table 1 scores, and deferred RTAR
evaluation to OVO-Bench's Forward Active Responding split. OmniPro does the same
job better and without OVO-Bench's blocker:

- Its **Online mode** issues the instruction once at t=0 and then requires the
  model to decide, unprompted, at which timestamps to respond — exactly
  `T_t = R_t ∧ D_t`.
- It permits **multiple responses per question** (3.4 ground-truth triggers per
  sample on average) and penalises over-triggering through precision, which
  OmniMMI-Pro's single-response protocol cannot.
- Its metric is fully specified, unlike OVO-Bench's timeliness exponent `p`
  (§8), which still blocks exact reproduction of the paper's Table 5.

So M5 was built instead of M4. OVO-Bench remains open but is no longer the only
route to a scored RTAR.

## 15.2 The audio problem is the whole framing

84% of OmniPro depends on audio, and QueryStream has none. The user's
instruction was explicit — evaluate the existing architecture, do not add an
audio encoder, and accept a low score on the audio-requiring tasks.

The temptation is to evaluate only `audio_dependency ∈ {none, helpful}` and
report that. **That would be dishonest**: it silently redefines the benchmark to
the 34.5% of it the system can attempt. The design decision is therefore:

> Every sample is evaluated. `required` is reported as its own slice, and the
> headline is reported under three explicit weightings, the primary one using
> the **full benchmark's** audio composition (required 65.5%, helpful 18.5%,
> none 16.0%) rather than the evaluated subset's.

The reason the distinction matters concretely: before the missing videos were
fetched, only 1,705 of 2,700 samples had a video on disk, and that on-disk set
was skewed *toward* the easy classes — 46.9% `required` against the true 65.5%.
Subset weighting would have quietly credited that skew. Benchmark weighting
corrects it. Both are reported, with the weights attached to the number.

The yardstick for a vision-only system comes from OmniPro's own Table 3, which
ablates input modality on MiniCPM-o 4.5 (the strongest online model):

| input | Mean online F1 |
|---|---|
| audio only | 10.9 |
| **video only** | **13.8** |
| audio + video | 20.9 |

That 7.1-point A+V→V drop is the size of the handicap, measured by the benchmark
authors on a model purpose-built for the task. Every table in the report places
our number next to the V-only row, not the A+V one.

## 15.3 Metric definitions, and the one that is easy to get wrong

OmniPro Sec. 3.2.1 specifies greedy temporal alignment at ±3 s with a match
valid only if the response is also content-correct. We decompose it:

```
time_P      = tp_time    / n_emits        n_emits = tp_time + fp
time_R      = tp_time    / n_gt           n_gt    = tp_time + fn
content_acc = tp_content / tp_time
joint_P     = tp_content / n_emits
joint_R     = tp_content / n_gt
```

**The trap.** Writing `joint_P = tp_content / (tp_content + fp)` looks like the
natural precision formula and is wrong: it drops every response that matched in
time but failed on content, instead of counting it as a false positive. On a
worked example (tp_time 6, tp_content 3, fp 4, fn 2) that gives 3/7 = 0.4286
instead of 3/10 = 0.30 — a 43% overstatement, and the overstatement grows as
content accuracy falls, i.e. it flatters you exactly where you are weakest.
`tests/test_omnipro_metrics.py::test_joint_precision_denominator_is_all_responses`
pins the correct denominator.

Content accuracy is **conditioned on a time match** by construction: a response
that matched nothing has no reference text to be scored against, so there is no
denominator in which it could appear.

**Micro vs macro.** `overall` pools counts across samples; the paper's Table 2
"Mean" is an equal-weight average over the 9 sub-tasks. Both are reported and
neither alone. A macro that had to exclude a withheld task reports how many
tasks it covered (`macro_*_n_tasks`), and the PDF says so in words — a partial
macro is not comparable to the paper's Mean.

**Unjudged ≠ zero.** A missing judge verdict leaves the match uncounted in both
directions; `content_acc` and `joint_f1` become `None` and only `*_lb` lower
bounds are published, under names that cannot be mistaken for the real thing.
Timing metrics never involve the judge and are always exact. This is the same
policy as §14.5's "the judge is a cross-check, not the scorer", inverted: here
the judge *is* the scorer for two sub-tasks, so its absence must be visible
rather than silently absorbed.

## 15.4 Decisions taken in Milestone 5

1. **Event debounce gap = the matching tolerance (3.0 s), not the
   StreamingBench harness's 1.0 s.** QueryStream Appendix A.4 groups consecutive
   triggers into event intervals and takes the first timestamp of each; the gap
   is unspecified. At 1 s, one real event emits up to three responses inside
   OmniPro's ±3 s window, of which at most one can match — the other two are
   *pure* false positives, an artefact of the gap rather than of the method.
   Mitigation against tuning-to-the-metric: **every raw `T_t` timestamp is
   written to the JSONL** (`raw_trigger_ts`), so any other gap can be re-scored
   offline for the two `time_only` sub-tasks without touching a GPU.
2. **Streams are never truncated.** Stopping a video at its last ground-truth
   trigger would remove the tail's false positives and inflate precision.
   `--max-duration` therefore excludes *whole samples* rather than truncating,
   and every exclusion is recorded in `ManifestStats`.
3. **Prompts carry the output-format constraint, never the answer vocabulary.**
   The benchmark scores counts, grid cells and state names by exact match, so
   the format is part of what is being tested. State is an open vocabulary — 611
   distinct `state_to` values — and pasting candidates into the prompt would be
   leaking ground truth. The model is told the shape of the answer, never its
   content.
4. **Extractors are transcribed strictly from the reference online parser.**
   Each rule that looks over-strict carries a comment naming what a lenient
   variant would inflate: the clock-strip before integer extraction, hyphenated-
   only position labels (so `"bottom center"` matches the bare `center` — the
   wrong cell — as upstream does), whole-reply equality for state rather than a
   substring search. An unparseable reply is **wrong**, not unscorable: failing
   the output format is a failure the benchmark is measuring.
5. **The judge sees the ground truth here** — the opposite of §13.7 decision 3.
   On StreamingBench the judge maps free text onto one of four options and must
   not see the key, or it rationalises agreement. On OmniPro the metric *is*
   "how well does this response match the reference", so the reference is the
   input. What it never sees is another model's answer or the timing verdict.
6. **Only time-matched responses on the two judged sub-tasks are sent.** An
   unmatched response has no reference and is already a false positive. This
   keeps the batch to a few hundred requests rather than tens of thousands.
7. **OpenAI substituted for the paper's Gemini-3-Flash judge**, with the rubric,
   the 1–5 scale and the ≥3 threshold unchanged. A different judge is a
   different instrument; the report states this rather than assuming it away.
8. **Per-task `max_new_tokens`** (8 for counts, 80 for narration). Letting a
   counting task ramble only creates text for the extractor to trip over, and
   decode is charged per token.
9. **Sharding is by video, strided, into one shared run directory.** A video's
   cache serves all its samples, and neighbouring videos have correlated
   durations so blocked shards would finish at very different times. Shard files
   are unioned by `sample_id` with last-write-wins, so a resumed shard
   supersedes rather than double-counts.
10. **No Stopwatch around the per-frame RTAR call.** It consumes Python floats;
    wrapping it would force two CUDA syncs per frame to measure ~10 µs of CPU
    work and serialise the loop. It is covered by the vectorised `qdp_rtar`
    stage. (The first version did wrap it, and measured 0.98 ms/frame against
    §14.2's 0.277 — the sync, not the work.)

## 15.5 Plane masks are a prefix, which is what makes this affordable

An emission at time *t* needs the QDP mask for every plane up to *t*. Plane *j*'s
mask depends only on plane *j*'s own similarity maps — the frame-adaptive
threshold in `M_sem` is a mean over that plane's tokens — so the mask for a
prefix of the stream is exactly the prefix of the whole-video mask tensor.

Consequence: `_plane_masks` is computed **once per (video, query)**, and each
emission is a `TokenBuffer.flush()` over the planes admitted so far. `flush`
already defaults to `consume=False`, so the buffer is appended to as time
advances and flushed non-destructively at each event. Without this, a sample
with 10 emissions would recompute its masks 10 times.

The §13.2 per-video cache is unchanged and still exact: decode, OpenCLIP, the
Qwen vision tower and the whole DSH recursion are query-independent.

## 15.6 Dataset — completed, not merely found

`/iopsstor/scratch/cscs/dbartaula/omnipro_data/` held `benchmark.json` (all
2,700 samples) but only **678 of 1,262 videos**, covering 1,705 samples. The
missing 584 were fetched from `RuixiangZhao/OmniPro` (`repo_type="dataset"`,
`hf_hub_download` with `local_dir`, 12 threads): **584/584 in 136 s, 0
failures**. Coverage is now 2,700/2,700 — 1,262 videos, 66.2 h, 9,051
ground-truth triggers.

This mattered more than convenience: the on-disk subset was biased toward the
audio classes the system can attempt (46.9% `required` against the true 65.5%),
so evaluating it would have needed the benchmark-weighted correction to mean
anything. With full coverage, subset and benchmark weighting agree by
construction.

Videos are 1280×720 @ ~30 fps, mean 189 s — materially cheaper than
StreamingBench's 1080p/504 s, which is why the whole benchmark fits in a
`debug` allocation where the StreamingBench split does not.

## 15.7 Measured behaviour (login-node smoke, n=14 across two runs)

```
token keep rate      0.495 - 0.519      (the QueryStream regime; paper 57.2%)
raw cos(q, v-bar)    0.162 - 0.315      tau_rel = 0.60 -> SIXTH confirmation of
                                        zero reachable frames
openclip             6.56 ms/frame      at --clip-batch 8
qwen vision tower    20.2 ms/frame
qdp + rtar           ~0.3 ms/frame      after removing the per-frame sync
llm prefill (TTFT)   51 - 75 ms         per emitted response, short streams
llm decode           ~19 ms/token
frame decode         43 ms/frame        worker time, 720p; 3 threads hide it
```

Output conformance across every task family was checked by hand: counting tasks
emit bare integers, Target-Ground emits hyphenated grid cells (`center`,
`center-right`), the alert tasks emit one-sentence alerts, and Event-Narration
emits two-sentence commentary. No format post-processing was needed.

## 15.8 RESULTS — full benchmark, 2,700/2,700 samples

SLURM 3175952 + 3175953 (`debug`, 3 nodes x 4 GPUs, 30 min each; the second
resumed the ~400 samples the first did not reach). `runs/op_full/`.
**0 errors, 0 zero-emit samples, 0 emit-cap hits, 0 zero-survivor guards.**

```
responses emitted 24,915     ground-truth triggers 9,051     time-matched 3,110
TIME    P 12.48   R 34.35   F1 18.31        macro 17.67
CONTENT ACC     36.60 (judge landed 2026-08-25; see 19.2)    macro 41.30
JOINT   F1       6.70                                        macro 6.70
token keep rate  47.98%   (2,700 samples, 512,540 frames)
```

| sub-task | n | emits | GT | time F1 | content | joint F1 | MiniCPM V | MiniCPM A+V |
|---|---|---|---|---|---|---|---|---|
| Event-Alert | 300 | 3037 | 395 | 8.2 | 100.0 | 8.2 | 14.9 | 44.2 |
| Target-Ground | 300 | 2782 | 332 | 6.9 | 19.6 | 1.4 | 8.7 | 13.9 |
| State-Monitor | 300 | 3135 | 1340 | 24.9 | 12.4 | 3.1 | 23.3 | 24.3 |
| Snap.-Count | 300 | 2941 | 300 | 7.5 | 13.1 | 1.0 | 16.0 | 21.2 |
| **Cond.-Alert** | 300 | 2461 | 984 | 17.6 | 100.0 | **17.6** | 15.7 | 33.1 |
| Cum.-Count | 300 | 2683 | 1435 | 22.1 | 29.7 | 6.6 | 7.6 | 16.4 |
| Event-Narr. | 300 | 2987 | 1322 | 19.9 | 36.4 | 7.2 | 3.5 | 6.9 |
| Dedup.-Count | 300 | 3124 | 1234 | 21.1 | 33.5 | 7.1 | 27.3 | 20.5 |
| Step-Inst. | 300 | 1765 | 1709 | 30.9 | 26.8 | 8.3 | 7.5 | 7.9 |

Slices: audio-required (n=1768) time F1 17.2 / joint 5.5; audio helpful+none
(n=932) time F1 19.1 / joint 7.3. Benchmark-weighted macro joint F1 **6.24**
(lower bound), against MiniCPM-o 4.5's published **13.8** video-only.

Cond.-Alert is the one sub-task where this beats the vision-only reference row.

## 15.9 THE CENTRAL FINDING — the RTAR claim cannot be tested in this configuration

Over **512,540 frames**:

| signal | fires on | why |
|---|---|---|
| `D_t` (density) | **96.4%** of frames | the CLIP-grid QDP keep rate averages **0.467**, far above `tau_den = 0.15`, so the comparison is almost always true |
| `R_t` (relevance) | 43.8% of frames | min-max calibration maps the stream onto [0, 1], so `tau_rel = 0.60` means "top 40% of the trailing window" and fires at about that rate *by construction* |
| `T_t = R AND D` | 42.5% of frames | indistinguishable from `R_t` alone |

Largest raw `cos(q, mean-pooled v)` seen **anywhere** in the benchmark:
**0.3647**, against a stated `tau_rel` of 0.60.

The paper's core RTAR claim (Table 5) is that AND-ing relevance with density
beats either alone. Here the density gate is near-vacuous, so the conjunction is
arithmetically almost equal to relevance alone — and the gate ablation,
reconstructed exactly from the logged per-frame `R_t`/`D_t` traces at zero GPU
cost, shows precisely that:

| gate policy | responses | per GT trigger | time P | time R | time F1 |
|---|---|---|---|---|---|
| **full RTAR (R AND D)** | 24,915 | 2.75 | 12.5 | 34.4 | **18.3** |
| relevance only (R) | 24,722 | 2.73 | 12.4 | 34.0 | 18.2 |
| density only (D) | 4,063 | 0.45 | 7.6 | 3.4 | 4.7 |

Timing only: the full gate is the intersection, so the single-gate policies fire
on a *superset* of its frames and no response text exists for their extra
firings. Density-only emits too few responses for the opposite reason — `D_t` is
true almost everywhere, so the debouncer collapses whole streams into one event.

**This is a finding about the reproduction gap, not about the method.** Taken
literally `R_t` is false on every frame and the model never speaks (§13.6, and
now confirmed a sixth time on 512k frames); *some* calibration is unavoidable,
and whichever one is chosen is this reproduction's invention and is what sets
the firing rate. Resolving it needs either the authors' calibration or a
principled selective one — see §15.11.

## 15.10 The debounce gap is not where the performance is lost

The Appendix A.4 gap is unspecified, so it was swept — **exactly, and without a
GPU**. For gaps g1 < g2 every g2-interval is a union of consecutive
g1-intervals, so the g2 response set is a strict subset of the g1 set, and every
one of those responses already has its generated text in the JSONL. Verified on
the run's own data (`emits == debounce(raw_trigger_ts, 3.0)` for every record;
subset property holds for every larger gap).

| gap (s) | responses | per GT | time P | time R | time F1 |
|---|---|---|---|---|---|
| **3.0 (shipped)** | 24,915 | 2.75 | 12.5 | 34.4 | **18.3** |
| 5.0 | 16,561 | 1.83 | 12.8 | 23.5 | 16.6 |
| 10.0 | 8,812 | 0.97 | 12.9 | 12.6 | 12.8 |
| 20.0 | 4,805 | 0.53 | 13.7 | 7.2 | 9.5 |

**Time precision is flat at 12.5–13.7% across the entire sweep while recall
falls monotonically.** If the responses were merely too dense around the right
instants, merging them would raise the fraction landing on a trigger. It does
not. The timing error is in *where* the gate fires, not in how often — so no
choice of gap recovers it, and there is nothing to gain by tuning it. The
shipped gap remains the headline.

## 15.11 What is NOT established, and what would move it

- ~~**The two open-ended sub-tasks are unjudged**~~ **RESOLVED 2026-08-25.**
  Batch `batch_6a8c894a07988190bb9aeb5fc2589f63` (965 requests) returned;
  `runs/op_full/` was rescored and its PDF regenerated. `n_unjudged` is 0 and
  `content_complete` is true, so the joint F1 is a real number (6.70), not a
  bound. See 19.2 for what the verdict distribution says.
- **The RTAR ablation is inconclusive** for the reason in §15.9. The productive
  next experiments, none of which need new streaming:
  1. a selective relevance calibration (e.g. `running_zscore`, or a quantile
     target chosen *a priori*, not from the score) — needs a re-run, ~30 min;
  2. `tau_den` raised until `D_t` actually binds — the current value cannot,
     given a 0.467 mean keep rate;
  3. `GATE_MODE=relevance_only` / `density_only` runs, which would add the
     content half the offline ablation cannot reach.
- **Long-horizon recall is flat here** (33.1 / 35.0 / 36.4 across short /
  medium / long), which contradicts the paper's Figure 3. Read it as an
  artefact: this system carries no state between responses — each is
  regenerated from the whole pruned buffer — and its trigger rate is
  near-uniform in time. A model that degrades is being penalised for
  maintaining state this one never had.
- **Backbone differs** from the paper's TimeChat-Online-7B (§15.1), and the
  OpenCLIP weights question from §13.4 applies identically.
- **No audio**, by design (§15.2). Every audio-required number is a floor.


---

# 16. The prefetcher memory leak (found on the full StreamingBench run)

`querystream/prefetch.py` is the threaded decode-ahead added in §14.1, and from
Milestone 3b until 2026-08-25 it retained **every decoded video for the entire
run**. The mechanism:

```python
futures = [self._pool.submit(self._decode, it) for it in self.items]
for fut in futures:              # <-- the list outlives every iteration
    dv = fut.result()            # <-- and a Future keeps its result alive
    yield dv
```

A `concurrent.futures.Future` holds `_result` for as long as the Future object
exists, so iterating a retained list pins all of them. The bounding semaphore
does not help: it caps concurrent **decodes**, never retained **results**.

## 16.1 Why it stayed hidden through OmniPro

Decoded frames are uint8 `[T, 3, H, W]` at native resolution, so the cost per
video is set by the benchmark's video profile:

| benchmark | profile | per video | per 125-video shard |
|---|---|---|---|
| OmniPro | 720p, 189 s mean | ~0.5 GB | ~55 GB — survived |
| StreamingBench | 1080p, 504 s mean | ~3.1 GB | **~390 GB** |

The whole 2,700-sample OmniPro run (§15.8) completed with zero errors while
leaking, because 720p at 189 s never accumulated enough to matter. The first
1080p run hit it immediately.

## 16.2 What the failure looked like

Job 3178994 (full split, `debug`, 1 node x 4 GPUs). Three of four shards simply
**stopped writing** — no traceback, no error rows, no non-zero exit. Only
shard 3 kept advancing, and its accuracy and keep rate looked entirely normal.
Every visible symptom pointed at Lustre or the cluster.

The diagnosis came from the node itself:

```
srun --jobid=3178994 --overlap -N1 -n1 nvidia-smi
  0, 0 %,     4 MiB      <- worker dead
  1, 0 %,    22 MiB      <- worker dead
  2, 0 %,     2 MiB      <- worker dead
  3, 33 %, 31424 MiB     <- the only survivor
top: PID 94801 dbartaula  RES 0.332t   <- 332 GB resident
```

The OOM killer had taken three workers. **A stalled shard is a dead process
until proven otherwise; check the GPUs before blaming the filesystem.** This
also casts doubt on the §14.3 diagnosis, where the M3b subset run died with cv2
`Stream timeout` messages at six concurrent readers — memory pressure is at
least as plausible a root cause there, and the timeout was a symptom.

## 16.3 The fix, its guard, and what it did NOT fix

Drop the list slot and the local reference around each yield, capping residency
at roughly `depth + workers` videos as originally intended.

`tests/test_prefetch_memory.py` measures live uint8 frame tensors while draining
the prefetcher, with `read_video_frames` stubbed so it needs no video, no GPU
and no network. It asserts two things, not one:

1. peak residency stays under `(depth + workers + 1)` videos;
2. **re-running the pre-fix iterator exceeds that ceiling** — otherwise the
   test would pass against a broken implementation and prove nothing.

Measured on 20 x 200 MB: **0.59 GB after the fix, 3.91 GB before.** Re-measured
on six real 1080p StreamingBench videos: peak 9.7 GB, **0.2 GB after the loop**.
The prefetcher is genuinely fixed.

**It was not sufficient.** The very next job (3179311) still lost a worker to
SIGKILL, with the survivor at **195.7 GB RSS after 111 videos** — about 1.7 GB
per video processed, which is the order of one decoded video. Four such workers
against 856 GB of node RAM is why one died.

I could not reproduce that growth, and three hypotheses were tested and
**eliminated**:

| hypothesis | test | result |
|---|---|---|
| glibc dynamic mmap threshold stops returning large blocks | 8 x 2 GB alloc/free | RSS 0.14 GB — returned cleanly |
| heap fragmentation from varying multi-GB sizes | 30 blocks, 0.6–3.2 GB | peak 0.42 GB, final 0.18 GB |
| GH200 unified memory puts GPU bytes in RSS | tensor census by device | 12.65 GB live on cuda while RSS was 6.44 GB — GPU is not counted |
| bounded residency of realistic 1080p videos | 12 blocks 1.6–7.9 GB, 3 live | peak 16.8 GB, final 0.18 GB |

A tensor census after `del vp` + `gc.collect()` shows exactly **one** live
`cpu/uint8/dim4` frames tensor — the one the prefetcher generator legitimately
still holds before it resumes — and no accumulation. An end-to-end probe of the
real loop over 9 videos did climb 0.5 -> 21.3 GB, but that is consistent with
the *bounded* high-water mark for `depth + workers` large videos rather than
with unbounded growth.

**So the 195 GB remains unexplained.** It is recorded here rather than papered
over with a guessed cause. What is established: the prefetcher no longer
retains every video, the run completes, and `--resume` absorbs a lost worker.

### Practical guidance until it is understood
- On 1080p material use `--decode-workers 1 --prefetch-depth 1` if a node is
  tight; it bounds residency to ~2-3 videos at the cost of unhidden decode.
- **Always submit a second, identical job.** It resumes and finishes whatever a
  killed worker left; that is what completed this run.
- A shard that stops writing with no traceback is a dead process until proven
  otherwise. Check with
  `srun --jobid=<id> --overlap -N1 -n1 nvidia-smi` — idle GPUs mean dead
  workers — and `ps -eo pid,rss,cmd --sort=-rss` for the RSS.

---

# 17. StreamingBench full split — the reproduction number

`runs/sb_score_mean/`, all 2,500 questions, 500 videos. SLURM 3179311 + 3179312
(`debug`, 1 node x 4 GPUs, 90 min each; the second resumed the shard whose
worker was OOM-killed). **0 unparsed, 0 errors, 100% coverage.**

```
accuracy 69.40%   1735/2500   95% CI 67.56 - 71.18
token keep rate 49.40%   41,058,754 / 83,110,428
```

| Model | Keep | OP | CR | CS | ATP | EU | TR | PR | SU | ACP | CT | All |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **QueryStream (ours)** | **49.4%** | 72.63 | 78.91 | 76.97 | 73.08 | 72.96 | 69.47 | 75.93 | 59.35 | 63.35 | **55.32** | **69.40** |
| paper QueryStream-7B | 57.2% | 82.38 | 84.38 | 79.18 | 82.37 | 77.99 | 81.31 | 78.70 | 65.04 | 69.32 | 47.34 | 75.32 |
| paper Qwen2.5-VL-7B | 100% | 78.32 | 80.47 | 78.86 | 80.45 | 76.73 | 78.50 | 79.63 | 63.41 | 66.19 | 53.19 | 73.68 |

## 17.1 Which row to compare against

**The Qwen2.5-VL-7B row (73.68), not the QueryStream row.** §3 records that the
paper's headline swaps QDP+RTAR into **TimeChat-Online-7B**, which is itself a
streaming fine-tune of Qwen2.5-VL; this reproduction bolts them onto **stock
Qwen2.5-VL-7B-Instruct**. The gap to the paper's QueryStream row therefore
contains the fine-tune as well as the method.

- vs the stock-Qwen baseline: **-4.28** while discarding **50.6%** of visual
  tokens.
- vs the paper's QueryStream row: **-5.92**, at a keep rate 7.8 points *lower*
  (49.4% vs 57.2%). The paper's own TimeChat-Online at 55.8% keep scores 74.32,
  so part of this gap is simply that we prune harder than the row we are
  compared to.

## 17.2 Counting is where this reproduction is ahead

**CT 55.32 against the paper's 47.34 (+7.98).** §4 flags Counting as the one
subtask that "regresses hard under any pruning" — 57.98 full -> 47.34 at 57.2%
keep -> 42.55 at 29.6%. Here it degrades far less: 55.32 at 49.4% keep, close to
the paper's *full-token* 57.98 and above its own 53.19 stock-Qwen figure.

The subset run saw the same sign (73.3%, but n=15, called out at the time as not
yet a finding). At n=188 on the full split it is a finding: **score-space
resampling appears to preserve counting-relevant tokens that binary-mask pooling
loses.** That is the one place this reproduction's forced deviation (§12.2)
looks like an improvement rather than a compromise.

## 17.3 Where it is behind, and the honest candidates

Largest deficits vs the paper's QueryStream row: **TR -11.84**, **OP -9.75**,
**ATP -9.29**, ACP -5.97, SU -5.69. Text-Rich and Object Perception are the
detail-sensitive subtasks — exactly what §9's limitation 1 predicts of a system
bounded by OpenCLIP's representational quality, and exactly where a lower keep
rate should bite hardest.

Unresolved contributors, none of which this run can separate:
1. **Backbone** — stock Qwen2.5-VL vs TimeChat-Online-7B (the largest single
   known difference).
2. **OpenCLIP weights** — §13.4: the paper cites Cherti et al. (LAION,
   `laion2b_s32b_b82k`); we load `openai` through open_clip. Never measured.
3. **Score-space resampling** — §12.2, our invention, unavoidable and
   unspecified by the paper.
4. **Keep rate** — 49.4% vs the 57.2% row. `--tau-temp` is the knob; a sweep
   toward 57% would make the comparison like-for-like.

## 17.4 Consistency with the subset

The 270-question subset (§14) reported 71.11% with a 95% CI of 65.4-76.2. The
full split's 69.40% sits inside it, so the subset was not misleading — it simply
could not resolve a gap of this size. **Quote 69.40%.**

---

# 18. Milestone 6 — OVO-Bench on TimeChat-Online-7B (the paper's own backbone)

Full detail in **[`OVOBENCH.md`](OVOBENCH.md)**; this section records *why*.

## 18.1 The gap this closes

§17.1 named the largest unresolved difference in the StreamingBench result: the
paper's headline puts QDP+RTAR into **TimeChat-Online-7B**, a streaming
fine-tune of Qwen2.5-VL-7B, while every arm of this reproduction so far used
**stock Qwen2.5-VL-7B-Instruct**. That difference was baked into every number
and could not be separated from the method.

This arm removes it. `wyccccc/TimeChatOnline-7B` (16 GB) declares
`architectures: ["Qwen2_5_VLForConditionalGeneration"]`, `model_type
qwen2_5_vl`, and the same vision geometry (`spatial_patch_size 14`,
`tokens_per_second 2`), so `QwenVLWrapper` loads it **unchanged** — only
`--backbone` differs. Confirmed at load: 286.9 s (contended filesystem), then
inference with no wrapper changes at all.

DTD is not disabled so much as **bypassed**. QDP prunes upstream and the
backbone receives an already-sparse token set through `generate_from_pruned`,
so it never sees the dense stream DTD would have filtered. That is precisely the
paper's construction.

## 18.2 The data is fully present — including the split I first thought was missing

```
chunked_videos/{id}.mp4      1,468 clips, the 9 single-timestamp tasks
chunked_videos/{id}_{k}.mp4  1,567 clips, the 3 forward tasks, one per probe
```

3,035 calls over 1,640 samples, 178 h, **0 missing**.

The forward samples (REC/SSR/CRR, ids 1468-1639) have no `{id}.mp4`, and a first
pass therefore concluded Forward Active Responding was unavailable and that
Overall could not be computed. That was wrong: the forward clips exist under
`{id}_{k}.mp4`, 0-indexed, exactly 1,567 of them for exactly 1,567 probes. All
three categories are scoreable and **Overall is computable**.

Because the clips are pre-trimmed to each query's `realtime`, "stream up to the
question" is just "decode the whole clip" — the streaming rule is enforced by
the dataset rather than by us, which removes a whole class of truncation error
that OmniPro required an explicit decision about (§15.4 decision 2).

## 18.3 Metrics are OVO-Bench's own, quirks included

Transcribed from `utils/OVOBenchScore.py` into `ovo_eval/scoring.py`:

```
9 MCQ tasks   score = int(ground_truth_letter in response)     # substring test
REC           digits extracted, JOINED, string-compared        # "1 or 2" -> "12"
SSR / CRR     "Y"/"N" shorthands, then int("Yes"/"No" in response)
category Avg  = MACRO mean over that category's tasks
Overall       = mean of the three category Avgs
```

Reconstructing the paper's printed averages from its own per-task numbers
reproduces them exactly (QueryStream 52.9%: 63.1 / 44.9 / 40.2 / 49.4;
TimeChat-Online 100%: 61.9 / 41.7 / 36.7 / 46.7), which validates the
transcription of both the table and the aggregation rule.

Decisions:
1. **The quirks are pinned by tests, not fixed.** The letter rule scores
   `"A photo of a dog"` when the answer is A; REC punishes hedging brutally.
   Matching the benchmark matters more than matching intuition, because Table 2
   was produced by these rules. `strict_letter_acc` is reported *alongside* as a
   diagnostic so it is visible how much the headline leans on the loose rule.
2. **Macro, not micro.** A test where the two differ by 49 points guards it:
   OCR has 149 calls against FPD's 101, and a micro pool would silently
   reweight every published comparison.
3. **Overall withheld unless all three categories are present** — a
   two-category mean is not the published quantity.
4. **Prompts verbatim**, upstream typos included (`repetively`, `You're task
   is`). A test asserts the typos survive, so a well-meaning cleanup fails
   loudly. The prompts are part of the measuring instrument: the scorer
   substring-matches a letter or a bare Yes/No, and that depends on the model
   having been asked in exactly this way.

## 18.4 What this arm cannot measure, and why

**Table 5 (the RTAR ablation) is out of reach with the released data.** The
Appendix A.4 protocol places a response at a timestamp *the model chooses*, but
every clip is pre-cut to the benchmark's own probe time, so there is no video
beyond the probe to respond at. It would need the untrimmed sources.
Independently, the timeliness exponent `p` is never specified (§8), so the Score
column is unreproducible even with full video. `R_t / D_t / T_t` are computed
and logged per frame anyway.

This is the third arm to run into the same wall from a different direction:
StreamingBench does not score triggering, OmniPro scores it but the calibration
gap makes the ablation vacuous (§15.9), and OVO-Bench would score it but ships
data that cannot support the protocol.

## 18.5 Scale and the subset

3,035 calls / 640,828 frames — 2.5x StreamingBench, far past one `debug`
allocation, so the user's suggested **20% subset**: 584 calls, 328 samples,
37.6 h.

Sampling is **per task, at the sample level**. Per task so the small ones (SSR
has 42 samples) keep a defined accuracy and the macro averages stay meaningful;
at the sample level so a forward sample's probes are never split across the
boundary, which would score a partial trajectory. Pinned by
`test_subset_keeps_a_samples_probes_together`.

## 18.6 Smoke result

12 calls, all EPM (the manifest is id-ordered and EPM starts at 0), 0 errors:

```
token keep rate 51.85%     (paper's comparable QueryStream row: 52.9%)
EPM 58.33                  (paper: QueryStream 56.9, TimeChat-Online 55.9)
responses: "B.", "C.", "A. hanging"  -- clean, format-conforming
```

n=12 on one task, so the accuracy is not a result. What it establishes is that
the paper's backbone loads and answers through the unchanged wrapper, and that
the keep rate lands in the intended regime on it.

## 18.7 RESULTS — 20% subset, 584 calls, 1 error

`runs/ovo_20pct/`. Four login-node GPUs (the `debug` queue was fully occupied by
the user's own self-chaining `sa_full` jobs, which are not ours to preempt).
The one error is a clip too short to form a 2-frame plane -- logged, excluded,
not scored zero.

```
                          Keep   OCR  ACR  ATR  STU  FPD  OJR  EPM  ASI  HLD  REC  SSR  CRR    RT   BW   FW   All
QueryStream (ours)       50.8%  56.7 63.6 39.1 30.6 79.0 54.0 55.9 73.3  2.7 33.0 36.0 36.0  53.8 44.0 35.0  44.3
paper QueryStream 52.9%         75.2 49.5 69.8 50.0 71.3 62.5 56.9 65.5 12.4 35.5 43.3 41.7  63.0 44.9 40.2  49.4
paper TimeChat-Online 100%      75.2 46.8 70.7 47.8 69.3 61.4 55.9 59.5  9.7 31.6 38.5 40.0  61.9 41.7 36.7  46.8
```

**Overall 44.3 vs 49.4 (-5.1)** at a comparable keep rate (50.8% vs 52.9%).
**Backward Tracing essentially reproduces: 44.0 vs 44.9**, with EPM 55.9 vs 56.9.

## 18.8 The deficit is bimodal, and it is not random

| ahead of the paper | behind the paper |
|---|---|
| ACR **+14.1** (63.6 vs 49.5) | ATR **-30.7** (39.1 vs 69.8) |
| ASI **+7.8** (73.3 vs 65.5) | STU **-19.4** (30.6 vs 50.0) |
| FPD **+7.7** (79.0 vs 71.3) | OCR **-18.5** (56.7 vs 75.2) |

We are **ahead** on action recognition, action sequencing and future prediction
-- semantic and temporal tasks -- and **behind** on attribute recognition,
spatial understanding and OCR -- fine-detail perception. A split that clean is
a systematic cause, not sampling noise.

The mechanics were checked before concluding anything: prompts, option
formatting, ground-truth letters and parsing are all correct, and the model
emits clean letters ("C.", "D. T spanner"). These are genuine model errors.

**Prime suspect: `max_pixels = 448^2`.** That cap is *this reproduction's*
choice (§12.6 decision 4), introduced so Qwen's `smart_resize` could not make
the token count -- and hence the keep-rate denominator -- drift between videos.
The paper never states it. Reading "1.88 kWh" off an energy meter, or judging a
surface texture, at 448x448 is close to hopeless, while recognising an action or
ordering a sequence is not. A control at 768^2 on exactly OCR/ATR/STU, same
samples and seed, is in `runs/ovo_detail768/`.

**HLD is a separate, real deficit: 2.7 vs 12.4.** Ground truth there is usually
"Unable to answer" -- the task tests refusal -- and the model confidently picks
a concrete option instead. Everyone is bad at HLD (the paper's TimeChat-Online
manages 9.7), but we are worse, and no amount of resolution fixes a refusal
calibration problem. `strict_letter_acc` is 0.0 on HLD, so it is not a parsing
artefact either.

## 18.9 A bug this run exposed

The OVO driver buffered its JSONL and flushed only after the loop, so three of
four shards sat at **0 bytes on disk while holding 146 completed calls each in
a Python file buffer**. A killed worker would have lost everything it computed
-- precisely the failure `--resume` exists to absorb, and OOM kills have taken
workers in this project more than once (§16). Fixed: flush per call. The
completed run was unaffected, and the fix was verified before any further run.

## 18.10 Operational note

`debug` allows two submissions per user, and both slots were held by the user's
own `sa_full` jobs (`streamagent/sa_full2700.sbatch`). Those are not ours to
cancel or reprioritise; a watcher submits the OVO run when a slot frees.

---

# 19. Milestone 7 — encoder control, judge closure, and a second backbone family

Two evals the user asked for, plus the OmniPro judge finally landing. Run as a
**chain** (`run_chain.py`) rather than as one-off commands, because every arm so
far has had the same two failure modes: a worker dies mid-run, or a run finishes
and the PDF never gets built because the shell went away.

## 19.1 The LAION encoder control — the hypothesis is largely refuted

`runs/ovo_laion/`, 584/584 calls, paired to `runs/ovo_20pct` on **exactly** the
same call ids via `--calls-from`, same backbone (TimeChat-Online-7B), same
thresholds. The encoder weights are the only variable.

§13.4 raised this as an open fork: the paper cites "OpenCLIP-ViT-L/14 (Cherti
et al. 2023)" — the LAION paper, whose ViT-L/14 tag is `laion2b_s32b_b82k` —
while every earlier run loaded the **OpenAI** weights through open_clip. §17.3
listed it second among the candidates for the ~5-point deficit.

```
                openai   laion   delta    dense   paper QS
Real-Time         53.8    54.1    +0.3     55.1      63.0
Backward          44.0    45.4    +1.4     44.4      44.9
Forward           35.0    36.7    +1.6     38.7      40.2
OVERALL           44.3    45.4    +1.1     46.0      49.4
keep rate        50.8%   50.6%
```

Per task, LAION is ahead on ATR (+4.3), CRR (+4.0), ASI (+3.3), HLD (+2.7),
STU (+2.8); behind on FPD (-5.3), EPM (-1.7), SSR (-0.8); identical on OCR, ACR
and OJR.

**It is not significant.** Paired over 583 calls: 25 flips openai-wrong ->
laion-right, 20 the other way, net +5. Exact binomial sign test **p = 0.551**.
The direction is consistent across all three categories, which is mildly
suggestive, but +1.1 closes about a fifth of a 5.1-point gap and cannot be
distinguished from noise at this n.

So of §17.3's four candidates, **the encoder-weights one is now largely
eliminated** and score-space resampling (§12.2) is the main one left standing,
alongside the backbone (which §18 already addressed) and the keep rate.

One thing LAION does change: the raw relevance range widens, per-clip max
`cos(q, v̄)` going from 0.130-0.345 to 0.151-0.396. Still nowhere near
`tau_rel = 0.60` — a **seventh** independent confirmation of §13.6.

And the control that matters most: **dense (46.0) still beats both pruned arms**
(44.3 / 45.4). The paper reports pruning *gaining* +2.6 over full tokens; here
it costs 0.6-1.8 whichever encoder is used. That inversion survives both of the
controls proposed for it (resolution in §18.8, encoder here) and is the single
most robust negative finding of the reproduction.

## 19.2 The OmniPro judge landed — the headline is no longer a bound

Batch `batch_6a8c894a07988190bb9aeb5fc2589f63` (965 requests) returned;
`runs/op_full/` was rescored and its PDF regenerated automatically.

```
n_unjudged 0     content_complete True
OVERALL   time F1 18.31   content acc 36.60   joint F1 6.70
macro     time F1 17.67   content acc 41.30   joint F1 6.70
benchmark-weighted macro joint F1   6.49      (MiniCPM-o 4.5 video-only: 13.8)
```

The two previously withheld sub-tasks: **Event-Narration content 36.4 -> joint
F1 7.2**, **Step-Instruction content 26.8 -> joint F1 8.3**. §15.8's published
lower bound of 4.93 was, as bounds go, loose — the true value is 6.70.

The verdict distribution is the informative part:

| judge score | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| count | 188 | 477 | 203 | 92 | 5 |

Pass rate 31.1% at the >=3 threshold, and **the mass sits on 2** — responses
that landed near the right moment but described it wrong. That is consistent
with §15.10's finding that time precision is flat at 12.5-13.7% across the
entire debounce sweep: the gate is firing at the wrong *instants*, not merely
too often, and the content follows the timing.

## 19.3 Qwen3-VL-8B-Instruct — a genuinely different backbone family

`querystream/qwen3_wrapper.py`. Qwen3-VL is **not** a drop-in for Qwen2.5-VL,
and §18.1's happy accident (TimeChat-Online loading through the unchanged
`QwenVLWrapper`) does not repeat. Four differences, each forcing code:

| | Qwen2.5-VL-7B | Qwen3-VL-8B |
|---|---|---|
| patch / merge | 14 / 2 | **16** / 2 |
| text hidden | 3584 | **4096** |
| video layout | ONE `<vision_start>…</vision_end>` block, `t` advancing by `tokens_per_second * second_per_grid_t` | **one block per temporal plane**, each `T=1`, separated by `<{t:.1f} seconds>` text |
| M-ROPE | sectioned | **interleaved**, `mrope_section [24, 20, 20]` |
| extra features | none | **deepstack** at layers [8, 16, 24], added into hidden states at visual positions |
| `tokens_per_second` | 2 | **absent** |

1. **Per-plane vision blocks.** `get_rope_index` splits `video_grid_thw` per
   plane and passes `time_interval=1`, so a plane's three axes all start from
   the running position and the temporal axis is *constant within a plane*. The
   running position advances by `max(H, W)` of the **dense** merged grid — never
   by how many tokens survived — which is what lets pruning preserve absolute
   coordinates, exactly as in the Qwen2.5 arm but by a different mechanism.
   `VideoChunk.mrope[:, 0]` therefore holds the **plane index**, not
   `t * time_interval`: a plane's absolute start is not knowable until prefill,
   because the timestamp text before it has to be tokenised first.

2. **DeepStack has to be pruned in lockstep.** The vision tower returns three
   extra feature maps, 1:1 with the merged tokens, which the text model *adds*
   into the hidden states at visual positions
   (`hidden_states[visual_pos_masks] += visual_embeds`). Prune the tokens
   without pruning these identically and every survivor gets another token's
   mid-level features added to it — plausible output, silently corrupt. They are
   carried by **concatenating them onto the token feature dimension** (row width
   `4096 * 4 = 16384`), so `TokenBuffer`'s existing row slicing prunes all four
   together and the pairing cannot drift. Costs 4x buffer bytes: ~540 MB for a
   189 s video at 1 FPS, which is what OmniPro averages.

3. **`generate()` cannot be used.** `Qwen3VLModel.forward` computes the
   deepstack features itself from pixels and passes them positionally to the
   text model; no kwarg path accepts precomputed ones. Prefill and greedy decode
   are written out by hand against `model.model.language_model`.

4. **`max_pixels` is a whole-chunk budget.** Qwen3-VL's video `smart_resize`
   bounds `t_bar * h_bar * w_bar`, not `h * w`. The wrapper multiplies the
   per-frame cap by `temporal_patch_size` so the `--max-pixels-side` argument
   keeps the meaning it has in every other arm (§12.6 decision 4). Because the
   constraint is per-frame after that multiplication, it is invariant to how
   many planes are batched into one tower call — which is what §13.2's
   vision-tower batching requires.

### 19.3.1 What the probe caught that review would not have

`tests/test_qwen3_geometry.py`, 23 checks, all passing. Two are load-bearing:

- **`position_ids` compared element-for-element against the model's own
  `get_rope_index`** on a dense prompt, `rope_delta` included. This is the
  Qwen3 counterpart of §12.1's `build_dense_mrope` assertion. A builder even
  slightly wrong about how the running position advances would place every
  plane after the first at the wrong coordinate, and nothing downstream would
  notice.
- **`Qwen3VLVideoProcessor.do_sample_frames` defaults to `True`**, and with no
  `video_metadata` it falls back to *resampling at fps=24*. We hand it chunks
  already sampled at the stream's 1 FPS. Left alone it would silently drop or
  duplicate frames while the token count stayed perfectly plausible. The
  wrapper now passes `do_sample_frames=False` explicitly, and the test pins
  both the plane arithmetic (2/4/8/16 frames -> 1/2/4/8 planes) and a content
  check: two clips differing only in frames 2-3 must produce an identical
  plane 0 and a different plane 1.

A third, caught before this session: the merged tokens live on
`.pooler_output`; `.last_hidden_state` is the **pre-merge** sequence, 4x longer
and 1152-wide. Pairing that with the deepstack maps raises in
`_extract_video_features` rather than corrupting the run.

### 19.3.2 Comparability, and the one thing that is not equal

`runs/op_qwen3` is pinned to `runs/op_full`'s configuration in every respect
except the backbone: same OpenAI ViT-L/14 encoder, same alpha / tau_temp /
tau_rel / tau_den, same 448 px cap, same 3.0 s debounce, no `--max-duration`
(all 2,700 samples evaluated).

What is **not** equal, and cannot be made equal without introducing a second
variable: at the same 448 px cap, Qwen3-VL yields **~180 visual tokens per
plane** against Qwen2.5-VL's **~252**, because patch 16 tiles a frame more
coarsely than patch 14. Equalising it would mean giving one arm a different
pixel budget, which is a worse confound than the one it removes. The keep
**rate** is a ratio and stays comparable; the absolute visual capacity does not.
Stated here rather than left for a reader to discover.

## 19.4 Compute — why this is on the login node

`debug` is the only partition available to account `a168` (`normal` carries QOS
`stop`, MaxJobsPU=0), it allows two submissions per user, and both slots are
held by the user's own `sa_full` jobs, which are not ours to cancel (§18.10).
The partition itself is saturated — 1018 nodes `alloc` and a pending queue
going back thousands of job ids.

So both stages run on the four **login-node** GH200s, as `ovo_laion` already
did. Resumability is at three levels:

1. per call / per sample — every worker passes `--resume` and skips ids already
   in its shard JSONL, which is flushed per row (§18.9);
2. per stage — `run_chain.py` skips a stage whose JSONL already holds `expect`
   rows, without loading any weights, and always runs score + report so a
   completed eval cannot end up without its PDF;
3. per chain — `scripts/run_chain_supervised.sh` re-enters the chain until
   every stage is complete, bounded by `MAX_ROUNDS` and stopping after two
   rounds that make no progress, so a genuinely broken stage surfaces instead
   of spinning.

`run_chain.py --progress` prints one integer (total deduped rows across the
requested stages) and is what lets the supervisor tell a resumed round that made
progress from one that stalled.
