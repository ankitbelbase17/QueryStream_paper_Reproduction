"""OmniPro manifest loading, task taxonomy, and evaluation slices.

OmniPro (arXiv 2605.18577, "OmniPro: A Comprehensive Benchmark for
Omni-Proactive Streaming Video Understanding") evaluates *when to speak* and
*what to say* on continuous audio-visual streams. 2,700 human-verified samples,
9 sub-tasks, 3 cognitive levels, 1,262 videos.

WHY THIS IS THE RIGHT BENCHMARK FOR RTAR
StreamingBench asks a question at a fixed timestamp, so Table 1 of the
QueryStream paper is a *pruning* result and our harness answers at the query
timestamp. OmniPro's Online mode issues the instruction once at t=0 and then
requires the model to decide, unprompted, at which timestamps to respond. That
is exactly what RTAR (`T_t = R_t AND D_t`) computes, so this is the first
benchmark in this reproduction where the trigger half of QueryStream is scored
rather than merely logged.

THE AUDIO CAVEAT, STATED UP FRONT
QueryStream is vision-only: an OpenCLIP-ViT-L/14 image tower in front of
Qwen2.5-VL-7B. No audio encoder exists anywhere in the architecture, and the
user's instruction was explicitly not to add one. OmniPro labels every sample
`audio_dependency` in {required, helpful, none}:

    required  the trigger cannot be perceived without audio (speech or sound)
    helpful   audio assists but the trigger is visually recoverable
    none      the trigger is purely visual

`required` samples are *unsolvable by construction* for this system. They are
NOT dropped -- suppressing them would flatter the result -- they are reported as
their own slice, with the vision-fair slices reported alongside. The paper's own
Table 3 supplies the right yardstick: MiniCPM-o 4.5 run video-only scores 13.8
mean online F1 against 20.9 with audio, so a large `required` deficit is the
expected behaviour of any vision-only system, not a bug in this one.
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Task taxonomy (paper Sec. 3.1.1, Figure 1)
# --------------------------------------------------------------------------- #

ALL_TASKS: Tuple[str, ...] = (
    "instant_event_alert",
    "realtime_state_monitor",
    "snapshot_counting",
    "explicit_target_grounding",
    "event_narration",
    "cumulative_counting",
    "semantic_condition_alert",
    "dedup_counting",
    "sequential_step_instruction",
)

# Paper column order in Table 2, grouped by cognitive level.
PAPER_ORDER: Tuple[str, ...] = (
    "instant_event_alert",        # Perception
    "explicit_target_grounding",
    "realtime_state_monitor",
    "snapshot_counting",
    "semantic_condition_alert",   # Comprehension
    "cumulative_counting",
    "event_narration",
    "dedup_counting",             # Reasoning
    "sequential_step_instruction",
)

SHORT_NAME: Dict[str, str] = {
    "instant_event_alert": "Event-Alert",
    "explicit_target_grounding": "Target-Ground",
    "realtime_state_monitor": "State-Monitor",
    "snapshot_counting": "Snap.-Count",
    "semantic_condition_alert": "Cond.-Alert",
    "cumulative_counting": "Cum.-Count",
    "event_narration": "Event-Narr.",
    "dedup_counting": "Dedup.-Count",
    "sequential_step_instruction": "Step-Inst.",
}

LEVEL: Dict[str, str] = {
    "instant_event_alert": "Perception",
    "explicit_target_grounding": "Perception",
    "realtime_state_monitor": "Perception",
    "snapshot_counting": "Perception",
    "semantic_condition_alert": "Comprehension",
    "cumulative_counting": "Comprehension",
    "event_narration": "Comprehension",
    "dedup_counting": "Reasoning",
    "sequential_step_instruction": "Reasoning",
}

# HOW EACH TASK'S CONTENT IS SCORED (paper Sec. 3.2.1).
#   time_only  the two alert tasks -- a time-matched response is correct
#              regardless of its wording; content is not part of the metric.
#   count / position / state
#              structured output, deterministic exact match, no judge, offline.
#   gpt_judge  the two open-ended generation tasks. The paper uses
#              Gemini-3-Flash on a 1-5 scale with >=3 counted correct; we use
#              the same rubric through OpenAI (see judge.py) because that is the
#              key the user supplied.
TASK_CONTENT_KIND: Dict[str, str] = {
    "instant_event_alert": "time_only",
    "semantic_condition_alert": "time_only",
    "explicit_target_grounding": "position",
    "snapshot_counting": "count",
    "cumulative_counting": "count",
    "dedup_counting": "count",
    "realtime_state_monitor": "state",
    "event_narration": "gpt_judge",
    "sequential_step_instruction": "gpt_judge",
}

JUDGE_TASKS = frozenset(t for t, k in TASK_CONTENT_KIND.items() if k == "gpt_judge")
TIME_ONLY_TASKS = frozenset(t for t, k in TASK_CONTENT_KIND.items() if k == "time_only")

AUDIO_CLASSES: Tuple[str, ...] = ("none", "helpful", "required")

# Audio-dependency composition of the FULL 2,700-sample benchmark. Used to
# reweight a subset's slices back onto the benchmark's own proportions, so a
# convenience sample that happens to be short-video-heavy cannot silently
# rebalance the headline. Verified against benchmark.json at build time.
FULL_BENCHMARK_AUDIO_SHARE: Dict[str, float] = {
    "required": 1768 / 2700,
    "helpful": 500 / 2700,
    "none": 432 / 2700,
}
FULL_BENCHMARK_N = 2700

# --------------------------------------------------------------------------- #
# Published baselines, for the report's comparison table
# --------------------------------------------------------------------------- #

# Table 2, Online-mode evaluation (metric: F1). Audio-visual input.
PAPER_ONLINE: Dict[str, Dict[str, float]] = {
    "LiveStar-8B": {
        "instant_event_alert": 9.7, "explicit_target_grounding": 0.8,
        "realtime_state_monitor": 0.0, "snapshot_counting": 0.0,
        "semantic_condition_alert": 14.7, "cumulative_counting": 0.0,
        "event_narration": 1.6, "dedup_counting": 0.0,
        "sequential_step_instruction": 6.0, "Mean": 3.6,
    },
    "MMDuet2-3B": {
        "instant_event_alert": 12.5, "explicit_target_grounding": 5.3,
        "realtime_state_monitor": 14.9, "snapshot_counting": 11.2,
        "semantic_condition_alert": 21.4, "cumulative_counting": 5.3,
        "event_narration": 3.7, "dedup_counting": 12.7,
        "sequential_step_instruction": 14.7, "Mean": 11.3,
    },
    "MiniCPM-o-4.5-9B": {
        "instant_event_alert": 44.2, "explicit_target_grounding": 13.9,
        "realtime_state_monitor": 24.3, "snapshot_counting": 21.2,
        "semantic_condition_alert": 33.1, "cumulative_counting": 16.4,
        "event_narration": 6.9, "dedup_counting": 20.5,
        "sequential_step_instruction": 7.9, "Mean": 20.9,
    },
}

# Table 3, MiniCPM-o 4.5 under input ablation -- the ONLY published online-mode
# row produced from video WITHOUT audio, and therefore the single directly
# comparable reference point for a vision-only system like QueryStream.
PAPER_ONLINE_VIDEO_ONLY: Dict[str, float] = {
    "instant_event_alert": 14.9, "explicit_target_grounding": 8.7,
    "realtime_state_monitor": 23.3, "snapshot_counting": 16.0,
    "semantic_condition_alert": 15.7, "cumulative_counting": 7.6,
    "event_narration": 3.5, "dedup_counting": 27.3,
    "sequential_step_instruction": 7.5, "Mean": 13.8,
}
PAPER_ONLINE_AUDIO_ONLY_MEAN = 10.9      # same model, audio-only input
PAPER_TOLERANCE_S = 3.0                  # Sec. 3.2.1 and Appendix A.1 default


# --------------------------------------------------------------------------- #
# Sample
# --------------------------------------------------------------------------- #

@dataclass
class Sample:
    """One OmniPro online-mode evaluation item."""

    sample_id: str
    task: str
    video_id: str
    video_path: str                  # absolute
    duration: float
    question: str                    # the standing instruction, issued at t=0
    event: str
    audio_dependency: str            # none | helpful | required
    ground_truth: List[dict]         # each carries a float t_sec, sorted

    @property
    def gt_times(self) -> List[float]:
        return [g["t_sec"] for g in self.ground_truth]

    @property
    def last_gt_time(self) -> float:
        return self.ground_truth[-1]["t_sec"] if self.ground_truth else 0.0

    @property
    def trigger_modalities(self) -> List[str]:
        return [g.get("trigger_type", "unknown") for g in self.ground_truth]

    def query_text(self) -> str:
        """Text handed to OpenCLIP for QDP's `M_sem` and RTAR's `R_t`.

        OpenCLIP's text encoder truncates at 77 BPE tokens, so this is the
        instruction plus the trigger-event phrase and nothing else -- no output
        formatting boilerplate, which would push the actual semantic content out
        of the window. Same rule as the StreamingBench harness. The *backbone*
        still receives the full prompt (see prompts.py).
        """
        q = (self.question or "").strip()
        ev = (self.event or "").strip()
        if ev and ev.lower() not in q.lower():
            return f"{q} {ev}"
        return q


def normalise_ground_truth(gt) -> List[dict]:
    """benchmark.json stores a list[dict]; add a float `t_sec` and sort by it."""
    if isinstance(gt, str):
        gt = json.loads(gt)
    out = []
    for g in gt:
        g = dict(g)
        if g.get("trigger_time_sec") is not None:
            g["t_sec"] = float(g["trigger_time_sec"])
        elif "trigger_time" in g:
            g["t_sec"] = _mmss_to_sec(g["trigger_time"])
        else:
            g["t_sec"] = 0.0
        out.append(g)
    out.sort(key=lambda x: x["t_sec"])
    return out


def _mmss_to_sec(s) -> float:
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    if ":" not in s:
        return float(s)
    sec = 0.0
    for p in s.split(":"):
        sec = sec * 60 + float(p)
    return sec


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

DEFAULT_DATASET_DIR = os.environ.get(
    "OMNIPRO_DATASET_DIR", "/iopsstor/scratch/cscs/dbartaula/omnipro_data")
DEFAULT_BENCHMARK_JSON = os.environ.get(
    "OMNIPRO_BENCHMARK_JSON",
    os.path.join(DEFAULT_DATASET_DIR, "benchmark.json"))


@dataclass
class ManifestStats:
    """Exactly what was kept and what was dropped, and why."""

    n_total: int = 0
    n_kept: int = 0
    dropped_missing_video: int = 0
    dropped_task: int = 0
    dropped_audio: int = 0
    dropped_duration: int = 0
    dropped_no_gt: int = 0
    by_task: Dict[str, int] = field(default_factory=dict)
    by_audio: Dict[str, int] = field(default_factory=dict)
    by_task_audio: Dict[str, int] = field(default_factory=dict)
    n_videos: int = 0
    video_seconds: float = 0.0
    gt_triggers: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def load_manifest(
    *,
    benchmark_json: str = DEFAULT_BENCHMARK_JSON,
    dataset_dir: str = DEFAULT_DATASET_DIR,
    tasks: Optional[Sequence[str]] = None,
    audio: Optional[Sequence[str]] = None,
    max_duration: Optional[float] = None,
    require_video_on_disk: bool = True,
) -> Tuple[List[Sample], ManifestStats]:
    """Load OmniPro samples, keeping only what is actually runnable.

    Nothing is filtered silently: `ManifestStats` records every drop reason, and
    the driver writes it into the run's JSON so the denominator behind any
    reported number is always recoverable.
    """
    with open(benchmark_json) as f:
        raw = json.load(f)

    keep_tasks = set(tasks) if tasks else set(ALL_TASKS)
    keep_audio = set(audio) if audio else set(AUDIO_CLASSES)

    st = ManifestStats(n_total=len(raw))
    out: List[Sample] = []
    for e in raw:
        if e["task"] not in keep_tasks:
            st.dropped_task += 1
            continue
        if e.get("audio_dependency") not in keep_audio:
            st.dropped_audio += 1
            continue
        dur = float(e.get("duration", 0.0))
        if max_duration is not None and dur > max_duration:
            st.dropped_duration += 1
            continue
        vp = e["video_path"]
        vp = vp if os.path.isabs(vp) else os.path.join(dataset_dir, vp)
        if require_video_on_disk and not os.path.exists(vp):
            st.dropped_missing_video += 1
            continue
        gt = normalise_ground_truth(e["ground_truth"])
        if not gt:
            st.dropped_no_gt += 1
            continue
        out.append(Sample(
            sample_id=e["id"], task=e["task"], video_id=e["video_id"],
            video_path=vp, duration=dur, question=e.get("question", ""),
            event=e.get("event") or e.get("question", ""),
            audio_dependency=e.get("audio_dependency", "unknown"),
            ground_truth=gt,
        ))

    _fill_stats(st, out)
    return out, st


def _fill_stats(st: ManifestStats, samples: Sequence[Sample]) -> None:
    st.n_kept = len(samples)
    st.by_task = dict(Counter(s.task for s in samples))
    st.by_audio = dict(Counter(s.audio_dependency for s in samples))
    st.by_task_audio = {
        f"{t}|{a}": n for (t, a), n in
        sorted(Counter((s.task, s.audio_dependency) for s in samples).items())
    }
    vids = {s.video_id: s.duration for s in samples}
    st.n_videos = len(vids)
    st.video_seconds = float(sum(vids.values()))
    st.gt_triggers = sum(len(s.ground_truth) for s in samples)


# --------------------------------------------------------------------------- #
# Subset selection
# --------------------------------------------------------------------------- #

def stratified_subset(
    samples: Sequence[Sample],
    *,
    n_target: int,
    seed: int = 0,
    strata: str = "task_audio",
    prefer_short: float = 0.0,
) -> Tuple[List[Sample], dict]:
    """Pick `n_target` samples spread evenly over strata, not over file order.

    OmniPro's task x audio grid is very uneven on disk -- dedup_counting has 5
    `required` samples and 156 `helpful` ones -- so a uniform random draw would
    leave several cells with an n too small to read. Largest-remainder
    allocation over the strata gives every cell its proportional share and
    hands leftovers to the cells that were rounded down hardest.

    `prefer_short` in [0, 1] biases *within* a stratum toward shorter videos:
    0.0 is a pure random draw, 1.0 takes the shortest available. Runtime is
    dominated by frame decode, which is linear in video seconds, so a small bias
    buys a materially larger n for the same wall clock. It also skews the sample
    toward early-trigger, easier items -- Figure 3 of the paper shows every model
    degrading with trigger time -- so anything above 0 is recorded in the run
    metadata and flagged in the report rather than left implicit.
    """
    rng = random.Random(seed)
    if n_target >= len(samples):
        return list(samples), {"mode": "all", "n": len(samples), "prefer_short": prefer_short}

    def key(s: Sample) -> str:
        if strata == "task_audio":
            return f"{s.task}|{s.audio_dependency}"
        if strata == "task":
            return s.task
        if strata == "audio":
            return s.audio_dependency
        raise ValueError(f"unknown strata {strata!r}")

    buckets: Dict[str, List[Sample]] = defaultdict(list)
    for s in samples:
        buckets[key(s)].append(s)

    total = len(samples)
    exact = {k: len(v) * n_target / total for k, v in buckets.items()}
    alloc = {k: min(len(buckets[k]), int(v)) for k, v in exact.items()}
    # Largest remainder, deterministic tie-break on the stratum name.
    short = n_target - sum(alloc.values())
    order = sorted(exact, key=lambda k: (-(exact[k] - int(exact[k])), k))
    i = 0
    while short > 0 and i < len(order) * 4:
        k = order[i % len(order)]
        if alloc[k] < len(buckets[k]):
            alloc[k] += 1
            short -= 1
        i += 1

    chosen: List[Sample] = []
    for k in sorted(buckets):
        pool = list(buckets[k])
        rng.shuffle(pool)
        if prefer_short > 0:
            # Rank-blend: interpolate between the shuffled order and the
            # shortest-first order, so prefer_short=0.5 is a genuine compromise
            # rather than a hard switch.
            by_dur = sorted(range(len(pool)), key=lambda j: pool[j].duration)
            dur_rank = {j: r for r, j in enumerate(by_dur)}
            pool_idx = sorted(
                range(len(pool)),
                key=lambda j: (1 - prefer_short) * j + prefer_short * dur_rank[j],
            )
            pool = [pool[j] for j in pool_idx]
        chosen.extend(pool[: alloc[k]])

    chosen.sort(key=lambda s: s.sample_id)
    info = {
        "mode": "stratified", "strata": strata, "seed": seed,
        "n_requested": n_target, "n_selected": len(chosen),
        "prefer_short": prefer_short,
        "allocation": {k: alloc[k] for k in sorted(alloc) if alloc[k]},
    }
    return chosen, info


def group_by_video(samples: Sequence[Sample]) -> List[Tuple[str, List[Sample]]]:
    """Group samples by video, so one decode+encode pass serves every question.

    Ordered by video_id for reproducible sharding. The per-video pass is what
    makes this affordable: decode, OpenCLIP, the Qwen vision tower and the whole
    DSH recursion are query-independent (see querystream/eval_cache.py), so only
    `M_sem` and the RTAR relevance re-run per sample.
    """
    by: Dict[str, List[Sample]] = defaultdict(list)
    for s in samples:
        by[s.video_id].append(s)
    return [(v, sorted(by[v], key=lambda s: s.sample_id)) for v in sorted(by)]


# --------------------------------------------------------------------------- #
# Slices
# --------------------------------------------------------------------------- #

@dataclass
class Slice:
    """A named subset of the evaluation, defined by a predicate on a record."""

    name: str
    title: str
    description: str
    kind: str                    # audio | task | modality | all
    predicate: object            # Callable[[dict], bool]
    weight_key: Optional[str] = None   # audio class to weight by, if any


def _audio_slice(names: Sequence[str], title: str, desc: str, key: str,
                 weight_key: Optional[str]) -> Slice:
    want = set(names)
    return Slice(name=key, title=title, description=desc, kind="audio",
                 predicate=lambda r, w=want: r.get("audio_dependency") in w,
                 weight_key=weight_key)


def build_slices(*, tasks: Sequence[str] = ALL_TASKS) -> List[Slice]:
    """The slice registry the report is built from.

    THE THREE HEADLINE SLICES, in the order the user asked for them:

      1. `audio_required`      the 65.5% of OmniPro whose triggers cannot be
                               perceived without audio. Unsolvable by
                               construction for a vision-only system; reported
                               so the cost of the missing modality is visible
                               rather than hidden by exclusion.
      2. `audio_helpful_none`  the vision-fair slice -- audio assists or is
                               irrelevant, so the trigger is recoverable from
                               frames alone. This is the slice on which
                               QueryStream can legitimately be judged.
      3. `event_narration_required`
                               Event-Narration restricted to its audio-required
                               samples: a single-task probe of the free-text
                               generation path under the hardest modality
                               condition, and the one place where the LLM judge
                               (rather than exact match) decides content.

    Everything below the first three is diagnostic breakdown: the audio classes
    separately, every sub-task on its own, and the trigger-modality split that
    mirrors the paper's Figure 4.
    """
    sl: List[Slice] = [
        Slice(name="all", title="All samples",
              description="Every evaluated sample, micro-pooled.",
              kind="all", predicate=lambda r: True),
        _audio_slice(["required"], "Audio required",
                     "Trigger cannot be perceived without speech or sound. "
                     "Unsolvable by a vision-only system; reported, not dropped.",
                     "audio_required", "required"),
        _audio_slice(["helpful", "none"], "Audio helpful / none",
                     "Trigger is recoverable from frames alone. The vision-fair "
                     "slice for QueryStream.",
                     "audio_helpful_none", None),
        Slice(name="event_narration_required", title="Event-Narration (audio required)",
              description="Free-text narration under the hardest modality "
                          "condition; content decided by the LLM judge.",
              kind="task",
              predicate=lambda r: (r.get("task") == "event_narration"
                                   and r.get("audio_dependency") == "required")),
    ]
    for a in AUDIO_CLASSES:
        sl.append(_audio_slice([a], f"Audio = {a}",
                               f"audio_dependency == {a}", f"audio_{a}", a))
    for t in tasks:
        sl.append(Slice(name=f"task_{t}", title=SHORT_NAME.get(t, t),
                        description=f"Sub-task {t} ({LEVEL.get(t, '?')} level).",
                        kind="task", predicate=lambda r, t=t: r.get("task") == t))
    for m in ("visual", "speech", "sound", "visual+speech", "visual+sound"):
        sl.append(Slice(
            name=f"modality_{m.replace('+', '_')}",
            title=f"Trigger modality: {m}",
            description="Per-trigger modality split (paper Figure 4). Scored "
                        "over the ground-truth triggers carrying this label.",
            kind="modality", predicate=lambda r, m=m: True))
    return sl


HEADLINE_SLICES: Tuple[str, ...] = (
    "audio_required", "audio_helpful_none", "event_narration_required")
