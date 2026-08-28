"""OVO-Bench manifest, taxonomy and the paper's reference numbers.

OVO-Bench (Li et al., CVPR 2025) evaluates online video understanding across
three categories and twelve sub-tasks. The QueryStream paper reports it in
Table 2, and it is the benchmark whose *Forward Active Responding* split carries
the RTAR ablation (Table 5).

WHY THIS ARM EXISTS
Milestones 3-5 ran QDP+RTAR on **stock Qwen2.5-VL-7B-Instruct**. The paper's
headline instead swaps QueryStream into **TimeChat-Online-7B**, a streaming
fine-tune of the same base model, replacing its query-agnostic DTD module with
QDP+RTAR. Every previous number in this reproduction therefore carried an
unquantified backbone gap. This arm closes it: same modules, same thresholds,
the paper's actual backbone.

DATA LAYOUT (already on disk, do not re-download)
    ovo_bench_new.json          1,640 samples
    chunked_videos/{id}.mp4     the 9 single-timestamp tasks, PRE-TRIMMED to
                                the question's `realtime`
    chunked_videos/{id}_{k}.mp4 the 3 forward tasks, one clip per probe in
                                `test_info`, 0-indexed, 1,567 in total

Because the clips are pre-trimmed, "stream up to the query time" is simply
"decode the whole clip" -- there is no truncation decision to make, and the
streaming rule is enforced by the dataset rather than by us.
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# taxonomy (OVO-Bench constant.py, verbatim)
# --------------------------------------------------------------------------- #

REAL_TIME_TASKS: Tuple[str, ...] = ("OCR", "ACR", "ATR", "STU", "FPD", "OJR")
BACKWARD_TASKS: Tuple[str, ...] = ("EPM", "ASI", "HLD")
FORWARD_TASKS: Tuple[str, ...] = ("REC", "SSR", "CRR")
ALL_TASKS: Tuple[str, ...] = REAL_TIME_TASKS + BACKWARD_TASKS + FORWARD_TASKS

CATEGORY: Dict[str, str] = {
    **{t: "realtime" for t in REAL_TIME_TASKS},
    **{t: "backward" for t in BACKWARD_TASKS},
    **{t: "forward" for t in FORWARD_TASKS},
}
CATEGORY_ORDER: Tuple[str, ...] = ("realtime", "backward", "forward")
CATEGORY_TITLE: Dict[str, str] = {
    "realtime": "Real-Time Visual Perception",
    "backward": "Backward Tracing",
    "forward": "Forward Active Responding",
}

TASK_NAME: Dict[str, str] = {
    "OCR": "Optical Character Recognition",
    "ACR": "Action Recognition",
    "ATR": "Attribute Recognition",
    "STU": "Spatial Understanding",
    "FPD": "Future Prediction",
    "OJR": "Object Recognition",
    "EPM": "Episodic Memory",
    "ASI": "Action Sequence Identification",
    "HLD": "Hallucination Detection",
    "REC": "Repetition Event Count",
    "SSR": "Sequential Steps Recognition",
    "CRR": "Clue-Reveal Responding",
}

# --------------------------------------------------------------------------- #
# paper Table 2 -- the rows this arm is measured against
# --------------------------------------------------------------------------- #

def _row(vals: Sequence[float]) -> Dict[str, float]:
    """(OCR ACR ATR STU FPD OJR | EPM ASI HLD | REC SSR CRR) -> dict + averages."""
    d = dict(zip(ALL_TASKS, vals))
    for c in CATEGORY_ORDER:
        ts = [t for t in ALL_TASKS if CATEGORY[t] == c]
        d[c] = round(sum(d[t] for t in ts) / len(ts), 2)
    d["overall"] = round(sum(d[c] for c in CATEGORY_ORDER) / 3, 2)
    return d


PAPER: Dict[str, Dict[str, float]] = {
    # name: (keep_rate_pct or None, row)
    "Human Agents": _row([94.0, 92.6, 94.8, 92.7, 91.1, 94.0,
                          92.6, 93.0, 91.4, 95.5, 89.7, 93.6]),
    "Gemini 1.5 Pro": _row([87.3, 67.0, 80.2, 54.5, 68.3, 67.4,
                            68.6, 75.7, 52.7, 35.5, 74.2, 61.7]),
    "GPT-4o (64f)": _row([69.1, 65.1, 65.5, 50.0, 68.3, 63.7,
                          49.8, 71.0, 55.4, 27.6, 73.2, 59.4]),
    "Qwen2-VL-7B (64f)": _row([69.1, 53.2, 63.8, 50.6, 66.3, 60.9,
                               44.4, 66.9, 34.4, 30.1, 65.7, 50.8]),
    "TimeChat-Online-7B (100%)": _row([75.2, 46.8, 70.7, 47.8, 69.3, 61.4,
                                       55.9, 59.5, 9.7, 31.6, 38.5, 40.0]),
    "TimeChat-Online-7B (55.4%)": _row([74.5, 48.6, 68.1, 48.3, 69.3, 59.8,
                                        56.9, 64.9, 11.8, 31.8, 38.5, 40.0]),
    "TimeChat-Online-7B (15.2%)": _row([69.8, 48.6, 64.7, 44.9, 68.3, 55.4,
                                        53.9, 62.8, 9.1, 32.5, 36.5, 40.0]),
    "QueryStream-7B (52.9%)": _row([75.2, 49.5, 69.8, 50.0, 71.3, 62.5,
                                    56.9, 65.5, 12.4, 35.5, 43.3, 41.7]),
    "QueryStream-7B (20.0%)": _row([74.5, 47.7, 70.7, 46.6, 71.3, 57.6,
                                    54.2, 63.5, 8.6, 33.2, 43.1, 40.8]),
}

# The two rows this reproduction is actually trying to hit, and the baseline it
# must beat for the method to have done anything on this backbone.
TARGET_QUERYSTREAM = "QueryStream-7B (52.9%)"
TARGET_BASELINE = "TimeChat-Online-7B (100%)"

# --------------------------------------------------------------------------- #
# samples
# --------------------------------------------------------------------------- #

@dataclass
class Call:
    """One inference: a pre-trimmed clip plus the prompt asked at its end.

    The 9 single-timestamp tasks yield exactly one Call per sample. The 3
    forward tasks yield one per probe in `test_info`, each with its own clip,
    which is why `n_calls != n_samples`.
    """

    call_id: str
    sample_id: int
    task: str
    category: str
    video_path: str
    realtime: float
    probe_index: Optional[int]          # None for single-timestamp tasks
    prompt_kind: str                    # br | REC | SSR | CRR
    question: str = ""
    options: List[str] = field(default_factory=list)
    ground_truth: str = ""              # "A".."D" for br; str(count) for REC;
                                        # "Yes"/"No" for SSR/CRR
    step: str = ""                      # SSR only
    activity: str = ""                  # REC only
    probe_type: Optional[int] = None    # SSR/CRR `type`; REC has `count`


@dataclass
class ManifestStats:
    n_samples_total: int = 0
    n_samples_kept: int = 0
    n_calls: int = 0
    video_seconds: float = 0.0
    dropped_missing_video: int = 0
    dropped_task: int = 0
    by_task_samples: Dict[str, int] = field(default_factory=dict)
    by_task_calls: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


DEFAULT_ROOT = os.environ.get(
    "OVO_ROOT",
    "/iopsstor/scratch/cscs/dbartaula/eval_7_10_8_19/simplestreamingeval/data/ovo_bench")
DEFAULT_ANNO = os.path.join(DEFAULT_ROOT, "ovo_bench_new.json")
DEFAULT_VIDEOS = os.path.join(DEFAULT_ROOT, "chunked_videos")


def load_manifest(
    *,
    anno: str = DEFAULT_ANNO,
    video_dir: str = DEFAULT_VIDEOS,
    tasks: Optional[Sequence[str]] = None,
    require_video: bool = True,
) -> Tuple[List[Call], ManifestStats]:
    """Expand the annotation file into flat `Call`s, one per inference.

    Ground truth is materialised here, in the benchmark's own encoding, so the
    scorer never has to re-derive it:
      * single-timestamp tasks -> `chr(65 + gt)`, i.e. the option LETTER, which
        is what OVO-Bench's scorer substring-matches against the reply;
      * REC  -> `str(count)`;
      * SSR / CRR -> "Yes" when `type == 1`, "No" when `type == 0`.
    """
    with open(anno) as f:
        raw = json.load(f)

    keep = set(tasks) if tasks else set(ALL_TASKS)
    st = ManifestStats(n_samples_total=len(raw))
    calls: List[Call] = []
    kept_samples = 0

    for e in raw:
        task = e["task"]
        if task not in keep:
            st.dropped_task += 1
            continue
        sid = int(e["id"])

        if task in FORWARD_TASKS:
            made = []
            for i, ti in enumerate(e["test_info"]):
                vp = os.path.join(video_dir, f"{sid}_{i}.mp4")
                if require_video and not os.path.exists(vp):
                    st.dropped_missing_video += 1
                    continue
                if task == "REC":
                    gt, kind = str(ti["count"]), "REC"
                else:
                    gt = "Yes" if int(ti["type"]) == 1 else "No"
                    kind = task
                made.append(Call(
                    call_id=f"{sid}_{i}", sample_id=sid, task=task,
                    category=CATEGORY[task], video_path=vp,
                    realtime=float(ti["realtime"]), probe_index=i,
                    prompt_kind=kind,
                    question=e.get("question", ""),
                    ground_truth=gt,
                    step=ti.get("step", ""),
                    activity=e.get("activity", ""),
                    probe_type=ti.get("type"),
                ))
            if made:
                kept_samples += 1
                calls.extend(made)
        else:
            vp = os.path.join(video_dir, f"{sid}.mp4")
            if require_video and not os.path.exists(vp):
                st.dropped_missing_video += 1
                continue
            kept_samples += 1
            calls.append(Call(
                call_id=str(sid), sample_id=sid, task=task,
                category=CATEGORY[task], video_path=vp,
                realtime=float(e["realtime"]), probe_index=None,
                prompt_kind="br",
                question=e.get("question", ""),
                options=list(e.get("options", [])),
                ground_truth=chr(65 + int(e["gt"])),
            ))

    st.n_samples_kept = kept_samples
    st.n_calls = len(calls)
    st.video_seconds = float(sum(c.realtime for c in calls))
    st.by_task_calls = dict(Counter(c.task for c in calls))
    st.by_task_samples = dict(Counter(
        t for t, _ in {(c.task, c.sample_id) for c in calls}))
    return calls, st


def subset(
    calls: Sequence[Call],
    fraction: float,
    *,
    seed: int = 0,
) -> Tuple[List[Call], dict]:
    """Take `fraction` of SAMPLES per task, keeping each sample's probes together.

    Sampling at the sample level, not the call level, matters for the forward
    tasks: REC/SSR/CRR accuracy is pooled over a sample's probes, and splitting
    a sample across the in/out boundary would score a partial trajectory. Every
    task is sampled independently so the macro averages stay defined even for
    the small tasks (SSR has 42 samples).
    """
    if fraction >= 1.0:
        return list(calls), {"mode": "full", "fraction": 1.0, "n_calls": len(calls)}

    by_task: Dict[str, Dict[int, List[Call]]] = defaultdict(lambda: defaultdict(list))
    for c in calls:
        by_task[c.task][c.sample_id].append(c)

    out: List[Call] = []
    alloc: Dict[str, str] = {}
    for task in sorted(by_task):
        sids = sorted(by_task[task])
        n = max(1, round(fraction * len(sids)))
        # PER-TASK RNG, seeded by (seed, task). A single RNG consumed
        # sequentially across tasks makes each task's draw depend on how many
        # tasks precede it, so `--tasks OCR` and a full 12-task run would pick
        # DIFFERENT OCR samples at the same seed -- which silently unpairs any
        # per-task control against a full run. Found the hard way: a 448-vs-768
        # resolution control shared only 18 of 89 samples with its baseline.
        rng = random.Random(f"{seed}:{task}")
        pick = rng.sample(sids, n)
        alloc[task] = f"{n}/{len(sids)}"
        for s in sorted(pick):
            out.extend(by_task[task][s])

    out.sort(key=lambda c: (c.sample_id, c.probe_index if c.probe_index is not None else -1))
    return out, {
        "mode": "stratified_by_task", "fraction": fraction, "seed": seed,
        "n_calls": len(out),
        "n_samples": len({c.sample_id for c in out}),
        "samples_per_task": alloc,
        "video_seconds": round(sum(c.realtime for c in out), 1),
    }
