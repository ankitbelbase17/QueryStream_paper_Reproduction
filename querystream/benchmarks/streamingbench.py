"""StreamingBench (Real-Time Visual Understanding) dataset, prompt and scoring.

Manifest shape (`questions_real.json`, 500 videos / 2500 questions):

    [{"video_path": "./videos/sample_1.mp4",
      "video_categories": "Real-Time Visual Understanding",
      "questions": [{"time_stamp": "00:00:04",
                     "question": "...",
                     "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
                     "answer": "D",
                     "task_type": "Object Perception"}, ...]}, ...]

Streaming protocol: a question stamped at `time_stamp` may only be answered from
frames at or before that second. The harness enforces this by truncating the
stream, never by prompting the model to ignore later frames.

Question counts per task match the paper's Table 1 exactly -- e.g. Counting has
188 questions and the paper reports 47.34%, which is 89/188 to the digit; Causal
Reasoning has 128 and 84.38% is 108/128. That is the cross-check that this file
is reading the same split the paper scored.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# Paper Table 1 column order and abbreviations.
TASK_CODE: Dict[str, str] = {
    "Object Perception": "OP",
    "Causal Reasoning": "CR",
    "Clips Summarize": "CS",
    "Attribute Perception": "ATP",
    "Event Understanding": "EU",
    "Text-Rich Understanding": "TR",
    "Prospective Reasoning": "PR",
    "Spatial Understanding": "SU",
    "Action Perception": "ACP",
    "Counting": "CT",
}
PAPER_ORDER: List[str] = ["OP", "CR", "CS", "ATP", "EU", "TR", "PR", "SU", "ACP", "CT"]

# Reference rows for the summary table (paper Table 1).
PAPER_REFERENCE: Dict[str, Dict[str, float]] = {
    "QueryStream-7B (57.2% keep)": {
        "OP": 82.38, "CR": 84.38, "CS": 79.18, "ATP": 82.37, "EU": 77.99,
        "TR": 81.31, "PR": 78.70, "SU": 65.04, "ACP": 69.32, "CT": 47.34,
        "All": 75.32,
    },
    "Qwen2.5-VL-7B (full tokens)": {
        "OP": 78.32, "CR": 80.47, "CS": 78.86, "ATP": 80.45, "EU": 76.73,
        "TR": 78.50, "PR": 79.63, "SU": 63.41, "ACP": 66.19, "CT": 53.19,
        "All": 73.68,
    },
}

# The StreamingBench MCQ prompt. Identical wording to the official harness so
# accuracy is comparable; changing it changes the numbers.
PROMPT_TEMPLATE = (
    "You are an advanced video question-answering AI assistant. "
    "You have been provided with some frames from the video and a multiple-choice question. "
    "Your task is to analyze the video and provide the best answer.\n\n"
    "Question: {question}\n\n"
    "Options:\n{options}\n\n"
    "Only give the best option's letter (A, B, C, or D) directly."
)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #

@dataclass
class Question:
    video_id: str
    question_id: str
    index_in_video: int
    timestamp_s: int
    question: str
    options: List[str]
    answer: str
    task_type: str

    @property
    def task_code(self) -> str:
        return TASK_CODE.get(self.task_type, self.task_type)

    def prompt(self) -> str:
        return PROMPT_TEMPLATE.format(
            question=self.question, options=format_options(self.options)
        )

    def query_text(self) -> str:
        """What QDP/RTAR score patches against.

        Deliberately NOT the full MCQ prompt: OpenCLIP's text encoder truncates
        at 77 tokens and the boilerplate instruction carries no visual signal,
        so feeding the whole prompt would push the actual question out of the
        context window. The question plus the option texts is the query.
        """
        opts = " ".join(_strip_letter(o) for o in self.options)
        return f"{self.question} {opts}".strip()


@dataclass
class VideoItem:
    video_id: str
    video_path: str
    questions: List[Question] = field(default_factory=list)

    @property
    def max_timestamp_s(self) -> int:
        return max((q.timestamp_s for q in self.questions), default=0)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def timestamp_to_seconds(ts: str) -> int:
    parts = str(ts).strip().split(":")
    return sum(int(float(x)) * 60 ** i for i, x in enumerate(reversed(parts)))


def _strip_letter(option: str) -> str:
    return re.sub(r"^\s*[A-Da-d][\.\):]\s*", "", str(option)).strip()


def format_options(options: Sequence[str]) -> str:
    out: List[str] = []
    for i, o in enumerate(options):
        text = str(o).strip()
        if not re.match(r"^[A-D][\.\):]", text):
            text = f"{chr(65 + i)}. {text}"
        out.append(text)
    return "\n".join(out)


def resolve_video_path(video_path: str, video_dir: str) -> str:
    if os.path.isabs(video_path):
        return video_path
    return os.path.join(video_dir, os.path.basename(video_path))


def load_streamingbench(
    anno_path: str,
    video_dir: str,
    *,
    limit_videos: Optional[int] = None,
    limit_questions: Optional[int] = None,
    task_filter: Optional[Sequence[str]] = None,
    shuffle_seed: Optional[int] = None,
    require_video: bool = True,
) -> List[VideoItem]:
    """Load the manifest into per-video question groups.

    Grouping by video is not cosmetic: frame decoding, OpenCLIP and Qwen's
    vision tower are all query-independent, so one visual pass serves every
    question on that video. See `eval_cache.VideoPass`.
    """
    with open(anno_path) as fh:
        raw = json.load(fh)

    want = {t.upper() for t in task_filter} if task_filter else None
    items: List[VideoItem] = []
    missing = 0

    for entry in raw:
        vp = resolve_video_path(entry["video_path"], video_dir)
        vid = os.path.splitext(os.path.basename(vp))[0]
        if require_video and not os.path.exists(vp):
            missing += 1
            continue
        qs: List[Question] = []
        for i, q in enumerate(entry.get("questions", [])):
            tt = str(q.get("task_type", "")).strip()
            if want is not None and TASK_CODE.get(tt, tt).upper() not in want:
                continue
            qs.append(
                Question(
                    video_id=vid,
                    question_id=f"{vid}_q{i}",
                    index_in_video=i,
                    timestamp_s=timestamp_to_seconds(q.get("time_stamp", "00:00:00")),
                    question=str(q.get("question", "")).strip(),
                    options=[str(o) for o in q.get("options", [])],
                    answer=str(q.get("answer", "")).strip().upper()[:1],
                    task_type=tt,
                )
            )
        if qs:
            items.append(VideoItem(video_id=vid, video_path=vp, questions=qs))

    if missing:
        print(f"[streamingbench] WARNING: {missing} videos missing from {video_dir}")

    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(items)
    if limit_videos is not None:
        items = items[:limit_videos]
    if limit_questions is not None:
        kept, budget = [], limit_questions
        for it in items:
            if budget <= 0:
                break
            it.questions = it.questions[:budget]
            budget -= len(it.questions)
            kept.append(it)
        items = kept
    return items


# --------------------------------------------------------------------------- #
# Answer parsing
# --------------------------------------------------------------------------- #

_PATTERNS = [
    re.compile(r"^\s*\(?([A-D])\)?\s*[\.\):,]?\s*$"),          # "C" / "(C)" / "C."
    re.compile(r"^\s*\(?([A-D])\)?\s*[\.\):]\s"),               # "C. the ball"
    re.compile(r"\b(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-D])\)?\b", re.I),
    re.compile(r"\*\*\s*\(?([A-D])\)?\s*\*\*"),                 # markdown bold
]


# Last resort: any isolated A-D. Runs only after every structured form failed.
_LOOSE = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])")
# "A person walks by" -- the English indefinite article, not an option letter.
_ARTICLE_A = re.compile(r"^A\s+[a-z]")


def parse_mcq(
    response: Optional[str], *, return_method: bool = False
):
    """Extract the chosen letter, or None if the response is not parseable.

    Ordered most-specific first. The loose fallback would happily match the "A"
    in "A person walks by", so it runs last and skips a bare "A" that reads as
    the indefinite article. A None return is a *parse failure*, not a wrong
    answer; the caller decides whether to send it to a judge or score it zero.

    Args:
        return_method: also return which rule fired, so the run log can show how
                       often the loose fallback was needed.
    """
    def _out(letter, method):
        return (letter, method) if return_method else letter

    if response is None:
        return _out(None, "empty")
    text = str(response).strip()
    if not text:
        return _out(None, "empty")

    for i, pat in enumerate(_PATTERNS):
        m = pat.search(text)
        if m:
            return _out(m.group(1).upper(), f"pattern{i}")

    for m in _LOOSE.finditer(text):
        if m.group(1) == "A" and _ARTICLE_A.match(text[m.start():m.start() + 8]):
            continue
        return _out(m.group(1).upper(), "loose")
    return _out(None, "unparsed")


def match_option_text(response: Optional[str], options: Sequence[str]) -> Optional[str]:
    """Fallback: the model restated an option verbatim instead of its letter."""
    if not response:
        return None
    low = str(response).strip().lower()
    hits = []
    for i, o in enumerate(options):
        body = _strip_letter(o).lower().rstrip(".").strip()
        if body and (body in low or low in body):
            hits.append(chr(65 + i))
    return hits[0] if len(hits) == 1 else None


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def aggregate(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by: Dict[str, Dict[str, int]] = {}
    total = correct = unparsed = errors = 0
    keep_num = keep_den = 0

    for r in records:
        code = r.get("task_code") or TASK_CODE.get(r.get("task_type", ""), "?")
        slot = by.setdefault(code, {"total": 0, "correct": 0})
        slot["total"] += 1
        total += 1
        if r.get("is_correct"):
            slot["correct"] += 1
            correct += 1
        if r.get("parse_failed"):
            unparsed += 1
        if r.get("error"):
            errors += 1
        keep_num += int(r.get("visual_tokens_kept", 0) or 0)
        keep_den += int(r.get("visual_tokens_dense", 0) or 0)

    tasks = {}
    for code in PAPER_ORDER + sorted(k for k in by if k not in PAPER_ORDER):
        if code not in by:
            continue
        t, c = by[code]["total"], by[code]["correct"]
        tasks[code] = {"total": t, "correct": c, "acc": 100.0 * c / t if t else 0.0}

    return {
        "tasks": tasks,
        "overall": {
            "total": total,
            "correct": correct,
            "acc": 100.0 * correct / total if total else 0.0,
        },
        "token_keep_rate_pct": 100.0 * keep_num / keep_den if keep_den else 0.0,
        "visual_tokens_kept": keep_num,
        "visual_tokens_dense": keep_den,
        "unparsed": unparsed,
        "errors": errors,
    }


def format_table(summary: Dict[str, Any], label: str = "QueryStream (ours)") -> str:
    codes = [c for c in PAPER_ORDER if c in summary["tasks"]]
    head = f"{'Model':<30}{'Keep':>7}" + "".join(f"{c:>7}" for c in codes) + f"{'All':>8}"
    lines = [head, "-" * len(head)]

    row = f"{label:<30}{summary['token_keep_rate_pct']:>6.1f}%"
    row += "".join(f"{summary['tasks'][c]['acc']:>7.2f}" for c in codes)
    row += f"{summary['overall']['acc']:>8.2f}"
    lines.append(row)

    for name, ref in PAPER_REFERENCE.items():
        if not all(c in ref for c in codes):
            continue
        r = f"{'  paper: ' + name:<30}{'':>7}"
        r += "".join(f"{ref[c]:>7.2f}" for c in codes)
        r += f"{ref['All']:>8.2f}"
        lines.append(r)

    lines.append("")
    lines.append(
        f"  n={summary['overall']['total']}  correct={summary['overall']['correct']}"
        f"  unparsed={summary['unparsed']}  errors={summary['errors']}"
    )
    lines.append(
        f"  token keep rate = {summary['visual_tokens_kept']}/"
        f"{summary['visual_tokens_dense']} = {summary['token_keep_rate_pct']:.2f}%"
    )
    return "\n".join(lines)


__all__ = [
    "TASK_CODE", "PAPER_ORDER", "PAPER_REFERENCE", "PROMPT_TEMPLATE",
    "Question", "VideoItem",
    "load_streamingbench", "timestamp_to_seconds", "format_options",
    "resolve_video_path", "parse_mcq", "match_option_text",
    "aggregate", "format_table",
]
