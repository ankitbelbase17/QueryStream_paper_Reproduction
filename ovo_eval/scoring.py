"""OVO-Bench scoring, transcribed from `utils/OVOBenchScore.py`.

THE RULES ARE THE BENCHMARK'S, NOT OURS. Each is reproduced exactly, including
the parts that look too lenient or too strict, because the published Table 2
numbers were produced by these rules and changing them would make our figures
incomparable.

    single-timestamp tasks (9)   score = int(ground_truth_letter in response)
                                 A bare substring test on "A".."D".
    REC                          digits are extracted from the reply, joined,
                                 and compared to str(count) as a string.
    SSR / CRR                    the literal shorthands "Y"/"N" are accepted
                                 first, then int("Yes"/"No" in response).

    per-task accuracy -> category average = MACRO mean over that category's
    tasks -> Overall = mean of the three category averages.

Two consequences of the official rules worth stating, because they look like
bugs and are not:

1. **The letter test is a substring test.** A reply of "A" scores, and so does
   "The answer is A." -- but so does "A photo of ..." if the ground truth is A.
   We report `strict_letter_acc` alongside as a diagnostic, but the headline
   uses the official rule.
2. **REC concatenates digits.** "1 or 2" becomes "12", not 1. This is upstream
   behaviour (`re.findall(r'\\d+')` then `"".join`), and it penalises hedged
   answers hard.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from .dataset import ALL_TASKS, CATEGORY, CATEGORY_ORDER

_DIGITS = re.compile(r"\d+")
_STRICT_LETTER = re.compile(r"^\s*\(?([A-D])\)?\s*[.):,]?\s*$")


def score_br(response: Optional[str], gt_letter: str) -> int:
    """`int(gt in response)` -- OVO-Bench's rule for the 9 MCQ tasks."""
    if response is None:
        return 0
    return int(gt_letter in response)


def score_rec(response: Optional[str], gt_count: str) -> int:
    """Digits extracted, concatenated, compared as a string."""
    if response is None:
        return 0
    return int("".join(_DIGITS.findall(response)) == str(gt_count))


def score_yesno(response: Optional[str], gt: str) -> int:
    """SSR / CRR: the "Y"/"N" shorthands first, then a substring test."""
    if response is None:
        return 0
    if response == "N" and gt == "No":
        return 1
    if response == "Y" and gt == "Yes":
        return 1
    return int(gt in response)


def score_call(task: str, response: Optional[str], ground_truth: str) -> int:
    if task == "REC":
        return score_rec(response, ground_truth)
    if task in ("SSR", "CRR"):
        return score_yesno(response, ground_truth)
    return score_br(response, ground_truth)


def strict_letter(response: Optional[str]) -> Optional[str]:
    """A reply that is ONLY a letter, for the parser-sensitivity diagnostic."""
    if not response:
        return None
    m = _STRICT_LETTER.match(response.strip())
    return m.group(1) if m else None


def aggregate(records: Sequence[dict]) -> dict:
    """Per-task accuracy, category macro averages, and the Overall mean.

    `records` are the JSONL rows: each carries `task`, `response`,
    `ground_truth` and (already computed, but recomputed here so the scorer is
    a pure function of the raw reply) nothing else.
    """
    per_task_hits: Dict[str, List[int]] = defaultdict(list)
    strict_hits: Dict[str, List[int]] = defaultdict(list)
    n_null = 0

    for r in records:
        if r.get("error"):
            continue
        task = r["task"]
        resp = r.get("response")
        if resp is None:
            n_null += 1
        s = score_call(task, resp, r["ground_truth"])
        per_task_hits[task].append(s)
        if CATEGORY[task] != "forward":
            sl = strict_letter(resp)
            strict_hits[task].append(int(sl is not None and sl == r["ground_truth"]))

    per_task = {
        t: {
            "n": len(v),
            "correct": int(sum(v)),
            "acc": round(100 * sum(v) / len(v), 2) if v else None,
        }
        for t, v in per_task_hits.items()
    }
    for t, v in strict_hits.items():
        if v:
            per_task[t]["strict_letter_acc"] = round(100 * sum(v) / len(v), 2)

    cats: Dict[str, Optional[float]] = {}
    cat_tasks: Dict[str, List[str]] = {}
    for c in CATEGORY_ORDER:
        ts = [t for t in ALL_TASKS if CATEGORY[t] == c and t in per_task]
        cat_tasks[c] = ts
        cats[c] = round(sum(per_task[t]["acc"] for t in ts) / len(ts), 2) if ts else None

    present = [c for c in CATEGORY_ORDER if cats[c] is not None]
    overall = (round(sum(cats[c] for c in present) / len(present), 2)
               if len(present) == 3 else None)

    return {
        "per_task": {t: per_task[t] for t in ALL_TASKS if t in per_task},
        "category": cats,
        "category_tasks": cat_tasks,
        "overall": overall,
        "overall_is_partial": len(present) < 3,
        "n_calls": sum(len(v) for v in per_task_hits.values()),
        "n_null_responses": n_null,
        "note": ("Overall is the mean of the three category averages, each a "
                 "macro mean over its tasks -- OVO-Bench's own aggregation. It "
                 "is withheld (None) unless all three categories are present, "
                 "because a two-category mean is not the published quantity."),
    }
