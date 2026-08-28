"""Backbone prompts for OmniPro Online mode.

Two separate texts exist per sample and must not be confused:

  * `Sample.query_text()` (dataset.py) -> OpenCLIP, for QDP's `M_sem` and RTAR's
    `R_t`. Truncated at 77 BPE tokens, so it carries only the instruction and
    the trigger phrase.
  * `build_prompt()` (here) -> Qwen2.5-VL, at each RTAR-selected timestamp.
    Carries the full instruction plus the output-format constraint that makes
    the response machine-scorable.

OmniPro Sec. 3.2.1: "For most sub-tasks, correctness is verified via exact match
on structured outputs (e.g., integer count, YES/NO). For open-ended generation
tasks (i.e., Event-Narr. and Step-Inst.) ... we employ [an] LLM judge."

So the format constraint is not our invention -- it is what the benchmark scores
against. Getting it wrong costs content accuracy on tasks where the model may
well have perceived the right thing, which is why each template below is written
to match its extractor in content.py exactly, and why the two files name each
other.

WHAT IS DELIBERATELY NOT IN THESE PROMPTS
No ground-truth vocabulary. State-Monitor is scored by exact string equality
against `state_to`, drawn from an open vocabulary of 611 distinct values across
the benchmark, and it would be trivial -- and dishonest -- to paste the
candidate states into the prompt. The model is told the shape of the answer,
never its content.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:      # pragma: no cover
    from .dataset import Sample

# The nine grid cells Target-Ground is scored against, hyphenated exactly as the
# ground truth spells them (content.py::extract_position scans for these forms).
POSITION_VOCAB = (
    "top-left, top-center, top-right, "
    "center-left, center, center-right, "
    "bottom-left, bottom-center, bottom-right"
)

_SYSTEM = (
    "You are a proactive streaming video assistant. You are watching a live "
    "video stream and have just decided that this is the moment to speak. "
    "Answer about what has happened up to now only -- you cannot see the future."
)

# Per-task output constraint. The key is the task; the value is appended after
# the user's standing instruction.
_FORMAT = {
    # ---- time_only: content is not scored, so ask for a short natural alert.
    "instant_event_alert":
        "The event you were asked to watch for has just occurred. In one short "
        "sentence, alert the user to it.",
    "semantic_condition_alert":
        "The condition you were asked to watch for is now satisfied. In one "
        "short sentence, alert the user and say briefly why.",

    # ---- count: extractor takes the first integer after clock times are stripped.
    "snapshot_counting":
        "Reply with ONLY a single integer: the number of designated targets "
        "visible in the current frame. No words, no punctuation.",
    "cumulative_counting":
        "Reply with ONLY a single integer: the running total of occurrences "
        "counted so far, including the one that just happened. No words.",
    "dedup_counting":
        "Reply with ONLY a single integer: how many DISTINCT targets have "
        "appeared so far in total, counting each one only once. No words.",

    # ---- position: extractor scans for a hyphenated grid label.
    "explicit_target_grounding":
        "Reply with ONLY the grid cell where the target is, chosen from exactly "
        f"this list: {POSITION_VOCAB}. Output the label alone.",

    # ---- state: extractor lowercases the WHOLE reply and compares to state_to.
    "realtime_state_monitor":
        "The monitored state has just changed. Reply with ONLY the NEW state, "
        "as a short lowercase word or noun phrase. No sentence, no punctuation, "
        "no mention of the previous state.",

    # ---- gpt_judge: free text, scored 1-5 by the LLM judge, >=3 correct.
    "event_narration":
        "Narrate what just happened in one or two sentences, as live "
        "commentary. Be specific about who and what.",
    "sequential_step_instruction":
        "State the single next step the user should take right now, in one "
        "or two sentences. Be concrete and actionable.",
}


def system_prompt() -> str:
    return _SYSTEM


def build_prompt(sample: "Sample", t_sec: float) -> str:
    """The full instruction handed to the backbone at response time `t_sec`.

    The timestamp is stated explicitly. Qwen2.5-VL receives M-ROPE temporal
    coordinates for the surviving tokens, but those are grid indices, not
    seconds, and the counting tasks in particular are much easier to answer
    coherently when the model knows how far into the stream it is.
    """
    fmt = _FORMAT.get(sample.task, "Answer in one short sentence.")
    mm, ss = divmod(int(t_sec), 60)
    return (
        f"Standing instruction from the user (given at the start of the video): "
        f"{sample.question.strip()}\n\n"
        f"Current stream time: {mm:02d}:{ss:02d}.\n"
        f"{fmt}"
    )


def max_new_tokens_for(task: str) -> int:
    """Generation budget per task.

    Structured tasks need a handful of tokens; letting them ramble only creates
    extra text for the extractors to trip over, and decode time is charged per
    token. The two judged tasks get room for two sentences.
    """
    kind_budget = {
        "snapshot_counting": 8, "cumulative_counting": 8, "dedup_counting": 8,
        "explicit_target_grounding": 12,
        "realtime_state_monitor": 16,
        "instant_event_alert": 40, "semantic_condition_alert": 48,
        "event_narration": 80, "sequential_step_instruction": 80,
    }
    return kind_budget.get(task, 48)
