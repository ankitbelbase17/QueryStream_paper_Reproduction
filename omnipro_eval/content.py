"""Deterministic content extraction for OmniPro's structured sub-tasks.

Three of the five content kinds are decided offline with no model in the loop:
`count`, `position`, `state`. Their rules are transcribed from the OmniPro
reference online parser so our numbers sit on the same scale as the published
baselines. Each rule below records what the strict form is and what a more
lenient variant would have cost, because every leniency here inflates content
accuracy against the benchmark's own scorer.

An emit the extractor cannot parse is scored WRONG, not unscorable: producing
an unparseable count is a failure to follow the required output format, and
following the format is part of what Online mode tests.
"""

from __future__ import annotations

import re
from typing import Optional

# Longest-first so compound cells are tested before the bare "center" they
# contain. In alphabetical order "center-right" and "bottom-center" both match
# "center" first and score as the wrong cell -- and both occur in the ground
# truth, so two of the nine cells would be unwinnable.
POSITIONS = sorted(
    ["top-left", "top-center", "top-right",
     "center-left", "center", "center-right",
     "bottom-left", "bottom-center", "bottom-right"],
    key=len, reverse=True,
)

_CLOCK = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_INT = re.compile(r"-?\d+")
_POS_ANCHOR = re.compile(r"position\s*:\s*([\w\-]+)")


def extract_count(text: str) -> Optional[int]:
    """First integer in the reply, AFTER clock times are stripped.

    The strip is load-bearing: a reply opening "At 01:23, 5 people" yields 1
    (from "01") without it, marking a correct answer wrong. Only affects the
    three counting tasks.
    """
    cleaned = _CLOCK.sub(" ", text or "")
    m = _INT.search(cleaned)
    return int(m.group(0)) if m else None


def extract_position(text: str) -> Optional[str]:
    """An explicit `Position: <cell>` anchor if present, else a longest-match scan.

    Only HYPHENATED labels are accepted, which is the reference behaviour. Also
    accepting the space-separated form ("bottom center") is more lenient than
    the benchmark's own scorer -- there it matches the bare "center" and scores
    as a different cell -- so it is deliberately not accepted here.
    """
    t = (text or "").lower()
    m = _POS_ANCHOR.search(t)
    if m and m.group(1).strip() in POSITIONS:
        return m.group(1).strip()
    for p in POSITIONS:                     # already longest-first
        if p in t:
            return p
    return None


def extract_state(text: str) -> Optional[str]:
    """The WHOLE reply, lowercased and unquoted, trailing punctuation removed.

    The reference rule compares the entire payload to `state_to` by exact
    equality. A substring search ("does the reply mention the target state
    anywhere?") is far more lenient -- a rambling reply that happens to contain
    the word scores correct -- and would inflate State-Monitor against the
    benchmark's scorer, so it is not used.

    The one liberty taken beyond the reference is stripping a trailing period or
    comma, since our prompt asks for a bare noun phrase and a model that obeys
    it but adds a full stop has not made a content error.
    """
    s = (text or "").strip().strip("'\"").strip()
    s = s.rstrip(".,;:!").strip()
    return s.lower() or None


def structured_correct(kind: str, reply: str, gt_item: dict) -> bool:
    """Exact-match verdict for the non-judged content kinds."""
    if not (reply or "").strip():
        return False

    if kind == "count":
        pc = extract_count(reply)
        gc = gt_item.get("count")
        if pc is None or gc is None:
            return False
        return int(pc) == int(gc)

    if kind == "position":
        pp = extract_position(reply)
        gp = (gt_item.get("position") or "").lower().strip()
        return bool(pp and gp and pp == gp)

    if kind == "state":
        ps = extract_state(reply)
        gs = (gt_item.get("state_to") or gt_item.get("state") or "").lower().strip()
        return bool(ps and gs and ps == gs)

    raise ValueError(f"structured_correct called with kind={kind!r}")
