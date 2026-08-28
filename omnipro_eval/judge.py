"""LLM-as-a-judge for OmniPro's two open-ended sub-tasks, over the Batch API.

WHICH RESPONSES ARE JUDGED, AND WHY ONLY THOSE
OmniPro Sec. 3.2.1: "For most sub-tasks, correctness is verified via exact match
on structured outputs ... For open-ended generation tasks (i.e., Event-Narr. and
Step-Inst.) where output cannot be constrained into a fixed format, we employ
Gemini-3-Flash as an LLM judge to score each prediction against the ground truth
on a 1-5 scale; a score >= 3 is considered correct."

So exactly two tasks reach the judge, and only their *time-matched* responses do
-- a response that matched no ground-truth trigger has no reference text to be
scored against and is already a false positive. That keeps the judged set small:
roughly (matched responses on 2 of 9 tasks), not (every response).

THE JUDGE SEES THE GROUND TRUTH HERE, AND SHOULD
This is the opposite policy from the StreamingBench judge in
`querystream/benchmarks/batch_judge.py`, deliberately. There the judge maps free
text onto one of four options and must not see the key, or it would rationalise
agreement. Here the benchmark's own metric IS "how well does this response match
the reference", so the reference is the input. What the judge never sees is any
other model's answer, the sample's other responses, or the timing verdict.

WE SUBSTITUTE OPENAI FOR THE PAPER'S GEMINI-3-FLASH
The rubric, the 1-5 scale and the >=3 threshold are unchanged, but a different
judge model is a different measuring instrument, and judged scores from
different judges are not strictly comparable. This is recorded in the run
metadata and stated in the report rather than quietly assumed away. Structured
Outputs with `strict=True` makes the response schema-guaranteed, so
"unparseable judge output" is not a failure mode.

NO FALLBACK JUDGE. If a verdict is missing the match is UNJUDGED: it counts as
neither correct nor incorrect, and `content_acc` / `joint_f1` are withheld in
favour of explicit lower bounds (see metrics.py). A lexical-overlap fallback
would silently turn "we could not measure this" into a number.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence

from .dataset import JUDGE_TASKS

JUDGE_PROMPT = (
    "You are evaluating whether a streaming video model's proactive response "
    "correctly describes the event it was supposed to report, compared to the "
    "ground-truth response.\n\n"
    "User's standing instruction: {question}\n"
    "Ground-truth response: {gt}\n"
    "Model response: {pred}\n\n"
    "Score 1-5:\n"
    "- 5: Perfect - same event/information, accurate\n"
    "- 4: Good - mostly correct, minor differences\n"
    "- 3: Acceptable - right event, notable inaccuracies\n"
    "- 2: Poor - partially relevant but significantly wrong or vague\n"
    "- 1: Wrong - irrelevant or wrong event\n"
)

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "explanation": {"type": "string"},
    },
    "required": ["score", "explanation"],
    "additionalProperties": False,
}

PASS_SCORE = 3          # paper: >= 3 counts as correct


def _client(api_key: Optional[str] = None):
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it in the shell; this module "
            "never reads a key from, or writes one to, any file."
        )
    from openai import OpenAI

    return OpenAI(api_key=key)


def judge_body(model: str, question: str, gt: str, pred: str,
               seed: Optional[int] = 1234) -> Dict[str, Any]:
    """The chat.completions body, shared by the sync and Batch paths.

    Shared on purpose: a batch verdict and a sync verdict for the same triple
    must be the same request, or the two paths quietly measure different things.
    """
    body: Dict[str, Any] = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                question=(question or "")[:1500],
                gt=(gt or "")[:1500],
                pred=(pred or "")[:1500] or "(the model said nothing)"),
        }],
        "max_completion_tokens": 200,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "verdict", "strict": True,
                            "schema": JUDGE_SCHEMA},
        },
    }
    if seed is not None:
        body["seed"] = seed
    return body


def custom_id(sample_id: str, emit_idx: int) -> str:
    """Batch custom_id. Sample ids contain '::' already, so '#' separates the
    emit index unambiguously and survives the round trip."""
    return f"{sample_id}#{emit_idx}"


def split_custom_id(cid: str) -> tuple[str, int]:
    sid, _, idx = cid.rpartition("#")
    return sid, int(idx)


# --------------------------------------------------------------------------- #
# request building
# --------------------------------------------------------------------------- #

def build_requests(
    records: Sequence[Dict[str, Any]],
    scores_by_id: Dict[str, Any],
    *,
    model: str = "gpt-4o-mini",
) -> List[Dict[str, Any]]:
    """One request per (judged task, time-matched response) pair.

    `scores_by_id` maps sample_id -> SampleScore, so only responses that
    actually matched a ground-truth trigger are sent -- which is both what the
    metric needs and what keeps the batch small.
    """
    out: List[Dict[str, Any]] = []
    for r in records:
        if r.get("task") not in JUDGE_TASKS or r.get("error"):
            continue
        sc = scores_by_id.get(r["sample_id"])
        if sc is None:
            continue
        emits = r.get("emits", [])
        gts = r.get("ground_truth", [])
        for m in sc.matches:
            ei, gj = m["emit_idx"], m["gt_idx"]
            if ei >= len(emits) or gj >= len(gts):
                continue
            out.append({
                "custom_id": custom_id(r["sample_id"], ei),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": judge_body(model, r.get("question", ""),
                                   gts[gj].get("response", ""),
                                   emits[ei].get("text", "")),
            })
    return out


# --------------------------------------------------------------------------- #
# Batch API
# --------------------------------------------------------------------------- #

def submit(requests: Sequence[Dict[str, Any]], work_dir: str, *,
           api_key: Optional[str] = None,
           completion_window: str = "24h") -> Dict[str, Any]:
    os.makedirs(work_dir, exist_ok=True)
    req_path = os.path.join(work_dir, "judge_requests.jsonl")
    with open(req_path, "w") as fh:
        for r in requests:
            fh.write(json.dumps(r) + "\n")

    client = _client(api_key)
    with open(req_path, "rb") as fh:
        up = client.files.create(file=fh, purpose="batch")
    batch = client.batches.create(
        input_file_id=up.id, endpoint="/v1/chat/completions",
        completion_window=completion_window,
        metadata={"description": "QueryStream OmniPro online content judge"})
    meta = {"batch_id": batch.id, "input_file_id": up.id, "status": batch.status,
            "n_requests": len(requests), "submitted_at": time.time(),
            "model": requests[0]["body"]["model"] if requests else None}
    with open(os.path.join(work_dir, "judge_batch_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def poll(batch_id: str, *, api_key: Optional[str] = None,
         timeout_s: float = 900.0, interval_s: float = 20.0,
         verbose: bool = True) -> Any:
    """Wait for the batch, returning the final batch object.

    `completion_window="24h"` is the SLA, not the typical latency: small batches
    often return in minutes but nothing guarantees it. Returns on any terminal
    status or on timeout -- an unfinished batch is not an error, because every
    timing metric and the seven exact-match tasks stand without it.
    """
    client = _client(api_key)
    t0 = time.time()
    terminal = {"completed", "failed", "expired", "cancelled"}
    while True:
        b = client.batches.retrieve(batch_id)
        c = getattr(b, "request_counts", None)
        if verbose:
            print(f"  [judge] {b.status:<12} "
                  f"{getattr(c, 'completed', 0)}/{getattr(c, 'total', 0)}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if b.status in terminal or (time.time() - t0) > timeout_s:
            return b
        time.sleep(interval_s)


def fetch(batch: Any, work_dir: str, *, api_key: Optional[str] = None
          ) -> Dict[str, Dict[str, Any]]:
    """Download results; return {custom_id: {"score": int, "explanation": str}}."""
    if not getattr(batch, "output_file_id", None):
        return {}
    client = _client(api_key)
    content = client.files.content(batch.output_file_id).text
    with open(os.path.join(work_dir, "judge_output.jsonl"), "w") as fh:
        fh.write(content)

    verdicts: Dict[str, Dict[str, Any]] = {}
    for line in content.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            txt = row["response"]["body"]["choices"][0]["message"]["content"]
            obj = json.loads(txt)
            verdicts[row["custom_id"]] = {
                "score": int(obj["score"]),
                "explanation": str(obj.get("explanation", ""))[:400],
            }
        except Exception:
            continue      # a malformed row is UNJUDGED, never a default verdict
    return verdicts


# --------------------------------------------------------------------------- #
# applying verdicts
# --------------------------------------------------------------------------- #

class CachedJudge:
    """A `JudgeFn` backed by a verdict table, for re-scoring without the API.

    Keyed by (sample_id, emit_idx), which the scorer supplies through
    `bind(sample_id)`. Returns None -- UNJUDGED -- for anything the table does
    not cover, which is what keeps a partially-returned batch from being
    silently treated as a set of failures.
    """

    def __init__(self, verdicts: Dict[str, Dict[str, Any]],
                 *, pass_score: int = PASS_SCORE):
        self.verdicts = verdicts
        self.pass_score = pass_score
        self._sample_id: Optional[str] = None
        self._emit_idx: Optional[int] = None
        self.n_hit = 0
        self.n_miss = 0

    def bind(self, sample_id: str, emit_idx: int) -> None:
        self._sample_id, self._emit_idx = sample_id, emit_idx

    def __call__(self, question: str, gt: str, pred: str) -> Optional[bool]:
        if self._sample_id is None:
            return None
        v = self.verdicts.get(custom_id(self._sample_id, self._emit_idx))
        if v is None:
            self.n_miss += 1
            return None
        self.n_hit += 1
        return v["score"] >= self.pass_score


def attach_verdicts(records: Sequence[Dict[str, Any]],
                    verdicts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Write each verdict onto its emit so the JSONL is self-contained."""
    n = 0
    dist: Dict[int, int] = {}
    for r in records:
        if r.get("task") not in JUDGE_TASKS:
            continue
        for i, e in enumerate(r.get("emits", [])):
            v = verdicts.get(custom_id(r["sample_id"], i))
            if v is None:
                continue
            e["judge_score"] = v["score"]
            e["judge_explanation"] = v["explanation"]
            e["judge_correct"] = v["score"] >= PASS_SCORE
            dist[v["score"]] = dist.get(v["score"], 0) + 1
            n += 1
    return {"attached": n, "score_distribution": {str(k): dist[k] for k in sorted(dist)},
            "pass_score": PASS_SCORE,
            "pass_rate": round(sum(c for s, c in dist.items() if s >= PASS_SCORE) / n, 4)
            if n else None}
