"""LLM-as-a-judge over the OpenAI **Batch API**.

WHY BATCH, AND WHAT IT IS FOR
StreamingBench is 4-way multiple choice and the official metric is exact-match
on the letter, so the judge is not the scorer -- it is an independent second
opinion used to (a) recover answers the regex could not parse and (b) quantify
how much the reported accuracy depends on the parser. Judging every evaluated
question with the synchronous API would be thousands of serial round-trips; the
Batch API takes one upload, runs them server-side at 50% cost, and returns one
file.

THE JUDGE NEVER SEES THE GROUND TRUTH. It is asked only to map a free-text reply
onto one of the four options. Scoring against the answer key happens locally.
Showing it the correct letter would let it rationalise agreement and would make
"judge accuracy" a measure of the judge's compliance, not of the model.

TIMING CAVEAT: `completion_window="24h"` is the SLA, not the typical latency.
Small batches usually return in minutes, but nothing guarantees it, so submit
and poll are separate steps and the run's exact-match numbers never depend on
the batch landing.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

MAP_PROMPT = (
    "A video question-answering model was shown a video and asked a "
    "multiple-choice question. Your job is ONLY to determine which option the "
    "model's reply corresponds to. Do not judge whether it is correct.\n\n"
    "Question: {question}\n\n"
    "Options:\n{options}\n\n"
    "Model reply: {response}\n\n"
    "Answer with exactly one character: A, B, C, or D. "
    "If the reply matches no option, answer X."
)


def _client(api_key: Optional[str] = None):
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it in the shell; this module "
            "never reads a key from, or writes one to, any file."
        )
    from openai import OpenAI

    return OpenAI(api_key=key)


def build_requests(
    records: Sequence[Dict[str, Any]],
    *,
    model: str = "gpt-4o-mini",
    max_tokens: int = 4,
) -> List[Dict[str, Any]]:
    """One Batch API request per evaluated question."""
    from .streamingbench import format_options

    out: List[Dict[str, Any]] = []
    for r in records:
        if r.get("error"):                       # nothing was generated
            continue
        prompt = MAP_PROMPT.format(
            question=r.get("question", ""),
            options=format_options(r.get("options", [])),
            response=(r.get("predicted_response") or "")[:2000],
        )
        out.append({
            "custom_id": r["question_id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": max_tokens,
                "temperature": 0.0,
            },
        })
    return out


def submit(
    requests: Sequence[Dict[str, Any]],
    work_dir: str,
    *,
    api_key: Optional[str] = None,
    completion_window: str = "24h",
) -> Dict[str, Any]:
    """Upload the request file and create the batch. Returns batch metadata."""
    os.makedirs(work_dir, exist_ok=True)
    req_path = os.path.join(work_dir, "batch_requests.jsonl")
    with open(req_path, "w") as fh:
        for r in requests:
            fh.write(json.dumps(r) + "\n")

    client = _client(api_key)
    with open(req_path, "rb") as fh:
        up = client.files.create(file=fh, purpose="batch")
    batch = client.batches.create(
        input_file_id=up.id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
        metadata={"description": "QueryStream StreamingBench MCQ mapping"},
    )
    meta = {"batch_id": batch.id, "input_file_id": up.id,
            "status": batch.status, "n_requests": len(requests),
            "submitted_at": time.time()}
    with open(os.path.join(work_dir, "batch_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def poll(
    batch_id: str,
    *,
    api_key: Optional[str] = None,
    timeout_s: float = 600.0,
    interval_s: float = 15.0,
    verbose: bool = True,
) -> Any:
    """Wait for the batch, returning the final batch object.

    Returns as soon as the batch leaves the in-progress states, or when
    `timeout_s` elapses -- the caller decides whether an unfinished batch is a
    problem. It usually is not: the exact-match result stands on its own.
    """
    client = _client(api_key)
    t0 = time.time()
    terminal = {"completed", "failed", "expired", "cancelled"}
    while True:
        b = client.batches.retrieve(batch_id)
        counts = getattr(b, "request_counts", None)
        if verbose:
            done = getattr(counts, "completed", 0) if counts else 0
            tot = getattr(counts, "total", 0) if counts else 0
            print(f"  [batch] {b.status:<12} {done}/{tot}  "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)
        if b.status in terminal or (time.time() - t0) > timeout_s:
            return b
        time.sleep(interval_s)


def fetch(batch: Any, work_dir: str, *, api_key: Optional[str] = None) -> Dict[str, str]:
    """Download the output file and return {question_id: letter-or-X}."""
    if not getattr(batch, "output_file_id", None):
        return {}
    client = _client(api_key)
    content = client.files.content(batch.output_file_id).text
    raw_path = os.path.join(work_dir, "batch_output.jsonl")
    with open(raw_path, "w") as fh:
        fh.write(content)

    letters: Dict[str, str] = {}
    for line in content.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            body = row["response"]["body"]
            text = (body["choices"][0]["message"]["content"] or "").strip().upper()
        except Exception:
            continue
        m = re.search(r"[A-DX]", text)
        if m:
            letters[row["custom_id"]] = m.group(0)
    return letters


def apply_to_records(
    records: Sequence[Dict[str, Any]], letters: Dict[str, str]
) -> Dict[str, Any]:
    """Score the judge's mapping against the answer key, in place.

    Adds `judge_letter` / `judge_is_correct` to every record the judge covered,
    and reports how often the judge and the regex parser disagree -- that
    disagreement rate is the real measure of how parser-sensitive the headline
    number is.
    """
    n = agree = judge_correct = exact_correct = recovered = 0
    disagreements: List[Dict[str, Any]] = []

    for r in records:
        letter = letters.get(r["question_id"])
        if letter is None:
            continue
        n += 1
        jl = None if letter == "X" else letter
        r["judge_letter"] = jl
        r["judge_is_correct"] = bool(jl is not None and jl == r["ground_truth"])
        judge_correct += r["judge_is_correct"]
        exact_correct += bool(r.get("is_correct"))
        if jl == r.get("predicted_letter"):
            agree += 1
        else:
            disagreements.append({
                "question_id": r["question_id"],
                "response": (r.get("predicted_response") or "")[:120],
                "parser": r.get("predicted_letter"),
                "judge": jl,
                "ground_truth": r["ground_truth"],
            })
        if r.get("parse_failed") and jl is not None:
            recovered += 1

    return {
        "judged": n,
        "judge_acc": 100.0 * judge_correct / n if n else 0.0,
        "exact_acc_on_judged": 100.0 * exact_correct / n if n else 0.0,
        "parser_judge_agreement_pct": 100.0 * agree / n if n else 0.0,
        "recovered_parse_failures": recovered,
        "disagreements": disagreements[:50],
        "n_disagreements": len(disagreements),
    }


__all__ = ["build_requests", "submit", "poll", "fetch", "apply_to_records", "MAP_PROMPT"]
