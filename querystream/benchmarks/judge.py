"""Optional LLM-as-a-judge fallback for unparseable MCQ responses.

StreamingBench is 4-way multiple choice and the prompt asks for a bare letter,
so exact-match letter parsing IS the official protocol and is what the reported
accuracy must be based on. The judge exists for one narrow job: when
`parse_mcq` returns None -- the model wrote prose instead of a letter -- decide
which option that prose meant, rather than scoring it zero by default.

It is therefore a *recovery* path, not the scorer. It is off unless
`--judge openai` is passed, every judged item is flagged `judged=True` in the
JSONL, and the run summary reports the judged count so its influence is visible
and auditable. If the judged fraction is not small, the harness has a prompting
problem the judge is papering over.

The API key is read from the OPENAI_API_KEY environment variable and is never
written to a config file, a log line or the JSONL.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

JUDGE_PROMPT = (
    "A video question-answering model was asked a multiple-choice question but "
    "did not reply with a bare option letter. Decide which option its reply "
    "corresponds to.\n\n"
    "Question: {question}\n\n"
    "Options:\n{options}\n\n"
    "Model reply: {response}\n\n"
    "Reply with exactly one character: A, B, C, D, or X if the reply does not "
    "correspond to any option."
)


@dataclass
class JudgeResult:
    letter: Optional[str]
    ok: bool
    error: str = ""


class OpenAIJudge:
    """Thin wrapper over the OpenAI chat API, used only on parse failures."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it in the shell (or the SLURM "
                "job env) -- the harness deliberately does not read a key from "
                "any file it might write."
            )
        from openai import OpenAI

        self.client = OpenAI(api_key=key, timeout=timeout, max_retries=max_retries)
        self.model = model
        self.n_calls = 0
        self.n_failed = 0

    def __call__(
        self, question: str, options: Sequence[str], response: str
    ) -> JudgeResult:
        from .streamingbench import format_options

        prompt = JUDGE_PROMPT.format(
            question=question,
            options=format_options(options),
            response=(response or "")[:2000],
        )
        self.n_calls += 1
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=4,
                temperature=0.0,
            )
            text = (r.choices[0].message.content or "").strip().upper()
        except Exception as exc:                       # network / quota / model
            self.n_failed += 1
            return JudgeResult(None, False, f"{type(exc).__name__}: {exc}")

        m = re.search(r"[A-DX]", text)
        if not m or m.group(0) == "X":
            return JudgeResult(None, True)
        return JudgeResult(m.group(0), True)

    def stats(self) -> dict:
        return {"judge_calls": self.n_calls, "judge_api_failures": self.n_failed}


def build_judge(kind: str, model: str = "gpt-4o-mini") -> Optional[OpenAIJudge]:
    if kind in ("none", "", None):
        return None
    if kind == "openai":
        return OpenAIJudge(model=model)
    raise ValueError(f"unknown judge {kind!r} (expected 'none' or 'openai')")


__all__ = ["OpenAIJudge", "JudgeResult", "build_judge", "JUDGE_PROMPT"]
