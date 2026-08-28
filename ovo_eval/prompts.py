"""OVO-Bench prompt templates, transcribed verbatim from the benchmark's source.

These come from `OVO-Bench/constant.py` and `utils/OVOBench.py::build_prompt`
and are reproduced **unchanged**, including the wording quirks
("repetively", "You're task is"). The prompts are part of the measuring
instrument: the official scorer substring-matches an option letter, or a bare
"Yes"/"No", or digits, and every one of those depends on the model having been
asked in exactly this way. Rewriting them to be cleaner would silently make our
numbers non-comparable with Table 2.

Only the ACTIVE templates are here. `constant.py` also carries two commented-out
earlier variants of the REC and SSR prompts; those are not what the published
numbers used, so they are not reproduced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:      # pragma: no cover
    from .dataset import Call

BR_PROMPT_TEMPLATE = """
Question: {}
Options:
{}

Respond only with the letter corresponding to your chosen option (e.g., A, B, C). 
Do not include any additional text or explanation in your response.
"""

REC_PROMPT_TEMPLATE = """
You're watching a video in which people may perform a certain type of action repetively. 
The person performing this kind of action are referred to as 'they' in the following statement.
You're task is to count how many times have different people in the video perform this kind of action in total.
One complete motion counts as one. 
Now, answer the following question: {}
Provide your answer as a single number (e.g., 0, 1, 2, 3…) indicating the total count.
Do not include any additional text or explanation in your response.
"""

SSR_PROMPT_TEMPLATE = """
You're watching a tutorial video which contain a sequential of steps. 
The following is one step from the whole procedures: 
{}
Your task is to determine if the man or woman in the video is currently performing this step.
Answer only with “Yes” or “No”.
Do not include any additional text or explanation in your response.
"""

CRR_PROMPT_TEMPLATE = """
You're responsible of answering questions based on the video content. 
The following question are relevant to the latest frames, i.e. the end of the video.
{}
Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.
Answer only with “Yes” or “No”.
Do not include any additional text or explanation in your response.
"""


def format_options(options) -> str:
    """`A. opt; B. opt; ...;` -- the exact join used by build_prompt."""
    return '; '.join(f'{chr(65 + i)}. {o}' for i, o in enumerate(options)) + ';'


def build_prompt(c: "Call") -> str:
    """The prompt handed to the backbone for one Call."""
    if c.prompt_kind == "br":
        return BR_PROMPT_TEMPLATE.format(c.question, format_options(c.options))
    if c.prompt_kind == "REC":
        # build_prompt synthesises the question from `activity`; the annotation
        # has no `question` field for REC.
        return REC_PROMPT_TEMPLATE.format(f"How many times did they {c.activity}?")
    if c.prompt_kind == "SSR":
        return SSR_PROMPT_TEMPLATE.format(c.step)
    if c.prompt_kind == "CRR":
        return CRR_PROMPT_TEMPLATE.format(c.question)
    raise ValueError(f"unknown prompt_kind {c.prompt_kind!r}")


def query_text(c: "Call") -> str:
    """Text handed to OpenCLIP for QDP's `M_sem` and RTAR's `R_t`.

    OpenCLIP truncates at 77 BPE tokens, so the instruction boilerplate above
    would push the actual content out of the window entirely -- every
    single-timestamp prompt shares the same 30-token tail. This is the semantic
    payload only, the same rule used on StreamingBench and OmniPro. The backbone
    still receives the complete official prompt.
    """
    if c.prompt_kind == "br":
        return (c.question + " " + " ".join(c.options)).strip()
    if c.prompt_kind == "REC":
        return f"How many times did they {c.activity}?"
    if c.prompt_kind == "SSR":
        return c.step
    return c.question


def max_new_tokens_for(kind: str) -> int:
    """The official prompts all demand a bare token, so the budget is tiny.

    A letter, a "Yes"/"No", or an integer. Anything longer is the model failing
    to follow the format, which the scorer already penalises; giving it room to
    ramble only creates text for the substring match to trip over.
    """
    return 8
