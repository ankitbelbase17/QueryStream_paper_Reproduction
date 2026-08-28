#!/usr/bin/env python3
"""Correctness anchors for the OVO-Bench scorer and manifest. No GPU, no network.

    python tests/test_ovo_scoring.py [-v]

The rules under test are OVO-Bench's own (`utils/OVOBenchScore.py`), including
the ones that look wrong. Each test that pins an odd rule says why it is pinned:
matching the benchmark matters more than matching intuition, because Table 2 was
produced by these rules.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ovo_eval import dataset as ds
from ovo_eval import prompts as pr
from ovo_eval import scoring as sc

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


@case
def test_letter_rule_is_a_substring_test():
    """OVO-Bench scores `int(gt in response)`. Bare letter and prose both pass."""
    assert sc.score_br("A", "A") == 1
    assert sc.score_br("The answer is C.", "C") == 1
    assert sc.score_br("B", "A") == 0
    assert sc.score_br(None, "A") == 0
    # The known looseness: any reply containing the letter scores. Pinned so a
    # future "tightening" is a deliberate, visible change and not a silent one.
    assert sc.score_br("A photo of a dog", "A") == 1


@case
def test_strict_letter_diagnostic_is_stricter_than_the_official_rule():
    assert sc.strict_letter("A") == "A"
    assert sc.strict_letter(" (B) ") == "B"
    assert sc.strict_letter("C.") == "C"
    assert sc.strict_letter("A photo of a dog") is None
    assert sc.strict_letter("") is None


@case
def test_rec_concatenates_digits():
    """Upstream does re.findall(r'\\d+') then ''.join -- '1 or 2' becomes '12'."""
    assert sc.score_rec("3", "3") == 1
    assert sc.score_rec("The count is 3", "3") == 1
    assert sc.score_rec("1 or 2", "12") == 1          # the documented quirk
    assert sc.score_rec("1 or 2", "1") == 0
    assert sc.score_rec("three", "3") == 0            # words are not numbers
    assert sc.score_rec(None, "3") == 0


@case
def test_yesno_accepts_the_shorthands_then_substring():
    assert sc.score_yesno("Y", "Yes") == 1
    assert sc.score_yesno("N", "No") == 1
    assert sc.score_yesno("Yes", "Yes") == 1
    assert sc.score_yesno("No", "No") == 1
    assert sc.score_yesno("No", "Yes") == 0
    # "Yes" is a substring of nothing else here, but "No" IS a substring of
    # "Not sure" -- upstream would score that as No. Pinned deliberately.
    assert sc.score_yesno("Not sure", "No") == 1
    assert sc.score_yesno(None, "Yes") == 0


@case
def test_score_call_dispatches_by_task():
    assert sc.score_call("OCR", "A", "A") == 1
    assert sc.score_call("REC", "5", "5") == 1
    assert sc.score_call("SSR", "Yes", "Yes") == 1
    assert sc.score_call("CRR", "No", "No") == 1


@case
def test_category_average_is_macro_over_tasks_not_micro():
    """A task with many calls must not dominate its category.

    OCR 100% on 1 call and ACR 0% on 99 calls is a 50.0 category average under
    the official macro rule, and 1.0 under a micro pool. Getting this wrong
    would silently reweight every published comparison.
    """
    recs = [{"task": "OCR", "response": "A", "ground_truth": "A"}]
    recs += [{"task": "ACR", "response": "B", "ground_truth": "A"} for _ in range(99)]
    agg = sc.aggregate(recs)
    assert agg["per_task"]["OCR"]["acc"] == 100.0
    assert agg["per_task"]["ACR"]["acc"] == 0.0
    assert agg["category"]["realtime"] == 50.0, agg["category"]


@case
def test_overall_is_withheld_unless_all_three_categories_present():
    recs = [{"task": "OCR", "response": "A", "ground_truth": "A"}]
    agg = sc.aggregate(recs)
    assert agg["overall"] is None and agg["overall_is_partial"] is True
    recs += [{"task": "EPM", "response": "A", "ground_truth": "A"},
             {"task": "REC", "response": "1", "ground_truth": "1"}]
    agg = sc.aggregate(recs)
    assert agg["overall"] == 100.0 and agg["overall_is_partial"] is False


@case
def test_errored_rows_are_excluded_not_zeroed():
    ok = [{"task": "OCR", "response": "A", "ground_truth": "A"}]
    bad = ok + [{"task": "OCR", "response": None, "ground_truth": "A",
                 "error": "decode failed"}]
    assert sc.aggregate(ok)["per_task"]["OCR"]["acc"] == 100.0
    assert sc.aggregate(bad)["per_task"]["OCR"]["acc"] == 100.0, \
        "an infrastructure error is not a wrong answer"


@case
def test_null_response_without_error_scores_zero():
    """A model that returned nothing IS wrong, unlike a crashed call."""
    recs = [{"task": "OCR", "response": None, "ground_truth": "A"}]
    agg = sc.aggregate(recs)
    assert agg["per_task"]["OCR"]["acc"] == 0.0
    assert agg["n_null_responses"] == 1


@case
def test_paper_table2_rows_reconstruct_their_printed_averages():
    """Transcription check on the numbers we compare against."""
    for name, want in (("QueryStream-7B (52.9%)", (63.1, 44.9, 40.2, 49.4)),
                       ("TimeChat-Online-7B (100%)", (61.9, 41.7, 36.7, 46.7)),
                       ("TimeChat-Online-7B (55.4%)", (61.4, 44.5, 36.8, 47.6)),
                       ("Human Agents", (93.2, 92.3, 92.9, 92.8))):
        row = ds.PAPER[name]
        got = (row["realtime"], row["backward"], row["forward"], row["overall"])
        for g, w in zip(got, want):
            assert abs(g - w) <= 0.06, f"{name}: {got} vs {want}"


@case
def test_taxonomy_is_complete():
    assert len(ds.ALL_TASKS) == 12
    assert set(ds.REAL_TIME_TASKS) | set(ds.BACKWARD_TASKS) | set(ds.FORWARD_TASKS) \
        == set(ds.ALL_TASKS)
    for t in ds.ALL_TASKS:
        assert t in ds.CATEGORY and t in ds.TASK_NAME


@case
def test_prompts_are_the_official_templates():
    """Guards against a well-meaning cleanup of the upstream wording."""
    assert "Respond only with the letter corresponding to your chosen option" \
        in pr.BR_PROMPT_TEMPLATE
    assert "repetively" in pr.REC_PROMPT_TEMPLATE      # upstream typo, kept
    assert "You're task is" in pr.REC_PROMPT_TEMPLATE  # upstream typo, kept
    assert "sequential of steps" in pr.SSR_PROMPT_TEMPLATE
    assert pr.format_options(["x", "y"]) == "A. x; B. y;"


@case
def test_query_text_strips_boilerplate_for_openclip():
    """OpenCLIP truncates at 77 tokens; the shared instruction tail must not
    crowd out the actual question."""
    c = ds.Call(call_id="1", sample_id=1, task="OCR", category="realtime",
                video_path="", realtime=10.0, probe_index=None, prompt_kind="br",
                question="What is written?", options=["cat", "dog"],
                ground_truth="A")
    qt = pr.query_text(c)
    assert "What is written?" in qt and "cat" in qt
    assert "Respond only with the letter" not in qt
    assert "Respond only with the letter" in pr.build_prompt(c)


@case
def test_manifest_ground_truth_encoding():
    """Letters for MCQ, str(count) for REC, Yes/No from `type` for SSR/CRR."""
    calls, st = ds.load_manifest()
    assert st.dropped_missing_video == 0, "every clip should be on disk"
    by = {}
    for c in calls:
        by.setdefault(c.task, c)
    assert by["OCR"].ground_truth in "ABCD"
    assert by["REC"].ground_truth.isdigit()
    for t in ("SSR", "CRR"):
        assert by[t].ground_truth in ("Yes", "No")
        expect = "Yes" if by[t].probe_type == 1 else "No"
        assert by[t].ground_truth == expect


@case
def test_subset_keeps_a_samples_probes_together():
    calls, _ = ds.load_manifest()
    sub, info = ds.subset(calls, 0.2, seed=0)
    whole = {c.sample_id for c in calls}
    chosen = {c.sample_id for c in sub}
    by_all, by_sub = {}, {}
    for c in calls:
        by_all.setdefault(c.sample_id, set()).add(c.call_id)
    for c in sub:
        by_sub.setdefault(c.sample_id, set()).add(c.call_id)
    for sid in chosen:
        assert by_sub[sid] == by_all[sid], f"sample {sid} was split across the subset"
    assert chosen <= whole
    for t in ds.ALL_TASKS:
        assert t in info["samples_per_task"], f"{t} missing from the subset"


@case
def test_subset_is_stable_under_task_filtering():
    """A task's sample draw must not depend on which OTHER tasks are present.

    With one RNG consumed sequentially across tasks, `--tasks OCR` and a full
    12-task run pick DIFFERENT OCR samples at the same seed, so a per-task
    control cannot be paired with a full-run baseline. Measured before the fix:
    a 448-vs-768 resolution control shared only 18 of 89 samples with its
    baseline, which made the comparison meaningless.
    """
    calls, _ = ds.load_manifest()
    full, _ = ds.subset(calls, 0.2, seed=0)
    three = [c for c in calls if c.task in ("OCR", "ATR", "STU")]
    filtered, _ = ds.subset(three, 0.2, seed=0)

    for t in ("OCR", "ATR", "STU"):
        a = {c.call_id for c in full if c.task == t}
        b = {c.call_id for c in filtered if c.task == t}
        assert a == b, (f"{t}: filtering changed the draw "
                        f"({len(a & b)} of {len(a)} shared)")


@case
def test_subset_is_reproducible_and_seed_sensitive():
    calls, _ = ds.load_manifest()
    a, _ = ds.subset(calls, 0.2, seed=0)
    b, _ = ds.subset(calls, 0.2, seed=0)
    c, _ = ds.subset(calls, 0.2, seed=1)
    assert [x.call_id for x in a] == [x.call_id for x in b], "not reproducible"
    assert {x.call_id for x in a} != {x.call_id for x in c}, "seed had no effect"


def main() -> int:
    verbose = "-v" in sys.argv
    p = f = 0
    for fn in CASES:
        try:
            fn(); p += 1
            if verbose:
                print(f"  ok   {fn.__name__}")
        except Exception as exc:
            f += 1
            print(f"  FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
            if verbose:
                import traceback; traceback.print_exc()
    print(f"\n{p}/{p+f} passed")
    return 1 if f else 0


if __name__ == "__main__":
    raise SystemExit(main())
