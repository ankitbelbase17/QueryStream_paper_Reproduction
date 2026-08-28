#!/usr/bin/env python3
"""Correctness anchors for the OmniPro scorer. No pytest, no GPU, no network.

    python tests/test_omnipro_metrics.py [-v]

These are the checks that stop a metric bug from producing a plausible number.
The dangerous failures here are not crashes -- they are formulas that are
*almost* right and quietly flatter the result, so each test names the wrong
formula it rules out.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omnipro_eval import content as C
from omnipro_eval import dataset as ds
from omnipro_eval import judge as J
from omnipro_eval import metrics as M

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


# --------------------------------------------------------------------------- #
# temporal matching
# --------------------------------------------------------------------------- #

@case
def test_matching_is_one_to_one():
    """Two responses inside one trigger's window must not both count.

    Without the used-set, a burst of responses around a single trigger would
    each score a true positive and precision would be unbounded above.
    """
    m, ue, ug = M.match_emits_to_gt([10.0, 10.5], [10.2], tolerance=3.0)
    assert len(m) == 1 and len(ue) == 1 and ug == []
    assert m[0][0] == 0, "closest-first should take the 10.0 pair (dt 0.2)"


@case
def test_matching_is_closest_first_not_first_come():
    """Greedy by |dt|, as the benchmark specifies -- not by emission order."""
    m, _, _ = M.match_emits_to_gt([12.0, 10.1], [10.0, 12.0], tolerance=3.0)
    pairs = {(i, j) for i, j, _ in m}
    assert pairs == {(0, 1), (1, 0)}, pairs


@case
def test_tolerance_boundary_is_inclusive():
    """|dt| == tolerance matches; anything beyond does not."""
    assert len(M.match_emits_to_gt([13.0], [10.0], tolerance=3.0)[0]) == 1
    assert len(M.match_emits_to_gt([13.001], [10.0], tolerance=3.0)[0]) == 0


@case
def test_no_emits_gives_all_false_negatives():
    m, ue, ug = M.match_emits_to_gt([], [1.0, 2.0], tolerance=3.0)
    assert m == [] and ue == [] and ug == [0, 1]


# --------------------------------------------------------------------------- #
# the metric formulas
# --------------------------------------------------------------------------- #

@case
def test_time_and_joint_formulas():
    """Hand-computed reference for every headline quantity.

    tp_time=6 tp_content=3 fp=4 fn=2  ->  n_emits=10, n_gt=8
        time_P = 6/10 = .60   time_R = 6/8 = .75   time_F1 = .666...
        content_acc = 3/6 = .50
        joint_P = 3/10 = .30  joint_R = 3/8 = .375  joint_F1 = .333...
    """
    b = M.block({"tp_time": 6, "tp_content": 3, "fp": 4, "fn": 2,
                 "n_unjudged": 0, "n": 5})
    approx(b["time_precision"], 0.6, 1e-4)
    approx(b["time_recall"], 0.75, 1e-4)
    approx(b["time_f1"], 0.6667, 1e-4)
    approx(b["content_acc"], 0.5, 1e-4)
    approx(b["joint_precision"], 0.3, 1e-4)
    approx(b["joint_recall"], 0.375, 1e-4)
    approx(b["joint_f1"], 0.3333, 1e-4)


@case
def test_joint_precision_denominator_is_all_responses():
    """THE inflation bug: joint_P must be tp_content / n_emits.

    Using tp_content / (tp_content + fp) drops the 3 responses that matched in
    time but failed on content, instead of counting them as false positives.
    Here that would give 3/7 = .4286 instead of the correct 3/10 = .30 -- a
    43% overstatement, largest exactly when content accuracy is worst.
    """
    b = M.block({"tp_time": 6, "tp_content": 3, "fp": 4, "fn": 2,
                 "n_unjudged": 0, "n": 5})
    approx(b["joint_precision"], 3 / 10, 1e-6)
    assert abs(b["joint_precision"] - 3 / 7) > 0.1, "wrong denominator in use"


@case
def test_joint_is_not_time_times_content():
    """Guards against anyone 'simplifying' joint_F1 to a product."""
    b = M.block({"tp_time": 6, "tp_content": 3, "fp": 4, "fn": 2,
                 "n_unjudged": 0, "n": 5})
    assert abs(b["joint_f1"] - b["time_f1"] * b["content_acc"]) > 1e-6


@case
def test_perfect_and_empty_runs():
    b = M.block({"tp_time": 5, "tp_content": 5, "fp": 0, "fn": 0,
                 "n_unjudged": 0, "n": 3})
    approx(b["time_f1"], 1.0)
    approx(b["joint_f1"], 1.0)
    e = M.block({"n": 0})
    approx(e["time_f1"], 0.0)
    approx(e["joint_f1"], 0.0)


@case
def test_unjudged_withholds_rather_than_zeroes():
    """An unjudged match must not be silently counted wrong.

    The lower bound stays available under a name that says what it is, and
    content_coverage reports how much of the metric is actually measured.
    """
    b = M.block({"tp_time": 4, "tp_content": 2, "fp": 1, "fn": 1,
                 "n_unjudged": 2, "n": 2})
    assert b["content_acc"] is None and b["joint_f1"] is None
    assert b["content_complete"] is False
    approx(b["content_acc_lb"], 0.5, 1e-4)          # 2/4, worst case
    approx(b["content_coverage"], 0.5, 1e-4)        # 2 of 4 matches judged


@case
def test_macro_excludes_withheld_tasks_and_reports_the_count():
    per_task = {"a": {"joint_f1": 0.4}, "b": {"joint_f1": None}, "c": {"joint_f1": 0.6}}
    v, n = M.macro_over_tasks(per_task, "joint_f1")
    approx(v, 0.5, 1e-9)
    assert n == 2, "a partial macro average must report how many tasks it covers"


# --------------------------------------------------------------------------- #
# per-sample scoring
# --------------------------------------------------------------------------- #

def _rec(task, emits, gts, **kw):
    r = {"sample_id": f"{task}::v::0", "task": task, "audio_dependency": "none",
         "question": "q", "emits": emits, "ground_truth": gts}
    r.update(kw)
    return r


@case
def test_time_only_tasks_ignore_content_entirely():
    """The two alert tasks score a time match correct whatever the text says."""
    r = _rec("instant_event_alert",
             [{"t_sec": 10.0, "text": "complete nonsense"}],
             [{"t_sec": 10.5, "response": "the bell rang"}])
    s = M.score_sample(r)
    assert s.tp_time == 1 and s.tp_content == 1 and s.n_unjudged == 0


@case
def test_count_task_exact_match_and_unparseable_is_wrong():
    gts = [{"t_sec": 10.0, "count": 3, "response": "three"}]
    ok = M.score_sample(_rec("snapshot_counting", [{"t_sec": 10.0, "text": "3"}], gts))
    assert ok.tp_content == 1
    bad = M.score_sample(_rec("snapshot_counting",
                              [{"t_sec": 10.0, "text": "several"}], gts))
    assert bad.tp_time == 1 and bad.tp_content == 0, \
        "an unparseable count is WRONG, not unscorable"


@case
def test_clock_strip_before_integer_extraction():
    """'At 01:23, 5 people' must yield 5, not 1 (from the '01')."""
    gts = [{"t_sec": 10.0, "count": 5, "response": "five"}]
    s = M.score_sample(_rec("snapshot_counting",
                            [{"t_sec": 10.0, "text": "At 01:23, 5 people"}], gts))
    assert s.tp_content == 1


@case
def test_position_matching_follows_upstream_strictness():
    assert C.extract_position("bottom-center") == "bottom-center"
    assert C.extract_position("Position: center-right") == "center-right"
    # Upstream scans only hyphenated labels: "bottom center" hits the bare
    # "center" and scores as a DIFFERENT cell. Accepting it would inflate.
    assert C.extract_position("bottom center") == "center"
    assert C.extract_position("top left") is None      # no bare "left" cell
    # Longest-first ordering: "center-right" must not be swallowed by "center".
    assert C.extract_position("it is at the center-right") == "center-right"


@case
def test_state_is_whole_reply_equality_not_substring():
    gt = {"state_to": "playing"}
    assert C.structured_correct("state", "playing", gt)
    assert C.structured_correct("state", "Playing.", gt)     # trailing period only
    assert not C.structured_correct(
        "state", "the background music is now playing loudly", gt), \
        "a substring search would inflate State-Monitor against the benchmark"


@case
def test_judged_task_without_a_judge_is_unjudged():
    r = _rec("event_narration", [{"t_sec": 10.0, "text": "something happened"}],
             [{"t_sec": 10.0, "response": "the ferry departed"}])
    s = M.score_sample(r, judge=None)
    assert s.tp_time == 1 and s.n_unjudged == 1 and s.tp_content == 0


@case
def test_cached_judge_is_bound_per_emit_index():
    """Verdicts are keyed by (sample_id, emit_idx), so emit 1 must not be scored
    with emit 0's verdict."""
    r = _rec("event_narration",
             [{"t_sec": 5.0, "text": "wrong"}, {"t_sec": 20.0, "text": "right"}],
             [{"t_sec": 5.0, "response": "a"}, {"t_sec": 20.0, "response": "b"}])
    verdicts = {J.custom_id(r["sample_id"], 0): {"score": 1, "explanation": ""},
                J.custom_id(r["sample_id"], 1): {"score": 5, "explanation": ""}}
    s = M.score_sample(r, judge=J.CachedJudge(verdicts))
    assert s.tp_time == 2 and s.tp_content == 1 and s.n_unjudged == 0
    ok = {m["emit_idx"]: m["content_ok"] for m in s.matches}
    assert ok == {0: False, 1: True}, ok


@case
def test_judge_pass_threshold_is_three():
    r = _rec("event_narration", [{"t_sec": 5.0, "text": "x"}],
             [{"t_sec": 5.0, "response": "y"}])
    for score, want in ((2, 0), (3, 1), (5, 1)):
        v = {J.custom_id(r["sample_id"], 0): {"score": score, "explanation": ""}}
        assert M.score_sample(r, judge=J.CachedJudge(v)).tp_content == want


# --------------------------------------------------------------------------- #
# slices and weighting
# --------------------------------------------------------------------------- #

@case
def test_headline_slices_partition_the_audio_axis():
    """required + (helpful|none) must cover every sample exactly once."""
    sl = {s.name: s for s in ds.build_slices()}
    for a in ds.AUDIO_CLASSES:
        r = {"task": "event_narration", "audio_dependency": a}
        hits = sum(bool(sl[n].predicate(r))
                   for n in ("audio_required", "audio_helpful_none"))
        assert hits == 1, f"{a} matched {hits} headline audio slices"


@case
def test_event_narration_required_slice_is_the_intersection():
    sl = {s.name: s for s in ds.build_slices()}
    p = sl["event_narration_required"].predicate
    assert p({"task": "event_narration", "audio_dependency": "required"})
    assert not p({"task": "event_narration", "audio_dependency": "helpful"})
    assert not p({"task": "dedup_counting", "audio_dependency": "required"})


@case
def test_benchmark_weights_match_the_published_composition():
    """1768 / 500 / 432 out of 2700, per benchmark.json and the paper's 84%."""
    w = ds.FULL_BENCHMARK_AUDIO_SHARE
    approx(sum(w.values()), 1.0, 1e-9)
    approx(w["required"] + w["helpful"], 0.84, 0.005)   # the paper's headline
    approx(w["required"], 1768 / 2700, 1e-9)


@case
def test_weighted_joint_uses_benchmark_not_subset_proportions():
    """A subset over-representing the easy slice must not lift the headline.

    Slices: none=0.60 (n=100), helpful=0.40 (n=100), required=0.00 (n=10).
    Subset-weighted -> (60+40+0)/210 = .476. Benchmark-weighted ->
    .16*.60 + .1852*.40 + .6548*0 = .1701. The gap IS the correction.
    """
    blocks = {
        "audio_none": {"macro_joint_f1": 0.60, "n_samples": 100},
        "audio_helpful": {"macro_joint_f1": 0.40, "n_samples": 100},
        "audio_required": {"macro_joint_f1": 0.00, "n_samples": 10},
    }
    w = M.weighted_joint(blocks)
    approx(w["subset_weighted"], round((0.60 * 100 + 0.40 * 100) / 210, 4), 1e-4)
    approx(w["benchmark_weighted"], 0.1701, 1e-3)
    approx(w["equal_weighted"], round((0.60 + 0.40 + 0.0) / 3, 4), 1e-4)
    assert w["benchmark_weighted"] < w["subset_weighted"]


@case
def test_modality_block_is_recall_side_only():
    """A false positive carries no modality, so precision is undefined there."""
    r = _rec("dedup_counting",
             [{"t_sec": 10.0, "text": "2"}, {"t_sec": 90.0, "text": "9"}],
             [{"t_sec": 10.0, "count": 2, "trigger_type": "visual"},
              {"t_sec": 50.0, "count": 5, "trigger_type": "visual"},
              {"t_sec": 60.0, "count": 6, "trigger_type": "speech"}])
    b = M.modality_block([M.score_sample(r)], "visual")
    assert b["n_gt"] == 2 and b["n_matched"] == 1
    approx(b["time_recall"], 0.5, 1e-9)
    assert "precision" not in " ".join(k for k in b if k != "note")


# --------------------------------------------------------------------------- #
# taxonomy integrity
# --------------------------------------------------------------------------- #

@case
def test_taxonomy_is_complete_and_consistent():
    assert len(ds.ALL_TASKS) == 9
    assert set(ds.PAPER_ORDER) == set(ds.ALL_TASKS)
    for t in ds.ALL_TASKS:
        assert t in ds.TASK_CONTENT_KIND and t in ds.SHORT_NAME and t in ds.LEVEL
        assert t in ds.PAPER_ONLINE_VIDEO_ONLY
        assert t in ds.PAPER_ONLINE["MiniCPM-o-4.5-9B"]
    assert ds.JUDGE_TASKS == {"event_narration", "sequential_step_instruction"}
    assert ds.TIME_ONLY_TASKS == {"instant_event_alert", "semantic_condition_alert"}


@case
def test_published_means_match_their_own_rows():
    """Transcription check on the reference tables we compare against."""
    for name, row in ds.PAPER_ONLINE.items():
        vals = [row[t] for t in ds.ALL_TASKS]
        approx(sum(vals) / 9, row["Mean"], 0.06)
    v = ds.PAPER_ONLINE_VIDEO_ONLY
    approx(sum(v[t] for t in ds.ALL_TASKS) / 9, v["Mean"], 0.06)


@case
def test_custom_id_round_trips_through_a_sample_id_containing_colons():
    sid = "event_narration::Q8FK2CtxfpY::12"
    assert J.split_custom_id(J.custom_id(sid, 7)) == (sid, 7)


def main() -> int:
    verbose = "-v" in sys.argv
    npass = nfail = 0
    for fn in CASES:
        try:
            fn()
            npass += 1
            if verbose:
                print(f"  ok   {fn.__name__}")
        except Exception as exc:
            nfail += 1
            print(f"  FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
            if verbose:
                import traceback
                traceback.print_exc()
    print(f"\n{npass}/{npass + nfail} passed")
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(main())
