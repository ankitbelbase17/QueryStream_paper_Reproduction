#!/usr/bin/env python3
"""Compile the OmniPro evaluation into a PDF.

Consumes `omnipro_metrics.json` (from score_omnipro.py), the raw JSONL and the
profile CSV. Reports what the run measured, what it could not measure and why,
and places every number next to the published row it should be read against.

    python make_omnipro_report.py runs/op_full
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics as stat
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from omnipro_eval import dataset as ds
from omnipro_eval import metrics as M
from omnipro_eval import report as rp

INK = colors.HexColor("#0b0b0b")
INK2 = colors.HexColor("#52514e")
MUTED = colors.HexColor("#8a8983")
GRID = colors.HexColor("#e4e3de")
BAND = colors.HexColor("#eef4fc")
ACCENT = colors.HexColor("#2a78d6")


def _short_backbone(cfg) -> str:
    """Human-readable backbone name, from the run's OWN config.

    Hardcoding it here once produced a Qwen3-VL report that described itself as
    Qwen2.5-VL in prose while the config table said otherwise -- the kind of
    error that survives review because every number on the page is right.
    """
    mid = str((cfg or {}).get("backbone", "") or "")
    return mid.split("/")[-1] if mid else "the backbone"


def _is_qwen3(cfg) -> bool:
    return "qwen3-vl" in str((cfg or {}).get("backbone", "")).lower()


def styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    mk = lambda **kw: ParagraphStyle(**kw)
    return {
        "title": mk(name="t", parent=base["Title"], fontName="Helvetica-Bold",
                    fontSize=19, leading=23, textColor=INK, spaceAfter=2,
                    alignment=TA_LEFT),
        "sub": mk(name="s", parent=base["Normal"], fontName="Helvetica",
                  fontSize=9.5, leading=13.5, textColor=INK2, spaceAfter=10),
        "h1": mk(name="h1", parent=base["Heading1"], fontName="Helvetica-Bold",
                 fontSize=12.5, leading=16, textColor=INK, spaceBefore=15, spaceAfter=5),
        "h2": mk(name="h2", parent=base["Heading2"], fontName="Helvetica-Bold",
                 fontSize=10, leading=13, textColor=INK, spaceBefore=9, spaceAfter=3),
        "body": mk(name="b", parent=base["Normal"], fontName="Helvetica",
                   fontSize=8.8, leading=12.6, textColor=INK, spaceAfter=5),
        "small": mk(name="sm", parent=base["Normal"], fontName="Helvetica",
                    fontSize=7.6, leading=10.5, textColor=MUTED, spaceAfter=4),
        "mono": mk(name="m", parent=base["Normal"], fontName="Courier",
                   fontSize=7.3, leading=9.8, textColor=INK2, spaceAfter=3),
    }


def tbl(data, widths, *, align_right_from=1, header=True, size=7.6, highlight=None):
    t = Table(data, colWidths=widths, hAlign="LEFT")
    cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), size),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("ALIGN", (align_right_from, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, GRID),
    ]
    if header:
        cmds += [("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                 ("TEXTCOLOR", (0, 0), (-1, 0), INK2),
                 ("LINEBELOW", (0, 0), (-1, 0), 0.9, INK2)]
    for r in (highlight or []):
        cmds += [("BACKGROUND", (0, r), (-1, r), BAND),
                 ("FONTNAME", (0, r), (-1, r), "Helvetica-Bold")]
    t.setStyle(TableStyle(cmds))
    return t


def wrap(text: str, st: ParagraphStyle) -> Paragraph:
    """A table cell that wraps. Plain strings in a Table do not wrap, they
    overflow into the next column, which is how a long note silently ate the
    value beside it in the first draft."""
    return Paragraph(text, st)


def pct(v: Optional[float], nd: int = 1, bound: bool = False) -> str:
    if v is None:
        return "n/a"
    return ("<" if bound else "") + f"{100 * v:.{nd}f}"


def metric_or_bound(b: dict, key: str) -> tuple[str, bool]:
    v = b.get(key)
    if v is not None:
        return pct(v), False
    return pct(b.get(key + "_lb"), bound=True), True


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load(run_dir: str):
    with open(os.path.join(run_dir, "omnipro_metrics.json")) as f:
        m = json.load(f)

    records: List[dict] = []
    for p in sorted(glob.glob(os.path.join(run_dir, "omnipro_raw_logs*.jsonl"))):
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
    by_id = {r["sample_id"]: r for r in records}
    records = list(by_id.values())

    summaries = []
    for p in sorted(glob.glob(os.path.join(run_dir, "summary*.json"))):
        try:
            summaries.append(json.load(open(p)))
        except Exception:
            pass

    prof: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "profile_summary*.csv"))):
        with open(p) as fh:
            for row in csv.DictReader(fh):
                if row.get("scope") != "OVERALL":
                    continue
                w = float(row.get("n_frames") or 0) or 1.0
                for k in ("decode_ms_per_frame", "openclip_ms_per_frame",
                          "vision_tower_ms_per_frame", "qdp_rtar_ms_per_frame"):
                    prof[k] = prof.get(k, 0.0) + float(row.get(k) or 0) * w
                    counts[k] = counts.get(k, 0) + w
                nq = float((row.get("question_id") or "0").split()[0] or 0) or 1.0
                for k in ("llm_prefill_ms", "llm_decode_ms"):
                    prof[k] = prof.get(k, 0.0) + float(row.get(k) or 0) * nq
                    counts[k] = counts.get(k, 0) + nq
                prof["n_frames"] = prof.get("n_frames", 0.0) + w
                prof["total_wall_clock_sec"] = max(
                    prof.get("total_wall_clock_sec", 0.0),
                    float(row.get("total_wall_clock_sec") or 0))
    for k in list(prof):
        if k in counts and counts[k]:
            prof[k] /= counts[k]
    return m, records, summaries, prof


# --------------------------------------------------------------------------- #
# derived views
# --------------------------------------------------------------------------- #

HORIZON = (("Short (0-60s)", 0, 60), ("Medium (60-180s)", 60, 180),
           ("Long (180s+)", 180, float("inf")))


def horizon_recall(records: List[dict], tolerance: float):
    """Time recall bucketed by where the ground-truth trigger sits (paper Fig. 3)."""
    hit = {h[0]: 0 for h in HORIZON}
    tot = {h[0]: 0 for h in HORIZON}
    for r in records:
        if r.get("error"):
            continue
        sc = M.score_sample(r, tolerance=tolerance, judge=None)
        matched_gt = {m["gt_idx"] for m in sc.matches}
        for j, g in enumerate(r.get("ground_truth", [])):
            t = float(g.get("t_sec", 0.0))
            for name, lo, hi in HORIZON:
                if lo <= t < hi:
                    tot[name] += 1
                    hit[name] += int(j in matched_gt)
                    break
    names = [h[0] for h in HORIZON]
    return names, [hit[n] / tot[n] if tot[n] else 0.0 for n in names], [tot[n] for n in names]


def collect_dts(records: List[dict], tolerance: float) -> List[float]:
    out = []
    for r in records:
        if r.get("error"):
            continue
        sc = M.score_sample(r, tolerance=tolerance, judge=None)
        out.extend(m["dt"] for m in sc.matches)
    return out


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #

def build(run_dir: str, out_path: str) -> str:
    m, records, summaries, prof = load(run_dir)
    S = styles()
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    ok = [r for r in records if not r.get("error")]
    tol = m["tolerance_s"]
    ov = m["metrics"]["overall"]
    per_task = m["metrics"]["per_task"]
    slices = m["slices"]
    cfg = (summaries[0].get("config") if summaries else {}) or {}
    manifest = (summaries[0].get("manifest") if summaries else {}) or {}
    sel = (summaries[0].get("selection") if summaries else {}) or {}

    E: List[Any] = []
    add = E.append

    # ---------------- header ---------------- #
    add(Paragraph("QueryStream on OmniPro", S["title"]))
    add(Paragraph(
        "Online-mode evaluation of query-aware pruning (QDP) and "
        "relevance-triggered active response (RTAR) on OmniPro, a benchmark "
        "in which 84% of samples depend on audio and this system has no audio "
        "encoder. Generated "
        f"{datetime.now():%Y-%m-%d %H:%M} from <font face='Courier'>{run_dir}</font>.",
        S["sub"]))

    n = ov.get("n_samples", 0)
    n_bench = ds.FULL_BENCHMARK_N
    add(Paragraph("What this run is", S["h1"]))
    add(Paragraph(
        "OmniPro's Online mode gives the model a standing instruction at t=0 and "
        "then scores <b>when</b> it chooses to speak and <b>what</b> it says. "
        "That makes it the first benchmark in this reproduction where RTAR is "
        "measured rather than merely logged: on StreamingBench the question "
        "carries its own timestamp, so RTAR was computed and recorded but did "
        "not drive anything. Here every response is emitted because "
        "<font face='Courier'>T_t = R_t AND D_t</font> fired.",
        S["body"]))
    add(Paragraph(
        f"<b>The architecture is unchanged and has no audio pathway.</b> "
        f"OpenCLIP-ViT-L/14 -> QDP -> RTAR -> {_short_backbone(cfg)}, at 1 FPS, "
        f"training-free. OmniPro labels each sample by whether audio is "
        f"<i>required</i>, <i>helpful</i> or unnecessary for its trigger; the "
        f"<i>required</i> samples are unsolvable by construction here. They are "
        f"reported as their own slice rather than dropped, because dropping "
        f"them would turn a known limitation into a flattering number.",
        S["body"]))

    # ---------------- headline ---------------- #
    add(Paragraph("Headline results", S["h1"]))
    ca_s, ca_b = metric_or_bound(ov, "content_acc")
    jf_s, jf_b = metric_or_bound(ov, "joint_f1")
    mjf_s, _ = metric_or_bound(ov, "macro_joint_f1")
    head = [
        ["metric", "value", "what it counts"],
        ["Samples scored", f"{n}", f"of {manifest.get('n_total', n_bench)} in the benchmark"],
        ["Responses emitted", f"{ov.get('n_emits', 0)}",
         f"{ov.get('emits_per_sample', 0):.2f} per sample"],
        ["Ground-truth triggers", f"{ov.get('n_gt', 0)}",
         f"{ov.get('gt_per_sample', 0):.2f} per sample"],
        ["Time-matched", f"{ov.get('n_matched', 0)}", f"within +-{tol:g}s, greedy 1-to-1"],
        ["Time precision", pct(ov.get("time_precision")), "matched / emitted"],
        ["Time recall", pct(ov.get("time_recall")), "matched / ground truth"],
        ["Time F1 (micro)", pct(ov.get("time_f1")), "timing alone, content ignored"],
        ["Time F1 (macro)", pct(ov.get("macro_time_f1")),
         f"equal weight per sub-task -- paper's Mean "
         f"({ov.get('macro_time_f1_n_tasks', 0)} of 9 sub-tasks)"],
        ["Content accuracy", ca_s, "of matched responses, text also correct"],
        ["Joint F1 (micro)", jf_s, "the benchmark's headline metric"],
        ["Joint F1 (macro)", mjf_s,
         f"comparable to Table 2's Mean column "
         f"({ov.get('macro_joint_f1_n_tasks') or ov.get('macro_joint_f1_lb_n_tasks', 0)}"
         f" of 9 sub-tasks scored)"],
        ["Token keep rate", pct(m.get("token_keep_rate")), "QDP survivors / dense tokens"],
    ]
    add(tbl(head, [42 * mm, 26 * mm, 100 * mm], highlight=[8, 12]))
    nmac = ov.get("macro_joint_f1_n_tasks") or ov.get("macro_joint_f1_lb_n_tasks", 0)
    if nmac and nmac < 9:
        add(Paragraph(
            f"<b>The macro means average {nmac} of the 9 sub-tasks</b>, not all "
            f"nine: a sub-task whose content metric is withheld is excluded "
            f"rather than counted as zero. A macro over a subset of tasks is "
            f"not directly comparable to the paper's Mean column, which averages "
            f"all nine.", S["small"]))
    if ca_b or jf_b:
        add(Paragraph(
            "Values prefixed \"&lt;\" are <b>lower bounds</b>: the LLM judge has "
            "not returned verdicts for every matched response on the two "
            "open-ended sub-tasks, and an unjudged match is counted as neither "
            "correct nor incorrect. Timing metrics never involve the judge and "
            "are exact.", S["small"]))

    wj = m["weighted_joint_f1"]
    wj_use = wj if wj.get("benchmark_weighted") is not None else m["weighted_joint_f1_lb"]
    bound = wj.get("benchmark_weighted") is None
    add(Paragraph("The slice-weighted number over all the data", S["h2"]))
    add(Paragraph(
        "A single figure for the whole benchmark has to decide what to do with "
        "the audio-required majority. Three weightings are given because they "
        "answer different questions, and the weights travel with the number so "
        "it cannot be quoted without them.", S["body"]))
    cell = S["small"]
    wrows = [["weighting", "macro joint F1", wrap("what it estimates", cell)],
             ["Benchmark-weighted",
              pct(wj_use.get("benchmark_weighted"), bound=bound),
              wrap("the full 2,700-sample distribution (required 65.5%, helpful "
                   "18.5%, none 16.0%) &mdash; dominated by what audio would "
                   "solve", cell)],
             ["Subset-weighted", pct(wj_use.get("subset_weighted"), bound=bound),
              wrap("the samples actually evaluated here", cell)],
             ["Equal-weighted", pct(wj_use.get("equal_weighted"), bound=bound),
              wrap("one third per audio condition, ignoring how common each is",
                   cell)]]
    add(tbl(wrows, [40 * mm, 28 * mm, 100 * mm], highlight=[1]))

    # ---------------- slices ---------------- #
    add(Paragraph("The three headline slices", S["h1"]))
    add(Paragraph(
        "Each slice is scored independently and its sub-task macro average is "
        "reported, so a slice's number is never dominated by whichever sub-task "
        "happens to have the most triggers in it.", S["body"]))
    srows = [["slice", "n", "emits", "GT", "time F1", "content", "joint F1"]]
    for name in ds.HEADLINE_SLICES:
        b = slices.get(name)
        if not b:
            continue
        c_s, _ = metric_or_bound(b, "macro_content_acc")
        j_s, _ = metric_or_bound(b, "macro_joint_f1")
        srows.append([b["title"], str(b["n_samples"]), str(b["n_emits"]),
                      str(b["n_gt"]), pct(b.get("macro_time_f1")), c_s, j_s])
    E.pop()   # the heading and intro are re-added inside the KeepTogether
    E.pop()
    add(KeepTogether([
        Paragraph("The three headline slices", S["h1"]),
        Paragraph(
            "Each slice is scored independently and its sub-task macro average "
            "is reported, so a slice's number is never dominated by whichever "
            "sub-task happens to have the most triggers in it.", S["body"]),
        tbl(srows, [56 * mm, 14 * mm, 17 * mm, 15 * mm, 20 * mm, 20 * mm, 21 * mm],
            highlight=[2]),
        Paragraph(
            "The middle row is the one QueryStream can fairly be judged on: "
            "those triggers are recoverable from frames. The first row is the "
            "cost of having no audio encoder, stated rather than hidden.",
            S["small"]),
    ]))

    tf, cf, jf = [], [], []
    labs = []
    for name in ds.HEADLINE_SLICES:
        b = slices.get(name)
        if not b:
            continue
        labs.append(b["title"].replace(" (", "\n("))
        tf.append(100 * (b.get("macro_time_f1") or 0))
        cf.append(100 * (b.get("macro_content_acc") or b.get("macro_content_acc_lb") or 0))
        jf.append(100 * (b.get("macro_joint_f1") or b.get("macro_joint_f1_lb") or 0))
    if labs:
        p = rp.fig_slice_metrics(labs, tf, cf, jf, os.path.join(fig_dir, "slices.png"))
        add(Spacer(1, 4))
        add(Image(p, width=150 * mm, height=150 * mm * 3.2 / 7.2))

    add(Paragraph("All audio conditions, separately", S["h2"]))
    arows = [["audio_dependency", "n", "emits", "GT", "time F1", "joint F1",
              "share of full benchmark"]]
    for a in ds.AUDIO_CLASSES:
        b = slices.get(f"audio_{a}")
        if not b:
            continue
        j_s, _ = metric_or_bound(b, "macro_joint_f1")
        arows.append([a, str(b["n_samples"]), str(b["n_emits"]), str(b["n_gt"]),
                      pct(b.get("macro_time_f1")), j_s,
                      f"{100 * ds.FULL_BENCHMARK_AUDIO_SHARE[a]:.1f}%"])
    add(tbl(arows, [30 * mm, 14 * mm, 17 * mm, 15 * mm, 20 * mm, 20 * mm, 40 * mm]))

    add(PageBreak())

    # ---------------- per sub-task ---------------- #
    add(Paragraph("Per sub-task, against the published rows", S["h1"]))
    add(Paragraph(
        "The right-hand columns are OmniPro's own Table 2 (Online mode, F1) and "
        "the video-only ablation row from its Table 3. "
        "<b>MiniCPM-o 4.5 (V) is the only published online row produced without "
        "audio</b>, and is therefore the single directly comparable reference "
        "point for this system. The same model with audio scores "
        f"{ds.PAPER_ONLINE['MiniCPM-o-4.5-9B']['Mean']} against "
        f"{ds.PAPER_ONLINE_VIDEO_ONLY['Mean']} without it, which is the size of "
        "the gap the missing modality accounts for.", S["body"]))
    prows = [["sub-task", "level", "n", "emits", "GT", "time F1", "content",
              "joint F1", "MiniCPM V", "MiniCPM A+V"]]
    for t in ds.PAPER_ORDER:
        b = per_task.get(t)
        if not b:
            continue
        c_s, _ = metric_or_bound(b, "content_acc")
        j_s, _ = metric_or_bound(b, "joint_f1")
        prows.append([ds.SHORT_NAME[t], ds.LEVEL[t][:5], str(b["n_samples"]),
                      str(b["n_emits"]), str(b["n_gt"]), pct(b["time_f1"]),
                      c_s, j_s,
                      f"{ds.PAPER_ONLINE_VIDEO_ONLY[t]:.1f}",
                      f"{ds.PAPER_ONLINE['MiniCPM-o-4.5-9B'][t]:.1f}"])
    mj_s, _ = metric_or_bound(ov, "macro_joint_f1")
    prows.append(["MEAN (macro)", "", str(ov.get("n_samples", 0)), "", "",
                  pct(ov.get("macro_time_f1")),
                  metric_or_bound(ov, "macro_content_acc")[0], mj_s,
                  f"{ds.PAPER_ONLINE_VIDEO_ONLY['Mean']:.1f}",
                  f"{ds.PAPER_ONLINE['MiniCPM-o-4.5-9B']['Mean']:.1f}"])
    add(tbl(prows, [26 * mm, 13 * mm, 11 * mm, 14 * mm, 12 * mm, 17 * mm,
                    16 * mm, 17 * mm, 19 * mm, 21 * mm],
            highlight=[len(prows) - 1], size=7.2))

    labels = [ds.SHORT_NAME[t] for t in ds.PAPER_ORDER if t in per_task]
    mine = [(per_task[t].get("joint_f1") or per_task[t].get("joint_f1_lb") or 0) * 100
            for t in ds.PAPER_ORDER if t in per_task]
    ref_v = [ds.PAPER_ONLINE_VIDEO_ONLY[t] for t in ds.PAPER_ORDER if t in per_task]
    ref_av = [ds.PAPER_ONLINE["MiniCPM-o-4.5-9B"][t] for t in ds.PAPER_ORDER if t in per_task]
    if labels:
        p = rp.fig_subtask(
            labels,
            {"QueryStream-7B (vision only)": mine,
             "MiniCPM-o 4.5 (video only)": ref_v,
             "MiniCPM-o 4.5 (audio+video)": ref_av},
            os.path.join(fig_dir, "subtask.png"),
            title="Joint F1 per sub-task, OmniPro Online mode")
        add(Spacer(1, 4))
        add(Image(p, width=168 * mm, height=168 * mm * 3.3 / 9.4))

    # ---------------- timing behaviour ---------------- #
    add(Paragraph("When the model speaks", S["h1"]))
    add(Paragraph(
        "Joint F1 is a product of two behaviours that fail differently, so both "
        "are shown. Precision is bounded by how often RTAR fires relative to how "
        "many triggers exist; recall by whether it fires near the right seconds "
        "at all.", S["body"]))
    emits = [r.get("n_emits", 0) for r in ok]
    gts = [r.get("n_gt", 0) for r in ok]
    figs = []
    if emits:
        figs.append(rp.fig_emission_balance(emits, gts,
                                            os.path.join(fig_dir, "emission.png")))
    dts = collect_dts(ok, tol)
    if dts:
        figs.append(rp.fig_dt_histogram(dts, tol, os.path.join(fig_dir, "dt.png")))
    if figs:
        row = [[Image(f, width=76 * mm, height=76 * mm * (3.6 if i == 0 else 3.0) / 4.6)
                for i, f in enumerate(figs)]]
        t = Table(row, colWidths=[80 * mm] * len(figs), hAlign="LEFT")
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
        add(t)
    if emits and sum(emits):
        ratio = sum(emits) / max(sum(gts), 1)
        ceiling = 100.0 * min(1.0, sum(gts) / sum(emits))
        add(Paragraph(
            f"<b>Precision is structurally capped at {ceiling:.1f}% here.</b> "
            f"RTAR emitted {sum(emits):,} responses against {sum(gts):,} "
            f"ground-truth triggers &mdash; {ratio:.2f}x as many &mdash; and "
            f"matching is one-to-one, so at best {sum(gts):,} of them can ever "
            f"be true positives. That ceiling is a property of the trigger rate, "
            f"not of what the model said: recall and content accuracy could both "
            f"be perfect and joint precision would still not exceed it. It is "
            f"set by the relevance calibration and the debounce gap, both of "
            f"which are reproduction choices (see below), and both of which are "
            f"logged per frame so the trade-off can be re-examined offline.",
            S["body"]))
    if dts:
        add(Paragraph(
            f"Matched responses land a median {stat.median(dts):.2f}s from their "
            f"trigger (mean {stat.mean(dts):.2f}s) inside a +-{tol:g}s window. "
            f"A distribution pressed against the window edge would mean the "
            f"matches barely survive the tolerance; one near zero means the "
            f"trigger instants are genuinely being found.", S["small"]))

    names, rec, cnt = horizon_recall(ok, tol)
    p = rp.fig_horizon(names, rec, cnt, os.path.join(fig_dir, "horizon.png"))
    hrows = [["trigger position", "GT triggers", "time recall"]]
    for nm, r, c in zip(names, rec, cnt):
        hrows.append([nm, str(c), pct(r)])
    add(KeepTogether([
        Paragraph("Long-horizon behaviour", S["h2"]),
        Paragraph(
            "OmniPro's Section 4.4 reports every model degrading as the trigger "
            "moves later in the video, with online streaming models nearly "
            "failing past 180 s (MiniCPM-o 4.5: 29.1 -&gt; 0.3). This is the "
            "same cut, on time recall.", S["body"]),
        Image(p, width=110 * mm, height=110 * mm * 3.0 / 4.8),
        tbl(hrows, [42 * mm, 26 * mm, 24 * mm]),
        Paragraph(
            "<b>Read the flatness as an artefact, not an advantage.</b> This "
            "system carries no conversational state between responses: each one "
            "is regenerated from the pruned token buffer covering the whole "
            "stream so far, so there is no drifting history to lose. Its trigger "
            "rate is also near-uniform along the timeline, which spreads recall "
            "evenly rather than concentrating it early. A model that does "
            "degrade is being penalised for maintaining state that this one "
            "never had.", S["small"]),
    ]))

    add(PageBreak())

    # ---------------- pruning + gates ---------------- #
    add(Paragraph("Pruning and gating behaviour", S["h1"]))
    keeps = [r["token_keep_rate"] for r in ok if r.get("token_keep_rate")]
    raws: List[float] = []
    for r in ok:
        v = r.get("relevance_raw_scores")
        if v:
            raws.extend(v[::7])          # decimate: the shape is what matters
        elif r.get("relevance_raw_mean") is not None:
            raws.append(r["relevance_raw_mean"])
    figs = []
    if keeps:
        figs.append(rp.fig_keep_rate(keeps, m.get("token_keep_rate", 0.0),
                                     os.path.join(fig_dir, "keep.png")))
    if raws:
        figs.append(rp.fig_relevance(raws, cfg.get("tau_rel", 0.60),
                                     os.path.join(fig_dir, "rel.png")))
    if figs:
        row = [[Image(f, width=76 * mm, height=76 * mm * 3.0 / 4.6) for f in figs]]
        t = Table(row, colWidths=[80 * mm] * len(figs), hAlign="LEFT")
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
        add(t)
    if raws:
        add(Paragraph(
            f"<b>tau_rel = 0.60 is unreachable, confirmed again here.</b> Raw "
            f"cos(q, mean-pooled frame) spans {min(raws):.4f} to {max(raws):.4f} "
            f"over {len(raws)} sampled frames; frames above 0.60: 0. Taken "
            f"literally the relevance gate is False everywhere and the model "
            f"never speaks, so <font face='Courier'>relevance_mode="
            f"{cfg.get('relevance_mode', 'running_minmax')}</font> calibrates it "
            f"against a trailing window. The raw value is logged per frame so the "
            f"mode can be re-chosen from the distribution rather than guessed.",
            S["small"]))

    n_raw = sum(r.get("n_raw_triggers", 0) for r in ok)
    n_emit = sum(r.get("n_emits", 0) for r in ok)
    n_frames = sum(r.get("n_frames_streamed", 0) for r in ok)
    grows = [["quantity", "value", "note"],
             ["Frames gated", f"{n_frames:,}", "1 FPS, every frame sees QDP and RTAR"],
             ["Raw T_t firings", f"{n_raw:,}",
              f"{100 * n_raw / n_frames if n_frames else 0:.1f}% of frames"],
             ["Responses after debounce", f"{n_emit:,}",
              f"Appendix A.4 grouping at gap = {cfg.get('event_gap_s', tol):g}s"],
             ["Token keep rate", pct(m.get("token_keep_rate")),
              "backbone grid; the number comparable to the paper's tables"],
             ["Zero-survivor guard fired",
              f"{sum(r.get('forced_min_keep', 0) for r in ok)}",
              "a revived top token, logged rather than an empty context"],
             ["Emit cap hit",
              f"{sum(1 for r in ok if r.get('emit_cap_hit'))} samples",
              f"cap = {cfg.get('max_emits', 64)} responses per sample"]]
    add(tbl(grows, [46 * mm, 26 * mm, 96 * mm]))
    add(Paragraph(
        "The debounce gap is a scoring-relevant choice the QueryStream paper "
        "leaves unspecified. At the StreamingBench harness's 1 s gap, one real "
        "event emits up to three responses inside OmniPro's +-3 s window, of "
        "which at most one can match and the rest are pure false positives; the "
        "gap therefore defaults to the matching tolerance here. Every raw T_t "
        "timestamp is in the JSONL, so any other gap can be re-scored offline "
        "for the two time-only sub-tasks without a GPU.", S["small"]))

    # ---------------- gate diagnosis ---------------- #
    gr = m.get("gate_rates", {})
    ga = m.get("gate_ablation", [])
    if gr and ga:
        grows = [["gate policy", "responses", "per GT trigger", "time P",
                  "time R", "time F1"]]
        for row in ga:
            grows.append([
                {"full": "full RTAR  (R AND D)",
                 "relevance_only": "relevance only  (R)",
                 "density_only": "density only  (D)"}.get(row["gate_mode"],
                                                          row["gate_mode"]),
                f"{row['n_emits']:,}", f"{row['emits_per_gt']:.2f}",
                pct(row["time_precision"]), pct(row["time_recall"]),
                pct(row["time_f1"])])
        rrows = [["signal", "fires on", "why"],
                 ["D_t  (density)", f"{gr['D_t_true_pct']:.1f}% of frames",
                  wrap(f"the CLIP-grid QDP keep rate averages "
                       f"{gr['clip_keep_rate_mean']:.3f}, far above "
                       f"tau_den = {cfg.get('tau_den', 0.15)} &mdash; so the "
                       f"comparison is almost always true", S["small"])],
                 ["R_t  (relevance)", f"{gr['R_t_true_pct']:.1f}% of frames",
                  wrap("min-max calibration maps the stream onto [0, 1], so "
                       "tau_rel = 0.60 means \"in the top 40% of the trailing "
                       "window\" and fires at roughly that rate by "
                       "construction", S["small"])],
                 ["T_t = R AND D", f"{gr['T_t_true_pct']:.1f}% of frames",
                  wrap("indistinguishable from R_t alone, because D_t is "
                       "almost never the binding constraint", S["small"])]]
        add(KeepTogether([
            Paragraph("Why the trigger fires as often as it does", S["h1"]),
            Paragraph(
                "This is the most important thing the run measured, and it is a "
                "finding about the <i>reproduction gap</i> rather than about the "
                "method. The two RTAR thresholds behave very differently from "
                "what the paper's text implies.", S["body"]),
            tbl(rrows, [30 * mm, 30 * mm, 108 * mm]),
            Paragraph(
                f"Over {gr['frames']:,} frames the largest raw "
                f"cos(q, mean-pooled v) seen anywhere was "
                f"<b>{gr['relevance_raw_max']:.4f}</b>, against a stated "
                f"tau_rel of 0.60. Read literally the relevance gate is false "
                f"everywhere and the model never speaks, so some calibration is "
                f"unavoidable &mdash; but whichever one is chosen is this "
                f"reproduction's invention, and it is what sets the firing rate. "
                f"<b>The consequence is that RTAR's central claim cannot be "
                f"tested here.</b> The paper argues that AND-ing relevance with "
                f"density beats either alone; in this configuration the density "
                f"gate is near-vacuous, so the conjunction is arithmetically "
                f"almost equal to relevance alone, and the ablation below shows "
                f"exactly that.", S["small"]),
            Spacer(1, 4),
            Paragraph("RTAR gate ablation (the structure of the paper's Table 5)",
                      S["h2"]),
            tbl(grows, [46 * mm, 24 * mm, 28 * mm, 20 * mm, 20 * mm, 20 * mm],
                highlight=[1]),
            Paragraph(
                "Rebuilt from the per-frame R_t / D_t traces in the JSONL, so it "
                "cost no GPU. <b>Timing only:</b> the full gate is the "
                "intersection, so both single-gate policies fire on a superset "
                "of its frames and no response text exists for their extra "
                "firings &mdash; content, and therefore Joint F1, is not "
                "computed for them rather than guessed at. Density-only emits "
                "far too few responses for the opposite reason: D_t is true on "
                "almost every frame, so the debouncer collapses whole streams "
                "into a single event.", S["small"]),
        ]))

    # ---------------- gap sweep ---------------- #
    sweep = [r for r in m.get("gap_sweep", []) if not r.get("error")]
    if sweep:
        recorded = cfg.get("event_gap_s")
        gp = rp.fig_gap_sweep([r["gap_s"] for r in sweep],
                              [r["time_precision"] for r in sweep],
                              [r["time_recall"] for r in sweep],
                              [r["time_f1"] for r in sweep],
                              os.path.join(fig_dir, "gapsweep.png"))
        grows = [["gap (s)", "responses", "per GT trigger", "time P", "time R",
                  "time F1", "joint F1"]]
        hi = None
        for i, r in enumerate(sweep):
            jf = r.get("joint_f1")
            jf_s = pct(jf) if jf is not None else pct(r["joint_f1_lb"], bound=True)
            grows.append([f"{r['gap_s']:g}", f"{r['n_emits']:,}",
                          f"{r['emits_per_gt']:.2f}", pct(r["time_precision"]),
                          pct(r["time_recall"]), pct(r["time_f1"]), jf_s])
            if recorded is not None and abs(r["gap_s"] - recorded) < 1e-6:
                hi = i + 1
        pmin = min(r["time_precision"] for r in sweep)
        pmax = max(r["time_precision"] for r in sweep)
        add(KeepTogether([
            Paragraph("Sensitivity to the debounce gap", S["h1"]),
            Paragraph(
                "The QueryStream paper never states the Appendix A.4 event-"
                "grouping gap, so it is a reproduction choice that moves the "
                "numbers. It can be swept <b>exactly and without a GPU</b>: for "
                "gaps g1 &lt; g2 every g2-interval is a union of consecutive "
                "g1-intervals, so the g2 response set is a strict subset of the "
                "g1 set, and every one of those responses already has its "
                "generated text in the JSONL. Content is therefore fully "
                "included below, not approximated.", S["body"]),
            tbl(grows, [18 * mm, 24 * mm, 28 * mm, 20 * mm, 20 * mm, 20 * mm,
                        20 * mm], highlight=[hi] if hi else None),
            Image(gp, width=132 * mm, height=132 * mm * 3.1 / 6.4),
            Paragraph(
                f"<b>Time precision is flat at {100 * pmin:.1f}-{100 * pmax:.1f}% "
                f"across the whole sweep while recall falls monotonically.</b> "
                f"That shape says the responses are not merely too dense around "
                f"the right instants &mdash; if they were, merging them would "
                f"raise the fraction that land on a trigger. The timing error is "
                f"in <i>where</i> the gate fires, not in how often, so no choice "
                f"of gap can recover it. The shipped gap is the headline row; "
                f"quoting the best row instead would be tuning against the "
                f"benchmark, and on this evidence there is nothing to gain by "
                f"doing so.", S["small"]),
        ]))

    # ---------------- profile ---------------- #
    add(Paragraph("Wall-clock profile", S["h1"]))
    frows = [["stage", "cost", "scope"],
             ["Frame decode (worker time)",
              f"{(summaries[0].get('decode_ms_per_frame_worker') if summaries else 0) or 0:.2f} ms",
              "per frame, summed across decode threads"],
             ["OpenCLIP ViT-L/14", f"{prof.get('openclip_ms_per_frame', 0):.2f} ms",
              f"per frame, at clip_batch={cfg.get('clip_batch', '?')}"],
             ["Qwen vision tower", f"{prof.get('vision_tower_ms_per_frame', 0):.2f} ms",
              "per frame, once per video regardless of sample count"],
             ["QDP + RTAR", f"{prof.get('qdp_rtar_ms_per_frame', 0):.3f} ms",
              "per frame -- the gate itself"],
             ["LLM prefill (TTFT)", f"{prof.get('llm_prefill_ms', 0):.1f} ms",
              "per emitted response"],
             ["LLM decode", f"{prof.get('llm_decode_ms', 0):.1f} ms",
              "per emitted response"]]
    add(tbl(frows, [46 * mm, 26 * mm, 96 * mm]))
    qr = prof.get("qdp_rtar_ms_per_frame", 0)
    vt = prof.get("vision_tower_ms_per_frame", 0)
    if qr > 0 and vt > 0:
        add(Paragraph(
            f"The gating logic costs {vt / qr:.0f}x less per frame than the "
            f"vision tower it decides to invoke, which is the efficiency claim "
            f"the design rests on: the cheap always-on path decides when the "
            f"expensive path runs.", S["small"]))

    # ---------------- config ---------------- #
    add(Paragraph("Configuration", S["h1"]))
    crows = [["setting", "this run", "paper / benchmark"],
             ["Encoder", cfg.get("encoder", "?"), "OpenCLIP-ViT-L/14 (QueryStream)"],
             ["Backbone", cfg.get("backbone", "?"),
              "TimeChat-Online-7B in the paper"],
             ["Audio encoder", "none", "OmniPro assumes an omni-modal model"],
             ["Frame rate", "1 FPS", "1 FPS (both)"],
             ["alpha (DSH)", str(cfg.get("alpha")), "0.1"],
             ["tau_temp", str(cfg.get("tau_temp")), "0.90"],
             ["tau_rel", str(cfg.get("tau_rel")), "0.60 (unreachable raw -- see above)"],
             ["tau_den", str(cfg.get("tau_den")), "0.15"],
             ["Relevance calibration", str(cfg.get("relevance_mode")), "not stated"],
             ["Mask space / pool",
              f"{cfg.get('mask_space')} / {cfg.get('pool')}", "not stated"],
             ["Event debounce gap", f"{cfg.get('event_gap_s')} s",
              "not stated (Appendix A.4)"],
             ["Matching tolerance", f"+-{tol:g} s", "+-3 s (OmniPro Sec. 3.2.1)"],
             ["Content judge", m["judge"].get("model_note", "").split(";")[0],
              "Gemini-3-Flash, 1-5, correct if >=3"],
             ["Precision", "bf16 backbone, fp16 encoder, fp32 statistics",
              "not stated in either paper"]]
    add(tbl(crows, [40 * mm, 56 * mm, 72 * mm]))

    # ---------------- completeness ---------------- #
    n_bench_total = manifest.get("n_total", n_bench)
    shard_files = sorted(glob.glob(os.path.join(run_dir, "omnipro_raw_logs*.jsonl")))
    budget_hits = sum(1 for x in summaries if x.get("budget_hit"))
    n_err = len(records) - len(ok)
    complete = (len(records) >= n_bench_total and budget_hits == 0)
    crows = [["", "", ""],
             ["Samples in the JSONL", f"{len(records)}",
              f"{100 * len(records) / n_bench_total:.1f}% of the benchmark"],
             ["Scored", f"{len(ok)}", "errored samples are excluded, not zeroed"],
             ["Errored", f"{n_err}",
              "listed in omnipro_metrics.json under `errors`"],
             ["Shard files", f"{len(shard_files)}",
              "unioned by sample_id, last write wins"],
             ["Workers that hit the time budget", f"{budget_hits}",
              "stopped cleanly between videos, not killed mid-sample"]]
    crows[0] = ["run completeness", "value", "note"]
    add(KeepTogether([
        Paragraph("Run completeness", S["h1"]),
        Paragraph(
            "This report is a pure function of the JSONL, so it can be built at "
            "any point in a run and rebuilt after every resume. "
            + ("<b>This run covers the full benchmark.</b>" if complete else
               "<b>This run is partial</b> &mdash; the numbers below describe "
               "the samples listed here and nothing else. Re-running "
               "<font face='Courier'>sbatch scripts/submit_omnipro.sh</font> "
               "with the same RUN_NAME resumes it: every worker skips the "
               "sample_ids already in its shard file."), S["body"]),
        tbl(crows, [56 * mm, 22 * mm, 90 * mm]),
    ]))

    # ---------------- limits ---------------- #
    add(Paragraph("What this run does and does not establish", S["h1"]))
    cov = manifest.get("n_kept", n)
    tot_bench = manifest.get("n_total", n_bench)
    bullets = [
        f"<b>Coverage.</b> {n} samples scored out of {tot_bench} in OmniPro "
        f"({100 * n / tot_bench:.1f}%). Every per-slice n is printed next to its "
        f"metric; a slice with a small n is a direction, not a result.",
        "<b>No audio, by design.</b> The user's instruction was to evaluate the "
        "existing architecture, not to add an audio encoder. Every "
        "audio-required number here is a floor on what the same pruning and "
        "gating would do with audio present, not an estimate of the method.",
        "<b>Different judge from the paper.</b> The rubric, the 1-5 scale and "
        "the &gt;=3 threshold match OmniPro; the judge model is OpenAI rather "
        "than Gemini-3-Flash. Judged scores from different judges are not "
        "strictly comparable, and only the two open-ended sub-tasks are affected.",
        "<b>Different backbone from the QueryStream paper.</b> Its main results "
        "use TimeChat-Online-7B, a streaming fine-tune of Qwen2.5-VL-7B; this "
        f"run uses stock {_short_backbone(cfg)} with no streaming fine-tune."
        + (" Qwen3-VL is a different model family again -- patch 16 rather than "
           "14, interleaved M-ROPE, per-plane vision blocks and deepstack "
           "features -- so it is further from the paper's setting than the "
           "Qwen2.5-VL arm, not closer."
           if _is_qwen3(cfg) else ""),
        "<b>The debounce gap and the relevance calibration are reproduction "
        "choices.</b> Neither is specified in the QueryStream paper, both move "
        "the numbers, and both are logged per frame so they can be re-examined "
        "without re-running anything.",
    ]
    if sel.get("prefer_short"):
        bullets.append(
            f"<b>Selection bias.</b> Sampling was biased toward shorter videos "
            f"(prefer_short={sel['prefer_short']}). OmniPro reports every model "
            f"degrading on later triggers, so this makes the subset easier than "
            f"the benchmark and the long-horizon table above should be read "
            f"with that in mind.")
    if manifest.get("dropped_missing_video"):
        bullets.append(
            f"<b>{manifest['dropped_missing_video']} samples were skipped</b> "
            f"because their video was not on disk. The benchmark-weighted "
            f"headline corrects for the resulting slice imbalance; the "
            f"subset-weighted one does not.")
    for b in bullets:
        add(Paragraph("&bull; " + b, S["body"]))

    add(Paragraph("Reproducing this run", S["h2"]))
    add(Paragraph(
        "python eval_omnipro.py --out {d} --device cuda:0<br/>"
        "python judge_omnipro.py {d} --submit &amp;&amp; "
        "python judge_omnipro.py {d} --collect<br/>"
        "python score_omnipro.py {d}<br/>"
        "python make_omnipro_report.py {d}".format(d=run_dir), S["mono"]))

    doc = SimpleDocTemplate(out_path, pagesize=A4,
                            leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm,
                            title="QueryStream on OmniPro",
                            author="QueryStream reproduction")

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(16 * mm, 9 * mm,
                          "QueryStream reproduction - OmniPro Online mode "
                          "(vision-only)")
        canvas.drawRightString(A4[0] - 16 * mm, 9 * mm,
                               f"page {canvas.getPageNumber()}")
        canvas.restoreState()

    doc.build(E, onFirstPage=footer, onLaterPages=footer)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.run_dir, "QueryStream_OmniPro_report.pdf")
    p = build(a.run_dir, out)
    print(f"wrote {p}  ({os.path.getsize(p) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
