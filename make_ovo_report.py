#!/usr/bin/env python3
"""Compile the OVO-Bench run into a PDF laid out like the paper's Table 2.

    python make_ovo_report.py runs/ovo_20pct
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from ovo_eval import dataset as ds
from ovo_eval import report as rp

INK = colors.HexColor("#0b0b0b"); INK2 = colors.HexColor("#52514e")
MUTED = colors.HexColor("#8a8983"); GRID = colors.HexColor("#e4e3de")
BAND = colors.HexColor("#eef4fc")


def styles():
    b = getSampleStyleSheet()
    mk = lambda **kw: ParagraphStyle(**kw)
    return {
        "title": mk(name="t", parent=b["Title"], fontName="Helvetica-Bold",
                    fontSize=18, leading=22, textColor=INK, alignment=TA_LEFT,
                    spaceAfter=2),
        "sub": mk(name="s", parent=b["Normal"], fontName="Helvetica", fontSize=9,
                  leading=13, textColor=INK2, spaceAfter=9),
        "h1": mk(name="h1", parent=b["Heading1"], fontName="Helvetica-Bold",
                 fontSize=12, leading=15, textColor=INK, spaceBefore=13, spaceAfter=5),
        "h2": mk(name="h2", parent=b["Heading2"], fontName="Helvetica-Bold",
                 fontSize=9.5, leading=12, textColor=INK, spaceBefore=8, spaceAfter=3),
        "body": mk(name="b", parent=b["Normal"], fontName="Helvetica", fontSize=8.5,
                   leading=12, textColor=INK, spaceAfter=5),
        "small": mk(name="sm", parent=b["Normal"], fontName="Helvetica", fontSize=7.3,
                    leading=10, textColor=MUTED, spaceAfter=4),
        "mono": mk(name="m", parent=b["Normal"], fontName="Courier", fontSize=7,
                   leading=9.5, textColor=INK2, spaceAfter=3),
    }


def tbl(data, widths, *, size=6.6, highlight=None, header=True, bold_rows=()):
    t = Table(data, colWidths=widths, hAlign="LEFT")
    cmds = [("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), size),
            ("TEXTCOLOR", (0, 0), (-1, -1), INK),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("LINEBELOW", (0, 0), (-1, -2), 0.35, GRID)]
    if header:
        cmds += [("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                 ("TEXTCOLOR", (0, 0), (-1, 0), INK2),
                 ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2)]
    for r in (highlight or []):
        cmds += [("BACKGROUND", (0, r), (-1, r), BAND),
                 ("FONTNAME", (0, r), (-1, r), "Helvetica-Bold")]
    for r in bold_rows:
        cmds += [("FONTNAME", (0, r), (-1, r), "Helvetica-Bold")]
    t.setStyle(TableStyle(cmds))
    return t


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - m), 100 * (c + m))


def load(run_dir: str):
    m = json.load(open(os.path.join(run_dir, "ovo_metrics.json")))
    recs: List[dict] = []
    for p in sorted(glob.glob(os.path.join(run_dir, "ovo_raw_logs*.jsonl"))):
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        recs.append(json.loads(line))
                    except Exception:
                        pass
    recs = list({r["call_id"]: r for r in recs}.values())
    summaries = []
    for p in sorted(glob.glob(os.path.join(run_dir, "summary*.json"))):
        try:
            summaries.append(json.load(open(p)))
        except Exception:
            pass
    prof: Dict[str, float] = {}
    w: Dict[str, float] = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "profile_summary*.csv"))):
        for row in csv.DictReader(open(p)):
            if row.get("scope") != "OVERALL":
                continue
            nf = float(row.get("n_frames") or 0) or 1.0
            for k in ("decode_ms_per_frame", "openclip_ms_per_frame",
                      "vision_tower_ms_per_frame", "qdp_rtar_ms_per_frame",
                      "llm_prefill_ms", "llm_decode_ms"):
                prof[k] = prof.get(k, 0.0) + float(row.get(k) or 0) * nf
                w[k] = w.get(k, 0.0) + nf
            prof["n_frames"] = prof.get("n_frames", 0.0) + nf
    for k in list(prof):
        if k in w and w[k]:
            prof[k] /= w[k]
    return m, recs, summaries, prof


def build(run_dir: str, out_path: str) -> str:
    m, recs, summaries, prof = load(run_dir)
    S = styles()
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    met = m["metrics"]
    cfg = m.get("config", {}) or {}
    sel = (summaries[0].get("selection") if summaries else {}) or {}
    man = (summaries[0].get("manifest") if summaries else {}) or {}
    keep = m["token_keep_rate"]

    E: List[Any] = []
    add = E.append
    add(Paragraph("QueryStream on OVO-Bench", S["title"]))
    add(Paragraph(
        f"Query-aware pruning (QDP) on <b>TimeChat-Online-7B</b> &mdash; the "
        f"backbone the QueryStream paper actually uses &mdash; scored under "
        f"OVO-Bench's own protocol. Generated {datetime.now():%Y-%m-%d %H:%M} "
        f"from <font face='Courier'>{run_dir}</font>.", S["sub"]))

    add(Paragraph("Why this run exists", S["h1"]))
    add(Paragraph(
        "Every earlier arm of this reproduction ran QDP+RTAR on stock "
        "Qwen2.5-VL-7B-Instruct, so each result carried an unquantified "
        "backbone gap: the paper's headline instead swaps QueryStream into "
        "<b>TimeChat-Online-7B</b>, a streaming fine-tune of the same base "
        "model, replacing its query-agnostic DTD module with QDP+RTAR. This run "
        "closes that gap &mdash; same modules, same thresholds, the paper's "
        "backbone &mdash; on the benchmark of its Table 2.", S["body"]))
    add(Paragraph(
        "TimeChat-Online-7B loads as <font face='Courier'>"
        "Qwen2_5_VLForConditionalGeneration</font> with the same vision "
        "geometry, so the existing wrapper takes it unchanged; only the weights "
        "differ. DTD is not disabled so much as bypassed: QDP does the pruning "
        "and the backbone is fed an already-sparse token set.", S["body"]))

    # ---- headline Table 2 ---- #
    add(Paragraph("Results, in the paper's Table 2 layout", S["h1"]))
    hdr = ["Model", "Keep"] + list(ds.ALL_TASKS) + ["RT", "BW", "FW", "All"]
    rows = [hdr]

    def fmt(v):
        return f"{v:.1f}" if isinstance(v, (int, float)) else "-"

    ours = [f"QueryStream (ours)", f"{100*keep:.1f}%"]
    for t in ds.ALL_TASKS:
        ours.append(fmt(met["per_task"].get(t, {}).get("acc")))
    for c in ds.CATEGORY_ORDER:
        ours.append(fmt(met["category"][c]))
    ours.append(fmt(met["overall"]))
    rows.append(ours)

    for name in ("TimeChat-Online-7B (100%)", "TimeChat-Online-7B (55.4%)",
                 "TimeChat-Online-7B (15.2%)", "QueryStream-7B (52.9%)",
                 "QueryStream-7B (20.0%)", "Qwen2-VL-7B (64f)",
                 "GPT-4o (64f)", "Gemini 1.5 Pro", "Human Agents"):
        pr = ds.PAPER[name]
        r = [f"paper: {name}", ""]
        r += [fmt(pr[t]) for t in ds.ALL_TASKS]
        r += [fmt(pr[c]) for c in ds.CATEGORY_ORDER]
        r.append(fmt(pr["overall"]))
        rows.append(r)

    widths = [46 * mm, 12 * mm] + [10.5 * mm] * 12 + [12 * mm] * 4
    add(tbl(rows, widths, highlight=[1]))
    add(Paragraph(
        "RT / BW / FW are the three category averages, each a <b>macro</b> mean "
        "over its tasks; <b>All</b> is the mean of the three. That is "
        "OVO-Bench's own aggregation (<font face='Courier'>utils/"
        "OVOBenchScore.py</font>), so a task with many calls cannot dominate "
        "its category.", S["small"]))

    # ---- deltas ---- #
    tgt = ds.PAPER[ds.TARGET_QUERYSTREAM]
    base = ds.PAPER[ds.TARGET_BASELINE]
    drows = [["comparison", "RT", "BW", "FW", "Overall", "what it isolates"]]
    if met["overall"] is not None:
        drows.append(["ours - QueryStream-7B (52.9%)"]
                     + [f"{met['category'][c]-tgt[c]:+.1f}" for c in ds.CATEGORY_ORDER]
                     + [f"{met['overall']-tgt['overall']:+.1f}",
                        "the reproduction gap on the paper's own backbone"])
        drows.append(["ours - TimeChat-Online-7B (100%)"]
                     + [f"{met['category'][c]-base[c]:+.1f}" for c in ds.CATEGORY_ORDER]
                     + [f"{met['overall']-base['overall']:+.1f}",
                        "what QDP adds over the un-pruned backbone"])
        add(tbl(drows, [50 * mm, 14 * mm, 14 * mm, 14 * mm, 18 * mm, 66 * mm],
                size=7.4, highlight=[1]))
        add(Paragraph(
            "The second row is the one that says whether the method did "
            "anything <i>here</i>: it holds the backbone fixed and varies only "
            "the pruning. The first row mixes in every remaining reproduction "
            "choice &mdash; OpenCLIP weights, score-space resampling, the "
            "relevance calibration &mdash; and a different keep rate.",
            S["small"]))

    # ---- per task ---- #
    add(Paragraph("Per sub-task detail", S["h1"]))
    prows = [["task", "name", "category", "n", "acc", "95% CI",
              "paper QS 52.9%", "delta", "strict-letter"]]
    for t in ds.ALL_TASKS:
        d = met["per_task"].get(t)
        if not d:
            continue
        lo, hi = wilson(d["correct"], d["n"])
        prows.append([t, ds.TASK_NAME[t], ds.CATEGORY_TITLE[ds.CATEGORY[t]][:18],
                      str(d["n"]), f"{d['acc']:.1f}", f"{lo:.0f}-{hi:.0f}",
                      f"{tgt[t]:.1f}", f"{d['acc']-tgt[t]:+.1f}",
                      f"{d['strict_letter_acc']:.1f}" if "strict_letter_acc" in d else "-"])
    add(tbl(prows, [12 * mm, 52 * mm, 30 * mm, 12 * mm, 14 * mm, 20 * mm,
                    22 * mm, 16 * mm, 20 * mm], size=7.2))
    add(Paragraph(
        "<b>strict-letter</b> is a diagnostic, not the metric. OVO-Bench scores "
        "a multiple-choice reply with <font face='Courier'>int(gt in response)"
        "</font> &mdash; a bare substring test, so any reply containing the "
        "letter counts. The strict column requires the reply to be <i>only</i> "
        "that letter. A large gap between them means the headline is leaning on "
        "the loose rule; a small one means the model is answering in the "
        "required format.", S["small"]))

    # ---- figures ---- #
    labels = [t for t in ds.ALL_TASKS if t in met["per_task"]]
    if labels:
        p = rp.fig_tasks(
            labels,
            {"QueryStream (ours, TimeChat-Online)":
                [met["per_task"][t]["acc"] for t in labels],
             "paper QueryStream-7B (52.9%)": [tgt[t] for t in labels],
             "paper TimeChat-Online-7B (100%)": [base[t] for t in labels]},
            os.path.join(fig_dir, "tasks.png"))
        add(Spacer(1, 3))
        add(Image(p, width=228 * mm, height=228 * mm * 3.2 / 10.0))

    keeps = [r["keep_rate"] for r in recs if r.get("keep_rate")]
    raws = [r["relevance_raw_max"] for r in recs if r.get("relevance_raw_max") is not None]
    figs = []
    if keeps:
        figs.append(rp.fig_keep(keeps, keep, os.path.join(fig_dir, "keep.png")))
    if raws:
        figs.append(rp.fig_relevance(raws, cfg.get("tau_rel", 0.60),
                                     os.path.join(fig_dir, "rel.png")))
    if figs:
        row = [[Image(f, width=95 * mm, height=95 * mm * 3.0 / 5.0) for f in figs]]
        t = Table(row, colWidths=[100 * mm] * len(figs), hAlign="LEFT")
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
        add(t)

    # ---- coverage / config / limits ---- #
    add(Paragraph("Coverage, configuration and limits", S["h1"]))
    crows = [["item", "value", "note"],
             ["Calls scored", f"{met['n_calls']}",
              f"of {man.get('n_calls', '?')} in the full benchmark"],
             ["Samples", f"{sel.get('n_samples', '?')}",
              f"stratified per task, fraction {sel.get('fraction', 1.0)}, "
              f"seed {sel.get('seed', 0)}"],
             ["Errored calls", f"{m['n_errored']}", "excluded, not scored zero"],
             ["Null responses", f"{met['n_null_responses']}",
              "scored zero -- the model produced nothing"],
             ["Token keep rate", f"{100*keep:.2f}%",
              "backbone grid; paper's comparable rows are 52.9% and 100%"],
             ["Backbone", str(m.get("backbone")), "the paper's own"],
             ["Encoder", str(cfg.get("encoder")), "OpenCLIP-ViT-L/14"],
             ["alpha / tau_temp", f"{cfg.get('alpha')} / {cfg.get('tau_temp')}",
              "paper: 0.1 / 0.90"],
             ["tau_rel / tau_den", f"{cfg.get('tau_rel')} / {cfg.get('tau_den')}",
              "paper: 0.60 / 0.15"],
             ["Relevance calibration", str(cfg.get("relevance_mode")),
              "not stated in the paper -- tau_rel=0.60 is unreachable on raw cosine"],
             ["Mask space / pool", f"{cfg.get('mask_space')} / {cfg.get('pool')}",
              "not stated in the paper"]]
    add(tbl(crows, [40 * mm, 44 * mm, 110 * mm], size=7.4))

    bullets = [
        "<b>Forward Active Responding is scored, but RTAR is not.</b> OVO-Bench "
        "ships every question as a clip pre-trimmed to its probe time, so the "
        "model answers where the benchmark asks. The paper's Table 5 RTAR "
        "ablation needs the Appendix A.4 simulated protocol, which places a "
        "response at a timestamp the model chooses &mdash; that requires the "
        "untrimmed source video, which the released clips are not. R_t / D_t / "
        "T_t are computed and logged per frame regardless.",
        "<b>The timeliness exponent p is never specified</b> in the paper, so "
        "Table 5's Score column cannot be reproduced even with full video.",
        "<b>Subset, not full benchmark.</b> Sampling is stratified per task at "
        "the sample level, keeping each forward sample's probes together, so "
        "every task retains a defined accuracy and the macro averages stay "
        "meaningful. Per-task 95% Wilson intervals are printed above; small "
        "tasks are wide and should be read as directions.",
        "<b>OpenCLIP weights remain unresolved.</b> The paper cites Cherti et "
        "al. (LAION, laion2b_s32b_b82k); this run loads the `openai` weights "
        "through open_clip. Never measured either way.",
    ]
    for b in bullets:
        add(Paragraph("&bull; " + b, S["body"]))

    if prof:
        add(Paragraph("Wall-clock profile", S["h2"]))
        frows = [["stage", "cost", "scope"],
                 ["Frame decode", f"{prof.get('decode_ms_per_frame',0):.2f} ms", "per frame"],
                 ["OpenCLIP ViT-L/14", f"{prof.get('openclip_ms_per_frame',0):.2f} ms", "per frame"],
                 ["Vision tower", f"{prof.get('vision_tower_ms_per_frame',0):.2f} ms", "per frame"],
                 ["QDP + RTAR", f"{prof.get('qdp_rtar_ms_per_frame',0):.3f} ms", "per frame"],
                 ["LLM prefill (TTFT)", f"{prof.get('llm_prefill_ms',0):.1f} ms", "per call"],
                 ["LLM decode", f"{prof.get('llm_decode_ms',0):.1f} ms", "per call"]]
        add(tbl(frows, [40 * mm, 26 * mm, 60 * mm], size=7.4))

    add(Paragraph("Reproducing this run", S["h2"]))
    add(Paragraph(
        f"python tests/test_ovo_scoring.py<br/>"
        f"FRACTION={sel.get('fraction', 1.0)} sbatch scripts/submit_ovo.sh<br/>"
        f"python score_ovo.py {run_dir}<br/>"
        f"python make_ovo_report.py {run_dir}", S["mono"]))

    doc = SimpleDocTemplate(out_path, pagesize=landscape(A4),
                            leftMargin=12 * mm, rightMargin=12 * mm,
                            topMargin=12 * mm, bottomMargin=12 * mm,
                            title="QueryStream on OVO-Bench",
                            author="QueryStream reproduction")

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(12 * mm, 7 * mm,
                          "QueryStream reproduction - OVO-Bench on TimeChat-Online-7B")
        canvas.drawRightString(landscape(A4)[0] - 12 * mm, 7 * mm,
                               f"page {canvas.getPageNumber()}")
        canvas.restoreState()

    doc.build(E, onFirstPage=footer, onLaterPages=footer)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.run_dir, "QueryStream_OVOBench_report.pdf")
    p = build(a.run_dir, out)
    print(f"wrote {p}  ({os.path.getsize(p)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
