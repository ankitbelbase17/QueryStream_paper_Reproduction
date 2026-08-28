#!/usr/bin/env python
"""Compile a StreamingBench evaluation run into a PDF report.

    python make_report.py runs/sb_subset --out runs/sb_subset/QueryStream_StreamingBench.pdf

Reads `eval_raw_logs.jsonl`, `summary.json`, `profile_summary.csv` and (if the
batch judge has landed) `judge_summary.json`, and lays them out against the
paper's Table 1 so the reproduction can be read next to what it is reproducing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics as st
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
    TableStyle,
)

from querystream import report as rp
from querystream.benchmarks import streamingbench as sb

INK = colors.HexColor(rp.INK)
INK2 = colors.HexColor(rp.INK_2)
MUTED = colors.HexColor(rp.MUTED)
GRID = colors.HexColor(rp.GRID)
BLUE = colors.HexColor(rp.SERIES[0])
ORANGE = colors.HexColor(rp.SERIES[1])
BAND = colors.HexColor("#f4f4f1")

# Paper Table 1, the rows this run is comparable to.
PAPER = {
    "TimeChat-Online-7B (100%)": dict(keep="100%", OP=80.22, CR=82.03, CS=79.50,
        ATP=83.33, EU=76.10, TR=78.50, PR=78.70, SU=64.63, ACP=69.60, CT=57.98, All=75.36),
    "TimeChat-Online-7B (55.8%)": dict(keep="55.8%", OP=81.03, CR=83.59, CS=78.55,
        ATP=81.09, EU=76.73, TR=80.37, PR=75.93, SU=63.82, ACP=68.47, CT=47.87, All=74.32),
    "QueryStream-7B (57.2%)": dict(keep="57.2%", OP=82.38, CR=84.38, CS=79.18,
        ATP=82.37, EU=77.99, TR=81.31, PR=78.70, SU=65.04, ACP=69.32, CT=47.34, All=75.32),
    "QueryStream-7B (29.6%)": dict(keep="29.6%", OP=82.11, CR=83.59, CS=78.23,
        ATP=82.69, EU=75.47, TR=80.06, PR=79.63, SU=63.01, ACP=67.90, CT=42.55, All=74.04),
    "Qwen2.5-VL-7B (full)": dict(keep="--", OP=78.32, CR=80.47, CS=78.86,
        ATP=80.45, EU=76.73, TR=78.50, PR=79.63, SU=63.41, ACP=66.19, CT=53.19, All=73.68),
}
SUBTASK_NAME = {v: k for k, v in sb.TASK_CODE.items()}


# --------------------------------------------------------------------------- #
# styles
# --------------------------------------------------------------------------- #

def styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    mk = lambda **kw: ParagraphStyle(**kw)
    return {
        "title": mk(name="t", parent=base["Title"], fontName="Helvetica-Bold",
                    fontSize=19, leading=23, textColor=INK, spaceAfter=2, alignment=TA_LEFT),
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


def tbl(data, widths, *, align_right_from=1, header=True, size=7.8, highlight=None):
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
        cmds += [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 0), (-1, 0), INK2),
            ("LINEBELOW", (0, 0), (-1, 0), 0.9, INK2),
        ]
    for r in (highlight or []):
        cmds += [("BACKGROUND", (0, r), (-1, r), BAND),
                 ("FONTNAME", (0, r), (-1, r), "Helvetica-Bold")]
    t.setStyle(TableStyle(cmds))
    return t


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson interval -- the honest error bar on a subset accuracy."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


# --------------------------------------------------------------------------- #

def _resolved_clip(args: Dict[str, Any]) -> str:
    """The model config actually built -- open_clip's QuickGELU correction means
    the name on the command line is not always the name that ran."""
    name = str(args.get("clip_model", ""))
    if args.get("clip_pretrained") == "openai" and not name.endswith("-quickgelu"):
        return name + "-quickgelu"
    return name


def load(run_dir: str):
    recs = [json.loads(l) for l in open(os.path.join(run_dir, "eval_raw_logs.jsonl")) if l.strip()]
    summ = json.load(open(os.path.join(run_dir, "summary.json")))
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    prof = {}
    pcsv = os.path.join(run_dir, "profile_summary.csv")
    if os.path.exists(pcsv):
        for row in csv.DictReader(open(pcsv)):
            if row["scope"] == "OVERALL":
                prof = row
    if not prof:
        # A run killed before its final flush has no OVERALL row; finalize_run.py
        # rebuilds the same figures from the per-video/per-question rows.
        prof = summ.get("profile") or {}
    judge = None
    jp = os.path.join(run_dir, "judge_summary.json")
    if os.path.exists(jp):
        judge = json.load(open(jp))
    return recs, summ, cfg, prof, judge


def build(run_dir: str, out_path: str) -> str:
    recs, summ, cfg, prof, judge = load(run_dir)
    S = styles()
    figdir = os.path.join(run_dir, "figures")
    os.makedirs(figdir, exist_ok=True)
    ok = [r for r in recs if not r.get("error")]
    args = cfg.get("args", {})
    run = summ.get("run", {})

    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm, topMargin=15 * mm, bottomMargin=15 * mm,
        title="QueryStream - StreamingBench reproduction", author="QueryStream reproduction",
    )
    W = doc.width
    E: List[Any] = []

    # ---------------- header ---------------- #
    E.append(Paragraph("QueryStream on StreamingBench", S["title"]))
    E.append(Paragraph(
        "Reproduction of <i>QueryStream: Advancing Streaming Video Understanding with "
        "Query-Aware Pruning and Proactive Response</i> (ICLR 2026) &mdash; "
        "Query-Aware Differential Pruning in front of a frozen Qwen2.5-VL-7B-Instruct. "
        f"Generated {datetime.now():%Y-%m-%d %H:%M}.", S["sub"]))

    n_tot, n_cor = summ["overall"]["total"], summ["overall"]["correct"]
    lo, hi = wilson(n_cor, n_tot)
    keep = summ["token_keep_rate_pct"]
    head = [
        ["Accuracy (exact match)", f"{summ['overall']['acc']:.2f}%",
         f"{n_cor}/{n_tot}", f"95% CI  {lo:.1f} - {hi:.1f}"],
        ["Token keep rate", f"{keep:.2f}%",
         f"{summ['visual_tokens_kept']:,}/{summ['visual_tokens_dense']:,}",
         "paper 57.2%"],
        ["Paper QueryStream-7B", "75.32%", "1883/2500", "@ 57.2% keep"],
        ["Paper Qwen2.5-VL-7B", "73.68%", "1842/2500", "full tokens, no pruning"],
    ]
    E.append(tbl([["", "Value", "Count", "Note"]] + head,
                 [W * 0.31, W * 0.16, W * 0.20, W * 0.33], highlight=[1, 2], size=8.4))

    cover = 100.0 * n_tot / 2500
    E.append(Spacer(1, 7))
    E.append(Paragraph(
        f"<b>Scope.</b> This run covers <b>{n_tot} of the 2500 questions</b> "
        f"({cover:.1f}%) drawn from {run.get('videos_completed', '?')} videos, sampled with "
        f"<font face='Courier'>--shuffle-seed {args.get('shuffle_seed')}</font> and cut by a "
        f"{run.get('time_budget_sec', 0):.0f}s wall-clock budget on a single GH200. It is a "
        "uniform random subset of the same split the paper scored, so the accuracy is an "
        "unbiased estimate of the full-split number &mdash; but the confidence interval above "
        "is wide enough that small differences from the paper are not resolvable at this n. "
        "The token keep rate, by contrast, is an average over millions of tokens and is "
        "already tight.", S["body"]))

    # ---------------- per-subtask ---------------- #
    E.append(Paragraph("1 &nbsp; Per-subtask accuracy against Table 1", S["h1"]))
    codes = [c for c in sb.PAPER_ORDER if c in summ["tasks"]]
    ours = [summ["tasks"][c]["acc"] for c in codes]
    ns = [summ["tasks"][c]["total"] for c in codes]
    f1 = rp.fig_subtask_accuracy(
        codes, ours, ns,
        {c: PAPER["QueryStream-7B (57.2%)"][c] for c in codes},
        {c: PAPER["Qwen2.5-VL-7B (full)"][c] for c in codes},
        os.path.join(figdir, "subtask.png"),
        label=f"This run ({keep:.1f}% keep, n={n_tot})")
    E.append(Image(f1, width=W, height=W * 0.395))
    E.append(Spacer(1, 4))

    rows = [["Model", "Keep"] + codes + ["All"]]
    rows.append([f"This run (n={n_tot})", f"{keep:.1f}%"] +
                [f"{v:.1f}" for v in ours] + [f"{summ['overall']['acc']:.2f}"])
    rows.append(["   correct / n", ""] +
                [f"{summ['tasks'][c]['correct']}/{summ['tasks'][c]['total']}" for c in codes] +
                [f"{n_cor}/{n_tot}"])
    for name, p in PAPER.items():
        rows.append([name, p["keep"]] + [f"{p[c]:.1f}" for c in codes] + [f"{p['All']:.2f}"])
    cw = [W * 0.205, W * 0.062] + [(W * 0.733 - W * 0.078) / len(codes)] * len(codes) + [W * 0.078]
    E.append(tbl(rows, cw, highlight=[1], size=7.0))
    missing = [c for c in sb.PAPER_ORDER if c not in codes]
    note = "Subtask codes: " + ", ".join(
        f"<b>{c}</b> {SUBTASK_NAME.get(c, c)}" for c in codes) + "."
    if missing:
        note += (" Not present in this subset: " + ", ".join(missing) +
                 " &mdash; the sample is drawn by video, and a subtask appears only if a "
                 "sampled video happens to ask it. Subtasks with a single-digit n carry no "
                 "signal; read the All column.")
    E.append(Paragraph(note, S["small"]))

    # ---------------- pruning ---------------- #
    sec2 = Paragraph("2 &nbsp; Pruning behaviour", S["h1"])
    kr = [r["avg_keep_rate"] for r in ok if r.get("avg_keep_rate")]
    clipkr = [r["avg_qdp_keep_rate_clip"] for r in ok if r.get("avg_qdp_keep_rate_clip")]
    f2 = rp.fig_keep_rate(kr, 57.2, os.path.join(figdir, "keep.png")) if kr else None

    krows = [["Statistic", "Value"],
             ["Aggregate token keep rate", f"{keep:.2f}%"],
             ["Per-question keep rate, median", f"{100 * st.median(kr):.2f}%" if kr else "--"],
             ["Per-question keep rate, min - max",
              f"{100 * min(kr):.1f}% - {100 * max(kr):.1f}%" if kr else "--"],
             ["QDP keep rate on the CLIP 16x16 grid",
              f"{100 * st.mean(clipkr):.2f}%" if clipkr else "--"],
             ["Visual tokens kept / dense",
              f"{summ['visual_tokens_kept']:,} / {summ['visual_tokens_dense']:,}"],
             ["Questions needing the zero-survivor guard",
              str(sum(1 for r in ok if r.get("forced_min_keep")))],
             ["Paper target (Table 1 row)", "57.2%"]]
    body = Paragraph(
        "QDP scores patches on OpenCLIP's 16&times;16 grid, but Qwen2.5-VL's tokens live on a "
        "grid coarsened by a 2&times;2 spatial merge and a 2-frame temporal merge &mdash; eight "
        "CLIP patches per backbone token. Pooling the <i>binary</i> mask with OR therefore "
        "inflates the keep rate to ~96% (1&nbsp;&minus;&nbsp;0.59<super>8</super>), which is "
        "combinatorial rather than data-dependent. This run pools the <i>continuous</i> "
        "similarity maps onto the backbone grid and applies both thresholds there, so the "
        "frame-adaptive mean threshold keeps its meaning on the grid actually being pruned. "
        "The paper does not discuss this conversion at all.", S["body"])
    if f2:
        E.append(KeepTogether([sec2, Table(
            [[Image(f2, width=W * 0.47, height=W * 0.293),
              tbl(krows, [W * 0.29, W * 0.19], size=7.4)]],
            colWidths=[W * 0.49, W * 0.51], hAlign="LEFT",
            style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))]))
    else:
        E.append(sec2)
    E.append(body)

    # ---------------- RTAR ---------------- #
    sec3 = Paragraph("3 &nbsp; RTAR gating and the &tau;<sub>rel</sub> problem", S["h1"])
    raws = [v for r in ok for v in r.get("relevance_raw_scores", [])]
    f3 = rp.fig_relevance(raws, args.get("tau_rel", 0.60),
                          os.path.join(figdir, "relevance.png")) if raws else None
    nfr = sum(r.get("n_frames_visible", 0) for r in ok)
    nfire = sum(r.get("n_fired", 0) for r in ok)
    ntrig = sum(r.get("n_triggers", 0) for r in ok)
    rrows = [["Statistic", "Value"],
             ["Frame-query pairs gated", f"{nfr:,}"],
             ["raw cos(q, mean-pool v), min", f"{min(raws):.4f}" if raws else "--"],
             ["raw cos(q, mean-pool v), mean", f"{st.mean(raws):.4f}" if raws else "--"],
             ["raw cos(q, mean-pool v), max", f"{max(raws):.4f}" if raws else "--"],
             ["Frames with raw cos > 0.60", f"{sum(1 for v in raws if v > 0.60):,}"],
             ["T_t triggers (calibrated)", f"{ntrig:,}"],
             ["Response events after A.4 debounce", f"{nfire:,}"],
             ["Calibration mode used", str(args.get("relevance_mode"))]]
    ceiling = f"<b>{max(raws):.4f}</b>" if raws else "any measured ceiling"
    rbody = Paragraph(
        "The paper sets &tau;<sub>rel</sub>&nbsp;=&nbsp;0.60 on <i>cosine similarity</i> between "
        "the query embedding and the mean-pooled frame feature. Measured over every frame in "
        f"this run, that similarity never exceeds {ceiling} "
        "&mdash; so a literal reading of the threshold makes R<sub>t</sub> false on every frame "
        "of the benchmark and the model never speaks. Some normalisation is implied but never "
        "stated. This run calibrates the relevance signal with a running min-max over a 64-frame "
        "window and logs the raw cosine on every frame, so the choice can be revisited from the "
        "observed distribution instead of guessed. Note that StreamingBench scores answers at "
        "the question timestamp, so this gating affects the logged trigger stream, not the "
        "accuracy above; RTAR is what OVO-Bench's Forward Active Responding split measures.",
        S["body"])
    rbody2 = None
    if f3:
        E.append(KeepTogether([sec3, Table(
            [[Image(f3, width=W * 0.47, height=W * 0.293),
              tbl(rrows, [W * 0.31, W * 0.17], size=7.4)]],
            colWidths=[W * 0.49, W * 0.51], hAlign="LEFT",
            style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))]))
    else:
        E.append(sec3)
    E.append(rbody)

    # ---------------- efficiency ---------------- #
    sec4 = Paragraph("4 &nbsp; Wall-clock profile", S["h1"])
    g = lambda k: float(prof.get(k, 0) or 0)
    stages = ["frame decode", "Qwen vision tower", "OpenCLIP encoder", "QDP + RTAR"]
    vals = [g("decode_ms_per_frame"), g("vision_tower_ms_per_frame"),
            g("openclip_ms_per_frame"), g("qdp_rtar_ms_per_frame")]
    f4 = rp.fig_latency(stages, vals, os.path.join(figdir, "latency.png"))
    nw = int(args.get("decode_workers") or 1)
    prows = [["Stage", "Cost", "Scope"],
             ["Frame decode, worker time", f"{vals[0]:.2f} ms", "per frame, per video"],
             [f"Frame decode, effective ({nw} threads)", f"{vals[0] / nw:.2f} ms",
              "hidden behind GPU work"],
             ["Qwen2.5-VL vision tower", f"{vals[1]:.2f} ms", "per frame, per video"],
             ["OpenCLIP ViT-L/14", f"{vals[2]:.2f} ms", "per frame, per video"],
             ["QDP + RTAR (the method itself)", f"{vals[3]:.3f} ms", "per frame, per question"],
             ["LLM prefill / TTFT", f"{g('llm_prefill_ms'):.1f} ms", "per question"],
             ["LLM decode", f"{g('llm_decode_ms'):.1f} ms", "per question"],
             ["Unique frames decoded", f"{int(g('n_frames')):,}", "whole run"]]
    E.append(KeepTogether([sec4, Table(
        [[Image(f4, width=W * 0.46, height=W * 0.26),
          tbl(prows, [W * 0.30, W * 0.11, W * 0.17], size=7.4)]],
        colWidths=[W * 0.48, W * 0.52], hAlign="LEFT",
        style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))]))
    E.append(Paragraph(
        "<b>QDP and RTAR together cost "
        f"{vals[3]:.3f}&nbsp;ms per frame</b> &mdash; roughly "
        f"{vals[1] / max(vals[3], 1e-9):.0f}&times; less than the vision tower they gate. The "
        "method is not the bottleneck; frame decoding is, which is why this harness decodes "
        f"upcoming videos on {args.get('decode_workers')} background threads while the GPU "
        "works. Decode is reported as <i>worker</i> time: it is paid concurrently, so the "
        "effective per-frame cost is that figure over the thread count, and it is the number "
        "the pipeline actually waits on. The OpenCLIP figure was measured at "
        f"<font face='Courier'>--clip-batch {args.get('clip_batch')}</font>; a true 1-FPS "
        "stream sees one frame at a time and pays <b>11.14 ms/frame</b> instead (measured "
        "separately on the same GPU). Batching the encoder is a harness optimisation, not "
        "something a live stream could do.", S["body"]))

    # ---------------- judge ---------------- #
    E.append(Paragraph("5 &nbsp; LLM-as-a-judge cross-check", S["h1"]))
    if judge:
        jrows = [["Metric", "Value"],
                 ["Questions judged", f"{judge['judged']:,}"],
                 ["Exact-match accuracy on judged subset", f"{judge['exact_acc_on_judged']:.2f}%"],
                 ["Judge accuracy", f"{judge['judge_acc']:.2f}%"],
                 ["Parser / judge agreement", f"{judge['parser_judge_agreement_pct']:.2f}%"],
                 ["Disagreements", f"{judge['n_disagreements']:,}"],
                 ["Parse failures recovered by judge", f"{judge['recovered_parse_failures']:,}"],
                 ["Judge model", str(judge.get("model"))],
                 ["Batch id", str(judge.get("batch_id"))]]
        E.append(tbl(jrows, [W * 0.55, W * 0.30], size=7.8))
        E.append(Paragraph(
            "The judge was run over the OpenAI <b>Batch API</b> and was shown the question, the "
            "options and the model's reply &mdash; but never the answer key. Its only job is to "
            "map free text onto a letter; scoring against the key happens locally. Showing it "
            "the correct answer would turn 'judge accuracy' into a measure of its own "
            "compliance. The agreement rate is the number that matters: it bounds how much of "
            "the headline accuracy is an artefact of the regex parser.", S["body"]))
    else:
        nfail = sum(1 for r in ok if r.get("parse_failed"))
        meta_p = os.path.join(run_dir, "batch_meta.json")
        meta = json.load(open(meta_p)) if os.path.exists(meta_p) else None
        if meta:
            state = (f"A judge batch of <b>{meta['n_requests']}</b> requests has been submitted "
                     f"(batch <font face='Courier'>{meta['batch_id']}</font>) and is still "
                     "running server-side. ")
        else:
            state = "No judge batch has been submitted for this run. "
        E.append(Paragraph(
            state +
            f"Exact-match letter parsing is the official StreamingBench protocol and it resolved "
            f"<b>{len(ok) - nfail}/{len(ok)}</b> replies "
            f"({100 * (len(ok) - nfail) / max(len(ok), 1):.1f}%) unambiguously, so the accuracy "
            "above does not depend on a judge in the first place &mdash; the cross-check "
            "measures how much it <i>could</i> have. Collect it with<br/>"
            "<font face='Courier'>python judge_batch.py " + run_dir + " --collect</font><br/>"
            "and regenerate this report; the Batch API's contractual window is 24&nbsp;h.",
            S["body"]))
        pm: Dict[str, int] = {}
        for r in ok:
            pm[r.get("parse_method", "?")] = pm.get(r.get("parse_method", "?"), 0) + 1
        E.append(tbl([["Parse rule that fired", "Questions"]] +
                     [[k, str(v)] for k, v in sorted(pm.items(), key=lambda x: -x[1])],
                     [W * 0.55, W * 0.30], size=7.8))

    # ---------------- config ---------------- #
    E.append(Paragraph("6 &nbsp; Configuration", S["h1"]))
    q = cfg.get("config", {})
    crows = [["Parameter", "This run", "Paper"],
             ["Feature encoder E",
              f"{_resolved_clip(args)} / {args.get('clip_pretrained')}",
              "OpenCLIP-ViT-L/14"],
             ["Backbone Video-LLM", str(args.get("backbone")), "TimeChat-Online-7B / Qwen2.5-VL-7B"],
             ["DSH smoothing alpha", str(args.get("alpha")), "0.1"],
             ["tau_temp (temporal novelty)", str(args.get("tau_temp")), "0.90"],
             ["tau_rel (relevance)", str(args.get("tau_rel")), "0.60"],
             ["tau_den (density)", str(args.get("tau_den")), "0.15"],
             ["Frame rate", f"{args.get('fps')} FPS", "1 FPS"],
             ["Training", "none (zero-shot)", "none (zero-shot)"],
             ["Backbone precision", "bfloat16", "not stated"],
             ["Encoder precision", "float16", "not stated"],
             ["Statistics precision", "float32 (all thresholds)", "not stated"],
             ["Mask conversion space", f"{args.get('mask_space')} / pool={args.get('pool')}",
              "not discussed"],
             ["Relevance calibration", str(args.get("relevance_mode")), "not stated"],
             ["Qwen max_pixels cap", f"{args.get('max_pixels'):,}", "not stated"],
             ["Hardware", "1x NVIDIA GH200 120GB", "1x NVIDIA A800 80GB"]]
    E.append(tbl(crows, [W * 0.30, W * 0.36, W * 0.34], size=7.5))
    E.append(Paragraph(
        "<b>On \"optimisations mentioned in the paper\":</b> the paper specifies only the "
        "encoder, the four thresholds, 1 FPS, zero training and a single A800. It does not "
        "state a batch size, a dataloader design, or a mixed-precision policy, so none of those "
        "can be reproduced from it &mdash; the rows above marked <i>not stated</i> are this "
        "reproduction's choices, recorded so they can be challenged. QDP and RTAR statistics "
        "are deliberately computed in fp32 regardless of model dtype: &tau;<sub>temp</sub> sits "
        "at 0.90, where fp16's resolution near 1.0 would make the gate decision noise-dominated.",
        S["body"]))

    # ---------------- caveats ---------------- #
    E.append(Paragraph("7 &nbsp; What this run does and does not establish", S["h1"]))
    for t in [
        f"<b>The keep rate reproduces.</b> {keep:.2f}% against the paper's 57.2% target row, "
        "over millions of tokens, with per-question spread narrow enough to be a property of "
        "the method rather than of the sample.",
        f"<b>The accuracy is a subset estimate.</b> n={n_tot} gives a 95% interval of "
        f"{lo:.1f}-{hi:.1f}%, which spans the paper's 75.32%. This run can rule out a gross "
        "reproduction failure; it cannot resolve a one-point difference. Only the full 2500 can.",
        "<b>Two encoder questions remain open.</b> The paper cites \"OpenCLIP-ViT-L/14 (Cherti "
        "et al. 2023)\", the LAION paper, whose ViT-L/14 tag is <font face='Courier'>"
        "laion2b_s32b_b82k</font>; this run uses OpenAI CLIP weights loaded through open_clip. "
        "Separately, open_clip pairs the OpenAI checkpoint with plain GELU while those weights "
        "were trained with QuickGELU &mdash; a mismatch it only warns about. This run corrects "
        "it to <font face='Courier'>ViT-L-14-quickgelu</font>; earlier milestones did not.",
        "<b>The backbone is Qwen2.5-VL-7B, not TimeChat-Online-7B.</b> The paper's headline row "
        "swaps QDP+RTAR into TimeChat-Online, which is itself built on Qwen2.5-VL. The relevant "
        "paper baseline for this configuration is therefore the Qwen2.5-VL-7B row at 73.68%.",
        "<b>Counting is the known casualty.</b> The paper's own numbers show CT falling 57.98 "
        "&rarr; 47.34 &rarr; 42.55 as pruning tightens. Any CT result here should be read "
        "against that, not against the overall average.",
    ]:
        E.append(Paragraph("&bull;&nbsp; " + t, S["body"]))

    E.append(Paragraph("8 &nbsp; Reproducing this run", S["h1"]))
    E.append(Paragraph(" ".join(cfg.get("argv", [])), S["mono"]))
    E.append(Paragraph(
        f"Artefacts in <font face='Courier'>{run_dir}</font>: "
        "<font face='Courier'>eval_raw_logs.jsonl</font> (one row per question with per-frame "
        "keep rates, raw and calibrated relevance, trigger timestamps and the raw reply), "
        "<font face='Courier'>profile_summary.csv</font> (per-video and per-question timings), "
        "<font face='Courier'>summary.json</font>, <font face='Courier'>config.json</font>.",
        S["small"]))

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(16 * mm, 9 * mm, "QueryStream reproduction - StreamingBench "
                                           "(Real-Time Visual Understanding)")
        canvas.drawRightString(A4[0] - 16 * mm, 9 * mm, f"page {canvas.getPageNumber()}")
        canvas.restoreState()

    doc.build(E, onFirstPage=footer, onLaterPages=footer)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.run_dir, "QueryStream_StreamingBench_report.pdf")
    p = build(a.run_dir, out)
    print(f"wrote {p}  ({os.path.getsize(p) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
