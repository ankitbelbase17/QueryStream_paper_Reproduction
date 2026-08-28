"""Figures for the StreamingBench evaluation report.

Colour comes from the validated reference palette, slots 1-3 (blue / orange /
aqua) -- the documented all-pairs-safe subset, used unchanged and never cycled.
Every figure is paired with a data table in the PDF, which also discharges the
relief rule for the aqua slot (below 3:1 on a light surface).

Design constraints applied throughout: one axis per chart, thin marks, recessive
grid and axes, a legend whenever two or more series are drawn, and labels only
where they carry information rather than on every mark.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Reference palette, light mode. Slots assigned by entity, fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8983"
GRID = "#e4e3de"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "600",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "legend.frameon": False,
    "figure.dpi": 200,
})


def _finish(ax) -> None:
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def fig_subtask_accuracy(
    codes: Sequence[str],
    ours: Sequence[float],
    ours_n: Sequence[int],
    paper_qs: Dict[str, float],
    paper_qwen: Dict[str, float],
    path: str,
    label: str = "This run",
) -> str:
    """Grouped bars: our per-subtask accuracy against the two paper rows."""
    import numpy as np

    x = np.arange(len(codes))
    w = 0.27
    fig, ax = plt.subplots(figsize=(9.2, 3.5))

    ax.bar(x - w, ours, w * 0.88, label=label, color=SERIES[0])
    ax.bar(x, [paper_qs.get(c, 0) for c in codes], w * 0.88,
           label="Paper: QueryStream-7B @57.2% keep", color=SERIES[1])
    ax.bar(x + w, [paper_qwen.get(c, 0) for c in codes], w * 0.88,
           label="Paper: Qwen2.5-VL-7B, full tokens", color=SERIES[2])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\nn={n}" for c, n in zip(codes, ours_n)], fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.19), ncol=3, fontsize=8.5)
    _finish(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_keep_rate(rates: Sequence[float], paper_ref: float, path: str) -> str:
    """Where the per-question Token Keep Rate actually lands. One series."""
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    ax.hist([100 * r for r in rates], bins=28, color=SERIES[0], edgecolor=SURFACE,
            linewidth=0.5)
    ax.axvline(paper_ref, color=INK, linewidth=1.4, linestyle=(0, (4, 2)))
    ax.annotate(f"paper {paper_ref:.1f}%", xy=(paper_ref, ax.get_ylim()[1] * 0.92),
                xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel("Token keep rate per question (%)")
    ax.set_ylabel("Questions")
    ax.yaxis.get_major_locator().set_params(integer=True)   # counts are integers
    _finish(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_relevance(raws: Sequence[float], tau_rel: float, path: str) -> str:
    """The tau_rel reachability problem, drawn to scale. One series."""
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    ax.hist(raws, bins=60, color=SERIES[0], edgecolor=SURFACE, linewidth=0.4)
    ax.axvline(tau_rel, color="#e34948", linewidth=1.6)
    ax.annotate(f"$\\tau_{{rel}}$ = {tau_rel:.2f}\nnever reached",
                xy=(tau_rel, ax.get_ylim()[1] * 0.72), xytext=(-6, 0),
                textcoords="offset points", ha="right", fontsize=8.5, color="#e34948")
    ax.set_xlim(0, max(0.7, tau_rel + 0.05))
    ax.set_xlabel(r"raw  $\cos(q,\ \bar{v}_t)$  per frame")
    ax.set_ylabel("Frames")
    _finish(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_latency(stages: Sequence[str], ms: Sequence[float], path: str) -> str:
    """Per-frame cost by stage. One series, direct-labelled."""
    import numpy as np

    y = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.barh(y, ms, 0.6, color=SERIES[0])
    ax.set_yticks(y)
    ax.set_yticklabels(stages, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("ms per frame")
    ax.set_xlim(0, max(ms) * 1.28 if ms else 1)
    for yi, v in zip(y, ms):
        ax.text(v + max(ms) * 0.02, yi, f"{v:.2f}", va="center", fontsize=8, color=INK_2)
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    _finish(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


__all__ = ["fig_subtask_accuracy", "fig_keep_rate", "fig_relevance", "fig_latency",
           "SERIES", "INK", "INK_2", "MUTED", "GRID", "SURFACE"]
