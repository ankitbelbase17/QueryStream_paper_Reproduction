"""Figures for the OVO-Bench report.

Palette slots 1-3 from the validated reference set, used unchanged. Every figure
is paired with a table in the PDF, which discharges the relief rule for the aqua
slot on a light surface.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8983", "#e4e3de", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "600",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "legend.frameon": False, "figure.dpi": 200,
})


def _finish(ax):
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_tasks(labels: Sequence[str], series: Dict[str, Sequence[float]],
              path: str) -> str:
    """Per-task accuracy, this run against the two published rows that matter."""
    names = list(series)
    x = range(len(labels))
    w = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(10.0, 3.2))
    for i, nm in enumerate(names):
        ax.bar([xi + i * w - 0.4 + w / 2 for xi in x],
               [0 if v is None else v for v in series[nm]],
               width=w * 0.9, label=nm, color=SERIES[i % len(SERIES)], linewidth=0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("OVO-Bench accuracy per sub-task", loc="left")
    ax.legend(loc="upper right", ncols=3, fontsize=7.5)
    _finish(ax)
    return _save(fig, path)


def fig_keep(rates: Sequence[float], mean_rate: float, path: str) -> str:
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.hist([100 * r for r in rates], bins=30, color=SERIES[0], linewidth=0)
    ax.axvline(100 * mean_rate, color=SERIES[1], linewidth=1.2,
               label=f"aggregate {100 * mean_rate:.1f}%")
    ax.set_xlabel("token keep rate (%)")
    ax.set_ylabel("calls")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("QDP token keep rate", loc="left")
    ax.legend(loc="upper right", fontsize=8)
    _finish(ax)
    return _save(fig, path)


def fig_relevance(raw: Sequence[float], tau_rel: float, path: str) -> str:
    """Per-clip MAXIMUM raw relevance against tau_rel.

    The maximum is the right statistic for the unreachability claim: if even the
    best-matching frame of a clip never reaches tau_rel, no frame in it can.
    """
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.hist(list(raw), bins=36, color=SERIES[0], linewidth=0,
            label="per-clip max cos(q, mean-pool v)")
    lo, hi = (min(raw), max(raw)) if raw else (0.0, 1.0)
    if tau_rel <= hi * 1.6:
        ax.axvline(tau_rel, color=SERIES[1], linewidth=1.2, label=f"tau_rel = {tau_rel:g}")
    else:
        ax.annotate(f"tau_rel = {tau_rel:g} is off-scale to the right",
                    xy=(0.98, 0.9), xycoords="axes fraction", ha="right",
                    fontsize=8, color=SERIES[1])
    ax.set_xlabel("raw cosine similarity")
    ax.set_ylabel("clips")
    ax.set_title(f"Frame relevance, per-clip max {lo:.3f}-{hi:.3f}", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    _finish(ax)
    return _save(fig, path)
