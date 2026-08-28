"""Figures for the OmniPro evaluation report.

Colour comes from the validated reference palette, slots 1-3 (blue / orange /
aqua) -- the documented all-pairs-safe subset, used unchanged and never cycled.
Every figure is paired with a data table in the PDF, which also discharges the
relief rule for the aqua slot (below 3:1 on a light surface).

One axis per chart, thin marks, recessive grid, a legend whenever two or more
series are drawn, and labels only where they carry information.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8983", "#e4e3de", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "600",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "legend.frameon": False, "figure.dpi": 200,
})


def _finish(ax) -> None:
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_subtask(labels: Sequence[str], series: Dict[str, Sequence[Optional[float]]],
                path: str, *, title: str, ylabel: str = "Joint F1 (%)") -> str:
    """Grouped bars: this run against the published rows, per sub-task."""
    names = list(series)
    n = len(names)
    x = range(len(labels))
    w = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(9.4, 3.3))
    for i, nm in enumerate(names):
        vals = [0.0 if v is None else v for v in series[nm]]
        ax.bar([xi + i * w - 0.4 + w / 2 for xi in x], vals, width=w * 0.9,
               label=nm, color=SERIES[i % len(SERIES)], linewidth=0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(loc="upper right", ncols=min(n, 3), fontsize=8)
    _finish(ax)
    return _save(fig, path)


def fig_slice_metrics(slices: Sequence[str], time_f1: Sequence[float],
                      content: Sequence[Optional[float]],
                      joint: Sequence[Optional[float]], path: str) -> str:
    """The three metrics side by side for each headline slice.

    Time F1 is always exact; content and joint may be lower bounds while the
    judge is outstanding, which the caller signals by passing the bound.
    """
    x = range(len(slices))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for i, (nm, vals) in enumerate((("Time F1", time_f1),
                                    ("Content acc.", content),
                                    ("Joint F1", joint))):
        v = [0.0 if u is None else u for u in vals]
        ax.bar([xi + (i - 1) * w for xi in x], v, width=w * 0.9, label=nm,
               color=SERIES[i], linewidth=0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(slices, fontsize=8)
    ax.set_ylabel("percent")
    ax.set_title("Timing, content and joint performance by slice", loc="left")
    ax.legend(loc="upper right", ncols=3, fontsize=8)
    _finish(ax)
    return _save(fig, path)


def fig_emission_balance(emits: Sequence[int], gts: Sequence[int], path: str) -> str:
    """Responses emitted vs ground-truth triggers, per sample.

    The single most explanatory plot for a low precision: everything above the
    diagonal is a sample where RTAR spoke more often than there was anything to
    speak about, and each excess response is a false positive by construction.
    """
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.scatter(gts, emits, s=9, alpha=0.35, color=SERIES[0], linewidths=0)
    hi = max([1] + list(gts) + list(emits))
    ax.plot([0, hi], [0, hi], color=MUTED, linewidth=0.9, linestyle="--",
            label="one response per trigger")
    ax.set_xlabel("ground-truth triggers")
    ax.set_ylabel("responses emitted")
    ax.set_title("Emission balance per sample", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    _finish(ax)
    return _save(fig, path)


def fig_dt_histogram(dts: Sequence[float], tolerance: float, path: str) -> str:
    """|t_response - t_trigger| for every matched pair.

    A distribution piled at the tolerance edge would mean the matches are barely
    matches and a tighter window would erase them; one piled near zero means the
    trigger timing is genuinely sharp.
    """
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.hist(list(dts), bins=max(6, int(tolerance * 4)), range=(0, tolerance),
            color=SERIES[0], linewidth=0)
    ax.set_xlabel(f"|response - trigger| (s), matched pairs only, tol +-{tolerance:g}s")
    ax.set_ylabel("matches")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Timing error of matched responses", loc="left")
    _finish(ax)
    return _save(fig, path)


def fig_keep_rate(rates: Sequence[float], mean_rate: float, path: str) -> str:
    """Per-sample backbone-grid token keep rate."""
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.hist([100 * r for r in rates], bins=28, color=SERIES[0], linewidth=0)
    ax.axvline(100 * mean_rate, color=SERIES[1], linewidth=1.2,
               label=f"aggregate {100 * mean_rate:.1f}%")
    ax.set_xlabel("token keep rate (%)")
    ax.set_ylabel("samples")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("QDP token keep rate", loc="left")
    ax.legend(loc="upper right", fontsize=8)
    _finish(ax)
    return _save(fig, path)


def fig_relevance(raw: Sequence[float], tau_rel: float, path: str) -> str:
    """Raw CLIP relevance against tau_rel -- the unreachability evidence."""
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.hist(list(raw), bins=40, color=SERIES[0], linewidth=0,
            label="observed cos(q, mean-pool v)")
    lo, hi = (min(raw), max(raw)) if raw else (0.0, 1.0)
    if tau_rel <= hi * 1.6:
        ax.axvline(tau_rel, color=SERIES[1], linewidth=1.2,
                   label=f"tau_rel = {tau_rel:g}")
    else:
        ax.annotate(f"tau_rel = {tau_rel:g} lies off-scale to the right",
                    xy=(0.98, 0.86), xycoords="axes fraction", ha="right",
                    fontsize=8, color=SERIES[1])
    ax.set_xlabel("raw cosine similarity")
    ax.set_ylabel("frames")
    ax.set_title(f"Frame relevance, observed range {lo:.3f}-{hi:.3f}", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    _finish(ax)
    return _save(fig, path)


def fig_horizon(buckets: Sequence[str], recall: Sequence[float],
                counts: Sequence[int], path: str) -> str:
    """Recall by where the trigger sits on the timeline (paper Figure 3).

    The paper's headline long-horizon finding is that every model retains only
    ~37% of its short-term performance on triggers past 180 s, and that online
    streaming models collapse almost completely. This is the same cut.
    """
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.bar(range(len(buckets)), [100 * r for r in recall], width=0.55,
           color=SERIES[0], linewidth=0)
    for i, (r, c) in enumerate(zip(recall, counts)):
        ax.annotate(f"n={c}", (i, 100 * r), ha="center", va="bottom",
                    fontsize=8, color=MUTED)
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels(buckets, fontsize=8)
    ax.set_ylabel("time recall (%)")
    ax.set_title("Long-horizon behaviour: recall by trigger time", loc="left")
    _finish(ax)
    return _save(fig, path)


def fig_gap_sweep(gaps, precision, recall, f1, path: str) -> str:
    """Time P / R / F1 against the Appendix A.4 debounce gap.

    A flat precision line across the sweep is the informative shape: it means
    responses are not merely too dense around the right instants -- collapsing
    them cannot raise the fraction that land on a trigger -- so the timing error
    is in *where* the gate fires, not in how often.
    """
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    for i, (nm, v) in enumerate((("Time precision", precision),
                                 ("Time recall", recall),
                                 ("Time F1", f1))):
        ax.plot(list(gaps), [100 * x for x in v], marker="o", markersize=3,
                linewidth=1.4, color=SERIES[i], label=nm)
    ax.set_xlabel("event debounce gap (s)")
    ax.set_ylabel("percent")
    ax.set_ylim(bottom=0)
    ax.set_title("Sensitivity to the unspecified Appendix A.4 gap", loc="left")
    ax.legend(loc="upper right", ncols=3, fontsize=8)
    _finish(ax)
    return _save(fig, path)
