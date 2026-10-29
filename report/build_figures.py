#!/usr/bin/env python
"""Figures for the technical report, drawn from live artefacts and measurements.

Run by `make report` since Round 80, before the PDF is built. Until then nothing ran it but a hand,
and `fig_baselines.pdf` was last drawn on 13 August: it showed the satellite forecaster at 0.45
where `models/metrics.json` has said 0.0 since. Every figure is also checked for overlapping text
before it is written (`check_layout`), because two labels on top of each other and a label drawn
across a curve were in the published PDF and no gate looked at a figure.

`--check` draws and checks the figures without writing them; the test suite runs it that way.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "report" / "generated"
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 200, "savefig.bbox": "tight"})

from qsh.models import metrics, registry  # noqa: E402

M = json.loads((REPO / "models" / "metrics.json").read_text())
CHECK_ONLY = "--check" in sys.argv


def _texts(fig) -> list[Text]:
    """Every visible, non-empty piece of text a reader sees: titles, axis labels, tick labels,
    annotations and legend entries."""
    out: list[Text] = []
    for ax in fig.axes:
        out += [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts]
        # Only the tick labels drawn: matplotlib keeps labels for ticks outside the view limits,
        # which are never shown and would otherwise be reported as overlapping the axis title.
        for axis, (lo, hi), i in ((ax.xaxis, sorted(ax.get_xlim()), 0), (ax.yaxis, sorted(ax.get_ylim()), 1)):
            eps = (hi - lo) * 1e-6
            out += [t for t in axis.get_ticklabels() if lo - eps <= t.get_position()[i] <= hi + eps]
        if ax.get_legend() is not None:
            out += list(ax.get_legend().get_texts())
    return [t for t in out if t.get_visible() and t.get_text().strip()]


def check_layout(fig, name: str) -> list[str]:
    """Text that overlaps other text, or a data line that runs through a label.

    Measured on the renderer, in display units, so it sees what the PDF shows: a bounding box per
    text, and each data line sampled finely along its path. Grid lines are not data and are skipped.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [(t, t.get_window_extent(r).shrunk(0.96, 0.9)) for t in _texts(fig)]
    problems = []
    for i, (a, ba) in enumerate(boxes):
        for b, bb in boxes[i + 1:]:
            if ba.overlaps(bb):
                problems.append(f"{name}: text {a.get_text()!r} overlaps text {b.get_text()!r}")
    for ax in fig.axes:
        # Every drawn line and marker, the annotated points included; grid lines are not data.
        grid = set(ax.get_xgridlines()) | set(ax.get_ygridlines())
        lines = [ln for ln in ax.get_lines() if ln.get_visible() and ln not in grid]
        inside = [(t, bx) for t, bx in boxes if t in ax.texts or t is ax.title]
        legend = ax.get_legend()
        if legend is not None:
            inside.append((Text(text="legend"), legend.get_window_extent(r)))
        for ln in lines:
            xy = ln.get_transform().transform(np.column_stack(ln.get_data()))
            if len(xy) > 1:          # densify, so a segment between two far points still counts
                seg = np.linspace(0, 1, 30)[:, None]
                xy = np.vstack([a + seg * (b - a) for a, b in zip(xy[:-1], xy[1:], strict=True)])
            for t, bx in inside:
                if any(bx.contains(x, y) for x, y in xy):
                    problems.append(f"{name}: line {ln.get_label()!r} runs through {t.get_text()!r}")
    return problems


def finish(fig, name: str) -> list[str]:
    problems = check_layout(fig, name)
    if not CHECK_ONLY:
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / name)
    plt.close(fig)
    return problems


problems: list[str] = []

# --- The achieved figure against the published baseline, per model -------------------------------
keys, got, base = [], [], []
for s in sorted(registry._SPECS, key=lambda x: x.key):
    a = M.get(s.key) or {}
    v = a.get(s.metric.name, a.get("student_metric"))
    if not isinstance(v, float) or s.metric.baseline is None:
        continue
    # Two short lines: the use case, then the model. The full key split at every underscore ran to
    # three lines and the neighbouring labels collided.
    usecase, model = s.key.split(".", 1)
    keys.append(f"{usecase.split('_')[0]}\n{model}")
    got.append(v)
    base.append(s.metric.baseline)
fig, ax = plt.subplots(figsize=(6.6, 2.7))
x = np.arange(len(keys))
ax.bar(x - 0.2, got, 0.4, label="measured here", color="#2b6cb0")
ax.bar(x + 0.2, base, 0.4, label="published baseline", color="#a0aec0")
# A measured zero is a result, not a missing bar: say so where the bar would be.
for xi, v in zip(x, got, strict=True):
    if v < 0.02:
        ax.annotate(f"{v:.2f}", (xi - 0.2, 0), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=5.5, color="#2b6cb0")
ax.set_xticks(x)
ax.set_xticklabels(keys, fontsize=5.5)
ax.set_ylabel("headline metric")
ax.set_ylim(0, 1.25)
ax.legend(fontsize=6.5, loc="upper left", ncol=2)
ax.set_title("Measured against the published baseline each model is compared to", fontsize=8)
problems += finish(fig, "fig_baselines.pdf")

# --- The decision curve, which is why net benefit replaced Youden's J ----------------------------
rng = np.random.default_rng(0)
n, prev = 400, 0.27
y = (rng.random(n) < prev).astype(int)
score = np.clip(np.where(y == 1, rng.normal(0.62, 0.18, n), rng.normal(0.34, 0.18, n)), 0, 1)
grid = np.linspace(0.02, 0.6, 120)
nb_model = [metrics.net_benefit(y, score, q) for q in grid]
nb_all = [metrics.net_benefit(y, np.ones(n), q) for q in grid]
fig, ax = plt.subplots(figsize=(3.3, 2.3))
ax.plot(grid, nb_model, color="#2b6cb0", lw=1.6, label="model")
ax.plot(grid, nb_all, color="#c05621", lw=1.2, ls="--", label="treat all")
ax.axhline(0, color="#4a5568", lw=1.0, ls=":", label="treat none")
k = int(np.argmax(np.array(nb_model) - np.maximum(np.array(nb_all), 0)))
ax.plot(grid[k], nb_model[k], "o", ms=4, color="#2b6cb0", label="_nolegend_")
ax.set_ylim(top=0.42)
ax.annotate("max margin", (grid[k], nb_model[k]), textcoords="offset points", xytext=(-10, 14),
            ha="right", fontsize=6, arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "#4a5568"})
ax.set_xlabel("threshold probability $p$")
ax.set_ylabel("net benefit")
ax.set_title("Decision curve: all three lines are functions of $p$", fontsize=7.5)
ax.legend(fontsize=6, loc="lower left")
problems += finish(fig, "fig_decision_curve.pdf")

# --- The permutation null: why "near 1.0" is the wrong pass condition ----------------------------
fig, ax = plt.subplots(figsize=(3.3, 2.3))
p = np.linspace(0.001, 0.5, 400)
ax.plot(p, p**2 + (1 - p)**2, color="#2b6cb0", lw=1.6, label="p^2 + (1-p)^2")
# Each label with a leader line, clear of the falling curve: OPS-SAT and AMLworld to the right, where
# the curve has already dropped below them, BreastMNIST below and to the left, under it. AMLworld sat
# under the title and the other two on the curve itself in the published figure.
for lbl, pv, col, off, ha in (("BreastMNIST 0.269", 0.2692, "#2f855a", (-4, -24), "right"),
                              ("OPS-SAT 0.214", 0.2136, "#b7791f", (10, 12), "left"),
                              ("AMLworld 0.001", 1/981, "#c53030", (12, 2), "left")):
    yv = pv**2 + (1 - pv)**2
    ax.plot(pv, yv, "o", ms=4, color=col, label="_nolegend_")
    ax.annotate(lbl, (pv, yv), textcoords="offset points", xytext=off, fontsize=5.5, color=col,
                ha=ha, va="center", arrowprops={"arrowstyle": "-", "lw": 0.6, "color": col})
ax.set_xlabel("positive rate $p$")
ax.set_ylabel("agreement under a random permutation")
ax.set_title(r"A shuffled harvest scores $p^2+(1-p)^2$", fontsize=7.5)
ax.set_ylim(0.45, 1.06)
problems += finish(fig, "fig_permutation_null.pdf")

if problems:
    print("\n".join(problems), file=sys.stderr)
    raise SystemExit(f"{len(problems)} layout problem(s) in the report's figures; nothing is right "
                     "until each is fixed")
print("  figures checked: no overlapping text" + ("" if CHECK_ONLY else
      "; wrote fig_baselines.pdf, fig_decision_curve.pdf, fig_permutation_null.pdf"))
