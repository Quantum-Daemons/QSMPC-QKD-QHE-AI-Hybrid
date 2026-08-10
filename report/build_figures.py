#!/usr/bin/env python
"""Figures for the technical report, drawn from live artefacts and measurements."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "report" / "generated"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 200, "savefig.bbox": "tight"})

from qsh.models import metrics, registry  # noqa: E402

M = json.loads((REPO / "models" / "metrics.json").read_text())

# --- Fig 1: achieved vs published baseline, per model -------------------------------------------
keys, got, base = [], [], []
for s in sorted(registry._SPECS, key=lambda x: x.key):
    a = M.get(s.key) or {}
    v = a.get(s.metric.name, a.get("student_metric"))
    if not isinstance(v, float) or s.metric.baseline is None:
        continue
    keys.append(s.key.replace("_", "\n")); got.append(v); base.append(s.metric.baseline)
fig, ax = plt.subplots(figsize=(6.6, 2.6))
x = np.arange(len(keys))
ax.bar(x - 0.2, got, 0.4, label="measured here", color="#2b6cb0")
ax.bar(x + 0.2, base, 0.4, label="published baseline", color="#a0aec0")
ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=5.5)
ax.set_ylabel("headline metric"); ax.set_ylim(0, 1.05); ax.legend(fontsize=7, loc="upper left")
ax.set_title("Measured against the published baseline each model is compared to", fontsize=8)
fig.savefig(OUT / "fig_baselines.pdf"); plt.close(fig)

# --- Fig 2: the decision curve, which is why net benefit replaced Youden's J ---------------------
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
ax.plot(grid[k], nb_model[k], "o", ms=4, color="#2b6cb0")
ax.annotate("max margin", (grid[k], nb_model[k]), textcoords="offset points",
            xytext=(6, 6), fontsize=6)
ax.set_xlabel("threshold probability $p$"); ax.set_ylabel("net benefit")
ax.set_title("Decision curve: all three lines are functions of $p$", fontsize=7.5)
ax.legend(fontsize=6)
fig.savefig(OUT / "fig_decision_curve.pdf"); plt.close(fig)

# --- Fig 3: permutation null - why "near 1.0" is the wrong pass condition ------------------------
fig, ax = plt.subplots(figsize=(3.3, 2.3))
p = np.linspace(0.001, 0.5, 400)
ax.plot(p, p**2 + (1 - p)**2, color="#2b6cb0", lw=1.6)
for lbl, pv, col in (("BreastMNIST\n0.269", 0.2692, "#2f855a"),
                     ("OPS-SAT\n0.214", 0.2136, "#b7791f"),
                     ("AMLworld\n0.001", 1/981, "#c53030")):
    ax.plot(pv, pv**2 + (1-pv)**2, "o", ms=4, color=col)
    ax.annotate(lbl, (pv, pv**2 + (1-pv)**2), textcoords="offset points",
                xytext=(4, -12 if pv > 0.2 else 4), fontsize=5.5, color=col)
ax.set_xlabel("positive rate $p$")
ax.set_ylabel("agreement under a random permutation")
ax.set_title(r"A shuffled harvest scores $p^2+(1-p)^2$", fontsize=7.5)
ax.set_ylim(0.45, 1.02)
fig.savefig(OUT / "fig_permutation_null.pdf"); plt.close(fig)

print("  wrote fig_baselines.pdf, fig_decision_curve.pdf, fig_permutation_null.pdf")
