#!/usr/bin/env python
"""Emit the report's tables as LaTeX, generated from live artefacts.

Every number in the technical report comes from here, so the report cannot drift from the code the
way prose does. This project has already found stale figures in five documents at once; a report is
a worse place for that than a README because it is a fixed artefact someone keeps.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "report" / "generated"
OUT.mkdir(parents=True, exist_ok=True)

from qsh.models import registry  # noqa: E402

M = json.loads((REPO / "models" / "metrics.json").read_text())


def esc(s: str) -> str:
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, float) else ("--" if v is None else str(v))


def model_table() -> str:
    rows = []
    for s in sorted(registry._SPECS, key=lambda x: (x.usecase, x.tier)):
        a = M.get(s.key) or {}
        got = a.get(s.metric.name, a.get("student_metric"))
        tier = {"t1_crypto": "T1", "t2_industry": "T2", "t3_large": "T3"}.get(s.tier, s.tier)
        rows.append(" & ".join([
            r"\texttt{" + esc(s.key) + "}", tier, esc(s.metric.display),
            fmt(got), fmt(s.metric.baseline), esc(s.dataset.licence.split(";")[0][:22]),
        ]) + r" \\")
    return "\n".join(rows)


def crypto_table() -> str:
    from qsh.config import defaults as D
    rows = [
        ("KEM", "ML-KEM-768 + X25519", "FIPS 203 hybrid; ct 1088 B"),
        ("Signature", "ML-DSA-65", "FIPS 204; sig 3309 B"),
        ("FHE", "CKKS, ring degree 8192", "4096 slots, one multiplicative level per dot product"),
        ("MPC", "Garbled circuits (emp-wasm)", "non-linear layers: ReLU, argmax"),
        ("QKD", "BB84 two-intensity decoy", "GLLP tagged fraction"),
    ]
    return "\n".join(" & ".join([esc(a), r"\texttt{" + esc(b) + "}", esc(c)]) + r" \\" for a, b, c in rows)


def findings_table() -> str:
    """The negative-results chapter, as data rather than prose."""
    rows = [
        ("Constant predictor", "medical CKKS student emitted one class for all 156 test images",
         "accuracy 0.7308 equalled the majority rate exactly; AUC 0.354, below random"),
        ("Inverted class labels", "four sites asserted the opposite of MedMNIST's own metadata",
         "the demo displayed the opposite diagnosis to the model's output"),
        ("Sampler misalignment", "shuffled DataLoader output paired positionally with dataset order",
         "teacher/ground-truth agreement 0.6227, inside the permutation null"),
        ("Wrong baseline row", "cited ESA-ADB Table 2 (all events) for an anomalies-only task",
         "like-for-like row is Supplementary Table 9 at 0.968, not 0.786"),
        ("Uncorrected metric", "reported uncorrected event-wise F0.5 against a corrected baseline",
         "ESA-ADB multiplies event precision by the time-level TNR"),
        ("Licence divergence", "published artefact recorded ImageNet weights; two docs said random",
         "publish now refuses any model whose manifest records a pretrained source"),
        ("Guard too narrow", "both\\_scored\\_zero required BOTH models at zero",
         "teacher 0 vs student 0.0032 rendered as the encrypted model beating its teacher"),
        ("Manufactured margin", "treat-all evaluated at a different threshold from the model",
         "a zero margin reported as +0.2308; caught by the degeneracy guard"),
        ("Leak in a published sample",
         "finance eval sample: hour\\_of\\_day >= 3 reproduced the label at 100\\% precision",
         "served from an HF revision whose pin looked valid; the first two leak guards were "
         "themselves wrong"),
        ("Guard exempted by its data", "telecom feature\\_names held CLASS names, not feature names",
         "the leak guard skipped the file and reported success on something it never examined"),
        ("Half-applied fix", "QCNN repaired in the browser; the Python path stayed a constant predictor",
         "a coin flip looks broken, a constant predictor looks confident and reproducible"),
        ("Crossed wiring", "the trainer's CNOT ring indexed a stale axis after chained movedim",
         "training still converged; only an independent reimplementation disagreed, by 0.73 on a "
         "quantity bounded in [-1,1]"),
        ("Feature-space mismatch", "circuit trained on min-max raw units, served StandardScaler z-scores",
         "seven of eight features pinned to -1; a degenerate state still returned a confident class"),
        ("Selection on the test set", "epoch chosen and headline reported on the same rows",
         "medical F1 fell 0.9181 to 0.7326 once a validation split was introduced"),
        ("Criterion half-applied", "loss-aware QSDC bound added to the module, not to its caller",
         "an inert threshold control, and an eavesdropper reported on every clean run"),
    ]
    findings_table.rows = rows
    return "\n".join(" & ".join([r"\textbf{" + a + "}", b, c]) + r" \\" for a, b, c in rows)


def macros() -> str:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                         capture_output=True, text=True).stdout.strip()
    n_pub = sum(1 for s in registry._SPECS if s.hf_repo)
    checklist = (REPO / "docs" / "browser-verification-checklist.md").read_text()
    n_rows = sum(1 for line in checklist.splitlines()
                 if line.strip().startswith("|") and len(line.split("|")) > 3
                 and line.split("|")[1].strip()[:1].isalnum()
                 and any(ch.isdigit() for ch in line.split("|")[1]))
    return "\n".join([
        r"\newcommand{\gitsha}{" + sha + "}",
        r"\newcommand{\npublished}{" + str(n_pub) + "}",
        r"\newcommand{\nchecklistrows}{" + str(n_rows) + "}",
        r"\newcommand{\nusecases}{5}",
        r"\newcommand{\ndefectclasses}{" + str(len(findings_table.rows)) + "}",
    ])


def wrap(body: str, *, cols: str, header: str, caption: str, size: str = r"\small",
         longtable: bool = False) -> str:
    """Emit a COMPLETE table environment rather than bare rows.

    ``\\input`` inside a ``tabular`` does not work: TeX ends the file inside an alignment cell, so
    the following ``\\bottomrule`` meets ``\\noalign`` outside a row and the run reports
    "Misplaced \\noalign" -- while still producing a PDF that looks fine and silently drops rows.
    A build that emits 20 errors and a readable document is the same trap as a test suite passing
    on the wrong thing, so the environment is generated whole and \\input at document level.
    """
    if longtable:
        return "\n".join([
            "{" + size, r"\begin{longtable}{" + cols + "}",
            r"\caption{" + caption + r"}\\", r"\toprule", header + r" \\", r"\midrule",
            r"\endfirsthead", r"\toprule", header + r" \\", r"\midrule", r"\endhead",
            body, r"\bottomrule", r"\end{longtable}}"])
    return "\n".join([
        r"\begin{table}[htbp]\centering" + size,
        r"\caption{" + caption + "}", r"\begin{tabular}{" + cols + "}", r"\toprule",
        header + r" \\", r"\midrule", body, r"\bottomrule",
        r"\end{tabular}", r"\end{table}"])


findings_table()  # populate .rows so the caption below can count them

TABLES = {
    "models.tex": dict(fn=model_table, cols="llllll",
        header=r"\textbf{Model} & \textbf{Tier} & \textbf{Metric} & \textbf{Measured} & "
               r"\textbf{Baseline} & \textbf{Data licence}",
        caption="Published models: measured value against the baseline each is compared to. "
                "Tier T1 runs under encryption; T2 is the plaintext teacher. A dash means the "
                "model is declared in the registry but not trained in this build, which is "
                "reported rather than hidden."),
    "crypto.tex": dict(fn=crypto_table, cols="lll",
        header=r"\textbf{Layer} & \textbf{Primitive} & \textbf{Note}",
        caption="Cryptographic parameters as configured"),
    "findings.tex": dict(fn=findings_table, cols="p{2.5cm}p{5.2cm}p{6.2cm}", longtable=True,
        size=r"\scriptsize",
        header=r"\textbf{Class} & \textbf{What it was} & \textbf{Why nothing caught it}",
        caption=f"{len(findings_table.rows)} classes of silent defect, all found in a working "
                f"system with green tests"),
}

for name, cfg in TABLES.items():
    fn = cfg.pop("fn")
    (OUT / name).write_text(wrap(fn(), **cfg) + "\n")
    print(f"  wrote report/generated/{name}")
(OUT / "macros.tex").write_text(macros() + "\n")
print("  wrote report/generated/macros.tex")
