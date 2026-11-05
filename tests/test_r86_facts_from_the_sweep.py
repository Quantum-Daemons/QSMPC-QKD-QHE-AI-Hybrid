"""Corrections from the fact-check sweep of 2026-11-03, where a test can hold them.

WHY THIS EXISTS (Round 86). Most of the sweep's findings are sentences, and their sources change. These are the ones
the code itself can get wrong again: values a published YANG module does not allow, a confidence bound that says it is
SP 800-90B's and was not, an equation pinned to a version of the paper that does not hold it, and a page repeating a
position BSI has moved on from.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def test_the_sdn_model_emits_only_what_the_yang_allows() -> None:
    """etsi-qkdn-types: link-status-types and iface-status-types (GS QKD 015 V2.1.1, revision 2022-01-30)."""
    from qsh.qkd.sdn_controller import QkdInterface, QkdLink

    assert QkdInterface(qkdi_id="i").qkdi_status in {"ENABLED", "DISABLED", "FAILED"}
    link = QkdLink(qkdl_id="l", qkdl_local_qkdn_id="a", qkdl_remote_qkdn_id="b")
    assert link.qkdl_status in {"ACTIVE", "PASSIVE", "PENDING", "OFF"}


def test_the_mcv_bound_is_sp_800_90b_section_6_3_1() -> None:
    """p_u = min(1, p + 2.576 sqrt(p (1 - p) / (L - 1))); it divided by L and called the Z one-sided."""
    from qsh.qrng.health import _MCV_Z, _min_entropy_mcv

    assert math.isclose(_MCV_Z, 2.5758293035489004)
    bits = np.array([1] * 60 + [0] * 40, dtype=np.uint8)
    p = 0.6
    expected = -math.log2(p + 2.5758293035489004 * math.sqrt(p * (1 - p) / 99))
    assert math.isclose(_min_entropy_mcv(bits), expected, rel_tol=1e-12)
    assert _min_entropy_mcv(np.array([1], dtype=np.uint8)) == 0.0, "one sample must not divide by zero"


def test_the_browser_bound_says_the_same() -> None:
    src = (REPO / "web/src/lib/clientQRNG.ts").read_text()
    assert "Math.max(n - 1, 1)" in src and "one-sided" not in src


@pytest.mark.parametrize(("path", "old"), [
    ("src/qsh/qkd/decoy_reference.py", "quant-ph/0503005v2."),
    ("src/qsh/qkd/qsdc.py", "Curras-Lorenzo"),
    ("web/src/pages/Overview.tsx", "the TLS hybrids BSI\n        intends to recommend"),
    ("docs/pqc-roadmap.md", "intends to recommend SecP256r1MLKEM768"),
    ("docs/pqc-roadmap.md", "sets PQC migration at **2035**"),
    ("docs/architecture.md", "1.5x to 10.5x\nlower latency than Orion."),
    ("docs/threat-model.md", "a refined polynomial noise"),
    ("references/REFERENCES.md", "node-seal/blob/main/BENCHMARK.md"),
    ("references/encrypted-inference-performance.md", "node-seal/blob/main/BENCHMARK.md"),
    ("DATA_LICENSES.md", "Alisah-Ozcan/HEonGPU | GPU CKKS"),
    ("docs/development.md", "AlmaLinux 9.7, i5-13600K"),
])
def test_a_corrected_sentence_does_not_come_back(path: str, old: str) -> None:
    assert old not in (REPO / path).read_text(encoding="utf-8"), f"{path} says {old!r} again"


def test_a_document_that_calls_a_figure_current_gives_the_artifacts() -> None:
    """Two references still called 0.883721 the shipped medical accuracy, two rounds after a retrain moved it."""
    import json

    shipped = json.loads((REPO / "models/qcnn_medical/weights.json").read_text(encoding="utf-8"))
    for name in ("references/qcnn-vs-published-wdbc.md", "references/qcnn-entanglement-ablation.md"):
        text = (REPO / name).read_text(encoding="utf-8")
        assert f"**{shipped['test_accuracy']:.6f}**" in text, name
        assert f"**{shipped['test_minority_f1']:.6f}**" in text, name


def test_the_xss_figure_is_the_f1_in_the_papers_table() -> None:
    """Engelen et al., WTMC 2021, Table II: Web Attack XSS P 0.52, R 0.11, F1 0.18; final F1 0.79. 0.11 is its recall."""
    text = (REPO / "src/qsh/models/metrics.py").read_text(encoding="utf-8")
    assert "Web Attack XSS 0.18 -> 0.79" in text and "XSS 0.11" not in text


@pytest.mark.parametrize(
    "phrase",
    [
        "regulator-facing",  # the key is per run with no trust anchor, and the backend path keeps only a prefix
        "Confirms nothing larger exists",  # a survey cannot show that
        "Q-Day is being pulled forward",  # the cards under the heading say estimates are falling
        "Resource estimates collapsing",
        "a measured AI optimizer",  # the page runs coordinate ascent, and the deployment a regression surrogate
        "Federated SHAP",  # nothing in `ai/federated.py` computes E[x] by secure aggregation
    ],
)
def test_an_overstatement_the_readers_found_stays_gone(phrase: str) -> None:
    """Records quote what they correct, so the register, the log and the checklist are not searched."""
    roots = [REPO / "src", REPO / "web" / "src", REPO / "references", REPO / "README.md"]
    roots += [p for p in (REPO / "docs").glob("*.md")
              if p.name not in {"deliverable-verification.md", "verification-log.md", "browser-verification-checklist.md"}]
    hits = []
    for root in roots:
        files = [root] if root.is_file() else [p for p in root.rglob("*") if p.suffix in {".py", ".ts", ".tsx", ".md"}]
        hits += [str(p.relative_to(REPO)) for p in files if phrase in p.read_text(encoding="utf-8", errors="ignore")]
    assert not hits, f"{phrase!r} is back in {hits}"
