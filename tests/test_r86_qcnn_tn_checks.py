"""The tensor-network checks of the medical, telecom and satellite QCNNs say what the circuits and their comparators do.

WHY THIS EXISTS (Round 86, batch 6; the owner's choice of 2026-11-26; references/qcnn-tensor-network-checks-preregistration.md)

Each trained QCNN outside finance now carries a check of how much entanglement it builds and a matrix-product-state
classifier on its inputs. The first run counted single-precision round-off as Schmidt rank, so every point, the
encoding's product state included, read the bound; the addendum records that and the trace runs in double precision.
This file holds the record to the properties that would have caught it, and the page to the record.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
RECORD = json.loads((REPO / "docs" / "measurements" / "qcnn-tn-checks-2026-11-26.json").read_text())
SITE = json.loads((REPO / "web" / "public" / "data" / "qcnn_tn_checks.json").read_text())


def _module(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_record_is_under_the_committed_preregistration() -> None:
    pre = RECORD["preregistration"]
    assert hashlib.sha256((REPO / pre["file"]).read_bytes()).hexdigest() == pre["sha256"]
    assert pre["sha256"] == _module("scripts/train/qcnn_tn_checks.py", "qcnn_tn_checks").PREREG_SHA256
    assert "Addendum, 2026-11-26" in (REPO / pre["file"]).read_text()
    assert set(RECORD["artifacts"]) == {"medical", "telecom", "telecom_q7", "satellite_opssat", "satellite_esa"}


@pytest.mark.parametrize("key", sorted(RECORD["artifacts"]))
def test_each_trace_starts_from_a_product_state_and_stays_under_the_bound(key: str) -> None:
    sc = RECORD["artifacts"][key]["schmidt"]
    assert sc["by_point"]["encode"] == 1, "the encoding is a product state; a higher rank is round-off"
    assert sc["largest_rank"] == max(sc["by_point"].values()) <= sc["bound"] == 2 ** (sc["wires"] // 2)
    assert any(p.startswith("conv") for p in sc["by_point"]) and any(p.startswith("pool") for p in sc["by_point"]), (
        "the trace skipped the convolution or pooling layers")
    if "recomputed" in sc:
        assert sc["recomputed"] == sc["recorded"], "the circuit traced is not the circuit shipped"


def test_the_single_precision_run_is_kept_as_invalid_and_read_by_nothing() -> None:
    inv = RECORD["invalid_single_precision_run"]
    assert "round-off" in inv["note"] and inv["artifacts"]["medical"]["by_point"]["encode"] > 1
    assert "invalid_single_precision_run" not in json.dumps(SITE)


def test_the_page_summary_is_the_records() -> None:
    for key, r in RECORD["artifacts"].items():
        s, h = SITE["artifacts"][key], r["headline"]
        assert (s["largest_schmidt_rank"], s["bound"]) == (r["schmidt"]["largest_rank"], r["schmidt"]["bound"])
        assert (s["qcnn"], s["logistic_regression"], s["mps"]) == (r["qcnn"][h], r["logistic_regression"][h], r["mps"][h])
        assert s["mps_average_precision"] == r["mps"].get("test_average_precision")
    assert "No quantum advantage is claimed" in SITE["statement"] and "Bermejo" in SITE["statement"]


def test_the_rank_counts_states_not_round_off() -> None:
    f = _module("scripts/train/finance_qcnn_comparison.py", "finance_qcnn_comparison_tn")
    product = np.ones(256, dtype=np.complex128) / 16
    bell = np.zeros(256, dtype=np.complex128)
    bell[0] = bell[(1 << 0) | (1 << 7)] = 1 / np.sqrt(2)
    assert f.schmidt_ranks(product[None]).max() == 1
    assert f.schmidt_ranks(bell[None])[0].tolist() == [2] * 7
    noisy = (product + 1e-7 * np.random.default_rng(0).standard_normal(256)).astype(np.complex64)
    assert f.schmidt_ranks(noisy[None].astype(np.complex128)).max() > 1, (
        "single-precision noise no longer reads as rank, so the addendum's reason needs re-reading")


def test_double_precision_is_an_option_and_single_stays_the_default() -> None:
    torch = pytest.importorskip("torch")
    import inspect
    import sys

    sys.path.insert(0, str(REPO / "scripts" / "train"))
    import qcnn_variational as Q

    assert inspect.signature(Q.circuit).parameters["complex_dtype"].default is torch.complex64
    w = json.loads((REPO / "models" / "qcnn_medical" / "weights.json").read_text())
    x = np.random.default_rng(1).uniform(-0.5, 0.5, (4, 8))
    lo = Q.circuit(torch.tensor(x, dtype=torch.float32), torch.tensor(w["theta"], dtype=torch.float32), 8, layers=3)
    hi = Q.circuit(torch.tensor(x, dtype=torch.float64), torch.tensor(w["theta"], dtype=torch.float64), 8, layers=3,
                   complex_dtype=torch.complex128)
    assert lo.dtype == torch.float32 and np.allclose(lo.numpy(), hi.numpy(), atol=1e-6)


def test_the_cards_show_the_check() -> None:
    web = REPO / "web" / "src" / "components"
    tn = (web / "QcnnTnCheck.tsx").read_text()
    assert 'qubits === 7 ? "telecom_q7" : "telecom"' in tn and 'usecase === "medical_fl") return "medical"' in tn
    assert "All three score the same." in tn, "a three-way tie must be named as one"
    assert "<QcnnTnCheck artifactKey=" in (web / "QcnnCircuitCard.tsx").read_text()
    sat = (web / "SatelliteQcnnCard.tsx").read_text()
    assert "An MPS classifier on the same eight inputs" in sat and "leadOf(fig)" in sat
    assert "which is ahead of plain logistic regression" not in sat, "the fixed sentence was false on ESA"
    for folder in ("qcnn_medical", "qcnn_telecom", "qcnn_telecom_q7", "qcnn_satellite_opssat", "qcnn_satellite_esa"):
        assert "## Tensor-network checks (2026-11-26)" in (REPO / "models" / folder / "README.md").read_text(), folder
