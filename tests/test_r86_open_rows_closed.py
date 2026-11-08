"""The last open rows of the 2026-11-08 audit, held so they stay closed (Round 86).

Each test names the private checklist row or the fact-check ledger item it closes. Most pin a sentence, because the defect
each one fixes was a sentence: a label that claimed more than the code does, or a caveat outside the block it qualifies.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from qsh.config import frontend_config
from qsh.models import registry

REPO = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO / path).read_text()


def test_qc33_no_flowchart_calls_the_comparison_circuit_a_qcnn():
    readme = _read("README.md")
    assert "QOTP and QCNN, T-gates counted" not in readme
    assert readme.count("untrained QOTP and Clifford+T circuit, T-gates counted") == 5


def test_sc23_the_satellite_summary_says_simulated():
    assert "<summary>Satellite: simulated free-space QKD and direct telemetry</summary>" in _read("README.md")


def test_ux6_the_sentence_the_page_shows_carries_the_covariates_and_the_snps():
    # The regular expression is RegistryNote's: the first sentence stays visible and the rest folds.
    first = re.match(r"^([\s\S]{40,}?[.!?])\s+(?=[A-Z(`\"])", registry.M_PRS.baseline_source).group(1)
    assert "0.637" in first and "WITH COVARIATES" in first and "162 of the score's 187" in first


def test_sc10_the_signed_number_is_not_called_a_capacity():
    for path in ("web/src/lib/clientCompute.ts", "web/src/lib/useCaseFlows.ts", "src/qsh/usecases/satellite_comms/__init__.py"):
        assert "qsdc_secrecy_capacity" not in _read(path), path
    assert '"qsdc_eq8_bound"' in _read("web/src/lib/clientCompute.ts")


def test_pf3_the_measurement_comes_before_the_projection():
    compute, flows = _read("web/src/lib/clientCompute.ts"), _read("web/src/lib/useCaseFlows.ts")
    assert 'metric(stage, "mpc_nonlinear_measured_ms", r2(real.mpcMs));' in compute
    assert 'metric(stage, "mpc_comm_measured_kb", r2(real.commBytes / 1024));' in compute
    assert '${f(m, "fhe_linear_measured_ms")} ms measured, ${f(m, "fhe_linear_ms")} ms${scaleTxt}' in flows
    assert '${f(m, "mpc_comm_measured_kb")} KiB measured; ' in flows


@pytest.mark.parametrize("usecase", sorted(registry.BUNDLED_DEMO_SAMPLES))
def test_rd23_the_data_line_has_the_credit_and_the_notice(usecase: str):
    entry = frontend_config()["demo_samples"][usecase]
    assert entry["attribution"] == registry.BUNDLED_DEMO_SAMPLES[usecase][1].attribution
    assert entry["attribution"], f"{usecase} has no credit"
    assert entry["file"].startswith("/data/"), entry["file"]
    assert (REPO / "web" / "public" / entry["file"].lstrip("/")).exists()


def test_rd23_the_line_prints_them():
    compute = _read("web/src/lib/clientCompute.ts")
    assert "Credit: ${ds.attribution}" in compute and "fields of ${ds.file}" in compute


def test_e14_nist_ir_8547_rows_are_a_schedule_not_a_requirement():
    clock = _read("web/src/lib/regulatoryClock.ts")
    rows = re.findall(r'\{ lane: "NIST", short: "(?:deprecate|disallow) RSA/ECC".*?firmness: "([a-z ]+)"', clock, flags=re.S)
    assert rows == ["announced", "announced"]


def test_e5_the_diagram_does_not_put_ml_kem_under_an_rfc():
    panel = _read("web/src/components/NetworkPositioning.tsx")
    assert "ML-KEM-768 + PPK (RFC 9370, 8784)" not in panel
    assert "ML-KEM-768 (draft) + PPK (RFC 8784)" in panel


def test_b27_the_clock_assumption_and_a_bar_label_that_is_not_geography():
    assert "assumes it is midnight" in registry.PAYSIM_MODEL_FEATURES["night_tx"]
    assert 'cross_border: "transfer_or_cash_out"' in _read("web/src/components/ShapPanel.tsx")


def test_d1_d3_the_corpus_is_described_as_its_own_design():
    assert "methodology-grounded" not in registry.DS_QKD_ATTACK_SYNTH.name
    assert "own design" in registry.DS_QKD_ATTACK_SYNTH.name
    assert "No public real QKD-attack capture was found" in registry.DS_QKD_ATTACK_SYNTH.notes
    assert "No commercial-OK real" not in registry.DS_QKD_ATTACK_SYNTH.notes


def test_the_live_walk_of_2026_11_08() -> None:
    """Four things the deployed site showed that no gate held."""
    assert "or a schedule its issuer has announced" in _read("web/src/components/RegulatoryClock.tsx")
    value = _read("web/src/pages/Value.tsx")
    assert "ReLUs and argmax in garbled circuits" in value, "the prior-art row still described only the decrypted path"
    assert '"calibration"' in _read("web/src/components/ModelPanel.tsx").split("NOT_A_SUBTREE = new Set(")[1].split(")")[0]
    compute = _read("web/src/lib/clientCompute.ts")
    assert re.search(r'if \(bscale !== 1\) \{\s*metric\(stage, "mpc_nonlinear_measured_ms"', compute), "a measured row beside an identical scaled one at x1"
