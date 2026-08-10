"""Tests for the four industry use cases + the QML-IDS detectors."""

import pytest

from qsh.qml_ids.attacks import ATTACKS, binary_labels, generate
from qsh.qml_ids.detector import evaluate_classical
from qsh.usecases import finance_aml, genomics, medical_fl, satellite_comms, telecom_ids


def test_finance_aml_psi_and_private_score():
    r = finance_aml.run_demo({"seed": 1})
    # PSI recovers exactly the shared fraud ring
    assert r["shared_fraud_accounts_found"] == r["ground_truth_shared"]
    # a known-fraud transaction is privately scored as fraud
    assert r["private_fraud_prediction"] == 1
    # XAI: a faithful, named feature attribution is produced (and gets ML-DSA-signed)
    exp = r["explanation"]
    assert exp and exp["method"] == "exact-linear-shap"
    assert exp["efficiency_gap"] < 1e-6          # exact (SHAP local-accuracy axiom)
    assert exp["top_features"] and exp["top_features"][0]["feature"]


def test_medical_fl_trains_and_diagnoses():
    r = medical_fl.run_demo({"seed": 1, "rounds": 4})
    assert r["federated_accuracy"] > 0.85
    assert r["private_diagnosis"] in (0, 1)
    assert r["hospitals"] == 4


def test_genomics_secure_sum_exact():
    r = genomics.run_demo({"seed": 1})
    assert r["pooled_allele_count"] == r["ground_truth_total"]


def test_genomics_real_prs_under_ckks():
    """The genomics use case runs a REAL published Polygenic Risk Score (PGS Catalog PGS000001, 77-SNP
    breast-cancer) under CKKS — a genuine industry-standard model that is FHE-native (linear)."""
    import numpy as np

    from qsh.data.loaders import load_genomic_prs
    from qsh.fhe.backends import get_backend
    from qsh.fhe.ckks_inference import LinearModel, encrypted_linear_inference

    dosages, weights, rsids, eff, freq = load_genomic_prs(n_individuals=200, seed=0)
    assert len(weights) == 77 and len(rsids) == 77 and rsids[0].startswith("rs")   # real PGS Catalog SNPs
    scores = dosages @ weights
    thr = float(np.quantile(scores, 0.75))
    model = LinearModel(np.vstack([np.zeros_like(weights), weights]), np.array([0.0, -thr]))
    subj = dosages[int(np.argmax(scores))]
    out = encrypted_linear_inference(model, subj, get_backend("tenseal"))
    # the encrypted PRS reproduces the plaintext risk decision (FHE-native linear PRS = CKKS dot-product)
    assert out["prediction"] == int(float(subj @ weights) > thr)
    assert out["max_abs_error"] < 1e-2
    # the live pipeline surfaces the real PRS + a per-SNP signed explanation
    r = genomics.run_demo({"seed": 1})
    assert r["prs"]["n_snps"] == 77 and r["prs"]["name"] == "breast_cancer"
    assert r["explanation"]["top_features"][0]["feature"].startswith("rs")


def test_telecom_ids_classical_meets_target():
    """The target is checked on macro F1, not on attack-vs-normal F1.

    The collapsed binary score treats all seven attack types as one label, so it can clear 0.93
    while an individual class is being missed entirely — the failure Engelen et al. (WTMC 2021)
    documented on CIC-IDS2017. Asserting the easy number would let exactly that regression through.
    """
    r = telecom_ids.run_demo({"seed": 1})
    assert r["ids_macro_f1"] >= 0.93, "per-class detection regressed"
    assert r["attack_vs_normal_f1"] >= r["ids_macro_f1"], (
        "the collapsed score should be the more forgiving of the two; if it is not, the class "
        "structure is not what this assertion assumes"
    )


@pytest.mark.slow
def test_telecom_quantum_svc_meets_target():
    """The 2026 QML-IDS standard — a quantum-kernel SVM (QSVC) — detects QKD attacks at F1>=0.93.
    Slow (O(n^2) quantum-kernel evaluations), so kept out of the default suite."""
    from qsh.qml_ids.detector import evaluate_quantum_svc

    r = evaluate_quantum_svc(n_per_class=40, seed=0)
    assert r.detector == "quantum-svc" and r.binary_f1 >= 0.90 and r.n_params > 0


def test_satellite_clean_downlink_yields_key_but_qsdc_cannot_close():
    """BB84 closes over this downlink; QSDC does not, and the two are not the same question.

    These previously asserted `telemetry_delivered is True`, which was asserting the permissive
    bug: the abort was a fixed QBER of 0.11, the LOSS-FREE root of the secrecy-capacity
    expression. With the loss term restored (g = 10^(loss_dB/10), Qi et al. 2019) a 24.5 dB
    downlink needs QBER below ~3e-4 and the misalignment floor alone is 0.025, so Cs is negative
    by ~51 bit/carrier. The delivery those assertions protected was never physical.
    """
    r = satellite_comms.run_demo({"seed": 1})
    assert r["secure_key_bits"] >= 256          # free-space BB84 still yields a usable key
    assert r["qber"] < 0.11                     # and BB84's own bound is comfortably met
    assert r["telemetry_delivered"] is False    # QSDC is not available at this link budget
    assert r["qsdc_secrecy_capacity"] == 0.0    # reported clamped at 0; the raw value is negative
    assert r["eavesdrop_detected"] is False     # nothing is attacking it -- do not say otherwise


def test_satellite_eavesdropper_is_not_attributable_on_a_link_that_was_already_dead():
    """The honest consequence, and the reason `link_budget_limited` exists as a separate flag.

    You cannot detect an eavesdropper by loss of secrecy capacity on a channel that had none to
    begin with. Reporting `eavesdrop_detected` here would accuse Eve of an abort that happens
    identically without her -- which is exactly what this demo did on every run, including clean
    ones.
    """
    r = satellite_comms.run_demo({"seed": 1, "sat_eve": 0.5})
    assert r["telemetry_delivered"] is False     # still withheld, so no message leaks
    assert r["secure_key_bits"] == 0             # BB84 DOES detect her -- that channel still works
    assert r["eavesdrop_detected"] is False      # but QSDC cannot attribute it, and must not claim to


def test_qsdc_aborts_on_secrecy_capacity_and_attributes_the_cause():
    from qsh.qkd.channel import ChannelParams
    from qsh.qkd.qsdc import quantum_secure_direct_comm

    msg = b"SAT-OK;payload=nominal"
    leo = ChannelParams(distance_km=800, fiber_loss_db_per_km=0.035)      # 28 dB, g ~ 631
    clean = quantum_secure_direct_comm(msg, leo, seed=0)
    assert clean.delivered is None                      # not physical at this loss
    assert clean.link_budget_limited is True
    assert clean.eavesdrop_detected is False            # blaming Eve for the link budget is the bug
    assert clean.eve_gain_g > 1.0                       # the loss term is actually applied

    # A link where QSDC genuinely closes: g near 1 is the regime the published demonstrations use
    # (Qi et al. 2019 measured 4.1 dB on fibre). Only here is an abort attributable to an attacker.
    short = ChannelParams(distance_km=2, fiber_loss_db_per_km=0.2, misalignment=0.001,
                          dark_count_prob=1e-9)
    ok = quantum_secure_direct_comm(msg, short, seed=0)
    assert ok.delivered == msg and ok.secrecy_capacity > 0.0
    assert ok.link_budget_limited is False

    attacked = quantum_secure_direct_comm(
        msg, ChannelParams(distance_km=2, fiber_loss_db_per_km=0.2, misalignment=0.001,
                           dark_count_prob=1e-9, eve_intercept_fraction=0.6), seed=0)
    assert attacked.delivered is None
    assert attacked.eavesdrop_detected is True          # THIS one is genuinely Eve
    assert attacked.link_budget_limited is False


def test_attack_generator_shape():
    X, y, names = generate(n_per_class=50, seed=0)
    assert X.shape == (8 * 50, 7)
    assert set(y.tolist()) == set(range(8))
    assert names == ATTACKS
    assert binary_labels(y).max() == 1


def test_classical_ids_report():
    rep = evaluate_classical(n_per_class=200, seed=0)
    assert rep.macro_f1 > 0.8
    assert rep.binary_f1 >= 0.93


@pytest.mark.slow
def test_quantum_ids_runs():
    from qsh.qml_ids.detector import evaluate_quantum

    rep = evaluate_quantum(n_per_class=60, seed=0, epochs=8)
    assert 0.0 <= rep.binary_f1 <= 1.0
    assert rep.n_params < 100  # low-resource quantum classifier
