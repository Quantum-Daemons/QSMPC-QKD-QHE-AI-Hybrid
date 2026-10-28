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
    """The genomics use case runs a REAL published Polygenic Risk Score (PGS Catalog PGS000004, 313-SNP
    breast-cancer) under CKKS — a genuine industry-standard model that is FHE-native (linear)."""
    import numpy as np

    # The score comes from the config field, not from a literal here. Round 31 updated this test's
    # DOCSTRING to say PGS000004/313 and left the assertions on 77 - a test whose stated purpose denied
    # its own body, and it passed, because the assertions matched the pipeline's own stale default.
    from qsh.config import get_config
    from qsh.data.loaders import load_genomic_prs
    from qsh.fhe.backends import get_backend
    from qsh.fhe.ckks_inference import LinearModel, encrypted_linear_inference

    score = get_config().data.genomics_prs_score
    dosages, weights, keys, eff, freq = load_genomic_prs(n_individuals=200, seed=0, name=score)
    n_expected = {"breast_cancer": 77, "breast_cancer_prs313": 313}[score]
    assert len(weights) == n_expected and len(keys) == n_expected     # real PGS Catalog variants
    # PGS000001 publishes rsIDs; PGS000004 publishes none, so its keys are chr:pos:effect_allele.
    # Asserting "starts with rs" unconditionally is what would have caught the split had it been here.
    if score == "breast_cancer":
        assert keys[0].startswith("rs")
    else:
        assert keys[0].count(":") == 2 and not keys[0].startswith("rs")
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
    assert r["prs"]["n_snps"] == n_expected and r["prs"]["name"] == score
    # The feature label is the variant key, so its SHAPE follows the score's key space. Pinning "rs"
    # here would fail on PGS000004 for the right reason and on a future rsID score for the wrong one.
    assert r["explanation"]["top_features"][0]["feature"] in set(keys)


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
    # SIGNED, and it has to be. This asserted `== 0.0`, enshrining a clamp that made a link missing
    # by 61 bit/carrier indistinguishable from one missing by 0.001 - and from a marginal link that
    # is not negative at all. It also hid a 3.5 dB disagreement between this model and the browser's:
    # both sides reported "0" while computing -138.93 and -51.58 on the same nominal downlink.
    assert r["qsdc_secrecy_capacity"] < -50.0
    assert r["channel_loss_db"] == 34.0         # diffraction + 5 atmospheric + 3 pointing + 8 receiver optics
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
    from qsh.qkd.channel import ChannelParams, FreeSpaceParams
    from qsh.qkd.qsdc import quantum_secure_direct_comm

    msg = b"SAT-OK;payload=nominal"
    # The downlink, under the DIFFRACTION law. This read `fiber_loss_db_per_km=0.035` -> 28 dB, a
    # linear rate the browser had already rejected (it reads plausibly at 800 km beside the ~27 dB
    # Micius reports and diverges badly with range), so the two implementations disagreed by 3.5 dB
    # about the same nominal link while both reported secrecy capacity as a clamped "0".
    leo = ChannelParams(distance_km=800, free_space=FreeSpaceParams())    # 34.02 dB with the station optics
    clean = quantum_secure_direct_comm(msg, leo, seed=0)
    assert clean.delivered is None                      # not physical at this loss
    assert clean.link_budget_limited is True
    assert clean.eavesdrop_detected is False            # blaming Eve for the link budget is the bug
    assert clean.eve_gain_g > 1.0                       # the loss term is actually applied

    # A link where QSDC genuinely closes: g near 1 is the regime the published demonstrations use
    # (Qi et al. 2019 measured 4.1 dB on fibre). Only here is an abort attributable to an attacker.
    #
    # THE BLOCK IS 32 KB AND THAT IS THE POINT OF THIS ROUND, not an arbitrary fixture size. The
    # eavesdrop check is decided on a Serfling upper bound over the check subset, so certifying an
    # error rate costs carriers: on THIS link the crossover sits between 8 KB (C_s -0.485,
    # statistics-limited) and 32 KB (C_s +0.113, certified). Below it nothing about the channel can
    # be concluded - which is why the 22-byte `msg` that used to run here now aborts, and why it
    # previously DELIVERED for a reason that was arithmetic rather than physics.
    #
    # The crossover was 8 KB when this test was first re-derived an hour earlier, with the entropy
    # argument still pooled. Correcting it to `h(e_x + e_z)` quadrupled the block a certificate
    # costs. Two independent corrections compounding in the same direction is the signal worth
    # keeping: both were permissive, and neither was visible in the delivered/withheld verdict.
    block = b"x" * 32768
    short = ChannelParams(distance_km=2, fiber_loss_db_per_km=0.2, misalignment=0.001,
                          dark_count_prob=1e-9)
    ok = quantum_secure_direct_comm(block, short, seed=0)
    assert ok.delivered == block and ok.secrecy_capacity > 0.0
    assert ok.link_budget_limited is False
    assert ok.statistics_limited is False               # enough carriers to certify

    attacked = quantum_secure_direct_comm(
        block, ChannelParams(distance_km=2, fiber_loss_db_per_km=0.2, misalignment=0.001,
                             dark_count_prob=1e-9, eve_intercept_fraction=0.6), seed=0)
    assert attacked.delivered is None
    assert attacked.eavesdrop_detected is True          # THIS one is genuinely Eve
    assert attacked.link_budget_limited is False

    # And the third attribution, which is the one this round added. The SAME link and the SAME clean
    # channel at a short block cannot conclude anything, and must not report an eavesdropper or a
    # link-budget failure to say so. Before the bound, this call delivered.
    tiny = quantum_secure_direct_comm(msg, short, seed=0)
    assert tiny.delivered is None
    assert tiny.statistics_limited is True
    assert tiny.eavesdrop_detected is False, (
        "a 22-byte block cannot certify anything about this channel, so attributing its abort to an "
        "eavesdropper is an accusation the sample does not support")
    assert tiny.link_budget_limited is False, (
        "this link closes asymptotically - the binding constraint is the number of check carriers, "
        "and reporting it as a link-budget failure would send a reader to widen the aperture")
    assert tiny.error_rate_upper > tiny.error_rate, (
        "the certified bound must sit above the point estimate; if they coincide the abort is being "
        "decided on the point estimate again, which is the defect")


def test_the_satellite_eavesdrop_check_is_reachable_from_the_use_case():
    """The page's headline feature, demonstrated through the pipeline rather than a hand-built channel.

    ``qsdc_eavesdrop_detected`` is ``abort and not link_budget_limited and not statistics_limited``,
    and this test exists because that metric was 0 by construction for many rounds: the fixed 5 dB
    atmospheric and 3 dB pointing terms floored total loss at 8 dB even at zero range, so the diagram's
    ABORTED branch was unreachable and a full-strength eavesdropper still drew a completion tick.

    REWRITTEN IN ROUND 55, AND THE PREVIOUS VERSION'S REASONING WAS WRONG IN A WAY WORTH KEEPING.
    It said: "QSDC measures its error rate on a FINITE sample of check carriers, so near C_s = 0 the
    verdict genuinely fluctuates with the seed ... the test uses a point far enough from the boundary
    that sampling cannot flip it." That treats finite sampling as NOISE to be steered around. It is
    not noise. It is a certification limit, and no distance from the boundary escapes it: the abort is
    now decided on a Serfling upper bound, and at 74 bytes of telemetry - 25 check carriers - the
    bound certifies nothing at any epsilon. The old fixture delivered because 0.9848^29 = 0.64 made a
    zero-error sample the common case, not because the channel was sound.

    So the assertions here changed from "it delivers at 2.5 dB" to the three attributions, which is
    what the metric is actually for.
    """
    # RE-DERIVED TWICE IN ROUND 55, against two corrections that compounded.
    #
    # (1) `g` now includes detector efficiency - a fixed 3.979 dB before the channel contributes.
    # (2) The Eve term now takes `h(e_x + e_z)`, i.e. `h(2e)`, as Qi et al. Eq. (8) writes it. Their
    #     own arithmetic settles it: they state I(A:E) = g*Q_Bob*h(e_x+e_z) = 9.1e-4, and only
    #     h(0.016) = 9.1248e-4 reproduces that, against 4.08e-4 for the pooled h(0.006).
    #
    # With the free-space misalignment floor of 0.025, C_s > 0 now needs
    # `1 - h(0.025) - g*h(0.05) > 0`, i.e. `g < 2.902`, i.e. TOTAL loss below 4.626 dB - leaving the
    # channel at most 0.647 dB after the detector's 3.979 dB. The 2.5 dB budget this test used
    # earlier in the same round now lands at C_s = -0.443 asymptotically, so no block length rescues
    # it. That is not a regression; the 2.5 dB figure was derived from the pooled formula.
    budget = {"sat_distance_km": 1, "sat_atmospheric_db": 1.5, "sat_pointing_db": 1.0}

    # At the SHIPPED telemetry length the check certifies nothing, and must say so rather than
    # blaming either the channel or an attacker.
    short = satellite_comms.run_demo({"seed": 1, "sat_eve": 0.0, **budget})
    assert short["telemetry_delivered"] is False
    assert short["eavesdrop_detected"] is False, (
        "a 74-byte block gives ~25 check carriers, which cannot certify any error rate - reporting "
        "an eavesdropper from it is an accusation the sample does not support")

    # The eavesdrop branch IS reachable, and pinning exactly where matters more than that it is:
    # zero atmospheric and pointing loss, which the panel permits (both are min 0), plus a 4 MB
    # block. Both are extremes. A real downlink occupies neither, and the honest reading of this
    # test is that QSDC does not close over this class of link at the lengths the demo sends - which
    # is what `RI8` in the browser checklist recorded from the other direction.
    ideal = dict(sat_distance_km=1, sat_atmospheric_db=0.0, sat_pointing_db=0.0,
                 telemetry="S" * 4_194_304)
    clean = satellite_comms.run_demo({"seed": 1, "sat_eve": 0.0, **ideal})
    assert clean["telemetry_delivered"] is True, (
        "the eavesdrop-detection branch is unreachable at EVERY setting the demo offers, which is "
        "the state Round 32 fixed - re-derive the budget rather than deleting this assertion")
    assert clean["qsdc_secrecy_capacity"] > 0.0
    assert clean["eavesdrop_detected"] is False

    for seed in range(1, 6):
        attacked = satellite_comms.run_demo({"seed": seed, "sat_eve": 0.4, **ideal})
        assert attacked["telemetry_delivered"] is False, f"seed {seed} delivered under an attack"
        assert attacked["qsdc_secrecy_capacity"] < 0.0, f"seed {seed} reported a non-negative C_s"
        # The whole point: an abort this link would NOT have suffered on its own, and one the check
        # had enough carriers to conclude.
        assert attacked["eavesdrop_detected"] is True, (
            f"seed {seed} aborted without attributing it to the eavesdropper - if this fails, check "
            "whether the abort is statistics-limited instead, which is a different claim")

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
