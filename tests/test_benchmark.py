"""Tests for the MPC-FHE vs QSMPC-FHE benchmark."""

import json

import numpy as np

from qsh.benchmark.alice_bob_e2e import run_alice_bob_e2e
from qsh.benchmark.classical_mpc_fhe import run_classical
from qsh.benchmark.gpu_mpc_adapter import available
from qsh.benchmark.quantum_qsmpc_fhe import run_quantum
from qsh.benchmark.runner import _linear_model_and_data, compute_advantage, run_benchmark
from qsh.benchmark.security import compute_security_score, kem_category
from qsh.benchmark.quantum_advantage import quantum_advantage
from qsh.fhe.ckks_inference import LinearModel


def test_classical_path_preserves_accuracy():
    model, X, y = _linear_model_and_data()
    res = run_classical(model, X, y, backend_name="tenseal", n_samples=8)
    assert res.path == "MPC-FHE"
    assert res.accuracy == 1.0
    assert res.security_model.startswith("computational")
    assert res.max_abs_error < 1e-2


def test_quantum_path_reports_quantum_resources():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(10, 8)), rng.integers(0, 2, 10)
    res = run_quantum(X, y, n_qubits=8, n_samples=5)
    assert res.path == "QSMPC-FHE"
    assert res.t_gate_count > 0
    assert res.qubits == 8
    assert "information-theoretic" in res.security_model
    # rate-1: quantum ciphertext ratio is far below classical CKKS expansion
    assert res.ciphertext_plaintext_ratio < 2.0


def test_run_benchmark_writes_artifacts(tmp_path):
    results = run_benchmark("linear", out_dir=tmp_path, n_samples=6, include_openfhe=False)
    assert any(r.path == "MPC-FHE" for r in results)
    assert any(r.path == "QSMPC-FHE" for r in results)
    data = json.loads((tmp_path / "benchmark_linear.json").read_text())
    assert len(data) == len(results)
    assert (tmp_path / "benchmark_linear.csv").exists()


def test_gpu_adapter_reports_state_without_crashing():
    state = available()
    assert "heongpu_present" in state and "hpmpc_present" in state


def test_security_score_orders_quantum_above_classical():
    classical = compute_security_score(
        security_model="computational (RLWE / FIPS-203 family)",
        pqc_hybrid=True, kem="ml-kem-768+x25519", qber=None)
    quantum = compute_security_score(
        security_model="information-theoretic (QOTP perfect hiding)",
        pqc_hybrid=True, kem="ml-kem-1024+x25519",
        qkd_key_bits=256, qber=0.0128, leakage_bits_per_weight=0.1)
    assert 0 <= classical["total"] <= 100
    assert quantum["total"] > classical["total"]      # info-theoretic + QKD + cat5 wins
    assert quantum["kem_category"] == 5
    assert kem_category("ml-kem-768") == 3 and kem_category(None) == 3


def test_a_dead_quantum_channel_cannot_score_full_eavesdropper_marks():
    """QBER below 0.11 but above this project's own zero-key point must not score full marks.

    0.11 is the Shor-Preskill one-way BB84 bound. This repository's GLLP model stops yielding secure
    key at 0.098106, so the band [0.098106, 0.11) used to award the maximum eavesdropper-detection
    score on a channel producing nothing - the score rewarding a link the same codebase reports as
    dead. The browser's fact display was guarded against exactly this and the score was not.
    """
    from qsh.config import get_config

    s = get_config().security_scoring
    full, warn = s.eavesdrop_secure_pts, s.eavesdrop_warn_pts
    assert warn < full, "the warn tier must be worth less than the secure tier for this to mean anything"

    args = dict(security_model="computational (RLWE / FIPS-203 family)", pqc_hybrid=True,
                kem="ml-kem-768+x25519")
    # In the dangerous band, with the key measured dead.
    dead = compute_security_score(**args, qkd_key_bits=0, qber=0.105)
    assert dead["components"]["eavesdropper_detection"] == warn, (
        "a run that produced no secure key must not score full eavesdropper-detection marks, "
        "whatever its QBER sits below"
    )
    # The same QBER with a live key still scores full marks: the guard is about the KEY, not the QBER.
    live = compute_security_score(**args, qkd_key_bits=256, qber=0.105)
    assert live["components"]["eavesdropper_detection"] == full
    assert dead["total"] < live["total"]

    # And the guard must not fire when key bits were never reported, which is a different situation
    # from a measured zero - a pqc-only profile has no quantum channel to judge.
    unknown = compute_security_score(**args, qkd_key_bits=None, qber=0.105)
    assert unknown["components"]["eavesdropper_detection"] == full


def test_quantum_advantage_rewards_communication_compactness():
    classical = run_classical(*_linear_model_and_data(), backend_name="tenseal", n_samples=6)
    rng = np.random.default_rng(0)
    quantum = run_quantum(rng.normal(size=(8, 8)), rng.integers(0, 2, 8),
                          n_qubits=8, n_samples=4)
    adv = quantum_advantage(classical, quantum)
    assert 0 <= adv["score"] <= 10
    assert adv["communication_ratio"] > 1.0            # rate-1 QFHE far more compact
    assert adv["tier"] in ("dominant", "strong", "moderate", "marginal")


def test_compute_advantage_from_results():
    results = run_benchmark("linear", out_dir="/tmp/qsh-bench-test", n_samples=6,
                            include_openfhe=False)
    adv = compute_advantage(results)
    assert adv is not None and 0 <= adv["score"] <= 10


def test_alice_bob_two_party_e2e():
    r = run_alice_bob_e2e(seed=0, n_shared=8)
    assert r["session_key_agreement"] is True         # PQC+QKD session keys match
    assert r["psi_correct"] is True                    # PSI finds exactly the shared accounts
    assert r["psi_intersection_size"] == r["psi_ground_truth"] == 8
    assert r["private_inference_error"] < 1e-2         # CKKS preserves the decision
    assert r["qber"] < 0.11                             # eavesdrop check passes
    assert 0 < r["security_score"] <= 100
    assert r["ttp_free"] is True and r["raw_data_shared"] is False
    assert len(r["phases"]) == 6 and r["total_ms"] > 0
