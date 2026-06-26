"""Tests for the QHE/FHE layer: QOTP security, homomorphic eval, Rate-1, CKKS, QCNN."""

import numpy as np
import pytest

from qsh.fhe.ckks_inference import LinearModel, encrypted_linear_inference
from qsh.qhe.qcnn import QCNN, qotp_clifford_homomorphic_demo
from qsh.qhe.qotp import encrypt, encrypted_density_matrix, pauli_key
from qsh.qhe.private_inference import private_inference


def test_qotp_is_perfectly_hiding():
    # averaging QOTP over all keys yields the maximally mixed state (perfect security)
    for n in (1, 2, 3):
        st = np.zeros(2 ** n, dtype=complex)
        st[0] = 1.0
        rho = encrypted_density_matrix(st)
        assert np.allclose(rho, np.eye(2 ** n) / 2 ** n, atol=1e-9)


def test_qhe_clifford_homomorphic_correct():
    rep = qotp_clifford_homomorphic_demo(n=3, seed=7)
    assert rep["correct"]
    assert rep["fidelity_recovered_vs_plaintext"] == pytest.approx(1.0, abs=1e-9)
    assert rep["encrypted_is_maximally_mixed"]


def test_ckks_linear_inference_accuracy():
    rng = np.random.default_rng(0)
    model = LinearModel(weight=rng.normal(size=(4, 16)), bias=rng.normal(size=4))
    x = rng.normal(size=16)
    out = encrypted_linear_inference(model, x)
    assert out["max_abs_error"] < 1e-3            # CKKS preserves the result
    assert out["prediction"] == int(np.argmax(model.weight @ x + model.bias))


def test_qcnn_runs_and_reports_tcount():
    qc = QCNN(n_qubits=8, seed=0)
    p = qc.predict_proba(np.random.default_rng(1).normal(size=8))
    assert 0.0 <= p <= 1.0
    assert qc.t_count() > 0


def test_qhe_private_inference_hides_input():
    qc = QCNN(n_qubits=2, seed=0)  # small so hiding is brute-force-verified
    rep = private_inference(qc, np.array([0.1, 0.2]))
    assert rep.input_hidden
    assert rep.probability == pytest.approx(rep.plain_probability)


def test_rate1_qfhe_roundtrip_and_rate():
    from qsh.pqc.hybrid_kem import HybridKEMRecipient
    from qsh.qhe.rate1_qfhe import (
        ciphertext_to_plaintext_ratio,
        decrypt_state,
        encrypt_state,
    )

    with HybridKEMRecipient() as rcpt:
        # large state => fixed KEM overhead is negligible => rate approaches 1
        psi = np.random.default_rng(3).normal(size=2 ** 12) + 0j
        psi /= np.linalg.norm(psi)
        ct, _, _ = encrypt_state(psi, rcpt.public_key)
        rec = decrypt_state(ct, rcpt)
        assert abs(np.vdot(psi, rec)) ** 2 == pytest.approx(1.0, abs=1e-9)
        assert ciphertext_to_plaintext_ratio(ct) < 1.2
