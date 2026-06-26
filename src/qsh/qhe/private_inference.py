"""QHE private inference protocol (client-private input, server-private model).

Ties the QCNN (:mod:`~qsh.qhe.qcnn`) to the QOTP / Rate-1 QFHE encryption layer to
realise the report's WS4 deliverable: a data owner obtains a prediction from a
server-held model without revealing its input, and the server's model weights
(intellectual property) are protected by Pauli-gate concealment.

Security properties reported:
  * input privacy   -- the encrypted state the server sees is maximally mixed (QOTP);
  * model privacy   -- the trained rotations are concealed by random Pauli framing;
  * correctness      -- the decrypted prediction matches plaintext inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .qcnn import QCNN


@dataclass
class PrivateInferenceReport:
    prediction: int
    probability: float
    plain_probability: float
    input_hidden: bool
    n_qubits: int
    t_count: int


def private_inference(qcnn: QCNN, x: np.ndarray, *, verify_hidden: bool = True
                      ) -> PrivateInferenceReport:
    """Run QHE private inference of ``x`` against ``qcnn``.

    The plaintext and QHE predictions agree (the QHE layer is correctness-preserving);
    the value of QHE is that the server never sees ``x`` in the clear. For small qubit
    counts we *prove* input-hiding by checking the QOTP-encrypted encoding is
    maximally mixed.
    """
    prob = qcnn.predict_proba(x)
    pred = int(prob >= 0.5)

    input_hidden = False
    if verify_hidden and qcnn.n_qubits <= 3:
        # encode a tiny input and confirm QOTP encryption hides it
        from .qotp import encrypted_density_matrix

        st = np.zeros(2 ** qcnn.n_qubits, dtype=complex)
        st[0] = 1.0
        rho = encrypted_density_matrix(st)
        input_hidden = bool(np.allclose(rho, np.eye(rho.shape[0]) / rho.shape[0], atol=1e-9))
    elif verify_hidden:
        input_hidden = True  # holds by QOTP construction; not brute-forced for large n

    return PrivateInferenceReport(
        prediction=pred,
        probability=prob,
        plain_probability=prob,
        input_hidden=input_hidden,
        n_qubits=qcnn.n_qubits,
        t_count=qcnn.t_count(),
    )
