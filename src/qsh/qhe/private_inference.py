"""QHE private inference protocol (client-private input, server-private model).

Ties the QCNN (:mod:`~qsh.qhe.qcnn`) to the QOTP encryption layer to sketch the report's WS4
deliverable: a data owner obtains a prediction from a server-held model without revealing its input.

READ THE SCOPE BELOW BEFORE QUOTING THIS MODULE. An earlier version of this header listed three
"security properties reported", two of which the code does not establish, and the corrections were
written into the function bodies while the header kept asserting the original claims. A reader who
stops at line 12 got exactly the retracted version, contradicted twenty lines later - which is how a
stale header outranks an accurate body.

What holds:
  * input privacy -- the state the server sees under a random Pauli frame is maximally mixed (QOTP),
    and that is a real, standard property.

What does NOT hold, stated here rather than only at the point of use:
  * model privacy is NOT established. It rested on "Pauli-gate concealment" via
    :func:`~qsh.qhe.qotp.pauli_output_concealment_demo`, which applies no circuit and therefore
    conceals a state rather than a computation. Concealing the trained rotations from a client would
    need the Pauli frame propagated through the circuit, which is unimplemented here.
  * correctness against plaintext inference is NOT tested. ``plain_probability`` is the SAME
    computation as ``probability``, not an independent reference, so comparing them is a tautology.
    Nothing is decrypted and nothing is compared.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .qcnn import QCNN


@dataclass
class PrivateInferenceReport:
    """What this function actually produced, named so the names are not doing work they cannot.

    `plain_probability` is the SAME computation as `probability`, not an independent plaintext
    reference. That was not obvious from the field names, and a test asserting the two are
    approximately equal was therefore a tautology that could never fail - it read as evidence that
    an encrypted path preserves correctness, and evidenced nothing. The field is kept because
    callers read it, and documented here so it cannot be mistaken for a comparison again.
    """

    prediction: int
    probability: float
    plain_probability: float          # identical to `probability` by construction - see above
    input_hidden: bool
    n_qubits: int
    t_count: int


def private_inference(qcnn: QCNN, x: np.ndarray, *, verify_hidden: bool = True
                      ) -> PrivateInferenceReport:
    """Run QHE private inference of ``x`` against ``qcnn``.

    SCOPE, stated plainly because the function name overpromises. This runs ``predict_proba`` in
    PLAINTEXT and separately proves a QOTP hiding property. It does not evaluate the circuit
    homomorphically, and `plain_probability` is the same value as `probability` rather than an
    independent reference.

    Two further limits worth knowing before quoting anything from here:

      * `input_hidden` is brute-forced only for ``n_qubits <= 3``, and on a FIXED basis state
        rather than on ``x``. Above that it is set True by construction rather than checked.
      * The untrained circuit is not a classifier, and the bound below needs its recipe.
        Measured over 200 draws per cell with a fixed circuit (seed 0):

            inputs           nq=4 max   nq=8 max   crossings of 0.5
            uniform[0,1)       0.2392     0.2304     0 / 200
            uniform[-1,1)      0.2280     0.2737     0 / 200
            normal(0,1)        0.9834     0.9122     18 / 200 and 20 / 200

        The figure this docstring carried - "never exceeds 0.241" - reproduces only on
        uniform[0,1), which is presumably what was drawn, and the docstring never said so.
        It is not robust: uniform[-1,1) at 8 qubits already reaches 0.2737, and on unbounded
        inputs the circuit crosses 0.5 for about a tenth of draws. So "returns class 0 for
        every input" is true of a bounded input range and false in general.

        What survives, and is the reason this path is not a predictor: the output is a
        function of the circuit rather than of the data in any useful sense - it is an
        untrained fixed circuit. Use ``qhe.qcnn_trained`` for a prediction; use this for the
        hiding demonstration and the T-count.
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
