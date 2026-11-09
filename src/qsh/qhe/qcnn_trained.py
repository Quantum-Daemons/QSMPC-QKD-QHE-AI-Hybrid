"""Trained variational QCNN inference for the Python pipeline.

WHY THIS EXISTS. Round 25 found that the browser's QCNN backend surfaced a class carrying no
information about its input, and fixed it there. The Python path had the same problem and was not
touched, so the fix reached only half the codebase. Reaching the Python path is not hypothetical:
``api/main.py`` passes ``config`` straight through and the Parameters panel offers ``qcnn``, so a
full-deployment run selects it exactly as a browser run does.

The two failures are NOT identical, and the difference matters:

    browser   ``qotpQcnnInfer`` applied a QOTP mask it never undid, and the class flipped with the
              run seed for 100% of inputs. Visibly random.
    python    ``QCNN.predict_proba`` never exceeds ~0.24 over 200 random inputs at 4 or 8 qubits,
              so it never crosses the 0.5 decision boundary and the class is **0 for every input**,
              at every seed. Correlation with the input is -0.05.

A coin flip looks broken. A CONSTANT PREDICTOR looks confident and reproducible, which is why it
survived longer - and it is a shape this project has been bitten by before, in the medical model
that scored exactly the majority-class rate.

WHAT THIS MODULE DOES. Loads the weights fitted by ``scripts/train/qcnn_variational.py`` and
evaluates the same circuit. The implementation is deliberately the plain-numpy one that was used as
the INDEPENDENT reimplementation when the trainer was verified: on the circuit of that time it
agreed with the torch training circuit to 2.4e-07, after a crossed-CNOT bug in the trainer was found
by exactly that comparison. On the current shared-filter circuit the cross-language test measures
about 7.6e-07 (finance, four wires) and 1.3e-06 (medical) over 24 rows. Two implementations that
were checked against each other are worth more than one that was not. It serves the two binary
artifacts only; the 8-class telecom circuit is implemented by the trainer and the browser.

WHAT IT DOES NOT DO. No QOTP mask on this path. Undoing a Pauli frame through Clifford+T is not
implemented anywhere in this repository - ``T X T-dagger`` is not Pauli - so hiding and prediction
stay separate rather than claiming a correction that does not exist. ``qhe/qotp.py`` remains the
hiding demonstration and proves its own claim properly.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]

#: Use case -> trained-weights file. A use case absent here has no trained circuit, and the caller
#: must decline to surface a class rather than reporting an untrained one.
_WEIGHTS = {"medical_fl": _REPO / "models" / "qcnn_medical" / "weights.json",
            "finance_aml": _REPO / "models" / "qcnn_finance" / "weights.json"}


@lru_cache(maxsize=8)
def load_weights(usecase: str) -> dict | None:
    """Trained weights for ``usecase``, or None when none have been fitted."""
    path = _WEIGHTS.get(usecase)
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def _finite(v, n: int) -> bool:
    try:
        a = np.asarray(v, dtype=float).ravel()
    except (TypeError, ValueError):
        return False
    return a.size >= n and bool(np.all(np.isfinite(a[:n])))


def validate(weights: dict) -> dict:
    """Refuse a truncated or malformed artifact BEFORE simulating it; return the layout it needs.

    Round 84. A short ``theta`` used to surface as whatever numpy raised first - a reshape error, an
    IndexError, or the fully-connected check below - and the browser twin of this function went
    further and returned class 0 with ``<Z0>`` = NaN. Both now refuse with one message naming what is
    wrong, checked before anything runs, on the same rules as ``validateQcnnWeights`` in
    ``web/src/lib/quantumSim.ts``.
    """
    def bad(what: str):
        raise ValueError(f"QCNN artifact is malformed: {what}. Refusing rather than reporting a class "
                         "from NaN or from a circuit the angles were not trained for.")

    if not isinstance(weights, dict):
        bad("no weights object")
    nq, layers = weights.get("n_qubits"), weights.get("layers")
    if not isinstance(nq, int) or not 2 <= nq <= 16:
        bad(f"n_qubits is {nq!r}, not an integer in 2..16")
    if not isinstance(layers, int) or layers < 1:
        bad(f"layers is {layers!r}, not a positive integer")
    nf = weights.get("n_features", nq)
    if not isinstance(nf, int) or not 1 <= nf <= nq:
        bad(f"n_features is {nf!r} for {nq} wires")
    circuit = (weights.get("circuit") or {}).get("name", "tree")
    # THIS MODULE IMPLEMENTS THE TREE ONLY. The SU(4) telecom circuit is the trainer's and the
    # browser's; running its 15-angle blocks through this file's 8-angle filter would be a different
    # circuit on the same numbers, which is the defect a name check exists to stop.
    if circuit != "tree":
        bad(f"circuit {circuit!r} is not implemented here (this module runs the tree only)")
    readout_q = int((weights.get("readout") or {}).get("readout_qubits", 0) or 0)
    if readout_q:
        bad(f"a {readout_q}-qubit multiclass readout; this module computes <Z0> only")
    probe, stages = nq, 0
    while probe // 2 >= 1 and probe > 1:
        stages += 1
        probe //= 2
    want = stages * layers * 8 + stages + 2 * probe
    theta = weights.get("theta")
    if not isinstance(theta, list) or len(theta) != want or not _finite(theta, want):
        n = len(theta) if isinstance(theta, list) else 0
        bad(f"theta carries {n} angles{' (some not finite numbers)' if n == want else ''}; the tree "
            f"circuit at {nq} wires needs {want} (conv {stages * layers * 8} + pool {stages} + fc 2x{probe})")
    for key in ("scale", "bias"):
        v = weights.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
            bad(f"the binary <Z0> head needs a finite scalar {key}, got {v!r}")
    c = weights.get("encoding_scale", 1.0)
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not 0.0 < float(c) <= 1.0:
        bad(f"encoding_scale is {c!r}, not in (0, 1]")
    moments = _finite(weights.get("feature_mean"), nf) and _finite(weights.get("feature_std"), nf)
    if weights.get("input_space") == "raw" and not moments:
        bad(f"raw input needs {nf} finite feature_mean and feature_std")
    return {"nq": nq, "layers": layers, "nf": nf, "stages": stages, "survivors": probe,
            "c": float(c)}


def infer(features, weights: dict) -> tuple[float, int, float]:
    """Return ``(expectation, cls, logit)`` for one feature vector; see :func:`_run_circuit`."""
    return _run_circuit(features, weights)[:3]


def final_state(initial: np.ndarray, weights: dict) -> tuple[np.ndarray, np.ndarray]:
    """The trained circuit applied to the basis-state amplitudes ``initial``: ``(re, im)``, with no encoding step.

    `qcnn_pauli` builds the circuit's readout as a linear form from this, through the same gates :func:`infer` applies.
    """
    return _run_circuit(None, weights, initial=np.asarray(initial, dtype=float))[3:]


def _run_circuit(features, weights: dict, initial: np.ndarray | None = None) -> tuple[float, int, float, np.ndarray, np.ndarray]:
    """Return ``(expectation, cls, logit, re, im)``; ``initial`` replaces the encoding.

    Mirrors ``scripts/train/qcnn_variational.py`` gate for gate: RY(c*tanh(z)*pi) data encoding,
    with c the artifact's ``encoding_scale`` (1 when absent); then
    per stage ``layers`` rounds of one shared 8-angle two-qubit filter (RY, RZ on each wire, CNOT,
    RZ, RY on each wire) on the disjoint pairs, and a CRY pool per pair with one shared angle per
    stage, halving the register; then a per-survivor RZ, RY block (and a CNOT ring only when more
    than one wire survives, which no binary artifact has); then <Z0>. Because the filter touches only
    disjoint pairs and a pooled wire is never touched again, this is a tree of two-qubit blocks:
    classically simulable in linear time at any width. No QOTP mask is applied.

    The tanh encoding is not cosmetic. The features are standardised to z-scores first, and the
    first version of the trained model min-max scaled z-scores against RAW units instead - which
    pinned seven of eight features to -1 and produced a degenerate state, making the TRAINED path
    exactly as meaningless as the untrained one it replaced. tanh bounds an unbounded z-score
    without reordering it.

    ``features`` are in the artifact's ``input_space``: raw units for every artifact shipped since
    Round 84 (finance and medical), standardised here with the artifact's own moments. A malformed
    artifact raises ``ValueError`` from ``validate`` before anything is simulated.
    """
    layout = validate(weights)
    nq = layout["nq"]
    theta = np.asarray(weights["theta"], dtype=float).ravel()
    n = 1 << nq
    re = np.zeros(n)
    im = np.zeros(n)
    re[0] = 1.0

    idx0 = {q: np.flatnonzero((np.arange(n) & (1 << q)) == 0) for q in range(nq)}

    def ry(q: int, th: float) -> None:
        c, s = np.cos(th / 2.0), np.sin(th / 2.0)
        i0 = idx0[q]
        i1 = i0 | (1 << q)
        r0, m0, r1, m1 = re[i0].copy(), im[i0].copy(), re[i1].copy(), im[i1].copy()
        re[i0] = c * r0 - s * r1
        im[i0] = c * m0 - s * m1
        re[i1] = s * r0 + c * r1
        im[i1] = s * m0 + c * m1

    def cnot(ctl: int, tgt: int) -> None:
        i = np.arange(n)
        sel = ((i & (1 << ctl)) != 0) & ((i & (1 << tgt)) == 0)
        src = i[sel]
        dst = src | (1 << tgt)
        re[src], re[dst] = re[dst].copy(), re[src].copy()
        im[src], im[dst] = im[dst].copy(), im[src].copy()

    x = np.zeros(0) if features is None else np.asarray(features, dtype=float).ravel()
    # The caller's units are the artifact's to name, and assuming them is how the first trained model
    # became as meaningless as the untrained one. finance arrives RAW and mixed-scale (a log amount
    # and three binary flags since its Round 84 retrain), so it must be standardised here with the
    # moments fitted at training time; medical, since Round 84, likewise.
    #
    # ROUND 84: medical declares `raw` too. It was `zscore`, fed z-scores made with the caller's own
    # StandardScaler rather than the trainer's training-split moments, and that moved 2 of 86
    # held-out classes in the browser. Raw units standardised here with the artifact's constants
    # cannot drift from the trainer, whoever scales their copy of the data.
    nf = layout["nf"]
    if features is not None and weights.get("input_space") == "raw":
        mu = np.asarray(weights["feature_mean"], dtype=float)[:nf]
        sd = np.asarray(weights["feature_std"], dtype=float)[:nf]
        sd = np.where(sd == 0, 1.0, sd)
        x = x[:nf]
        x = (x - mu[:x.size]) / sd[:x.size]
    # Wires past n_features stay in |0>, as the trainer's zero pad leaves them. The angle carries the
    # artifact's encoding scale: c = 1 folds a feature's two tails onto one state, and medical ships 0.5.
    c = layout["c"]
    if initial is None:
        for q in range(nq):
            ry(q, c * float(np.tanh(x[q] if q < min(x.size, nf) else 0.0)) * np.pi)
    else:
        re[:] = initial
        im[:] = 0.0
    # QCNN, mirroring scripts/train/qcnn_variational.py and web/src/lib/quantumSim.ts.
    #
    # ROUND 46: THIS WAS THE THIRD COPY OF THE CIRCUIT. The trainer, the browser and this inference
    # path each carried their own state-vector implementation, and this one still read `theta` as
    # `(layers, n_qubits)` after the other two moved to a shared-filter QCNN - so it raised
    # IndexError on every retrained artifact. "Two implementations of one thing" is this project's
    # named recurring defect; here it was three, and only a test caught the third.
    def rz(q: int, th: float) -> None:
        c, s = np.cos(th / 2.0), np.sin(th / 2.0)
        bit = 1 << q
        idx = np.arange(n)
        hi = (idx & bit) != 0
        r, m = re.copy(), im.copy()
        re[hi], im[hi] = c * r[hi] - s * m[hi], c * m[hi] + s * r[hi]
        re[~hi], im[~hi] = c * r[~hi] + s * m[~hi], c * m[~hi] - s * r[~hi]

    def cry(ctl: int, tgt: int, th: float) -> None:
        c, s = np.cos(th / 2.0), np.sin(th / 2.0)
        cb, tb = 1 << ctl, 1 << tgt
        idx = np.arange(n)
        sel = ((idx & cb) != 0) & ((idx & tb) == 0)
        i0, i1 = idx[sel], idx[sel] | tb
        r0, m0, r1, m1 = re[i0].copy(), im[i0].copy(), re[i1].copy(), im[i1].copy()
        re[i0], im[i0] = c * r0 - s * r1, c * m0 - s * m1
        re[i1], im[i1] = s * r0 + c * r1, s * m0 + c * m1

    layers = int(weights["layers"])
    readout_q = max(1, int((weights.get("readout") or {}).get("readout_qubits", 0) or 0))
    probe, stages = nq, 0
    while probe // 2 >= readout_q and probe > 1:
        stages += 1
        probe //= 2
    n_conv = stages * layers * 8
    conv = theta[:n_conv].reshape(stages, layers, 8)
    pool = theta[n_conv:n_conv + stages]
    # THE FULLY-CONNECTED BLOCK IS (survivors, 2), NOT ONE SCALAR. This read
    # `fc = float(theta[n_conv + stages])` and applied that single angle to every survivor, which is
    # the pre-Round-47 block. Round 47 made it a per-survivor RY+RZ in the trainer and in
    # `quantumSim.ts` and left this THIRD implementation behind - the same defect, a sixth time in
    # five rounds, in the language that had already been fixed twice.
    #
    # It was not producing a wrong number, and that is the part worth recording. Both artifacts this
    # module serves are BINARY, so they pool all the way to a single survivor; `theta[n_conv+stages]`
    # is then exactly that survivor's RY angle. HISTORICALLY - under the RY-then-RZ ordering in force
    # while the old form was live - the RZ it dropped was the LAST gate before the readout and so
    # commuted with the <Z0>. That argument does not describe the current tree: later in the same
    # round the fc block was reordered to RZ-then-RY, and a trailing-diagonal RZ no longer exists to
    # commute. Against the trainer of the revision that carried the old form, on that revision's
    # weights and 24 rows, the gap was 6.9e-07 on finance and 1.4e-06 on medical - float32 noise, and
    # correct by two coincidences rather than by construction. Both figures are HISTORICAL; the
    # equivalent measurement on the current tree is the shipped-vs-numpy parity below.
    #
    # Seeded proof that it was one artifact away from breaking: at `readout_qubits=2` on 8 wires the
    # register pools to TWO survivors, and on the seed the repo's own gate pins -
    # `tests/test_the_qcnn_agrees_across_languages.py::
    #  test_the_old_fc_block_diverges_once_there_are_two_survivors`, torch seed 1, 16 rows - the old
    # form diverges from the trainer by **0.0335** in class probability (0.0946 summed per row).
    # That is the TEST's figure, re-measured on the current tree, and it is four orders of magnitude
    # above the ~1e-06 float32 agreement the shipped binary artifacts show. An earlier revision of
    # this comment said 0.2701, which is roughly 8x the reproducible value and matches no draw the
    # repo pins; the divergence is seed-dependent, so only the pinned seed's number is quotable.
    fc = theta[n_conv + stages:].reshape(-1, 2)

    active = list(range(nq))
    for st in range(stages):
        for r in range(layers):
            w8 = conv[st, r]
            for a, b in zip(active[0::2], active[1::2]):
                ry(a, float(w8[0])); rz(a, float(w8[1]))
                ry(b, float(w8[2])); rz(b, float(w8[3]))
                cnot(a, b)
                # RZ before RY in the second half; see scripts/train/qcnn_variational.py.
                rz(a, float(w8[5])); ry(a, float(w8[4]))
                rz(b, float(w8[7])); ry(b, float(w8[6]))
        survivors = []
        for a, b in zip(active[0::2], active[1::2]):
            cry(b, a, float(pool[st]))
            survivors.append(a)
        if len(active) % 2 == 1:
            cry(active[-1], active[0], float(pool[st]))
        active = survivors
    if fc.shape[0] != len(active):
        raise ValueError(
            f"the fully-connected block carries {fc.shape[0]} survivor(s) but pooling left "
            f"{len(active)} ({active}). The artifact's parameter layout does not match the circuit "
            "this module implements, and continuing would silently apply the wrong angles.")
    # RZ before RY: a diagonal gate applied last is unobservable through |amplitude|^2 and a CNOT
    # ring. See the note in scripts/train/qcnn_variational.py.
    for j, q in enumerate(active):
        rz(q, float(fc[j, 1])); ry(q, float(fc[j, 0]))
    if len(active) > 1:
        for j in range(len(active)):
            cnot(active[j], active[(j + 1) % len(active)])

    p = re * re + im * im
    sign = np.where((np.arange(n) & 1) == 0, 1.0, -1.0)
    expectation = float((p * sign).sum())
    logit = expectation * float(weights["scale"]) + float(weights["bias"])
    return expectation, int(logit > 0), logit, re, im
