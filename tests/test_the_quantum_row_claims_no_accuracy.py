"""The QSMPC-FHE benchmark row must not report an accuracy, in EITHER language.

WHY THIS EXISTS

`run_quantum` returned `accuracy=1.0` with the comment "QHE preserves the plaintext decision
exactly", backed by a `correct` counter that did `correct += 1` unconditionally inside the loop and
was never read. There is nothing there to be accurate about: the QCNN carries no trained weights, and
`qsh/qhe/private_inference.py` says so in its own docstring - "Nothing is decrypted and nothing is
compared" - and names the probability-versus-plaintext comparison a tautology.

The browser twin was corrected to `accuracy: null`, with a comment recording that it had "reported
1.0 for three rounds while the MPC-FHE row eight lines above carries a comment forbidding exactly
that". THAT FIX REACHED ONE LANGUAGE. `web/src/lib/usePipelineRun.ts` PREFERS
`st.artifacts.benchmark_result`, so the full deployment served the Python row: the identical line of
the identical table on the identical page printed 1.0 through one transport and an em dash through
the other, and 1.0 is the flattering side of the contradiction - it made the quantum path look
perfectly accurate, fastest AND most secure, with one of the three invented.

WHAT IS ASSERTED

That the Python row's accuracy is None, that the browser's is null, and that no `correct`-style
counter has come back. The third is asserted on the source because a counter that is computed and
discarded produces no observable output - which is exactly why this one survived.

WHAT IS DELIBERATELY NOT ASSERTED

Not that the other fields are right. `test_benchmark.py::test_quantum_path_reports_quantum_resources`
covers the T-gate count, qubits and security model. Only that the row makes no accuracy claim.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from pathlib import Path

import numpy as np

from qsh.benchmark.quantum_qsmpc_fhe import run_quantum

REPO = Path(__file__).resolve().parents[1]
CLIENT = REPO / "web" / "src" / "lib" / "clientCompute.ts"


def _row():
    rng = np.random.default_rng(0)
    return run_quantum(rng.random((6, 8)), rng.integers(0, 2, 6), n_qubits=4, n_samples=3, seed=0)


def test_the_python_row_reports_no_accuracy():
    """The outcome, on the transport the full deployment actually serves."""
    got = _row().accuracy
    assert got is None, (
        f"the QSMPC-FHE benchmark row reports accuracy={got!r}. The QCNN in this path has no trained "
        "weights and nothing is compared against a plaintext label, so any number here is invented. "
        "`MetricRow.accuracy` is `float | None` and `BenchmarkPanel` renders null as an em dash.")


def test_the_browser_row_agrees():
    """Both languages, because the fix reaching one of them is what this file is about."""
    src = CLIENT.read_text()
    m = re.search(r'path:\s*"QSMPC-FHE".*?accuracy:\s*(\w+)', src, re.S)
    assert m, "could not find the QSMPC-FHE row's accuracy in clientCompute.ts; re-derive this test"
    assert m.group(1) == "null", (
        f"the browser's QSMPC-FHE row reports accuracy: {m.group(1)}, and Python reports None. These "
        "two render the SAME row of the SAME table - `usePipelineRun.ts` prefers the Python artifact "
        "when it exists - so a disagreement means one page shows two different numbers for one "
        "quantity depending on which transport answered.")


def test_no_unconditional_correctness_counter_has_returned():
    """The mechanism, asserted on the source because it produces no output to assert on.

    `correct += 1` with no condition, inside the sample loop, never read. A counter that is computed
    and discarded is invisible to every behavioural test; the only trace it left was the 1.0 it once
    justified. So this checks the code rather than the result - the same technique the QRNG register
    gate uses, and for the same reason.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(run_quantum)))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            if re.search(r"correct|accurate|hit|match", name, re.I):
                bad.append(f"{name} at line {node.lineno} of run_quantum")
    assert not bad, (
        "a correctness-style counter is back in `run_quantum`: " + ", ".join(bad)
        + ". If something genuinely IS being compared now, this test should be re-derived to assert "
        "what it compares against - but an unconditional increment is not a measurement, and the "
        "last one spent three rounds justifying an accuracy of 1.0.")
