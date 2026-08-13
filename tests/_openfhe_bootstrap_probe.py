"""Standalone probe: does OpenFHE CKKS bootstrapping actually work on this host?

Run as a subprocess by ``tests/test_round2.py::test_openfhe_bootstrap``, deliberately NOT as a
test itself (the leading underscore keeps pytest from collecting it).

WHY OUT OF PROCESS. Misconfigured bootstrapping does not raise -- it SIGSEGVs the interpreter
from inside the C++ library, which no ``try``/``except`` in the test can catch. In-process that
kills the whole pytest run (exit 139) and takes the other 290 tests with it, so the only way to
finish a local suite was to deselect this test, i.e. to stop checking. Out of process the crash
is just a return code the parent can assert on and report as an ordinary failure.

Protocol: one JSON object on the last stdout line. Exit 0 = bootstrapping verified,
42 = openfhe not importable (the one honest reason to skip), anything else = failure.
"""

from __future__ import annotations

import json
import sys

EXIT_OK = 0
EXIT_NO_OPENFHE = 42

VALUES = [0.5, 0.25, 0.1]


def main() -> int:
    try:
        import openfhe  # noqa: F401
    except ImportError as e:
        print(json.dumps({"import_error": str(e)}))
        return EXIT_NO_OPENFHE

    from qsh.fhe.backends import OpenFHEBackend

    b = OpenFHEBackend(use_bootstrap=True)
    if not b.bootstrap_enabled:
        # Unreachable by construction: the backend raises rather than reporting False. Kept so
        # that reintroducing a silent-degrade path fails here instead of skipping upstream.
        print(json.dumps({"error": "bootstrap_enabled is False after use_bootstrap=True"}))
        return 1

    ct = b.encrypt(VALUES)
    # Spend levels using only the public API: multiplying by 1.0 leaves the plaintext alone and
    # costs exactly one level, so this depletes the ciphertext without changing what it holds.
    # Stop at total_depth - 2: on the final tower EvalBootstrap cannot run at all.
    # The bound turns "mul_plain stopped consuming levels" into a fast, legible failure rather
    # than a loop that spins until the parent's timeout fires.
    for _ in range(b.total_depth + 2):
        if ct.GetLevel() >= b.total_depth - 2:
            break
        ct = b.mul_plain(ct, [1.0] * len(VALUES))
    else:
        print(json.dumps({"error": f"level stuck at {ct.GetLevel()} of {b.total_depth}; "
                                   f"mul_plain is not consuming levels"}))
        return 1
    level_before = ct.GetLevel()
    depleted = b.decrypt(ct)[: len(VALUES)]

    out = b.bootstrap(ct)
    level_after = out.GetLevel()
    refreshed = b.decrypt(out)[: len(VALUES)]

    # A refresh that does not buy back usable levels is not a refresh. Spending one on the
    # bootstrapped ciphertext is the cheapest proof that the level budget really was restored.
    reused = b.decrypt(b.mul_plain(out, [2.0] * len(VALUES)))[: len(VALUES)]

    print(json.dumps({
        "ring_dimension": b._cc.GetRingDimension(),
        "mult_depth": b.mult_depth,
        "total_depth": b.total_depth,
        "level_before": level_before,
        "level_after": level_after,
        "levels_regained": level_before - level_after,
        "expected": VALUES,
        "depleted": depleted,
        "refreshed": refreshed,
        "reused": reused,
    }))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
