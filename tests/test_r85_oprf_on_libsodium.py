"""The full deployment's OPRF runs on libsodium, and agrees with the pure-Python one byte for byte.

WHY THIS EXISTS (Round 85)

The Python OPRF that the full deployment's PSI uses (:mod:`qsh.qsmpc.oprf`) was pure Python, which
is not constant-time: Python integers are not, and its scalar multiplication branches on the
scalar, while RFC 9497 section 7.4 requires constant-time operations on secret data. The owner chose
libsodium for it, through pysodium (PyNaCl, the first choice, exposes none of libsodium's
ristretto255 functions). The pure-Python code stays, as the fallback where the host has no
libsodium and as the cross-check here.

WHAT IS ASSERTED

- libsodium is what runs wherever pysodium loads it, and the run reports it (``psi_oprf_backend``).
  In CI ``QSH_REQUIRE_LIBSODIUM=1`` makes its absence a failure rather than a skip.
- Both backends reproduce RFC 9497's ristretto255-SHA512 OPRF vectors (Appendix A.1.1).
- On random keys, blinds and inputs the two produce the same blinded and evaluated elements and
  outputs, and a run that mixes them (one bank on each) still finds the intersection.
- They reject the same inputs: non-canonical scalars, a zero blind, RFC 9496's invalid encodings and
  the identity element.
- On libsodium no secret scalar becomes a Python integer.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import secrets
from pathlib import Path

import pytest

from qsh.qsmpc import oprf

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("r84_oprf_vectors", REPO / "tests" / "test_r84_oprf_psi.py")
_r84 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_r84)


def _libsodium_or_skip():
    if oprf.LIBSODIUM is not None:
        return oprf.LIBSODIUM
    msg = ("libsodium 1.0.18 or later is not loadable here (pysodium absent, or no libsodium shared "
           "library), so the OPRF runs on pure Python. This is a SKIP, not a pass.")
    if os.environ.get("QSH_REQUIRE_LIBSODIUM") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


def _backends():
    return [oprf.PURE_PYTHON, _libsodium_or_skip()]


def test_libsodium_is_what_runs_and_what_the_run_reports() -> None:
    lib = _libsodium_or_skip()
    assert oprf.BACKEND == lib.name and lib.name.startswith("libsodium 1.0.")
    assert (oprf.blind, oprf.blind_evaluate, oprf.finalize, oprf.evaluate) == (
        lib.blind, lib.blind_evaluate, lib.finalize, lib.evaluate)
    ctx, m, logs = _r84._python_stage(["acct-1", "acct-2"], ["acct-2", "acct-3"], os.urandom(32))
    assert m["psi_oprf_backend"] == lib.name
    assert f"its OPRF on {lib.name}" in next(x for x in logs if x.startswith("PSI complete"))


@pytest.mark.parametrize("which", ["pure-python", "libsodium"])
def test_each_backend_reproduces_rfc9497_a11(which) -> None:
    impl = oprf.PURE_PYTHON if which == "pure-python" else _libsodium_or_skip()
    sk, pk = impl.derive_key_pair(bytes.fromhex(_r84.RFC9497_SEED), bytes.fromhex(_r84.RFC9497_KEYINFO))
    assert sk.hex() == _r84.RFC9497_SKSM
    assert pk == (oprf.GENERATOR * int.from_bytes(sk, "little")).encode()
    for v in _r84.RFC9497_A11:
        x = bytes.fromhex(v["input"])
        blind, blinded = impl.blind(x, bytes.fromhex(_r84.RFC9497_BLIND))
        assert blinded.hex() == v["blinded"], "BlindedElement"
        evaluated = impl.blind_evaluate(sk, blinded)
        assert evaluated.hex() == v["evaluated"], "EvaluationElement"
        assert impl.finalize(x, blind, evaluated).hex() == v["output"], "Output (Finalize)"
        assert impl.evaluate(sk, x).hex() == v["output"], "Output (Evaluate)"


def test_the_backends_agree_on_random_keys_blinds_and_inputs() -> None:
    py, na = _backends()
    for _ in range(40):
        sk, pk = py.derive_key_pair(secrets.token_bytes(32), secrets.token_bytes(secrets.randbelow(20)))
        x = secrets.token_bytes(secrets.randbelow(80))
        r = oprf.serialize_scalar(oprf.random_scalar())
        b_py, m_py = py.blind(x, r)
        b_na, m_na = na.blind(x, r)
        assert (b_py, m_py) == (b_na, m_na)
        z_py, z_na = py.blind_evaluate(sk, m_py), na.blind_evaluate(sk, m_na)
        assert z_py == z_na
        out = py.finalize(x, b_py, z_py)
        assert out == na.finalize(x, b_na, z_na) == py.evaluate(sk, x) == na.evaluate(sk, x)
        # One bank on each backend: the messages are the RFC's, so they interoperate.
        assert na.finalize(x, b_py, py.blind_evaluate(sk, m_py)) == out
    seed, info = secrets.token_bytes(32), b"qsh psi test"
    assert py.derive_key_pair(seed, info) == na.derive_key_pair(seed, info)
    # Fresh draws on libsodium are valid and consistent with the pure-Python arithmetic.
    sk, pk = na.generate_key_pair()
    assert pk == (oprf.GENERATOR * oprf.deserialize_scalar(sk)).encode()
    x = b"acct-0001"
    blind, blinded = na.blind(x)
    assert py.finalize(x, blind, py.blind_evaluate(sk, blinded)) == py.evaluate(sk, x)


def test_the_backends_reject_the_same_inputs() -> None:
    py, na = _backends()
    sk, _ = py.derive_key_pair(bytes(32), b"")
    _, blinded = py.blind(b"x", oprf.serialize_scalar(7))
    bad_scalars = [oprf.L.to_bytes(32, "little"), (oprf.L + 1).to_bytes(32, "little"),
                   bytes([0xFF] * 32), bytes(31)]
    for impl in (py, na):
        for s in bad_scalars:
            with pytest.raises((oprf.DecodeError, ValueError)):
                impl.blind_evaluate(s, blinded)
            with pytest.raises((oprf.DecodeError, ValueError)):
                impl.blind(b"x", s)
        with pytest.raises(oprf.InvalidInputError):
            impl.blind(b"x", bytes(32))
        with pytest.raises(oprf.InvalidInputError):
            impl.finalize(b"x", bytes(32), blinded)
        for enc in _r84.RFC9496_A2 + ["00" * 32]:
            with pytest.raises(oprf.DecodeError):
                impl.blind_evaluate(sk, bytes.fromhex(enc))
            with pytest.raises(oprf.DecodeError):
                impl.finalize(b"x", oprf.serialize_scalar(7), bytes.fromhex(enc))
        with pytest.raises(oprf.InvalidInputError):
            impl.evaluate(sk, bytes(oprf.MAX_INPUT + 1))


def test_no_secret_scalar_becomes_a_python_integer_on_libsodium() -> None:
    _libsodium_or_skip()
    for name in oprf._API:
        src = inspect.getsource(getattr(oprf, f"_na_{name}"))
        for integer_arithmetic in ("int.from_bytes", "% L", "pow(", "deserialize_scalar(", "random_scalar("):
            assert integer_arithmetic not in src, (name, integer_arithmetic)
    for helper in (oprf._na_scalar, oprf._na_nonzero, oprf._na_mul, oprf._na_hash_to_group):
        src = inspect.getsource(helper)
        assert "int.from_bytes" not in src and "% L" not in src, helper.__name__
