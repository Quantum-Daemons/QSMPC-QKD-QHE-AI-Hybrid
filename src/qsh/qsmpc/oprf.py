"""ristretto255 and the RFC 9497 OPRF (mode 0x00, suite ristretto255-SHA512), on libsodium or in Python.

The full deployment's private set intersection (:mod:`qsh.qsmpc.psi`) needs an oblivious
pseudorandom function. ``cryptography`` exposes no hash-to-curve and no prime-order group, and
PyNaCl exposes none of libsodium's ristretto255 functions (1.6.2, read 2026-11-01). This module
implements exactly what the PSI uses, from the primary sources, twice: in pure Python, below, and on
libsodium's ristretto255 through ``pysodium`` (Round 85), which the PSI uses whenever the host has
libsodium (:data:`BACKEND` says which ran, and the run reports it). The browser runs the same
functions from ``@noble/curves`` (``ristretto255_oprf``); ``tests/test_r84_oprf_psi.py`` checks the
tiers byte for byte, and ``tests/test_r85_oprf_on_libsodium.py`` checks the two Python backends
against each other and the RFC vectors.

- **ristretto255** (RFC 9496, section 4): the prime-order group of order
  ``2^252 + 27742317777372353535851937790883648493`` built on Curve25519 points; field arithmetic
  modulo ``2^255 - 19`` (RFC 9496 section 2), SQRT_RATIO_M1 (section 4.2), Decode (4.3.1), Encode
  (4.3.2), Equals (4.3.3) and the element derivation from 64 bytes (4.3.4). Points are kept in
  extended twisted Edwards coordinates with the addition and doubling formulas of RFC 8032 section
  5.1.4 (a = -1).
- **expand_message_xmd** with SHA-512 (RFC 9380 section 5.3.1) and **hash_to_ristretto255**
  (RFC 9380 Appendix B: the element derivation of 64 expanded bytes).
- **The OPRF** (RFC 9497): the context string ``"OPRFV1-" || I2OSP(0x00, 1) || "-ristretto255-SHA512"``
  (section 3.1), HashToGroup and HashToScalar with the DSTs of section 4.1, DeriveKeyPair (3.2.1),
  and Blind, BlindEvaluate, Finalize and Evaluate (3.3.1). The verifiable and partially oblivious
  modes are not implemented: the PSI runs mode 0x00 only.

CONSTANT TIME. RFC 9497 section 7.4 requires constant-time operations on secret data. The pure-Python
code does not meet it: Python integers are not constant-time, and its scalar multiplication is a
4-bit fixed-window walk that branches on the scalar. On libsodium every operation on a secret (the
blind, its inverse, the key, the hashed input) is libsodium's, which is written to run in constant
time, and the secret scalars stay byte strings: they are checked by reducing them and comparing with
:func:`hmac.compare_digest`, never converted to Python integers. What is left in Python is SHA-512
(``hashlib``) and length checks. Side channels on the host stay out of scope for this project
(``docs/threat-model.md``, "Explicitly out of scope"); the pure-Python code remains the fallback where
libsodium is absent, and the cross-check where it is present.

NOT POST-QUANTUM. Its security rests on a discrete-logarithm assumption (the One-More Gap
Computational Diffie-Hellman assumption, RFC 9497 section 7.2.1), which a quantum computer running
Shor's algorithm would break, on either backend.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from types import SimpleNamespace

try:  # libsodium's ristretto255, through pysodium (Round 85)
    import pysodium as _sodium
except (ImportError, OSError, ValueError):  # pysodium absent, or no libsodium on the host
    _sodium = None
if _sodium is not None and not _sodium.sodium_version_check(1, 0, 18):  # ristretto255 is 1.0.18+
    _sodium = None

# --------------------------------------------------------------------------- #
#  Field and group constants (RFC 9496 sections 2, 4 and 4.1; RFC 8032 section 5.1)
# --------------------------------------------------------------------------- #
P = 2**255 - 19
#: The group order l (RFC 9496 section 4; RFC 9497 section 4.1 Order()).
L = 2**252 + 27742317777372353535851937790883648493

D = 37095705934669439343138083508754565189542113879843219016388785533085940283555
SQRT_M1 = 19681161376707505956807079304988542015446066515923890162744021073123829784752
SQRT_AD_MINUS_ONE = 25063068953384623474111414158702152701244531502492656460079210482610430750235
INVSQRT_A_MINUS_D = 54469307008909316920995813868745141605393597292927456921205312896311721017578
ONE_MINUS_D_SQ = 1159843021668779879193775521855586647937357759715417654439879720876111806838
D_MINUS_ONE_SQ = 40440834346308536858101042469323190826248399146238708352240133220865137265952

#: The Curve25519 (edwards25519) base point, which internally represents the canonical
#: ristretto255 generator (RFC 9496 section 4); coordinates from RFC 8032 section 5.1.
_BX = 15112221349535400772501151409588531511454012693041857206046113283949847762202
_BY = 46316835694926478169428394003475163141307993866256225615783033603165251855960

_D2 = 2 * D % P


def _is_negative(e: int) -> bool:
    """IS_NEGATIVE (RFC 9496 section 2.1): the least nonnegative representative is odd."""
    return (e % P) & 1 == 1


def _ct_abs(e: int) -> int:
    """CT_ABS (RFC 9496 section 2.2), without the constant time."""
    e %= P
    return (-e) % P if e & 1 else e


def sqrt_ratio_m1(u: int, v: int) -> tuple[bool, int]:
    """SQRT_RATIO_M1(u, v), RFC 9496 section 4.2, returning (was_square, r)."""
    u %= P
    v %= P
    v3 = v * v % P * v % P
    v7 = v3 * v3 % P * v % P
    r = u * v3 % P * pow(u * v7 % P, (P - 5) // 8, P) % P
    check = v * r % P * r % P
    correct_sign_sqrt = check == u
    flipped_sign_sqrt = check == (-u) % P
    flipped_sign_sqrt_i = check == (-u * SQRT_M1) % P
    if flipped_sign_sqrt or flipped_sign_sqrt_i:
        r = r * SQRT_M1 % P
    return correct_sign_sqrt or flipped_sign_sqrt, _ct_abs(r)


class DecodeError(ValueError):
    """A byte string that is not the canonical encoding of a ristretto255 element (RFC 9496 4.3.1)."""


class InvalidInputError(ValueError):
    """An input that maps to the group identity, or one RFC 9497 does not accept (section 3.3.1)."""


class Element:
    """A ristretto255 group element, held as an internal representation in extended coordinates.

    Two internal representations may stand for the same element; compare with :meth:`__eq__`
    (RFC 9496 section 4.3.3) or by encoding, never by coordinates.
    """

    __slots__ = ("X", "Y", "Z", "T")

    def __init__(self, X: int, Y: int, Z: int, T: int) -> None:
        self.X, self.Y, self.Z, self.T = X % P, Y % P, Z % P, T % P

    # -- group law: RFC 8032 section 5.1.4, a = -1 -------------------------------------------- #
    def __add__(self, other: "Element") -> "Element":
        A = (self.Y - self.X) * (other.Y - other.X) % P
        B = (self.Y + self.X) * (other.Y + other.X) % P
        C = self.T * _D2 % P * other.T % P
        D_ = 2 * self.Z * other.Z % P
        E, F, G, H = B - A, D_ - C, D_ + C, B + A
        return Element(E * F, G * H, F * G, E * H)

    def double(self) -> "Element":
        A = self.X * self.X % P
        B = self.Y * self.Y % P
        C = 2 * self.Z * self.Z % P
        H = A + B
        s = self.X + self.Y
        E = H - s * s
        G = A - B
        F = C + G
        return Element(E * F, G * H, F * G, E * H)

    def __neg__(self) -> "Element":
        return Element(-self.X, self.Y, self.Z, -self.T)

    def __mul__(self, scalar: int) -> "Element":
        """Scalar multiplication by a 4-bit fixed window. Not constant-time (see the module note)."""
        k = scalar % L
        table = [IDENTITY, self]
        for _ in range(14):
            table.append(table[-1] + self)
        acc = IDENTITY
        for shift in range(252, -1, -4):
            acc = acc.double().double().double().double()
            nibble = (k >> shift) & 0xF
            if nibble:
                acc = acc + table[nibble]
        return acc

    __rmul__ = __mul__

    def __eq__(self, other: object) -> bool:
        """Equals, RFC 9496 section 4.3.3."""
        if not isinstance(other, Element):
            return NotImplemented
        return (self.X * other.Y - self.Y * other.X) % P == 0 or (self.Y * other.Y - self.X * other.X) % P == 0

    def __hash__(self) -> int:
        return hash(self.encode())

    def is_identity(self) -> bool:
        return self == IDENTITY

    # -- encoding: RFC 9496 sections 4.3.1 and 4.3.2 ------------------------------------------ #
    def encode(self) -> bytes:
        x0, y0, z0, t0 = self.X, self.Y, self.Z, self.T
        u1 = (z0 + y0) * (z0 - y0) % P
        u2 = x0 * y0 % P
        _, invsqrt = sqrt_ratio_m1(1, u1 * u2 % P * u2 % P)
        den1 = invsqrt * u1 % P
        den2 = invsqrt * u2 % P
        z_inv = den1 * den2 % P * t0 % P
        ix0 = x0 * SQRT_M1 % P
        iy0 = y0 * SQRT_M1 % P
        enchanted_denominator = den1 * INVSQRT_A_MINUS_D % P
        rotate = _is_negative(t0 * z_inv)
        x = iy0 if rotate else x0
        y = ix0 if rotate else y0
        z = z0
        den_inv = enchanted_denominator if rotate else den2
        if _is_negative(x * z_inv):
            y = -y % P
        s = _ct_abs(den_inv * (z - y))
        return s.to_bytes(32, "little")

    @staticmethod
    def decode(data: bytes) -> "Element":
        if len(data) != 32:
            raise DecodeError(f"a ristretto255 encoding is 32 bytes, got {len(data)}")
        s = int.from_bytes(data, "little")
        if s >= P:
            raise DecodeError("non-canonical field encoding")
        if _is_negative(s):
            raise DecodeError("negative field element")
        ss = s * s % P
        u1 = (1 - ss) % P
        u2 = (1 + ss) % P
        u2_sqr = u2 * u2 % P
        v = (-(D * u1 % P * u1) - u2_sqr) % P
        was_square, invsqrt = sqrt_ratio_m1(1, v * u2_sqr % P)
        den_x = invsqrt * u2 % P
        den_y = invsqrt * den_x % P * v % P
        x = _ct_abs(2 * s * den_x)
        y = u1 * den_y % P
        t = x * y % P
        if not was_square or _is_negative(t) or y == 0:
            raise DecodeError("not the encoding of a ristretto255 element")
        return Element(x, y, 1, t)

    def __repr__(self) -> str:
        return f"Element({self.encode().hex()})"


IDENTITY = Element(0, 1, 1, 0)
GENERATOR = Element(_BX, _BY, 1, _BX * _BY)


def _map(b32: bytes) -> Element:
    """MAP, RFC 9496 section 4.3.4: one 32-byte half of the element derivation."""
    t = (int.from_bytes(b32, "little") & ((1 << 255) - 1)) % P
    r = SQRT_M1 * t % P * t % P
    u = (r + 1) * ONE_MINUS_D_SQ % P
    v = (-1 - r * D) * (r + D) % P
    was_square, s = sqrt_ratio_m1(u, v)
    s_prime = -_ct_abs(s * t) % P
    if not was_square:
        s = s_prime
    c = P - 1 if was_square else r
    N = (c * (r - 1) % P * D_MINUS_ONE_SQ - v) % P
    w0 = 2 * s * v % P
    w1 = N * SQRT_AD_MINUS_ONE % P
    w2 = (1 - s * s) % P
    w3 = (1 + s * s) % P
    return Element(w0 * w3, w2 * w1, w1 * w3, w0 * w2)


def element_derivation(b64: bytes) -> Element:
    """The element derivation function of RFC 9496 section 4.3.4: MAP(b[0:32]) + MAP(b[32:64])."""
    if len(b64) != 64:
        raise ValueError(f"the element derivation takes 64 bytes, got {len(b64)}")
    return _map(b64[:32]) + _map(b64[32:])


# --------------------------------------------------------------------------- #
#  Hashing (RFC 9380 section 5.3.1 and Appendix B)
# --------------------------------------------------------------------------- #
def expand_message_xmd(msg: bytes, dst: bytes, len_in_bytes: int) -> bytes:
    """expand_message_xmd with H = SHA-512 (b_in_bytes 64, s_in_bytes 128), RFC 9380 section 5.3.1."""
    ell = -(-len_in_bytes // 64)
    if ell > 255 or len_in_bytes > 65535 or len(dst) > 255:
        raise ValueError("expand_message_xmd: requested length or DST out of range")
    dst_prime = dst + bytes([len(dst)])
    msg_prime = bytes(128) + msg + len_in_bytes.to_bytes(2, "big") + b"\x00" + dst_prime
    b0 = hashlib.sha512(msg_prime).digest()
    blocks = [hashlib.sha512(b0 + b"\x01" + dst_prime).digest()]
    for i in range(2, ell + 1):
        mixed = bytes(x ^ y for x, y in zip(b0, blocks[-1]))
        blocks.append(hashlib.sha512(mixed + bytes([i]) + dst_prime).digest())
    return b"".join(blocks)[:len_in_bytes]


def hash_to_ristretto255(msg: bytes, dst: bytes) -> Element:
    """hash_to_ristretto255, RFC 9380 Appendix B: the element derivation of 64 expanded bytes."""
    return element_derivation(expand_message_xmd(msg, dst, 64))


# --------------------------------------------------------------------------- #
#  The OPRF, mode 0x00, suite ristretto255-SHA512 (RFC 9497)
# --------------------------------------------------------------------------- #
SUITE_ID = b"ristretto255-SHA512"
MODE_OPRF = 0x00
#: CreateContextString(modeOPRF, "ristretto255-SHA512"), RFC 9497 section 3.1.
CONTEXT_STRING = b"OPRFV1-" + bytes([MODE_OPRF]) + b"-" + SUITE_ID
_DST_GROUP = b"HashToGroup-" + CONTEXT_STRING
_DST_SCALAR = b"HashToScalar-" + CONTEXT_STRING
_DST_DERIVE = b"DeriveKeyPair" + CONTEXT_STRING
#: Ne, Ns and Nh for this suite (RFC 9497 section 4.1).
NE, NS, NH = 32, 32, 64
#: PrivateInput values are length-prefixed with two bytes, so at most 2^16 - 1 bytes (section 1.2).
MAX_INPUT = 0xFFFF


def hash_to_group(msg: bytes) -> Element:
    """HashToGroup of RFC 9497 section 4.1, DST = "HashToGroup-" || contextString."""
    return hash_to_ristretto255(msg, _DST_GROUP)


def hash_to_scalar(msg: bytes, dst: bytes = _DST_SCALAR) -> int:
    """HashToScalar of RFC 9497 section 4.1: 64 expanded bytes, little-endian, reduced mod l."""
    return int.from_bytes(expand_message_xmd(msg, dst, 64), "little") % L


def serialize_scalar(s: int) -> bytes:
    return (s % L).to_bytes(NS, "little")


def deserialize_scalar(buf: bytes) -> int:
    """DeserializeScalar (RFC 9497 section 4.1): 32 bytes little-endian, rejected unless below l."""
    if len(buf) != NS:
        raise DecodeError(f"a scalar is {NS} bytes, got {len(buf)}")
    s = int.from_bytes(buf, "little")
    if s >= L:
        raise DecodeError("scalar not below the group order")
    return s


def serialize_element(e: Element) -> bytes:
    return e.encode()


def deserialize_element(buf: bytes) -> Element:
    """DeserializeElement (RFC 9497 section 4.1): Decode, then reject the identity (section 3.3)."""
    e = Element.decode(buf)
    if e.is_identity():
        raise DecodeError("the group identity is not accepted over the wire")
    return e


def random_scalar() -> int:
    """RandomScalar: a uniform nonzero scalar (RFC 9497 sections 2.1 and 4.7.2).

    48 bytes from the OS CSPRNG reduced mod l: section 4.7.2's ceil((3 * ceil(log2 l) / 2) / 8)
    = ceil(379.5 / 8) = 48 bytes, so the reduction bias is below 2^-128. Zero is redrawn.
    """
    while True:
        s = int.from_bytes(secrets.token_bytes(48), "little") % L
        if s:
            return s


def _check_input(data: bytes) -> bytes:
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("OPRF inputs are byte strings")
    if len(data) > MAX_INPUT:
        raise InvalidInputError(f"an OPRF input is at most {MAX_INPUT} bytes (RFC 9497 section 1.2)")
    return bytes(data)


def _py_derive_key_pair(seed: bytes, info: bytes) -> tuple[bytes, bytes]:
    """DeriveKeyPair, RFC 9497 section 3.2.1: (serialized skS, serialized pkS)."""
    if len(seed) != 32:
        raise ValueError("DeriveKeyPair takes a 32-byte seed")
    info = _check_input(info)
    derive_input = seed + len(info).to_bytes(2, "big") + info
    for counter in range(256):
        sk = hash_to_scalar(derive_input + bytes([counter]), _DST_DERIVE)
        if sk:
            return serialize_scalar(sk), serialize_element(GENERATOR * sk)
    raise ValueError("DeriveKeyPairError")


def _py_generate_key_pair() -> tuple[bytes, bytes]:
    """GenerateKeyPair, RFC 9497 section 3.2: a fresh random (skS, pkS)."""
    sk = random_scalar()
    return serialize_scalar(sk), serialize_element(GENERATOR * sk)


def _py_blind(data: bytes, blind_scalar: bytes | None = None) -> tuple[bytes, bytes]:
    """Blind, RFC 9497 section 3.3.1: (serialized blind, serialized blindedElement).

    ``blind_scalar`` fixes the blind, as the RFC's test vectors do; a run leaves it None and draws a
    fresh RandomScalar.
    """
    data = _check_input(data)
    r = random_scalar() if blind_scalar is None else deserialize_scalar(blind_scalar)
    if r == 0:
        raise InvalidInputError("the blind must be nonzero")
    point = hash_to_group(data)
    if point.is_identity():
        raise InvalidInputError("the input maps to the group identity")
    return serialize_scalar(r), serialize_element(point * r)


def _py_blind_evaluate(sk: bytes, blinded_element: bytes) -> bytes:
    """BlindEvaluate, RFC 9497 section 3.3.1: skS * blindedElement, serialized."""
    return serialize_element(deserialize_element(blinded_element) * deserialize_scalar(sk))


def _finalize_hash(data: bytes, element: bytes) -> bytes:
    return hashlib.sha512(len(data).to_bytes(2, "big") + data + len(element).to_bytes(2, "big")
                          + element + b"Finalize").digest()


def _py_finalize(data: bytes, blind_scalar: bytes, evaluated_element: bytes) -> bytes:
    """Finalize, RFC 9497 section 3.3.1: Hash of the input and ScalarInverse(blind) * evaluatedElement."""
    data = _check_input(data)
    r = deserialize_scalar(blind_scalar)
    if r == 0:
        raise InvalidInputError("the blind must be nonzero")
    unblinded = deserialize_element(evaluated_element) * pow(r, -1, L)
    return _finalize_hash(data, serialize_element(unblinded))


def _py_evaluate(sk: bytes, data: bytes) -> bytes:
    """Evaluate, RFC 9497 section 3.3.1: the PRF output computed by a holder of skS directly."""
    data = _check_input(data)
    point = hash_to_group(data)
    if point.is_identity():
        raise InvalidInputError("the input maps to the group identity")
    return _finalize_hash(data, serialize_element(point * deserialize_scalar(sk)))


# --------------------------------------------------------------------------- #
#  The same OPRF on libsodium (Round 85)
# --------------------------------------------------------------------------- #
#: The encoding of the identity element: 32 zero bytes (RFC 9496 section 4.3.2).
_IDENTITY_BYTES = bytes(32)


def _na_scalar(buf: bytes) -> bytes:
    """DeserializeScalar on libsodium: 32 bytes, rejected unless below l.

    A canonical scalar is the one that reducing leaves unchanged, and the comparison is
    :func:`hmac.compare_digest`, so the check says whether the scalar was canonical and nothing else
    about it; the pure-Python twin converts it to an integer and compares."""
    buf = bytes(buf)
    if len(buf) != NS:
        raise DecodeError(f"a scalar is {NS} bytes, got {len(buf)}")
    if not hmac.compare_digest(_sodium.crypto_core_ristretto255_scalar_reduce(buf + bytes(NS)), buf):
        raise DecodeError("scalar not below the group order")
    return buf


def _na_nonzero(scalar: bytes) -> bytes:
    if hmac.compare_digest(scalar, bytes(NS)):
        raise InvalidInputError("the blind must be nonzero")
    return scalar


def _na_element(buf: bytes) -> bytes:
    """DeserializeElement on libsodium: a canonical encoding, and not the identity (section 3.3)."""
    buf = bytes(buf)
    if len(buf) != NE or not _sodium.crypto_core_ristretto255_is_valid_point(buf):
        raise DecodeError("not a canonical ristretto255 encoding")
    if buf == _IDENTITY_BYTES:          # public data: what crossed the wire
        raise DecodeError("the group identity is not accepted over the wire")
    return buf


def _na_mul(scalar: bytes, element: bytes) -> bytes:
    """scalar * element. libsodium refuses a product that is the identity, which a nonzero scalar
    times a non-identity element of a prime-order group never is."""
    try:
        return _sodium.crypto_scalarmult_ristretto255(scalar, element)
    except ValueError as e:
        raise InvalidInputError("the product is the group identity") from e


def _na_hash_to_group(msg: bytes) -> bytes:
    """HashToGroup: libsodium's element derivation (``crypto_core_ristretto255_from_hash``, RFC 9496
    section 4.3.4) of the same 64 expanded bytes as :func:`hash_to_group`."""
    point = _sodium.crypto_core_ristretto255_from_hash(expand_message_xmd(msg, _DST_GROUP, 64))
    if hmac.compare_digest(point, _IDENTITY_BYTES):
        raise InvalidInputError("the input maps to the group identity")
    return point


def _na_derive_key_pair(seed: bytes, info: bytes) -> tuple[bytes, bytes]:
    if len(seed) != 32:
        raise ValueError("DeriveKeyPair takes a 32-byte seed")
    info = _check_input(info)
    derive_input = seed + len(info).to_bytes(2, "big") + info
    for counter in range(256):
        sk = _sodium.crypto_core_ristretto255_scalar_reduce(
            expand_message_xmd(derive_input + bytes([counter]), _DST_DERIVE, 64))
        if not hmac.compare_digest(sk, bytes(NS)):
            return sk, _sodium.crypto_scalarmult_ristretto255_base(sk)
    raise ValueError("DeriveKeyPairError")


def _na_generate_key_pair() -> tuple[bytes, bytes]:
    sk = _sodium.crypto_core_ristretto255_scalar_random()      # uniform in [1, l)
    return sk, _sodium.crypto_scalarmult_ristretto255_base(sk)


def _na_blind(data: bytes, blind_scalar: bytes | None = None) -> tuple[bytes, bytes]:
    data = _check_input(data)
    r = (_sodium.crypto_core_ristretto255_scalar_random() if blind_scalar is None
         else _na_nonzero(_na_scalar(blind_scalar)))
    return r, _na_mul(r, _na_hash_to_group(data))


def _na_blind_evaluate(sk: bytes, blinded_element: bytes) -> bytes:
    return _na_mul(_na_scalar(sk), _na_element(blinded_element))


def _na_finalize(data: bytes, blind_scalar: bytes, evaluated_element: bytes) -> bytes:
    data = _check_input(data)
    inverse = _sodium.crypto_core_ristretto255_scalar_invert(_na_nonzero(_na_scalar(blind_scalar)))
    return _finalize_hash(data, _na_mul(inverse, _na_element(evaluated_element)))


def _na_evaluate(sk: bytes, data: bytes) -> bytes:
    data = _check_input(data)
    return _finalize_hash(data, _na_mul(_na_scalar(sk), _na_hash_to_group(data)))


# --------------------------------------------------------------------------- #
#  Which one runs
# --------------------------------------------------------------------------- #
_API = ("derive_key_pair", "generate_key_pair", "blind", "blind_evaluate", "finalize", "evaluate")
#: The pure-Python implementation: the fallback, and the cross-check of libsodium.
PURE_PYTHON = SimpleNamespace(name="pure-python", **{f: globals()[f"_py_{f}"] for f in _API})
#: libsodium's, or None where the host has no libsodium 1.0.18 or later.
LIBSODIUM = (SimpleNamespace(name=f"libsodium {_sodium.sodium.sodium_version_string().decode()}",
                             **{f: globals()[f"_na_{f}"] for f in _API})
             if _sodium is not None else None)
_ACTIVE = LIBSODIUM or PURE_PYTHON
#: What runs, as the run reports it (metric ``psi_oprf_backend``): libsodium whenever the host has it.
BACKEND = _ACTIVE.name

derive_key_pair = _ACTIVE.derive_key_pair
generate_key_pair = _ACTIVE.generate_key_pair
blind = _ACTIVE.blind
blind_evaluate = _ACTIVE.blind_evaluate
finalize = _ACTIVE.finalize
evaluate = _ACTIVE.evaluate
