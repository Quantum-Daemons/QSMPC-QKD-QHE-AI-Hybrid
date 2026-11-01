"""Private set intersection on the RFC 9497 OPRF: a semi-honest DH-OPRF PSI between two banks.

The finance use case's matching of flagged accounts, in the full deployment. The browser runs the
same protocol (``web/src/lib/oprfPsi.ts``, on ``@noble/curves``' ``ristretto255_oprf``), and the two
produce byte-identical OPRF messages for the same key, blinds and inputs
(``tests/test_r84_oprf_psi.py``).

THE PROTOCOL, one run. Bank A is the OPRF client and Bank B the OPRF server, in RFC 9497's mode
0x00 with the suite ristretto255-SHA512 (RFC 9497 section 4.1; the group is RFC 9496's
ristretto255; :mod:`qsh.qsmpc.oprf` implements both):

1. Bank B draws a fresh OPRF key k for this run (GenerateKeyPair, RFC 9497 section 3.2).
2. Bank A blinds each of its IDs x, M_x = r_x * HashToGroup(x) with a fresh random r_x (Blind,
   section 3.3.1), and sends the M_x.
3. Bank B returns Z_x = k * M_x for each one (BlindEvaluate), and, for each of its own IDs y, the
   PRF output F_k(y) (Evaluate), in an order shuffled with the OS CSPRNG, so the order says nothing
   about its list.
4. Bank A computes F_k(x) = Finalize(x, r_x, Z_x) for each of its IDs and keeps those whose output
   is among Bank B's: that is the intersection.

WHAT EACH BANK LEARNS, when both follow the protocol (semi-honest):

- Bank A learns the intersection and |B|, the number of outputs it receives. It never holds k, so it
  cannot compute F_k offline: it can test an ID against Bank B's list only by having Bank B answer a
  query for it, and Bank B answers exactly the |A| queries Bank A sends.
- Bank B learns |A|, the number of blinded elements, and nothing else about Bank A's IDs: a blinded
  element is a uniform group element whatever the ID, even to an unbounded adversary (RFC 9497
  section 7.1, "Unconditional input secrecy"). Bank B does not learn the intersection; if it needs
  it, Bank A has to send it, which this code does not do.

WHAT IT DOES NOT PROVIDE:

- No malicious security. A bank that deviates is not detected: Bank A could blind IDs it does not
  hold, testing up to |A| candidates of its choosing, and Bank B could evaluate under a different key
  or send made-up outputs and corrupt Bank A's result. Mode 0x00 carries no proof; the verifiable
  mode 0x01 is not used.
- Not post-quantum. The pseudorandomness of F_k, which is what keeps Bank B's list from Bank A, rests
  on the One-More Gap Computational Diffie-Hellman assumption in ristretto255 (RFC 9497 section
  7.2.1). A Bank A with a quantum computer could solve k from any (M_x, Z_x) pair it received and
  then evaluate every candidate ID against Bank B's outputs. Bank A's own input secrecy (above) holds
  regardless.
- Here both banks run in one process: they are notional parties, and no message crosses a network.

THE PQC SESSION KEY. The three messages are sealed with AES-256-GCM (``SecureChannel``, the
transport's AEAD record layer) under two keys HKDF-SHA256 derives from the run's PQC session key, one
per direction (:func:`transcript_keys`). That is what the key does here: it keeps the transcript from
anyone who is not one of the two banks, including someone who records it now and later attacks
ristretto255 with a quantum computer, for as long as the ML-KEM-768 + X25519 hybrid holds. What it
does not do is hide anything between the banks, since each opens what the other sealed; that
privacy comes from the OPRF alone. In one process no network carries the messages, so the sealing
shows the path they would take rather than protecting a wire.

The key k is fresh per run: rotating the server's key is one of the mitigations RFC 9497 section
7.2.3 names for the static Diffie-Hellman attack, under which Q BlindEvaluate queries to one key
reduce the group's security by about log2(Q)/2 bits against the best-known attacks. The
implementation is not constant-time (see :mod:`qsh.qsmpc.oprf`).
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from . import oprf

#: The protocol, as the run reports it (metric ``psi_protocol``); the browser reports the same string.
PROTOCOL = "DH-OPRF PSI (RFC 9497 OPRF, mode 0x00, ristretto255-SHA512), semi-honest"

#: HKDF info strings for the two directions of the transcript; the browser uses the same two.
_INFO_A_TO_B = b"qsh/qsmpc/oprf-psi/bank-a-to-bank-b"
_INFO_B_TO_A = b"qsh/qsmpc/oprf-psi/bank-b-to-bank-a"
#: Associated data that binds each sealed record to its place in the protocol.
_AAD_BLINDED = b"qsh-psi/blinded-elements"
_AAD_EVALUATED = b"qsh-psi/evaluated-elements"
_AAD_OUTPUTS = b"qsh-psi/server-outputs"


@dataclass
class PSIResult:
    """One run of the OPRF PSI, as Bank A, the party that learns the intersection, sees it."""

    intersection: list
    size_a: int
    size_b: int
    intersection_size: int
    #: BlindEvaluate calls Bank B answered: one per ID Bank A blinded, so |A|.
    blind_evaluations: int
    protocol: str = PROTOCOL
    #: Sealed bytes each way, which is all a third party on the channel sees.
    transcript_bytes: dict = field(default_factory=dict)


def _ids(values) -> list:
    """Distinct values in first-seen order: the protocol is over sets."""
    return list(dict.fromkeys(values))


def _encode(value) -> bytes:
    """An ID as the OPRF's input: the UTF-8 bytes of its string form (the browser does the same)."""
    return str(value).encode("utf-8")


def transcript_keys(session_key: bytes) -> tuple[bytes, bytes]:
    """The two AES-256-GCM keys the transcript is sealed under: (Bank A to B, Bank B to A).

    HKDF-SHA256 (RFC 5869) of the PQC session key with no salt and one info string per direction, so
    each key seals one direction only and both counters can start at zero.
    """
    if not session_key:
        raise ValueError("the OPRF PSI is sealed under the PQC session key, and none was given")
    return tuple(HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info).derive(session_key)
                 for info in (_INFO_A_TO_B, _INFO_B_TO_A))


def _split(blob: bytes, width: int) -> list[bytes]:
    if len(blob) % width:
        raise ValueError(f"a sealed PSI message of {len(blob)} bytes is not a whole number of "
                         f"{width}-byte items")
    return [blob[i:i + width] for i in range(0, len(blob), width)]


class OPRFClient:
    """Bank A: blinds its IDs, finalizes Bank B's evaluations, and intersects."""

    def __init__(self, ids) -> None:
        self.ids = _ids(ids)
        self._blinds: list[bytes] = []

    def blinded_elements(self) -> list[bytes]:
        """One fresh blinded element per ID (Blind, RFC 9497 section 3.3.1)."""
        out = []
        self._blinds = []
        for x in self.ids:
            r, m = oprf.blind(_encode(x))
            self._blinds.append(r)
            out.append(m)
        return out

    def intersect(self, evaluated: list[bytes], server_outputs: list[bytes]) -> list:
        """Finalize each evaluation and keep the IDs whose PRF output Bank B also sent."""
        if len(evaluated) != len(self.ids):
            raise ValueError(f"Bank B answered {len(evaluated)} of {len(self.ids)} blinded elements")
        theirs = set(server_outputs)
        return [x for x, r, z in zip(self.ids, self._blinds, evaluated)
                if oprf.finalize(_encode(x), r, z) in theirs]


class OPRFServer:
    """Bank B: holds the OPRF key for one run and never sees Bank A's IDs, only blinded elements."""

    def __init__(self, ids, secret_key: bytes | None = None) -> None:
        self.ids = _ids(ids)
        # Fresh per run; `secret_key` exists so a test can fix it, as the RFC's vectors do.
        self._sk = secret_key if secret_key is not None else oprf.generate_key_pair()[0]

    def evaluate_blinded(self, blinded: list[bytes]) -> list[bytes]:
        """BlindEvaluate each blinded element (RFC 9497 section 3.3.1)."""
        return [oprf.blind_evaluate(self._sk, m) for m in blinded]

    def own_outputs(self) -> list[bytes]:
        """F_k(y) for each of Bank B's IDs (Evaluate), shuffled with the OS CSPRNG."""
        outs = [oprf.evaluate(self._sk, _encode(y)) for y in self.ids]
        secrets.SystemRandom().shuffle(outs)
        return outs


def oprf_psi(set_a, set_b, session_key: bytes) -> PSIResult:
    """Run the OPRF PSI between Bank A (``set_a``) and Bank B (``set_b``), sealed under ``session_key``."""
    from ..pqc.transport import SecureChannel

    key_ab, key_ba = transcript_keys(session_key)
    a_sends, b_opens = SecureChannel(key_ab), SecureChannel(key_ab)
    b_sends, a_opens = SecureChannel(key_ba), SecureChannel(key_ba)
    bank_a, bank_b = OPRFClient(set_a), OPRFServer(set_b)

    # Bank A to Bank B: the blinded elements, and nothing else.
    msg_blinded = a_sends.seal(b"".join(bank_a.blinded_elements()), _AAD_BLINDED)
    blinded = _split(b_opens.open(msg_blinded, _AAD_BLINDED), oprf.NE)

    # Bank B to Bank A: its evaluations of them, then its own outputs in shuffled order.
    msg_evaluated = b_sends.seal(b"".join(bank_b.evaluate_blinded(blinded)), _AAD_EVALUATED)
    msg_outputs = b_sends.seal(b"".join(bank_b.own_outputs()), _AAD_OUTPUTS)

    evaluated = _split(a_opens.open(msg_evaluated, _AAD_EVALUATED), oprf.NE)
    outputs = _split(a_opens.open(msg_outputs, _AAD_OUTPUTS), oprf.NH)
    inter = bank_a.intersect(evaluated, outputs)
    return PSIResult(
        intersection=inter,
        size_a=len(bank_a.ids),
        size_b=len(outputs),
        intersection_size=len(inter),
        blind_evaluations=len(blinded),
        transcript_bytes={"bank_a_to_bank_b": len(msg_blinded),
                          "bank_b_to_bank_a": len(msg_evaluated) + len(msg_outputs)},
    )


def learned_summary(res: PSIResult, a: str = "Bank A", b: str = "Bank B") -> str:
    """What each party learned in this run, in the words the run log uses (both tiers say the same).

    ``a`` is the OPRF client, ``b`` the server: the banks in the pipeline, Alice and Bob in the
    two-party demo.
    """
    return (f"{a} learns the intersection ({res.intersection_size}) and |B| = {res.size_b}; "
            f"{b} learns |A| = {res.size_a} from the {res.blind_evaluations} blinded elements it "
            f"evaluated, and nothing else about {a}'s IDs")
