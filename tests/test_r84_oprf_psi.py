"""Round 84 (psi): the private set intersection is a semi-honest DH-OPRF PSI on RFC 9497, in both tiers.

WHY THIS EXISTS

The finance PSI was HMAC-SHA256 of both banks' ID lists under the one PQC session key both banks
held, so over the demo's small ID space either bank could hash every candidate ID and read the
other's list from its tags. The owner chose to replace it, in the browser and in the full
deployment, with an OPRF-based PSI: RFC 9497's OPRF in mode 0x00 with the suite ristretto255-SHA512
(the group is RFC 9496's ristretto255). Bank B holds a fresh OPRF key k per run; Bank A blinds each
of its IDs, Bank B blind-evaluates them and sends F_k(y) for each of its own IDs in shuffled order,
and Bank A finalizes and intersects. Bank A learns the intersection and |B|, Bank B learns |A|. The
three messages are sealed under keys derived from the PQC session key.

The browser runs @noble/curves' `ristretto255_oprf` (web/src/lib/oprfPsi.ts). Python had no library
for it (`cryptography` has no hash-to-curve), so src/qsh/qsmpc/oprf.py implements RFC 9496's group,
RFC 9380's expand_message_xmd and hash_to_ristretto255, and RFC 9497's OPRF, and src/qsh/qsmpc/psi.py
the protocol.

WHAT IS ASSERTED

1. The Python group and OPRF against the primary sources' own vectors, fetched from rfc-editor.org
   on 2026-11-01: RFC 9496 Appendix A.1 (generator multiples), A.2 (invalid encodings), A.3 (element
   derivation) and A.4 (SQRT_RATIO_M1); RFC 9380 Appendix K.3 (expand_message_xmd, SHA-512); RFC 9497
   Appendix A.1.1 (ristretto255-SHA512, OPRF mode: DeriveKeyPair's skSm, BlindedElement,
   EvaluationElement and Output for both vectors).
2. The browser module against RFC 9497 Appendix A.1.1, and byte for byte against Python for the same
   key, blinds and inputs: blinded elements, evaluated elements, outputs, DeriveKeyPair and the two
   transcript keys. Bundled with esbuild and run in node, as tests/test_the_fibre_key_is_a_finite_key.py
   does.
3. The PSI equals the plain set intersection in both languages, for empty, disjoint, equal, nested
   and overlapping sets with duplicates, and for the finance use case's own sets.
4. What each tier reports: `psi_intersection_size` (the name the pages read), `psi_set_size_a`,
   `psi_set_size_b`, `psi_blind_evaluations` and `psi_protocol`; a log line that says what each bank
   learns; the canvas row and caption of the OPRF; and no keyed hash anywhere in them.
5. The messages are sealed under the session key, with no unsealed mode: a tampered record and
   another session key both fail to open, the sealed sizes are the AEAD's, and both stages refuse to
   run the PSI without a key.
6. The keyed hash is gone from the code of both tiers and from the visitor-facing text that described
   it, which now names the OPRF and its limits.

WHAT EACH FAILS ON AT THE BASE COMMIT (f5f1fce)

src/qsh/qsmpc/oprf.py, src/qsh/qsmpc/psi.py and web/src/lib/oprfPsi.ts do not exist there, so every
vector, identity and correctness test fails to import or to bundle. The Python stage, the two-party
demo and the browser pipeline report no `psi_protocol` and log the keyed hash; the texts name the
HMAC and "a production PSI would use an OPRF-based protocol".
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
SRC = WEB / "src"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"

# --------------------------------------------------------------------------------------------------
# Primary-source vectors
# --------------------------------------------------------------------------------------------------

#: RFC 9496 Appendix A.1: the encodings of 0 to 15 times the ristretto255 generator.
RFC9496_A1 = [
    "0000000000000000000000000000000000000000000000000000000000000000",
    "e2f2ae0a6abc4e71a884a961c500515f58e30b6aa582dd8db6a65945e08d2d76",
    "6a493210f7499cd17fecb510ae0cea23a110e8d5b901f8acadd3095c73a3b919",
    "94741f5d5d52755ece4f23f044ee27d5d1ea1e2bd196b462166b16152a9d0259",
    "da80862773358b466ffadfe0b3293ab3d9fd53c5ea6c955358f568322daf6a57",
    "e882b131016b52c1d3337080187cf768423efccbb517bb495ab812c4160ff44e",
    "f64746d3c92b13050ed8d80236a7f0007c3b3f962f5ba793d19a601ebb1df403",
    "44f53520926ec81fbd5a387845beb7df85a96a24ece18738bdcfa6a7822a176d",
    "903293d8f2287ebe10e2374dc1a53e0bc887e592699f02d077d5263cdd55601c",
    "02622ace8f7303a31cafc63f8fc48fdc16e1c8c8d234b2f0d6685282a9076031",
    "20706fd788b2720a1ed2a5dad4952b01f413bcf0e7564de8cdc816689e2db95f",
    "bce83f8ba5dd2fa572864c24ba1810f9522bc6004afe95877ac73241cafdab42",
    "e4549ee16b9aa03099ca208c67adafcafa4c3f3e4e5303de6026e3ca8ff84460",
    "aa52e000df2e16f55fb1032fc33bc42742dad6bd5a8fc0be0167436c5948501f",
    "46376b80f409b29dc2b5f6f0c52591990896e5716f41477cd30085ab7f10301e",
    "e0c418f7c8d9c4cdd7395b93ea124f3ad99021bb681dfc3302a9d99a2e53e64e",
]

#: RFC 9496 Appendix A.2: encodings Decode (section 4.3.1) MUST reject - four non-canonical field
#: encodings, eight negative field elements, eight non-square x^2, eight negative x*y, and s = -1.
RFC9496_A2 = [
    "00ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
    "f3ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
    "edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
    "0100000000000000000000000000000000000000000000000000000000000000",
    "01ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
    "ed57ffd8c914fb201471d1c3d245ce3c746fcbe63a3679d51b6a516ebebe0e20",
    "c34c4e1826e5d403b78e246e88aa051c36ccf0aafebffe137d148a2bf9104562",
    "c940e5a4404157cfb1628b108db051a8d439e1a421394ec4ebccb9ec92a8ac78",
    "47cfc5497c53dc8e61c91d17fd626ffb1c49e2bca94eed052281b510b1117a24",
    "f1c6165d33367351b0da8f6e4511010c68174a03b6581212c71c0e1d026c3c72",
    "87260f7a2f12495118360f02c26a470f450dadf34a413d21042b43b9d93e1309",
    "26948d35ca62e643e26a83177332e6b6afeb9d08e4268b650f1f5bbd8d81d371",
    "4eac077a713c57b4f4397629a4145982c661f48044dd3f96427d40b147d9742f",
    "de6a7b00deadc788eb6b6c8d20c0ae96c2f2019078fa604fee5b87d6e989ad7b",
    "bcab477be20861e01e4a0e295284146a510150d9817763caf1a6f4b422d67042",
    "2a292df7e32cababbd9de088d1d1abec9fc0440f637ed2fba145094dc14bea08",
    "f4a9e534fc0d216c44b218fa0c42d99635a0127ee2e53c712f70609649fdff22",
    "8268436f8c4126196cf64b3c7ddbda90746a378625f9813dd9b8457077256731",
    "2810e5cbc2cc4d4eece54f61c6f69758e289aa7ab440b3cbeaa21995c2f4232b",
    "3eb858e78f5a7254d8c9731174a94f76755fd3941c0ac93735c07ba14579630e",
    "a45fdc55c76448c049a1ab33f17023edfb2be3581e9c7aade8a6125215e04220",
    "d483fe813c6ba647ebbfd3ec41adca1c6130c2beeee9d9bf065c8d151c5f396e",
    "8a2e1d30050198c65a54483123960ccc38aef6848e1ec8f5f780e8523769ba32",
    "32888462f8b486c68ad7dd9610be5192bbeaf3b443951ac1a8118419d9fa097b",
    "227142501b9d4355ccba290404bde41575b037693cef1f438c47f8fbf35d1165",
    "5c37cc491da847cfeb9281d407efc41e15144c876e0170b499a96a22ed31e01e",
    "445425117cb8c90edcbc7c1cc0e74f747f2c1efa5630a967c64f287792a48a4b",
    "ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
]

#: RFC 9496 Appendix A.3: element-derivation inputs (64 bytes) and their encoded outputs.
RFC9496_A3 = [
    ("5d1be09e3d0c82fc538112490e35701979d99e06ca3e2b5b54bffe8b4dc772c1"
     "4d98b696a1bbfb5ca32c436cc61c16563790306c79eaca7705668b47dffe5bb6",
     "3066f82a1a747d45120d1740f14358531a8f04bbffe6a819f86dfe50f44a0a46"),
    ("f116b34b8f17ceb56e8732a60d913dd10cce47a6d53bee9204be8b44f6678b27"
     "0102a56902e2488c46120e9276cfe54638286b9e4b3cdb470b542d46c2068d38",
     "f26e5b6f7d362d2d2a94c5d0e7602cb4773c95a2e5c31a64f133189fa76ed61b"),
    ("8422e1bbdaab52938b81fd602effb6f89110e1e57208ad12d9ad767e2e25510c"
     "27140775f9337088b982d83d7fcf0b2fa1edffe51952cbe7365e95c86eaf325c",
     "006ccd2a9e6867e6a2c5cea83d3302cc9de128dd2a9a57dd8ee7b9d7ffe02826"),
    ("ac22415129b61427bf464e17baee8db65940c233b98afce8d17c57beeb7876c2"
     "150d15af1cb1fb824bbd14955f2b57d08d388aab431a391cfc33d5bafb5dbbaf",
     "f8f0c87cf237953c5890aec3998169005dae3eca1fbb04548c635953c817f92a"),
    ("165d697a1ef3d5cf3c38565beefcf88c0f282b8e7dbd28544c483432f1cec767"
     "5debea8ebb4e5fe7d6f6e5db15f15587ac4d4d4a1de7191e0c1ca6664abcc413",
     "ae81e7dedf20a497e10c304a765c1767a42d6e06029758d2d7e8ef7cc4c41179"),
    ("a836e6c9a9ca9f1e8d486273ad56a78c70cf18f0ce10abb1c7172ddd605d7fd2"
     "979854f47ae1ccf204a33102095b4200e5befc0465accc263175485f0e17ea5c",
     "e2705652ff9f5e44d3e841bf1c251cf7dddb77d140870d1ab2ed64f1a9ce8628"),
    ("2cdc11eaeb95daf01189417cdddbf95952993aa9cb9c640eb5058d09702c7462"
     "2c9965a697a3b345ec24ee56335b556e677b30e6f90ac77d781064f866a3c982",
     "80bd07262511cdde4863f8a7434cef696750681cb9510eea557088f76d9e5065"),
]
#: RFC 9496 Appendix A.3, last block: four inputs that all derive the same element.
RFC9496_A3_SAME = (
    ["edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
     "1200000000000000000000000000000000000000000000000000000000000000",
     "edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f"
     "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
     "0000000000000000000000000000000000000000000000000000000000000080"
     "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
     "0000000000000000000000000000000000000000000000000000000000000000"
     "1200000000000000000000000000000000000000000000000000000000000080"],
    "304282791023b73128d277bdcb5c7746ef2eac08dde9f2983379cb8e5ef0517f",
)

_Z = "00" * 32
#: RFC 9496 Appendix A.4: SQRT_RATIO_M1(u, v) as little-endian field encodings.
RFC9496_A4 = [
    (_Z, _Z, True, _Z),
    (_Z, "01" + "00" * 31, True, _Z),
    ("01" + "00" * 31, _Z, False, _Z),
    ("02" + "00" * 31, "01" + "00" * 31, False, "3c5ff1b5d8e4113b871bd052f9e7bcd0582804c266ffb2d4f4203eb07fdb7c54"),
    ("04" + "00" * 31, "01" + "00" * 31, True, "02" + "00" * 31),
    ("01" + "00" * 31, "04" + "00" * 31, True, "f6ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff3f"),
]

#: RFC 9380 Appendix K.3, expand_message_xmd(SHA-512), DST QUUX-V01-CS02-with-expander-SHA512-256.
XMD_DST = b"QUUX-V01-CS02-with-expander-SHA512-256"
RFC9380_K3 = [
    (b"", 0x20, "6b9a7312411d92f921c6f68ca0b6380730a1a4d982c507211a90964c394179ba"),
    (b"abc", 0x20, "0da749f12fbe5483eb066a5f595055679b976e93abe9be6f0f6318bce7aca8dc"),
    (b"", 0x80, "41b037d1734a5f8df225dd8c7de38f851efdb45c372887be655212d07251b921b052b62eaed99b46f72f2ef4"
                "cc96bfaf254ebbbec091e1a3b9e4fb5e5b619d2e0c5414800a1d882b62bb5cd1778f098b8eb6cb399d5d9d18f5"
                "d5842cf5d13d7eb00a7cff859b605da678b318bd0e65ebff70bec88c753b159a805d2c89c55961"),
]

#: RFC 9497 Appendix A.1.1, ristretto255-SHA512, OPRF mode: the key material and both vectors.
RFC9497_SEED = "a3" * 32
RFC9497_KEYINFO = "74657374206b6579"                  # "test key"
RFC9497_SKSM = "5ebcea5ee37023ccb9fc2d2019f9d7737be85591ae8652ffa9ef0f4d37063b0e"
RFC9497_BLIND = "64d37aed22a27f5191de1c1d69fadb899d8862b58eb4220029e036ec4c1f6706"
RFC9497_A11 = [
    {"input": "00",
     "blinded": "609a0ae68c15a3cf6903766461307e5c8bb2f95e7e6550e1ffa2dc99e412803c",
     "evaluated": "7ec6578ae5120958eb2db1745758ff379e77cb64fe77b0b2d8cc917ea0869c7e",
     "output": "527759c3d9366f277d8c6020418d96bb393ba2afb20ff90df23fb7708264e2f3"
               "ab9135e3bd69955851de4b1f9fe8a0973396719b7912ba9ee8aa7d0b5e24bcf6"},
    {"input": "5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a",
     "blinded": "da27ef466870f5f15296299850aa088629945a17d1f5b7f5ff043f76b3c06418",
     "evaluated": "b4cbf5a4f1eeda5a63ce7b77c7d23f461db3fcab0dd28e4e17cecb5c90d02c25",
     "output": "f4a74c9c592497375e796aa837e907b1a045d34306a749db9f34221f7e750cb4"
               "f2a6413a6bf6fa5e19ba6348eb673934a722a7ede2e7621306d18951e7cf2c73"},
]

#: Set pairs both languages intersect: empty, one side empty, disjoint, equal, nested, overlapping,
#: duplicates, and IDs shaped like each page's (the finance stage's `acct-`, the demo's `ACCT-`).
SET_CASES = [
    ([], []),
    ([], ["b-1", "b-2"]),
    (["a-1", "a-2"], []),
    (["a-1", "a-2", "a-3"], ["b-1", "b-2"]),
    (["x", "y", "z"], ["z", "y", "x"]),
    (["acct-1", "acct-2"], ["acct-2", "acct-1", "b-9", "b-10"]),
    (["a", "b", "c", "x"], ["c", "x", "z"]),
    (["acct-7", "acct-7", "a-1", "acct-9"], ["acct-9", "b-4", "acct-7", "acct-9"]),
    (["ACCT-1234567", "A-7654321", "", "unicode-\u00e9"], ["", "ACCT-1234567", "B-1111111", "unicode-\u00e9"]),
]


def _plain(a, b) -> set[str]:
    return set(map(str, a)) & set(map(str, b))


# --------------------------------------------------------------------------------------------------
# 1. The Python group and OPRF against the RFCs
# --------------------------------------------------------------------------------------------------

def test_ristretto255_generator_multiples_rfc9496_a1() -> None:
    from qsh.qsmpc import oprf

    acc = oprf.IDENTITY
    for i, want in enumerate(RFC9496_A1):
        assert acc.encode().hex() == want, f"B[{i}] by repeated addition"
        assert (oprf.GENERATOR * i).encode().hex() == want, f"B[{i}] by scalar multiplication"
        if i:
            assert oprf.Element.decode(bytes.fromhex(want)).encode().hex() == want, f"B[{i}] round trip"
        acc = acc + oprf.GENERATOR
    # The group order l (RFC 9496 section 4): (l - 1) * B + B is the identity, and (l - 1) * B is -B.
    almost = oprf.GENERATOR * (oprf.L - 1)
    assert (almost + oprf.GENERATOR).is_identity() and almost == -oprf.GENERATOR
    assert not almost.is_identity()


def test_ristretto255_rejects_every_invalid_encoding_rfc9496_a2() -> None:
    from qsh.qsmpc import oprf

    accepted = []
    for enc in RFC9496_A2:
        try:
            oprf.Element.decode(bytes.fromhex(enc))
            accepted.append(enc)
        except oprf.DecodeError:
            pass
    assert len(RFC9496_A2) == 29 and not accepted, f"Decode accepted encodings RFC 9496 A.2 rejects: {accepted}"
    # The identity decodes (it is a valid element) but the OPRF refuses it over the wire (RFC 9497 3.3).
    assert oprf.Element.decode(bytes(32)).is_identity()
    with pytest.raises(oprf.DecodeError):
        oprf.deserialize_element(bytes(32))


def test_ristretto255_element_derivation_rfc9496_a3() -> None:
    from qsh.qsmpc import oprf

    for inp, want in RFC9496_A3:
        assert oprf.element_derivation(bytes.fromhex(inp)).encode().hex() == want
    inputs, want = RFC9496_A3_SAME
    assert {oprf.element_derivation(bytes.fromhex(i)).encode().hex() for i in inputs} == {want}


def test_sqrt_ratio_m1_rfc9496_a4_and_the_constants() -> None:
    from qsh.qsmpc import oprf

    le = lambda h: int.from_bytes(bytes.fromhex(h), "little")  # noqa: E731
    for u, v, was_square, r in RFC9496_A4:
        got_square, got_r = oprf.sqrt_ratio_m1(le(u), le(v))
        assert (got_square, got_r.to_bytes(32, "little").hex()) == (was_square, r), (u, v)
    # RFC 9496 section 4.1's constants are the relations they are named for, mod p = 2^255 - 19.
    p, d = oprf.P, oprf.D
    assert d == -121665 * pow(121666, -1, p) % p
    assert oprf.SQRT_M1 == pow(2, (p - 1) // 4, p)
    assert oprf.SQRT_AD_MINUS_ONE ** 2 % p == (-d - 1) % p
    assert oprf.INVSQRT_A_MINUS_D ** 2 * (-1 - d) % p == 1
    assert oprf.ONE_MINUS_D_SQ == (1 - d * d) % p and oprf.D_MINUS_ONE_SQ == (d - 1) ** 2 % p


def test_expand_message_xmd_sha512_rfc9380_k3() -> None:
    from qsh.qsmpc import oprf

    for msg, length, want in RFC9380_K3:
        assert oprf.expand_message_xmd(msg, XMD_DST, length).hex() == want, (msg, length)


def test_the_python_oprf_reproduces_rfc9497_a11() -> None:
    from qsh.qsmpc import oprf

    assert oprf.CONTEXT_STRING == b"OPRFV1-\x00-ristretto255-SHA512"
    sk, pk = oprf.derive_key_pair(bytes.fromhex(RFC9497_SEED), bytes.fromhex(RFC9497_KEYINFO))
    assert sk.hex() == RFC9497_SKSM
    assert pk == (oprf.GENERATOR * int.from_bytes(sk, "little")).encode()
    for v in RFC9497_A11:
        x = bytes.fromhex(v["input"])
        blind, blinded = oprf.blind(x, bytes.fromhex(RFC9497_BLIND))
        assert blinded.hex() == v["blinded"], "BlindedElement"
        evaluated = oprf.blind_evaluate(sk, blinded)
        assert evaluated.hex() == v["evaluated"], "EvaluationElement"
        assert oprf.finalize(x, blind, evaluated).hex() == v["output"], "Output (Finalize)"
        assert oprf.evaluate(sk, x).hex() == v["output"], "Output (Evaluate)"


# --------------------------------------------------------------------------------------------------
# 2. The browser module, and byte identity between the tiers
# --------------------------------------------------------------------------------------------------

def _require_node_or_skip() -> None:
    missing = [name for name, ok in (("web/node_modules/.bin/esbuild", ESBUILD.exists()),
                                     ("node", shutil.which("node") is not None)) if not ok]
    if not missing:
        return
    msg = f"{' and '.join(missing)} absent, so the browser module cannot run. This is a SKIP, not a pass."
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


_RUNNER = r"""
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
const [BUNDLE, INPUT] = process.argv.slice(2);
const m = await import(pathToFileURL(BUNDLE).href);
const input = JSON.parse(readFileSync(INPUT, "utf8"));
const hx = (b) => Buffer.from(b).toString("hex");
const fx = (h) => Uint8Array.from(Buffer.from(h, "hex"));
const out = { context: hx(m.OPRF_CONTEXT), protocol: m.PSI_PROTOCOL, hasEvaluate: typeof m.oprf.evaluate === "function" };
out.derive = input.derive.map(([seed, info]) => hx(m.oprf.deriveKeyPair(fx(seed), fx(info)).secretKey));
out.cases = input.cases.map(([sk, blind, x]) => {
  const blinded = m.blindWith(fx(x), fx(blind));
  const evaluated = m.oprf.blindEvaluate(fx(sk), blinded);
  return { blinded: hx(blinded), evaluated: hx(evaluated), output: hx(m.oprf.finalize(fx(x), fx(blind), evaluated)),
           evaluate: hx(m.oprf.evaluate(fx(sk), fx(x))) };
});
// noble's own blind draws a fresh scalar: its blinded element is blindWith's for that scalar, two blinds
// of one input differ, and Finalize over BlindEvaluate is Evaluate.
out.randomBlind = input.inputs.map((x) => {
  const sk = m.oprf.generateKeyPair().secretKey;
  const b1 = m.oprf.blind(fx(x)), b2 = m.oprf.blind(fx(x));
  return { same: hx(m.blindWith(fx(x), b1.blind)) === hx(b1.blinded),
           differ: hx(b1.blinded) !== hx(b2.blinded),
           finalizeIsEvaluate: hx(m.oprf.finalize(fx(x), b1.blind, m.oprf.blindEvaluate(sk, b1.blinded)))
             === hx(m.oprf.evaluate(sk, fx(x))) };
});
out.keys = input.sessionKeys.map((k) => { const t = m.psiTranscriptKeys(fx(k)); return [hx(t.aToB), hx(t.bToA)]; });
out.psi = [];
for (const [a, b] of input.sets) {
  const r = await m.oprfPSI(a, b, fx(input.sessionKeys[0]));
  out.psi.push({ ...r, learned: m.psiLearnedSummary(r) });
}
try { await m.oprfPSI(["a"], ["a"], new Uint8Array(0)); out.emptyKey = "no error"; }
catch (e) { out.emptyKey = String(e?.message ?? e); }
// The record layer: open what Python sealed, seal what Python will open, and reject a changed byte,
// another key and other associated data.
const rec = input.record, key = fx(rec.key), aad = new TextEncoder().encode(rec.aad);
out.record = { opened: hx(await m.openPsiRecord(key, fx(rec.sealedByPython), aad)),
               sealed: hx(await m.sealPsiRecord(key, fx(rec.plaintext), aad, rec.counter)), rejects: [] };
for (const [k, r, a] of [[key, fx(rec.tampered), aad], [fx(rec.otherKey), fx(rec.sealedByPython), aad],
                         [key, fx(rec.sealedByPython), new TextEncoder().encode("qsh-psi/server-outputs")]]) {
  try { await m.openPsiRecord(k, r, a); out.record.rejects.push(false); } catch { out.record.rejects.push(true); }
}
const stepOf = (uc) => m.visibleSteps(uc, {}).find((st) => st.stageId === "qsmpc_compute");
out.canvas = Object.fromEntries(["finance_aml", "demo"].map((uc) => {
  const s = stepOf(uc);
  return [uc, { row: m.canvasRow(uc, s, m.kindOf(s, {}), {}, {}), label: s.label({ psi_intersection_size: 7 }, {}),
                caption: m.captionOf(uc) }];
}));
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def browser(tmp_path_factory) -> dict:
    """The browser module's answers to the vectors, the cross-language cases and the PSI cases."""
    _require_node_or_skip()
    from qsh.qsmpc import oprf

    d = tmp_path_factory.mktemp("r84psi")
    entry, bundle, runner, inp = d / "entry.ts", d / "bundle.mjs", d / "run.mjs", d / "input.json"
    entry.write_text(
        "export { oprf, blindWith, psiTranscriptKeys, oprfPSI, PSI_PROTOCOL, OPRF_CONTEXT, psiLearnedSummary,"
        " openPsiRecord, sealPsiRecord }"
        f" from {json.dumps(str(SRC / 'lib' / 'oprfPsi'))};\n"
        "export { canvasRow, captionOf, kindOf, visibleSteps }"
        f" from {json.dumps(str(SRC / 'lib' / 'useCaseFlows'))};\n")
    subprocess.run([str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node",
                    f"--outfile={bundle}", "--log-level=error"], cwd=WEB, check=True, capture_output=True)
    runner.write_text(_RUNNER)
    # The cross-language cases: the RFC's key and blind, a key from DeriveKeyPair on another seed, and
    # fresh random keys and blinds from Python, over the RFC's inputs and the pages' ID shapes.
    sk2, _ = oprf.derive_key_pair(bytes(range(32)), b"qsh psi test")
    inputs = ["00", "5a" * 17, "", b"acct-123456".hex(), b"ACCT-1234567".hex(), "unicode-\u00e9".encode().hex()]
    keys = [RFC9497_SKSM, sk2.hex(), oprf.generate_key_pair()[0].hex()]
    blinds = [RFC9497_BLIND, oprf.serialize_scalar(oprf.random_scalar()).hex()]
    cases = [(k, b, x) for k in keys for b in blinds for x in inputs]
    session_keys = [bytes(range(32)).hex(), "ff" * 32, os.urandom(32).hex()]
    # One record sealed by Python's SecureChannel under a transcript key, and one the browser seals
    # at counter 1 for Python to open (`test_the_record_layer_is_one_format_in_both_tiers`).
    from qsh.pqc.transport import SecureChannel
    from qsh.qsmpc.psi import transcript_keys

    rkey = transcript_keys(bytes(range(32)))[0]
    sealed = SecureChannel(rkey).seal(b"\x05" * 96, b"qsh-psi/blinded-elements")
    tampered = bytearray(sealed)
    tampered[40] ^= 0x80
    record = {"key": rkey.hex(), "aad": "qsh-psi/blinded-elements", "sealedByPython": sealed.hex(),
              "tampered": bytes(tampered).hex(), "otherKey": transcript_keys(b"\x09" * 32)[0].hex(),
              "plaintext": (b"\x06" * 64).hex(), "counter": 1}
    payload = {"derive": [[RFC9497_SEED, RFC9497_KEYINFO], [bytes(range(32)).hex(), b"qsh psi test".hex()]],
               "cases": cases, "inputs": inputs, "sessionKeys": session_keys, "sets": SET_CASES,
               "record": record}
    inp.write_text(json.dumps(payload))
    proc = subprocess.run(["node", str(runner), str(bundle), str(inp)], cwd=WEB, capture_output=True,
                          text=True, timeout=600)
    assert proc.returncode == 0, f"the browser module did not run:\n{proc.stderr[-3000:]}"
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    out["_payload"] = payload
    return out


def test_the_browser_oprf_reproduces_rfc9497_a11(browser) -> None:
    assert bytes.fromhex(browser["context"]) == b"OPRFV1-\x00-ristretto255-SHA512"
    assert browser["hasEvaluate"], "@noble/curves no longer exposes Evaluate for mode 0x00"
    assert browser["derive"][0] == RFC9497_SKSM
    rfc = [c for c, (k, b, _x) in zip(browser["cases"], browser["_payload"]["cases"])
           if k == RFC9497_SKSM and b == RFC9497_BLIND]
    by_input = {x: c for c, (k, b, x) in zip(browser["cases"], browser["_payload"]["cases"])
                if k == RFC9497_SKSM and b == RFC9497_BLIND}
    assert len(rfc) == len(browser["_payload"]["inputs"])
    for v in RFC9497_A11:
        got = by_input[v["input"]]
        assert (got["blinded"], got["evaluated"], got["output"], got["evaluate"]) == (
            v["blinded"], v["evaluated"], v["output"], v["output"]), v["input"]


def test_python_and_the_browser_are_byte_identical(browser) -> None:
    """For the same key, blind and input both tiers send the same blinded and evaluated elements and
    compute the same output; DeriveKeyPair and the transcript keys agree too."""
    from qsh.qsmpc import oprf
    from qsh.qsmpc.psi import transcript_keys

    payload = browser["_payload"]
    assert len(payload["cases"]) == 36
    for (sk, blind, x), got in zip(payload["cases"], browser["cases"]):
        b, blinded = oprf.blind(bytes.fromhex(x), bytes.fromhex(blind))
        evaluated = oprf.blind_evaluate(bytes.fromhex(sk), blinded)
        want = {"blinded": blinded.hex(), "evaluated": evaluated.hex(),
                "output": oprf.finalize(bytes.fromhex(x), b, evaluated).hex(),
                "evaluate": oprf.evaluate(bytes.fromhex(sk), bytes.fromhex(x)).hex()}
        assert got == want, (sk, blind, x)
        assert want["output"] == want["evaluate"]
    for (seed, info), got in zip(payload["derive"], browser["derive"]):
        assert oprf.derive_key_pair(bytes.fromhex(seed), bytes.fromhex(info))[0].hex() == got
    for k, got in zip(payload["sessionKeys"], browser["keys"]):
        assert [t.hex() for t in transcript_keys(bytes.fromhex(k))] == got
    for r in browser["randomBlind"]:
        assert r == {"same": True, "differ": True, "finalizeIsEvaluate": True}, r


# --------------------------------------------------------------------------------------------------
# 3. The PSI is the plain intersection, in both languages
# --------------------------------------------------------------------------------------------------

def _sealed_sizes(size_a: int, size_b: int) -> dict:
    """Bytes on the wire: each record is a 12-byte nonce, the plaintext and a 16-byte GCM tag; Bank A
    sends |A| 32-byte elements, Bank B |A| elements and |B| 64-byte outputs in two records."""
    return {"bank_a_to_bank_b": 28 + 32 * size_a, "bank_b_to_bank_a": 56 + 32 * size_a + 64 * size_b}


@pytest.mark.parametrize("i", range(len(SET_CASES)))
def test_the_python_psi_is_the_plain_intersection(i) -> None:
    from qsh.qsmpc.psi import PROTOCOL, oprf_psi

    a, b = SET_CASES[i]
    r = oprf_psi(a, b, bytes(range(32)))
    want = _plain(a, b)
    assert set(r.intersection) == want and len(r.intersection) == r.intersection_size == len(want)
    assert (r.size_a, r.size_b, r.blind_evaluations) == (len(set(a)), len(set(b)), len(set(a)))
    assert r.protocol == PROTOCOL
    assert r.transcript_bytes == _sealed_sizes(len(set(a)), len(set(b)))


def test_the_browser_psi_is_the_plain_intersection(browser) -> None:
    from qsh.qsmpc.psi import PROTOCOL, learned_summary, oprf_psi

    assert browser["protocol"] == PROTOCOL
    for (a, b), r in zip(SET_CASES, browser["psi"]):
        want = _plain(a, b)
        assert set(r["intersection"]) == want and r["intersection_size"] == len(want), (a, b)
        assert (r["size_a"], r["size_b"], r["blind_evaluations"]) == (len(set(a)), len(set(b)), len(set(a)))
        assert r["protocol"] == PROTOCOL
        assert r["transcript_bytes"] == _sealed_sizes(len(set(a)), len(set(b)))
        # The run log's sentence is the same in both tiers.
        assert r["learned"] == learned_summary(oprf_psi(a, b, b"k" * 32))


def test_the_python_psi_finds_the_finance_use_cases_shared_accounts() -> None:
    from qsh.qsmpc.psi import oprf_psi
    from qsh.usecases.finance_aml import _bank_blacklists

    bank_a, bank_b, shared = _bank_blacklists(0)
    r = oprf_psi(bank_a, bank_b, os.urandom(32))
    assert set(r.intersection) == shared == set(bank_a) & set(bank_b)
    assert (r.size_a, r.size_b, r.blind_evaluations) == (len(bank_a), len(bank_b), len(bank_a))


# --------------------------------------------------------------------------------------------------
# 4. The roles: what each bank holds and sees
# --------------------------------------------------------------------------------------------------

def test_bank_b_sees_only_blinded_elements_and_its_outputs_are_its_prf() -> None:
    from qsh.qsmpc import oprf
    from qsh.qsmpc.psi import OPRFClient, OPRFServer

    ids_a, ids_b = ["acct-1", "acct-2", "a-3"], ["acct-2", "b-4", "acct-1", "b-5"]
    client, again = OPRFClient(ids_a), OPRFClient(ids_a)
    blinded, blinded_again = client.blinded_elements(), again.blinded_elements()
    # A fresh blind per run: the same IDs blind to different elements, none of them the ID's hash.
    assert all(x != y for x, y in zip(blinded, blinded_again))
    assert not {oprf.hash_to_group(i.encode()).encode() for i in ids_a} & set(blinded)
    sk, _ = oprf.derive_key_pair(bytes(32), b"")
    server = OPRFServer(ids_b, secret_key=sk)
    outputs = server.own_outputs()
    assert sorted(outputs) == sorted(oprf.evaluate(sk, y.encode()) for y in ids_b)
    assert set(client.intersect(server.evaluate_blinded(blinded), outputs)) == {"acct-1", "acct-2"}
    # Bank A cannot finish without Bank B's key: evaluations under another key match nothing.
    other = OPRFServer(ids_b)
    assert client.intersect(other.evaluate_blinded(blinded), outputs) == []


def test_the_transcript_is_sealed_under_the_session_key() -> None:
    from cryptography.exceptions import InvalidTag

    from qsh.pqc.transport import SecureChannel
    from qsh.qsmpc.psi import oprf_psi, transcript_keys

    k_ab, k_ba = transcript_keys(b"\x07" * 32)
    assert k_ab != k_ba and len(k_ab) == len(k_ba) == 32
    assert transcript_keys(b"\x08" * 32)[0] != k_ab
    record = SecureChannel(k_ab).seal(b"\x01" * 64, b"qsh-psi/blinded-elements")
    assert SecureChannel(k_ab).open(record, b"qsh-psi/blinded-elements") == b"\x01" * 64
    tampered = bytearray(record)
    tampered[20] ^= 1
    for key, rec, aad in ((k_ab, bytes(tampered), b"qsh-psi/blinded-elements"),
                          (transcript_keys(b"\x08" * 32)[0], record, b"qsh-psi/blinded-elements"),
                          (k_ab, record, b"qsh-psi/server-outputs")):
        with pytest.raises(InvalidTag):
            SecureChannel(key).open(rec, aad)
    with pytest.raises(ValueError, match="sealed under the PQC session key"):
        oprf_psi(["a"], ["a"], b"")


def test_the_browser_psi_refuses_to_run_unsealed(browser) -> None:
    assert "sealed under the PQC session key" in browser["emptyKey"], browser["emptyKey"]


def test_the_record_layer_is_one_format_in_both_tiers(browser) -> None:
    """The browser opens a record Python's SecureChannel sealed, Python opens one the browser sealed,
    and the browser rejects a changed byte, another key and other associated data."""
    from qsh.pqc.transport import SecureChannel

    rec, got = browser["_payload"]["record"], browser["record"]
    assert got["opened"] == "05" * 96
    sealed = bytes.fromhex(got["sealed"])
    assert sealed[:12] == (1).to_bytes(12, "big"), "the nonce is the 12-byte big-endian counter"
    assert SecureChannel(bytes.fromhex(rec["key"])).open(sealed, rec["aad"].encode()) == bytes.fromhex(rec["plaintext"])
    assert got["rejects"] == [True, True, True], got["rejects"]


# --------------------------------------------------------------------------------------------------
# 5. What each tier reports
# --------------------------------------------------------------------------------------------------

_RETIRED = ("HMAC", "keyed-hash", "keyed hash", "OPRF-based protocol", "recover the other's list")


def _python_stage(set_a, set_b, session_key):
    from qsh.orchestration.crypto_agility import CryptoRegistry
    from qsh.orchestration.events import EventBus
    from qsh.orchestration.stages import Context, QSMPCComputeStage

    ctx = Context(usecase="finance_aml", config={"qsmpc_op": "psi", "seed": 0})
    ctx.data.update(set_a=set_a, set_b=set_b, session_key=session_key)
    bus = EventBus()
    QSMPCComputeStage().run(ctx, bus, CryptoRegistry())
    logs = [e.message for e in bus.history if e.type.value == "log"]
    return ctx, bus.metrics(), logs


def test_the_python_stage_reports_the_oprf_psi_and_what_each_bank_learns() -> None:
    from qsh.qsmpc.psi import PROTOCOL
    from qsh.usecases.finance_aml import _bank_blacklists

    bank_a, bank_b, shared = _bank_blacklists(0)
    ctx, m, logs = _python_stage(bank_a, bank_b, os.urandom(32))
    assert set(ctx.artifacts["psi_intersection"]) == shared
    assert m["psi_intersection_size"] == len(shared)
    assert (m["psi_set_size_a"], m["psi_set_size_b"], m["psi_blind_evaluations"]) == (len(bank_a), len(bank_b), len(bank_a))
    assert m["psi_protocol"] == PROTOCOL
    line = next(x for x in logs if x.startswith("PSI complete"))
    for part in (f"Bank A learns the intersection ({len(shared)}) and |B| = {len(bank_b)}",
                 f"Bank B learns |A| = {len(bank_a)}", "nothing else about Bank A's IDs",
                 "sealed under the PQC session key", "Bank A and Bank B run in this process (notional parties)",
                 "not post-quantum", PROTOCOL):
        assert part in line, (part, line)
    assert not [r for r in _RETIRED if r in line], line


def test_the_python_stage_refuses_the_psi_without_a_session_key() -> None:
    with pytest.raises(RuntimeError, match="sealed under it, and there is no unsealed mode"):
        _python_stage(["a"], ["a"], None)


def test_the_full_deployment_demo_pipeline_reports_the_oprf_psi(tmp_path) -> None:
    from qsh.orchestration.pipeline import PipelineState, build_pipeline
    from qsh.qsmpc.psi import PROTOCOL

    pipe, ctx = build_pipeline("demo")
    pipe.run(ctx, out_dir=tmp_path)
    assert pipe.state is PipelineState.DONE
    m = pipe.bus.metrics()
    want = _plain(ctx.data["set_a"], ctx.data["set_b"])
    assert m["psi_intersection_size"] == len(want) and set(ctx.artifacts["psi_intersection"]) == want
    assert m["psi_protocol"] == PROTOCOL and m["psi_blind_evaluations"] == len(set(ctx.data["set_a"]))
    # The generic demo has parties, not banks, as its canvas does.
    line = next(e.message for e in pipe.bus.history if e.type.value == "log" and e.message.startswith("PSI complete"))
    assert f"Party A learns the intersection ({len(want)})" in line and "Bank" not in line, line


def test_the_two_party_demo_runs_the_oprf_psi_in_python() -> None:
    from qsh.benchmark.alice_bob_e2e import run_alice_bob_e2e
    from qsh.qsmpc.psi import PROTOCOL

    r = run_alice_bob_e2e(seed=0, n_shared=8)
    assert r["psi_protocol"] == PROTOCOL and r["psi_correct"] is True
    assert r["psi_intersection_size"] == r["psi_ground_truth"] == 8
    assert (r["psi_set_size_a"], r["psi_set_size_b"], r["psi_blind_evaluations"]) == (208, 188, 208)
    assert r["psi_learned"].startswith("Alice learns the intersection (8) and |B| = 188; Bob learns |A| = 208")
    psi_phase = next(p for p in r["phases"] if p["name"].startswith("Private set intersection"))
    assert psi_phase["party"] == "Alice+Bob"


def test_the_canvas_draws_the_oprf_roles_and_the_caption_states_them(browser) -> None:
    for uc, c in browser["canvas"].items():
        row = c["row"]
        assert row["a"] == "OPRF client: blinds its IDs, intersects", (uc, row)
        assert row["b"] == "OPRF server: evaluates under its own key", (uc, row)
        assert row["flow"] == "both" and row["channel"] == "classical"
        assert c["label"].startswith("OPRF PSI (RFC 9497)") and c["label"].endswith("shared flagged = 7")
        cap = c["caption"]
        for part in ("side B evaluates them under an OPRF key only B holds", "RFC 9497 over ristretto255",
                     "A learns the shared IDs and the size of B's list and B only the size of A's",
                     "sealed under the session key", "semi-honest and not post-quantum"):
            assert part in cap, (uc, part, cap)
        assert not [r for r in _RETIRED if r in cap + c["label"] + json.dumps(row)], (uc, cap)


# The browser pipeline, run in node as tests/test_r84_finance_browser_run.py runs it, stopped once the
# QSMPC stage has ended (the stages after it are that file's subject). The two-party mirror it also ran,
# `clientAliceBob`, had no caller in the app and was removed in Round 86.
_PIPELINE = r"""
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
const WEB = process.argv[2];
const { build } = createRequire(WEB + "/package.json")("esbuild");
const dir = mkdtempSync(join(tmpdir(), "qsh-r84-psi-"));
const out = join(dir, "bundle.mjs");
const sealWasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export { runClientPipeline } from ${JSON.stringify(resolve(WEB, "src/lib/clientCompute"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: out, logLevel: "error",
  define: { "import.meta.env": '{"BASE_URL":"/"}', "process.env.NODE_ENV": '"production"' },
  plugins: [{ name: "node-stand-ins", setup(b) {
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" },
             () => ({ contents: `export default ${JSON.stringify(sealWasm)};`, loader: "js" }));
    b.onResolve({ filter: /(^|\/)clientMPC$/ }, () => ({ path: "fakempc", namespace: "fakempc" }));
    b.onLoad({ filter: /.*/, namespace: "fakempc" }, () => ({ loader: "js", contents: `
      export async function warmMPC() {}
      export async function secureArgmax(l) {
        let b = 0; for (let i = 1; i < l.length; i++) if (l[i] > l[b]) b = i;
        return { out: [b], compares: Math.max(0, l.length - 1), commBytes: 0, mpcMs: 0, messages: 0, clearOut: [b] };
      }` }));
    b.onResolve({ filter: /\.wasm\?url$|qsh_mpc\.js$/ }, (a) => ({ path: a.path, namespace: "stub" }));
    b.onLoad({ filter: /.*/, namespace: "stub" }, () => ({ contents: "export default '';", loader: "js" }));
  } }],
});
globalThis.fetch = async (url) => {
  const p = String(url);
  const miss = { ok: false, status: 404, headers: { get: () => null },
                 json: async () => { throw new Error("404"); }, text: async () => "" };
  if (/^[a-z]+:/i.test(p)) return miss;
  let text;
  try { text = readFileSync(join(WEB, "public", p.replace(/^\/+/, "")), "utf8"); } catch { return miss; }
  return { ok: true, status: 200, headers: { get: () => "application/json" },
           json: async () => JSON.parse(text), text: async () => text };
};
const m = await import(pathToFileURL(out).href);
rmSync(dir, { recursive: true, force: true });
const result = {};
for (const [name, usecase, config] of [["finance", "finance_aml", { seed: 0, qfl: true }], ["demo", "demo", { seed: 0 }]]) {
  const logs = [], metrics = {}, times = {};
  let ctl = null;
  const artifacts = await new Promise((done) => {
    const timer = setTimeout(() => done(null), 600_000);
    ctl = m.runClientPipeline(usecase, config, {
      onEvent: (e) => {
        if (e.stage === "qsmpc_compute") {
          if (e.type === "log") logs.push(String(e.message ?? ""));
          if (e.type === "metric") metrics[e.data.name] = e.data.value;
          if (e.type === "stage_start" || e.type === "stage_end") times[e.type] = e.t;
          if (e.type === "stage_end") ctl?.stop();
        }
      },
      onEnd: (s) => { clearTimeout(timer); done(s.artifacts); },
    });
  });
  result[name] = { logs, metrics, stageSeconds: times.stage_end - times.stage_start,
                   intersection: artifacts?.psi_intersection ?? null };
}
console.log(JSON.stringify(result));
process.exit(0);
"""


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> dict:
    _require_node_or_skip()
    script = tmp_path_factory.mktemp("r84psipipe") / "harness.mjs"
    script.write_text(_PIPELINE)
    proc = subprocess.run(["node", str(script), str(WEB)], cwd=WEB, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"the browser pipeline did not run:\n{proc.stderr[-3000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_the_browser_pipeline_runs_the_oprf_psi_and_says_what_each_bank_learns(pipeline) -> None:
    from qsh.qsmpc.psi import PROTOCOL

    # finance_aml at the default 4000 transactions: 200 per bank plus 60 shared (clientCompute's
    # `perParty`), so 260 IDs a side unless the seeded draw repeats one; demo: 40 plus 12. The log
    # names the parties as the canvas does: the banks on finance, Party A and Party B on the demo.
    for name, per_party, n_shared, (pa, pb) in (("finance", 200, 60, ("Bank A", "Bank B")),
                                                ("demo", 40, 12, ("Party A", "Party B"))):
        r = pipeline[name]
        m = r["metrics"]
        assert not [x for x in r["logs"] if x.startswith("error:")], r["logs"]
        assert m["psi_protocol"] == PROTOCOL, m
        assert m["psi_set_size_a"] <= per_party + n_shared and m["psi_set_size_b"] <= per_party + n_shared
        assert m["psi_blind_evaluations"] == m["psi_set_size_a"]
        assert m["psi_intersection_size"] == len(r["intersection"]) >= 1
        assert all(x.startswith("acct-") for x in r["intersection"]), "an ID outside the shared draw intersected"
        assert "psi_set_size" not in m, "the retired single-set metric is reported again"
        line = next(x for x in r["logs"] if x.startswith("PSI complete"))
        for part in (f"{pa} learns the intersection ({m['psi_intersection_size']}) and |B| = {m['psi_set_size_b']}",
                     f"{pb} learns |A| = {m['psi_set_size_a']}", f"nothing else about {pa}'s IDs",
                     "sealed under the PQC session key", f"{pa} and {pb} run in this tab (notional parties)",
                     "not post-quantum", PROTOCOL):
            assert part in line, (name, part, line)
        assert not [x for x in _RETIRED if x in line], line


# --------------------------------------------------------------------------------------------------
# 6. The keyed hash is gone, and the texts state the OPRF
# --------------------------------------------------------------------------------------------------

def _code(path: Path) -> str:
    """Source with comments blanked (TS block, line and JSX comments, Python comments), flattened."""
    text = path.read_text()
    text = re.sub(r"/\*[\s\S]*?\*/", " ", text)
    text = re.sub(r"(?<![:\"'`])//[^\n]*", " ", text)
    return re.sub(r"\s+", " ", text)


def test_the_keyed_hash_is_gone_from_the_code_of_both_tiers() -> None:
    qot = (REPO / "src" / "qsh" / "qsmpc" / "qot.py").read_text()
    for gone in ("def private_set_intersection", "def multiparty_psi", "import hmac", "hmac.new"):
        assert gone not in qot, gone
    stages = (REPO / "src" / "qsh" / "orchestration" / "stages.py").read_text()
    assert "private_set_intersection" not in stages and "multiparty_psi" not in stages
    cc = _code(SRC / "lib" / "clientCompute.ts")
    for gone in ("keyedHashPSI", "hmacHex", '"HMAC"', "HMAC key"):
        assert gone not in cc, gone
    assert "oprfPSI(" in cc


def test_the_visitor_facing_texts_state_the_oprf_psi() -> None:
    texts = {
        "useCases.ts": _code(SRC / "lib" / "useCases.ts"),
        "Overview.tsx": _code(SRC / "pages" / "Overview.tsx"),
        "Value.tsx": _code(SRC / "pages" / "Value.tsx"),
        "OrchestrationGraph.tsx": _code(SRC / "components" / "OrchestrationGraph.tsx"),
        "ArchitectureDiagram.tsx": _code(SRC / "components" / "ArchitectureDiagram.tsx"),
        "useCaseFlows.ts": _code(SRC / "lib" / "useCaseFlows.ts"),
    }
    for name, t in texts.items():
        assert not [r for r in ("HMAC", "keyed-hash", "OPRF-based protocol") if r in t], name
    blurb = re.search(r'id: "finance_aml".*?blurb: "([^"]*)"', texts["useCases.ts"]).group(1)
    for part in ("OPRF-based PSI (RFC 9497 over ristretto255)", "Bank A learns the shared accounts and the size of "
                 "Bank B's list, Bank B learns only the size of Bank A's", "semi-honest and not post-quantum",
                 "sealed under the PQC session key"):
        assert part in blurb, part
    assert "The PSI runs RFC 9497&apos;s OPRF over ristretto255" in texts["Overview.tsx"]
    assert "quantum-safe private set intersection" not in texts["Overview.tsx"]
    assert "The PSI is semi-honest and not post-quantum" in texts["Value.tsx"]
    assert '"PSI\\n(OPRF, classical)"' in texts["OrchestrationGraph.tsx"]
    assert '"PSI: RFC 9497 OPRF,"' in texts["ArchitectureDiagram.tsx"]


#: The finance flowchart's PSI node, and the wording of the keyed-hash PSI it replaced. Module level so
#: the technical report's own tests (outside this repository since Round 86) hold its copy to the same.
PSI_NODE = "PSI: OPRF (RFC 9497, ristretto255-SHA512) over two banks' account lists"
RETIRED_PSI_WORDING = ("keyed-hash", "HMAC-SHA256 keyed by the session key", "OPRF-based protocol",
                       "HMAC of both lists", "keyedHashPSI", "private_set_intersection")


def test_the_documents_state_the_oprf_psi() -> None:
    readme = (REPO / "README.md").read_text()
    assert PSI_NODE in readme, "the finance flowchart's PSI node"
    for rel in ("README.md", "docs/threat-model.md", "docs/use-cases.md", "docs/findings.md",
                "references/REFERENCES.md"):
        flat = re.sub(r"\s+", " ", (REPO / rel).read_text())
        for retired in RETIRED_PSI_WORDING:
            assert retired not in flat, (rel, retired)
    tm = (REPO / "docs" / "threat-model.md").read_text()
    row = next(line for line in tm.splitlines() if line.startswith("| PSI (finance) |"))
    for part in ("RFC 9497's OPRF, mode 0x00, suite ristretto255-SHA512", "RFC 9496", "`web/src/lib/oprfPsi.ts`",
                 "`qsmpc/oprf.py`", "`qsmpc/psi.py`", "**No malicious security**", "**Not post-quantum**",
                 "Unconditional input secrecy", "sealed with AES-256-GCM", "hides nothing between the banks"):
        assert part in row, part
    refs = (REPO / "references" / "REFERENCES.md").read_text()
    for rfc in ("RFC 9497, \"Oblivious Pseudorandom Functions (OPRFs) Using Prime-Order Groups\"",
                "RFC 9496, \"The ristretto255 and decaf448 Groups\"", "RFC 9380, \"Hashing to Elliptic Curves\""):
        assert rfc in refs, rfc


def test_every_file_this_round_wrote_is_plain_ascii() -> None:
    for rel in ("src/qsh/qsmpc/oprf.py", "src/qsh/qsmpc/psi.py", "web/src/lib/oprfPsi.ts", "tests/test_r84_oprf_psi.py"):
        assert (REPO / rel).read_text().isascii(), rel
