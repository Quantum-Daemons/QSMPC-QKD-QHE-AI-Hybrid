# Network key schedules: sources, vectors, and what is validated (checked 2026-10-24)

This records where every formula and every stored test vector in `web/src/lib/netkeys/` comes from,
and draws one line through the result: **validated against an official or published vector** on one
side, **computed from the specification text only** on the other. The line matters more than the
list. Two of the layers below (RFC 9370 and RFC 8784) have no official vectors anywhere, and a seeded
transposition in either passes every check this repository runs. That is measured, not assumed; the
table at the end shows it.

Every file listed was downloaded on 2026-10-24 and hashed on arrival. The SHA-256 values are what make
"the upstream file changed" a question with an answer: re-fetch, hash, compare.

## Modules and how they are checked

| File | What it computes |
|---|---|
| `web/src/lib/netkeys/mlkemKat.ts` | `runMlKemAcvp()`: NIST ACVP ML-KEM-768 keyGen, encapsulation, decapsulation through `@noble/post-quantum` |
| `web/src/lib/netkeys/noiseIKpsk2.ts` | Noise_IKpsk2_25519_ChaChaPoly_BLAKE2s, initiator and responder, Split(); `runNoiseVectors()` |
| `web/src/lib/netkeys/wireguard.ts` | `wireguardHandshake({ psk, seed })`: both peers in one tab on the Noise core |
| `web/src/lib/netkeys/ikev2.ts` | prf+, SKEYSEED, SK_*, RFC 9370, RFC 8784; `runIkev2Cavp()`, `ikev2KeySchedule({ seed, ppk })` |
| `web/scripts/check_net_key_schedules.mjs` | `npm run check:netkeys`: bundles the four modules and runs every stored vector through them |
| `tests/test_net_key_schedules.py` | runs that check, asserts provenance, and runs the seeded negative for every vector family |

The node check compares bytes itself and then requires each in-tab runner to reach the same verdict, so
a runner that compared a value with itself would be caught. `--corrupt=<family>` (or
`QSH_NETKEYS_CORRUPT`) flips one byte of one expected value in memory; families `mlkem`, `noise`,
`ikev2`, `rfc5903`, `wginit`. Each must make the check exit 1 and name the case, and the pytest file
asserts that it does.

## Specifications (formulas confirmed against these texts)

| Document | Sections used | URL | Bytes | SHA-256 |
|---|---|---|---|---|
| FIPS 203 (ML-KEM), August 2024 | 6.1-6.3 Algorithms 16-18 (KeyGen_internal(d, z), Encaps_internal(ek, m), Decaps_internal(dk, c) with implicit rejection), 8 Table 3 sizes | https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf | 1252341 | `fe1f12f32a7e44ec9fdebbf400cda843a40b506dee676725234dc6f7923b6cac` |
| RFC 7296 (IKEv2) | 2.10 nonces, 2.13 prf+, 2.14 SKEYSEED and SK_*, 2.17 Child SA KEYMAT, 2.18 rekey | https://www.rfc-editor.org/rfc/rfc7296.txt | 354358 | `b6c1bf3921e85dd3674f427ffcaded0209e1f9718e8d601e2f918371f46329a1` |
| RFC 9370 (multiple key exchanges) | 2.2.2 IKE_INTERMEDIATE keying, Appendix A (flows only, no values) | https://www.rfc-editor.org/rfc/rfc9370.txt | 81487 | `16160fa320dc06b75a759cc06a21d138bf126ec4c129b2193a8f413c190b1904` |
| RFC 8784 (PPK) | 3 key derivation, 6 PPK entropy | https://www.rfc-editor.org/rfc/rfc8784.txt | 42663 | `e12eae449f9a96743d8edb38a944d438815b01592314a46731a8fa380f676538` |
| RFC 5903 (ECP groups) | 7 KE data and shared secret format, 8.1 group 19 example | https://www.rfc-editor.org/rfc/rfc5903.txt | 29175 | `939fab548a6e6bb49a5b3c4dd24a3c5df54a46645447b2d6f4df4fd88ff2d69f` |
| RFC 9242 (IKE_INTERMEDIATE) | the exchange RFC 9370 runs in | https://www.rfc-editor.org/rfc/rfc9242.txt | 35365 | `712d0899c26a1be55a295834b1a72331a97e04d6828002fd58fef759558bc1dd` |
| RFC 9867 (PPK in IKE_INTERMEDIATE) | 3.1 negotiation and Table 1, 3.1.1 ordering after RFC 9370 | https://www.rfc-editor.org/rfc/rfc9867.txt | 29918 | `10af5b184e1f2849e5bb14f0636b9f9049a7914640c8e117ced5d3043a5b1d92` |
| RFC 5282 (AEAD in IKEv2) | 7.1 SK_ai/SK_ar are zero octets, SK_e carries salt | https://www.rfc-editor.org/rfc/rfc5282.txt | 42137 | `cb0bf136b49b0b635aa90f8fd74b0405681a7dae214990e41c1ba1da876164a4` |
| RFC 4106 (AES-GCM ESP) | 8.1 KEYMAT for a 256-bit key is 36 octets | https://www.rfc-editor.org/rfc/rfc4106.txt | 23399 | `2a0d8b0274cf307156e18708a13e6f9bfd0ed5124dca234b0a1ed1590a5a886a` |
| RFC 4868 (HMAC-SHA-2 in IPsec) | 2.1.2 PRF usage | https://www.rfc-editor.org/rfc/rfc4868.txt | 41432 | `40feb304d348e909ad224a79507ebd56012b257f03c88fc472c753a000dfadbe` |
| Noise Protocol Framework, revision 34, 2018-07-11 | 4.3 HKDF, 5.1-5.3 state objects, 6 prologue, 7.5 IK, 9.2 psk token and psk-mode `e`, 9.4 IKpsk2, 12.1 / 12.3 / 12.7 | https://noiseprotocol.org/noise.html | 142833 | `05d1f2305f81f49ce516530fd70884014f9bae54f8f290a4274b3392b1313d68` |
| (same, PDF) | | https://noiseprotocol.org/noise.pdf | 389431 | `7562d34c04a661f5679918e56b139b066bd9fa6b821080dfe59eb8f80c7dccb7` |
| WireGuard whitepaper, draft revision e2da747, 2020-06-01 | 5.4 constants, 5.4.2 initiation, 5.4.3 response, 5.4.4 mac1/mac2, 5.4.5 transport keys | https://www.wireguard.com/papers/wireguard.pdf | 550175 | `b4cf398c21d054e8774af568395b0f7cd0e2b01322809c5dbd66c4135021ca57` |
| draft-ietf-ipsecme-ikev2-mlkem-09, 2026-07-05 | 2 shared secret used like g^ir, 2.1 KE payloads, 2.2 recipient tests, 4 IANA, Appendix A | https://www.ietf.org/archive/id/draft-ietf-ipsecme-ikev2-mlkem-09.txt | 31307 | `f93e8c536b34ffe52dd6a281212b278a16573cbf532465ac473789e7951b155e` |
| IANA IKEv2 parameters (Last-Modified 2026-10-21) | Transform Type 4: 19 = 256-bit random ECP, 36 = ml-kem-768; Type 2: 6 = PRF_HMAC_SHA2_384; Type 1: 20 = ENCR_AES_GCM_16 | https://www.iana.org/assignments/ikev2-parameters/ikev2-parameters.xml | 93531 | `34c2d1e89049a167b51296b82c155f3a3cbab669f9ef8b701fc8a03c4af52545` |

**Status of ML-KEM in IKEv2, as of 2026-10-24.** Not an RFC. The IETF datatracker
(https://datatracker.ietf.org/doc/draft-ietf-ipsecme-ikev2-mlkem/) shows draft-ietf-ipsecme-ikev2-mlkem-09,
latest revision 2026-07-05, intended status Proposed Standard, IESG state "RFC Ed Queue". IANA already
lists `ml-kem-768` as Key Exchange Method 36 with the draft as its reference. The datatracker page is
generated per request, so it is recorded by date and state rather than by hash.

**draft-ietf-ipsecme-ikev2-qr-alt is now RFC 9867** (Proposed Standard, November 2025). It is relevant
because this module combines RFC 9370 with a PPK; see the composition note below.

## Test vectors (stored subsets in `web/src/lib/netkeys/vectors/`)

### NIST ACVP, ML-KEM (FIPS 203) - `mlkem768_acvp.json`

Repository https://github.com/usnistgov/ACVP-Server at commit `975de31eb83d87039ec88934fdc47d8c312b892d`
(2026-08-12, "RELEASE/v1.1.0.43 including hotfix patches"). Raw URLs are
`https://raw.githubusercontent.com/usnistgov/ACVP-Server/975de31eb83d87039ec88934fdc47d8c312b892d/<path>`.
Each file's git blob id was recomputed and equals the one the GitHub tree reports at that commit, so the
bytes are the repository's.

| Path (under `gen-val/json-files/`) | Bytes | SHA-256 |
|---|---|---|
| `ML-KEM-keyGen-FIPS203/prompt.json` | 16066 | `3f9ce34f6c836c77958bad2729e837c3b213f44ac36c3065976e7acca6389523` |
| `ML-KEM-keyGen-FIPS203/expectedResults.json` | 544032 | `a253d0ad91c95ebea5b409673defef0aa49d65d4ed72286399e2e798ddf073a4` |
| `ML-KEM-encapDecap-FIPS203/prompt.json` | 624189 | `998e22dfb12efb14ce9fdff911ca634b13612819a1806f25da69adba7e16db91` |
| `ML-KEM-encapDecap-FIPS203/expectedResults.json` | 190940 | `9089ec6ff2424da9f2782b89b2f831a329a3e28d6e5e24b802b78ff36ac61cdf` |
| `ML-KEM-encapDecap-FIPS203/internalProjection.json` (only the decapsulation "reason" label) | 1465634 | `a556952ce869bb89c3a3196a701dad89647c193a34c86eafb61a9d710d5b810f` |

Stored, all ML-KEM-768: keyGen tgId 2 tcIds 26, 27, 28; encapsulation tgId 2 tcIds 26, 27, 28;
decapsulation tgId 5 tcIds 86 and 87 ("valid decapsulation") and 88 and 91 ("modified ciphertext", whose
answer is the implicit-rejection key). Before the subset was cut, all 25 + 25 + 10 ML-KEM-768 cases in
these files were run through `@noble/post-quantum` 0.7.0 and all passed; the subset is a selection, not
the survivors of one. Licence: the repository README's "License" section states that software developed
by NIST employees is not subject to copyright protection within the United States.

The 64-byte seed noble accepts is `d || z`. FIPS 203 passes d and z as an ordered pair, so the packing is
a convention; the check asserts that `z || d` does not reproduce the answer.

### Noise_IKpsk2_25519_ChaChaPoly_BLAKE2s - `noise_ikpsk2.json`

The Noise protocol wiki's test-vector page (https://github.com/noiseprotocol/noise_wiki/wiki/Test-vectors)
lists cacophony, three noise-c files and snow-multipsk, and designates no set as authoritative. Only
cacophony has a vector for exactly this protocol name: the noise-c files (searched at `master` on
2026-10-24) use the pre-revision-32 `NoisePSK_` scheme, a different construction; the snow-multipsk link
returns HTTP 404; and snow's current `snow.txt` and `snow-extended.txt` carry only multi-PSK variants.

| | |
|---|---|
| Repository | https://github.com/haskell-cryptography/cacophony (the wiki links its former name, centromere/cacophony) |
| Commit | `8ee9d41e34a1a596cfa3ab12aa4069ff87dc1247` (2025-01-09, "Bump version to 0.11.0") |
| File | `vectors/cacophony.txt`, 1709817 bytes, SHA-256 `3bde7c09a6f349ee11c825c50fcc02649f8f02a47c857a459206b357f9386cae`, git blob `b8a271ed1aba8b4a56bf429e559d7947827123b4` |
| Stored | `vectors[505]` of 944, verbatim: both handshake messages, four transport messages, the handshake hash |
| Licence | Unlicense (repository licence per the GitHub API) |

snow (https://github.com/mcginty/snow, Rust, Apache-2.0) vendors a byte-identical copy at commit
`8ac60f51cfe3e010c84f0a454cc575ad9204fa12` (same git blob id, `tests/vectors/cacophony.txt`) and runs it
from `test_vectors_cacophony` in `tests/vectors.rs`, so the vector is produced by one implementation and
consumed by another. snow's own `snow.txt` (SHA-256
`69da433305fd045f6c9f01b656662a389d022688986fd39fbe7af009cd402fd3`) has no vector with this exact name.

### NIST CAVP, SP 800-135 IKEv2 KDF - `ikev2_cavp.json`

| | |
|---|---|
| Page | https://csrc.nist.gov/projects/cryptographic-algorithm-validation-program/component-testing ("Key Derivation Functions (SP 800-135)", "IKEv2 KDF Test Vectors") |
| File | https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Algorithm-Validation-Program/documents/components/800-135testvectors/ikev2.zip |
| Retrieved | 2026-10-24; HTTP Last-Modified Sun, 29 Sep 2024 23:13:46 GMT |
| ikev2.zip | 890382 bytes, SHA-256 `813c9fa9bf99a78659143d1898b3adffd09ad235be1580cf0ba8cd3663ae2950` |
| ikev2.rsp inside | 715805 bytes, SHA-256 `1ba2b56dc5b94bc796aa8ebf3c3f8d1ae76e2eafb27a8f8ef9fa82fc8ba0431c`, "CAVS 12.0", generated 2012-03-16; 28 sections, 280 cases |
| Field definitions | ASKDFVS section 6.3, https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Algorithm-Validation-Program/documents/components/askdfvs.pdf, SHA-256 `e75a5a92f52ce27e823f98c458f6bb6fa94f9b5d99e1b74cb8887d874dc2b304` |

Stored: COUNT 0 of six sections - HMAC-SHA-384 with g^ir of 1024 bits (64-bit and 2048-bit nonces), 185
bits and 2048 bits; HMAC-SHA-256 with g^ir 2048 / nonces 2048; HMAC-SHA-512 with g^ir 2048 / nonces 64.
Every PRF `ikev2.ts` accepts has a case. Each case checks five outputs: SKEYSEED, DKM, DKM(Child SA),
DKM(Child SA D-H), SKEYSEED(Rekey). Before the subset was cut, all 280 cases were reproduced with
Python's own `hmac`/`hashlib`, which is what fixed the reading of each field (SK_d is the first
prf-length octets of DKM). The CSRC page says these vectors are for informal verification and do not
replace CAVP validation. Half the sections use 64-bit nonces, below RFC 7296 section 2.10's minimum; the
KDF does not depend on nonce length.

The same file carries RFC 5903 section 8.1's group 19 example (i, r, g^i, g^r, girx), copied from the
RFC text above.

### WireGuard initial state - `wireguard_initial_state.json` (partial, not authoritative)

boringtun (https://github.com/cloudflare/boringtun, BSD-3-Clause) at commit
`6dcc889a95ad82400932f785d874421d70308195` (2026-06-15), file `boringtun/src/noise/handshake.rs`
(36430 bytes, SHA-256 `6c2e3eada1e0095f6dae7829540c135c6251f71602f439d818ad6ca3c7a2a262`), hardcodes
`INITIAL_CHAIN_KEY` = Hash(Construction) and `INITIAL_CHAIN_HASH` = Hash(Ci || Identifier). The two
32-byte constants were parsed from the source, converted to hex, and confirmed with Python's
`hashlib.blake2s` before being stored. wireguard-go computes the same values at start-up
(`device/noise-protocol.go`, `init()`) instead of publishing them, so no authoritative implementation
offers them as constants.

## What is validated, and what is computed per spec only

| Layer | Where | Held to | Kind |
|---|---|---|---|
| ML-KEM-768 KeyGen (d, z -> ek, dk) | `mlkemKat.ts` | ACVP tcIds 26-28 | official vector |
| ML-KEM-768 Encaps (ek, m -> c, K) | `mlkemKat.ts` | ACVP tcIds 26-28 | official vector |
| ML-KEM-768 Decaps, including implicit rejection | `mlkemKat.ts` | ACVP tcIds 86, 87, 88, 91 | official vector |
| Noise IKpsk2: HKDF, MixKey / MixHash / MixKeyAndHash, EncryptAndHash, psk2, both messages, handshake hash, Split(), transport nonces | `noiseIKpsk2.ts` | cacophony vector 505 (listed by the Noise wiki, consumed by snow) | published vector |
| WireGuard Ci and Hi (Construction, Identifier as prologue) | `wireguard.ts` | boringtun constants | independent implementation, one step |
| WireGuard framing: type, reserved, indices, TAI64N, mac1, mac2, the 148 / 92 sizes | `wireguard.ts` | nothing | per whitepaper only |
| WireGuard transport keys | `wireguard.ts` | Split() of the vector-checked core | per spec on a validated core |
| IKEv2 prf, prf+ | `ikev2.ts` | CAVP, three PRFs | official vector |
| SKEYSEED, SK_* stream, Child SA KEYMAT, rekey SKEYSEED | `ikev2.ts` | CAVP SKEYSEED, DKM, DKM(Child SA), DKM(Child SA D-H), SKEYSEED(Rekey) | official vector |
| ECP-256 KE data and g^ir = x coordinate | `ikev2.ts` | RFC 5903 section 8.1 | official example |
| SK_* lengths for the demo suite (48 / 0 / 0 / 36 / 36 / 48 / 48) | `ikev2.ts` | RFC 7296 2.13-2.14, RFC 5282 7.1, RFC 4106 8.1 | per spec only |
| RFC 9370 additional key exchange: SKEYSEED(n), SK_*(n) | `ikev2.ts` | nothing - no official vectors exist | per spec only |
| RFC 8784 PPK mixing of SK_d, SK_pi, SK_pr | `ikev2.ts` | nothing - no official vectors exist | per spec only |
| ML-KEM in IKEv2 (ek in KEi(1), ciphertext in KEr(1), SK(1) = ss) | `ikev2.ts` | draft-09 text; the draft has no vectors | per draft only |

## Seeded defects: what the checks catch and what they cannot

Each defect was seeded into a scratch copy of the modules and run through the real check script. The
two that pass are the point of the table, not a failure of it: they sit in the layers with no vectors.

| Seeded defect | Result | Caught by |
|---|---|---|
| Noise: drop the psk-mode MixKey(e.public_key) | caught | message 0 differs from byte 32 |
| Noise: big-endian nonce | caught | message 4 (the first with n = 1; n = 0 encodes the same both ways) |
| Noise: responder sends with c1 | caught | transport decrypt fails |
| Noise: HKDF outputs swapped | caught | message 0 |
| Noise: prologue not hashed | caught | message 0 differs from byte 64 |
| IKEv2: prf+ concatenates S before T(n-1) | caught | DKM differs from byte 48 (T1 is unaffected) |
| IKEv2: SKEYSEED with key and data swapped | caught | SKEYSEED |
| WireGuard: empty prologue instead of the Identifier | caught | boringtun Hi; passed every other check before that file was added |
| WireGuard: mac1 keyed with the sender's static key | caught | the responder rejects mac1 |
| ML-KEM: 64-byte seed packed as z then d | caught | keyGen tcId 26 |
| RFC 8784: prf+(SK_d', PPK) instead of prf+(PPK, SK_d') | **not caught** | nothing can be: no vector exists |
| RFC 9370: SKEYSEED(1) keyed with SK_pi(0) instead of SK_d(0) | **not caught** | nothing can be: no vector exists |

## Notes a reader of the code needs

**RFC 8784's primes point the other way.** In section 3, SK_d' is the key the standard derivation
produced and SK_d = prf+(PPK, SK_d') is the result. `ikev2.ts` follows the RFC and calls the primed
keys `beforePpk`. SK_ei, SK_er, SK_ai and SK_ar are not touched.

**The RFC 9370 + RFC 8784 composition is a reading, not a rule.** Neither RFC defines their
combination. `ikev2.ts` applies RFC 8784 to the keys of the last RFC 9370 round, because RFC 9370
section 2.2.2 says those are the keys used in the next exchange and RFC 8784 mixes the PPK at IKE_AUTH.
RFC 9867 is consistent with it: its Table 1 keeps RFC 8784 available alongside IKE_INTERMEDIATE, a
responder picks one PPK method, and for its own method (SKEYSEED' = prf+(PPK, SK_d), then all seven
keys re-derived) section 3.1.1 requires the PPK step to follow every RFC 9370 recalculation. RFC 9867's
method is not implemented.

**WireGuard has no public fixed handshake vector from an authoritative implementation.** The
whitepaper has none; wireguard-go's `device/noise_test.go` (commit
`2631ce99a06f27120d581611cf125d68bc6aa565`, SHA-256
`3c4f711b91a6adc1a011da8c92eb344abd5699178b3fb329b613ac6d86bb9e21`) handshakes between randomly keyed
devices. `wireguardHandshake()` therefore reports `validatedAgainst: "Noise vectors only; WireGuard
framing per the whitepaper"`, plus the boringtun cross-check as a separate, labelled field.
wireguard-go's `device/noise-protocol.go` (SHA-256
`f0dcd7b46f776d2f554ea5a638fa33382c8cba5eb74edd983e5e87fb85705009`) states the same construction,
identifier, label and the 148 / 92 sizes, and its `tai64n/tai64n.go` (SHA-256
`536f5f88a4c2eb68b23ba140276534ab0cf1a47c31b8444372e58e7ad80edf01`) writes the TAI64N label as
0x400000000000000a plus Unix seconds, which `wireguard.ts` follows. No code was copied from either.
No interoperability run against a live WireGuard implementation was made: no Go toolchain was present
and bringing up a kernel interface was out of scope.

**Demo inputs are derived, not random.** Both demos derive every key, nonce, SPI, index and the default
PPK from a seed so a page is reproducible. Real IKEv2 and WireGuard draw them at random. A PPK derived
from a typed seed has only the seed's entropy; RFC 8784 section 6 calls at least 256 bits the strongest
practice.

## Re-checking the upstream files

```
curl -sL <url> | sha256sum      # compare with the tables above
cd web && npm run -s check:netkeys
node scripts/check_net_key_schedules.mjs --corrupt=mlkem   # must exit 1 and name ACVP tcId 88
python -m pytest -q tests/test_net_key_schedules.py
```

If a re-fetched file hashes differently, upstream changed it. That is information, not a failure:
re-run the subset against the new file, then update the hash and the date here.
