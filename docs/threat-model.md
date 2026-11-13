# Threat model

## Adversaries
- **Harvest-Now-Decrypt-Later (HNDL)**: records ciphertext today to decrypt once a CRQC
  exists. Mitigation: PQC-first hybrid (ML-KEM+X25519) on every channel - confidential
  as long as *either* primitive holds; the X25519 half covers today, ML-KEM the quantum future.
- **Quantum adversary at Q-Day**: breaks RSA/ECC. Mitigation: ML-KEM/ML-DSA (FIPS 203/204);
  QKD is an optional *add-on* to the PQC hybrid, never standalone - this project's choice, which
  neither BSI nor NSA recommends ([their positions, in the roadmap](pqc-roadmap.md#standards-alignment-done--in-progress)).
  Real QKD would add key material whose secrecy rests on physics rather than on a hardness
  assumption; here the quantum channel is simulated, so the add-on contributes none of that security.
- **A substituted model file**: a Hub, its CDN or the network serves other bytes for a pinned revision. A pin names a commit and says nothing about what comes back for it. Mitigation: the browser checks every model file, weights file and evaluation sample against a SHA-256 that ships with this site, and refuses a file that does not match or has no entry (`web/src/lib/pinnedFetch.ts`; the record is `models/hub_file_hashes.json`, written by `scripts/pin_hub_hashes.py`). A refused file is reported as refused, and no other figure is put in its place. Not covered: the in-browser LLM's weight shards, which the WebLLM engine fetches itself and whose integrity option covers only its configuration, tokenizer and model library. The record is as trustworthy as the site that serves it.
- **Honest-but-curious compute server** (delegated AI): would read inputs/model. Mitigation:
  QHE (QOTP, perfectly hiding as a scheme; the demo's pad is seeded) for the quantum path; CKKS for
  the classical path, with masked HE-to-MPC conversions in the full deployment; secure aggregation
  hides per-client updates in federated learning from an honest-but-curious aggregator, each pair of
  clients agreeing its mask seed by X25519, though here every client runs in one tab or one process
  and a client that drops breaks the round. What each of these hides, in the browser and in the full
  deployment: [the table below](#what-each-step-protects-in-the-browser-and-in-the-full-deployment).
- **Malicious MPC participant**: injects bad inputs / aborts. No mitigation here: the protocols assume
  semi-honest parties, and neither identifiable abort nor QOT test rounds exist in this repository. The OPRF PSI is secure against semi-honest banks only: a
  bank that deviates from it is **not** detected
  ([the table below](#what-each-step-protects-in-the-browser-and-in-the-full-deployment)).
- **QKD link attacker**: intercept-resend, PNS, Trojan-horse, detector-blinding, RNG
  manipulation. Mitigation: QBER monitoring + two-intensity decoy-state single-photon bounds; QRNG
  health + **SP 800-90B min-entropy** and section 4.4 continuous health checks guard the entropy source; the
  config-gated **Toeplitz+QLHL PQC+QKD combiner** (`its_toeplitz`). Note the property carefully: `src/qsh/pqc/transport.py`
  explicitly declines the "secure-if-either" framing, because plain XOR already gives
  that and the cited paper never claims superiority over XOR. What the paper claims is narrower -- if the
  combined output AND one input key are compromised, the remaining key material retains quantifiable
  min-entropy -- and that claim needs a seed independent of the input. Here the extractor's seed is
  `SHAKE-256` over the same two keys it extracts from (`transport.py:64`), so as implemented it is a
  **computationally-seeded** extractor, not an information-theoretic or leakage-resilient one, and its
  QKD input is a fixed constant on the MCP channel. The docstring says so; the label "ITS" does not; + QML-IDS (WS2) classifying 7 attack types against normal traffic (8 classes).
- **Classical control-plane adversary against the QKD stack** - recorded 2026-08-12 as a **GAP**,
  because until then this document had no such adversary at all. Every QKD adversary above attacks the
  quantum channel or the network crypto; none attacks the **classical procedure text** that drives the
  quantum layer. Malik, Anwar and Raza, "Beyond the Quantum Promise: A Security Analysis of Classical
  Control in Quantum Key Distribution" (arXiv:2608.07626, 2026-08-07), build the first hybrid Tamarin
  model of the ETSI and ITU-T QKD specifications and find **three specification-level
  vulnerabilities** - subverted entanglement injection, basis-deferred measurement, and message
  reflection - each arising from *"a classical control-plane omission in the procedure text"*, with
  verified countermeasures (**measurement commitment**, **identity-bound MACs**). This lands on
  `src/qsh/qkd/sdn_controller.py`, which models ETSI GS QKD 015 `PHYS`/`VIRT` links, and on
  `qkd/kme.py` (GS QKD 014). **Mitigation: none implemented**, and the uncomfortable part is why -
  the findings are in the specifications this project implements faithfully, so implementing them
  faithfully reproduces them. **Resolving condition**, so this is falsifiable rather than a standing
  worry: it closes when the controller and KME either (a) bind SAE identity into the authentication of
  every control-plane message instead of accepting it as a parameter - the same missing identity
  binding the GS QKD 014 gap below already records, so one fix serves both - or (b) declare in-band
  which of the three findings a given payload is exposed to, the way `trust_note()` already declares
  trusted-relay exposure. Until one of those exists, the control plane here is **modelled, not
  defended**.
- **Network MITM / tampering**: Mitigation: AEAD (AES-256-GCM) record layer + ML-DSA
  message authentication over the hybrid session.

## Trust assumptions
- Endpoints (clients) are honest and hold their own secret keys.
- The QKD hardware is modelled by simulation; real deployments must address device
  imperfections (the WS2 IDS targets exactly these).
- The crypto/compute libraries are trusted as implemented, after review: server-side
  TenSEAL / OpenFHE / liboqs / MPyC (and HEonGPU with this project's binding, where it is built); client-side **node-seal** (CKKS) + **Polytune /
  Garble** (garbled-circuit 2PC, Rust compiled to wasm from `wasm/qsh-mpc`, Round 65) for the in-browser
  demo; the **MCP Python SDK** (`MCPServer`, named `FastMCP` before v2.0.0) for
  the full-deployment MCP server. Only some of them are pinned. The browser's npm packages and the
  Rust crates are fixed by lockfiles (`web/package-lock.json`, `wasm/qsh-mpc/Cargo.lock`), Polytune by
  commit, and the image's liboqs C library by tag (0.16.0). The Python packages are version floors
  (OpenFHE is also capped below 1.6) with no lockfile, so a fresh image can resolve newer releases
  than the ones measured here: TenSEAL
  0.3.16, which this repository's environment runs, bundles Microsoft SEAL 4.1.1, and TenSEAL 0.3.17
  and later, which its floor admits, bundle SEAL 4.3.3. That release itself introduced process termination on
  truncated compressed input, fixed in SEAL 4.5.1 (2026-10-09); the same "never deserializes foreign objects" bound
  covers it.
- **Server-tier MPC** (full deployment, `src/qsh/mpc/parties.py`): three MPyC parties, the client
  who holds the CKKS key, the server who holds the model, and a helper; Shamir with t = 1, so
  **semi-honest with an honest majority**. What the masked conversions hide is in
  [the table below](#what-each-step-protects-in-the-browser-and-in-the-full-deployment). Here the
  three run on one host over loopback; a deployment that relies on the honest majority needs them on
  three hosts under three operators.
- **MCP tool dispatch** (full deployment): the quantum-safe channel authenticates + seals the
  `tools/call` and its result (ML-DSA + AES-GCM), but the tool *logic* runs server-side and is trusted;
  the official filesystem MCP server is given one folder, fixed on the server (`MCP_FS_ROOT`, the bundled
  public sample data), and `/api/mcp/filesystem` accepts only its read-only tools, so a client can read
  that folder and nothing else. Until Round 86 the caller chose the folder, the tool (write tools
  included) and the arguments, unauthenticated, with CORS open to every origin and the port published
  on every interface; this sentence said the server was sandboxed. CORS is now off unless the operator
  names origins, and the compose file binds the API to loopback. The public demo runs no
  server-side MCP (client mirror only), so it adds no server trust surface.

## What each step protects, in the browser and in the full deployment

The public demo runs every step of a use case in one browser tab; the full deployment runs the same
pipeline on a server. A step's name does not say what it hides, and the two tiers differ, so this
table states it per step. In both tiers the quantum channel is simulated: no QKD hardware is
involved anywhere in this project.

| Step | In the browser (the public demo) | In the full deployment |
|---|---|---|
| Session key | ML-KEM-768 + X25519, combined by HKDF-SHA256 over both shared secrets and the transcript. **No QKD bit enters it**: the browser's simulated channel reports the key length a block would carry and derives no key material. | The same hybrid. When the simulated block yields 256 secure bits or more, a 256-bit key derived from the simulated sifted bits is folded in by the configured combiner: HKDF-SHA256 by default, or the Toeplitz extractor `its_toeplitz`, seeded with SHAKE-256 of its own inputs. The channel is simulated, so this adds no physical secrecy, and no key here is information-theoretically secure. |
| Agent (MCP) channel | Two endpoints in the tab, each with its own keys, establish an ML-KEM-768 + X25519 session, optionally mixed with a fixed 32-byte "QKD-domain" constant, and exchange an ML-DSA-65-signed, AES-256-GCM-sealed `tools/call` request of MCP revision 2026-07-28 (`web/src/lib/clientMcp.ts`). No QKD key reaches it; no tool is executed, so no response follows. | The same channel, with the same constant and no QKD key, carries the `tools/call` to the in-process registry of the tools the MCP server also exposes, and its signed response; the tool logic is trusted ([Trust assumptions](#trust-assumptions)). It serves that one method and is not a complete MCP implementation; the SDK server does not run over it. |
| PSI (finance) | A semi-honest DH-OPRF PSI on RFC 9497's OPRF, mode 0x00, suite ristretto255-SHA512 (the group is RFC 9496's ristretto255), run with `@noble/curves`' `ristretto255_oprf` (`web/src/lib/oprfPsi.ts`), both banks in one tab. Bank B, the OPRF server, draws a fresh key $`k`$ for the run; Bank A blinds each of its IDs; Bank B evaluates each blinded element and sends $`F_k(y)`$ for each of its own IDs $`y`$ in an order shuffled with the CSPRNG; Bank A finalizes $`F_k(x)`$ for its own IDs and intersects. **Bank A learns the intersection and $`\lvert B\rvert`$; Bank B learns $`\lvert A\rvert`$** and nothing else about Bank A's IDs, since a blinded element is uniform whatever the ID (RFC 9497 section 7.1, "Unconditional input secrecy"). Bank B does not learn the intersection. Bank A cannot compute $`F_k`$ without Bank B, so it can test an ID against Bank B's list only by having Bank B answer a query for it. **No malicious security**: a bank that deviates (blinding IDs it does not hold, answering under another key, sending made-up outputs) is not detected, and mode 0x00 carries no proof. **Not post-quantum**: the pseudorandomness of $`F_k`$, which keeps Bank B's list from Bank A, rests on the One-More Gap Computational Diffie-Hellman assumption (RFC 9497 section 7.2.1), so Bank A, with a quantum computer, could solve for $`k`$ from a blinded element and its evaluation and test every candidate against Bank B's outputs; Bank B's view of Bank A's IDs stays hidden even then. The three messages are sealed with AES-256-GCM under two keys HKDF-SHA256 derives from the session key above, one per direction: that keeps the transcript from third parties, including one who records it now and attacks ristretto255 later, for as long as the hybrid holds, and hides nothing between the banks. | The same protocol in pure Python: `qsmpc/oprf.py` implements ristretto255 and the OPRF from RFC 9496 and RFC 9497, checked against both RFCs' test vectors and byte for byte against the browser, and `qsmpc/psi.py` runs it between the two banks in one process, sealed with the transport's `SecureChannel` under keys derived as in the browser. The OPRF runs on libsodium's ristretto255 (through `pysodium`, Round 85), which is written to run in constant time as RFC 9497 section 7.4 requires, with the pure-Python code as the fallback where the host has no libsodium and as the cross-check; the run reports which (`psi_oprf_backend`). Side channels on the host stay out of scope here. A malicious-secure PSI and a post-quantum OPRF are future work. |
| Secure aggregation (finance, medical) | Each client draws an X25519 key pair per run; each pair of clients derives its mask seed from their X25519 shared secret by HKDF-SHA256, and each round's mask from the seed by SHAKE-256 (`web/src/lib/clientQFL.ts`), the pairwise key agreement of Bonawitz et al. (CCS 2017). Client $`i`$ adds the mask for every $`j > i`$ and subtracts it for every $`j < i`$, modulo $`2^{32}`$, so the masks cancel exactly. The aggregator receives only the public keys and the masked updates, so it cannot remove a mask on its own, and the PQC session key seeds no mask. **Not post-quantum**: the seeds come from X25519 alone and the masked updates are not sealed, so the aggregator, or anyone who records the public keys and the updates, could recompute every seed with a quantum computer and unmask each update; agreeing the seeds by ML-KEM-768 + X25519, as the session key is, would close it. All clients run in one tab, which holds every private key; the aggregator's step reads none. Not implemented: the double masking and the secret-shared keys that recover from a dropout (a client that drops breaks the round), security against a malicious aggregator, and authentication of the public keys (an aggregator that substitutes its own keys can unmask; that adversary is out of scope). | Medical only. The same key agreement, seeds and mask words (`SecureAggregator`, `federated.py`, byte-identical to the browser's for the same private keys), with fresh keys every run unless a caller passes fixed ones; every client and the aggregator run in one process. Since Round 85 the shares are the browser's to the bit: each weighted update is quantised to $`2^{-20}`$ and masked modulo $`2^{32}`$, so the masks cancel exactly and the aggregate is the exact sum of the quantised updates (until then the masks were added as reals in float64 and cancelled only to rounding). The same three omissions apply. |
| Secure sum (genomics) | Additive shares modulo a prime above any total the four cohorts can reach, each mask uniform from `crypto.getRandomValues` by rejection; the cohorts are notional parties in one tab. | QSMS (Shamir and a d-level QFT), with its measurement outcomes sampled analytically rather than measured on a device. |
| Private inference | The tab generates the CKKS keys, evaluates and decrypts, so the key holder, the decryptor and the viewer are one party. Each encrypted layer decrypts its outputs only, its bias added under encryption and every other slot flooded; the medical CNN's layer-by-layer path applies ReLU and pooling in plaintext between its encrypted layers ([the CNN under MPC](#the-medical-cnn-under-mpc) is the path that does not), and the garbled-circuit argmax (Polytune) runs on logits the tab has already decrypted, its two parties in one tab, so it measures the protocol's cost and hides nothing. The CKKS half is post-quantum at these parameters ($`N = 8192`$ with a 200-bit modulus, inside the HE Standard v1.1 bound for 128-bit quantum security); the 2PC is not, since Polytune's OT extension (KOS) runs over Chou-Orlandi base OTs on ristretto255, which rest on discrete log. | Three MPyC parties ([Trust assumptions](#trust-assumptions)). Every CKKS slot the key holder can decode carries an independent mask from the OS CSPRNG, real and imaginary parts, added as a vector encrypted under the client's key, which also re-randomises the ciphertext (`src/qsh/fhe/masking.py`, used by the dense layer, the convolution and, since Round 86, the linear model whose class the stage reports, `src/qsh/fhe/ckks_inference.py`, which until then decrypted the rotate-and-sum whole, partial sums included, review PI-1; there the client decrypts the logits themselves and takes the class in the clear, so it learns the logits, not only the class). One decryption is deliberately not masked: genomics' panel check (`encrypted_scores`) reads each person's score from a block whose other slots hold partial sums of the published PGS weights' products, which the page prints anyway. The MPC opens re-masked values and, at the end, the class index to the client alone. The client's whole view is hidden statistically rather than perfectly, with its bound (`view_statistical_distance`) reported per run; on OpenFHE the context is real-typed, so the imaginary parts are not flooded, and the run says so in `masked_slots` and reports no flood margin. A client and helper who pool their Shamir shares recover the server's masks, and with them every intermediate. |

Each key's lifetime and source is in the next section; what would close each gap is under the
residual risks below. The masks' statistical bounds, and what they do not cover, are derived in
[`architecture.md`](architecture.md#where-the-non-linear-half-goes-and-why-that-choice-is-a-security-decision).

## Key management (what each key is, and what it is not)

Enumerated because "key management is out of scope" reads as *not yet built*, when in fact several
keys exist and some of them are recomputable by a party they were meant to hide from.

| Key | Lifetime | Where it lives | Caveat |
|---|---|---|---|
| Hybrid KEM (ML-KEM-768 + X25519) | Per session, ephemeral | Process / tab memory | Both halves genuinely computed on both sides. Never persisted, never escrowed. |
| QKD-derived key material | Per run, full deployment only | Memory | Simulated channel - no QKD hardware is involved anywhere in this project. Which keys it enters, and why none is information-theoretically secure: [the table above](#what-each-step-protects-in-the-browser-and-in-the-full-deployment). |
| ML-DSA signing keypair | Per run, discarded | Generated inside the signature stage | The demo now records the public half and the exact signed payload; the **backend** path still discards the public key and keeps only a 12-byte signature prefix, so a full-mode run's signature is not third-party verifiable. |
| Secure-aggregation X25519 key pairs (`clientQFL.ts`, `federated.py`) | Per run, one per client | Tab or process memory, drawn by each client from the CSPRNG (`crypto.getRandomValues` in the browser, `X25519PrivateKey.generate()` in Python) unless a caller passes fixed private keys, as the tests do; Python drops the private keys once the seeds are derived | The aggregator receives the public keys alone, and they are not authenticated, so an aggregator that substitutes its own could unmask; that adversary is out of scope ([the table above](#what-each-step-protects-in-the-browser-and-in-the-full-deployment)). In the browser the tab holds every client's private key. |
| Secure-aggregation pair seeds and round masks (both tiers) | Seed per run per pair, mask per round | Each client's memory: HKDF-SHA256 of the pair's X25519 shared secret (salt `qsh/secagg/salt/v1`, info `qsh/secagg/pair/v1` and the two indices), and each round's mask SHAKE-256 of the seed, `qsh/secagg/mask/v1` and the round | Held only by the pair's two clients; the PQC session key seeds no mask. Both tiers add the masks modulo $`2^{32}`$ to the update quantised to $`2^{-20}`$ ([the table above](#what-each-step-protects-in-the-browser-and-in-the-full-deployment)). |
| Browser secure-argmax share masks (`clientMPC.ts`) | Per run | `crypto.getRandomValues`, 32 bits per logit; shares modulo $`2^{32}`$ (since Round 66) | Each share alone is uniform whatever the logit. Until Round 66 the mask came from the run's seeded generator (mulberry32 on the seed the page shows) and was bounded to $`2^{16}`$, so it was recomputable and only statistically hiding (distance up to 1/64). Both parties are in one tab, so this is a property of the sharing, not of two hosts. |
| Browser secure-sum masks (`clientCompute.ts`) | Per run | `crypto.getRandomValues`, uniform over $`\mathbb{Z}_p`$ by rejection (since Round 66) | Until Round 66 `floor(u() * p)` from the same seeded generator: recomputable, and biased because no odd prime divides $`2^{32}`$. |
| PSI OPRF key $`k`$ (Bank B; `oprfPsi.ts`, `qsmpc/psi.py`) | Per run, fresh | Bank B's memory: `crypto.getRandomValues` in the browser, the OS CSPRNG in Python | Never sent: Bank A receives evaluations under it, not the key. A fresh key per run is one of the mitigations RFC 9497 section 7.2.3 names for the static Diffie-Hellman attack. Its secrecy against Bank A is computational and not post-quantum ([the table above](#what-each-step-protects-in-the-browser-and-in-the-full-deployment)). |
| PSI blinds (Bank A) | Per ID, per run | Bank A's memory, from the same CSPRNGs | Each blinded element is uniform whatever the ID (RFC 9497 section 7.1). |
| PSI transcript keys | Per run | HKDF-SHA256 of the PQC session key, one info string per direction | Keep the PSI's three messages from third parties for as long as the session key holds. Both banks hold them, so they hide nothing between the banks. |
| QOTP demo key (`quantumSim.ts`) | Per run | The run's seeded generator | **Disclosed in place since Round 40, not changed.** The pad is perfectly hiding as a scheme, but this key decides the circuit's output, so it stays reproducible and anyone with the seed can recompute it. |
| ETSI KME key IDs (`qkd/kme.py`) | Per issued key, in memory | The KME's own store | Bound to one `(master_SAE_ID, slave_SAE_ID)` pair, so a `key_ID` observed in transit is not a bearer token. That binding is the only thing protecting it, because **mutual TLS is absent** - see below. |

**The GS QKD 014 authentication gap, stated rather than implied.** The spec mandates mutual TLS with
the SAE identity taken from the client certificate. This project's KME runs in-process and has no
transport to attach a certificate to, so **SAE identity here is a parameter, not an authenticated
fact**: any caller can name itself `sae-master`. The key-ID binding above is therefore meaningful
only against a caller who does not lie about its own identity, which is not a threat model - it is a
convention. A deployment must add mTLS before that binding means anything, and
`KeyManagementEntity.authentication_note()` says so at the call site so the gap cannot be read as an
oversight.

**Trusted relays are an assumption, not a mechanism.** When the configured distance exceeds a single
quantum span, `qkd/sdn_controller.py` creates a VIRT link through a relay node, and that relay holds
the key **in plaintext**. End-to-end secrecy then rests entirely on the relay's physical security,
which no cryptography in this repository provides. The controller is built to surface this rather
than smooth it over: `trust_note()` names every intermediate node, and the browser panel prints it.
ANSSI and BSI both press on exactly this point, and it is the honest limit of trusted-node QKD.

## What CKKS does and does not promise here

The product's headline is privacy-preserving inference under CKKS, so the security notion CKKS
actually satisfies belongs in this document. It was absent until Round 23.

**CKKS is IND-CPA, not IND-CPA-D.** The distinction is not academic for a system that returns
decrypted results:

- Li and Micciancio (EUROCRYPT 2021, ePrint 2020/1533) showed approximate FHE fails IND-CPA-D -
  security when the adversary sees DECRYPTION RESULTS. It is a **key-recovery** attack, demonstrated
  against the CKKS implementations in HEAAN, SEAL, HElib and PALISADE, with **"very modest running
  times"**.
  Two claims this document used to make were checked at primary source on 2026-08-12 and did not
  survive, so they are corrected rather than left: the abstract says "very modest running times" and
  **not "within several seconds"**, which was quoted here as if verbatim; and the key-recovery
  expression is `[UNVERIFIED]` - this document rendered it $`s = a^{-1}(m'' - b)`$ while secondary
  sources render it $`s = a^{-1} b'`$, and the PDF is unreachable (eprint returns HTTP 403, the UCSD
  mirror 404), so **neither form is asserted here**. Nothing below depends on which it is: the
  conclusion that decryption results leak the key is the abstract's own.
- **The "needs one decryption query" figure is verifiable, but from a different source.** It is
  attributed here to IBM **HElib's `CKKS-security.md`**, which states the attack can succeed with
  **"sometimes as few as a single decryption"**. Li and Micciancio is the result; HElib is the
  implementation that writes the query count down in its own security documentation.
- Li, Micciancio, Schultz and Sorrell (CRYPTO 2022, ePrint 2022/816) give the noise-flooding
  countermeasure and its precision cost.
- Cheon, Choe, Passelegue, Stehle and Suvanto (ACM CCS 2024, ePrint 2024/127) extended the attack
  class to EXACT FHE schemes.
- Checri, Clet, Renard and Sirdey (ePrint 2026/203) prove an **impossibility**: no "HELLHO" scheme -
  one whose homomorphic addition is linear over the ciphertext domain, which covers the basic
  variants of BFV, BGV, **CKKS** and TFHE - can achieve CPAD security at all. They also demonstrate
  a practical key-recovery attack on DE-CKKS, a scheme OpenFHE supports, achieving full key recovery
  "in a matter of hours on an average laptop PC".
- Ogilvie (ASIACRYPT 2025, ePrint 2025/1618) is the constructive counterpoint: in regimes where
  rescaling noise dominates, "an additional precision loss of as little as two bits is sufficient to
  restore a high level of security against passive key-recovery attacks for standard parameters".
- Bian, Choe, Kim and Lee (ePrint 2026/2490, 2026-10, to appear at IEEE S&P 2027, implemented on Lattigo v6.2.0) need no decrypted value at all: a malicious
  evaluator that holds the public coefficient-to-slot (bootstrapping) keys and sees only whether the
  client accepted a result, a 1-bit validity response, recovers a ternary secret key in a number of
  queries of the order of the key's entropy. It does not break IND-CPA. The abstract does not discuss
  noise flooding; this project's reading, not the paper's, is that flooding does not close it, because
  what leaks is the reaction rather than a value.

**The implementations say this in their own security documentation, and node-seal IS SEAL.** The demo
runs Microsoft SEAL compiled to WASM, so SEAL's `SECURITY.md` is not background reading, it is the
upstream security statement for the cipher on the page. Verbatim: *"The CKKS scheme requires
particular care because it is approximate: decryption returns the message plus a secret-key-dependent
error term rather than the exact message"*; *"Releasing such decryptions to any party other than the
secret key owner can therefore leak the secret key, even when only ordinary computation results are
shared"*; and *"Formally, the CKKS implementation in Microsoft SEAL does not provide IND-CPA-D
security ...; it targets only IND-CPA security"*. IBM HElib's `CKKS-security.md` states the mechanism
in one sentence: *"since the scheme itself adds some error, the attacker could learn something from
the decryption result even if it knows what the decrypted result was supposed to be. Specifically, it
learns the error, which may leak information about the secret key."*

**How much of this reaches this project.** The public demo is the benign case and should be stated
as such rather than claimed as a defence: the CKKS keys are generated in the browser tab, used in
that tab, and the decrypted class is shown to the same person who holds the secret key. There is no
decryption oracle exposed to anyone else, so an IND-CPA-D adversary would have to be the key holder
attacking themselves.

**The displayed approximation error is in scope of this section, and it is safe. Both halves of that
belong in writing.** The demo does not merely decrypt a result, it reports the per-sample CKKS
approximation error itself: `web/src/lib/clientFHE.ts` computes `maxAbsError` against a plaintext
reference and `clientCompute.ts` emits it as the run metrics `fhe_max_abs_error` and
`fhe_conv_max_abs_error` (about 1e-9), which the run panels render. By HElib's sentence above that
quantity - the error - IS
the leaking object, so a reader who knows this literature is right to stop on it, and a document that
listed only "decrypted results" would have skipped the sharpest case. It is nonetheless safe here, and
the reason is the predicate SEAL states: *"Any mitigation, such as adding noise to decrypted values
before they are shared (noise flooding), must be implemented by the application before a decrypted
CKKS value crosses a **trust boundary**."* The trigger is **crossing a trust boundary**, not being
displayed. A user reading their own approximation error, produced from a key generated in their own
tab, crosses nothing: the key holder, the decryptor and the viewer are one party. The moment any of
those three separate - a server-side key, a shared session, a value posted anywhere - the same number
becomes a key-recovery oracle and the mitigation below is owed. That is the falsifiable condition,
and it is a property of the architecture rather than of the cipher.

**The server-side deployment is the case that would need work.** If a model owner ran CKKS on a
server and returned decrypted results to a client, that is precisely the CPAD setting, and the
countermeasure - noise flooding, with its documented precision cost - would be required before any
production claim. This project does not use DE-CKKS and does not perform CKKS-to-FHEW scheme
switching (ePrint 2026/285 breaks PEGASUS-style switching in OpenFHE, so this is worth stating
explicitly rather than leaving to inference).

**One rule follows from the reaction attack (Round 85): the client never tells the evaluator whether a
decryption decoded.** Not a success flag, not a failure, not an error text. OpenFHE's precision guard
("approximation error is too high"), which `OpenFHEBackend.decrypt` lets propagate, is exactly such a
bit once the parties are separate, and the bootstrap path holds public coefficient-to-slot keys with a
uniform ternary secret, the attack's setting. The setting is not OpenFHE's bootstrap keys alone
(Round 86, review PI-9). The abstract names the ingredient as *"public homomorphic linear
transformations"*, and TenSEAL, the default backend, calls `generate_galois_keys()` with no step list
(`src/qsh/fhe/backends.py`). That asks SEAL for its default set, which SEAL's `keygenerator.h` says
*"is sufficient to apply any Galois automorphism"*: here, left and right rotations by every power of
two from 1 to 2048 and the conjugation (read off the context on 2026-11-02). With plaintext
multiplications, those compose any linear map of the slots, so an evaluator holding that set holds
the attack's linear layer. That is this project's inference; no attack on TenSEAL is demonstrated.
The browser generates only the rotation steps its sums use. Today every party runs in one tab or one
process, so nothing reaches an evaluator; a deployment that separates them has to catch the failure
on the client and answer the evaluator the same way either way. Separately, Noh, Chabounia, Hanrot
and Passelegue (ePrint 2026/2575) argue that flooding needs a rigorous upper bound on the
computational error. The server tier's flood is a fixed $`2^{30}`$ (`benchmark.mpc_mask_bits`); each
run reports its margin over the largest error it measured (`flood_margin_bits`, from
`ClientView.err_max` in `src/qsh/fhe/masking.py`), and nothing fails a run whose margin is small.
That framework would call the margin heuristic. Until Round 86 (review PI-7) this said the flood was
sized from the measured error.

**"This project does not implement noise flooding" was an understatement, and it named the wrong
subject.** The three CKKS implementations are not in the same position, and this deployment runs the
one with nothing:

- **SEAL has no noise flooding at all.** Its `SECURITY.md` does not describe a parameter or a mode; it
  says the mitigation *"must be implemented by the application"*. So there is nothing to enable in
  node-seal, and the gap is upstream rather than a configuration this project declined to set.
- **HElib has it built into decryption**, with a settable accuracy target, per its `CKKS-security.md`.
  That document also gives the scaling law, which is the honest cost of the countermeasure: *"to
  withstand an attack that has access to the results from decryption of D ciphertexts, the bits
  parameter should be increased roughly by a factor of sqrt(D)"*.
- **OpenFHE has `NOISE_FLOODING_DECRYPT` plus `CKKS_M_FACTOR`**, and its security documentation states
  the tolerated query count: *"By default, OpenFHE chooses a configuration where a relatively large
  number of decryption queries of the same or related ciphertexts can be tolerated. The lower bound
  for the tolerated number of such decryption queries is N_d = 128 (but in practical computations, N_d
  is at least 10,000)."* It also scopes the attack the way this document should: it *"applies to the
  setting where decryption results need to be shared between multiple parties, e.g., in the threshold
  FHE setting"* - which is the server-side deployment, not the single-tab demo. OpenFHE is one of the
  server tier's two CKKS backends (`qsh/fhe/backends.py`; TenSEAL is the default), so it is the one
  place in this project where the mitigation can be configured at all; the backend sets neither
  option.

**Two things that stop noise flooding being read as a solved fix.** First, the countermeasure is itself
under attack: Guo, Nabokov, Suvanto and Johansson, *"Key Recovery Attacks on Approximate Homomorphic
Encryption with Non-Worst-Case Noise Flooding Countermeasures"*, USENIX Security 2024, targets exactly
the mitigation named above when the flooding is parameterised on average-case rather than worst-case
noise. Its [abstract](https://www.usenix.org/conference/usenixsecurity24/presentation/guo-qian) (read 2026-11-03)
attacks OpenFHE's noise flooding on average-case estimates, recovers the key from a single shared decryption, and
finds that PRG-derived noise adds no protection; the body could not be loaded, so no parameter or bound from it is
quoted here. It bears on the server tier's flood, a fixed $`2^{30}`$ sized against no worst case. Second, the game is weaker than "malicious": Cheon, Choe,
Passelegue, Stehle and Suvanto (ePrint 2024/127, CCS 2024) give the adversary *"access to a decryption
oracle for ciphertexts for which it should know the underlying plaintexts. This includes ciphertexts
that it (honestly) encrypted and those obtained from the latter by evaluating circuits that it
chose."* **Honestly generated inputs suffice, and the attack is passive**, so "the parties here are
only semi-honest" is not a defence against it - which is worth stating because that sentence appears
elsewhere in this document as a real mitigation for other adversaries.

**The threshold setting OpenFHE points at now has a citation here.** This section had none, while the
quote above scopes the attack to exactly that case: Zhang, Zhou, Han, Liu and Huang, "Adaptively CCA-Secure
Threshold Decryption from Lattices with Polynomial Modulus" (IACR ePrint 2026/1627, received 2026-08-06, its
fourth revision 2026-09-19, retitled), treats adaptive corruption and leakage from **decryption queries** in
the random-oracle model with bounded queries, using polynomial noise flooding and a coupled refined leftover
hash lemma. It matters here rather than in general
because HEonGPU ships N-out-of-N collective bootstrapping, so the threshold case is one dependency away
rather than hypothetical. Its collective path is correct only from v2.1.1 (2026-11-02), which fixes the round-2
relinearization-key share reading error polynomials from uninitialized memory. Nothing here calls it.

**What SEAL itself says it does not give.** The same SEAL commit as the IND-CPA-D note above (16cb511, 2026-07-25)
added two more sections to its `SECURITY.md`. *Ciphertext Authenticity*: ciphertexts are not authenticated, and crafted
ones can pass validation. *Evaluator Operand Privacy*: SEAL gives no circuit privacy, and `multiply_plain`'s timing
depends on the plaintext operand. Both bear on the full deployment, which multiplies by the model's weights as
plaintexts and claims the weights stay with the server. The masks and the flood address what the key holder decodes; they
do not address timing, which stays out of scope here.

**Newer than this section's attacks, none of them changing it** (read 2026-11-03):

- ePrint 2026/2546 (CCS 2026): IND-CPA-D attacks on BFV and BGV, including bootstrapping-failure attacks, with OpenFHE BFV
  key recovery in minutes to hours. So an exact scheme is no way out of this section (`references/he-scheme-selection.md`).
- ePrint 2026/1671: a machine-checked flooding bound with a square-root-of-queries loss, beside HElib's $`\sqrt{D}`$ rule.
- ePrint 2026/2524 (GCKKS) and 2026/2675 (noise cleaning, threshold CKKS): keeping precision after flooding.

**No standard requires testing for this.** The HomomorphicEncryption.org standard contains no
testing or conformance clause and names CKKS with others as schemes that "are not described in this standard"; ISO/IEC 28033 is not published, and the part that matters here is **Part 3** - mechanisms for arithmetic on approximate numbers, i.e. CKKS - which is at its FDIS ballot, **stage 50.20** since 2026-09-23, closing around 2026-11-18, so it may publish soon (read at iso.org on 2026-11-03). The other parts, same reading: Part 1 general DIS 40.60 (2025-12-11), Part 2 DIS 40.60 (2026-04-25), Part 4 FDIS 50.20 (2026-09-08), Part 5 WD 20.60 (2025-08-16). Part 1 scopes correctness out. Naming "Parts 1-4 at DIS" was imprecise in both directions: Part 3 is further along than DIS and is the CKKS part, so this claim has a shelf life of months and carries a date for that reason. Checked 2026-08-10. So this section is a
statement of known limitations, not a conformance gap against any published requirement.

## The medical page's federation, aggregated five ways

`web/src/lib/clientFedAgg.ts` reruns the four-hospital federation from the page's seed and changes only how each
round's size-weighted FedAvg is aggregated. What each design hides, and from whom (all parties run in one tab, so every
separation below is notional, and none of the designs authenticates a party or survives a dropout):

| design | who can read one hospital's update | what the aggregator learns |
|---|---|---|
| pairwise masks | nobody alone: a share is masked until all four are summed | the sum, in the clear |
| CKKS, one shared key (NVIDIA FLARE's design) | every hospital, since all four hold the one secret key | nothing; but with any one hospital it reads every update |
| threshold CKKS or BGV (Lattigo, 4-out-of-4) | nobody alone: decrypting needs all four hospitals' shares | the sum, which it combines from the four decryption shares |

**The decryption shares carry flooding noise, and its size is the security.** A share reveals its party's secret-key
share masked only by noise, so each is flooded with a Gaussian whose deviation is $`2^{\lambda}`$ times the sum's noise
bound for $`\lambda`$ bits of statistical security; the bound used is $`2^{14}`$, above the $`2^{12.3}`$ to $`2^{12.8}`$
measured on four fresh encryptions summed. The page shows three settings: $`\lambda = 16`$ ($`\sigma = 2^{30}`$, the
deviation Lattigo's multiparty examples use: a demonstration), $`\lambda = 40`$ and $`\lambda = 128`$. CKKS cannot
reach 128 bits here: the decoded error is about $`\sigma\sqrt{N}/\Delta`$, so a usable aggregate needs a scale near
$`2^{168}`$, and Lattigo caps the scale at $`2^{128}`$, where the measured error was about $`2.8 	imes 10^{6}`$.
BGV decrypts exactly whenever the noise stays below $`Q/(2t)`$, so it carries the 128-bit flooding in a 180-bit
modulus, inside the 218 bits the HomomorphicEncryption.org standard allows at $`N = 8192`$: with a 33-bit $`t`$,
$`Q/(2t)`$ is $`2^{146}`$, eight standard deviations of the four shares' summed flooding noise ($`2^{143}`$).

**A share is never made twice for the same ciphertext.** Lattigo's SECURITY.md says repeating a multiparty protocol must
be considered insecure, since a second share of the same ciphertext under the same key share leaks; each round here
encrypts fresh updates and decrypts their sum once, and the aggregate ciphertext never leaves the function that
decrypts it. **Every prime of the modulus stays above six times the flooding deviation**: below that, Lattigo
v6.2.0's Gaussian sampler writes unreduced residues and the decryption fails (reported as tuneinsight/lattigo#596), and
`wasm/qsh-fedthresh/agg/agg_test.go` holds every setting to it.

## The medical CNN under MPC

`web/src/lib/clientHybridCnn.ts` runs one held-out image through the published student without either party holding an activation, in the shape of Gazelle (USENIX Security 2018). A client holds
the image and the CKKS secret key, and a server holds the weights. The server evaluates a linear layer under CKKS, adds an encrypted mask to every slot and returns the ciphertext. The client decrypts $`x + r`$, so the two hold
additive shares of $`x`$. A Polytune garbled circuit reconstructs $`x`$ modulo $`2^{20}`$, applies the ReLU and adds a fresh mask of the server's, and only the client receives the result. Average pooling is linear, so each party applies it to its own
share. The client encrypts its share of the next layer's input and the server removes its own under encryption. The last layer's shares go into the existing 2PC argmax.

**What it hides, and how well.** The masks are bounded reals, uniform on $`[-R, R)`$, with $`R = 2^{36}`$ on the convolution and the first dense layer and $`2^{30}`$ on the logits. Such a mask hides a value only statistically: $`\mathrm{SD}(x + r, x' + r) = |x - x'| / (2R)`$. The run sums
this over every masked value, every slot of every ciphertext and every return mask, and reports the total: $`2^{-22.5}`$ to $`2^{-24.2}`$ over the 156 test images. It cannot reach the garbled circuit's $`2^{-40}`$, because the decode error of a real mask is about $`6 \times 10^{-16} R`$, measured.

**Where this departs from the literature, and why it is stated.** Gazelle, Delphi, CrypTFlow2 and Cheetah mask over a plaintext field or ring, and BLB, EncFormer and FESC mask uniformly over the CKKS plaintext ring, each of which hides the value and the CKKS noise perfectly. BLB's Section 6 calls
MP2ML's slot-encoded masks insecure, because such a mask is concentrated in the coefficient domain, and its theorem does not cover a mask of this kind. The argument concerns a message that is large relative to the mask, which this is not, but no published proof covers this construction: the bound above is this project's own. Ring-uniform masks would need about two million AND gates per ciphertext in Polytune to convert, so they do not fit in a tab.

**What the masks cover.** A key holder sees every slot's real and imaginary parts, and the CKKS noise in them depends on the weights. Masks encoded from real values leave the imaginary half alone, and node-seal 7.0.0 encodes only real vectors, so the server adds the conjugate of the ciphertext first, which cancels the imaginary half. The masks then flood the noise on the coordinates that remain. That is a measured
margin and not a worst-case bound: Guo et al. (USENIX Security 2024) recover keys from flooding based on non-worst-case noise estimates, although that attack needs a decryption shared with another party, which this protocol does not do (only the key holder decrypts).

**What it does not do.**

- **Both parties run in one tab**, so the separation is notional, and the page's comparison with the clear network is possible only for that reason.
- **Semi-honest only.** Polytune's garbling is designed against malicious parties, but the CKKS half and the shares are not authenticated. MUSE (USENIX Security 2021) extracts model weights from protocols of this shape with a malicious client.
- **Not post-quantum as a whole.** The CKKS half is at these parameters, and Polytune's OT extension runs over Chou-Orlandi base OTs on ristretto255, which rest on discrete log.
- **No protection against queries.** Anyone who can run inference learns the class, and model extraction by repeated queries is out of scope in every source read.

## The finance MLP under MPC

`web/src/lib/clientHybridMlp.ts` runs one held-out row through the published MLP in the protocol shape of the medical CNN above, without the convolution: the client scales the row and encrypts it, the server evaluates each dense layer under CKKS and masks every slot, a garbled circuit applies the ReLUs on additive shares, and the last layer's shares go into the 2PC argmax. The same limits apply, and the page prints them: both parties run in this tab, the protocol is semi-honest, and the masks are bounded reals that hide a value only statistically.

**What the row is.** A per-transaction feature for this model sees transactions up to the end of its batch of 2,048, as the Graph Feature Preprocessor paper processes them, so a real deployment of it would decide per micro-batch. The features are computed in the clear by the bank that holds the transactions: encryption protects them from the party with the weights, not from the bank that computes them, and building features across banks is a separate problem this demo does not address.

## The PSI among banks under MPC

The finance page's main PSI is two-party, RFC 9497's OPRF. Beside it, when ticked (the default), the page runs a PSI among three to five banks: one garbled circuit under Polytune's WRK17b protocol with every bank a party (`psi_source` in `wasm/qsh-mpc/src/lib.rs`).

- **What each bank learns.** The accounts all the banks hold, and the list size, which is public here (8 each). It does not learn which accounts it shares with only some of the others, which a star of two-party PSIs would reveal to its centre.
- **Position in the output.** The output has one slot per element of the first bank's list, so that bank shuffles its list with the CSPRNG.
- **Accounts are hashed to 32 bits.** A false match needs two different accounts to collide; over the $`k^2(n-1)`$ comparisons of a run the union bound is below $`2^{-22}`$ at $`n = 5`$, $`k = 8`$.
- **What it does not provide.** Every bank runs in this tab, so the separation is notional. WRK17b's security against all but one malicious party is the protocol's property, and at the pinned pre-release commit nothing here claims it between hosts. Its oblivious transfers rest on a discrete-log assumption, so it is not post-quantum, and a recording could be broken later; nothing is sent between hosts here to record.
- **Size.** The circuit compares every pair, so it is quadratic in the list. Dedicated protocols scale far beyond this (Kolesnikov et al., CCS 2017: five parties with $`2^{20}`$ items each in 72 s, semi-honest), and the 10,000-accounts target of the original plan is not met here.

## The private liquidation check (DeFi lending page)

Implemented in `web/src/lib/lending/privateCheck.ts`, in two arrangements. The sources behind the choice are in [`references/defi-lending-key-custody.md`](../references/defi-lending-key-custody.md).

**Carol holds the key (the default).** Prices, thresholds and the interest index are public, as on a real market, and Carol stands in for the threshold committees confidential-DeFi systems use.

1. Carol generates the CKKS keys. Bob encrypts $`[c_1, c_2, \tilde d / 1000]`$ once under her public key, signs it with ML-DSA and seals it to the pool under the PQC session key, and takes no further part.
2. At each check the pool applies what has been settled in the open as a plaintext update, multiplies by the public weights $`[p_1 \mathrm{LT}_1 / 1000,\ p_2 \mathrm{LT}_2 / 1000,\ -I_t]`$, sums the three slots with two rotations, adds the conjugate to cancel the imaginary half, and adds a mask on every slot, uniform on $`[-2^{36}, 2^{36})`$, which it keeps.
3. The masked ciphertext goes to Carol under a session of its own, a second hybrid handshake. She decrypts $`M + r`$.
4. The pool and Carol enter the existing two-party argmax over $`(M + \text{offset},\ \text{offset})`$. Index 1 means $`M < 0`$ strictly. Both receive the bit and Carol signs it, binding the hour and a digest of what she decrypted.

- **What each side learns.** The pool: Bob's ciphertext and the bit. Carol: the masked margin and the bit. Bob needs to be online only for his one submission.
- **What it depends on.** The pool and Carol must not collude: together they see everything. Carol must never hold Bob's ciphertext unmasked. Here only the pool keeps it; a deployment enforces that with access control, as Zama's does.
- **A single Carol.** node-seal has no threshold CKKS, so she is one party. A deployment splits her across a committee.

**Bob holds the key (the bilateral mode).** The pool's thresholds are its own. Bob encrypts afresh at each check, the pool evaluates and masks, Bob decrypts $`M + r`$, and the two run the circuit.

- **When it protects nothing.** With public thresholds Bob already knows $`M`$, so the mask and the circuit have nothing to hide. The mode only earns its place when the thresholds are the pool's own.
- **Bob has to answer.** He holds the key and he loses by a liquidation, so he could refuse or go offline. A deployment would need a rule that no answer counts as a default.
- **Probing.** A malicious Bob could probe the thresholds with crafted ciphertexts, the shape of MUSE (Lehmkuhl et al., USENIX Security 2021), and repeated bits narrow them down.

**In both.**

- **Semi-honest only.** Nothing proves Bob encrypted his real balances; commitments and zero-knowledge proofs would be needed. Every party runs in one tab, so the separation is notional, and Carol's key crosses to each worker for that reason.
- **The mask is statistical.** For $`r`$ uniform on $`[-R, R)`$, $`\mathrm{SD}(x + r, x' + r) = |x - x'| / (2R)`$, and the run reports the sum over the slots it masks. At double precision the width is capped by the decode error, which is why a margin within a few hundredths of a stable unit of zero can come out either way.
- **Post-quantum status.** ML-KEM (with X25519 as a hedge), ML-DSA and CKKS rest on lattice assumptions. The garbled circuit's oblivious transfers rest on discrete logs: a deployment's sealed session would keep them from outsiders, but a counterparty that records them could open them with a future quantum computer.
- **No chain.** There is no contract, no oracle and no real price. Aave V4's target health factor and auction-style bonus are not modelled. The closest published design found is Martinico, Calabrese, Rubin and Ciampi (ACNS 2026), which proves a stronger notion against a corrupted decryptor in the UC model; this page gives no such guarantee.

## One-way quasi-QSDC on the satellite page

Where two-way QSDC cannot close, the page sends its telemetry by STIKE (Pan et al., Sci. Adv. 11, eadt4627, 2025). What is computed is statistics: no photons and no bytes cross a channel, and in the browser the session key stays the PQC handshake's.

**What it rests on.**

- **A pre-shared key.** The message is one-time padded from a pool both ends already hold, which is why this is quasi-QSDC and not the keyless communication Qi et al.'s Eq. 8 bounds. The model assumes the pool was provisioned before the pass, by a means it does not model. Pan et al. leave the strategy for filling it to future work. Whoever holds the pool reads the telemetry.
- **Reservation before the pass.** A frame reserves a key as long as its spread codeword, 98.5 Mbit for the 74-byte frame here, since the pad is applied before anyone knows which pulses will arrive. Only the part for pulses Bob retains is spent.
- **The channel check.** As in BB84, the secrecy comes from comparing a sample of basis-matched events, with an authenticated classical channel assumed for the timestamps and bases. A run with no positive Eq. 3 rate is refused and spends nothing.

**What it does not give.** The rate is asymptotic, as in the source: a finite-length analysis is left open there and would lower it. The spreading and majority stand in for their LDPC code. The sources measure fibre and 1.4 km of urban air, so nothing read shows the protocol over a satellite link.

## The trained QCNN under CKKS

`Inference` offers the trained QCNN under CKKS where one ships. Every shipped circuit encodes a feature on its own wire as $`\mathrm{RY}(\theta)|0\rangle`$, so its readout is exactly a linear form in the $`3^n`$ products of sines and cosines of the encoded angles ([`references/REFERENCES.md`](../references/REFERENCES.md)). The client computes those products from its features and encrypts them. The server, which holds the circuit, returns one encrypted inner product per readout at one multiplicative level, and only the client can decrypt it.

**What it hides.** The input from the evaluator, under RLWE at the demo's frozen parameters. **What it does not.** The circuit is simulated classically, so this is classical computation under RLWE and not quantum homomorphic encryption; the evaluator knows the circuit; the readout reveals its value to the client, and the form's $`3^n`$ coefficients can be recovered from that many chosen scores, which matters only for a private model and is why a deployment with one would add output perturbation; and both parties run in one tab, so the separation is notional. The accuracy is the plaintext circuit's, which the class reproduces, so nothing here makes the model more accurate than it is.

## Explicitly out of scope (PoC)
- Side-channel / fault attacks on the host.
- Formal verification of the protocols (referenced papers carry the proofs).
- Production key management / HSM integration (crypto-agility hooks are provided).

## Residual risk & honest limitations
- **Accidental hardware faults.** Nothing here detects a transient bit flip. In fault injection on the
  authors' own CKKS, one flip during ciphertext multiplication could reach the decrypted output as
  silent data corruption (arXiv:2608.11147), and with NTT or RNS+NTT arithmetic, which SEAL uses, any
  single flip ended in catastrophic failure (arXiv:2608.11155); both are 2026 preprints and neither
  tested SEAL or node-seal. `encrypted_agreement` compares a CKKS decision with the same model's
  plaintext decision, so it would show a flip that changes a decision and cannot show one that does not.
- Quantum-channel results (QKD distance/QBER gains) and the QSMPC-FHE comparison row are
  **simulated**, not hardware-validated; photonic inference and qOLE are not implemented at all.
- The QHE homomorphic evaluation is exact for Clifford circuits with key tracking (the Python
  demo); non-Clifford (trainable) layers require magic-state/T-gate corrections in a real device
  (their cost is reported as a T-count, not executed fault-tolerantly here). The browser's
  comparison circuit applies its T gates to the padded state and never corrects or decrypts it, and
  the trained QCNN that predicts runs unmasked.
- The PSI is semi-honest and not post-quantum
  ([the table above](#what-each-step-protects-in-the-browser-and-in-the-full-deployment)): a bank
  that deviates from it is not detected, and a quantum computer would break the discrete-log
  assumption its OPRF rests on. What would close each is a verifiable OPRF with checks on what each
  bank sends, and a post-quantum OPRF.
- **Federated averaging is not a privacy guarantee, and the secure aggregation hides an individual
  update only from an honest-but-curious aggregator with every client online**
  ([the table above](#what-each-step-protects-in-the-browser-and-in-the-full-deployment)).
  Each pair of clients agrees its mask seed by X25519, as Bonawitz et al. (CCS 2017) specify, so the
  aggregator cannot derive a mask from what it receives. The rest of their protocol is not
  implemented: the double masking and secret-shared keys that let a round survive a dropout, the
  security against a malicious server, and authenticated public keys, without which an aggregator
  that substitutes keys can unmask. Every client runs in one tab, or in one process in Python. Even
  a complete protocol would not bound what the trained model itself leaks: gradient inversion
  against a shared model is an established
  attack. A formal claim would require a stated DP-SGD budget, which this path does not provide, so
  none is made.
- **Not everything the demo displays was computed.** **One** thing cannot be done in a browser at all
  and is labelled `SIMULATED` rather than quietly approximated: QKD has no browser API and cannot have
  one, so every QKD figure comes from an analytic channel model. *This bullet used to list three.* The
  other two were claims about specific libraries, and they no longer describe what ships: TFHE-rs' WASM
  build does not support FHE computations, but the demo does not use TFHE-rs - it runs **node-seal**
  (Microsoft SEAL compiled to WASM) for real CKKS (`web/src/lib/clientFHE.ts`); and JIFF is indeed
  unmaintained and the PSE MPC Framework sunset, but the demo runs **Polytune** authenticated garbling
  (`web/src/lib/clientMPC.ts`; it ran `mpc-framework` + `emp-wasm-engine` + `summon-ts` until Round 65).
  Both are real, measured, and
  described as such in `architecture.md` and `crypto-agility.md` - this bullet contradicted them.
  Caveat that *does* apply to the MPC path: both "parties" share one JS heap, so the party separation
  is notional (stated at `clientMPC.ts:13-15`), and it is a demonstration of the protocol, not two
  mutually distrusting hosts.
  **A SECOND CAVEAT, ADDED ROUND 55 AND RESOLVED IN ROUND 65, AND IT WAS A SUPPLY-CHAIN ONE.** The
  correction above replaced
  "JIFF is unmaintained and the PSE MPC Framework sunset" with the three packages that actually ship -
  and all three of those are now themselves archived upstream. Verified at the GitHub API on
  2026-08-22: `emp-wasm` (archived, last push **2025-06-13**), `mpc-framework` (archived,
  **2025-07-22**) and `summon` (archived, **2025-07-09**), each present under both `voltrevo/*` and the
  `privacy-ethereum/*` org it moved to. So the whole garbled-circuit tier - `emp-wasm-engine ^0.5.4`,
  `mpc-framework ^0.3.0`, `summon-ts ^0.6.2` - rested on code with no upstream maintainer and no
  security-patch path. That does not make the protocol wrong or the measurements false, and nothing
  here is being withdrawn; it means the tier carried an unmaintained-upstream risk. The CKKS tier
  carries a related one: `node-seal` 7.0.0, still its newest release, is aligned with Microsoft SEAL
  4.1.2 (2024), and upstream SEAL is at 4.7.2 (2026-10-28), past 4.4.0's critical security update
  (`docs/development.md`). **The residual risk, with versions (Round 86, review PI-2).** The browser
  runs node-seal 7.0.0 (SEAL 4.1.2). The API image and CI run TenSEAL 0.3.18, the version `requirements.lock` names,
  which bundles SEAL 4.3.3, and a development environment may still hold TenSEAL 0.3.16 (SEAL 4.1.1), which the floor admits. No TenSEAL
  release bundles 4.4.0 or later. node-seal's main branch moved to SEAL 4.7.2 on 2026-10-31 (PR #168;
  its `CHANGES.md` calls the release 7.1.0, and that release also rejects NaN and infinities at CKKS
  encode), and npm's newest node-seal was still 7.0.0 on 2026-11-02. What bounds the exposure until
  then: neither tier deserializes a SEAL object it did not create (each serializes only to measure a
  ciphertext's size), so 4.4.0's hardened deserialization guards an input this project never takes,
  while 4.4.0's fixes to ciphertext evaluation and 4.7.2's sampler fix are missing from every build either tier can run.
  The upgrade waits for the release, and a check makes the release hard to miss:
  `scripts/audit_deps.sh seal`, run daily by the audit workflow, fails once npm's newest node-seal is
  not the locked one. Note the shape: the previous bullet was corrected by naming the maintained
  replacements for two sunset dependencies, and the replacements have since been sunset in turn - a
  correction can go stale in the same way the claim it fixed did. The project's archived-upstream
  query covered submodules and not npm dependencies, which is why this went a year unnoticed. **Replacements were researched in Round 60, and one was adopted in Round 65 (below).** The closest fit
  is `sine-fdn/polytune` with the Garble language (MIT, maintained, WRK17b multi-party authenticated
  garbling against malicious adversaries), at the cost of a Rust and wasm-pack build step and an alpha
  API; mpz would be a downgrade to semi-honest security. The register's Round 60 section carries the
  comparison and what remains unverified. **Round 63 prototyped it** (`prototypes/polytune-argmax/`, not
  part of the build): two Polytune parties complete the demo's argmax in one tab, correctly, about six
  times faster than emp-wasm at 2 classes, while sending about five times the bytes - the cost that
  would dominate on separate hosts, which was not timed. **Round 64 timed it**, between two tabs over
  netem-shaped real TCP: at 2 classes Polytune still finished first on all 18 links (1.09 to 3.33
  times faster), and at 10 classes emp-wasm did (1.47 and 2.75 times); the demo runs 2. **Round 65
  replaced the tier**: `clientMPC.ts` now runs Polytune, built by the web image's Rust stage from
  `wasm/qsh-mpc`, whose `cargo test` runs on every build, the public demo's included, and the three
  archived packages were removed. The residual risk moves with it: Polytune's latest release is a
  pre-release, so the crate is pinned to a commit, and a 10-class model under MPC would be slower than
  it was.
  **Round 68 reviewed the integration against that commit (47dc378).**
  - What held:
    - `mpc()`, Polytune's only public entry, always runs the preprocessing among the parties by OT
      extension. Its trusted-dealer mode is crate-private, so no dealer is involved.
    - Its randomness is `rand`'s thread-local CSPRNG, seeded from the OS, which on wasm32 means
      `crypto.getRandomValues`. Shared randomness comes from a commit-then-reveal coin toss. The fixed
      seeds in its source are in tests.
    - The statistical security parameter is 40.
  - What qualifies the claim: no security advisory is published for Polytune, but two issues are open.
    - #63: the OT extension uses KOS15's original consistency check, whose supporting lemma the
      SoftSpokenOT paper proved false. The attack it gives is impractical. Its practical attack needs
      a case the maintainers believe Polytune does not have. The corrected check exists in libOTe and
      CryProt, but not yet in Polytune.
    - #100: AES in counter mode as a PRG never repeats a block, which a birthday-bound test could
      notice. That is negligible at the output lengths here.
  - What it means here: neither issue changes the demo, whose two parties share one tab. Both qualify
    "malicious security" for a two-host deployment, as do the pre-release status and the absence of
    any audit in the repository.
  **Round 72 read further, at the pinned commit and in Polytune's history.**
  - No independent audit is published. Two pull requests (#34, merged 2025-01-07, and #36) apply
    feedback "from our security audit", but who audited, what was in scope and what was found are not
    disclosed. There is no SECURITY.md and no published advisory, and private vulnerability reporting
    is off.
  - #340 (2025-12-22) removed a fixed seed from the OT-extension sender's generator, a seed that a
    since-yanked pre-release had shipped. The pinned commit postdates it: its only fixed seeds are in
    test code.
  - #63, measured against the literature:
    - It is a gap in the proof at Polytune's parameter of 128, not a practical break. The one
      better-than-negligible attack (SoftSpokenOT, Appendix D) needs a field size divisible by 20.
    - On two hosts either party can play the attacking receiver, because each runs the extension in
      both directions. What is at stake is the other party's global key: its MAC key, and for the
      garbler its free-XOR offset.
  - A further question, ours and not yet answered:
    - The correlation check's challenge is drawn from the pairwise stream agreed by a coin toss at the
      start of preprocessing. That is before the receiver sends its matrix, so the receiver can
      compute the challenge in advance.
    - KOS15 samples the challenge after the matrix, and so did Polytune until commit 1f56564
      (2024-11-06), which moved it to save communication rounds.
    - None of the published analyses covers the earlier ordering. Its consequences are not assessed
      here.
    - A comment asking the maintainers was posted on `sine-fdn/polytune#63` on 2026-10-29 (issuecomment-6499773496), and none had come by 2026-11-09. `scripts/audit_deps.sh polytune` runs daily and fails once that issue, or #100, changes, or once protocol source changes past the pin.
  - So nothing here claims malicious security for two hosts at this commit. The demo, with both
    parties in one tab, is unaffected.
  - What was fixed: the integration, not the library.
    - A failure of either half of the hybrid is now reported as that half's failure, so a protocol
      abort no longer reads as "node-seal did not load".
    - The cost model's split is published only under `_representative` keys.
    - Each run checks the circuit's index against the same argmax in the clear (`mpc_matches_clear`).
    - The crate refuses more than 255 values before compiling.
    - `tests/test_the_polytune_integration_reports_what_happened.py` fails if the Polytune commit
      moves without this review being redone.
  Each result says whether it was measured or simulated, in
  [its provenance badge](#what-a-results-provenance-badge-says).
- **Published metrics are what this repository measured, not what the papers report.** Where a figure
  is carried from the literature instead of reproduced - the breast-cancer PRS AUROC being the clearest
  case, since 1000 Genomes has no case/control status to compute one from - it is labelled as external.
  Where a reproduction falls short of its published baseline, the gap and its cause are printed rather
  than closed by changing the metric.

- **No operator visibility into the public demo.** The client-only build serves no `/metrics` (404 by
  `Caddyfile.client-only`) and logs nothing server-side, so a demo that breaks in a visitor's browser is
  undetectable by the operator. Accepted for now; recorded here so it is a choice rather than an omission.
- **The demo's own attack surface belongs in this document.** Until Round 17 the demo stack ran an API
  container that left `/api/run/*` writable by anonymous visitors while the page displayed "No computation
  is sent to a server". The container was removed and `/api` now 404s, but that exposure was recorded only
  in `architecture.md`; a threat model should carry it.
- **No dependency advisory stays open (Rounds 59 to 65).** Round 59 found three production advisories
  and two development-tooling ones. The two react-router ones (an open redirect through a link target
  built from attacker input, and constructor injection during server-side hydration) were fixed by the
  upgrade to react-router 7.18.4 in Round 61; before that they were unreachable, because the app builds
  no link target from input and renders a declarative `<BrowserRouter>` with no hydration.
  `tests/test_navigation_targets_are_not_built_from_input.py` still fails the day a link target is
  built from a query parameter, because that would be an open redirect under any router version. The
  vite and esbuild advisories, dev-server issues unreachable from the public image, were fixed by vite
  6.4.5 in Round 62. The zod one, reached through the archived `mpc-framework`, left with that package
  in Round 65. `npm audit` now reports none, development tooling included; the register's Round 59 to
  65 sections carry the evidence.
  **Kept at zero since Round 67, rather than measured at zero by hand.** `scripts/audit_deps.sh` runs
  daily in `.github/workflows/audit.yml`, and whenever a manifest, a lockfile or the policy changes.
  It runs npm audit over `web/`, and cargo-deny over the Rust crates compiled into the MPC wasm, which
  had never been audited before. The cargo-deny policy (`wasm/qsh-mpc/deny.toml`) fails on every
  RustSec advisory and on any source other than crates.io and the pinned Polytune repository. It
  ignores one advisory: RUSTSEC-2025-0141, which says bincode 2.0.1 (pulled by Polytune) is
  unmaintained. That advisory reports no vulnerability, and there is no safe upgrade. A vulnerability
  in bincode would be a separate advisory, and it would fail the check.
- **The headline comparison row is simulated.** The QSMPC-FHE row is a state-vector simulation
  (8 qubits by default) of an untrained QOTP + Clifford+T circuit, and the simulated-vs-hardware caveat above
  covers it. The quantum-advantage score once derived from it is retired (no run computes it, in
  either tier), and so is the 0-100 security score; see
  `../references/security-scoring-fact-check.md`.

## What a result's provenance badge says

Every result carries a `ProvenanceBadge` (`web/src/components/ProvenanceBadge.tsx`) with one of
exactly **four** states - `MEASURED (local WebGPU)`, `MEASURED (local WASM)`, `MEASURED (plain JS)`
and `SIMULATED` - together with the runtime evidence for that claim (`adapter.info`,
`crossOriginIsolated`, `ort.env.wasm.numThreads`, bytes transferred against bytes served from cache).
A result labelled `MEASURED` that produced no compute or network activity is treated as a defect.
`MEASURED (plain JS)` exists for the bundled distilled model, a hand-written loop over plain JS
numbers with no wasm module and no GPU, which reported `measured-wasm` until that state was added
(Round 21): a claim wrong in the flattering direction. There is **no `REMOTE` state**. It was removed
from the type, not merely left unset, when the hosted-GPU tier was declined (Round 24d): nothing in
`web/src` calls the ZeroGPU Space, that Space returns HTTP 401, and a type member no code path can
produce is the same defect as a metric with no reader. The decision against a remote tier, and what
would reopen it, are recorded in [`findings.md`](findings.md#concerns--recommendations).

## Genomics: this system protects the computation, and the computation is not where the leak is

Recorded 2026-08-10 after a literature check, because it is the sharpest gap in the project's threat
model and it is a gap of FRAMING rather than of implementation.

The genomics use case computes pooled allele counts under secure summation, so no party sees another
site's genotypes. That is a real input-confidentiality property and it is implemented. It is also
not the threat the genomics-privacy literature is concerned with.

**Pooled allele counts are the canonical genomic privacy failure.** Homer et al., PLoS Genetics
4(8):e1000167 (2008) - the paper that caused NIH to withdraw public aggregate GWAS data - states
that composite statistics across cohorts, such as allele frequency or genotype counts, do not mask
identity within genome-wide association studies. It detects an individual contributing under 0.1% of
a mixture from roughly 50,000 SNPs.

**Releasing frequencies is worse than releasing booleans.** A 2025 beacon-reconstruction result
(DOI 10.1093/bioinformatics/btaf273) reports genome reconstruction at **70% F1 using allele
frequencies against 53.4% using binary yes/no responses** - so publishing the counts this pipeline
computes raises reconstruction risk by about 17 points over publishing mere presence.

**A single released score can be enough, for a small score.** A Feb 2026 preprint (Nikitin and
Gursoy, DOI 10.64898/2026.02.16.706191) reconstructs genotypes from polygenic risk scores by solving
a subset-sum problem: of 4,723 PGS Catalog models, at least 447 would be vulnerable if patient-level
scores were shared, and a typical 27-locus PRS uniquely identifies 95% of individuals in a
450,000-participant cohort. The recovery covers scores of up to about 50 variants and the
vulnerability assessment about 80, so PGS000072's 183 scored variants sit outside the recovery regime,
though a high-precision individual score still carries uniqueness and linkage risk. Non-European
ancestries are at greater risk, because GWAS panels are less diverse.

### What this means here, stated precisely

Encryption of the computation is **orthogonal** to this threat. The secure summation is not
weakened by any of the above; it simply addresses a different adversary. The exposure is the
released aggregate, and the mitigation for that is **output perturbation** - differential privacy on
the published counts - which this project does **not** implement.

Nor does anyone else in deployed crypto-genomics: SF-GWAS (Nature Genetics, Feb 2025), PP-GWAS,
SECRET-GWAS, privateQTL and the CKKS polygenic-score system HEPRS (Cell Reports Methods, Jan 2026)
all decrypt and release plaintext statistics with no output noise. SF-GWAS argues explicitly that DP
is impractical at GWAS scale because releasing statistics for many variants would require
substantial noise; that view is now contested, since GOPHER (Nandi, Neel and Cho, arXiv:2512.04225)
releases complete genome-wide statistics at $`\epsilon = 3`$, reporting r = 0.85 to 0.86 in its own
evaluation. So this is a field-wide gap rather than an oversight peculiar to this project -
which is a reason to state it, not a reason to be quiet about it.

### The honest claim, and the one to avoid

**Defensible:** "no party learns another site's genotypes during the computation."

**Not defensible:** any implication that pooling under encryption makes the *released* counts safe to
publish. It does not, and the literature is unambiguous.

Closing this properly means adding calibrated noise to the released aggregate with a stated $`\epsilon`$,
and reporting the utility cost - a UK Biobank study (DOI 10.1093/jamia/ocaf090, Jun 2025) found odds
ratios destabilising below $`\epsilon \approx \ln 2`$, with some reversing sign. That is a design decision
with a real accuracy price, and it is recorded here as open rather than silently absent.
