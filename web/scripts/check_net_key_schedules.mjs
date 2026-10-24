// Every stored official vector for the network key schedules, run through the REAL browser modules.
//
// WHAT THIS COVERS
//
//   web/src/lib/netkeys/mlkemKat.ts     NIST ACVP ML-KEM-768 keyGen / encapsulation / decapsulation
//   web/src/lib/netkeys/noiseIKpsk2.ts  cacophony's Noise_IKpsk2_25519_ChaChaPoly_BLAKE2s vector
//   web/src/lib/netkeys/ikev2.ts        NIST CAVP SP 800-135 IKEv2 KDF cases + RFC 5903 section 8.1
//   web/src/lib/netkeys/wireguard.ts    NO handshake vector exists - structural checks, plus the one
//                                       step boringtun publishes (Ci, Hi after the Identifier)
//
// Sources, commits, retrieval dates and SHA-256 of every upstream file: references/net-key-schedules.md.
//
// WHY IT COMPARES BYTES ITSELF AND ALSO CHECKS THE MODULES' OWN RUNNERS
//
// The modules carry in-tab runners (`runMlKemAcvp`, `runNoiseVectors`, `runIkev2Cavp`) because the page
// shows their verdicts. A runner is code too: one that compared a value with itself would report every
// case green forever. So this script drives the module PRIMITIVES with the stored inputs and compares
// the outputs to the stored answers on its own, then requires each runner to reach the same verdict and
// to report as "computed" the digest of what the primitive actually computed. Same approach as
// check_hybrid_combiner.mjs: bundle the real module graph with esbuild and import it, never a
// re-implementation - a copy would agree with itself while the browser drifted.
//
// AND THAT IT CAN FAIL
//
// `--corrupt=<family>` (or QSH_NETKEYS_CORRUPT=<family>) flips one byte of one EXPECTED value in memory
// before anything runs; families are mlkem, noise, ikev2, rfc5903, wginit. The check must then exit
// non-zero and name the case. tests/test_net_key_schedules.py runs every family. A vector check nobody
// has seen fail is indistinguishable from one that cannot.
//
// Every section runs inside `section()`, so a module that THROWS is reported as a named failure rather
// than a stack trace: seeding a wrong mac1 key first surfaced here as a bare Node crash.
//
// WHAT IT DOES NOT CLAIM
//
// The WireGuard framing and the RFC 9370 / RFC 8784 layers have no official vectors. Their checks below
// are properties (sizes from the whitepaper's tables, both peers agreeing, which keys a layer may
// change), and the OK line says "per spec only" for them so a log reader cannot mistake them for KATs.
//
// Run: node web/scripts/check_net_key_schedules.mjs   (npm run check:netkeys)

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const LIB = join(WEB, "src/lib/netkeys");
const VECTORS = join(LIB, "vectors");
const failures = [];
const check = (name, ok, detail) => { if (!ok) failures.push(`${name}: ${detail}`); };
const section = (name, body) => {
  try { body(); } catch (err) { failures.push(`${name}: threw ${err?.message ?? err}`); }
};

const hex = (u8) => Buffer.from(u8).toString("hex");
const unhex = (s) => Uint8Array.from(Buffer.from(s, "hex"));
const sha256 = (u8) => createHash("sha256").update(u8).digest("hex");
const same = (u8, expectedHex) => hex(u8) === expectedHex.toLowerCase();
const short = (s) => `${s.slice(0, 16)}...`;
const readJson = (name) => JSON.parse(readFileSync(join(VECTORS, name), "utf8"));

/** A mismatch a CI-log reader can act on: short values in full, long ones by digest, and WHERE they part. */
function describe(got, wantHex) {
  const want = unhex(wantHex);
  let at = 0;
  while (at < Math.min(got.length, want.length) && got[at] === want[at]) at += 1;
  const where = got.length !== want.length ? `lengths ${got.length} vs ${want.length}` : `first difference at byte ${at}`;
  return got.length <= 48
    ? `computed ${hex(got)} vs expected ${hex(want)} (${where})`
    : `computed sha256 ${short(sha256(got))} vs expected ${short(sha256(want))} (${where})`;
}

// ---------------------------------------------------------------------------------------------
// The negative control: one byte of one expected value, flipped before anything reads it.
// ---------------------------------------------------------------------------------------------
const CORRUPTIONS = {
  // tcId 88 is a "modified ciphertext" case: its K is the implicit-rejection key, the branch a
  // valid-only comparison never reaches - so this also proves that branch's comparison is live.
  mlkem: { file: "mlkem768_acvp.json", path: ["decapsulation", "cases", 2, "k"], label: "ML-KEM decapsulation tcId 88" },
  noise: { file: "noise_ikpsk2.json", path: ["vectors", 0, "messages", 4, "ciphertext"], label: "Noise message 4" },
  ikev2: { file: "ikev2_cavp.json", path: ["cases", 0, "DKM"], label: "HMAC-SHA-384/g^ir1024/N64/COUNT0" },
  rfc5903: { file: "ikev2_cavp.json", path: ["ecp256Rfc5903", "cases", 0, "girx"], label: "RFC5903-8.1" },
  wginit: { file: "wireguard_initial_state.json", path: ["initialHash", "hex"], label: "WireGuard Hi = Hash(Ci || Identifier)" },
};
const corruptArg = process.argv.find((a) => a.startsWith("--corrupt="));
const corrupt = corruptArg ? corruptArg.slice("--corrupt=".length) : (process.env.QSH_NETKEYS_CORRUPT || "");
if (corrupt && !CORRUPTIONS[corrupt]) {
  console.error(`check:netkeys: unknown corruption family "${corrupt}"; expected one of ${Object.keys(CORRUPTIONS).join(", ")}`);
  process.exit(2);
}

const files = {
  "mlkem768_acvp.json": readJson("mlkem768_acvp.json"),
  "noise_ikpsk2.json": readJson("noise_ikpsk2.json"),
  "ikev2_cavp.json": readJson("ikev2_cavp.json"),
  "wireguard_initial_state.json": readJson("wireguard_initial_state.json"),
};

if (corrupt) {
  const { file, path, label } = CORRUPTIONS[corrupt];
  let parent = files[file];
  for (const key of path.slice(0, -1)) parent = parent[key];
  const leaf = path[path.length - 1];
  const bytes = unhex(parent[leaf]);
  const at = bytes.length >> 1;
  bytes[at] ^= 0x01;
  parent[leaf] = hex(bytes);
  console.error(`check:netkeys NEGATIVE CONTROL: flipped byte ${at} of ${file}:${path.join(".")} (${label}); this run must FAIL`);
}

// ---------------------------------------------------------------------------------------------
// Bundle the four modules exactly as the browser gets them.
// ---------------------------------------------------------------------------------------------
function load() {
  const dir = mkdtempSync(join(tmpdir(), "qsh-netkeys-"));
  const entry = join(dir, "entry.ts");
  writeFileSync(entry, [
    `export * as mlkem from ${JSON.stringify(join(LIB, "mlkemKat"))};`,
    `export * as noise from ${JSON.stringify(join(LIB, "noiseIKpsk2"))};`,
    `export * as wg from ${JSON.stringify(join(LIB, "wireguard"))};`,
    `export * as ike from ${JSON.stringify(join(LIB, "ikev2"))};`,
  ].join("\n") + "\n");
  const out = join(dir, "bundle.mjs");
  execFileSync("npx", ["esbuild", entry, "--bundle", "--format=esm", "--platform=node",
                       `--outfile=${out}`, "--log-level=error"], { cwd: WEB, stdio: "pipe" });
  return { out, dir };
}

const { out, dir } = load();
const { mlkem, noise, wg, ike } = await import(`file://${out}`);

// ---------------------------------------------------------------------------------------------
// 0. Provenance is recorded, and recorded where the bytes came from.
// ---------------------------------------------------------------------------------------------
const isSha = (s) => typeof s === "string" && /^[0-9a-f]{64}$/.test(s);
const isCommit = (s) => typeof s === "string" && /^[0-9a-f]{40}$/.test(s);
section("provenance", () => {
  const f = files["mlkem768_acvp.json"];
  check("mlkem provenance", isCommit(f.source.commit) && f.source.files.length >= 4
        && f.source.files.every((x) => isSha(x.sha256) && x.url.startsWith(`https://raw.githubusercontent.com/usnistgov/ACVP-Server/${f.source.commit}/`)),
        "every ACVP file needs a commit-pinned raw URL and a SHA-256");
  for (const fn of ["keyGen", "encapsulation", "decapsulation"]) {
    const block = f[fn];
    check(`mlkem ${fn} tcIds`, block.cases.length >= 2 && block.cases.length <= 4
          && JSON.stringify(block.tcIds) === JSON.stringify(block.cases.map((c) => c.tcId)),
          `tcIds ${JSON.stringify(block.tcIds)} do not list the ${block.cases.length} stored cases`);
  }
  const n = files["noise_ikpsk2.json"];
  check("noise provenance", isCommit(n.source.commit) && isSha(n.source.sha256) && n.source.url.includes(n.source.commit),
        "the cacophony URL must carry the pinned commit and the file's SHA-256");
  const k = files["ikev2_cavp.json"];
  check("ikev2 provenance", isSha(k.source.sha256) && isSha(k.source.member.sha256) && k.source.url.startsWith("https://csrc.nist.gov/")
        && isSha(k.ecp256Rfc5903.source.sha256) && k.ecp256Rfc5903.source.url.startsWith("https://www.rfc-editor.org/"),
        "CAVP zip, its .rsp member and RFC 5903 each need a URL and a SHA-256");
  const w = files["wireguard_initial_state.json"];
  check("wireguard initial-state provenance", isCommit(w.source.commit) && isSha(w.source.sha256)
        && w.source.url.includes(w.source.commit) && w.authoritative === false,
        "boringtun's constants need a commit-pinned URL, a SHA-256, and must stay marked non-authoritative");
});

// ---------------------------------------------------------------------------------------------
// 1. ML-KEM-768 against NIST ACVP.
// ---------------------------------------------------------------------------------------------
const mlkemVerdicts = new Map();
let mlkemCount = 0;
section("ML-KEM", () => {
  const f = files["mlkem768_acvp.json"];
  const S = mlkem.MLKEM768_SIZES;
  for (const c of f.keyGen.cases) {
    const { ek, dk } = mlkem.mlkemKeyGen(unhex(c.d), unhex(c.z));
    const ok = same(ek, c.ek) && same(dk, c.dk) && ek.length === S.ek && dk.length === S.dk;
    check(`ML-KEM keyGen tcId ${c.tcId}`, ok, `ek: ${describe(ek, c.ek)}; dk: ${describe(dk, c.dk)}`);
    // d || z is noble's packing, not FIPS 203's; the swapped order must NOT reproduce the answer.
    const swapped = mlkem.mlkemKeyGen(unhex(c.z), unhex(c.d));
    check(`ML-KEM keyGen tcId ${c.tcId} order`, !same(swapped.ek, c.ek), "z || d gave the same ek; the seed order is not being tested");
    mlkemVerdicts.set(`keyGen/${c.tcId}`, { ok, digests: { ek: sha256(ek), dk: sha256(dk) } });
    mlkemCount += 1;
  }
  for (const c of f.encapsulation.cases) {
    const r = mlkem.mlkemEncaps(unhex(c.ek), unhex(c.m));
    const ok = same(r.c, c.c) && same(r.k, c.k) && r.c.length === S.c && r.k.length === S.k;
    check(`ML-KEM encapsulation tcId ${c.tcId}`, ok, `c: ${describe(r.c, c.c)}; K: ${describe(r.k, c.k)}`);
    mlkemVerdicts.set(`encapsulation/${c.tcId}`, { ok, digests: { c: sha256(r.c), k: sha256(r.k) } });
    mlkemCount += 1;
  }
  const reasons = new Set();
  for (const c of f.decapsulation.cases) {
    const k = mlkem.mlkemDecaps(unhex(c.dk), unhex(c.c));
    const ok = same(k, c.k);
    check(`ML-KEM decapsulation tcId ${c.tcId} (${c.reason})`, ok, `K: ${describe(k, c.k)}`);
    mlkemVerdicts.set(`decapsulation/${c.tcId}`, { ok, digests: { k: sha256(k) } });
    reasons.add(c.reason);
    mlkemCount += 1;
  }
  check("ML-KEM decapsulation coverage", reasons.has("valid decapsulation") && reasons.has("modified ciphertext"),
        `stored reasons ${JSON.stringify([...reasons])}; both branches of Decaps must be present`);

  // The in-tab runner must reach the same verdicts and report the digests of what was COMPUTED.
  const run = mlkem.runMlKemAcvp(f);
  check("runMlKemAcvp count", run.total === mlkemCount, `runner ran ${run.total} cases, the file holds ${mlkemCount}`);
  for (const rc of run.cases) {
    const mine = mlkemVerdicts.get(`${rc.function}/${rc.tcId}`);
    check(`runMlKemAcvp ${rc.function} ${rc.tcId}`, mine && mine.ok === rc.passed,
          `runner says ${rc.passed}, this script says ${mine?.ok}`);
    for (const o of rc.outputs) {
      check(`runMlKemAcvp ${rc.function} ${rc.tcId} ${o.name} digest`, mine && mine.digests[o.name] === o.computedSha256,
            "the runner's computedSha256 is not the digest of the computed output");
    }
  }
  check("runMlKemAcvp allPassed", run.allPassed === [...mlkemVerdicts.values()].every((v) => v.ok),
        `runner allPassed=${run.allPassed} disagrees with this script`);
});

// ---------------------------------------------------------------------------------------------
// 2. Noise_IKpsk2_25519_ChaChaPoly_BLAKE2s against cacophony's vector.
// ---------------------------------------------------------------------------------------------
let noiseMessages = 0;
let noiseOk = true;
section("Noise", () => {
  const f = files["noise_ikpsk2.json"];
  const vs = f.vectors.filter((v) => v.protocol_name === noise.NOISE_PROTOCOL_NAME);
  check("noise vector present", vs.length >= 1, "no vector for Noise_IKpsk2_25519_ChaChaPoly_BLAKE2s");
  for (const v of vs) {
    const fail = (name, detail) => { noiseOk = false; failures.push(`Noise ${name}: ${detail}`); };
    try {
      const init = new noise.IKpsk2Handshake({
        initiator: true, prologue: unhex(v.init_prologue), s: noise.keyPairFromPrivate(unhex(v.init_static)),
        e: noise.keyPairFromPrivate(unhex(v.init_ephemeral)), rs: unhex(v.init_remote_static), psk: unhex(v.init_psks[0]),
      });
      const resp = new noise.IKpsk2Handshake({
        initiator: false, prologue: unhex(v.resp_prologue), s: noise.keyPairFromPrivate(unhex(v.resp_static)),
        e: noise.keyPairFromPrivate(unhex(v.resp_ephemeral)), psk: unhex(v.resp_psks[0]),
      });
      const w0 = init.writeMessage(unhex(v.messages[0].payload));
      if (!same(w0.message, v.messages[0].ciphertext)) fail("message 0", `initiator: ${describe(w0.message, v.messages[0].ciphertext)}`);
      const r0 = resp.readMessage(unhex(v.messages[0].ciphertext));
      if (!same(r0.payload, v.messages[0].payload)) fail("message 0 payload", "responder did not recover the payload");
      const w1 = resp.writeMessage(unhex(v.messages[1].payload));
      if (!same(w1.message, v.messages[1].ciphertext)) fail("message 1", `responder: ${describe(w1.message, v.messages[1].ciphertext)}`);
      const r1 = init.readMessage(unhex(v.messages[1].ciphertext));
      if (!same(r1.payload, v.messages[1].payload)) fail("message 1 payload", "initiator did not recover the payload");
      noiseMessages += 2;
      for (const [who, hs] of [["initiator", init], ["responder", resp]]) {
        if (!same(hs.getHandshakeHash(), v.handshake_hash)) fail(`handshake hash (${who})`, describe(hs.getHandshakeHash(), v.handshake_hash));
      }
      const iT = r1.transport, rT = w1.transport;
      let fromInit = 0, fromResp = 0;
      for (let idx = 2; idx < v.messages.length; idx += 1) {
        const m = v.messages[idx];
        const initiatorSends = idx % 2 === 0;
        const [send, recv] = initiatorSends ? [iT.send, rT.recv] : [rT.send, iT.recv];
        const ct = send.encryptWithAd(new Uint8Array(0), unhex(m.payload));
        if (!same(ct, m.ciphertext)) fail(`message ${idx}`, `${initiatorSends ? "initiator" : "responder"}: ${describe(ct, m.ciphertext)}`);
        if (!same(recv.decryptWithAd(new Uint8Array(0), unhex(m.ciphertext)), m.payload)) fail(`message ${idx} payload`, "receiver did not recover the payload");
        if (initiatorSends) fromInit += 1; else fromResp += 1;
        noiseMessages += 1;
      }
      if (!(fromInit >= 1 && fromResp >= 1)) fail("transport coverage", "the vector must exercise both Split() directions");
    } catch (err) {
      fail("vector", `threw ${err?.message ?? err} - a decrypt failure here IS a mismatch`);
    }
  }
  const run = noise.runNoiseVectors(f);
  check("runNoiseVectors verdict", run.total === vs.length && run.allPassed === noiseOk,
        `runner allPassed=${run.allPassed} over ${run.total}, this script says ${noiseOk}`);
});

// ---------------------------------------------------------------------------------------------
// 3. IKEv2 KDF against NIST CAVP SP 800-135, and ECP-256 against RFC 5903 section 8.1.
// ---------------------------------------------------------------------------------------------
let cavpCases = 0;
let ikeOk = true;
section("IKEv2 CAVP", () => {
  const f = files["ikev2_cavp.json"];
  const prfs = new Set();
  for (const c of f.cases) {
    const alg = c.prf;
    prfs.add(alg);
    const [ni, nr, gir, girNew, spii, spir] = [c.Ni, c.Nr, c["g^ir"], c["g^ir (new)"], c.SPIi, c.SPIr].map(unhex);
    const skeyseed = ike.computeSkeyseed(alg, ni, nr, gir);
    const dkm = ike.prfPlus(alg, skeyseed, Buffer.concat([ni, nr, spii, spir]), c.section["DKM length"] / 8);
    const skd = dkm.slice(0, ike.prfLength(alg));
    const childLen = c.section["Child SA DKM length"] / 8;
    const results = [
      ["SKEYSEED", skeyseed, c.SKEYSEED],
      ["DKM", dkm, c.DKM],
      ["DKM(Child SA)", ike.childSaKeymat(alg, skd, ni, nr, childLen), c["DKM(Child SA)"]],
      ["DKM(Child SA D-H)", ike.childSaKeymat(alg, skd, ni, nr, childLen, girNew), c["DKM(Child SA D-H)"]],
      ["SKEYSEED(Rekey)", ike.rekeySkeyseed(alg, skd, girNew, ni, nr), c["SKEYSEED(Rekey)"]],
    ];
    for (const [name, got, want] of results) {
      const ok = same(got, want);
      if (!ok) ikeOk = false;
      check(`IKEv2 ${c.id} ${name}`, ok, describe(got, want));
    }
    cavpCases += 1;
  }
  check("IKEv2 PRF coverage", ["HMAC-SHA-256", "HMAC-SHA-384", "HMAC-SHA-512"].every((p) => prfs.has(p)),
        `stored PRFs ${JSON.stringify([...prfs])}; every PRF ikev2.ts accepts needs a case`);
  for (const c of f.ecp256Rfc5903.cases) {
    const i = unhex(c.i), r = unhex(c.r);
    const pairs = [
      ["g^i", ike.ecp256KeData(i), c.gix + c.giy],
      ["g^r", ike.ecp256KeData(r), c.grx + c.gry],
      ["g^ir initiator", ike.ecp256SharedSecret(i, unhex(c.grx + c.gry)), c.girx],
      ["g^ir responder", ike.ecp256SharedSecret(r, unhex(c.gix + c.giy)), c.girx],
    ];
    for (const [name, got, want] of pairs) {
      const ok = same(got, want);
      if (!ok) ikeOk = false;
      check(`RFC 5903 ${c.id} ${name}`, ok, describe(got, want));
    }
  }
  const run = ike.runIkev2Cavp(f);
  check("runIkev2Cavp verdict", run.cases.length === cavpCases && run.allPassed === ikeOk,
        `runner allPassed=${run.allPassed}, this script says ${ikeOk}`);
  check("runIkev2Cavp says what it does not validate",
        run.notValidated.some((s) => s.includes("RFC 9370") && s.includes("no official test vectors"))
        && run.notValidated.some((s) => s.includes("RFC 8784") && s.includes("no official test vectors")),
        "the RFC 9370 and RFC 8784 layers must be reported as unvalidated");
});

// ---------------------------------------------------------------------------------------------
// 4. WireGuard: properties, plus one cross-checked step. No public fixed handshake vector from an
//    authoritative implementation exists to compare against (see wireguard.ts).
// ---------------------------------------------------------------------------------------------
section("WireGuard", () => {
  const psk = Uint8Array.from({ length: 32 }, (_, i) => i);
  const a = wg.wireguardHandshake({ seed: "check:netkeys", psk });
  const again = wg.wireguardHandshake({ seed: "check:netkeys", psk: hex(psk) });
  const other = wg.wireguardHandshake({ seed: "check:netkeys/other", psk });
  check("WG labelled", a.validatedAgainst === "Noise vectors only; WireGuard framing per the whitepaper",
        `validatedAgainst is ${JSON.stringify(a.validatedAgainst)}`);
  check("WG sizes", a.messageSizes.initiation === 148 && a.messageSizes.response === 92 && a.sizesMatchWhitepaper,
        `initiation ${a.messageSizes.initiation}, response ${a.messageSizes.response}; whitepaper 5.4.2/5.4.3 give 148 and 92`);
  check("WG agreement", a.transportKeysAgree && a.handshakeHashesAgree && a.initiatorStaticRecovered && a.timestampRecovered
        && a.mac1Valid.initiation && a.mac1Valid.response, JSON.stringify({ k: a.transportKeysAgree, h: a.handshakeHashesAgree, mac1: a.mac1Valid }));
  check("WG noise core", a.noiseCore.passed, "wireguardHandshake ran on a Noise core that failed its vector");
  const pc = a.pskChange;
  check("WG psk change", pc.sizesUnchanged && pc.transportKeysChanged && pc.transportKeysAgree && pc.responseChanged,
        JSON.stringify(pc));
  check("WG psk2 placement", pc.initiationUnchanged, "the initiation changed with the PSK; psk2 mixes it only into message 2");
  check("WG psk mismatch", a.pskMismatch.rejected && !/mac1/.test(a.pskMismatch.reason),
        `mismatched PSKs gave ${JSON.stringify(a.pskMismatch)}; the initiator must reject msg.empty's tag, not mac1`);
  check("WG deterministic", JSON.stringify(a.fingerprints) === JSON.stringify(again.fingerprints),
        "the same seed and PSK (bytes vs hex) gave different keys");
  check("WG seed matters", a.fingerprints.initiatorSend !== other.fingerprints.initiatorSend, "a different seed gave the same keys");
  check("WG default psk", wg.wireguardHandshake({ seed: "x" }).psk.isDefaultZero, "no PSK given must mean Q = 0^32");

  // Framing, read off the bytes against the whitepaper's field tables.
  const k = (label) => noise.keyPairFromPrivate(Uint8Array.from(createHash("sha256").update(label).digest()));
  const [iS, rS, iE, rE] = ["iS", "rS", "iE", "rE"].map(k);
  const ts = wg.tai64n(Date.UTC(2026, 0, 1));
  const { message: init, handshake } = wg.createInitiation({ senderIndex: 0x01020304, s: iS, e: iE, responderStatic: rS.publicKey, psk, timestamp: ts });
  check("WG initiation header", init[0] === 1 && init[1] === 0 && init[2] === 0 && init[3] === 0
        && hex(init.slice(4, 8)) === "04030201" && hex(init.slice(8, 40)) === hex(iE.publicKey)
        && init.slice(132).every((b) => b === 0),
        "type, reserved, little-endian sender index, cleartext ephemeral or zero mac2 out of place");
  const tampered = init.slice();
  tampered[20] ^= 0x01;
  let mac1Rejects = false;
  try { wg.consumeInitiation(tampered, { s: rS, e: rE, psk }); } catch (err) { mac1Rejects = /mac1/.test(err.message); }
  check("WG mac1 covers msg_alpha", mac1Rejects, "a flipped ephemeral byte was not rejected by mac1");
  const consumed = wg.consumeInitiation(init, { s: rS, e: rE, psk });
  const { message: resp } = wg.createResponse(consumed, 0x0a0b0c0d);
  check("WG response header", resp[0] === 2 && hex(resp.slice(4, 8)) === "0d0c0b0a" && hex(resp.slice(8, 12)) === "04030201"
        && resp.slice(76).every((b) => b === 0), "type, sender/receiver indices or zero mac2 out of place");
  check("WG response consumed", wg.consumeResponse(resp, handshake, iS.publicKey).receiverIndex === 0x01020304, "receiver index lost");

  // The one WireGuard-specific step with a published answer: each handshake OBJECT's state after the
  // prologue against boringtun's precomputed constants. Without this an empty prologue passed every
  // check above (measured), because two copies of the same code agree on any prologue they share.
  const ws = files["wireguard_initial_state.json"];
  for (const [who, hs] of [["initiator", handshake], ["responder", consumed.handshake]]) {
    const ckOk = same(hs.afterPrologue.ck, ws.initialChainingKey.hex);
    const hOk = same(hs.afterPrologue.h, ws.initialHash.hex);
    check(`WG initial state (${who}) Ci = Hash(Construction)`, ckOk, describe(hs.afterPrologue.ck, ws.initialChainingKey.hex));
    check(`WG initial state (${who}) Hi = Hash(Ci || Identifier)`, hOk, describe(hs.afterPrologue.h, ws.initialHash.hex));
    const viaModule = wg.wgInitialStateCheck(hs, ws);
    check(`wgInitialStateCheck (${who})`, viaModule.chainingKeyMatches === ckOk && viaModule.hashMatches === hOk,
          "the module's verdict disagrees with this script");
  }
  check("WG result carries the cross-check", a.initialStateCrossCheck.bothPeers, JSON.stringify(a.initialStateCrossCheck));
});

// ---------------------------------------------------------------------------------------------
// 5. IKEv2 demo flow: the RFC 9370 and RFC 8784 layers. Properties of the RFC text, NOT vectors.
// ---------------------------------------------------------------------------------------------
section("IKEv2 demo", () => {
  const d = ike.ikev2KeySchedule({ seed: "check:netkeys" });
  const noPpk = ike.ikev2KeySchedule({ seed: "check:netkeys", ppk: null });
  const otherPpk = ike.ikev2KeySchedule({ seed: "check:netkeys", ppk: "ab".repeat(32) });
  check("IKEv2 exchange", d.exchange.girAgrees && d.exchange.mlkemSecretAgrees && d.exchange.keiOctets === 64
        && d.exchange.kei1Octets === 1184 && d.exchange.ker1Octets === 1088, JSON.stringify(d.exchange));
  check("IKEv2 stages", d.stages.length === 3 && d.stages.every((s) => s.peersAgree), "a stage where the two peers disagree");
  check("IKEv2 lengths", Object.entries({ SK_d: 48, SK_ai: 0, SK_ar: 0, SK_ei: 36, SK_er: 36, SK_pi: 48, SK_pr: 48 })
        .every(([n, l]) => d.stages.every((s) => s.keys[n].length === l)), "SK_* lengths are not the suite's");
  check("IKEv2 validated flags", d.stages[0].validated && !d.stages[1].validated && !d.stages[2].validated,
        "only the RFC 7296 / RFC 5903 stage has official vectors");
  check("IKEv2 RFC 9370 re-derives everything", d.properties.additionalKeyExchangeChangesAll,
        `ML-KEM round changed only ${JSON.stringify(d.stages[1].changedFromPrevious)}`);
  check("IKEv2 RFC 8784 touches exactly d/pi/pr", d.properties.ppkChangesOnlyDPiPr,
        `PPK changed ${JSON.stringify(d.stages[2].changedFromPrevious)}`);
  check("IKEv2 PPK off", noPpk.stages.length === 2 && noPpk.ppk.used === false
        && JSON.stringify(noPpk.stages[1].keys) === JSON.stringify(d.stages[1].keys), "ppk: null must remove only the PPK stage");
  check("IKEv2 PPK matters", otherPpk.stages[2].keys.SK_d.fingerprint !== d.stages[2].keys.SK_d.fingerprint
        && otherPpk.stages[2].keys.SK_ei.fingerprint === d.stages[2].keys.SK_ei.fingerprint, "a different PPK must change SK_d and leave SK_ei");
  check("IKEv2 demo labels", d.notValidated.length >= 2 && d.validation.cavpAllPassed && d.validation.rfc5903Passed,
        "the demo must carry its not-validated statements and the CAVP verdict");
});

rmSync(dir, { recursive: true, force: true });

if (failures.length) {
  console.error("check:netkeys FAILED");
  for (const f of failures) console.error("  - " + f);
  process.exit(1);
}
console.log(
  `check:netkeys OK (ML-KEM-768 ACVP ${mlkemCount} cases; Noise IKpsk2 cacophony ${noiseMessages} messages + handshake hash;`
  + ` IKEv2 CAVP ${cavpCases} cases x 5 values + RFC 5903 8.1; WireGuard Ci/Hi vs boringtun constants;`
  + " WireGuard framing and RFC 9370 / RFC 8784 layers: per spec only, no official vectors)");
