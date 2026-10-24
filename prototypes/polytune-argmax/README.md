# Polytune argmax prototype

A measurement prototype, not part of the demo. Nothing under `web/` imports it, the web image does not
build it, and `tests/test_prototypes_stay_out_of_the_web_build.py` keeps it that way.

## Why it exists

The demo's garbled-circuit tier (`web/src/lib/clientMPC.ts`) runs on `mpc-framework`, `emp-wasm-engine`
and `summon-ts`, and all three are archived upstream; `mpc-framework` also pulls the zod release that
carries GHSA-mrhx-c7xc-m3w5, with no fix coming. Round 60 found `sine-fdn/polytune` with the Garble
language to be the closest maintained replacement: MIT, WRK17b multi-party authenticated garbling,
secure against malicious adversaries controlling all but one party. This prototype answers the
question that research left open - can two Polytune parties run in one browser tab, and at what cost -
before anything is replaced.

It computes exactly what `secureArgmax` computes: the argmax of N logits that are additively shared
between two parties (`q_k = a_k + b_k - 65536`), with only the index revealed. The shares are built the
same way (fixed-point scale 64, offset 8, clamp 1024). One difference: Garble's integers are 32-bit,
where the summon circuit uses a 24-bit width, so this side does somewhat more arithmetic.

## Build (official Rust container)

Requires Docker. Nothing is installed on the host.

```bash
./build.sh
```

`build.sh` builds a builder image from `Dockerfile` - the official `rust:1.99.1-bookworm` image, pinned
by digest, plus the `wasm32-unknown-unknown` target, `wasm-pack` 0.15.0 and `wasm-bindgen-cli`
0.2.129 - then runs the native test and `wasm-pack build --target web --release` inside it, as the
calling user. Outputs go to `target/` and `pkg/`, and Cargo's download cache to `.cargo-home/`; all
three are gitignored. The builder image is about 2.4 GB; a cold build of the crate takes under a minute
on the project's development machine, and `./build.sh test` runs the native test alone.

Why a container: Polytune requires Rust 1.89 or later, and the host's system Rust is 1.88. The host
route - `sudo dnf install rust-std-static-wasm32-unknown-unknown` plus `cargo install wasm-pack` - works
for crates whose minimum Rust version the host compiler meets, but not for this one.

## Run the measurement

```bash
python3 -m http.server 4319 --bind 127.0.0.1
```

Open `http://127.0.0.1:4319/index.html`, choose the class count and the number of runs, and press Run.
Each run builds fresh shares, runs both parties in the tab over an in-memory channel, and checks the
index against the plaintext argmax. The runs are left in `window.__runs`.

## Results (2026-10-24, Chrome 152 on the development machine)

| implementation | classes | correct | protocol time | bytes exchanged | messages |
|---|---|---|---|---|---|
| Polytune (this prototype) | 2 | 6 of 6 | 117.5 to 120.1 ms warm, 150.5 ms first | 1,018,898 | 69 |
| emp-wasm (the demo today) | 2 | 3 of 3 | 690 to 730 ms in the browser | 214,861 | 43 |
| Polytune | 10 | 3 of 3 | 589 to 603 ms | 7,184,705 | 195 |
| emp-wasm | 10 | 3 of 3 | 745 to 767 ms (Node 22) | 1,241,896 | 55 |

The emp-wasm browser times come from the demo's own medical pipeline in the same browser; its byte and
message counts, and the 10-class times, come from replicating `secureArgmax` in Node 22 against the
same installed packages (the byte count matches the 209.83 KiB the pipeline reports). Polytune's
circuit has 400 AND gates at 2 classes and 2,997 at 10.

Read together: in one tab, where the two parties exchange messages through memory, Polytune is about
six times faster at 2 classes and about 1.3 times faster at 10. It sends 4.7 to 5.8 times as many bytes
and 1.6 to 3.5 times as many messages, and both of those cost real time once the parties sit on
separate hosts. The wasm package is 458 KB gzipped (the Garble compiler included) against 486 KB
gzipped for the chunk that carries emp-wasm and mpc-framework (summon not included).

## Not established here

- Over a real network the ordering may reverse: bytes and round trips were counted, not timed on a link.
- No security review of the prototype's integration, and none of Polytune's alpha API.
- `wasm-opt` was not run, so the package size and speed are before that optimisation.
- What adopting it would take in the web build: a Rust and wasm-pack step, and a decision on whether
  the web image or a separate artefact carries it.
