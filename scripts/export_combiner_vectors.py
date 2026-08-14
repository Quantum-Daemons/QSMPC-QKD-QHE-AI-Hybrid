#!/usr/bin/env python3
"""Emit the hybrid-KEM combiner's expected output as test vectors, from the production function.

WHY THIS EXISTS

Round 38 fixed the combiner across what it believed were all its call sites, twice, and was wrong
both times: `1252dde` changed four and missed a fifth; `5efa74a` then added the output-length tag to
the Python side ONLY, one commit after a message reading "All four sites change together - they
must, or backend and browser derive different keys." They did not. Measured on the tree that
shipped:

    python info : qsh/hybrid-kem/v2 ML-KEM-768+X25519/L32|<transcript>
    browser info: qsh/hybrid-kem/v2 ML-KEM-768+X25519|<transcript>
    same session key?  False

Three shipped strings asserted the opposite, one of them rendered in the Validations panel. Nothing
could catch it: `tests/test_hybrid_combiner_binding.py` is Python-only, and the browser has no test
that touches the combiner at all.

So the two languages need a shared artifact rather than a shared intention. This writes one:
`_combine` is called - not described, not re-implemented - and its bytes are committed.
`web/scripts/check_hybrid_combiner.mjs` bundles the real browser module and reproduces them, and CI
fails on `git diff --quiet` if Python changed and this file was not regenerated. That is the same
mechanism `scripts/export_config.py` uses for `config.defaults.json`, which has held since Round 20.

VECTOR CHOICE

Fixed, obviously-not-secret inputs: this asserts plumbing and byte-order, not key quality. Three
cases, and the third is the one that matters:

  - ML-KEM-768 at length 32   - the transport session key, by far the most-travelled path
  - ML-KEM-1024 at length 32  - the category-5 selection, which `kemImpl()` genuinely switches to
  - ML-KEM-768 at length 51   - `max(32, (2n+7)//8)` from `qhe/rate1_qfhe.py`, the only caller that
                                varies the length, and therefore the ONLY case that exercises the
                                length tag. Without it a regression that dropped `_length_tag`
                                would leave every vector unchanged.

Run after editing the combiner:

    python scripts/export_combiner_vectors.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qsh.pqc.hybrid_kem import _combine, _combiner_info  # noqa: E402
from qsh.pqc.transport import _mix_qkd_its  # noqa: E402

OUT = ROOT / "web" / "scripts" / "vectors" / "hybrid_combiner_vectors.json"

#: Deterministic stand-ins for the two shared secrets and the four transcript elements. Lengths are
#: the real ones for ML-KEM-768 so the concatenation is the shape the browser will build, but the
#: contents are a counter pattern - nothing here is or resembles key material.
def _pat(n: int, seed: int) -> bytes:
    return bytes((i * seed + 17) & 0xFF for i in range(n))


CASES = [
    {"name": "ml-kem-768/L32", "kem": "ML-KEM-768", "length": 32},
    {"name": "ml-kem-1024/L32", "kem": "ML-KEM-1024", "length": 32},
    {"name": "ml-kem-768/L51", "kem": "ML-KEM-768", "length": 51},
]


def main() -> int:
    k1, k2 = _pat(32, 3), _pat(32, 5)
    # c1 || c2 || ek1 || ek2, the SP 800-227 s4.6.3 order. Sizes are ML-KEM-768's real ones
    # (ct 1088, ek 1184) with X25519's 32-byte ephemeral and public.
    c1, c2, ek1, ek2 = _pat(1088, 7), _pat(32, 11), _pat(1184, 13), _pat(32, 19)
    transcript = c1 + c2 + ek1 + ek2

    vectors = []
    for case in CASES:
        info_prefix = _combiner_info(case["kem"])
        derived = _combine(k1, k2, transcript, case["length"], info_prefix)
        vectors.append({
            **case,
            "k1": k1.hex(), "k2": k2.hex(),
            "c1": c1.hex(), "c2": c2.hex(), "ek1": ek1.hex(), "ek2": ek2.hex(),
            "expected": derived.hex(),
        })

    # The ITS combiner, for the same reason and found the same way. `clientMcp.ts`'s `itsCombine`
    # called itself a "mirror of transport._mix_qkd_its" while seeding its Toeplitz matrix from a
    # SHA-256 counter chain against Python's SHAKE-256, so the two derived different keys from the
    # same inputs. Nothing caught it - the hybrid-KEM gate covers only the KEM combiner - and it is
    # the identical defect one module over. Two lengths, because the extractor's `need_t` depends on
    # the input size and a wrong slice would survive a single-length vector.
    its_vectors = []
    for n in (32, 48):
        p, q = _pat(n, 3), _pat(n, 5)
        its_vectors.append({
            "name": f"its_toeplitz/{n}B",
            "pqc": p.hex(), "qkd": q.hex(),
            "expected": _mix_qkd_its(p, q).hex(),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "_generated_by": "scripts/export_combiner_vectors.py (do not edit by hand; edit "
                         "src/qsh/pqc/hybrid_kem.py or src/qsh/pqc/transport.py and regenerate)",
        "_what_this_is": "Bytes produced by the PYTHON combiners. web/scripts/check_hybrid_combiner.mjs "
                         "bundles the real browser modules and must reproduce them exactly.",
        "vectors": vectors,
        "its_vectors": its_vectors,
    }, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} "
          f"({len(vectors)} hybrid-KEM, {len(its_vectors)} ITS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
