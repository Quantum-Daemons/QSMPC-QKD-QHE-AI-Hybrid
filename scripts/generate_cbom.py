#!/usr/bin/env python
"""Write the CycloneDX 1.7 cryptography bill of materials from `qsh.crypto_inventory` (Round 86).

The output is deterministic (no timestamp; the serial number is derived from the content), so `tests/test_r86_cbom.py` can require the file to be what this writes.

    .venv/bin/python scripts/generate_cbom.py
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OUTPUT = REPO / "web" / "public" / "cbom" / "qsh.cdx.json"
NOTE = ("Generated from this repository's own list of the cryptography its code calls (src/qsh/crypto_inventory.py), each entry checked against the source. Not the output of a scanner; "
        "library internals are not listed, and nothing here is FIPS 140-3 validated.")


def _component(a) -> dict:
    alg: dict = {"primitive": a.primitive, "executionEnvironment": "software-plain-ram", "implementationPlatform": "generic", "cryptoFunctions": list(a.functions)}
    if a.family:
        alg["algorithmFamily"] = a.family
    if a.parameter_set:
        alg["parameterSetIdentifier"] = a.parameter_set
    if a.mode:
        alg["mode"] = a.mode
    if a.curve:
        alg["ellipticCurve"] = a.curve
    if a.nist_quantum_level is not None:
        alg["nistQuantumSecurityLevel"] = a.nist_quantum_level
    if a.classical_bits is not None:
        alg["classicalSecurityLevel"] = a.classical_bits
    crypto: dict = {"assetType": "algorithm", "algorithmProperties": alg}
    if a.oid:
        crypto["oid"] = a.oid
    return {
        "type": "cryptographic-asset",
        "bom-ref": f"crypto/{a.ref}",
        "name": a.name,
        "description": a.role,
        "cryptoProperties": crypto,
        "properties": [{"name": "qsh:standard", "value": a.standard}, {"name": "qsh:scope", "value": ", ".join(a.scope)}],
        "evidence": {"occurrences": [{"location": path} for path in dict.fromkeys(p for p, _ in a.evidence)]},
    }


def build() -> dict:
    import qsh
    from qsh.crypto_inventory import ASSETS

    components = [_component(a) for a in ASSETS]
    root = "qsh"
    body = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.7",
        "version": 1,
        "metadata": {
            "component": {"type": "application", "bom-ref": root, "name": "qsh", "version": qsh.__version__},
            "properties": [{"name": "qsh:scope", "value": NOTE}],
        },
        "components": components,
        "dependencies": [{"ref": root, "dependsOn": [c["bom-ref"] for c in components]}],
    }
    serial = uuid.uuid5(uuid.NAMESPACE_URL, "qsh-cbom:" + json.dumps(body, sort_keys=True))
    return {"bomFormat": body["bomFormat"], "specVersion": body["specVersion"], "serialNumber": f"urn:uuid:{serial}", **{k: v for k, v in body.items() if k not in ("bomFormat", "specVersion")}}


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {OUTPUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
