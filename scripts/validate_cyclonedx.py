#!/usr/bin/env python
"""Validate CycloneDX 1.7 JSON documents against the vendored schema, offline (Round 86). Exits 1 and prints the first errors if any file fails.

    python scripts/validate_cyclonedx.py sbom-out/*.cdx.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT7

SCHEMAS = Path(__file__).resolve().parents[1] / "schemas" / "cyclonedx"


def validator():
    registry = Registry()
    for name in ("bom-1.7.schema.json", "spdx.schema.json", "jsf-0.82.schema.json", "cryptography-defs.schema.json"):
        data = json.loads((SCHEMAS / name).read_text())
        resource = Resource.from_contents(data, default_specification=DRAFT7)
        registry = registry.with_resource(uri=data.get("$id", name), resource=resource).with_resource(uri=name, resource=resource)
    root = json.loads((SCHEMAS / "bom-1.7.schema.json").read_text())
    return jsonschema.validators.validator_for(root)(root, registry=registry)


def main(paths: list[str]) -> int:
    v, failed = validator(), 0
    for path in paths:
        doc = json.loads(Path(path).read_text())
        if doc.get("specVersion") != "1.7":
            print(f"FAIL {path}: specVersion {doc.get('specVersion')!r}, this checks 1.7")
            failed += 1
            continue
        errors = list(v.iter_errors(doc))
        print(f"{'FAIL' if errors else 'ok  '} {path}: {len(doc.get('components', []))} components")
        for e in errors[:5]:
            print(f"     {'/'.join(map(str, list(e.path)[:6]))}: {e.message[:160]}")
        failed += bool(errors)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
