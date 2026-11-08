"""The cryptography bill of materials says what the code calls, and only that (Round 86).

The document is generated from `qsh.crypto_inventory`, which is hand-written, so what keeps it true is this file: every entry's evidence must still be in the source, every cryptographic import
must belong to an entry, the families and curves must be CycloneDX's own, and the document must validate against the vendored 1.7 schema.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
from pathlib import Path

import jsonschema
import pytest
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT7

from qsh.crypto_inventory import ASSETS, PY_HASHLIB, PY_IMPORTS, TS_IMPORTS, TS_UTILITY_IMPORTS

REPO = Path(__file__).resolve().parents[1]
SCHEMAS = REPO / "schemas" / "cyclonedx"
CBOM = REPO / "web" / "public" / "cbom" / "qsh.cdx.json"


def _validator():
    registry = Registry()
    for name in ("bom-1.7.schema.json", "spdx.schema.json", "jsf-0.82.schema.json", "cryptography-defs.schema.json"):
        data = json.loads((SCHEMAS / name).read_text())
        resource = Resource.from_contents(data, default_specification=DRAFT7)
        registry = registry.with_resource(uri=data.get("$id", name), resource=resource).with_resource(uri=name, resource=resource)
    root = json.loads((SCHEMAS / "bom-1.7.schema.json").read_text())
    return jsonschema.validators.validator_for(root)(root, registry=registry)


def _tracked(*patterns: str) -> list[Path]:
    out = subprocess.run(["git", "ls-files", *patterns], cwd=REPO, capture_output=True, text=True, check=True).stdout
    return [REPO / line for line in out.splitlines() if line]


def _build() -> dict:
    spec = importlib.util.spec_from_file_location("generate_cbom", REPO / "scripts" / "generate_cbom.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build()


def test_the_file_is_what_the_generator_writes() -> None:
    assert json.loads(CBOM.read_text()) == _build(), "run scripts/generate_cbom.py"


def test_the_document_validates_against_the_vendored_schema() -> None:
    errors = [e.message[:160] for e in _validator().iter_errors(json.loads(CBOM.read_text()))]
    assert not errors, errors[:5]


@pytest.mark.parametrize(("path", "value"), [
    (("components", 0, "cryptoProperties", "algorithmProperties", "primitive"), "nonsense"),
    (("components", 0, "cryptoProperties", "algorithmProperties", "algorithmFamily"), "NOT-A-FAMILY"),
    (("components", 0, "type"), "bogus"),
])
def test_the_schema_would_catch_a_bad_document(path: tuple, value: str) -> None:
    doc = json.loads(CBOM.read_text())
    node = doc
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    assert list(_validator().iter_errors(doc)), "the validator accepted a wrong value, so a pass means nothing"


@pytest.mark.parametrize("asset", ASSETS, ids=lambda a: a.ref)
def test_every_entry_is_still_in_the_source(asset) -> None:
    for path, text in asset.evidence:
        assert text in (REPO / path).read_text(), f"{asset.ref}: {path} no longer contains {text!r}"


def test_every_cryptographic_import_belongs_to_an_entry() -> None:
    refs = {a.ref for a in ASSETS}
    problems: list[str] = []
    pattern = re.compile(r"^import\s*\{([^}]*)\}\s*from\s*[\"']@noble/([^\"']+)[\"']", re.M)
    for path in _tracked("web/src/**/*.ts", "web/src/**/*.tsx"):
        for names, _lib in pattern.findall(path.read_text()):
            for raw in names.split(","):
                name = raw.strip().split(" as ")[0].strip()
                if not name or name in TS_UTILITY_IMPORTS:
                    continue
                if TS_IMPORTS.get(name) not in refs:
                    problems.append(f"{path.relative_to(REPO)}: {name}")
    for path in _tracked("src/qsh/**/*.py"):
        text = path.read_text()
        for names in re.findall(r"^from cryptography\S*\s+import\s+\(?([^)\n]*(?:\n[^)\n]*)*?)\)?$", text, re.M):
            for raw in names.replace("\n", ",").split(","):
                name = raw.strip()
                if name and name != "hashes" and PY_IMPORTS.get(name) not in refs:
                    problems.append(f"{path.relative_to(REPO)}: {name}")
        for name in re.findall(r"hashlib\.([a-z0-9_]+)\(", text):
            if PY_HASHLIB.get(name) not in refs:
                problems.append(f"{path.relative_to(REPO)}: hashlib.{name}")
    assert not problems, f"cryptography with no entry in qsh.crypto_inventory: {sorted(set(problems))}"


def test_families_and_curves_are_cyclonedx_s_own() -> None:
    defs = json.loads((SCHEMAS / "cryptography-defs.schema.json").read_text())["definitions"]
    families, curves = set(defs["algorithmFamiliesEnum"]["enum"]), set(defs["ellipticCurvesEnum"]["enum"])
    for a in ASSETS:
        assert a.family is None or a.family in families, a.ref
        assert a.curve is None or a.curve in curves, a.ref


def test_the_quantum_exposure_is_stated_for_what_shor_breaks() -> None:
    by_ref = {a.ref: a for a in ASSETS}
    for ref in ("x25519", "ecdh-p256", "oprf-ristretto255", "wrk17b"):
        assert by_ref[ref].nist_quantum_level == 0, ref
    for ref, level in (("ml-kem-768", 3), ("ml-kem-1024", 5), ("ml-dsa-65", 3), ("ml-dsa-87", 5)):
        assert by_ref[ref].nist_quantum_level == level, ref


def test_the_page_renders_it_and_claims_no_compliance() -> None:
    page = (REPO / "web" / "src" / "pages" / "Validations.tsx").read_text()
    panel = (REPO / "web" / "src" / "components" / "CbomPanel.tsx").read_text()
    assert "<CbomPanel" in page
    assert "cbom/qsh.cdx.json" in panel and "not the output of a scanner" in panel
    assert not re.search(r"\bcomplian|\bconformant", panel, re.I), "the page must not claim CBOM compliance"


def test_the_vendored_schema_carries_its_licence() -> None:
    assert (SCHEMAS / "LICENSE").read_text().lstrip().startswith("Apache License")
    assert "schemas/cyclonedx" in (REPO / "DATA_LICENSES.md").read_text()
