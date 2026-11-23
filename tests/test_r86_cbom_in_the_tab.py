"""The page checks the CBOM against the CycloneDX 1.7 schema in the tab, and judges it as CI does (Round 86, 2026-11-23).

WHY THIS EXISTS

The owner's choice of 2026-11-23: the Validations page validates the CBOM in the browser with Ajv 8.20.0. The page's
Content-Security-Policy has no 'unsafe-eval', so the schema is compiled ahead of time into
`web/src/lib/generated/cdx17Validate.js` by `web/scripts/build_cdx_validator.mjs`, from the vendored schemas
`tests/test_r86_cbom.py` validates against. Before compiling, the generator drops the keywords that only annotate a
schema (titles, descriptions), which were most of the 1.2 MB it first wrote. A validator in the page is worth
something only if it is that schema and judges as CI does, so this file holds:

1. the committed file to be what the generator writes, with the pinned Ajv;
2. the annotation stripping to keep every property name (CycloneDX has a property called "description", which a
   careless strip would delete, and with additionalProperties false every component would then be refused);
3. the page's validator to accept the published CBOM and to say what Python's jsonschema says on fourteen variants of
   it, twelve of them made wrong, each with its outcome written here, so a pair that agreed by both being broken
   still fails;
4. the page to load it lazily through `checkCbom` and say on the page what checked the document.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

import jsonschema
import pytest
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT7

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
SCHEMAS = REPO / "schemas" / "cyclonedx"
CBOM = WEB / "public" / "cbom" / "qsh.cdx.json"
GENERATED = WEB / "src" / "lib" / "generated" / "cdx17Validate.js"
GENERATOR = WEB / "scripts" / "build_cdx_validator.mjs"
AJV = "8.20.0"


def _python_validator():
    registry = Registry()
    for name in ("bom-1.7.schema.json", "spdx.schema.json", "jsf-0.82.schema.json", "cryptography-defs.schema.json"):
        data = json.loads((SCHEMAS / name).read_text())
        resource = Resource.from_contents(data, default_specification=DRAFT7)
        registry = registry.with_resource(uri=data.get("$id", name), resource=resource).with_resource(uri=name, resource=resource)
    root = json.loads((SCHEMAS / "bom-1.7.schema.json").read_text())
    return jsonschema.validators.validator_for(root)(root, registry=registry)


def _alg(d: dict) -> dict:
    return d["components"][0]["cryptoProperties"]["algorithmProperties"]


def _set(fn):
    def mutate(d: dict) -> dict:
        fn(d)
        return d
    return mutate


# (what was done, the outcome the schema gives, how)
CASES = [
    ("the CBOM as published", True, _set(lambda d: None)),
    ("a primitive the schema does not list", False, _set(lambda d: _alg(d).__setitem__("primitive", "nonsense"))),
    ("a family not in CycloneDX's registry", False, _set(lambda d: _alg(d).__setitem__("algorithmFamily", "NOT-A-FAMILY"))),
    ("a component type the schema does not list", False, _set(lambda d: d["components"][0].__setitem__("type", "bogus"))),
    ("no bomFormat", False, _set(lambda d: d.pop("bomFormat"))),
    ("specVersion as a number", False, _set(lambda d: d.__setitem__("specVersion", 1.7))),
    ("a top-level property the schema does not define", False, _set(lambda d: d.__setitem__("surprise", 1))),
    ("a certification level the schema does not list", False, _set(lambda d: _alg(d).__setitem__("certificationLevel", ["fips140-9-l1"]))),
    ("a quantum security level of 7", False, _set(lambda d: _alg(d).__setitem__("nistQuantumSecurityLevel", 7))),
    ("one dependency named twice", False,
     _set(lambda d: d["dependencies"][1].__setitem__("dependsOn", d["dependencies"][1]["dependsOn"][:1] * 2))),
    ("a licence ID SPDX does not list", False,
     _set(lambda d: d["metadata"]["component"].__setitem__("licenses", [{"license": {"id": "NOT-A-LICENSE"}}]))),
    ("a licence ID SPDX lists", True, _set(lambda d: d["metadata"]["component"].__setitem__("licenses", [{"license": {"id": "Apache-2.0"}}]))),
    ("a description on the application", True, _set(lambda d: d["metadata"]["component"].__setitem__("description", "the application"))),
    ("a dependency with a key the schema does not define", False, _set(lambda d: d["dependencies"][0].__setitem__("surprise", 1))),
    ("a signature whose algorithm is a number", False, _set(lambda d: d.__setitem__("signature", {"algorithm": 5, "value": "AA"}))),
]

RUNNER = r"""
import { readFileSync } from "node:fs";
const { validate } = await import(%(bundle)s);
const { schemas, stripAnnotations } = await import(%(generator)s);
const docs = JSON.parse(readFileSync(%(docs)s, "utf8"));
const out = { verdicts: docs.map((d) => validate(d)), firstError: null };
validate(docs[1]); out.firstError = validate.errors?.[0] ?? null;
// Every property name, at every schema position of every schema, before and after the strip. A `properties` map's
// keys are names and its values are schemas; CycloneDX even has a property called "properties", whose schema's keys
// ("type", "title", ...) are not names. Values that are data or annotation (enum, const, examples...) hold no schema.
const MAPS = new Set(["properties", "patternProperties", "definitions", "$defs", "dependencies"]);
const NOT_SCHEMA = new Set(["enum", "const", "required", "type", "examples", "default", "title", "description", "$comment", "meta:enum"]);
const names = (schema, acc) => {
  if (Array.isArray(schema)) { for (const x of schema) names(x, acc); return acc; }
  if (!schema || typeof schema !== "object") return acc;
  for (const [k, v] of Object.entries(schema)) {
    if (NOT_SCHEMA.has(k)) continue;
    if (MAPS.has(k) && v && typeof v === "object" && !Array.isArray(v)) {
      for (const [name, sub] of Object.entries(v)) { if (k === "properties") acc.push(name); names(sub, acc); }
    } else names(v, acc);
  }
  return acc;
};
const raw = schemas();
out.before = raw.map((s) => names(s, []).sort());
out.after = raw.map((s) => names(stripAnnotations(s), []).sort());
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def page(tmp_path_factory) -> dict:
    missing = [n for n, ok in (("web/node_modules/.bin/esbuild", ESBUILD.exists()), ("node", shutil.which("node"))) if not ok]
    if missing:
        msg = f"{' and '.join(missing)} absent, so the browser code cannot run. This is a SKIP, not a pass."
        if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
            pytest.fail(msg)
        pytest.skip(msg)
    d = tmp_path_factory.mktemp("cbomtab")
    bundle, runner, docs = d / "validator.mjs", d / "run.mjs", d / "docs.json"
    subprocess.run([str(ESBUILD), str(GENERATED), "--bundle", "--format=esm", "--platform=node", f"--outfile={bundle}",
                    "--log-level=error"], cwd=WEB, check=True, capture_output=True)
    base = json.loads(CBOM.read_text(encoding="utf-8"))
    docs.write_text(json.dumps([how(copy.deepcopy(base)) for _, _, how in CASES]))
    runner.write_text(RUNNER % {"bundle": json.dumps(bundle.as_uri()), "generator": json.dumps(GENERATOR.as_uri()),
                                "docs": json.dumps(str(docs))})
    proc = subprocess.run(["node", str(runner)], capture_output=True, text=True, timeout=300, cwd=WEB)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout)


def test_the_committed_validator_is_what_the_generator_writes() -> None:
    if not (WEB / "node_modules" / "ajv").exists() or not shutil.which("node"):
        msg = "web/node_modules/ajv or node absent, so the generator cannot run. This is a SKIP, not a pass."
        if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
            pytest.fail(msg)
        pytest.skip(msg)
    proc = subprocess.run(["node", str(GENERATOR), "--check"], cwd=WEB, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    head = GENERATED.read_text(encoding="utf-8")[:400]
    assert f"with Ajv {AJV} (MIT)" in head
    assert json.loads((WEB / "node_modules" / "ajv" / "package.json").read_text())["version"] == AJV


def test_ajv_is_pinned() -> None:
    pkg = json.loads((WEB / "package.json").read_text())
    assert pkg["dependencies"]["ajv"] == AJV, "ajv's runtime helpers ship in the bundle, so it is a dependency, pinned exactly"
    lock = json.loads((WEB / "package-lock.json").read_text())
    assert lock["packages"]["node_modules/ajv"]["version"] == AJV


def test_stripping_the_annotations_keeps_every_property_name(page: dict) -> None:
    assert page["before"] == page["after"], "a property name was stripped as if it were an annotation"
    assert any("description" in names for names in page["before"]), "the check would not have caught the case it exists for"


@pytest.mark.parametrize(("i", "case"), list(enumerate(CASES)), ids=[c[0] for c in CASES])
def test_the_page_judges_each_document_as_python_does(page: dict, i: int, case: tuple) -> None:
    what, expected, how = case
    doc = how(json.loads(CBOM.read_text(encoding="utf-8")))
    python_valid = not list(_python_validator().iter_errors(doc))
    assert python_valid == expected, f"{what}: Python's jsonschema says {python_valid}, this file expects {expected}"
    assert page["verdicts"][i] == expected, f"{what}: the page's validator says {page['verdicts'][i]}, Python says {expected}"


def test_the_page_says_where_a_document_is_wrong(page: dict) -> None:
    e = page["firstError"]
    assert e["instancePath"] == "/components/0/cryptoProperties/algorithmProperties/primitive", e
    assert e["keyword"] == "enum" and e["message"], e


def test_the_page_loads_it_lazily_and_names_it() -> None:
    check = (WEB / "src" / "lib" / "cbomSchemaCheck.ts").read_text(encoding="utf-8")
    panel = (WEB / "src" / "components" / "CbomPanel.tsx").read_text(encoding="utf-8")
    assert 'await import("./generated/cdx17Validate.js")' in check, "the validator must stay out of the first load"
    assert "compiled ahead of time by Ajv 8.20.0" in check
    assert "checkCbom(" in panel and "SCHEMA_CHECK" in panel
    assert "cdx17Validate" not in panel, "the panel reaches the validator only through checkCbom's dynamic import"
    attrs = (REPO / ".gitattributes").read_text(encoding="utf-8")
    assert "web/src/lib/generated/cdx17Validate.js linguist-generated=true" in attrs
