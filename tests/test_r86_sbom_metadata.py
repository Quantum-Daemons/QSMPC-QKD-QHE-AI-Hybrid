"""The SBOMs get the metadata their generators leave out, and nothing that is not true (Round 86, 2026-11-23).

WHY THIS EXISTS

sbomqs, report only in CI, graded the four SBOMs F, F, D and F on 2026-11-23, and the owner chose to look into it. Part
of the gap was deliberate (the generators write reproducibly: no creation time, no random serial number) and part was
wrong: the WASM SBOM said it was made on 1970-01-01, because `SOURCE_DATE_EPOCH=0` stood in for a time.
`scripts/sbom_metadata.py` fills in only what is true and the same on every run of a commit. This file holds:

1. a missing or 1970-01-01 creation time replaced by the commit's own time, and a real one kept;
2. a serial number derived from the content where none was written, the same on a second run, and a written one kept;
3. the supplier and the "build" lifecycle phase; the Python SBOM's primary component (the `qsh` package, at its version);
   the threshold-aggregation module's version where it had none;
4. no hash and no licence from the metadata itself. The web SBOM's component hashes are `web/package-lock.json`'s
   SHA-512s, nested packages included, and the step stops where the lockfile disagrees with what cyclonedx-npm wrote;
   every integrity in the real lockfile is one it reads. The Python SBOM's licences are copied only from the same
   package at the same version in cyclonedx-py's environment-mode SBOM (the owner's choice of 2026-11-23: read in CI's
   python job, where `requirements.lock` is installed);
5. the result still valid against the vendored CycloneDX 1.7 schema, the generator script calling the step and no
   longer setting `SOURCE_DATE_EPOCH=0`, and CI writing the environment SBOM in the python job before its tests, with
   the sbom job waiting for that job, still running when it fails, and passing the file on;
6. the threshold-aggregation module's SBOM (`wasm/qsh-fedthresh/sbom.py`, the owner's choice of 2026-11-23): each Go
   module's hash is the SHA-256 of the zip `go mod download` reports, and a module with an error, no zip or another
   version stops it. Go is not needed here: the go command's output is replayed.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

import qsh

REPO = Path(__file__).resolve().parents[1]


def _module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


META = _module("sbom_metadata", "scripts/sbom_metadata.py")
WHEN = "2026-11-23T06:36:09Z"


def _doc(**metadata) -> dict:
    return {"bomFormat": "CycloneDX", "specVersion": "1.7", "version": 1, "metadata": dict(metadata),
            "components": [{"type": "library", "bom-ref": "pkg:pypi/a@1", "name": "a", "version": "1"}]}


def test_the_creation_time_is_the_commit_s_where_it_was_missing_or_1970() -> None:
    for before in (None, "1970-01-01T00:00:00Z", "1970-01-01T00:00:00.000000000Z"):
        doc = _doc() if before is None else _doc(timestamp=before)
        assert META.complete(doc, "wasm", WHEN, "0.85.0")["metadata"]["timestamp"] == WHEN
    real = _doc(timestamp="2026-11-20T01:02:03Z")
    assert META.complete(real, "wasm", WHEN, "0.85.0")["metadata"]["timestamp"] == "2026-11-20T01:02:03Z"


def test_the_serial_number_comes_from_the_content_and_does_not_change() -> None:
    a = META.complete(_doc(), "web", WHEN, "0.85.0")
    b = META.complete(_doc(), "web", WHEN, "0.85.0")
    assert a["serialNumber"].startswith("urn:uuid:") and a["serialNumber"] == b["serialNumber"]
    again = META.complete(copy.deepcopy(a), "web", WHEN, "0.85.0")
    assert again == a, "a second run changes nothing"
    other = META.complete(_doc(), "python", WHEN, "0.85.0")
    assert other["serialNumber"] != a["serialNumber"]
    kept = _doc()
    kept["serialNumber"] = "urn:uuid:28827157-a345-55b8-833d-6644a89c338d"
    assert META.complete(kept, "fedthresh", WHEN, "0.85.0")["serialNumber"] == "urn:uuid:28827157-a345-55b8-833d-6644a89c338d"


def test_the_supplier_lifecycle_and_components_are_the_project_s() -> None:
    py = META.complete(_doc(), "python", WHEN, qsh.__version__)
    assert py["metadata"]["supplier"] == {"name": "Quantum Daemons", "url": ["https://quantum.daemons.jp/"]}
    assert py["metadata"]["lifecycles"] == [{"phase": "build"}]
    assert py["metadata"]["component"]["name"] == "qsh" and py["metadata"]["component"]["version"] == qsh.__version__
    go = META.complete(_doc(component={"type": "library", "bom-ref": "m", "name": "qsh-fedthresh"}), "fedthresh", WHEN, qsh.__version__)
    assert go["metadata"]["component"]["version"] == qsh.__version__
    web = META.complete(_doc(component={"type": "application", "name": "qsh-web", "version": "0.85.0"}), "web", WHEN, "9.9.9")
    assert web["metadata"]["component"] == {"type": "application", "name": "qsh-web", "version": "0.85.0"}, "a named component is kept as it is"


def test_the_metadata_itself_adds_no_hash_or_licence() -> None:
    doc = META.complete(_doc(), "python", WHEN, "0.85.0")
    for c in doc["components"]:
        assert "hashes" not in c and "licenses" not in c


def _npm(name: str, version: str, path: str, written: str | None = None) -> dict:
    c = {"type": "library", "bom-ref": f"qsh-web|{name}@{version}", "name": name, "version": version,
         "properties": [{"name": "cdx:npm:package:path", "value": path}]}
    if written is not None:
        c["externalReferences"] = [{"type": "distribution", "url": f"https://registry.npmjs.org/{name}/-/{name}-{version}.tgz",
                                    "hashes": [{"alg": "SHA-512", "content": written}]}]
    return c


TARBALL = hashlib.sha512(b"a-1.0.0.tgz").digest()
LOCK = {"node_modules/a": {"version": "1.0.0", "integrity": "sha512-" + base64.b64encode(TARBALL).decode()},
        "node_modules/a/node_modules/b": {"version": "2.0.0", "integrity": "sha512-" + base64.b64encode(TARBALL).decode()}}


def test_the_web_sbom_s_hashes_are_the_lockfile_s_and_agree_with_cyclonedx_npm_s() -> None:
    a = _npm("a", "1.0.0", "node_modules/a", TARBALL.hex())
    a["components"] = [_npm("b", "2.0.0", "node_modules/a/node_modules/b")]
    doc = _doc()
    doc["components"] = [a, _npm("c", "3.0.0", "node_modules/c")]
    META.lockfile_hashes(doc, LOCK)
    assert a["hashes"] == [{"alg": "SHA-512", "content": TARBALL.hex()}]
    assert a["components"][0]["hashes"] == [{"alg": "SHA-512", "content": TARBALL.hex()}], "a nested package gets its hash too"
    assert "hashes" not in doc["components"][1], "a package the lockfile has no entry for gets none"
    again = copy.deepcopy(doc)
    META.lockfile_hashes(again, LOCK)
    assert again == doc, "a second run changes nothing"
    for written, version in ((hashlib.sha512(b"another tarball").hexdigest(), "1.0.0"), (TARBALL.hex(), "1.0.1")):
        broken = _doc()
        broken["components"] = [_npm("a", version, "node_modules/a", written)]
        with pytest.raises(ValueError):
            META.lockfile_hashes(broken, LOCK)


def test_every_integrity_in_the_web_lockfile_is_one_the_step_reads() -> None:
    packages = json.loads((REPO / "web" / "package-lock.json").read_text(encoding="utf-8"))["packages"]
    read = 0
    for path, entry in packages.items():
        if entry.get("integrity"):
            alg, digest = META.sri(entry["integrity"])
            assert len(digest) == {"SHA-512": 128, "SHA-384": 96, "SHA-256": 64, "SHA-1": 40}[alg], path
            read += 1
    assert read > 100, "the lockfile's integrity values were not found"


def test_the_python_sbom_s_licences_come_from_the_same_package_at_the_same_version() -> None:
    mit = [{"license": {"acknowledgement": "declared", "id": "MIT"}}]
    environment = {"components": [
        {"type": "library", "bom-ref": "Foo_Bar==1.0", "name": "Foo_Bar", "version": "1.0", "licenses": mit},
        {"type": "library", "bom-ref": "baz==2.1", "name": "baz", "version": "2.1", "licenses": [{"expression": "Apache-2.0 OR MIT"}]},
        {"type": "library", "bom-ref": "qux==3", "name": "qux", "version": "3"}]}
    doc = _doc()
    doc["components"] = [{"type": "library", "bom-ref": f"requirements-L{i}", "name": n, "version": v}
                         for i, (n, v) in enumerate((("foo-bar", "1.0"), ("baz", "2.0"), ("qux", "3")))]
    META.environment_licences(doc, environment)
    got = {c["name"]: c.get("licenses") for c in doc["components"]}
    assert got == {"foo-bar": mit, "baz": None, "qux": None}, "only the same normalised name at the same version"
    got["foo-bar"][0]["license"]["id"] = "changed"
    assert environment["components"][0]["licenses"][0]["license"]["id"] == "MIT", "copied, not shared"


def test_the_result_validates_and_the_generator_runs_the_step() -> None:
    validate = _module("validate_cyclonedx", "scripts/validate_cyclonedx.py").validator()
    for name in ("python", "web", "wasm", "fedthresh"):
        doc = META.complete(_doc(timestamp="1970-01-01T00:00:00Z"), name, WHEN, "0.85.0")
        errors = [e.message for e in validate.iter_errors(doc)]
        assert not errors, (name, errors[:3])
    web = _doc(timestamp="1970-01-01T00:00:00Z")
    web["components"] = [_npm("a", "1.0.0", "node_modules/a", TARBALL.hex())]
    META.lockfile_hashes(web, LOCK)
    errors = [e.message for e in validate.iter_errors(META.complete(web, "web", WHEN, "0.85.0"))]
    assert not errors, ("web, with its hashes", errors[:3])
    script = (REPO / "scripts" / "generate_sbom.sh").read_text(encoding="utf-8")
    assert "python3 scripts/sbom_metadata.py" in script
    assert "SOURCE_DATE_EPOCH=0" not in script and 'SOURCE_DATE_EPOCH="$(git log -1 --format=%ct)"' in script
    assert '${env_sbom:+--python-environment "$env_sbom"}' in script and "QSH_PYTHON_ENVIRONMENT_SBOM" in script


def test_ci_reads_the_licences_in_the_python_job_and_the_sbom_job_waits_for_it() -> None:
    yaml = pytest.importorskip("yaml", reason="PyYAML is needed to parse the workflow")
    jobs = yaml.safe_load((REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))["jobs"]
    steps = jobs["python"]["steps"]
    at = next(i for i, s in enumerate(steps) if "cyclonedx-py\" environment" in s.get("run", ""))
    assert '"cyclonedx-bom==9.0.0"' in steps[at]["run"] and "--sv 1.7" in steps[at]["run"] and "$(command -v python)" in steps[at]["run"]
    upload = steps[at + 1]
    assert upload["uses"].startswith("actions/upload-artifact") and upload["with"]["name"] == "python-environment-sbom-${{ matrix.python }}"
    assert at < next(i for i, s in enumerate(steps) if s.get("name") == "Test suite"), "written before the tests, so a failing test still leaves it"
    sbom = jobs["sbom"]
    assert sbom["needs"] == "python" and sbom["if"] == "${{ !cancelled() }}"
    down = next(s for s in sbom["steps"] if str(s.get("uses", "")).startswith("actions/download-artifact"))
    versions = jobs["python"]["strategy"]["matrix"]["python"]
    assert down["continue-on-error"] is True and down["with"]["name"] == f"python-environment-sbom-{versions[0]}"
    gen = next(s for s in sbom["steps"] if s.get("name") == "Generate the SBOMs")
    assert gen["env"]["QSH_PYTHON_ENVIRONMENT_SBOM"] == down["with"]["path"] + "/python-environment.cdx.json"


def _fedthresh_sbom():
    import sys

    sys.path.insert(0, str(REPO / "wasm" / "qsh-fedthresh"))
    try:
        return _module("fedthresh_sbom", "wasm/qsh-fedthresh/sbom.py")
    finally:
        sys.path.remove(str(REPO / "wasm" / "qsh-fedthresh"))


def test_the_go_modules_hashes_are_their_verified_zips(tmp_path, monkeypatch) -> None:
    import subprocess

    sbom = _fedthresh_sbom()
    zips = {}
    for name, body in (("a", b"module a zip"), ("b", b"module b zip")):
        zips[name] = tmp_path / f"{name}.zip"
        zips[name].write_bytes(body)
    modules = {"example.com/a": ("v1.0.0", tmp_path), "example.com/b/v2": ("v2.1.0", tmp_path)}

    def replay(*reports, code=None):
        printed = "".join(json.dumps(r, indent="\t") + "\n" for r in reports)
        failed = 1 if code is None and any("Error" in r for r in reports) else (code or 0)
        monkeypatch.setattr(sbom.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, failed, stdout=printed, stderr="go: failed"))

    replay({"Path": "example.com/a", "Version": "v1.0.0", "Zip": str(zips["a"]), "Sum": "h1:x"},
           {"Path": "example.com/b/v2", "Version": "v2.1.0", "Zip": str(zips["b"]), "Sum": "h1:y"})
    assert sbom.zip_hashes(modules) == {"example.com/a": hashlib.sha256(b"module a zip").hexdigest(),
                                        "example.com/b/v2": hashlib.sha256(b"module b zip").hexdigest()}
    for broken in ([{"Path": "example.com/a", "Version": "v1.0.0", "Error": "checksum mismatch"}],
                   [{"Path": "example.com/a", "Version": "v1.0.0", "Dir": str(tmp_path)}],
                   [{"Path": "example.com/a", "Version": "v1.0.1", "Zip": str(zips["a"])},
                    {"Path": "example.com/b/v2", "Version": "v2.1.0", "Zip": str(zips["b"])}],
                   [{"Path": "example.com/a", "Version": "v1.0.0", "Zip": str(zips["a"])}]):
        replay(*broken)
        with pytest.raises(SystemExit):
            sbom.zip_hashes(modules)
    replay({"Path": "example.com/a", "Version": "v1.0.0", "Zip": str(zips["a"])},
           {"Path": "example.com/b/v2", "Version": "v2.1.0", "Zip": str(zips["b"])}, code=1)
    with pytest.raises(SystemExit):
        sbom.zip_hashes(modules)
    script = (REPO / "wasm" / "qsh-fedthresh" / "sbom.py").read_text(encoding="utf-8")
    assert '"hashes": [{"alg": "SHA-256", "content": hashes[path]}]' in script

