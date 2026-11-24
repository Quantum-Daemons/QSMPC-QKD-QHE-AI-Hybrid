"""The SBOM tools CI runs on the CBOM and the SBOMs, and what the repository keeps of them (Round 86, 2026-11-23).

WHY THIS EXISTS

The owner's choices of 2026-11-23: sbom-tools 0.4.0 reads the CBOM against NIST IR 8547 in CI, offline, and its
reading is compared with the Validations page's, every difference reviewed by hand and none adopted automatically;
the CBOM part of its Markdown view is committed under docs/ as dated tool output, its whole view and JSON kept as CI
artefacts; sbom-utility v1.0.1 checks the CBOM against rules this repository writes; sbomqs 2.1.3 scores the four SBOMs;
cdxgen 13.8.0 reads a copy of web/ and only reports. Nothing any of them says is shown on the site. This file holds:

1. `scripts/compare_sbom_tools.py` reading a report the way its docstring says, passing reviewed differences and failing
   an unreviewed one, on a small report written here;
2. the committed comparison: every difference reviewed, the notes in the script exactly those differences, and its
   reading of each asset the one the page gives now;
3. the committed view: ASCII, labelled as dated tool output and not a compliance statement, the CRA section left out,
   naming the release digest CI pins and the CBOM digest the comparison read;
4. the workflow's pins and digests, sbom-tools run offline, cdxgen unable to fail the build, and the rules file;
5. no mention of sbom-tools in what the site serves.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import yaml

from qsh.pqc import ir8547 as py

REPO = Path(__file__).resolve().parents[1]
MEASURED = REPO / "docs" / "measurements"
RECORD = MEASURED / "sbom-tools-comparison-2026-11-23.json"
VIEW = MEASURED / "sbom-tools-0.4.0-cbom-view-2026-11-23.md"
RULES = REPO / "schemas" / "sbom-utility" / "qsh-cbom-rules.json"
CI = REPO / ".github" / "workflows" / "ci.yml"
PINS = {  # release asset: (version in its URL, SHA-256 GitHub records)
    "sbom-utility": ("v1.0.1", "05e62a9b79929203451c25b0ad11cdb3dc3c827cfee024297d1dc12fa25c53a6"),
    "sbom-tools": ("v0.4.0", "94e2447f7048059b26f23dbbca87537f3711ddf2871abf668c27829b6f3b86f8"),
    "sbomqs": ("v2.1.3", "c96f07a6a27314c1c268af141af04f1c1b49e61400d7058f6419ca4f7f3bdd6b"),
    "cdxgen": ("v13.8.0", "01f2a7f822449242987942ae81da7464dc9d8eed27ae36d33b0f4ddce95e5b43"),
}


def _compare_module():
    spec = importlib.util.spec_from_file_location("compare_sbom_tools", REPO / "scripts" / "compare_sbom_tools.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finding(ref: str, rule: str, severity: str, deadline: str | None = None) -> dict:
    return {"component_id": ref, "rule_id": rule, "severity": severity, "deadline": deadline, "message": "-"}


def _report(violations: list[dict], passed: list[dict], sha: str) -> dict:
    return {"schema": "sbom-tools/compliance/v2", "tool": {"name": "sbom-tools", "version": "0.4.0"}, "as_of": "2026-11-23T00:00:00Z",
            "subject": {"sha256": sha},
            "results": [{"standard": "pqc", "score": 0, "counts": {}, "violations": violations, "passed": passed}]}


def test_the_comparison_reads_a_report_as_its_docstring_says() -> None:
    m = _compare_module()
    sha = m.hashlib.sha256(m.CBOM.read_bytes()).hexdigest()
    report = _report(
        violations=[_finding("crypto/x25519", "SBOM-PQC-001", "Info", "2030-12-31"),
                    _finding("crypto/hybrid-combiner", "SBOM-PQC-001", "Info", "2030-12-31"),
                    _finding("crypto/hybrid-combiner", "SBOM-PQC-012", "Warning"),
                    _finding("crypto/ckks", "SBOM-PQC-ALG-UNKNOWN", "Warning"),
                    _finding("crypto/sha-256", "SBOM-PQC-012", "Warning")],
        passed=[{"component_id": "crypto/ml-kem-768", "rule_id": "SBOM-PQC-009"}], sha=sha)
    rec = m.compare(report)
    rows = {a["ref"]: a for a in rec["assets"]}
    assert rows["crypto/ml-kem-768"]["agreement"] == "agree"
    assert rows["crypto/x25519"]["agreement"] == "agree"
    assert rows["crypto/ckks"]["agreement"] == "agree", "neither classifies homomorphic encryption"
    assert rows["crypto/sha-256"]["agreement"] == "no verdict", "a missing-field warning is not a verdict"
    assert rows["crypto/aes-256-gcm"]["agreement"] == "no verdict", "no finding at all is not a verdict"
    combiner = rows["crypto/hybrid-combiner"]
    assert combiner["agreement"] == "differ" and combiner["reviewed"]
    assert combiner["sbom_tools"]["severity"] == "Info", "the severity is the deciding finding's, not the field warning's"
    assert rec["unreviewed"] == []
    # A difference nobody reviewed fails: ML-KEM-1024 called quantum-vulnerable.
    report["results"][0]["violations"].append(_finding("crypto/ml-kem-1024", "SBOM-PQC-001", "Error"))
    assert m.compare(report)["unreviewed"] == ["crypto/ml-kem-1024"]


def test_the_committed_comparison_is_reviewed_and_current() -> None:
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    m = _compare_module()
    differ = {a["ref"] for a in rec["assets"] if a["agreement"] == "differ"}
    assert rec["unreviewed"] == [] and all(a.get("reviewed") for a in rec["assets"] if a["agreement"] == "differ")
    assert differ == set(m.REVIEWED), "the script's review notes must be exactly the recorded differences"
    assert rec["counts"] == {k: sum(1 for a in rec["assets"] if a["agreement"] == k) for k in ("agree", "differ", "no verdict")}
    assert rec["tool"] == {"name": "sbom-tools", "version": "0.4.0"} and rec["tool_read_sha256"] == rec["cbom_sha256"]
    assert [a["ref"] for a in rec["assets"]] == [r.ref for r in py.RULES]
    for a in rec["assets"]:
        standing, label = py.standing_on(py.BY_REF[a["ref"]], rec["as_of"])
        assert (a["in_house"]["class"], a["in_house"]["standing"], a["in_house"]["label"]) == (py.BY_REF[a["ref"]].cls, standing, label), \
            f"{a['ref']}: the page's reading changed since the comparison; run sbom-tools again (scripts/ci_local.sh step 11n)"


def test_the_committed_view_is_labelled_and_the_tool_s_own_text() -> None:
    text = VIEW.read_text(encoding="utf-8")
    assert text.isascii(), "tracked Markdown is ASCII"
    label, _, body = text.partition("\n---\n")
    assert label.startswith("> **Dated tool output, not a compliance statement.**")
    assert PINS["sbom-tools"][1] in label
    assert json.loads(RECORD.read_text(encoding="utf-8"))["cbom_sha256"] in label
    assert "--as-of 2026-11-23 --offline" in label and "CRA section, which is left out here" in label
    assert body.lstrip().startswith("# SBOM Report") and "## Cryptographic Inventory" in body
    assert "## CRA Compliance" not in body and "Non-compliant" not in body
    assert body.rstrip().endswith("**Quantum Readiness:** 20% (5 safe / 25 total algorithms)")


def test_the_workflow_pins_each_tool_and_cdxgen_cannot_fail_the_build() -> None:
    text = CI.read_text(encoding="utf-8")
    jobs = yaml.safe_load(text)["jobs"]
    for tool, (version, sha) in PINS.items():
        assert re.search(rf"releases/download/{re.escape(version)}/", text), tool
        assert sha in text, f"{tool}: its SHA-256 is not checked"
    assert jobs["cdxgen"]["continue-on-error"] is True
    sbom = "\n".join(s.get("run", "") for s in jobs["sbom"]["steps"])
    assert "--offline" in sbom and "sha256sum -c" in sbom and "scripts/compare_sbom_tools.py" in sbom
    assert "--custom schemas/sbom-utility/qsh-cbom-rules.json" in sbom
    assert jobs["sbom"]["steps"][-1]["with"]["retention-days"] == 90
    assert "sbom-tools-cbom-view.md" in jobs["sbom"]["steps"][-1]["with"]["path"], "the whole view is kept as an artefact"


def test_the_rules_file_uses_only_what_sbom_utility_has() -> None:
    rules = json.loads(RULES.read_text(encoding="utf-8"))["validation"]["actions"]
    assert [r["id"] for r in rules] == ["qsh-cbom-document", "qsh-cbom-application", "qsh-cbom-every-asset",
                                        "qsh-cbom-the-application-uses-the-assets", "qsh-cbom-the-combiner-names-its-inputs"]
    for r in rules:
        assert set(r["functions"]) <= {"hasProperties", "isUnique"} and "path" in r["selector"], r["id"]
        for p in r.get("properties", []):
            if "value" in p:
                re.compile(p["value"])


def test_nothing_from_sbom_tools_is_on_the_site() -> None:
    served = [REPO / "web" / "index.html", *(REPO / "web" / "src").rglob("*"), *(REPO / "web" / "public" / "cbom").rglob("*"),
              *(REPO / "web" / "public" / "licenses").rglob("*")]
    for path in served:
        if path.is_file() and path.suffix in {".ts", ".tsx", ".js", ".json", ".html", ".css", ".md", ".txt"}:
            assert "sbom-tools" not in path.read_text(encoding="utf-8", errors="replace").lower(), path
