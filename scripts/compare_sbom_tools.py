#!/usr/bin/env python
"""sbom-tools' NIST PQC reading of the CBOM beside this repository's own, asset by asset (Round 86, 2026-11-23).

The owner's choice of 2026-11-23: sbom-tools (MIT, v0.4.0 pinned in CI) reads `web/public/cbom/qsh.cdx.json` offline
with `validate --standard pqc --pqc-mode deadline --as-of <date>`, and its findings are compared with the reading the
Validations page shows (`web/src/lib/ir8547.ts`, whose Python twin is `qsh.pqc.ir8547`). Where the two differ, the
difference is reviewed by hand and the note is written in REVIEWED below; a difference with no note fails, so nothing
sbom-tools says is adopted automatically, and nothing it says is shown on the site.

Each asset's sbom-tools verdict is read from its findings: SBOM-PQC-009 (approved post-quantum algorithm) is
"post-quantum", SBOM-PQC-001 (quantum-vulnerable, with a deadline where it knows one) is "quantum-vulnerable",
SBOM-PQC-ALG-UNKNOWN is "unclassified", and an asset with only SBOM-PQC-012 (no nistQuantumSecurityLevel field) or no
finding at all has "no verdict". This repository leaves that field out where it does not rate an asset, so the
field-missing warnings are counted but not compared.

    python scripts/compare_sbom_tools.py <sbom-tools-pqc.json>                    # check; exit 1 on an unreviewed difference
    python scripts/compare_sbom_tools.py <sbom-tools-pqc.json> --write <record>   # also write the comparison record
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qsh.pqc.ir8547 import BY_REF, standing_on  # noqa: E402

CBOM = REPO / "web" / "public" / "cbom" / "qsh.cdx.json"

#: Each difference between the two readings, reviewed by hand on 2026-11-23, and why the repository keeps its own.
REVIEWED = {
    "crypto/hybrid-combiner": (
        "sbom-tools finds X25519 in the combiner's name and dates the whole combiner as quantum-vulnerable (Table 4). "
        "IR 8547 section 3.2.1 allows a composite whose other input comes from an approved scheme even when one input is "
        "disallowed, and against a quantum attacker this one rests on ML-KEM; the page shows the combination and both "
        "parts. Kept."),
    "crypto/oprf-ristretto255": (
        "Both readings call it quantum-vulnerable. sbom-tools dates it as elliptic-curve DH (Table 4, deprecated after "
        "2030); IR 8547 does not classify oblivious PRFs, so the page gives it no dates. Kept."),
    "crypto/wrk17b": (
        "Both readings call it quantum-vulnerable. sbom-tools, finding no algorithm it knows, says no transition applies "
        "and makes it an error; IR 8547 does not classify secure computation, so the page gives it no dates. Kept."),
}


VERDICT_RULES = (("SBOM-PQC-009", "post-quantum"), ("SBOM-PQC-001", "quantum-vulnerable"), ("SBOM-PQC-ALG-UNKNOWN", "unclassified"))


def tool_verdicts(report: dict) -> dict[str, dict]:
    """Per component: the verdict, the finding that decided it (its severity and any deadline), and every rule that fired."""
    result = next(r for r in report["results"] if r["standard"] == "pqc")
    found: dict[str, dict[str, dict]] = {}
    for kind, findings in (("violation", result.get("violations", [])), ("passed", result.get("passed", []))):
        for f in findings:
            if f.get("component_id"):
                found.setdefault(f["component_id"], {})[f["rule_id"]] = {
                    "severity": f.get("severity") if kind == "violation" else "Pass", "deadline": f.get("deadline")}
    out: dict[str, dict] = {}
    for ref, by_rule in found.items():
        rule, verdict = next(((r, v) for r, v in VERDICT_RULES if r in by_rule), (None, "no verdict"))
        decided = by_rule.get(rule, {}) if rule else {}
        out[ref] = {"verdict": verdict, "rule": rule, "severity": decided.get("severity"), "deadline": decided.get("deadline"),
                    "rules": sorted(by_rule)}
    return out


def in_house(ref: str, as_of: str) -> dict:
    r = BY_REF[ref]
    standing, label = standing_on(r, as_of)
    return {"class": r.cls, "basis": r.basis, "quantum_vulnerable": r.quantum_vulnerable, "standing": standing, "label": label,
            "deprecated_after": r.deprecated_after}


def agreement(ours: dict, theirs: dict) -> str:
    verdict = theirs["verdict"]
    if verdict == "no verdict":
        return "no verdict"
    if verdict == "post-quantum":
        return "agree" if ours["class"] == "post-quantum" else "differ"
    if verdict == "unclassified":
        return "agree" if ours["class"] == "outside" and not ours["quantum_vulnerable"] else "differ"
    # quantum-vulnerable: the same class, and the same first deadline where sbom-tools gives one.
    if ours["class"] != "quantum-vulnerable":
        return "differ"
    deadline = theirs["deadline"]
    return "agree" if deadline is None or deadline == f"{ours['deprecated_after']}-12-31" else "differ"


def compare(report: dict) -> dict:
    as_of = report["as_of"][:10]
    cbom = json.loads(CBOM.read_text(encoding="utf-8"))
    tool = tool_verdicts(report)
    assets = []
    for c in cbom["components"]:
        ref = c["bom-ref"]
        theirs = tool.get(ref, {"verdict": "no verdict", "rule": None, "severity": None, "deadline": None, "rules": []})
        ours = in_house(ref, as_of)
        a = agreement(ours, theirs)
        row = {"ref": ref, "name": c["name"], "in_house": ours, "sbom_tools": theirs, "agreement": a}
        if a == "differ":
            row["reviewed"] = REVIEWED.get(ref)
        assets.append(row)
    counts = {k: sum(1 for x in assets if x["agreement"] == k) for k in ("agree", "differ", "no verdict")}
    return {
        "what": "sbom-tools' NIST PQC reading of the CBOM beside the Validations page's, asset by asset; dated tool output, "
                "not a compliance statement",
        "tool": report["tool"],
        "as_of": as_of,
        "cbom_sha256": hashlib.sha256(CBOM.read_bytes()).hexdigest(),
        "tool_read_sha256": report["subject"]["sha256"],
        "tool_summary": {k: report["results"][0][k] for k in ("score", "counts")},
        "assets": assets,
        "counts": counts,
        "unreviewed": [x["ref"] for x in assets if x["agreement"] == "differ" and not x.get("reviewed")],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", type=Path, help="sbom-tools validate --standard pqc -o json output (compliance/v2)")
    ap.add_argument("--write", type=Path, help="write the comparison record here")
    a = ap.parse_args()
    report = json.loads(a.report.read_text(encoding="utf-8"))
    if report.get("schema") != "sbom-tools/compliance/v2":
        print(f"not an sbom-tools compliance/v2 report: {report.get('schema')!r}", file=sys.stderr)
        return 2
    record = compare(report)
    if record["tool_read_sha256"] != record["cbom_sha256"]:
        print("sbom-tools read a different CBOM from the one in this tree", file=sys.stderr)
        return 1
    if a.write:
        a.write.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    c = record["counts"]
    print(f"sbom-tools {record['tool']['version']} as of {record['as_of']}: {c['agree']} agree, {c['differ']} differ "
          f"({len(record['unreviewed'])} unreviewed), {c['no verdict']} without a verdict from sbom-tools")
    for x in record["assets"]:
        if x["agreement"] == "differ":
            print(f"  differ: {x['name']}: sbom-tools {x['sbom_tools']['verdict']} ({x['sbom_tools']['severity']}), "
                  f"this repository {x['in_house']['class']}; {'reviewed' if x.get('reviewed') else 'NOT REVIEWED'}")
    if record["unreviewed"]:
        print(f"unreviewed differences: {record['unreviewed']}; review each by hand and add a note to REVIEWED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
