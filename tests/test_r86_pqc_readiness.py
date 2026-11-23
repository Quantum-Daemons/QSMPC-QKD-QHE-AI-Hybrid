"""The CBOM panel's PQC-readiness report says what NIST IR 8547 says, the same way in both languages (Round 86, 2026-11-23).

WHY THIS EXISTS

The owner's choice of 2026-11-23: the Validations page's CBOM panel reads every asset of `web/public/cbom/qsh.cdx.json`
against NIST IR 8547 (final, October 2026, doi:10.6028/NIST.IR.8547), naming the table or section each row rests on
and what IR 8547 says of it on a date the reader picks. A report like that is only worth its transcription and its
dates, so this file holds:

1. one rule per CBOM asset in `web/src/lib/ir8547.ts` (the page's, bundled by esbuild and run in node) and in its
   Python twin `qsh.pqc.ir8547`, field for field, and the same standing in both on 2026-11-23, 2030-12-31, 2031-01-01,
   2035-12-31 and 2036-01-01;
2. the rows to IR 8547's own tables, transcribed below from the final PDF read on 2026-11-23 (Tables 3, 4, 5, 6, 8),
   and the dates to its words: "deprecated after 2030" starts on 2031-01-01 and "disallowed after 2035" on 2036-01-01;
3. the CBOM's own quantum field to the rules (a level 0 asset is quantum-vulnerable; a rated one has the table's
   category);
4. the hybrid read by both of its parts, from the document's own dependency graph: ML-KEM (Table 5), X25519
   (quantum-vulnerable, with section 4.1.2's dates) and the combination, allowed as a composite and not validated;
5. every asset outside IR 8547's tables saying so, and the report and the page claiming no certification;
6. the Markdown and CSV downloads carrying every row, the date and the served file's SHA-256, which the page computes
   as Python does;
7. the combiner described by its form, as the owner chose on 2026-11-23 ("in the form of SP 800-227 s4.6.2's two-step
   combiner, over the s4.6.3 transcript; not validated"), with no "approves" or "conformance" left in its descriptions.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from qsh.pqc import ir8547 as py

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
CBOM = WEB / "public" / "cbom" / "qsh.cdx.json"
DAYS = ("2026-11-23", "2030-12-31", "2031-01-01", "2035-12-31", "2036-01-01")
FIELDS = (("cls", "cls"), ("basis", "basis"), ("category", "category"), ("preimageCategory", "preimage_category"),
          ("classicalBits", "classical_bits"), ("named", "named"), ("deprecatedAfter", "deprecated_after"),
          ("disallowedAfter", "disallowed_after"), ("quantumVulnerable", "quantum_vulnerable"))

# NIST IR 8547 (final, October 2026), read on the PDF on 2026-11-23: (classical bits, security category).
TABLE_3 = {"ML-DSA-44": (128, 2), "ML-DSA-65": (192, 3), "ML-DSA-87": (256, 5)}
TABLE_5 = {"ML-KEM-512": (128, 1), "ML-KEM-768": (192, 3), "ML-KEM-1024": (256, 5)}
TABLE_6 = {"AES-128": (128, 1), "AES-192": (192, 3), "AES-256": (256, 5)}
# Table 8: (collision bits, collision category, preimage bits, preimage category).
TABLE_8 = {"SHA-256": (128, 2, 256, 5), "SHA-384": (192, 4, 384, 5), "SHA-512": (256, 5, 512, 5), "SHAKE256": (256, 5, 256, 5)}
# Table 4's elliptic-curve row and section 4.1.2: deprecated after 2030, disallowed after 2035.
QUANTUM_VULNERABLE_DATES = (2030, 2035)
IN_TABLES = {"crypto/ml-dsa-65": ("Table 3", "ML-DSA-65"), "crypto/ml-dsa-87": ("Table 3", "ML-DSA-87"),
             "crypto/ml-kem-768": ("Table 5", "ML-KEM-768"), "crypto/ml-kem-1024": ("Table 5", "ML-KEM-1024"),
             "crypto/aes-256-gcm": ("Table 6", "AES-256"), "crypto/sha-256": ("Table 8", "SHA-256"),
             "crypto/sha-384": ("Table 8", "SHA-384"), "crypto/sha-512": ("Table 8", "SHA-512"),
             "crypto/shake256": ("Table 8", "SHAKE256")}

RUNNER = r"""
import { readFileSync } from "node:fs";
const m = await import(%(bundle)s);
const bytes = readFileSync(%(cbom)s);
const doc = JSON.parse(new TextDecoder().decode(bytes));
const days = %(days)s;
const out = { rules: m.RULES, ir8547: m.IR8547 };
out.standings = Object.fromEntries(days.map((d) => [d, Object.fromEntries(m.RULES.map((r) => [r.ref, m.standingOn(r, d)]))]));
out.rows = Object.fromEntries(days.map((d) => [d, m.reportRows(doc, d)]));
out.counts = Object.fromEntries(days.map((d) => [d, m.reportCounts(m.reportRows(doc, d))]));
out.sha = await m.sha256Hex(bytes);
out.markdown = m.reportMarkdown(doc, "2031-01-01", { file: "qsh.cdx.json", sha256: out.sha, builtSha256: out.sha });
out.markdownStale = m.reportMarkdown(doc, "2031-01-01", { file: "qsh.cdx.json", sha256: out.sha, builtSha256: "0".repeat(64) });
out.csv = m.reportCsvRows(doc, "2036-01-01");
out.badDates = ["2031/01/01", "2031-1-1", ""].map((d) => { try { m.yearOf(d); return false; } catch { return true; } });
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
    d = tmp_path_factory.mktemp("pqcreport")
    entry, bundle, runner = d / "entry.ts", d / "bundle.mjs", d / "run.mjs"
    entry.write_text(f"export * from {json.dumps(str(WEB / 'src/lib/ir8547'))};\n"
                     f"export * from {json.dumps(str(WEB / 'src/lib/cbomReport'))};\n")
    subprocess.run([str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node", f"--outfile={bundle}",
                    "--log-level=error"], cwd=WEB, check=True, capture_output=True)
    runner.write_text(RUNNER % {"bundle": json.dumps(bundle.as_uri()), "cbom": json.dumps(str(CBOM)), "days": json.dumps(DAYS)})
    proc = subprocess.run(["node", str(runner)], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout)


def _cbom() -> dict:
    return json.loads(CBOM.read_text(encoding="utf-8"))


def test_every_cbom_asset_has_one_rule_in_each_language(page: dict) -> None:
    refs = [c["bom-ref"] for c in _cbom()["components"]]
    assert [r["ref"] for r in page["rules"]] == refs, "the page's rules are not the CBOM's assets, in order"
    assert [r.ref for r in py.RULES] == refs, "the Python twin's rules are not the CBOM's assets, in order"


def test_the_two_languages_agree_on_every_row(page: dict) -> None:
    for ts in page["rules"]:
        twin = py.BY_REF[ts["ref"]]
        for ts_name, py_name in FIELDS:
            assert ts[ts_name] == getattr(twin, py_name), f"{ts['ref']}: {ts_name} {ts[ts_name]!r} vs {getattr(twin, py_name)!r}"


@pytest.mark.parametrize("day", DAYS)
def test_the_standing_agrees_on_every_date(page: dict, day: str) -> None:
    for ref, s in page["standings"][day].items():
        assert (s["standing"], s["label"]) == py.standing_on(py.BY_REF[ref], day), (day, ref)


def test_the_transition_dates_fall_where_ir_8547_puts_them(page: dict) -> None:
    st = page["standings"]
    for ref in ("crypto/ecdh-p256", "crypto/x25519"):
        before = st["2030-12-31"][ref]["standing"]
        assert before in ("approved-quantum-vulnerable", "quantum-vulnerable"), (ref, before)
        assert st["2026-11-23"][ref]["standing"] == before
        assert st["2031-01-01"][ref]["standing"] == "deprecated", ref
        assert st["2035-12-31"][ref]["standing"] == "deprecated", ref
        assert st["2036-01-01"][ref]["standing"] == "disallowed", ref
        assert (py.BY_REF[ref].deprecated_after, py.BY_REF[ref].disallowed_after) == QUANTUM_VULNERABLE_DATES
    # ECDH P-256 is in Table 4; X25519 is not named, and its labels say whose dates they are.
    assert st["2030-12-31"]["crypto/ecdh-p256"]["label"] == "Listed as approved; deprecated after 2030"
    assert st["2030-12-31"]["crypto/x25519"]["label"] == "Not named; deprecated after 2030 (section 4.1.2)"
    assert st["2036-01-01"]["crypto/x25519"]["label"] == "Disallowed (section 4.1.2)"
    # Nothing else has a date: post-quantum, symmetric, the hybrid and the assets outside the tables stand still.
    for ref, s in st["2026-11-23"].items():
        if ref in ("crypto/ecdh-p256", "crypto/x25519"):
            continue
        assert all(st[d][ref] == s for d in DAYS), ref


def test_the_rows_are_ir_8547_s_tables(page: dict) -> None:
    rules = {r["ref"]: r for r in page["rules"]}
    for ref, (table, name) in IN_TABLES.items():
        r = rules[ref]
        assert r["basis"] == table, ref
        if table == "Table 8":
            bits, cat, pre_bits, pre_cat = TABLE_8[name]
            assert (r["classicalBits"], r["category"], r["preimageCategory"]) == (bits, cat, pre_cat), ref
            assert f"{bits} bits" in r["reading"] and f"{pre_bits} bits" in r["reading"], ref
        else:
            bits, cat = {"Table 3": TABLE_3, "Table 5": TABLE_5, "Table 6": TABLE_6}[table][name]
            assert (r["classicalBits"], r["category"]) == (bits, cat), ref
            assert f"{bits} bits" in r["reading"] and f"security category {cat}" in r["reading"], ref
    assert rules["crypto/ecdh-p256"]["basis"] == "Table 4" and "112 bits" in rules["crypto/ecdh-p256"]["reading"]
    assert "default parameter set" in rules["crypto/ml-kem-768"]["reading"], "section 4.1.2 recommends ML-KEM-768 as the default"
    allowed = {"Table 3", "Table 4", "Table 5", "Table 6", "Table 8", "Section 4.1.2", "Section 4.1.3", "Sections 3.2.1 and 4.1.2", "none"}
    assert {r["basis"] for r in page["rules"]} <= allowed


def test_the_cbom_s_own_quantum_field_agrees_with_the_rules(page: dict) -> None:
    rules = {r["ref"]: r for r in page["rules"]}
    for c in _cbom()["components"]:
        level = c["cryptoProperties"]["algorithmProperties"].get("nistQuantumSecurityLevel")
        r = rules[c["bom-ref"]]
        if level == 0:
            assert r["quantumVulnerable"], c["bom-ref"]
        elif level is not None:
            assert r["category"] == level and not r["quantumVulnerable"], c["bom-ref"]
        if r["quantumVulnerable"]:
            assert level == 0, f"{c['bom-ref']}: the rule says quantum-vulnerable and the CBOM does not"


def test_the_hybrid_is_read_by_both_parts(page: dict) -> None:
    rows = page["rows"]["2036-01-01"]
    at = next(i for i, r in enumerate(rows) if r["ref"] == "crypto/hybrid-combiner")
    combination, pq, classical = rows[at], rows[at + 1], rows[at + 2]
    assert combination["standing"] == "allowed-composite" and "not validated" in combination["standingLabel"]
    assert "In the form of SP 800-227 section 4.6.2's two-step combiner, over the section 4.6.3 transcript; not validated." in combination["reading"]
    assert "rests on ML-KEM alone" in combination["reading"]
    assert pq["partOf"] == classical["partOf"] == "crypto/hybrid-combiner"
    assert pq["name"] == "ML-KEM-768 or ML-KEM-1024, its post-quantum part" and pq["standing"] == "post-quantum"
    assert "category 3 with ML-KEM-768, 5 with ML-KEM-1024" in pq["reading"]
    assert classical["name"] == "X25519, its classical part" and classical["standing"] == "disallowed"
    assert "Section 3.2.1 allows the composite even when this part is disallowed" in classical["reading"]
    assert "combined by HKDF-SHA256" in classical["reading"], "the KDF is neither part, and the row names it"
    assert page["rows"]["2030-12-31"][at + 2]["standing"] == "quantum-vulnerable"
    assert sum(1 for r in rows if r.get("partOf")) == 2 and len(rows) == len(_cbom()["components"]) + 2


def test_every_asset_outside_the_tables_says_so(page: dict) -> None:
    outside = [r for r in page["rules"] if r["basis"] == "none"]
    assert {r["cls"] for r in outside} == {"outside"} and len(outside) == 10
    for r in outside:
        assert r["reading"].startswith("IR 8547 does not "), r["ref"]
        assert not r["named"], r["ref"]
    assert {r["ref"] for r in outside if r["quantumVulnerable"]} == {"crypto/oprf-ristretto255", "crypto/wrk17b"}
    for ref in ("crypto/ckks", "crypto/ckks-threshold", "crypto/bgv-threshold", "crypto/bgv-threshold-committee"):
        assert "post-quantum limit for 128-bit security, which the build checks" in next(r for r in outside if r["ref"] == ref)["reading"]
    counts = {c["standing"]: c["count"] for c in page["counts"]["2026-11-23"]}
    assert counts == {"post-quantum": 4, "symmetric": 7, "approved-quantum-vulnerable": 1, "quantum-vulnerable": 1,
                      "allowed-composite": 1, "not-classified": 10}, counts
    assert {c["standing"]: c["count"] for c in page["counts"]["2036-01-01"]}["disallowed"] == 2


def test_the_downloads_carry_every_row_the_date_and_the_digest(page: dict) -> None:
    raw = CBOM.read_bytes()
    assert page["sha"] == hashlib.sha256(raw).hexdigest(), "the page's SHA-256 of the served file is not the file's"
    md = page["markdown"]
    doc = _cbom()
    assert f"- SHA-256 of the file read: {page['sha']} (the file this build was made from)" in md
    assert "NOT the file this build was made from" in page["markdownStale"]
    assert "- Standing on: 2031-01-01" in md and doc["serialNumber"] in md
    assert "It is not a certification, and nothing here is FIPS 140-3 validated." in md
    assert page["ir8547"]["url"] in md
    for c in doc["components"]:
        assert f"| {c['name']} |" in md, c["name"]
    assert "| - X25519, its classical part |" in md
    table = md.split("## Assets", 1)[1].splitlines()
    cells = [len(re.findall(r"(?<!\\)\|", line)) for line in table if line.startswith("| ")]
    assert cells and set(cells) == {9}, "an asset row has a stray or a missing cell"
    csv = page["csv"]
    assert len(csv) == len(doc["components"]) + 2
    assert {r["standing_on"] for r in csv} == {"2036-01-01"}
    assert list(csv[0]) == ["bom_ref", "part_of", "name", "kind", "ir8547_basis", "ir8547_class", "reading", "standing_on", "standing",
                            "standing_label", "standard", "used_for", "scope"]
    assert page["badDates"] == [True, True, True]
    for bad in ("2031/01/01", "2031-1-1", ""):
        with pytest.raises(ValueError):
            py.year_of(bad)


def test_the_cbom_gives_the_combiner_its_parts_and_claims_no_certification() -> None:
    doc = _cbom()
    refs = {c["bom-ref"] for c in doc["components"]}
    deps = {d["ref"]: d.get("dependsOn", []) for d in doc["dependencies"]}
    assert sorted(deps[doc["metadata"]["component"]["bom-ref"]]) == sorted(refs)
    assert deps["crypto/hybrid-combiner"] == ["crypto/ml-kem-768", "crypto/ml-kem-1024", "crypto/x25519", "crypto/hkdf-sha256"]
    assert set(deps) == {doc["metadata"]["component"]["bom-ref"], "crypto/hybrid-combiner"}, "only the combiner's own code combines assets"
    for c in doc["components"]:
        assert c["cryptoProperties"]["algorithmProperties"]["certificationLevel"] == ["none"], c["bom-ref"]
    combiner = next(c for c in doc["components"] if c["bom-ref"] == "crypto/hybrid-combiner")
    assert {p["name"]: p["value"] for p in combiner["properties"]}["qsh:standard"] == "In the form of NIST SP 800-227 section 4.6.2; not validated"


def test_the_combiner_is_described_by_its_form() -> None:
    """The owner's wording of 2026-11-23: the combiner has the form of SP 800-227 s4.6.2's two-step combiner over the s4.6.3
    transcript, and no CAVP or CMVP validation, so nothing describing it may say a section "approves" it or call the project's
    claim conformance."""
    texts = {p: (REPO / p).read_text(encoding="utf-8") for p in (
        "src/qsh/pqc/hybrid_kem.py", "web/src/lib/hybridCombiner.ts", "web/src/lib/crypto.ts", "web/src/lib/clientCompute.ts",
        "src/qsh/crypto_inventory.py", "web/src/lib/ir8547.ts", "docs/pqc-roadmap.md")}
    pinned = {
        "src/qsh/pqc/hybrid_kem.py": ["The combiner is in the form of NIST SP 800-227 section 4.6.2's\ntwo-step combiner",
                                      "It is not validated (it has no CAVP or CMVP certificate)",
                                      "in the form of section 4.6.2's two-step combiner; not validated."],
        "web/src/lib/hybridCombiner.ts": ["in the form of SP 800-227 section 4.6.2's two-step combiner (equations 13 and 14); not\n// validated."],
        "web/src/lib/crypto.ts": ["in the form of SP 800-227 s4.6.2's two-step combiner, over the s4.6.3 transcript; not validated"],
        "web/src/lib/clientCompute.ts": ["in the form of NIST SP 800-227 s4.6.2's two-step combiner, over the s4.6.3 transcript; not validated"],
        "src/qsh/crypto_inventory.py": ['"In the form of NIST SP 800-227 section 4.6.2; not validated"'],
        "web/src/lib/ir8547.ts": ["In the form of SP 800-227 section 4.6.2's two-step combiner, over the section 4.6.3 transcript; not validated."],
        "docs/pqc-roadmap.md": ["is in the form of section 4.6.2's two-step KDF (equations 13-14)", "and it is not validated."],
    }
    for path, sentences in pinned.items():
        for s in sentences:
            assert s in texts[path], f"{path} no longer says {s!r}"
    banned = re.compile(r"4\.6\.2[^.\n]{0,40}approv|approves: an|which that section approves|approved two-step form|standards-conformance|"
                        r"a conformance and claim-accuracy defect|s4\.6\.2 two-step combiner over", re.I)
    for path, text in texts.items():
        assert not banned.search(text), f"{path}: {banned.search(text).group(0)!r}"


def test_the_panel_renders_the_report_and_claims_no_certification() -> None:
    panel = (WEB / "src" / "components" / "CbomPanel.tsx").read_text(encoding="utf-8")
    graph = (WEB / "src" / "components" / "CbomGraph.tsx").read_text(encoding="utf-8")
    for needed in ("reportRows(", "reportMarkdown(", "reportCsvRows(", "sha256Hex(", "__CBOM_SHA256__", "<CbomGraph",
                   "not a certification", "2030-12-31", "2031-01-01", "2036-01-01", 'type="date"'):
        assert needed in panel, needed
    assert "doc.dependencies" in graph, "the graph must be drawn from the document's own dependencies"
    for text in (panel, graph, (WEB / "src" / "lib" / "cbomReport.ts").read_text(encoding="utf-8"),
                 (WEB / "src" / "lib" / "ir8547.ts").read_text(encoding="utf-8")):
        assert not re.search(r"\bcomplian|\bconformant|\bcertified\b|\bconforms\b", text, re.I)
