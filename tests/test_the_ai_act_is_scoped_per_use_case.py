"""What the EU AI Act asks of a use case is stated per use case, with its own grounds and date.

WHY THIS EXISTS

The SHAP panel printed "for EU AI Act Art. 13, applicable 2 Dec 2027" on all five use-case pages, and
it was right for none of them (Round 69, read at the primary sources):
- Art. 13 binds high-risk systems only. Finance is not one: Annex III 5(b) covers the
  creditworthiness and credit scoring of natural persons, fraud detection excepted, and AML
  monitoring is not creditworthiness. Telecom is not one either: recital 55 says a component used
  solely for cybersecurity is not a safety component, so Annex III 2 does not reach an intrusion
  detector.
- Medical and genomics would be high-risk, through Art. 6(1) and Annex I: diagnostic software is MDR
  class IIa or higher (Rule 11), and polygenic risk software is IVDR class C (Rule 3). Their date
  under Regulation (EU) 2026/1744 is 2 August 2028, the Annex I one, not the Annex III one.

WHAT IS ASSERTED

- Behaviourally, through aiActScope.ts bundled with the real regulatory clock: there is one entry per
  use case; the two Annex I cases carry the clock's Annex I date and say so on the page; finance and
  telecom say Art. 13 does not bind them in the deployment shown, and name the deployment that changes
  that (law-enforcement profiling, Annex III 6(e); defence-only use, outside the Act under
  Art. 2(3)); satellite is stated as depending on deployment. Every sentence is followed by the
  not-legal-advice caveat, by the user's decision before the merge.
- The clock still dates the two deferrals as Regulation (EU) 2026/1744 set them.
- The panel takes its sentence from the module, and no blanket Art. 13 date is hard-coded in it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
LIB = WEB / "src" / "lib"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"

USE_CASES = {"finance_aml", "genomics", "medical_fl", "satellite_comms", "telecom_ids"}


def _scope() -> dict:
    if not ESBUILD.exists():
        msg = "web/node_modules/.bin/esbuild is absent; run `npm ci` in web/. This is a SKIP rather than a pass."
        if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
            pytest.fail(msg + " QSH_REQUIRE_BROWSER_GATES=1 is set, so this is a FAILURE.")
        pytest.skip(msg)
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "scope.mjs"
        subprocess.run([str(ESBUILD), str(LIB / "aiActScope.ts"), "--bundle", "--format=esm", "--platform=node",
                        f"--outfile={bundle}", "--log-level=error"], cwd=WEB, check=True, capture_output=True)
        out = subprocess.run(["node", "--input-type=module", "-e",
                              f"const m = await import({json.dumps(bundle.as_uri())});"
                              "console.log(JSON.stringify({ scope: m.AI_ACT_SCOPE, i: m.ANNEX_I_DATE, iii: m.ANNEX_III_DATE,"
                              " caveat: m.AI_ACT_CAVEAT }));"],
                             check=True, capture_output=True, text=True)
        return json.loads(out.stdout)


def test_every_use_case_has_its_own_grounds_and_date() -> None:
    res = _scope()
    scope = res["scope"]
    assert set(scope) == USE_CASES, f"one entry per use case, no more: {sorted(scope)}"
    assert (res["i"], res["iii"]) == ("2028-08-02", "2027-12-02")
    for uc in ("medical_fl", "genomics"):
        e = scope[uc]
        assert e["highRisk"] == "yes" and e["from"] == res["i"], f"{uc} is high-risk through Annex I"
        assert "Annex I" in e["text"] and "2 August 2028" in e["text"], e["text"]
    assert "MDR" in scope["medical_fl"]["basis"] and "Rule 11" in scope["medical_fl"]["basis"]
    assert "IVDR" in scope["genomics"]["basis"] and "Rule 3" in scope["genomics"]["basis"]
    for uc, ground in (("finance_aml", "5(b)"), ("telecom_ids", "recital 55")):
        e = scope[uc]
        assert e["highRisk"] == "no" and e["from"] is None, f"{uc} is not high-risk in the deployment shown"
        assert "does not bind" in e["text"] and ground in e["text"], e["text"]
    fin = scope["finance_aml"]["text"]
    assert "Annex III 6(e)" in fin and "2 December 2027" in fin, (
        "finance must say that law-enforcement profiling would make it high-risk, and from when")
    tel = scope["telecom_ids"]["text"]
    assert "reading recital 55 gives" in tel, "a recital is a reading, not the rule; attribute it"
    assert "Art. 2(3)" in tel and not re.search(r"20\d\d", tel), "telecom: defence-only use is outside the Act, no date"
    sat = scope["satellite_comms"]
    assert sat["highRisk"] == "depends" and "depends on how it is deployed" in sat["text"]
    assert "Art. 2(3)" in sat["text"] and "2 December 2027" in sat["text"]
    assert res["caveat"] == "A reading of the Act for this demonstration, not legal advice." 


def test_the_clock_dates_the_deferrals_as_the_omnibus_set_them() -> None:
    clock = (LIB / "regulatoryClock.ts").read_text()
    assert re.search(r'short: "Art\. 13 \(deferred\)", when: "2027-12-02"', clock)
    assert re.search(r'short: "Annex I \(deferred\)", when: "2028-08-02"', clock)
    assert "Regulation (EU) 2026/1744" in clock


def test_the_panel_takes_its_sentence_from_the_module() -> None:
    panel = (WEB / "src" / "components" / "ShapPanel.tsx").read_text()
    code = "\n".join(line for line in panel.splitlines() if not line.lstrip().startswith("//"))
    assert 'import { AI_ACT_CAVEAT, AI_ACT_SCOPE } from "../lib/aiActScope";' in code
    assert "{aiAct.text} {AI_ACT_CAVEAT}" in code, "every sentence carries the caveat on the page"
    assert not re.search(r"Art\. 13, applicable|2 Dec 2027|2027-12-02", code), (
        "a blanket Art. 13 date is back in the panel; it differs by use case")
    page = (WEB / "src" / "pages" / "UseCasePage.tsx").read_text()
    assert re.search(r"<ShapPanel[^>]*usecase=\{usecase\}", page, re.S), "the page no longer tells the panel its use case"


def test_no_tracked_text_calls_aml_high_risk_or_dates_art_13_without_its_annex() -> None:
    """The family, not the one copy: the blanket claim had four more copies outside the panel."""
    files = subprocess.run(["git", "ls-files", "README.md", "docs/*.md", "src/**/*.py", "web/src/**/*.ts",
                            "web/src/**/*.tsx", "report/*.tex"], cwd=REPO, check=True, capture_output=True,
                           text=True).stdout.split()
    # The register and the log record what was wrong, in its own words.
    files = [f for f in files if f not in ("docs/deliverable-verification.md", "docs/verification-log.md")]
    assert len(files) > 50, f"only {len(files)} files scanned; the listing is looking at nothing"
    bad = []
    for f in files:
        for n, line in enumerate((REPO / f).read_text(errors="replace").splitlines(), 1):
            if re.search(r'(?<!")credit/AML(?!")', line):
                bad.append(f"{f}:{n}: lists AML as high-risk")
            if re.search(r"Art(icle|\.) 13", line):
                if re.search(r"2 Dec(ember)? 2027|2027-12-02", line) and "Annex III" not in line:
                    bad.append(f"{f}:{n}: dates Art. 13 to 2 Dec 2027 without saying it is the Annex III date")
                if re.search(r"2 Aug(ust)? 2028|2028-08-02", line) and not re.search(r"Annex I\b", line):
                    bad.append(f"{f}:{n}: dates Art. 13 to 2 Aug 2028 without saying it is the Annex I date")
    assert not bad, "\n".join(bad)
