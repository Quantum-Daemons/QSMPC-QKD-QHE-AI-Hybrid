"""The Jinan-1 pass's second analysis (Round 86, batch 8; checklist MS28), held to its pre-registration and its record.

`references/satellite-jinan1-pass-v2-preregistration.md` fixed four questions and two rules before the run;
`scripts/measure_jinan1_pass_v2.py` wrote `docs/measurements/satellite-jinan1-pass-v2-2026-11-27.json` and the page's
`web/public/data/satellite/jinan1_pass_v2.json`. The source data are not in CI, so what is held here is what the
committed files say about each other:

  - the record ran under the committed pre-registration (its SHA-256 is the file's and the script's);
  - each rule's verdict is what its own stated thresholds give on the record's numbers, so neither verdict can drift from
    the figures beside it (rule 1: the joint fit must lower the residual deviation by 0.05 dB with the zenith loss in
    0 to 3 dB; rule 2: every packet ratio within a factor of 1.5, judged with the set rule 1 chose);
  - the first analysis it cites is the committed first record, figure for figure;
  - the page file is the record's sections, unchanged, with the corpus's credit; the satellite card mounts the section.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "measure_jinan1_pass_v2.py"
RECORD = REPO / "docs" / "measurements" / "satellite-jinan1-pass-v2-2026-11-27.json"
SITE = REPO / "web" / "public" / "data" / "satellite" / "jinan1_pass_v2.json"
SECTIONS = ("q1_atmosphere", "q2_sifted_bits", "q3_keys_by_block", "q4_other_passes", "rules", "first_analysis")


def _constant(name: str):
    """A module-level constant of the script, read without running it; `A, B = x, y` assignments included."""
    for node in ast.parse(SCRIPT.read_text()).body:
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if getattr(t, "id", None) == name:
                return ast.literal_eval(node.value)
            if isinstance(t, ast.Tuple) and isinstance(node.value, ast.Tuple):
                for el, val in zip(t.elts, node.value.elts, strict=True):
                    if getattr(el, "id", None) == name:
                        return ast.literal_eval(val)
    raise KeyError(name)


def _record() -> dict:
    return json.loads(RECORD.read_text())


def test_the_record_ran_under_the_committed_preregistration() -> None:
    r = _record()
    prereg = REPO / r["preregistration"]["file"]
    assert hashlib.sha256(prereg.read_bytes()).hexdigest() == r["preregistration"]["sha256"] == _constant("PREREG_SHA256")


def _verdicts(r: dict) -> tuple[bool, str, dict, bool]:
    """The two rules as the pre-registration states them, on a record's numbers."""
    sd_db, (zlo, zhi), factor = _constant("RULE_1_SD_DB"), _constant("RULE_1_ZENITH_DB"), _constant("RULE_2_FACTOR")
    q1 = r["q1_atmosphere"]
    rule1 = q1["sd_change_db"] <= -sd_db and zlo <= q1["fitted_zenith_loss_db"] <= zhi
    which = "fitted" if rule1 else "profile"
    ratios = {"2022-09-25": r["q2_sifted_bits"][which]["model_over_measured"],
              **{p["date"]: p[which]["packets_ratio"] for p in r["q4_other_passes"]["passes"]}}
    return rule1, which, ratios, all(1 / factor <= x <= factor for x in ratios.values())


def test_each_rule_verdict_follows_from_the_record_numbers() -> None:
    r = _record()
    q1 = r["q1_atmosphere"]
    assert abs(q1["sd_change_db"] - (q1["residual_sd_db"] - q1["first_analysis_one_parameter_sd_db"])) < 1e-12
    rule1, which, ratios, rule2 = _verdicts(r)
    assert r["rules"]["1_profile_takes_the_fitted_pair"] is rule1
    assert r["rules"]["2_judged_with"] == which
    assert ratios == r["rules"]["2_ratios"]
    assert r["rules"]["2_key_model_gains_the_factors"] is rule2
    # The other passes are the preprint's Table 1, as the script states them.
    table = {d: (e, p, b) for d, e, p, b in _constant("OTHER_PASSES")}
    for p in r["q4_other_passes"]["passes"]:
        assert table[p["date"]] == (p["max_elevation_deg"], p["published_packets"], p["published_key_bits"])
        assert abs(p[which]["packets_ratio"] - p[which]["model_packets"] / p["published_packets"]) < 1e-12


def test_the_first_analysis_it_cites_is_the_committed_one() -> None:
    first = _record()["first_analysis"]
    committed = json.loads((REPO / first["record"]).read_text())
    assert first["fitted_efficiency"] == committed["fitted_efficiency"]
    assert first["residuals_after"] == committed["residuals_after"]
    assert _record()["q1_atmosphere"]["first_analysis_one_parameter_sd_db"] == committed["residuals_after"]["sd_db"]


def test_the_page_file_is_the_record_sections_with_the_credit() -> None:
    r, s = _record(), json.loads(SITE.read_text())
    for k in SECTIONS:
        assert s[k] == r[k], k
    assert s["record"] == str(RECORD.relative_to(REPO)) and s["preregistration"] == r["preregistration"]["file"]
    assert s["licence"] == "CC-BY-4.0" and s["attribution"]["citation"] == r["attribution"]
    assert "no per-second row" in s["modifications"]


def test_the_satellite_card_mounts_the_section() -> None:
    card = (REPO / "web" / "src" / "components" / "SatJinanPassCard.tsx").read_text()
    section = (REPO / "web" / "src" / "components" / "SatJinanV2Section.tsx").read_text()
    assert "<SatJinanV2Section />" in card and "data/satellite/jinan1_pass_v2.json" in section


def test_the_verdict_check_would_notice_a_different_outcome() -> None:
    """Seeded: a fall of 0.06 dB passes rule 1, and every ratio at 1.2 passes rule 2; the record's verdicts are both no."""
    r = json.loads(RECORD.read_text())
    r["q1_atmosphere"]["sd_change_db"] = -0.06
    assert _verdicts(r)[0] is True and _verdicts(r)[0] is not _record()["rules"]["1_profile_takes_the_fitted_pair"]
    r = _record()
    r["q2_sifted_bits"]["profile"]["model_over_measured"] = 1.2
    for p in r["q4_other_passes"]["passes"]:
        p["profile"]["packets_ratio"] = 1.2
    assert _verdicts(r)[3] is True and _record()["rules"]["2_key_model_gains_the_factors"] is False
