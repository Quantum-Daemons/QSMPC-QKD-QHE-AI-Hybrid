"""Three checks against outside data reach their pages from their records (Round 86, batch 13).

The owner's choice of 2026-11-28: the BUS-BRA external test on the medical page, the SatQKD recomputation on the
satellite page and the PFCP held-out days on the telecom page. `scripts/export_outside_data_cards.py` writes each page's
file from its committed record; each card reads its file and types no figure. Held here: the files are what the script
writes from the records, their figures are the records', each carries its dataset's licence and credit, each card reads
its file and is placed on its page only, and none ships an image, a flow, a row or a per-patient figure.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
PAGE = (WEB / "src" / "pages" / "UseCasePage.tsx").read_text()
CARDS = {
    "BusbraExternalCard": ("medical_fl", "data/medical/busbra_external.json"),
    "SatqkdCheckCard": ("satellite_comms", "data/satellite/satqkd_check.json"),
    "PfcpHeldOutDayCard": ("telecom_ids", "data/telecom/pfcp_held_out_day.json"),
}


def _page(rel: str) -> dict:
    return json.loads((WEB / "public" / rel).read_text())


def _record(name: str) -> dict:
    return json.loads((REPO / "docs" / "measurements" / name).read_text())


def test_each_page_file_is_what_the_script_writes_from_its_record() -> None:
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "export_outside_data_cards.py"), "--check"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_figures_are_the_records() -> None:
    b, rec = _page("data/medical/busbra_external.json"), _record("medical-busbra-external-2026-11-28.json")
    assert b["images"] == rec["images"] == 1875 and b["patients"] == rec["patients"]
    for m in b["models"]:
        r = rec["models"][m["key"]]
        assert m["busbra_auc"] == round(r["busbra"]["auc"], 4) and m["breastmnist_auc"] == round(r["breastmnist_test_same_graph"]["auc"], 4)
    s, a = _page("data/satellite/satqkd_check.json"), _record("satellite-satqkd-key-length-2026-11-28.json")
    assert s["check_a"]["meets"] is a["check_a"]["meets"] is True
    assert s["check_a"]["keyed_equal"] == s["check_a"]["rows_with_a_key"]
    assert s["check_b_found_a_key"] is False and len(s["check_b_prime"]) == 4
    # The second addendum's Check B'' (SatQuMA's differences one at a time), each ratio its record's.
    c = _record("satellite-satqkd-key-length-b-double-prime-2026-11-28.json")
    assert c["check_b_prime_reproduced"] is True and len(s["addenda"]) == 2
    by = {(x["file"], x["dt_s"]): x["variants"] for x in c["check_b_double_prime"]["rows"]}
    for r in s["check_b_prime"]:
        v = by[(r["file"], r["half_window_s"])]
        assert r["key_ratio"] == round(v["b_prime"]["key_ratio"], 4)
        assert r["with_cap_key_ratio"] == round(v["a"]["key_ratio"], 4)
        assert r["with_cap_and_satellite_statistics_key_ratio"] == round(v["b"]["key_ratio"], 4)
        assert r["with_cap_statistics_and_slots_key_ratio"] == round(v["c"]["key_ratio"], 4)
        assert v["a"]["cap_single_photon_errors"] and not v["b_prime"]["cap_single_photon_errors"]
        assert v["c"]["window"] == "satquma_slots" and v["c"]["deviation"] == "chernoff"


def test_the_satellite_card_says_satquma_caps_and_the_page_does_not() -> None:
    """The owner's choice of 2026-11-28: the page keeps the papers' bound, the cap is a named option, and the card says so."""
    card = (REPO / "web" / "src" / "components" / "SatqkdCheckCard.tsx").read_text(encoding="utf-8")
    assert "SatQuMA caps; this page does not." in card
    assert "not resolved here" not in card
    for key in ("with_cap_key_ratio", "with_cap_and_satellite_statistics_key_ratio", "with_cap_statistics_and_slots_key_ratio"):
        assert f"r.{key}" in card, key
    p, pr = _page("data/telecom/pfcp_held_out_day.json"), _record("telecom-pfcp-held-out-day-2026-11-28.json")
    assert len(p["days"]) == 4 == len(pr["held_out_days"])
    assert sum(d["flows"] for d in p["days"]) == sum(v["held_out_rows"] for v in pr["held_out_days"].values())


def test_each_file_carries_its_licence_and_credit_and_no_row() -> None:
    for rel, credit in (("data/medical/busbra_external.json", "doi:10.1002/mp.16812"),
                        ("data/satellite/satqkd_check.json", "Zenodo 8101679"),
                        ("data/telecom/pfcp_held_out_day.json", "doi:10.21227/e0gj-ev03")):
        f = _page(rel)
        assert f["licence"] == "CC-BY-4.0" and credit in f["attribution"], rel
        text = json.dumps(f)
        assert not re.search(r"bus_\d{4}|Flow ID|Src IP|\b\d{1,3}(\.\d{1,3}){3}\b", text), rel
        assert (WEB / "public" / rel).stat().st_size < 8000, rel


def test_each_card_reads_its_file_types_no_figure_and_sits_on_its_page() -> None:
    for card, (page, rel) in CARDS.items():
        src = (WEB / "src" / "components" / f"{card}.tsx").read_text()
        assert rel in src, card
        body = src[src.index("export default function"):]
        assert not re.search(r"\b0\.\d{3,}\b", body), f"{card} types a figure"
        uses = re.findall(rf'usecase === "([a-z_]+)" && <{card} />', PAGE)
        assert uses == [page], (card, uses)
