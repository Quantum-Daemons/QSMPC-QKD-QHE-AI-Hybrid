"""The satellite page's Jinan-1 card states its post-hoc figures as the record has them (Round 86, batch 6).

The downlink's check against Jinan-1's measured pass was pre-registered (references/satellite-jinan1-pass-preregistration.md),
its windows failed on the paper's block-wise QBER, and addendum 2 named a post-hoc analysis. On the owner's choice of
2026-11-26 the page shows that analysis, labelled post hoc. Held here: the page file is the post-hoc record's, the
robustness summary is the record's grid in brief, the file carries CC BY 4.0's credit and changes, and the card says what
it is.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "web" / "public" / "data" / "satellite" / "jinan1_pass.json"
RECORD = REPO / "docs" / "measurements" / "satellite-jinan1-pass-posthoc-2026-11-26.json"
CARD = REPO / "web" / "src" / "components" / "SatJinanPassCard.tsx"
PREREG = "references/satellite-jinan1-pass-preregistration.md"


def test_the_page_file_is_the_post_hoc_record() -> None:
    from test_markdown_math_uses_the_house_style import FROZEN

    site, rec = json.loads(SITE.read_text()), json.loads(RECORD.read_text())
    assert site["post_hoc"] is True and "POST HOC" in site["windows"] and site["windows"] == rec["windows"]
    assert site["preregistration"] == rec["preregistration"] == {"file": PREREG, "sha256": FROZEN[PREREG]}
    for k in ("background_per_s", "profile_efficiency", "fitted_efficiency", "fit_minus_profile_db", "residuals_after",
              "trend_after_db_per_air_mass", "falls", "qber", "key", "rules"):
        assert site["summary"][k] == rec[k], k
    grid = rec["robustness"]["grid"]
    means = [g[m]["mean_db"] for g in grid for m in g if m.startswith("model_")]
    falls = [abs(g[m][f]) for g in grid for m in g if m.startswith("model_")
             for f in ("fall_first_minus_model_db", "fall_last_minus_model_db")]
    assert site["summary"]["robustness"] == {"choices": 2 * len(grid), "mean_residual_db": [min(means), max(means)],
                                             "largest_fall_difference_db": max(falls)}


def test_the_per_second_series_hold_together() -> None:
    site = json.loads(SITE.read_text())
    n = len(site["seconds"])
    assert n == json.loads(RECORD.read_text())["link_seconds"]["used"]
    for k in ("range_km", "measured_detections_per_s", "model_detections_per_s", "model_detections_per_s_refitted",
              "measured_qber", "model_qber"):
        assert len(site[k]) == n, k
    assert all(b > a for a, b in pairwise(site["seconds"]))
    assert min(site["measured_detections_per_s"]) > 0 and min(site["model_detections_per_s"]) > 0


def test_the_file_carries_its_licence_and_the_card_says_post_hoc() -> None:
    site = json.loads(SITE.read_text())
    assert site["licence"] == "CC-BY-4.0" and site["licence_url"] == "https://creativecommons.org/licenses/by/4.0/"
    assert "zenodo.14732295" in site["attribution"] and "Nature 640" in site["attribution"]
    assert "background" in site["modifications"] and "model" in site["modifications"]
    card = CARD.read_text().replace("&apos;", "'")
    for phrase in ("post hoc", "was pre-registered, but its windows did not hold", "one value per key block",
                   "800-ps gate", "ten percent of detections"):
        assert phrase in card, phrase
