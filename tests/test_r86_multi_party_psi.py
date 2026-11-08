"""The PSI among three to five banks, and the WebM export (Round 86).

The protocol is checked where it runs, by `cargo test` in the wasm crate and by `web/scripts/check_multi_psi.mjs` on the module the page imports. This holds what
those cannot see: that the config is the one the crate enforces, that a run only happens on the finance page, and that the card says what the run cannot prove.
"""

from __future__ import annotations

import re
from pathlib import Path

from qsh.config import frontend_config, get_config

REPO = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO / path).read_text()


def test_the_config_stays_inside_what_the_crate_accepts() -> None:
    crate = _read("wasm/qsh-mpc/src/lib.rs")
    max_parties = int(re.search(r"PSI_MAX_PARTIES: usize = (\d+)", crate).group(1))
    max_ids = int(re.search(r"PSI_MAX_IDS: usize = (\d+)", crate).group(1))
    c = get_config().multi_party_psi
    assert c.max_parties == max_parties
    assert 3 <= c.parties <= c.max_parties
    assert 1 <= c.shared < c.set_size <= max_ids


def test_the_browser_gets_it_from_the_config() -> None:
    assert frontend_config()["multi_party_psi"]["set_size"] == get_config().multi_party_psi.set_size
    source = _read("web/src/lib/clientCompute.ts")
    assert "const cfg = multiPartyPsiCfg();" in source
    assert not re.search(r"multiPartyPsiOffThread\(\[", source), "the lists come from the config's sizes"


def test_a_run_needs_three_banks_and_the_finance_page() -> None:
    compute = _read("web/src/lib/clientCompute.ts")
    assert 'if (usecase === "finance_aml" && Number(config.multi_party_psi) >= 3)' in compute
    assert "if (!(parties >= 3 && parties <= cfg.max_parties))" in compute
    run = _read("web/src/lib/usePipelineRun.ts")
    assert 'if (multiPsi && DEMO && usecase === "finance_aml") c.multi_party_psi = multiPsiParties;' in run


def test_ticking_it_does_not_move_any_other_stage_s_draws() -> None:
    compute = _read("web/src/lib/clientCompute.ts")
    body = compute[compute.index("async function runMultiPartyPsi("):compute.index("/** One held-out image through the student")]
    assert "const u = rng(seed + 0x6d70);" in body, "its accounts have a generator of their own"


def test_the_first_bank_shuffles_its_list() -> None:
    source = _read("web/src/lib/clientMultiPsi.ts")
    assert "const sets = hashed.map((h) => shuffled(Uint32Array.from(h)));" in source
    assert "randomWords(1)" in source, "from the CSPRNG, not the run's seeded generator"
    assert "Math.random" not in source


def test_the_card_says_what_a_run_cannot_prove() -> None:
    panel = _read("web/src/components/MultiPsiPanel.tsx")
    for needle in ("so the separation is notional", "not post-quantum", "pre-release library", "Kolesnikov", "matches the intersection in the clear",
                   "DISAGREES with the intersection in the clear"):
        assert needle in panel, needle


def test_the_gate_is_in_both_pipelines() -> None:
    assert "npm run check:multi-psi" in _read(".github/workflows/ci.yml")
    assert "check:multi-psi" in _read("scripts/ci_local.sh")
    assert '"check:multi-psi"' in _read("web/package.json")


def test_webm_uses_the_same_frames_and_is_offered_only_where_it_works() -> None:
    export = _read("web/src/lib/export.ts")
    assert "MediaRecorder.isTypeSupported" in export
    assert "canvas.captureStream(0)" in export
    for page in ("web/src/pages/UseCasePage.tsx", "web/src/pages/Orchestration.tsx"):
        source = _read(page)
        assert "{webmType() ? (" in source, page
        assert "downloadWebm(" in source, page
