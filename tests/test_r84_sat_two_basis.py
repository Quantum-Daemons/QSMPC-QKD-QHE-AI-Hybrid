"""The finite key on one key basis and on both, and the satellite Network tab that sets both beside the
published keys (Round 84, the user's choice: "add a two-basis mode").

WHY THIS EXISTS. Lim et al. 2014's protocol, as this project ran it, keys the Z basis and spends X on
the phase-error estimate. The keys it was compared with did not: Liao et al. 2017 distilled their
300,939 bits from every sifted bit, and Khmelev et al. 2024 state that "both bases are used for key
generation". So the Network tab compared a one-basis key with two-basis ones and read the shortfall,
0.75 of Liao's key, as a gap "the statistics close 45 percent" of. Keyed on both bases, each basis's
phase error from the other basis's counts and each at half of eps_sec, the model's key on Liao's pass
is 1.50 times the published one, and the drop from the asymptotic row to the one-basis row is 95
percent the basis convention.

WHAT IS ASSERTED

- The option, in Python: "one" is every default; "both" is the sum of the Z key and the X key, each
  the same bound with the bases exchanged at half of eps_sec and eps_cor, for the vacuum + weak bound
  and the 1-decoy one; a basis with no key adds none; an unknown basis is refused.
- The numbers, on Liao's pass and Khmelev et al.'s session, under Chernoff and Hoeffding.
- The page: the Network tab prints both conventions beside Liao's and Khmelev et al.'s keys, anchors
  Khmelev on its final 310,400 bits, names 629,000 as the key before the mismatch analysis and f_EC
  1.44 against 1.16, computes its gap paragraph (other profiles print other figures), and no longer
  tells the 45 and 29 percent story; the relay's finite keys use the satellite's statistics.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from qsh.orchestration.stages import Context
from qsh.qkd.channel import ChannelParams, link_elevation_deg
from qsh.qkd.finite_key import (
    KEY_BASES,
    FiniteKeyParams,
    VacuumDecoyParams,
    block_counts,
    blocks_for_positive_key,
    expected_counts,
    finite_key_over,
    key_length,
    one_decoy_key_length,
    params_for_source,
    vacuum_decoy_key_length,
)
from qsh.qkd.satellite_pass import simulate_pass
from qsh.usecases import satellite_comms
from qsh.usecases.satellite_comms import _sat_channel, _sat_finite_key_params, _sat_source

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
PANEL = WEB / "src" / "components" / "SatNetworkPanel.tsx"


def _liao_segments():
    ch = _sat_channel(Context(usecase="satellite_comms", config={}))
    fs = ch.free_space
    return simulate_pass(ch, link_elevation_deg(fs, 645.0), window_elevation_deg=link_elevation_deg(fs, 1200.0),
                         step_s=0.25).segments


def _khmelev_segments():
    ch = _sat_channel(Context(usecase="satellite_comms", config={"sat_profile": "micius_2022_zvenigorod"}))
    fs = ch.free_space
    # Rising from the 28 deg where their tracking started, setting at 1,100 km (Round 86).
    return simulate_pass(ch, link_elevation_deg(fs, 600.0), window_elevation_deg=link_elevation_deg(fs, 1100.0),
                         rise_elevation_deg=28.0).segments


def _params(deviation: str, profile: str | None = None):
    ctx = Context(usecase="satellite_comms", config={"sat_profile": profile, "sat_fk_deviation": deviation})
    return _sat_finite_key_params(ctx, _sat_source(ctx))


# --- the option, in Python ------------------------------------------------------------------------

def test_one_basis_is_every_default_and_an_unknown_basis_is_refused() -> None:
    assert KEY_BASES == ("one", "both")
    assert VacuumDecoyParams().key_basis == FiniteKeyParams().key_basis == "one"
    assert _params("chernoff").key_basis == "one"
    with pytest.raises(ValueError):
        VacuumDecoyParams(key_basis="X")
    with pytest.raises(ValueError):
        FiniteKeyParams(key_basis="two")
    # The satellite stage's key is unchanged by the option's existence: one basis, as before.
    m = satellite_comms.run_demo({"seed": 1})["metrics"]
    z, x = block_counts(satellite_comms._sat_block(Context(usecase="satellite_comms", config={}),
                                                   _sat_channel(Context(usecase="satellite_comms", config={}))).unit,
                        _params("chernoff"))
    assert m["secure_key_bits"] == int(vacuum_decoy_key_length(z, x, _params("chernoff")).key_bits)


@pytest.mark.parametrize("vacuum", [True, False])
def test_both_bases_is_each_basis_at_half_the_budget_summed(vacuum: bool) -> None:
    ch = ChannelParams(distance_km=20)
    if vacuum:
        p = VacuumDecoyParams(mu_signal=0.6, mu_decoy=0.1, p_signal=0.6, p_decoy=0.3, p_z=0.7, deviation="chernoff")
        bound = vacuum_decoy_key_length
    else:
        p = FiniteKeyParams(mu_signal=0.5, mu_decoy=0.15, p_signal=0.6, p_z=0.7, deviation="chernoff")
        bound = one_decoy_key_length
    z, x = expected_counts(5e8, ch, p)
    half = replace(p, eps_sec=p.eps_sec / 2, eps_cor=p.eps_cor / 2)
    kz, kx = bound(z, x, half).key_bits, bound(x, z, half).key_bits
    assert kz > 0 and kx > 0 and kz != pytest.approx(kx, rel=1e-3)   # p_Z = 0.7: the bases differ
    both = replace(p, key_basis="both")
    r = key_length(z, x, both)
    assert r.key_bits == pytest.approx(kz + kx, rel=1e-12)
    assert r.basis_keys == pytest.approx((kz, kx), rel=1e-12)
    assert r.lambda_ec == pytest.approx(bound(z, x, half).lambda_ec + bound(x, z, half).lambda_ec, rel=1e-12)
    # Every entry point honours the option: the bound itself, the pass helper, the pooling search.
    assert bound(z, x, both).key_bits == pytest.approx(r.key_bits, rel=1e-12)
    assert finite_key_over([(5e8, ch)], both).key_bits == pytest.approx(r.key_bits, rel=1e-12)
    assert key_length(z, x, p).key_bits == pytest.approx(bound(z, x, p).key_bits, rel=1e-12)
    assert blocks_for_positive_key(z, x, both) == 1
    # And the source's builder carries it.
    src = satellite_comms._sat_source(Context(usecase="satellite_comms", config={}))
    assert params_for_source(src, key_basis="both").key_basis == "both"


def test_a_basis_without_a_key_adds_none() -> None:
    """A block too short for either basis has no key, and no bound at all reads -inf as one basis does.
    Biased to Z, the Z key's phase error rests on the few X counts and fails first, while the X key's
    rests on the many Z counts: the pair is the X key alone, not that less the Z shortfall."""
    ch = ChannelParams(distance_km=20)
    p = VacuumDecoyParams(mu_signal=0.6, mu_decoy=0.1, p_signal=0.6, p_decoy=0.3, p_z=0.5, deviation="chernoff",
                          key_basis="both")
    z, x = expected_counts(1e3, ch, p)
    r = key_length(z, x, p)
    assert r.key_bits == -math.inf and r.basis_keys == (-math.inf, -math.inf)
    lop = replace(p, p_z=0.9)
    z, x = expected_counts(4e7, ch, lop)
    r = key_length(z, x, lop)
    assert max(r.basis_keys) < 0 and r.key_bits == max(r.basis_keys), r.basis_keys
    z, x = expected_counts(6e7, ch, lop)
    r = key_length(z, x, lop)
    kz, kx = r.basis_keys
    assert kz < 0 < kx, (kz, kx)
    assert r.key_bits == pytest.approx(kx, rel=1e-12)
    assert blocks_for_positive_key(*expected_counts(4e7, ch, lop), lop) == 2


@pytest.mark.parametrize("deviation,liao,khmelev", [
    # Khmelev's session rises from the 28 deg where their tracking started (Round 86): until then it counted from
    # 1,100 km on both sides, and read (435_095, 868_462) and (354_672, 705_542).
    ("chernoff", (226_072, 450_923), (415_669, 829_658)),
    ("hoeffding", (164_183, 325_556), (337_103, 670_497)),
])
def test_the_flown_passes_on_one_basis_and_on_both(deviation, liao, khmelev) -> None:
    """The figures the Network tab prints at the defaults, to the bit."""
    for segments, profile, want in ((_liao_segments(), None, liao), (_khmelev_segments(), "micius_2022_zvenigorod", khmelev)):
        p = _params(deviation, profile)
        z, x = block_counts(segments, p)
        got = (key_length(z, x, p).key_bits, key_length(z, x, replace(p, key_basis="both")).key_bits)
        assert tuple(round(v) for v in got) == want, (deviation, profile, got)


# --- the page -------------------------------------------------------------------------------------

def _render(params_list: list[dict]) -> list[str]:
    """The Network tab rendered to text by React's server renderer, for each run's params."""
    if not ESBUILD.exists():
        pytest.skip("web/node_modules/.bin/esbuild is absent; run `npm ci` in web/")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        bundle, runner = d / "bundle.mjs", d / "run.mjs"
        # On stdin, so the bare imports ("react") resolve from web/, as the node checks' resolveDir does.
        entry = (f"export {{ default as SatNetworkPanel }} from {json.dumps(str(PANEL.with_suffix('')))};\n"
                 'export { createElement } from "react";\n'
                 'export { renderToStaticMarkup } from "react-dom/server.browser";\n')
        subprocess.run([str(ESBUILD), "--bundle", "--loader=tsx", "--format=esm", "--platform=node", "--jsx=automatic",
                        '--define:import.meta.env={"BASE_URL":"/"}', '--define:process.env.NODE_ENV="production"',
                        f"--outfile={bundle}", "--log-level=error"], cwd=WEB, check=True, capture_output=True,
                       input=entry.encode())
        runner.write_text(f"import * as m from {json.dumps(bundle.as_uri())};\n" + f"""
const out = {json.dumps(params_list)}.map((params) => m.renderToStaticMarkup(m.createElement(m.SatNetworkPanel, {{ params }}))
  .replace(/<[^>]+>/g, " ").replace(/&#x27;|&apos;/g, "'").replace(/&quot;/g, '"').replace(/&amp;/g, "&")
  .replace(/\\s+/g, " "));
console.log(JSON.stringify(out));
""")
        return json.loads(subprocess.run(["node", str(runner)], check=True, capture_output=True, text=True).stdout)


@pytest.fixture(scope="module")
def pages() -> dict[str, str]:
    profiles = [None, "micius_2017_upper_atmosphere"]
    texts = _render([{} if p is None else {"sat_profile": p} for p in profiles])
    return dict(zip(["default", "upper"], texts, strict=True))


def test_the_tab_prints_both_conventions_beside_the_published_keys(pages) -> None:
    t = pages["default"]
    assert "Secret key, finite, one basis (Z) 226,072 300,939 (finite-key) -24.9%" in t
    assert "Secret key, finite, both bases 450,923 300,939 (finite-key) +49.8%" in t
    assert "Secret key, asymptotic (both bases) 462,290" in t
    # Khmelev et al., anchored on their final key, with the key before the mismatch analysis named.
    assert "Khmelev et al., Opt. Express 32, 11964 (2024)" in t
    # Against their key before the detector-mismatch analysis, which this model has no mismatch to need (Round 86).
    assert "Secret key, finite, one basis (Z) 415,669 629,000 (before the mismatch analysis) -33.9%" in t
    assert "Secret key, finite, both bases 829,658 629,000 (before the mismatch analysis) +31.9%" in t
    assert "Their final key is 310,400 bits" in t and "f_EC 1.44, where this model uses 1.16" in t
    # The run's own pass gives both too.
    assert re.search(r"keyed on the Z basis as the run keys it; on both bases, [\d.,]+ kbit", t)


def test_the_gap_paragraph_is_computed_and_tells_the_basis_story(pages) -> None:
    t = pages["default"]
    for story in ("45 percent", "29 percent", "the statistics close", "not most of either", "0.75 and 0.69"):
        assert story not in t, story
    assert "so 95% of the step from the asymptotic row to the one-basis row is the basis convention" in t
    assert ("the model's key is 1.50 times Liao's, and on Khmelev et al.'s session 1.32 times their key before the "
            "detector-mismatch correction") in t
    # On another profile the same sentence carries that profile's figures, not the default's.
    u = pages["upper"]
    one = re.search(r"Secret key, finite, one basis \(Z\) ([\d,]+) 300,939", u).group(1)
    both = re.search(r"Secret key, finite, both bases ([\d,]+) 300,939", u).group(1)
    asym = re.search(r"Secret key, asymptotic \(both bases\) ([\d,]+)", u).group(1)
    one, both, asym = (float(v.replace(",", "")) for v in (one, both, asym))
    assert f"{both / 300_939:.2f} times Liao's" in u and "1.50 times Liao's" not in u
    share = round(100 * (both - one) / (asym - one))
    assert f"so {share}% of the step" in u


def test_the_source_holds_no_literal_ratio_story() -> None:
    src = PANEL.read_text()
    for literal in ("0.55 of Liao", "0.56 of Khmelev", "45 percent of the gap", "29 percent of the gap", "629,000 bits, a key"):
        assert literal not in src, literal


def test_the_relay_uses_the_satellites_statistics(pages) -> None:
    """The relay's finite keys named the fibre's union of two statistics while the pass section above
    named the satellite's Chernoff bounds, on one tab: it read `fk_deviation`, the fibre's control."""
    t = pages["default"]
    row = re.search(r"Pass pairs for Liao's 100 kB(.{0,500})", t).group(1)
    assert "Chernoff-type bounds, Yin et al. 2020" in row and "the tighter of Chernoff and Hoeffding" not in row
    assert "params.fk_deviation" not in PANEL.read_text()


def test_the_tab_and_python_agree_on_the_both_bases_keys(pages) -> None:
    """The figures the page prints are the Python bound's on the same counts."""
    t = pages["default"]
    for segments, profile in ((_liao_segments(), None), (_khmelev_segments(), "micius_2022_zvenigorod")):
        p = replace(_params("chernoff", profile), key_basis="both")
        bits = key_length(*block_counts(segments, p), p).key_bits
        assert f"{round(bits):,}" in t, (profile, bits)
        assert math.isfinite(bits)
