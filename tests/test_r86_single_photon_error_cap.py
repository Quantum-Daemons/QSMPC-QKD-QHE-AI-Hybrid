"""The single-photon error cap is a named option, off by default, the same in both languages (Round 86,
batch 13; the owner's choice of 2026-11-28: "A named option, page unchanged").

SatQuMA caps the single-photon errors in the test basis at every error observed there
(cnqo-qcomms/SatQuMA v2.0.0, key/protocols/key_efficient_BB84.py, line 371); the papers this project
reproduces print the decoy bound alone. Applied everywhere, the cap moved Rusca's Table I 8 to 16
percent above the paper and Liao's pass above the key Liao reports, so the default keeps the papers'
bound, and only the SatQuMA comparison names the option. This file holds:

  - the default is off in both parameter classes, in `params_for_source` and in the browser's source;
  - with the option on, the capped bound is never above every error observed, never gives less key
    than the uncapped one, and both languages compute the same key, on both bounds;
  - with it off, the result is the uncapped bound's, unchanged;
  - nothing the page runs names the option: no browser source outside `finiteKey.ts`, and no
    Python source under `src/`, sets it.
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

from qsh.qkd.channel import ChannelParams
from qsh.qkd.finite_key import (
    DecoyCounts,
    FiniteKeyParams,
    VacuumDecoyParams,
    expected_counts,
    key_length,
    one_decoy_key_length,
    params_for_source,
    vacuum_decoy_key_length,
)

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"


def _cases() -> list[tuple[str, DecoyCounts, DecoyCounts, FiniteKeyParams | VacuumDecoyParams]]:
    """Counts each bound sees: fibre at two distances and blocks under both bounds, and a test basis
    whose decoy saw few errors, as SatQuMA's zenith pass did, where Eq. (4) exceeds every error."""
    out = []
    for km, pulses in ((25.0, 1e10), (50.0, 1e11), (100.0, 1e12)):
        ch = ChannelParams(distance_km=km)
        for p in (FiniteKeyParams(), VacuumDecoyParams(mu_signal=0.5, mu_decoy=0.1, p_signal=0.6, p_decoy=0.3)):
            z, x = expected_counts(pulses, ch, p)
            out.append((f"{type(p).__name__} {km:g} km {pulses:.0e}", z, x, p))
    vac = VacuumDecoyParams(mu_signal=0.5, mu_decoy=0.1, p_signal=0.7, p_decoy=0.2, deviation="chernoff")
    z = DecoyCounts((4.2e5, 2.0e4, 30.0), (2.6e3, 1.5e2, 15.0))
    x = DecoyCounts((4.0e4, 2.1e3, 3.0), (2.4e2, 13.0, 1.5))
    out.append(("few test-basis errors", z, x, vac))
    out.append(("few test-basis errors, both bases", z, x, replace(vac, key_basis="both")))
    return out


CASES = _cases()


def _source(p: FiniteKeyParams | VacuumDecoyParams, cap: bool) -> dict:
    return {"intensities": [list(v) for v in p.intensities], "pZ": p.p_z, "epsSec": p.eps_sec,
            "epsCor": p.eps_cor, "fEC": p.f_ec, "vacuum": isinstance(p, VacuumDecoyParams),
            "deviation": p.deviation, "keyBasis": p.key_basis, **({"capSinglePhotonErrors": True} if cap else {})}


@pytest.fixture(scope="module")
def browser() -> dict:
    if not ESBUILD.exists():
        pytest.skip("web/node_modules/.bin/esbuild is absent; run `npm ci` in web/")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        entry, bundle, runner = d / "entry.ts", d / "bundle.mjs", d / "run.mjs"
        entry.write_text(f"export * from {json.dumps(str(WEB / 'src/lib/finiteKey'))};\n")
        subprocess.run([str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node",
                        f"--outfile={bundle}", "--log-level=error"], cwd=WEB, check=True, capture_output=True)
        payload = [{"z": {"n": list(z.n), "m": list(z.m)}, "x": {"n": list(x.n), "m": list(x.m)},
                    "off": _source(p, False), "on": _source(p, True)} for _, z, x, p in CASES]
        runner.write_text(f"import * as m from {json.dumps(bundle.as_uri())};\n" + f"""
const cases = {json.dumps(payload)};
const fin = (v) => (Number.isFinite(v) ? v : null);
const out = cases.map((c) => ({{ off: fin(m.keyLength(c.z, c.x, c.off).keyBits),
                                 on: fin(m.keyLength(c.z, c.x, c.on).keyBits) }}));
console.log(JSON.stringify({{ cases: out }}));
""")
        return json.loads(subprocess.run(["node", str(runner)], check=True, capture_output=True,
                                         text=True).stdout)


def test_the_default_is_off_everywhere() -> None:
    assert FiniteKeyParams().cap_single_photon_errors is False
    assert VacuumDecoyParams().cap_single_photon_errors is False

    class _Src:
        mu_signal, mu_decoy, p_signal, p_decoy, basis_prob = 0.5, 0.1, 0.7, 0.2, 0.5

    for vac in (False, True):
        s = _Src()
        s.vacuum_decoy = vac
        assert params_for_source(s).cap_single_photon_errors is False
        assert params_for_source(s, cap_single_photon_errors=True).cap_single_photon_errors is True
    ts = (WEB / "src/lib/finiteKey.ts").read_text()
    body = ts[ts.index("export function finiteKeySourceOf"):]
    body = body[:body.index("\n}\n")]
    assert "capSinglePhotonErrors" not in body, "the browser's source must not set the cap"


@pytest.mark.parametrize("i", range(len(CASES)), ids=[c[0] for c in CASES])
def test_the_cap_is_an_upper_bound_and_never_costs_key(i) -> None:
    label, z, x, p = CASES[i]
    off = key_length(z, x, p).key_bits
    on = key_length(z, x, replace(p, cap_single_photon_errors=True)).key_bits
    if math.isfinite(off):
        assert on >= off - 1e-9, label
    else:
        assert not math.isfinite(on), label
    # Off is the bound as it was: the same function with the field left at its default.
    assert key_length(z, x, replace(p, cap_single_photon_errors=False)).key_bits == off or (
        not math.isfinite(off)), label


def test_the_cap_binds_where_the_decoy_saw_few_errors() -> None:
    """Eq. (4) alone exceeds every error the test basis saw here, so the cap lowers the phase error."""
    label, z, x, p = next(c for c in CASES if c[0] == "few test-basis errors")
    off, on = vacuum_decoy_key_length(z, x, p), vacuum_decoy_key_length(z, x, replace(p, cap_single_photon_errors=True))
    assert on.phi_z < off.phi_z, label
    assert on.key_bits > off.key_bits, label


def test_the_one_decoy_bound_takes_the_option_too() -> None:
    ch = ChannelParams(distance_km=100.0)
    p = FiniteKeyParams()
    z, x = expected_counts(1e12, ch, p)
    off = one_decoy_key_length(z, x, p)
    on = one_decoy_key_length(z, x, replace(p, cap_single_photon_errors=True))
    assert on.phi_z <= off.phi_z
    assert on.key_bits >= off.key_bits


@pytest.mark.parametrize("i", range(len(CASES)), ids=[c[0] for c in CASES])
def test_both_languages_compute_the_same_key_with_the_option_on_and_off(browser, i) -> None:
    label, z, x, p = CASES[i]
    js = browser["cases"][i]
    for key, got in (("off", js["off"]), ("on", js["on"])):
        want = key_length(z, x, replace(p, cap_single_photon_errors=(key == "on"))).key_bits
        if math.isfinite(want):
            assert got == pytest.approx(want, rel=1e-9, abs=1e-6), (label, key)
        else:
            assert got is None, (label, key)


def test_nothing_the_page_runs_names_the_option() -> None:
    """The browser's sources other than the bound itself never set it, nor does any Python source the
    pipeline runs: only scripts (the SatQuMA comparison) and tests may."""
    web_hits = [str(f.relative_to(REPO)) for f in (WEB / "src").rglob("*.ts*")
                if f.name != "finiteKey.ts" and "capSinglePhotonErrors" in f.read_text(encoding="utf-8")]
    assert web_hits == []
    py_hits = [str(f.relative_to(REPO)) for f in (REPO / "src").rglob("*.py")
               if f.name != "finite_key.py"
               and re.search(r"cap_single_photon_errors\s*=\s*True", f.read_text(encoding="utf-8"))]
    assert py_hits == []
    fk = (REPO / "src/qsh/qkd/finite_key.py").read_text(encoding="utf-8")
    assert re.search(r"cap_single_photon_errors:\s*bool\s*=\s*False", fk)
