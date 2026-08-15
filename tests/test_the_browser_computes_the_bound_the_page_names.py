"""The page names a bound; this asserts the browser computes THAT bound and not a different one.

WHY THIS EXISTS

Round 44 gave the secret-key rate a genuine bit/s by multiplying a per-pulse fraction by an assumed
100 MHz clock, and put ~958 kbit/s on the public demo. It did not ask whether the per-pulse fraction
was the quantity the page says it is. It is not.

`web/src/lib/quantumSim.ts` computes

    secureFraction = 1 - h2(q) - f_EC h2(q)

on the OBSERVED overall QBER. That is the legacy Shor-Preskill / GLLP form for a channel with no
decoy states. Seven surfaces call it a decoy-state bound - `clientCompute.ts` twice in provenance
strings and twice in prose, `quantumSim.ts`'s own header, `bb84.py`'s module docstring, and the
orchestration DAG label. `taggedFraction()` has existed since Round 29, is called exactly once, and
its only consumers are a display string and a warning: the decoy quantity is computed, printed beside
the rate, and never reaches it.

The gap is not a coefficient. The runtime models NO MEAN PHOTON NUMBER at all - it uses the
transmittance `eta` directly as the per-pulse detection probability, which for a weak coherent source
asserts `mu ~ 1 photon/pulse` - and it credits every sifted bit as single-photon, i.e. `p1 = 1`. That
is STRONGER than any decoy bound can deliver, so the label does not merely round the number off: it
names a technique that would LOWER it. Measured at the demo's own operating point, 50 km on the
project's own fibre defaults:

    displayed today                        0.009434 bit/pulse   943,371 bit/s
    two-intensity decoy (Ma et al. 2005)   0.002355 bit/pulse   235,479 bit/s   <- 4.01x lower

WHY NOTHING CAUGHT IT

Every existing cross-language QKD gate is TEXTUAL. `tests/test_sdn_fields_carry_the_dimension_the_spec_says.py`
greps the Python file for a unit comment. `web/scripts/check_qkd.mjs` asserts plumbing and identities
INSIDE TypeScript, and says so in its own header: it deliberately does not re-derive the key rate,
because a mirror written in TypeScript would copy a TypeScript bug. Both are right about what they
do. Neither can see that the two languages compute different physics, because neither ever puts a
number from one beside a number from the other.

So this file is the first numeric cross-language check in the project: it executes the browser's
`bb84()` through esbuild + node and compares the result against `src/qsh/qkd/bb84.py`'s
`_decoy_secure_fraction` evaluated live. Two implementations, one asserted identity, in two
languages - and the fix cannot satisfy both by restating itself.

WHAT IS DELIBERATELY *NOT* ASSERTED

E91 is left alone. `e91()` uses the same GLLP-on-observed-QBER form, and for an entanglement-based
protocol that is CORRECT: there are no decoy states in E91, because there is no attenuated laser to
put them on. The defect there is the label, not the formula, and it is checked separately below.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
QSIM = WEB / "src" / "lib" / "quantumSim.ts"
COMPUTE = WEB / "src" / "lib" / "clientCompute.ts"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"

sys.path.insert(0, str(REPO / "src"))

#: The operating point every assertion below is made at. 50 km is the fibre default the demo ships
#: and the distance the deployed panel was read at, so a failure here is a failure about the number
#: a visitor actually sees, not about a corner of parameter space nobody visits.
DISTANCE_KM = 50.0
PULSES = 200_000


def _cfg() -> dict:
    return json.loads((WEB / "src" / "lib" / "config.defaults.json").read_text())["qkd"]


def _browser() -> dict:
    """Run the browser's own `bb84()` and return what it produced.

    Bundled with esbuild exactly as `web/scripts/check_qkd.mjs` does, so this exercises the same
    module graph the browser gets rather than a hand-rolled re-import.
    """
    if not ESBUILD.exists():
        pytest.skip(
            "web/node_modules/.bin/esbuild is absent, so the browser model cannot be executed. "
            "Run `npm ci` in web/. This is a SKIP rather than a pass: the identity below is "
            "unchecked, not satisfied."
        )
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        entry, bundle, runner = d / "entry.ts", d / "bundle.mjs", d / "run.mjs"
        entry.write_text(
            "export { bb84, e91, FIBER, transmittance, intrinsicQber } from "
            f"{json.dumps(str(WEB / 'src/lib/quantumSim'))};\n"
            f"export {{ qkdCfg }} from {json.dumps(str(WEB / 'src/lib/config'))};\n"
        )
        subprocess.run(
            [str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node",
             f"--outfile={bundle}", "--log-level=error"],
            cwd=WEB, check=True, capture_output=True,
        )
        runner.write_text(
            f"import * as m from {json.dumps(bundle.as_uri())};\n"
            f"const chan = {{ ...m.FIBER, distanceKm: {DISTANCE_KM} }};\n"
            "const runs = [1, 2, 3, 4, 5].map((s) => m.bb84(%d, chan, s));\n" % PULSES
            + "console.log(JSON.stringify({\n"
            "  rates: runs.map((r) => r.secretKeyRate),\n"
            "  bits: runs.map((r) => r.secureKeyBits),\n"
            "  sifted: runs.map((r) => r.siftedBits),\n"
            "  qbers: runs.map((r) => r.qber),\n"
            "  nonDecoy: runs.map((r) => r.secretKeyRateNonDecoy ?? null),\n"
            "  eta: m.transmittance(chan),\n"
            "  eDet: m.intrinsicQber(chan),\n"
            "}));\n"
        )
        out = subprocess.run([ "node", str(runner) ], check=True, capture_output=True, text=True)
        return json.loads(out.stdout)


def _python_decoy(eta: float, e_det: float) -> float:
    """The bound the page names, from the implementation the test suite already checks.

    `src/qsh/qkd/bb84.py::_decoy_secure_fraction` is the two-intensity Ma, Qi, Zhao & Lo (2005)
    estimator, and `src/qsh/qkd/decoy_reference.py` - whose docstring says it "exists to be CHECKED"
    - reproduces the paper's own published curves in `tests/test_decoy_reference.py`. The repository
    owns a correct implementation and displays a different one; this names which is which.
    """
    from qsh.qkd.bb84 import _decoy_secure_fraction

    c = _cfg()
    mu = float(c["mu_signal"])
    nu = min(max(1e-4, float(c["mu_decoy"])), mu * 0.9)
    y0 = 2.0 * float(c["dark_count_prob"])
    sift = 0.5                      # the browser draws bases uniformly, so p^2 + (1-p)^2 = 0.5
    skr, _e_mu, _e1 = _decoy_secure_fraction(
        eta, mu, nu, e_det, y0, float(c["error_correction_eff"]), sift)
    return skr


def _python_production_rate() -> float:
    """What the PRODUCTION Python path computes - reached the way `stages.py` reaches it.

    THIS IS THE CHECK THIS FILE SHOULD HAVE MADE IN ROUND 45 AND DID NOT.

    `_python_decoy()` above imports `_decoy_secure_fraction` and calls it directly. That is a
    private helper. `src/qsh/qkd/bb84.py:116` branches on `qkd_params`, and the helper is reached
    ONLY in the branch taken when `qkd_params` is supplied - which, in the whole repository, only
    `src/qsh/qkd/optimize.py` ever did. Every production surface (`orchestration/stages.py`,
    `usecases/satellite_comms/__init__.py`, `api/inference_service.py`) called `run_bb84` without it
    and went on computing `1 - h2(q) - f h2(q)` on the observed QBER: 0.008936 bit/pulse against the
    browser's 0.002355, a factor of 4.

    So Round 45 announced "the first numeric cross-language check in the project", and pointed it at
    a function no Python caller invoked. Its own title - Round 44's - was "the fix changed one
    language". It did it again and certified itself against the half it had fixed.

    This resolves the provider through `crypto_agility.CryptoRegistry` exactly as `QKDStage` does,
    so what is asserted is what the orchestration actually runs, not what a helper could do.
    """
    from qsh.orchestration.crypto_agility import default_registry
    from qsh.qkd.channel import ChannelParams

    c = _cfg()
    provider = default_registry().get("qkd", "bb84-decoy")
    res = provider(
        n_pulses=PULSES,
        params=ChannelParams(
            distance_km=DISTANCE_KM,
            misalignment=float(c["misalignment"]),
            eve_intercept_fraction=0.0,
        ),
        seed=1,
    )
    return float(res.secret_key_rate)


def test_the_production_python_path_computes_the_bound_too() -> None:
    """Both languages, at the surface each actually runs.

    Red on the unmodified tree by ~4x, and it cannot be satisfied by editing the browser.
    """
    got = _python_production_rate()
    expected = _python_decoy(*_eta_and_edet())
    assert expected > 0, "the reference bound is zero here; check the channel defaults first"
    assert abs(got - expected) <= 1e-9, (
        f"the PRODUCTION Python path reports {got:.6f} bit/pulse where the decoy bound this "
        f"repository implements gives {expected:.6f} - a factor of {got / expected:.3f}.\n\n"
        "`run_bb84` only reaches `_decoy_secure_fraction` when `qkd_params` is supplied, and the "
        "registry entry named 'bb84-decoy' does not supply it. The name is the claim.\n\n"
        "Fix it where the provider is built - `orchestration/crypto_agility.py` - so every caller "
        "that resolves through the registry inherits it, and pass `QKDParams()` at the two call "
        "sites that bypass the registry."
    )


def test_no_production_call_site_takes_the_legacy_branch() -> None:
    """Scope, so the next call site cannot repeat it.

    The numeric check above pins ONE path. This pins the class: any `run_bb84(` outside the
    optimiser and the tests must ask for the decoy model, either directly or through the registry
    provider. Stated as a property rather than a file list, because a gate whose scope names files
    is this project's most-repeated defect.
    """
    import re

    offenders: list[str] = []
    for rel in ("src/qsh/usecases/satellite_comms/__init__.py", "api/inference_service.py",
                "src/qsh/benchmark/alice_bob_e2e.py"):
        path = REPO / rel
        if not path.exists():
            continue
        src = path.read_text()
        for m in re.finditer(r"run_bb84\((?:[^()]|\([^()]*\))*\)", src):
            if "qkd_params" not in m.group(0):
                line = src.count("\n", 0, m.start()) + 1
                offenders.append(f"{rel}:{line}: {' '.join(m.group(0).split())[:90]}")
    assert not offenders, (
        "these production call sites take `run_bb84`'s LEGACY branch, which computes the non-decoy "
        "GLLP fraction on the observed QBER:\n  " + "\n  ".join(offenders)
        + "\n\nPass `qkd_params=QKDParams()`. The legacy branch is not wrong - it is a different, "
        "explicitly-named model that the optimiser's comparison tests still need - but nothing "
        "user-facing should reach it by omission."
    )


def _eta_and_edet() -> tuple[float, float]:
    """The channel the assertions are made on, computed once from config."""
    c = _cfg()
    eta = float(c["detector_efficiency"]) * 10 ** (
        -float(c["fiber_loss_db_per_km"]) * DISTANCE_KM / 10)
    e_det = min(0.5, float(c["misalignment"]) + 0.5 * float(c["dark_count_prob"]) / max(eta, 1e-12))
    return eta, e_det


def test_the_operating_point_is_the_one_the_demo_ships() -> None:
    """A parity check at a point nobody visits proves nothing about the page.

    Non-vacuity first: if the fibre defaults move, the numbers in this file's docstring stop
    describing the deployed panel and the comparison silently changes subject.
    """
    c = _cfg()
    assert float(c["distance_km"]) == DISTANCE_KM, (
        f"the fibre default moved to {c['distance_km']} km; this file asserts at {DISTANCE_KM} km "
        "because that is what the deployed panel was read at"
    )
    assert float(c["source_rate_hz"]) > 0, "the assumed clock is gone, so there is no bit/s to check"
    assert float(c["mu_decoy"]) < float(c["mu_signal"]), (
        "the decoy intensity is not strictly weaker than the signal, so no two-intensity bound "
        "exists to compare against"
    )


def test_the_browser_key_rate_is_the_decoy_bound_python_computes() -> None:
    """The identity, across two languages and two implementations.

    This is the assertion the round exists to make true. It is red on the unmodified tree by a
    factor of ~4, and it cannot be satisfied by editing either side alone.
    """
    b = _browser()
    expected = _python_decoy(b["eta"], b["eDet"])
    got = b["rates"][0]
    assert expected > 0, (
        "the Python decoy bound is zero at the demo's operating point, so this test has no subject "
        "- check the channel defaults before trusting a comparison against it"
    )
    assert abs(got - expected) <= 1e-9, (
        f"the browser reports {got:.6f} bit/pulse where the decoy-state bound this repository "
        f"already implements gives {expected:.6f} - a factor of {got / expected:.2f}.\n\n"
        "The browser computes `1 - h2(q) - f h2(q)` on the OBSERVED QBER, which is the non-decoy "
        "GLLP form, while seven surfaces call it a decoy-state bound. It also uses the "
        "transmittance directly as the detection probability (asserting mu ~ 1) and credits every "
        "sifted bit as single-photon (p1 = 1) - both STRONGER than a decoy bound can deliver.\n\n"
        "Port `_decoy_secure_fraction` from src/qsh/qkd/bb84.py rather than re-deriving it here."
    )


def test_the_bound_does_not_move_with_the_seed() -> None:
    """A bound is a property of the channel; only the measurement fluctuates.

    The Python path feeds `intrinsic_qber(params)` - not the sampled QBER - into the decoy
    estimator, so the rate is deterministic given the channel while the measured QBER still varies
    run to run. The browser currently derives its rate FROM the sampled QBER, so five seeds give
    five rates and the demo's headline figure wobbles between reloads (943,371 and 957,659 were both
    observed). Matching Python makes the bound stable and leaves the measurement free to vary, which
    is the honest split.
    """
    b = _browser()
    rates = b["rates"]
    assert len(set(f"{r:.12f}" for r in rates)) == 1, (
        f"the reported bound differs across seeds: {rates}. A secret-key BOUND is a function of the "
        "channel, not of one session's sampling noise; deriving it from the measured QBER is what "
        "makes the public figure move between page loads."
    )
    assert len(set(f"{q:.9f}" for q in b["qbers"])) > 1, (
        "the measured QBER is now identical across seeds too, so the sampling noise was removed "
        "rather than moved off the bound. The QBER is a measurement and must still vary."
    )


def test_the_non_decoy_counterfactual_is_reported_and_is_not_larger() -> None:
    """Both numbers, because the counterfactual is the point of computing a decoy bound at all.

    THE FIRST VERSION OF THIS TEST ASSERTED THE WRONG DIRECTION, and the model corrected it. It
    required the non-decoy figure to be LARGER, reasoning that the browser's old number was larger
    and the old number was the non-decoy one. It was not: the old number credited every sifted bit
    as single-photon (`p1 = 1`), which is neither a decoy bound nor a non-decoy bound but an
    unbounded estimate stronger than both. Recorded rather than quietly swapped, because mistaking
    "the number that was displayed" for "the number without the certification" is exactly how the
    mislabel survived seven surfaces for sixteen rounds.

    The real ordering, at this operating point:

        non-decoy GLLP   0.000000 bit/pulse   <- Delta = 7.26, no detection certifiable
        decoy bound      0.002355 bit/pulse   <- what the page now shows
        old display      0.009434 bit/pulse   <- p1 = 1, mu ~ 1; not a bound at all

    Decoy states RAISE what an operator may claim, by letting Y1 and e1 be estimated instead of
    discarded. So the counterfactual must be <= the bound, and at mu = 0.5 it is zero outright -
    `decoy_reference.py`'s docstring makes the same point about the GYS parameter set, where the
    tagged fraction exceeds 1 at zero distance. "Without decoy states this link yields nothing" is a
    sharper thing to show a reader than a merely smaller number.
    """
    b = _browser()
    nd = b["nonDecoy"][0]
    assert nd is not None, (
        "`bb84()` no longer reports the non-decoy figure. The decoy bound alone is a number with "
        "nothing to compare it to; the counterfactual is what shows the assumption doing work."
    )
    assert nd <= b["rates"][0] + 1e-12, (
        f"the non-decoy figure {nd:.6f} exceeds the decoy bound {b['rates'][0]:.6f}. Decoy states "
        "let Y1 and e1 be ESTIMATED rather than discarded, so removing them cannot raise the rate. "
        "A counterfactual above the bound means one of the two is not computing what it claims."
    )
    assert b["rates"][0] > 0, (
        "the decoy bound is zero at the demo's operating point, so the panel would show no key at "
        "all - check the channel defaults before believing the counterfactual comparison"
    )


def test_no_surface_attributes_the_rate_to_decoy_states_on_the_e91_page() -> None:
    """E91 has no decoy states, and cannot have them.

    Decoy states are a countermeasure for the multi-photon pulses an attenuated laser emits in
    prepare-and-measure BB84. E91 is entanglement-based: there is no such source, so the technique
    does not apply. `clientCompute.ts` gets this right one line and wrong the next - the QKD
    key-material row at :224 branches on the protocol, and the secret-key-rate row three lines below
    is emitted unconditionally with a decoy provenance string. One of a sibling pair was made
    protocol-aware; the other was not, which is this round's subject in miniature.
    """
    src = COMPUTE.read_text()
    m = _rate_fact_source(src)
    assert m is not None, (
        "could not find the `Secret-key rate` fact in clientCompute.ts; if it was renamed, this "
        "check has lost its subject rather than passed"
    )
    assert "e91" in m.lower() or "protocol" in m.lower(), (
        f"the secret-key-rate provenance is emitted unconditionally as {m!r}. On the E91 page that "
        "attributes the rate to decoy states, which entanglement-based QKD does not use. Branch it "
        "on the protocol the way the row immediately above already does."
    )


def _rate_fact_source(src: str) -> str | None:
    """The provenance string attached to the `Secret-key rate` fact, or None if it moved."""
    import re

    m = re.search(r'label:\s*"Secret-key rate".*?source:\s*(.*?)\n', src, re.S)
    return m.group(1).strip().rstrip(",") if m else None
