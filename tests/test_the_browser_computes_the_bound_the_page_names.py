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

E91's SECURE-FRACTION FORM is left alone. `e91()` uses the same GLLP-on-observed-QBER form, and for
an entanglement-based protocol that is CORRECT: there are no decoy states in E91, because there is no
attenuated laser to put them on. The defect there is the label, not the formula.

**Its QBER is NOT left alone, since Round 54.** The harness already exported `e91Qbers` and nothing
asserted on them, so when Round 53 corrected the visibility composition in `qkd/e91.py` and not in
the browser's two copies, this file collected the disagreement and stayed green. That is the ninth
instance of one fix reaching one language, and the worst-directed one: the demo is CLIENT-SIDE ONLY,
so the Python copy it fixed does not run for a visitor and the two it missed do.
"""

from __future__ import annotations

import ast
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


def _require_browser_or_skip() -> None:
    """Skip when the browser toolchain is absent - unless the environment forbids skipping.

    THE SKIP WAS THE HOLE. These gates are the only ones that can see a change reaching one
    language and not the other, and they guarded on `if not ESBUILD.exists(): pytest.skip(...)`.
    `web/node_modules` is gitignored and the CI python job installed no Node, so on every GitHub
    run they reported SKIP and the workflow went green. Measured on a clone of the tip with no
    `web/node_modules`: **21 collected, 11 passed, 10 skipped** - 6 here and 4 in
    `test_the_qcnn_agrees_across_languages.py`. An earlier revision said 20 / 11 / 9, which was the
    count one commit before this docstring was written.

    Seeding the pre-Round-47 shared-scalar fully-connected block into `quantumSim.ts` fails FOUR of
    the QCNN gates locally - maximum readout differences 0.399170 (telecom-q7), 0.331032 (telecom),
    0.016233 (medical), 0.008699 (finance) - and passes the entire CI workflow. The single "0.585"
    an earlier revision quoted belongs to the PRE-RETRAIN weights of f5861c9, where the same seeding
    gives 0.585404 on telecom; the commit that published it retrained all six artifacts.

    `QSH_REQUIRE_BROWSER_GATES=1` turns the skip into a failure. CI sets it; a developer without
    node still gets the skip, with the message saying what is unchecked.
    """
    import os

    if ESBUILD.exists():
        return
    msg = (
        "web/node_modules/.bin/esbuild is absent, so the browser model cannot be executed. "
        "Run `npm ci` in web/. This is a SKIP rather than a pass: the identity is unchecked, "
        "not satisfied."
    )
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(
            msg + " QSH_REQUIRE_BROWSER_GATES=1 is set, so this is a FAILURE: the cross-language "
            "checks are mandatory in this environment and skipping them is what let a "
            "one-language fix ship green."
        )
    pytest.skip(msg)


def _cfg() -> dict:
    return json.loads((WEB / "src" / "lib" / "config.defaults.json").read_text())["qkd"]


def _browser() -> dict:
    """Run the browser's own `bb84()` and return what it produced.

    Bundled with esbuild exactly as `web/scripts/check_qkd.mjs` does, so this exercises the same
    module graph the browser gets rather than a hand-rolled re-import.
    """
    _require_browser_or_skip()
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
            # E91 was exported into this bundle by Round 45 and never called - the harness
            # imported the model it said it was not checking. It is evaluated now.
            + "const e91runs = [1, 2, 3, 4, 5].map((s) => m.e91(%d, chan, s));\n" % PULSES
            + "console.log(JSON.stringify({\n"
            "  rates: runs.map((r) => r.secretKeyRate),\n"
            "  bits: runs.map((r) => r.secureKeyBits),\n"
            "  sifted: runs.map((r) => r.siftedBits),\n"
            "  qbers: runs.map((r) => r.qber),\n"
            "  nonDecoy: runs.map((r) => r.secretKeyRateNonDecoy ?? null),\n"
            "  eta: m.transmittance(chan),\n"
            "  e91Rates: e91runs.map((r) => r.secretKeyRate),\n"
            "  e91Qbers: e91runs.map((r) => r.qber),\n"
            "  e91E0s: e91runs.map((r) => r.e0),\n"
            "  eDet: chan.misalignment,\n"
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
    return skr * _dead_time(eta)


def _dead_time(eta: float) -> float:
    """The dead-time factor both runs charge since Round 76 (Rusca et al. Eq. B3, the fibre clock).

    The bound the page names is the one-decoy bound on dead-time-corrected gains, which is the bare
    bound times this factor: every gain scales by it and the error rates do not.
    """
    import math

    from qsh.qkd.channel import ChannelParams, dead_time_factor

    c = _cfg()
    mu = float(c["mu_signal"])
    nu = min(max(1e-4, float(c["mu_decoy"])), mu * 0.9)
    y0 = 2.0 * float(c["dark_count_prob"])
    clicks = 0.5 * ((y0 + 1 - math.exp(-eta * mu)) + (y0 + 1 - math.exp(-eta * nu)))
    return dead_time_factor(ChannelParams(), clicks)


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


def _tracked_python() -> list[str]:
    """The corpus, asked of git rather than maintained by hand.

    ROUND 47 REWROTE THIS. The previous version iterated a three-element tuple of paths under a
    docstring reading "stated as a property rather than a file list, because a gate whose scope
    names files is this project's most-repeated defect". It was a file list. It could only ever
    fail on the files already known to be wrong, and in the same commit that wrote it Round 46
    raised a block size in the browser and left both Python twins on the old literal - inside two
    of the three files this very tuple names.

    Asking git means a file added tomorrow is in scope the moment it is committed.
    """
    out = subprocess.run(
        ["git", "ls-files", "--", "*.py"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.split()
    return [p for p in out if not p.startswith("tests/")]


def _run_bb84_call_sites() -> list[tuple[str, int, "ast.Call"]]:
    """Every `run_bb84(...)` call in tracked non-test Python, by AST rather than by regex.

    The regex the previous version used - `run_bb84\\((?:[^()]|\\([^()]*\\))*\\)` - cannot nest more
    than one level of parentheses, so a call whose channel argument gained a second nested call
    would silently stop matching and the site would leave the gate's scope without anyone editing
    the gate. The parser has no such depth limit.
    """
    sites: list[tuple[str, int, ast.Call]] = []
    for rel in _tracked_python():
        path = REPO / rel
        if not path.exists():
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:                                  # pragma: no cover - not our source
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name == "run_bb84":
                sites.append((rel, node.lineno, node))
    return sites


def test_the_scan_this_file_performs_has_a_subject() -> None:
    """Non-vacuity, first, because every check below is a loop over what this returns.

    A `git ls-files` that returns nothing - wrong cwd, no git, a submodule - would turn all three
    scope assertions green while asserting nothing at all. That failure mode is exactly why the
    file list was replaced, so it does not get to come back through the corpus.
    """
    corpus = _tracked_python()
    assert len(corpus) >= 20, (
        f"git ls-files returned only {len(corpus)} tracked non-test Python files; the scan below "
        "has effectively no subject. Check that REPO is a git worktree."
    )
    sites = _run_bb84_call_sites()
    assert len(sites) >= 2, (
        f"found {len(sites)} `run_bb84` call sites in {len(corpus)} files. The protocol entry "
        "point is reached from the orchestration registry, the satellite stage, the benchmark and "
        "the API at minimum; finding fewer means the AST walk is not seeing calls it should."
    )


def test_no_production_call_site_takes_the_legacy_branch() -> None:
    """Scope, so the next call site cannot repeat it.

    The numeric check above pins ONE path. This pins the class: any `run_bb84(` outside the tests
    must ask for the decoy model, either directly or through the registry provider.
    """
    offenders: list[str] = []
    for rel, line, node in _run_bb84_call_sites():
        if not any(kw.arg == "qkd_params" for kw in node.keywords):
            offenders.append(f"{rel}:{line}")
    assert not offenders, (
        "these production call sites take `run_bb84`'s LEGACY branch, which computes the non-decoy "
        "GLLP fraction on the observed QBER:\n  " + "\n  ".join(offenders)
        + "\n\nPass `qkd_params=QKDParams()`. The legacy branch is not wrong - it is a different, "
        "explicitly-named model that the optimiser's comparison tests still need - but nothing "
        "user-facing should reach it by omission."
    )


def test_no_call_site_hardcodes_its_block_size() -> None:
    """The block size is a security parameter, so config owns it and no call site may spell it.

    THIS IS THE CHECK ROUND 46 NEEDED AND DID NOT HAVE. `demo_alice_bob_pulses` was raised from
    40,000 to 400,000 with the reason written directly above it in `config/defaults.py`: at 40,000
    the decoy bound yields **79** secure bits at the intensities shipped now (94 at Round 46's
    mu = 0.5 / nu = 0.1), below the 256 a session key needs. The browser reads
    that key. Both Python twins kept the literal, so the deployed page derived a QKD-mixed session
    and the Python benchmark of the same protocol silently did not.

    A literal is what let the two drift. The number is not merely a tuning constant - it decides
    whether the demonstration demonstrates anything - and a value that appears in three places
    stays equal only for as long as someone remembers all three.
    """
    offenders: list[str] = []
    for rel, line, node in _run_bb84_call_sites():
        block = node.args[0] if node.args else next(
            (kw.value for kw in node.keywords if kw.arg == "n_pulses"), None)
        if isinstance(block, ast.Constant) and isinstance(block.value, int):
            offenders.append(f"{rel}:{line}: run_bb84({block.value:_}, ...)")
    assert not offenders, (
        "these call sites spell their BB84 block size as a literal:\n  " + "\n  ".join(offenders)
        + "\n\nRead it from config instead - `WebConfig.demo_alice_bob_pulses` for the two-party "
        "demo, `sat_pulses` for the satellite downlink - so raising it raises it everywhere. "
        "Round 46 raised the browser's copy and left two Python copies behind, which is the fourth "
        "time in four rounds that a fix changed one language."
    )


def test_the_python_two_party_demo_derives_the_session_key_the_browser_does() -> None:
    """The identity that matters to a visitor: both demos of the same protocol reach the same
    verdict about whether QKD contributed to the session.

    This is the numeric complement to the structural checks above, and it is the one that is
    actually red on the unmodified tree: at 40,000 pulses the decoy bound gives **79** secure bits
    at the shipped intensities (94 at Round 46's mu = 0.5 / nu = 0.1), `derive_shared_key` is
    skipped, `qkd_mixed` comes back False and the security score drops to 73.0 - while the browser,
    at the same nominal operating point, reports a mixed session.

    Asserted on the SHIPPED config value rather than on a number written here, so raising the
    block size again cannot leave this file behind in the same way it left the call sites.
    """
    import json as _json

    from qsh.benchmark.alice_bob_e2e import run_alice_bob_e2e

    web_cfg = _json.loads((WEB / "src" / "lib" / "config.defaults.json").read_text())
    browser_pulses = int(web_cfg["web"]["demo_alice_bob_pulses"])
    assert browser_pulses > 0, "the browser config no longer carries a two-party block size"

    res = run_alice_bob_e2e(seed=0)
    bits = int(res["qkd_secure_key_bits"])
    assert bits >= 256, (
        f"the Python two-party demo yields {bits} secure bits, below the 256 "
        "`derive_shared_key` needs, so `qkd_mixed` is "
        f"{res['qkd_mixed']} and the run scores {res['security_score']}. The browser runs the same "
        f"protocol at {browser_pulses:_} pulses and reports a mixed session. Two demonstrations of "
        "one protocol must not disagree about whether the quantum half contributed."
    )
    assert res["qkd_mixed"] is True, (
        f"{bits} secure bits were available and the session still did not mix them"
    )


def _eta_and_edet() -> tuple[float, float]:
    """The channel the assertions are made on, computed once from config.

    `e_det` is the MISALIGNMENT alone. It was `misalignment + 0.5 * p_dark / eta` - the observed
    QBER - until Round 47, which is the quantity `intrinsic_qber` returns and the wrong argument
    for the decoy bound: `_decoy_secure_fraction` adds the background back as `e0 * y0`, so the
    dark counts were charged twice. This reference helper carried the same error as the two
    implementations it was checking, which is why the identity held while both sides were wrong.
    """
    c = _cfg()
    eta = float(c["detector_efficiency"]) * 10 ** (
        -float(c["fiber_loss_db_per_km"]) * DISTANCE_KM / 10)
    return eta, float(c["misalignment"])


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


def test_the_e91_bound_does_not_move_with_the_seed_either() -> None:
    """E91 was held to a lower standard than BB84, and the harness could not see it.

    Round 45 made the BB84 bound seed-independent - it takes the intrinsic error rate, not the
    session's draw - and left `e91()` computing its secure fraction from the SAMPLED qber. So the
    E91 page's rate moved between reloads for exactly the reason the BB84 page's had.

    `src/qsh/qkd/e91.py:54` has always computed its fraction from the visibility-derived error rate
    and never from a sample, so the BROWSER was the divergent one. That is the shape this round is
    about: one language moved.

    The harness could not have caught it. `e91` was exported into the esbuild bundle by Round 45 and
    never called - the model the file's own docstring said it was "checked separately below" was
    imported and evaluated zero times. It is evaluated now.
    """
    b = _browser()
    rates = b.get("e91Rates")
    assert rates, "the harness no longer evaluates e91; it was imported-but-unused for a whole round"
    assert len(set(f"{r:.12f}" for r in rates)) == 1, (
        f"the E91 bound differs across seeds: {rates}. A secret-key bound is a property of the "
        "channel and the source visibility, not of one session's sampling noise."
    )
    assert len(set(f"{q:.9f}" for q in b["e91Qbers"])) > 1, (
        "the E91 measured QBER is now identical across seeds too, so the noise was removed rather "
        "than moved off the bound. The QBER is a measurement and must still vary."
    )


def test_the_decoy_bound_does_not_count_dark_counts_twice() -> None:
    """`e_det` in Ma et al. is the MISALIGNMENT, not the observed QBER.

    WHY THIS EXISTS

    `_decoy_secure_fraction` computes the observed error rate itself:

        e_mu = (e0 * y0 + e_det * (1 - exp(-eta * mu))) / q_mu

    The `e0 * y0` term IS the dark-count contribution - background clicks are unbiased, so they
    err half the time. `e_det` is the probability that a detected SIGNAL photon lands in the wrong
    detector, which is the misalignment and nothing else.

    Both languages passed `intrinsic_qber(params)`, which is
    `misalignment + 0.5 * dark_count_prob / eta` - the dark-count term already folded in. So the
    background was charged twice and the bound came out too low. `intrinsic_qber` is the right
    quantity for the OBSERVED QBER the panel displays, and both files still use it there; it is
    only wrong as this function's argument.

    Measured cost, in the conservative direction, WITH THE INTENSITIES NAMED - an earlier revision
    of this docstring gave "0.57% and 18.9%" without them, and 18.9% is not a "low by" figure at any
    intensity pair. It is the same gap with the WRONG bound as denominator, i.e. how much the fix
    RAISES the number, which is the convention `docs/verification-log.md` uses.

        mu = 0.5    / nu = 0.1     (in force while the double charge shipped):
            fibre 50 km       0.5673% low   (0.5705% raise)
            800 km downlink  15.9152% low  (18.9276% raise)
        mu = 0.3416 / nu = 0.1417  (SHIPPED NOW, the finite-key optimum):
            fibre 50 km       0.4984% low   (0.5009% raise)
            800 km downlink  13.6638% low  (15.8262% raise)

    The body below computes with the config's own `mu_signal` / `mu_decoy`, so what it demonstrates
    is the 0.4984% figure and not the retired 0.57%. eta is 17.5x smaller on the downlink than on
    the fibre and the `0.5 * p_dark / eta` term is correspondingly larger, which is why the same
    defect costs an order of magnitude more there. A bound that understates itself is safe but not
    correct, and the size of the error grows exactly where the link is hardest.

    ASSERTED NUMERICALLY, not by grepping the call sites. A textual version of this check caught
    Python and missed the browser: `quantumSim.ts` assigns `const eDet = intrinsicQber(p)` on one
    line and passes `eDet` on another, so the argument's provenance is invisible to a regex over
    the call span. That is the same blind spot a helper-taking-a-key created in Round 46. Two
    numbers from two languages cannot hide behind a local variable.
    """
    from qsh.qkd.bb84 import _decoy_secure_fraction

    c = _cfg()
    eta = float(c["detector_efficiency"]) * 10 ** (
        -float(c["fiber_loss_db_per_km"]) * DISTANCE_KM / 10)
    mu = float(c["mu_signal"])
    nu = min(max(1e-4, float(c["mu_decoy"])), mu * 0.9)
    y0 = 2.0 * float(c["dark_count_prob"])
    f_ec = float(c["error_correction_eff"])
    mis = float(c["misalignment"])
    double_counted = mis + 0.5 * float(c["dark_count_prob"]) / max(eta, 1e-12)

    right = _decoy_secure_fraction(eta, mu, nu, mis, y0, f_ec, 0.5)[0] * _dead_time(eta)
    wrong = _decoy_secure_fraction(eta, mu, nu, double_counted, y0, f_ec, 0.5)[0] * _dead_time(eta)
    assert right > wrong > 0, (
        "the two readings of `e_det` are indistinguishable at this operating point, so this check "
        f"cannot tell them apart: misalignment-only {right}, double-counted {wrong}. Non-vacuity "
        "first - without a gap there is nothing to assert."
    )

    got = _python_production_rate()
    assert abs(got - right) <= 1e-12, (
        f"the production Python path reports {got:.8f} bit/pulse; the decoy bound with `e_det` as "
        f"the MISALIGNMENT gives {right:.8f} and with the observed QBER (dark counts charged "
        f"twice) gives {wrong:.8f}. `e0 * y0` inside `_decoy_secure_fraction` is already the "
        "background term."
    )
    b = _browser()
    assert abs(b["rates"][0] - right) <= 1e-9, (
        f"the browser reports {b['rates'][0]:.8f} against {right:.8f}. If it matches "
        f"{wrong:.8f} instead, it is still passing `intrinsicQber` where the misalignment belongs "
        "- which is what Python was doing until this round, in the other language."
    )


#: Eve fractions the sweep below is made at. 0.5 is well past the textbook intercept-resend point
#: at which BB84 yields nothing, so a key surviving there is not a modelling nuance.
EVE_SWEEP = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)


def _browser_eve_sweep() -> list[dict]:
    """The browser's own bb84() across the eavesdropper control, one run per setting."""
    _require_browser_or_skip()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        entry, bundle, runner = d / "entry.ts", d / "bundle.mjs", d / "run.mjs"
        entry.write_text(
            "export { bb84, FIBER } from "
            f"{json.dumps(str(WEB / 'src/lib/quantumSim'))};\n")
        subprocess.run(
            [str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node",
             f"--outfile={bundle}", "--log-level=error"],
            cwd=WEB, check=True, capture_output=True)
        runner.write_text(
            f"import * as m from {json.dumps(bundle.as_uri())};\n"
            f"const eves = {json.dumps(list(EVE_SWEEP))};\n"
            f"const out = eves.map((e) => {{\n"
            f"  const ch = {{ ...m.FIBER, distanceKm: {DISTANCE_KM}, eveFraction: e }};\n"
            f"  const r = m.bb84({PULSES}, ch, 1);\n"
            "  return { eve: e, qber: r.qber, skr: r.secretKeyRate, bits: r.secureKeyBits };\n"
            "});\n"
            "console.log(JSON.stringify(out));\n")
        out = subprocess.run(["node", str(runner)], check=True, capture_output=True, text=True)
        return json.loads(out.stdout)


def test_an_eavesdropper_destroys_the_key_in_both_languages() -> None:
    """The eavesdropper control must reach the number it is a control for.

    WHY THIS EXISTS

    Round 47 corrected `e_det` from `intrinsic_qber` to the misalignment, because
    `_decoy_secure_fraction` adds the dark-count contribution back itself and the background was
    being charged twice. That correction was right about dark counts and wrong about Eve.

    `intrinsic_qber` is `misalignment + 0.5 * p_dark / eta + 0.25 * eve_intercept_fraction`. It was
    the ONLY path by which the eavesdropper fraction reached the secret-key bound - `transmittance`
    does not depend on it, and `E_mu` is modelled from `e_det` rather than measured from the
    sampled QBER. So removing it removed the dark-count double charge AND unhooked the
    eavesdropper, and the browser went on publishing a full-strength key under a full
    intercept-resend attack.

    MEASURED on the unmodified tree, 200,000 pulses at 50 km, seed 1:

        eve   browser qber   browser skr     browser secure bits
        0.0   0.0137         0.001976547     395
        0.5   0.1360         0.001976547     395
        1.0   0.2598         0.001976547     395   <- unchanged at every setting

    Python survived only by accident: `run_bb84` aborts when the SAMPLED qber exceeds
    `pp_threshold`, so the key died at eve ~0.35 through a threshold test rather than through the
    bound. `quantumSim.ts` has no such abort, so nothing stopped it at all.

    The fix is to put Eve back where the physics puts her. `e_det` in Ma et al. is the probability
    that a DETECTED SIGNAL photon lands in the wrong detector, and intercept-resend causes exactly
    that, at the textbook rate of 0.25 per intercepted pulse. Dark counts stay in the separate
    `e0 * y0` term, so the double charge does not come back.
    """
    py = []
    from qsh.qkd.bb84 import QKDParams, run_bb84
    from qsh.qkd.channel import ChannelParams

    for eve in EVE_SWEEP:
        r = run_bb84(PULSES, ChannelParams(distance_km=DISTANCE_KM, eve_intercept_fraction=eve),
                     seed=1, qkd_params=QKDParams())
        py.append({"eve": eve, "qber": r.qber, "skr": r.secret_key_rate,
                   "bits": r.secure_key_bits})
    br = _browser_eve_sweep()

    # NON-VACUITY: an eavesdropper must actually disturb the channel, or nothing below has a subject.
    assert py[-1]["qber"] > py[0]["qber"] + 0.1, (
        f"a full intercept-resend attack moved the Python QBER from {py[0]['qber']:.4f} to "
        f"{py[-1]['qber']:.4f}; the channel model is not modelling Eve at all"
    )
    assert br[-1]["qber"] > br[0]["qber"] + 0.1, "the browser channel model is not modelling Eve"

    for lang, rows in (("Python", py), ("browser", br)):
        rates = [r["skr"] for r in rows]
        assert rates == sorted(rates, reverse=True), (
            f"{lang}: the secret-key rate is not monotonically non-increasing in the eavesdropper "
            f"fraction: {[f'{x:.6f}' for x in rates]}"
        )
        dead = [r for r in rows if r["eve"] >= 0.5]
        assert all(r["skr"] == 0.0 and r["bits"] == 0 for r in dead), (
            f"{lang} still publishes a secret key at eve >= 0.5:\n  "
            + "\n  ".join(f"eve={r['eve']}: qber={r['qber']:.4f} skr={r['skr']:.9f} "
                          f"bits={r['bits']}" for r in dead)
            + "\n\nIntercept-resend on half the pulses puts the QBER far above any threshold at "
            "which BB84 yields a key. If the rate has not moved at all, the eavesdropper fraction "
            "is not reaching the bound - check what `e_det` is being passed."
        )

    for a, b in zip(py, br, strict=True):
        assert abs(a["skr"] - b["skr"]) <= 1e-9, (
            f"at eve={a['eve']} Python reports {a['skr']:.9f} bit/pulse and the browser "
            f"{b['skr']:.9f}. The eavesdropper must reach both bounds the same way; one language "
            "aborting on a sampled QBER while the other carries Eve in the bound is not agreement."
        )


def test_the_browser_e91_qber_is_the_one_python_computes() -> None:
    """Both languages must degrade the Werner visibility the same way.

    WHY THIS IS A SEPARATE ASSERTION from the decoy identity above: E91's secure-fraction FORM is
    deliberately not compared, because GLLP-on-observed-QBER is correct for an entanglement protocol.
    What must agree is the OBSERVED QBER itself, which is pure channel physics and has no protocol
    freedom in it.

    THE DEFECT THIS PINS. A channel that flips a bit with probability e has visibility `(1 - 2e)`,
    not `(1 - e)` - visibility spans [-1, 1] of correlation while an error probability spans [0, 1/2]
    of QBER. All three implementations used `(1 - e)` and understated QBER by 24.8%, always in the
    demo's favour. Round 53 fixed `qkd/e91.py`; `clientOptiQKD.ts` and `quantumSim.ts` kept the bug
    for one commit, and because this demo runs client-side only, the fixed copy was the one nobody
    executes.

    Checked against independent composition rather than against the other implementation's algebra,
    so agreeing on a shared mistake cannot satisfy it: two independent bit-flip sources compose as
    `q = q_V + e - 2*q_V*e`, and `(1 - V(1-2e))/2` equals that exactly, not to first order.
    """
    b = _browser()
    got = b["e91E0s"][0]

    from qsh.qkd.channel import ChannelParams, intrinsic_qber

    c = _cfg()
    # The same channel the harness hands the browser, built the way `_python_production_rate` does.
    e = intrinsic_qber(ChannelParams(
        distance_km=DISTANCE_KM,
        misalignment=float(c["misalignment"]),
        eve_intercept_fraction=0.0,
    ))
    v = float(c["e91_visibility"])
    expected = min(0.5, (1.0 - max(0.0, min(1.0, v * (1.0 - 2.0 * e)))) / 2.0)
    independent = (1.0 - v) / 2.0 + e - 2.0 * ((1.0 - v) / 2.0) * e
    assert abs(expected - independent) < 1e-12, (
        f"the Python form {expected:.9f} and independent composition {independent:.9f} disagree; "
        "the correction itself needs re-deriving before this comparison means anything")
    assert abs(got - expected) <= 1e-9, (
        f"the browser reports E91 QBER {got:.9f} where Python computes {expected:.9f}. If the "
        f"browser is low by about 24.8%, it is still composing visibility as `(1 - e)` instead of "
        "`(1 - 2e)` - check `clientOptiQKD.ts` and `quantumSim.ts`, BOTH of which carry a copy."
    )
