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
    the decoy bound yields 94 secure bits, below the 256 a session key needs. The browser reads
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
    actually red on the unmodified tree: at 40,000 pulses the decoy bound gives 94 secure bits,
    `derive_shared_key` is skipped, `qkd_mixed` comes back False and the security score drops to
    73.0 - while the browser, at the same nominal operating point, reports a mixed session.

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

    Measured cost, in the conservative direction: 0.57% on the 50 km fibre, where eta is 0.025,
    and **18.9%** on the 800 km free-space downlink, where eta is 17x smaller and the
    `0.5 * p_dark / eta` term is correspondingly larger. A bound that understates itself is safe
    but not correct, and the size of the error grows exactly where the link is hardest.

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

    right = _decoy_secure_fraction(eta, mu, nu, mis, y0, f_ec, 0.5)[0]
    wrong = _decoy_secure_fraction(eta, mu, nu, double_counted, y0, f_ec, 0.5)[0]
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
