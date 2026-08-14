"""Display prose decides a score here, so it must not be editable by accident.

WHY THIS EXISTS

`security_model` is a human-readable string rendered in the benchmark table's last column. It is
ALSO parsed, in two languages, to decide a number:

  - `web/src/lib/clientCompute.ts:302`  `(o.securityModel || "").toLowerCase().includes("information-theoretic")`
    -> selects `s.info_theoretic` or `s.computational` for the security score
  - `web/src/lib/clientCompute.ts:342`  same test -> `qa.info_theoretic_multiplier` in the quantum-advantage
  - `src/qsh/benchmark/security.py:48`  `"information-theoretic" in (security_model or "").lower()`

So an editor improving the wording of a rendered sentence moves a displayed score, in the direction
nobody chose, with every gate green. Round 40 walked into exactly this: correcting the "QOTP
perfect-hiding" claim looked like a prose fix, and deleting the phrase from `security_model` would
have silently dropped the quantum row's security score and its advantage multiplier - for a reason
that does not hold, because the quantum one-time pad IS information-theoretically hiding as a scheme.
What was wrong was never the label; it was that nothing beside it disclosed that the demo's mask is
drawn from a seeded, non-secret generator.

WHY THIS PINS THE MAPPING RATHER THAN REFACTORING THE SCORERS

The obvious fix is a `security_class` enum the scorers read instead of prose. That is five producer
sites in TypeScript plus the Python twin plus the table rendering - a scoring change, in the round
whose plan says a scoring change is where a number moves by accident. So this pins the observable
property instead: the set of security-model strings is closed, and each maps to the class intended
for it. Adding or rewording one fails here, which forces the decision to be deliberate. If the enum
refactor is done later, this test is what proves it changed no score.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TS = REPO / "web" / "src" / "lib" / "clientCompute.ts"
PY = REPO / "src" / "qsh" / "benchmark" / "security.py"

#: The Python files that PRODUCE a security-model string. Round 40 read only `clientCompute.ts`, so
#: rewording any of these moved the Python security score and the quantum-advantage multiplier with
#: every gate green - the exact scenario this file's docstring says it exists to prevent, on the
#: other side of the language boundary.
_PY_PRODUCERS = [
    REPO / "src" / "qsh" / "benchmark" / "quantum_qsmpc_fhe.py",
    REPO / "src" / "qsh" / "benchmark" / "classical_mpc_fhe.py",
    REPO / "src" / "qsh" / "benchmark" / "gpu_mpc_adapter.py",
    REPO / "src" / "qsh" / "benchmark" / "alice_bob_e2e.py",
]
#: Every place either language TESTS the string to pick a number. `quantum_advantage.py` is the
#: fourth and was unpinned: it is the Python twin of `clientCompute.ts:342`.
_PARSE_SITES = [
    (TS, 'includes("information-theoretic")', 2),
    (PY, '"information-theoretic" in (security_model or "").lower()', 1),
    (REPO / "src" / "qsh" / "benchmark" / "quantum_advantage.py", '"information-theoretic" in (', 1),
]

#: Every security-model string the code may produce, and the class each must score as.
#: "it" = information-theoretic (the higher-scoring branch), "comp" = computational, "none" = neither.
EXPECTED: dict[str, str] = {
    "information-theoretic (QOTP perfect hiding)": "it",
    "computational (RLWE) + secret-sharing MPC": "comp",
    "computational (RLWE / FIPS-203 family)": "comp",
    "computational + QKD (two-party, TTP-free)": "comp",
    "none (plaintext reference)": "none",
}

#: `security_model: "..."` and `securityModel: "..."` producers.
_PRODUCER = re.compile(r"security_?[Mm]odel:\s*\"([^\"]+)\"")
#: The substring both languages test for.
_TOKEN = "information-theoretic"


#: `security_model="..."` - the Python producers' form.
_PY_PRODUCER = re.compile(r"security_model\s*=\s*\"([^\"]+)\"")


def _produced() -> set[str]:
    return set(_PRODUCER.findall(TS.read_text()))


def _produced_py() -> set[str]:
    out: set[str] = set()
    for path in _PY_PRODUCERS:
        out |= set(_PY_PRODUCER.findall(path.read_text()))
    return out


def test_the_producer_parse_finds_something() -> None:
    """A regex that matched nothing would make every assertion below vacuous."""
    found = _produced()
    assert len(found) >= 4, f"only {len(found)} security-model strings parsed from {TS.name}: {found}"


def test_the_set_of_security_model_strings_is_closed() -> None:
    """New or reworded strings must be declared here, because they decide a score."""
    found = _produced()
    unexpected = sorted(found - EXPECTED.keys())
    vanished = sorted(EXPECTED.keys() - found)
    assert not unexpected, (
        f"security-model strings not declared in this test: {unexpected}\n"
        "These are parsed for `information-theoretic` at clientCompute.ts:302 and :342 and at "
        "security.py:48, so each one silently selects a score branch. Add it to EXPECTED with the "
        "class you intend, so the choice is recorded rather than inferred from wording."
    )
    assert not vanished, (
        f"declared security-model strings no longer produced: {vanished}\n"
        "If a row was removed that is fine - delete it here too, in the same commit, so this list "
        "keeps describing the code."
    )


def test_each_string_scores_as_the_class_intended_for_it() -> None:
    """The substring test both languages run must agree with the declared class."""
    wrong = []
    for text, cls in EXPECTED.items():
        scores_it = _TOKEN in text.lower()
        if scores_it != (cls == "it"):
            wrong.append(f"{text!r} declared {cls!r} but the {_TOKEN!r} test says {scores_it}")
    assert not wrong, (
        "a security-model string scores as a different class than declared:\n  " + "\n  ".join(wrong)
        + "\n\nThis is the failure the test exists for: the wording and the score disagree, and only "
        "the wording is visible."
    )


def test_the_python_producers_are_in_the_same_closed_set() -> None:
    """The other side of the boundary Round 40's version stopped at.

    Rewording `quantum_qsmpc_fhe.py`'s string drops the Python security score AND the advantage
    multiplier, and the TypeScript-only version of this test could not see it.
    """
    found = _produced_py()
    assert len(found) >= 3, f"only {len(found)} Python security-model strings parsed: {found}"
    unexpected = sorted(found - EXPECTED.keys())
    assert not unexpected, (
        f"Python produces security-model strings not declared here: {unexpected}\n"
        "They are parsed for `information-theoretic` at security.py:48 and quantum_advantage.py:41, "
        "so each silently selects a score branch. Declare it with the class you intend."
    )


def test_every_parse_site_is_pinned() -> None:
    """A fourth site existed and nothing named it.

    `quantum_advantage.py:41` is the Python twin of `clientCompute.ts:342`. Round 40's version
    asserted the token in `security.py` alone, so the two PYTHON scorers could have diverged from
    each other with every gate green.
    """
    wrong = []
    for path, needle, count in _PARSE_SITES:
        got = path.read_text().count(needle)
        if got != count:
            wrong.append(f"{path.name}: expected {count} occurrence(s) of {needle!r}, found {got}")
    assert not wrong, (
        "a security-model parse site changed:\n  " + "\n  ".join(wrong)
        + "\n\nIf the scorers were refactored onto an explicit class field, retire these assertions "
        "and assert instead that the refactor moved no score."
    )


def test_both_languages_still_key_on_the_same_token() -> None:
    """If one side's token changes, the two implementations score the same row differently."""
    ts, py = TS.read_text(), PY.read_text()
    assert ts.count(f'includes("{_TOKEN}")') == 2, (
        "clientCompute.ts no longer has exactly two `information-theoretic` score tests; if the "
        "scorers were refactored onto an explicit class field, retire this test and assert the "
        "refactor moved no score instead"
    )
    assert f'"{_TOKEN}" in (security_model or "").lower()' in py, (
        "src/qsh/benchmark/security.py no longer keys on the same token as the browser; the two "
        "implementations of one score have diverged - the defect class this repository has shipped "
        "five times across language twins"
    )
