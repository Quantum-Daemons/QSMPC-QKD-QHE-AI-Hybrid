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


def _produced() -> set[str]:
    return set(_PRODUCER.findall(TS.read_text()))


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
