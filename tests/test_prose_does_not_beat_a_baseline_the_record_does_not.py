"""A docstring must not claim a model beats a number the shipped record puts it below.

WHY THIS EXISTS

`src/qsh/models/distil.py` and `scripts/train/satellite_opssat.py` both stated, of
`satellite_comms.opssat_student`:

    "minority-class F1 0.8485 -> 0.9391 and AUC-PR 0.9235 -> 0.9810, which beats both the teacher
     (0.9189) and the published baseline (0.979)"

Against `models/metrics.json`, three things are wrong and they compound:

    auc_pr            0.968055    claimed 0.9810
    minority_class_f1 0.904348    claimed 0.9391
    teacher auc_pr    0.968107    the student is BELOW it, not above
    published         0.979       0.968055 is BELOW it

So the strongest claim - beating the published baseline - is **false**. "Beats the teacher (0.9189)"
compares the teacher's F1 against the student's AUC-PR, which is the two-fields-one-word trap. And
the degree-1 "before" figures are recorded in no committed artefact, so they cannot be checked at all.

WHAT IS ASSERTED

That for every model whose registry carries a published `metric.baseline`, no tracked source file
claims that model beats or passes that baseline unless the achieved value actually exceeds it.

WHAT IS DELIBERATELY NOT ASSERTED

Not that any number is right - the trainers decide those. Not that prose may never mention a
baseline; naming one and falling short of it is this project's normal, honest posture and most of the
registry does exactly that. Only that "beats" is not written where the record says "below".

HOW A WITHDRAWAL IS TOLD APART FROM A CLAIM: AN EXPLICIT MARKER, NOT A HEURISTIC

A correction has to quote what it withdraws, so the corrected text contains the very sentence the gate
looks for. The first version told them apart by counting double quotes from the start of the comment
block, and that is not a sound discriminator in Python source: apostrophes, string literals and
docstrings put quote characters everywhere, and one stray quote flips every verdict after it. It
misfired on the corrected text it was written to protect.

So the convention is explicit. A retracted claim is quoted between two marker lines,
`WITHDRAWN-CLAIM-BEGIN` and `WITHDRAWN-CLAIM-END` (in a comment, `# WITHDRAWN-CLAIM-BEGIN`). The gate
removes those spans before it scans, and nothing else is exempt. A dangling BEGIN would silence the rest
of the file, so the markers must balance, and a span must hold a quotation - an empty or unquoted span
would be an exemption with nothing to withdraw.
"""

from __future__ import annotations

import re
from pathlib import Path

from qsh.models.registry import achieved, specs

REPO = Path(__file__).resolve().parents[1]

#: Words that assert superiority over a named figure. Deliberately narrow: "against", "compared to"
#: and "short of" are the honest forms this project uses everywhere and must not trip the gate.
BEATS = re.compile(r"\b(beat|beats|beating|passes|passing|surpass\w*|exceeds?|outperforms?)\b", re.I)

#: Where prose about models lives. `.md` is excluded on purpose - documents carry long historical
#: narratives that quote superseded claims in order to withdraw them, and this gate would fire on the
#: withdrawal. Source files carry the operative statement.
SOURCE_GLOBS = ("src/**/*.py", "scripts/**/*.py")


def _sources():
    for pat in SOURCE_GLOBS:
        for p in sorted(REPO.glob(pat)):
            yield p, p.read_text()


BEGIN, END = "WITHDRAWN-CLAIM-BEGIN", "WITHDRAWN-CLAIM-END"
_SPAN = re.compile(rf"{BEGIN}.*?{END}", re.S)


def _without_withdrawals(src: str) -> str:
    """Blank every marked withdrawal, keeping line breaks so line numbers stay true."""
    return _SPAN.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), src)


def test_the_withdrawal_markers_are_well_formed():
    """Balanced, non-empty, and quoting something - otherwise the exemption is a loophole."""
    problems = []
    for path, src in _sources():
        if src.count(BEGIN) != src.count(END):
            problems.append(f"{path.relative_to(REPO)}: {src.count(BEGIN)} BEGIN against {src.count(END)} END")
            continue
        for m in _SPAN.finditer(src):
            body = m.group(0)[len(BEGIN):-len(END)]
            # A PAIRED quotation: an opening and a closing mark. One stray quote character is not a
            # quotation, and checking for "any quote" let a span with its opening mark deleted pass.
            if body.count('"') < 2 or body.count('"') % 2:
                line = src[: m.start()].count("\n") + 1
                problems.append(f"{path.relative_to(REPO)}:{line} marks a span with no paired quotation in it")
    assert not problems, "malformed withdrawal markers:\n  " + "\n  ".join(problems)


def test_the_scan_has_subjects():
    """Anti-tautology: no baselines, or no sources, would make the gate vacuous."""
    with_baseline = [s.key for s in specs()
                     if s.metric and isinstance(s.metric.baseline, (int, float))]
    assert len(with_baseline) >= 5, f"only {len(with_baseline)} models carry a published baseline"
    assert sum(1 for _ in _sources()) > 20, "the source scan found almost nothing; re-derive the globs"


def test_no_source_claims_a_model_beats_a_baseline_it_is_below():
    """The property: prose and record agree on the direction of the comparison."""
    offenders = []
    for spec in specs():
        m = spec.metric
        if not m or not isinstance(m.baseline, (int, float)):
            continue
        rec = achieved(spec.key) or {}
        got = rec.get(m.name)
        if not isinstance(got, (int, float)) or got >= m.baseline:
            continue                      # no baseline, not measured, or genuinely above it

        for path, raw in _sources():
            src = _without_withdrawals(raw)
            for line_no, line in enumerate(src.splitlines(), 1):
                if not BEATS.search(line):
                    continue
                # The baseline must be named ON or NEAR the boast for this to be that claim.
                window = "\n".join(src.splitlines()[max(0, line_no - 4): line_no + 3])
                # WHOLE-NUMBER MATCH, not a substring. The first version used `str(baseline) in
                # window` and fired on the CORRECTED text, because telemanom's baseline `0.968` is a
                # substring of the opssat student's measured `0.968055` and `0.968107`. A gate that
                # flags a correction as the defect it corrected gets switched off - that is the
                # fifth probe of mine this round to trip on its own loose matching.
                if not re.search(rf"(?<![\d.]){re.escape(str(m.baseline))}(?![\d])", window):
                    continue
                # A marked withdrawal was blanked above; anything still here is an assertion.
                offenders.append(
                    f"{path.relative_to(REPO)}:{line_no} claims {spec.key} beats {m.baseline}, "
                    f"but the record has {m.name} = {got}")

    assert not offenders, (
        "these sources claim a model beats a published baseline the record puts it below:\n  "
        + "\n  ".join(offenders)
        + "\n\nNaming a baseline and falling short of it is this project's normal posture and most "
        "of the registry does exactly that. Writing 'beats' where the record says 'below' is the "
        "one thing that is not allowed - and it survived in TWO source files carrying the identical "
        "sentence, so check for a twin before calling it fixed."
    )


def test_the_opssat_student_case_is_pinned_by_its_numbers():
    """The specific instance, so the general gate above cannot quietly stop covering it."""
    rec = achieved("satellite_comms.opssat_student") or {}
    spec = next(s for s in specs() if s.key == "satellite_comms.opssat")
    assert rec.get("auc_pr") is not None, "the student no longer records auc_pr; re-derive"
    assert rec["auc_pr"] < spec.metric.baseline, (
        f"the student's auc_pr {rec['auc_pr']} now meets or exceeds the published "
        f"{spec.metric.baseline}. If a retrain genuinely achieved that, this test and the corrected "
        "docstrings should be re-derived together - but check the estimator first, because the last "
        "time this was claimed the figure came from nowhere reproducible.")
    assert rec.get("student_metric") is not None and rec.get("teacher_metric") is not None
    assert rec["student_metric"] < rec["teacher_metric"], (
        "the student's decision metric now exceeds the teacher's; the corrected prose says it tracks "
        "rather than beats, and that would need updating with the measurement that changed it.")
