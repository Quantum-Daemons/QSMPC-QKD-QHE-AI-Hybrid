#!/usr/bin/env python
"""Emit the report's tables as LaTeX, generated from live artefacts.

Every number in the technical report comes from here, so the report cannot drift from the code the
way prose does. This project has already found stale figures in five documents at once; a report is
a worse place for that than a README because it is a fixed artefact someone keeps.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "report" / "generated"
OUT.mkdir(parents=True, exist_ok=True)

from qsh.models import registry  # noqa: E402

M = json.loads((REPO / "models" / "metrics.json").read_text())


def esc(s: str) -> str:
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, float) else ("--" if v is None else str(v))


def model_table(specs=None) -> str:
    """One row per published model. The licence column carries the DATASET NAME, not the licence.

    IT USED TO CARRY `s.dataset.licence.split(";")[0][:22]` - two truncations in one expression, both
    cutting from the end, which is where a hedge lives. Measured over the seven licences that reach
    this table, four were altered and three lost their qualifier:

        "CC-BY-4.0 asserted by MedMNIST; the upstream BUSI grant is NOT verifiable"
            -> "CC-BY-4.0 asserted by "   the `;` took the caveat, [:22] took the attributor
        "per-score; EMBL-EBI Terms of Use by default"  -> "per-score"
        "provider's own pages state conflicting terms" -> "provider's own pages s"

    "CC-BY-4.0 asserted by " tells a reader the corpus is CC-BY-4.0. The sentence exists to say that
    the upstream grant behind that assertion is NOT verifiable. **Truncation is biased toward
    overclaiming, because the caveat comes last** - and this was published in the PDF.

    The column is now the dataset name, which is short by nature and needs no cutting, and the
    licences are printed whole by `licence_table()` below. The licence is a property of the DATASET,
    which is why the old column repeated and re-truncated the same string once per model.

    `specs` is a parameter so a test can render a synthetic spec and check the rendering itself,
    rather than only checking the committed output.
    """
    rows = []
    for s in sorted(specs if specs is not None else registry._SPECS, key=lambda x: (x.usecase, x.tier)):
        a = M.get(s.key) or {}
        got = a.get(s.metric.name, a.get("student_metric"))
        tier = {"t1_crypto": "T1", "t2_industry": "T2", "t3_large": "T3"}.get(s.tier, s.tier)
        rows.append(" & ".join([
            r"\texttt{" + esc(s.key) + "}", tier, esc(s.metric.display),
            fmt(got), fmt(s.metric.baseline), esc(s.dataset.name),
        ]) + r" \\")
    return "\n".join(rows)


def licence_table(specs=None) -> str:
    """Every dataset licence that reaches the report, in full, once each.

    Deduplicated by dataset, because a licence belongs to the corpus and not to each model trained on
    it. Sorted so the output is stable across runs; `p{}` columns in the wrapper let the long strings
    wrap rather than be cut, which is the whole point of the table.
    """
    seen: dict[str, str] = {}
    for s in specs if specs is not None else registry._SPECS:
        seen.setdefault(s.dataset.name, s.dataset.licence)
    return "\n".join(
        " & ".join([esc(name), esc(lic)]) + r" \\" for name, lic in sorted(seen.items())
    )


def crypto_table() -> str:
    # The FHE row is DERIVED, because a generator that hardcodes the parameter set it is describing
    # publishes the old one the moment the code changes. Both numbers here were literals - "ring
    # degree 8192" and "4096 slots" - written inside this function, matching a literal in
    # `clientFHE.ts` and a default argument in `fhe/backends.py`, none of which read
    # `FHEConfig.poly_modulus_degree`. Raising the ring would have left this table asserting a
    # parameter set nothing runs.
    from qsh.config import get_config

    fhe = get_config().fhe
    rows = [
        ("KEM", "ML-KEM-768 + X25519", "FIPS 203 hybrid; ct 1088 B"),
        ("Signature", "ML-DSA-65", "FIPS 204; sig 3309 B"),
        ("FHE", f"CKKS, ring degree {fhe.poly_modulus_degree}",
         f"{fhe.poly_modulus_degree // 2} slots, one multiplicative level per dot product"),
        ("MPC", "Garbled circuits (Polytune)", "non-linear layers: ReLU, argmax"),
        ("QKD", "BB84 two-intensity decoy", "GLLP tagged fraction"),
    ]
    return "\n".join(" & ".join([esc(a), r"\texttt{" + esc(b) + "}", esc(c)]) + r" \\" for a, b, c in rows)


def findings_table() -> str:
    """The negative-results chapter, as data rather than prose."""
    rows = [
        ("Constant predictor", "medical CKKS student emitted one class for all 156 test images",
         "accuracy 0.7308 equalled the majority rate exactly; AUC 0.354, below random"),
        ("Inverted class labels", "four sites asserted the opposite of MedMNIST's own metadata",
         "the demo displayed the opposite diagnosis to the model's output"),
        ("Sampler misalignment", "shuffled DataLoader output paired positionally with dataset order",
         "teacher/ground-truth agreement 0.6227, inside the permutation null"),
        ("Wrong baseline row", "cited ESA-ADB Table 2 (all events) for an anomalies-only task",
         "like-for-like row is Supplementary Table 9 at 0.968, not 0.786"),
        ("Uncorrected metric", "reported uncorrected event-wise F0.5 against a corrected baseline",
         "ESA-ADB multiplies event precision by the time-level TNR"),
        ("Licence divergence", "published artefact recorded ImageNet weights; two docs said random",
         "publish now refuses any model whose manifest records a pretrained source"),
        ("Guard too narrow", "both\\_scored\\_zero required BOTH models at zero",
         "teacher 0 vs student 0.0032 rendered as the encrypted model beating its teacher"),
        ("Manufactured margin", "treat-all evaluated at a different threshold from the model",
         "a zero margin reported as +0.2308; caught by the degeneracy guard"),
        ("Leak in a published sample",
         "finance eval sample: hour\\_of\\_day >= 3 reproduced the label at 100\\% precision",
         "served from an HF revision whose pin looked valid; the first two leak guards were "
         "themselves wrong"),
        ("Guard exempted by its data", "telecom feature\\_names held CLASS names, not feature names",
         "the leak guard skipped the file and reported success on something it never examined"),
        ("Half-applied fix", "QCNN repaired in the browser; the Python path stayed a constant predictor",
         "a coin flip looks broken, a constant predictor looks confident and reproducible"),
        ("Crossed wiring", "the trainer's CNOT ring indexed a stale axis after chained movedim",
         "training still converged; only an independent reimplementation disagreed, by 0.73 on a "
         "quantity bounded in [-1,1]"),
        ("Feature-space mismatch", "circuit trained on min-max raw units, served StandardScaler z-scores",
         "seven of eight features pinned to -1; a degenerate state still returned a confident class"),
        ("Selection on the test set", "epoch chosen and headline reported on the same rows",
         "medical F1 fell 0.9181 to 0.7326 once a validation split was introduced"),
        ("Criterion half-applied", "loss-aware QSDC bound added to the module, not to its caller",
         "an inert threshold control, and an eavesdropper reported on every clean run"),
        ("False provenance",
         "a panel announced that the quantum circuit produced a class CKKS had produced",
         "it read the CONTROL value; the artifact recording which backend ran was never consulted"),
        ("Tautological verdict",
         "circuit\\_private rested half on decrypt(encrypt(x)) == x, which is 1.0 by construction",
         "a passing test asserted both the tautology and the verdict, so green read as confirmation"),
        ("Guard that cannot fail", "r.ok tested on an SPA host that answers 200 with HTML for any path",
         "it degraded correctly only because JSON parsing then threw and a catch swallowed it"),
        ("Label true of another artefact", "Clifford+T named a Python module containing zero T gates",
         "the same label IS accurate for the browser circuit, which counts 56 of them"),
    ]
    findings_table.rows = rows
    return "\n".join(" & ".join([r"\textbf{" + a + "}", b, c]) + r" \\" for a, b, c in rows)


#: Fallback for the checklist row count, kept in the COMMITTED generated macros.
#: `docs/browser-verification-checklist.md` is maintained privately and is no longer tracked, so a
#: fresh clone does not have it. Reading it unconditionally made the report unbuildable for anyone but
#: this working copy - a report that only its author can build is not a deliverable. When the file is
#: present the count is measured from it, exactly as before; when it is absent the previously measured
#: value is reused from `report/generated/macros.tex`, which IS committed. That keeps the number
#: derived from the artefact rather than typed into the source, which is the property that matters.
CHECKLIST = REPO / "docs" / "browser-verification-checklist.md"


def _from_committed(macro: str) -> int | None:
    """The value this macro had last time the checklist WAS present."""
    committed = OUT / "macros.tex"
    if not committed.exists():
        return None
    m = re.search(r"\\newcommand\{\\" + macro + r"\}\{([\d{},]+)\}", committed.read_text())
    return int(m.group(1).replace("{,}", "").replace(",", "")) if m else None


def _checklist_scalar(macro: str, measure) -> int:
    """Measure it from the checklist when present, else reuse the committed generated value."""
    if CHECKLIST.exists():
        return measure(CHECKLIST.read_text())
    prior = _from_committed(macro)
    if prior is not None:
        print(f"  checklist absent (maintained privately): reusing {macro}={prior} from "
              f"{(OUT / 'macros.tex').relative_to(REPO)}")
        return prior
    raise SystemExit(
        f"cannot determine \\{macro}: {CHECKLIST.relative_to(REPO)} is absent and "
        f"{(OUT / 'macros.tex')} carries no prior value. Commit the generated macros, or restore the "
        "checklist, rather than hardcoding a number here."
    )


#: A check row's first cell is its ID: one to three capitals then digits - ``RS26``, ``QC14``, ``CS9``,
#: ``M4`` - with an optional lowercase suffix where a check was inserted between two existing ones,
#: ``RS2b``. Nothing else in the checklist matches that shape as an entire cell.
#:
#: The suffix used to be excluded, and that cost twice. ``RS2b`` went uncounted, so the published
#: total read 320 against a true 321; and the duplicate guard below could not SEE a suffixed ID, so
#: the one collision a suffix makes likely - a second ``RS2b`` written beside the first - was exactly
#: the case the guard was blind to. A pattern that decides what counts as a check has to admit every
#: shape the file uses.
_CHECK_ID = re.compile(r"^[A-Z]{1,3}[0-9]{1,3}[a-z]?$")


def _count_rows(text: str) -> int:
    """Count CHECK rows, which is what the report calls them.

    The earlier predicate was "first cell starts alphanumeric and contains a digit". That also matched
    25 reference-value rows - the ``800``/``1200``/``2000``/``5000`` km distance table, the
    loss-versus-``g`` table, and other measured-value tables carrying no check ID - so the published
    PDF stated 225 checklist rows where there were 200 checks. The surrounding sentence calls them
    verification items, so the number read as a count of checks and was wrong by an eighth.

    Matching the ID shape ties the count to the thing named. A table row with no ID is not a check: it
    cannot be assigned, referenced from the run sheet, or reported as passed.

    AND IT ASSERTS DISTINCTNESS, because the report says it does. `technical-report.tex` renders
    "every one of its \nchecklistrows{} rows has a unique identifier" - a claim this function
    supplies the number for and never checked. It was false when found: RG1-RG4 named both the page
    regions and the artifact-regenerability block, so 297 rows carried 293 distinct IDs and the
    published PDF asserted uniqueness over a set that did not have it.

    The checklist itself claims uniqueness "is checked by extracting every ID". That extraction is
    this function, and until now it extracted without comparing. Raising here means the generator
    that prints the claim is the thing that enforces it - a build failure rather than a sentence
    nobody re-derives.
    """
    seen: dict[str, int] = {}
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = line.split("|")
        if len(cells) > 3 and _CHECK_ID.match(cells[1].strip()):
            seen[cells[1].strip()] = seen.get(cells[1].strip(), 0) + 1
    dupes = sorted(k for k, v in seen.items() if v > 1)
    if dupes:
        raise SystemExit(
            f"checklist has duplicate check IDs: {dupes}. The report claims every row has a unique "
            f"identifier; give the newer family its own prefix before rebuilding.")
    return sum(seen.values())


def macros() -> str:
    # `\gitsha` NAMES THE COMMIT THE MEASUREMENTS CAME FROM, which is the PARENT of the commit that
    # ships this output. That is a fact to state, not a bug to fix: this runs before the commit
    # containing its own result exists, so it can never name that commit. The report said
    # "measurements produced by the code at revision <sha>" and pointed at HEAD-1 - or further back
    # whenever a regenerated file was not committed, which is how it ended up four behind and
    # asserting a row count three short of the tree.
    #
    # `--dirty` is the load-bearing part. If the working tree differs from HEAD then the measurements
    # did not come from HEAD either, and a clean sha would be a claim about a commit the numbers do
    # not describe.
    sha = subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=7"], cwd=REPO,
                         capture_output=True, text=True).stdout.strip()
    if not sha:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                             capture_output=True, text=True).stdout.strip()
    n_pub = sum(1 for s in registry._SPECS if s.hf_repo)
    n_rows = _checklist_scalar("nchecklistrows", _count_rows)
    n_lines = _checklist_scalar("nchecklistlines", lambda t: len(t.splitlines()))
    # The report's date, from the HEAD COMMIT rather than the wall clock. `\date{10 August 2026}`
    # was a literal in technical-report.tex and had gone three days stale by 2026-08-13, which is the
    # same failure this whole directory exists to prevent - every other number here is generated and
    # the date was not. Commit date rather than `today` so that rebuilding an old commit reproduces
    # its own date instead of stamping it with whenever the rebuild happened.
    #
    # ISO from git, month name from a fixed table in Python. `--date=format:%B` asks git for a
    # LOCALISED month name, and this host's shell locale is not English - the first attempt emitted
    # a month name in the local script, which would have put non-Latin characters into the public
    # PDF and broken the rule that no tracked public file contains any. Formatting here removes the
    # locale from the question entirely rather than depending on LC_ALL being set by whoever runs
    # the build.
    #
    # The example is described rather than quoted on purpose: pasting the offending output into this
    # comment is what put those characters into a tracked source file the first time round.
    iso = subprocess.run(["git", "log", "-1", "--format=%ad", "--date=short"],
                         cwd=REPO, capture_output=True, text=True).stdout.strip()
    _MONTHS = ("January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December")
    y, m, d = (int(x) for x in iso.split("-"))
    when = f"{d} {_MONTHS[m - 1]} {y}"
    return "\n".join([
        r"\newcommand{\gitsha}{" + sha + "}",
        r"\newcommand{\reportdate}{" + when + "}",
        r"\newcommand{\npublished}{" + str(n_pub) + "}",
        r"\newcommand{\nchecklistrows}{" + str(n_rows) + "}",
        r"\newcommand{\nchecklistlines}{" + f"{n_lines:,}".replace(",", "{,}") + "}",
        r"\newcommand{\nusecases}{5}",
        r"\newcommand{\ndefectclasses}{" + str(len(findings_table.rows)) + "}",
    ])


def wrap(body: str, *, cols: str, header: str, caption: str, size: str = r"\small",
         longtable: bool = False) -> str:
    """Emit a COMPLETE table environment rather than bare rows.

    ``\\input`` inside a ``tabular`` does not work: TeX ends the file inside an alignment cell, so
    the following ``\\bottomrule`` meets ``\\noalign`` outside a row and the run reports
    "Misplaced \\noalign" -- while still producing a PDF that looks fine and silently drops rows.
    A build that emits 20 errors and a readable document is the same trap as a test suite passing
    on the wrong thing, so the environment is generated whole and \\input at document level.
    """
    if longtable:
        return "\n".join([
            "{" + size, r"\begin{longtable}{" + cols + "}",
            r"\caption{" + caption + r"}\\", r"\toprule", header + r" \\", r"\midrule",
            r"\endfirsthead", r"\toprule", header + r" \\", r"\midrule", r"\endhead",
            body, r"\bottomrule", r"\end{longtable}}"])
    return "\n".join([
        r"\begin{table}[htbp]\centering" + size,
        r"\caption{" + caption + "}", r"\begin{tabular}{" + cols + "}", r"\toprule",
        header + r" \\", r"\midrule", body, r"\bottomrule",
        r"\end{tabular}", r"\end{table}"])


findings_table()  # populate .rows so the caption below can count them

TABLES = {
    "models.tex": dict(fn=model_table, cols="llllll",
        header=r"\textbf{Model} & \textbf{Tier} & \textbf{Metric} & \textbf{Measured} & "
               r"\textbf{Baseline} & \textbf{Dataset}",
        caption="Published models: measured value against the baseline each is compared to. "
                "Tier T1 runs under encryption; T2 is the plaintext teacher. A dash means the "
                "model is declared in the registry but not trained in this build, which is "
                "reported rather than hidden. Dataset licences are given in full in the next "
                "table; this column names the corpus so the licence is never abbreviated."),
    "licences.tex": dict(fn=licence_table, cols=r"p{4.2cm}p{9.5cm}",
        header=r"\textbf{Dataset} & \textbf{Licence, in full}",
        caption="Dataset licences, quoted whole. An earlier revision printed this column inside the "
                "models table through split(\";\")[0][:22], which removed the qualifier from three "
                "of the seven licences that reach this report - and a qualifier is what makes a "
                "licence string long, so truncation is biased toward overclaiming."),
    "crypto.tex": dict(fn=crypto_table, cols="lll",
        header=r"\textbf{Layer} & \textbf{Primitive} & \textbf{Note}",
        caption="Cryptographic parameters as configured"),
    "findings.tex": dict(fn=findings_table, cols="p{2.5cm}p{5.2cm}p{6.2cm}", longtable=True,
        size=r"\scriptsize",
        header=r"\textbf{Class} & \textbf{What it was} & \textbf{Why nothing caught it}",
        caption=f"{len(findings_table.rows)} classes of silent defect, all found in a working "
                f"system with green tests"),
}

def main() -> int:
    """Regenerate every `report/generated/*.tex`.

    BEHIND A GUARD, and that is not style. These writes ran at module level, so `import
    report.build_data` - which a read-only audit does to reuse `_count_rows` - silently rewrote four
    tracked files. An import is not a request to modify the working tree, and a tool that edits the
    repository as a side effect of being read makes every "the tree is clean" observation
    conditional on nobody having imported it.
    """
    for name, cfg in TABLES.items():
        conf = dict(cfg)
        fn = conf.pop("fn")
        (OUT / name).write_text(wrap(fn(), **conf) + "\n")
        print(f"  wrote report/generated/{name}")
    (OUT / "macros.tex").write_text(macros() + "\n")
    print("  wrote report/generated/macros.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
