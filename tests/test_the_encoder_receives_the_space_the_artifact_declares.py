"""A QCNN artifact declares an input space; the page must feed it that space.

WHY THIS EXISTS

`web/public/data/finance/qcnn_weights.json` declares `input_space: "raw"`, and
`web/src/lib/realData.ts:116` hands it rows from `paysim_sample.json` - which are StandardScaler
output, and say so in that file's own `scaling` field. So `qcnnInferTrained` standardised
already-standardised data, a second time, with a DIFFERENT scaler: the artifact's moments were
fitted by the trainer on the full raw TSV, the sample's were fitted on the first 70% of 400 picked
rows.

MEASURED CONSEQUENCE, through the browser's own `qcnnInferTrained` on the shipped artifact and the
shipped sample, comparing the array the page fed against the array the artifact declares:

    wire 0 (`amount`) tanh range   [-0.99999992, -0.99982701]   sd 3.3e-05
    saturated |tanh| > 0.999       556 of 3200
    rows given a different class   101 of 400   (25.25%)
    accuracy on the bundled labels 0.8200 fed `samples`, 0.9125 fed `samples_raw`
    minority-class F1              0.7857 fed `samples`, 0.9041 fed `samples_raw`

TWO DIFFERENT COMPARISONS EXIST HERE AND THEY MUST NOT BE CONFLATED. The figures above compare the
DEPLOYED feed against the CORRECT feed - two different arrays through one encoder. An earlier
measurement compared `input_space: "raw"` against `"zscore"` on the SAME standardised array - one
array through two encoders - and gave 69 of 400, 17.25%, with 0.9875 accuracy. Both are true of what
they measure. The first is the one that describes what a visitor saw, because it is the one where
the rows differ. Quoting the second as "the deployed error rate" would be the mistake the previous
round's audit was written to catch.

These 400 rows overlap the artifact's training split, so neither accuracy is a generalisation
estimate; the held-out figure is on the model card. What they measure is whether the two feeds are
the same model, and they are not.

The model's most informative input carried no information at all. Medical is correct - it declares
`zscore` and is fed z-scores - and telecom is correct, because `clientIDS.ts` hands over a genuinely
raw row. Finance was the only mismatched pair, and nothing anywhere compared the two.

WHAT IS ASSERTED, AND WHY IT IS NOT A STRING COMPARISON

Comparing `input_space` against the sample's `scaling` prose would be checking one label against
another. What matters is the ENCODER'S INPUT: after the declared preprocessing, every wire that
carries variance in the source must still carry variance once encoded. A space mismatch shows up
there immediately and unmistakably, because standardising twice pushes a column into tanh's flat
region and the wire goes dead.

That also catches mismatches this file's authors did not think of - a future artifact declaring a
third space, a sample regenerated in different units - without needing a rule per case.

SCOPE IS DERIVED, NOT LISTED

The pairs come from globbing `web/public/data/*/qcnn_weights*.json` and looking for a sibling sample
in the same directory. A use case added tomorrow is checked the day its files land. "A gate whose
scope names files" is this project's most-repeated defect and the previous branch was named for it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PUBLIC = REPO / "web" / "public" / "data"

#: A wire is "dead" below this. The shipped finance defect sat at 3.3e-05; a healthy wire on the
#: same data sits at 0.6. Two orders of magnitude of headroom either way.
MIN_ENCODED_SD = 1e-3

#: Columns that are constant in the SOURCE cannot be expected to carry variance after encoding, and
#: excluding them is not a loophole: it is the difference between "this wire is dead because the
#: feed is in the wrong space" and "this wire is dead because the data has no signal in it". The
#: second is a data problem, reported separately by `test_no_wire_is_silently_constant`.
MIN_SOURCE_SD = 1e-9


def _pairs() -> list[tuple[str, Path, Path]]:
    """(use case, weights, sample) for every artifact with a sibling sample file, by glob."""
    out: list[tuple[str, Path, Path]] = []
    for w in sorted(PUBLIC.glob("*/qcnn_weights*.json")):
        samples = [p for p in sorted(w.parent.glob("*_sample.json"))]
        if len(samples) == 1:
            out.append((w.parent.name, w, samples[0]))
        elif len(samples) > 1:
            pytest.fail(
                f"{w.parent.name} ships {len(samples)} sample files ({[s.name for s in samples]}); "
                "this gate cannot tell which one the page feeds the QCNN. Name the pairing "
                "explicitly rather than letting the glob guess."
            )
    return out


def _encode(row: list[float], w: dict, space_of_data: str) -> list[float]:
    """The browser's encoder, exactly: standardise if the artifact says the input is raw, then tanh.

    Mirrors `quantumSim.ts` - `if (raw && feature_mean && feature_std) v = (v - mean) / sd;` then
    `RY(tanh(v) * pi)`. The angle is monotone in `tanh(v)`, so the tanh value is the quantity that
    decides whether the wire carries anything.
    """
    declared = w.get("input_space")
    mean, std = w.get("feature_mean") or [], w.get("feature_std") or []
    out = []
    for i, v in enumerate(row):
        x = float(v)
        if declared == "raw" and i < len(mean) and i < len(std):
            sd = float(std[i]) or 1.0
            x = (x - float(mean[i])) / sd
        out.append(math.tanh(x))
    _ = space_of_data
    return out


def _sd(col: list[float]) -> float:
    n = len(col)
    if n < 2:
        return 0.0
    m = sum(col) / n
    return math.sqrt(sum((v - m) ** 2 for v in col) / n)


def test_the_scan_has_a_subject() -> None:
    """Non-vacuity: a glob that finds nothing would make every assertion below pass."""
    pairs = _pairs()
    assert pairs, (
        "no (qcnn_weights, sample) pair found under web/public/data/*/. Either the artifacts moved "
        "or the naming convention changed; this gate is asserting nothing."
    )
    names = {uc for uc, _, _ in pairs}
    assert "finance" in names and "medical" in names, (
        f"expected finance and medical to ship both a QCNN artifact and a sample; found {names}. "
        "Those are the two the browser feeds from a bundled file."
    )


@pytest.mark.parametrize("case", _pairs(), ids=lambda c: f"{c[0]}/{c[1].name}")
def test_every_encoded_wire_survives_the_declared_preprocessing(case) -> None:
    """The property: a wire with variance in the source keeps it through the encoder.

    Red on the unmodified tree for both finance artifacts, at wire 0 (`amount`), sd 3.3e-05.
    """
    usecase, wpath, spath = case
    w = json.loads(wpath.read_text())
    ds = json.loads(spath.read_text())
    # THE ARRAY THE BROWSER SELECTS, chosen the same way `clientCompute.ts` chooses it: an artifact
    # declaring `raw` is fed `samples_raw` when the sample ships one. Reading `samples`
    # unconditionally would test an array the page does not use, which is a gate measuring the wrong
    # thing while looking green.
    by_space = dict(ds.get("samples_by_space") or {})
    if ds.get("feature_space"):
        by_space[ds["feature_space"]] = ds.get("samples")
    declared = w.get("input_space")
    rows = by_space.get(declared)
    assert rows, (
        f"{wpath.name} declares `input_space: {declared!r}` and {spath.name} ships rows only in "
        f"{sorted(k for k in by_space)}. The page can then feed it only an array in some other "
        "space, which applies a second transform with the wrong constants - the defect this gate "
        "exists for. Selection is by NAME, so an unshipped space is a refusal, not a fallback.")

    n_feat = int(w.get("n_features") or len(rows[0]))
    src_cols = [[float(r[i]) for r in rows] for i in range(min(n_feat, len(rows[0])))]
    enc = [_encode(r[:n_feat], w, str(ds.get("scaling", ""))) for r in rows]
    enc_cols = [[r[i] for r in enc] for i in range(len(enc[0]))]

    dead = []
    for i, (s, e) in enumerate(zip(src_cols, enc_cols, strict=False)):
        if _sd(s) <= MIN_SOURCE_SD:
            continue                      # constant at source - a data question, not a space one
        if _sd(e) < MIN_ENCODED_SD:
            name = (w.get("feature_names") or ds.get("names") or [])
            label = name[i] if i < len(name) else f"col {i}"
            dead.append(
                f"wire {i} ({label}): source sd {_sd(s):.4f} -> encoded sd {_sd(e):.2e}, "
                f"range [{min(e):.8f}, {max(e):.8f}]"
            )

    saturated = sum(1 for r in enc for v in r if abs(v) > 0.999)
    assert not dead, (
        f"{usecase}: these wires carry variance in {spath.name} and none after the encoder, which "
        f"means the data is not in the space `{wpath.name}` declares "
        f"(`input_space: {w.get('input_space')!r}`):\n  " + "\n  ".join(dead)
        + f"\n\n{saturated} of {len(enc) * len(enc[0])} encoded values are saturated (|tanh| > "
        "0.999). Standardising an already-standardised column pushes it into tanh's flat region and "
        "the wire goes dead. Feed the array whose space the artifact declares - the sample may ship "
        "more than one."
    )


@pytest.mark.parametrize("case", _pairs(), ids=lambda c: f"{c[0]}/{c[1].name}")
def test_no_wire_is_silently_constant_or_duplicated(case) -> None:
    """A feature that is constant, or an exact copy of another, must be declared.

    NOT the same failure as the one above, and separated deliberately. A wire dead because the feed
    is in the wrong space is a bug. A wire dead because the column has no signal is a property of
    the data - but it is still a wire the page shows, and the SHAP panel still splits attribution
    across a duplicated pair as though the two were independent evidence.

    MEASURED on the shipped finance sample: `velocity_24h` is identically 0 in all 400 rows, and
    `amount_zscore` equals `amount` to 1.0e-05 - the file's own 5-decimal rounding. Two of eight
    wires, undeclared anywhere in `names`, `provenance` or `eda`.
    """
    usecase, wpath, spath = case
    ds = json.loads(spath.read_text())
    rows = ds["samples"]
    names = ds.get("names") or []
    n = len(rows[0])
    cols = [[float(r[i]) for r in rows] for i in range(n)]
    eda = ds.get("eda") or {}
    # DECLARED means named in the field that exists to say so, NOT merely appearing in the column
    # list. A first version of this check accepted any mention anywhere in the `eda` block, and
    # since `eda.columns` names every column by construction, it passed unconditionally - it could
    # not have failed on anything. `velocity_24h` really is declared, in `eda.degenerate`; the
    # `amount`/`amount_zscore` duplication is declared nowhere, and the weak check hid that.
    degenerate = set(eda.get("degenerate") or [])
    duplicates = {tuple(sorted(p)) for p in (eda.get("duplicates") or []) if len(p) == 2}

    problems = []
    for i in range(n):
        label = names[i] if i < len(names) else f"col {i}"
        if _sd(cols[i]) <= MIN_SOURCE_SD and label not in degenerate:
            problems.append(
                f"{label} (col {i}) is constant in all {len(rows)} rows and is not in "
                "`eda.degenerate`")
        for j in range(i + 1, n):
            if max(abs(a - b) for a, b in zip(cols[i], cols[j], strict=False)) <= 1e-4:
                other = names[j] if j < len(names) else f"col {j}"
                if tuple(sorted((label, other))) not in duplicates:
                    problems.append(
                        f"{label} (col {i}) and {other} (col {j}) are identical to 1e-4 - one "
                        "feature occupying two wires - and the pair is not in `eda.duplicates`")
    assert not problems, (
        f"{usecase}: {spath.name} ships wires that carry no independent signal:\n  "
        + "\n  ".join(problems)
        + "\n\nDeclare them in the file's `eda` block, or regenerate the payload without them. The "
        "page renders one wire per feature and the SHAP panel attributes to each independently, so "
        "an undeclared duplicate is shown to a visitor as two pieces of evidence."
    )
