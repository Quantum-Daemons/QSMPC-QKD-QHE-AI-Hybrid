"""Every published evaluation sample says which split it was drawn from, and the page says the same.

WHY THIS EXISTS

Round 71. The run log said "re-scored all N published held-out rows" for every model it verified.
Two of the samples were not held out. The finance teacher's 8,192 rows (and so the CKKS student's,
which reuses them) and Telemanom's 128 windows were drawn from the whole corpus, so about 70 percent
of each were rows the model had been fitted on. Both generators also kept their own copy of the
trainer's preprocessing, and both copies had drifted: the finance one computed per-account aggregates
over the whole timeline, the leaky form the model was not trained on, and the Telemanom one
back-filled and fitted its z-score on the whole series. Drawn from the test window with the
trainer's own functions, the finance teacher scores rare-class F1 0.2032 where the old sample gave
0.5439.

WHAT IS ASSERTED

- Every published model declares its split (`eval_split`) or why it publishes no sample
  (`eval_sample_reason`), never both and never neither, and the browser config carries both.
- The page takes "held-out" from the declaration: the two log lines and the three captions read
  the split from the model, and the old blanket phrases are gone.
- The trainers' shared functions behave as the samples need, on synthetic data so CI can run them:
  Telemanom's `preprocess` fits its z-score on the training window and back-fills nothing, and the
  finance `temporal_windows` cuts the test window off the end. The generators call them and draw
  from their test output.
- A sample on disk states the registry's split, and the finance teacher's states the decision
  threshold its feature_spec publishes, which the browser now classifies at.
- The finance notes quote the figures beside them: their numbers were literals and outlived a
  retrain ("about 19x" next to figures 6.1x apart).
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from qsh.models import registry  # noqa: E402

WEB = REPO / "web" / "src"


def _without_docstrings(source: str) -> ast.Module:
    """The module as code: docstrings dropped here, comments by `ast.unparse`, so a check on what the
    generator DOES is not tripped by the history its docstrings record in the old phrases."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return tree


PREP_TREE = _without_docstrings((REPO / "scripts" / "prep_model_inputs.py").read_text())
PREP = ast.unparse(PREP_TREE)

#: Samples built before Round 71 and not rebuilt in it. Their files carry no `split` key; the page
#: prints the registry's declaration for them, which is what the second test pins.
BUILT_BEFORE_ROUND_71 = {"satellite_comms.opssat", "satellite_comms.opssat_student", "telecom_ids.qml",
                         "telecom_ids.nidd"}


def _published():
    return [s for s in registry.specs() if s.hf_repo]


def test_every_published_model_declares_its_split_or_why_it_has_none() -> None:
    specs = _published()
    assert len(specs) >= 11, f"only {len(specs)} published models found; the registry moved"
    for s in specs:
        assert bool(s.eval_split) != bool(s.eval_sample_reason), (
            f"{s.key}: exactly one of eval_split (where its sample comes from) and eval_sample_reason "
            f"(why it has none) must be set")
    exported = registry.frontend_models()
    flat = {m["key"]: m for uc in exported.values() for m in (uc if isinstance(uc, list) else uc.get("models", []))}
    for s in specs:
        assert flat[s.key]["eval_split"] == (s.eval_split or None), f"{s.key}: the config drops eval_split"


def _code(path: Path) -> str:
    return "\n".join(line for line in path.read_text().splitlines() if not line.lstrip().startswith("//"))


def test_the_page_says_held_out_only_through_the_declaration() -> None:
    compute = _code(WEB / "lib" / "clientCompute.ts")
    assert "published evaluation sample (${p.split" in compute, "the one-row log no longer prints the split"
    assert "rows of its published evaluation sample (${v.split" in compute, (
        "the verification log no longer prints the split")
    for old in ("published held-out rows", "of its own `\n        + `held-out sample"):
        assert old not in compute, f"the blanket phrase is back in the run log: {old!r}"
    chart = _code(WEB / "components" / "VerificationChart.tsx")
    assert "held-out rows scored in this tab" not in chart and chart.count("res.split") >= 2
    page = _code(WEB / "pages" / "UseCasePage.tsx")
    assert "published held-out sample" not in page and "artifacts.verification.split" in page
    panel = _code(WEB / "components" / "ModelPanel.tsx")
    assert "published held-out sample" not in panel and "spec.eval_split" in panel
    assert panel.index("if (spec.eval_sample_reason)") < panel.index("re-scored on its"), (
        "a model with no sample by design must not be promised a re-score")
    infer = _code(WEB / "lib" / "onnxInference.ts")
    assert "base.split = target.spec.eval_split" in infer and "split: target.spec.eval_split" in infer


def test_telemanom_preprocess_fits_on_the_training_window_and_back_fills_nothing() -> None:
    pd = pytest.importorskip("pandas")
    from train.satellite_telemanom import TEST_FRACTION, preprocess, windows

    idx = pd.date_range("2020-01-01", periods=2000, freq="60s")
    rng = np.random.default_rng(0)
    raw = pd.DataFrame(rng.standard_normal((2000, 2)).cumsum(0), index=idx, columns=["a", "b"])
    raw.iloc[:7, 0] = np.nan                       # a leading gap: dropped, never filled from later
    raw.iloc[500:520, 1] = np.nan                  # an interior gap: carried forward
    df, values, cut, mu, sigma = preprocess(raw, "300s")

    resampled = raw.resample("300s").mean()
    assert df.index[0] == resampled.dropna().index[0], "a leading row was back-filled from the future"
    assert cut == int(len(df) * (1.0 - TEST_FRACTION))
    plain = df.to_numpy(dtype=np.float32)
    assert np.allclose(mu, plain[:cut].mean(0)) and np.allclose(sigma, plain[:cut].std(0) + 1e-8)
    assert not np.allclose(mu, plain.mean(0)), "the z-score was fitted on the whole series"
    assert np.allclose(values, (plain - mu) / sigma)
    xte, _ = windows(values[cut:], 10, 1)
    assert np.array_equal(xte[0], values[cut:cut + 10]), "test windows must start at the cut"


def test_finance_temporal_windows_cut_the_test_window_off_the_end() -> None:
    pd = pytest.importorskip("pandas")
    from train.finance_aml import TEST_FRACTION, temporal_windows

    df = pd.DataFrame({"row": np.arange(1000)})
    fit, val, te, tr = temporal_windows(df)
    cut = int(1000 * (1.0 - TEST_FRACTION))
    assert te["row"].tolist() == list(range(cut, 1000))
    assert tr["row"].tolist() == list(range(cut))
    assert fit["row"].tolist() + val["row"].tolist() == tr["row"].tolist()


def _body(name: str) -> str:
    fn = next(n for n in PREP_TREE.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.unparse(fn)


def test_the_generators_draw_from_the_trainers_test_window() -> None:
    fin = _body("build_finance")
    assert "temporal_windows(df)" in fin and "load_sorted()" in fin
    assert "graph_features(te, fit_on=tr, causal=causal_account_features(df))" in fin, (
        "the finance sample must be featurised exactly as main() featurises its test rows")
    assert "graph_features(df)" not in fin, "whole-timeline aggregates are the leaky arm"
    assert re.search(r"pos = np\.flatnonzero\(y_te == 1\)", fin), "positives must come from the test window"
    sat = _body("build_satellite")
    assert "preprocess(D.load_esa_channels(), resample)" in sat
    assert "windows(values[cut:], SEQ_LEN, 1)" in sat, "windows must come from the test window"
    assert ".bfill()" not in sat and "values.mean(0)" not in sat, "a private copy of the preprocessing is back"
    for body in (fin, sat):
        assert "fixtures.npy" in body, "the sample is no longer checked against the trainer's own fixtures"


def _sample(key: str) -> dict | None:
    p = REPO / registry.ARTIFACT_DIRS[key] / "eval_sample.json"
    return json.loads(p.read_text()) if p.exists() else None


def test_a_sample_on_disk_states_the_registry_split() -> None:
    seen = 0
    for s in _published():
        sample = _sample(s.key)
        if s.eval_sample_reason:
            assert sample is None, f"{s.key} says it publishes no sample but has one"
            continue
        assert sample is not None, f"{s.key} declares a split and ships no sample"
        if s.key in BUILT_BEFORE_ROUND_71:
            assert "split" not in sample, f"{s.key} was rebuilt; take it off BUILT_BEFORE_ROUND_71"
            continue
        assert sample.get("split") == s.eval_split, f"{s.key}: the file and the registry name different splits"
        seen += 1
    assert seen >= 6, f"only {seen} samples state their split"


def test_the_finance_teacher_is_classified_at_its_published_threshold() -> None:
    sample = _sample("finance_aml.gbm")
    spec = json.loads((REPO / registry.ARTIFACT_DIRS["finance_aml.gbm"] / "feature_spec.json").read_text())
    assert sample["expected"]["decision_threshold"] == spec["decision_threshold"]
    infer = _code(WEB / "lib" / "onnxInference.ts")
    assert infer.count("file.expected.decision_threshold") >= 2, (
        "both the whole-sample verification and the one-row run must classify at the threshold")


def test_the_finance_notes_quote_the_figures_beside_them() -> None:
    rec = json.loads(registry.METRICS_PATH.read_text())
    teacher, student = _sample("finance_aml.gbm"), _sample("finance_aml.student")
    t_note, s_note = teacher["expected"]["note"], student["expected"]["note"]
    assert f"{rec['finance_aml.gbm']['minority_class_f1']:.4f}" in t_note
    assert f"{student['expected']['minority_class_f1']:.4f} on THESE rows" in s_note
    assert f"not the {rec['finance_aml.student']['student_metric']:.4f} the registry records" in s_note
    ratio = student["expected"]["minority_class_f1"] / rec["finance_aml.student"]["student_metric"]
    assert f"differ by {ratio:.1f}x" in s_note
    for literal in ("about 19x", "roughly 8.7x", "0.0178"):
        assert literal not in PREP, f"a stale literal is back in the generator: {literal!r}"


def test_the_readme_counts_the_published_repos_and_their_one_exception() -> None:
    """README's own sentence says it is "the second copy - and the second copy is the one that rots",
    and it rotted again: "All ten repos" with eleven published, and "the evaluation sample for each"
    with one published by design without one."""
    words = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15}
    # The sentence moved from the README to docs/findings.md in Round 80, with the findings it
    # belongs to; the README now links there.
    readme = (REPO / "docs" / "findings.md").read_text()
    m = re.search(r"All (\w+) repos are public and\s+pinned", readme)
    assert m, "the sentence counting the published repos moved; update this gate with it"
    assert words.get(m.group(1)) == len(_published()), (
        f"README says {m.group(1)} published repos, the registry has {len(_published())}")
    no_sample = [s.key for s in _published() if s.eval_sample_reason]
    tail = readme[m.end():m.end() + 400]
    assert no_sample == ["satellite_comms.student"] and "satellite student" in tail, (
        f"the README's exception no longer matches the registry's: {no_sample}")
