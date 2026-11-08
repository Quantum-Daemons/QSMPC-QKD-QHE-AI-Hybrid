"""What a re-read of the checklist's pending rows found still live, fixed where it lived (Round 86).

WHY THIS EXISTS

A Round 82 walk wrote thirty-nine findings into the private checklist as "PENDING (defects round)". Later rounds fixed
most of them and nobody updated the rows; re-read against main on 2026-11-06, twenty-six were fixed, eleven in part
and two not at all. These are the parts that could be fixed without republishing anything on the Hub:

- C5: the medical precision-recall curve scores class 1, which BreastMNIST calls "normal, benign", while the Wisconsin
  charts on the same page call 1 malignant, and the caption named neither.
- B6: the medical student, a 14 x 14 single-convolution CNN, was drawn against ResNet-18 at 28 px under "Both bars are
  the same metric on the same axis".
- L3: `build_resnet()` defaulted to ImageNet weights, and DATA_LICENSES named a `--from-scratch` flag that never existed.
- UX19: most of a run's log lines read a bare "metric <stage>".
- QN6: the QRNG line did not name its three SP 800-22 tests or its generator.
- PP14: the SHAP bars printed AML-style names for PaySim quantities and never said what they were.
- Four documents that contradicted the code or themselves (SW6 D8, AM2, RD1, and a datasets line Round 86 made stale).
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _code(rel: str) -> str:
    """The source without its comments, so a check reads what runs and not what a comment says ran."""
    return "\n".join(line for line in _read(rel).splitlines() if not line.lstrip().startswith(("//", "*", "/*", "{/*")))


# ---- C5 --------------------------------------------------------------------------------------------------------------

def test_the_precision_recall_figure_names_its_positive_class() -> None:
    infer = _code("web/src/lib/onnxInference.ts")
    assert 'file.classes?.[String(c)] ?? file.class_names?.[c]' in infer and "plot.classLabels = [labelOf(0), labelOf(1)]" in infer
    chart = _code("web/src/components/VerificationChart.tsx")
    assert "const [neg, pos] = plot.classLabels ?? [null, null];" in chart
    assert "The positive class is class 1, <strong>{pos}</strong>" in chart
    assert "name={`true class 1${pos ? `, ${pos}` : \"\"} (hatched)`}" in chart
    # The sample the medical page scores names its classes in the form the code reads.
    sample = json.loads(_read("models/medical_breastmnist_resnet18/eval_sample.json"))
    assert sample["classes"] == {"0": "malignant", "1": "normal, benign"}


# ---- B6 --------------------------------------------------------------------------------------------------------------

def test_the_medical_student_is_not_drawn_as_like_for_like() -> None:
    cfg = json.loads(_read("web/src/lib/config.defaults.json"))
    by_key = {m["key"]: m for m in cfg["models"]["medical_fl"]}
    caveat = by_key["medical_fl.student"]["metric"]["comparison_caveat"]
    assert "14 x 14" in caveat and "28 x 28" in caveat, caveat
    # The ResNet-18 is compared with ResNet-18 at the baseline's own input, so its claim stands.
    assert by_key["medical_fl.resnet18"]["metric"]["comparison_caveat"] == ""


def test_the_caveat_leaves_the_published_card_unchanged() -> None:
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    import publish_hf  # noqa: E402

    from qsh.models import registry

    spec = registry.get("medical_fl.student")
    card = publish_hf.model_card(spec, registry.achieved("medical_fl.student"))
    assert card == _read(registry.ARTIFACT_DIRS["medical_fl.student"] + "/README.md")


# ---- L3 --------------------------------------------------------------------------------------------------------------

def test_the_resnet_builder_defaults_to_random_initialisation() -> None:
    tree = ast.parse(_read("scripts/train/medical_breastmnist.py"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "build_resnet")
    defaults = dict(zip([a.arg for a in fn.args.kwonlyargs], fn.args.kw_defaults))
    assert isinstance(defaults["pretrained"], ast.Constant) and defaults["pretrained"].value is False
    # The row told a reader to run a flag that never existed; it may say so, not ask for it.
    assert "medical_breastmnist.py --from-scratch" not in _read("DATA_LICENSES.md")


# ---- UX19 ------------------------------------------------------------------------------------------------------------

def _require_browser_or_skip() -> None:
    if ESBUILD.exists():
        return
    msg = "web/node_modules/.bin/esbuild is absent, so the run log's line text cannot be executed. Run `npm ci` in web/."
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(msg + " QSH_REQUIRE_BROWSER_GATES=1 is set, so this is a failure.")
    pytest.skip(msg)


def test_a_metric_line_says_which_metric_and_its_value() -> None:
    _require_browser_or_skip()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        entry, bundle, runner = d / "entry.ts", d / "bundle.mjs", d / "run.mjs"
        entry.write_text(f"export {{ lineText }} from {json.dumps(str(WEB / 'src/components/RunMonitor'))};\n")
        subprocess.run([str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node", "--jsx=automatic",
                        f"--outfile={bundle}", "--log-level=error"], cwd=WEB, check=True, capture_output=True)
        cases = [
            {"type": "metric", "stage": "s", "message": "", "data": {"name": "encrypted_agreement", "value": 1}, "t": 0},
            {"type": "metric", "stage": "s", "message": "", "data": {"name": "fhe_ms", "value": 12.3456789}, "t": 0},
            {"type": "metric", "stage": "s", "message": "", "data": {"name": "model", "value": "finance_aml.gbm"}, "t": 0},
            {"type": "log", "stage": "s", "message": "a sentence", "data": {}, "t": 0},
        ]
        runner.write_text(f"import {{ lineText }} from {json.dumps(bundle.as_uri())};\n"
                          f"console.log(JSON.stringify({json.dumps(cases)}.map(lineText)));\n")
        got = json.loads(subprocess.run(["node", str(runner)], check=True, capture_output=True, text=True).stdout)
    assert got == ["encrypted_agreement = 1", "fhe_ms = 12.3457", "model = finance_aml.gbm", "a sentence"]
    assert "{lineText(e)}" in _code("web/src/components/RunMonitor.tsx")


# ---- QN6 -------------------------------------------------------------------------------------------------------------

def test_the_qrng_line_names_its_tests_and_its_generator() -> None:
    compute = _code("web/src/lib/clientCompute.ts")
    assert "SP 800-22 subset (monobit, runs and block frequency: 3 of its 15 tests)" in compute
    assert "draws from the run-seeded mulberry32 generator" in compute


# ---- PP14 ------------------------------------------------------------------------------------------------------------

def test_the_shap_bars_say_what_each_feature_is() -> None:
    payload = json.loads(_read("web/public/data/finance/paysim_sample.json"))
    assert set(payload["feature_definitions"]) == set(payload["names"])
    assert "definitions: ds.feature_definitions ?? undefined" in _code("web/src/lib/realData.ts")
    assert "artifacts.explanation_feature_definitions = realIn?.definitions ?? null;" in _code("web/src/lib/clientCompute.ts")
    assert "definitions={artifacts?.explanation_feature_definitions ?? null}" in _code("web/src/pages/UseCasePage.tsx")
    panel = _code("web/src/components/ShapPanel.tsx")
    assert "{definitions[t.feature]}</dd>" in panel


# ---- the documents ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(("path", "old"), [
    ("docs/pqc-roadmap.md", "machine-readable marking of generated content, is **not applicable**"),
    ("docs/pqc-roadmap.md", "`provenance_note` block is voluntary"),
    ("docs/findings.md", "unpublished on licence grounds (the PRS)"),
    ("docs/findings.md", "The decision rests on the FHE\n  path"),
    ("docs/models-and-benchmarks.md", "PneumoniaMNIST as the full deployment's use-case default"),
    ("DATA_LICENSES.md", "the script's *default* is still `IMAGENET1K_V1`"),
])
def test_a_corrected_sentence_does_not_come_back(path: str, old: str) -> None:
    assert old not in _read(path), f"{path} says {old!r} again"


# ---- the three republished Hub files (the owner's choice, 2026-11-06) ------------------------------------------------

def test_the_opssat_credit_has_one_owner_and_reaches_every_file() -> None:
    """RD18: the dataset record was appended by two payload builders and missing from the registry, so the cards and
    the evaluation samples credited the article alone."""
    from qsh.models import registry

    record = "doi:10.5281/zenodo.12588359 (version 1)"
    assert record in registry.DS_OPSSAT_AD.attribution
    for builder in ("scripts/data/build_heldout_payloads.py", "scripts/prep_real_data_2d.py"):
        assert "OPSSAT-AD - anomaly detection dataset" not in _read(builder), f"{builder} writes the credit again"
    for shipped in ("models/satellite_opssat/README.md", "models/satellite_opssat_student/README.md",
                    "models/satellite_opssat/eval_sample.json", "models/satellite_opssat_student/eval_sample.json",
                    "web/public/data/satellite/student_heldout.json", "web/public/data/satellite/opssat_segments.json",
                    "web/public/data/satellite/real_2d.json"):
        assert record in _read(shipped), f"{shipped} credits the article without the dataset record"


def test_the_opssat_split_says_how_it_was_drawn() -> None:
    """NS29: the verification line printed "the benchmark's own test partition, whole" for a stratified random split."""
    from qsh.models import registry

    for key in ("satellite_comms.opssat", "satellite_comms.opssat_student"):
        split = registry.get(key).eval_split
        assert "stratified random" in split, key
        sample = json.loads(_read(registry.ARTIFACT_DIRS[key] + "/eval_sample.json"))
        assert sample["split"] == split, key


def test_the_telemanom_sample_does_not_deny_the_satellite_ckks_check() -> None:
    """HF8: the published sample said no FHE stage runs on the satellite page, which its CKKS check made false."""
    text = _read("models/satellite_telemanom/eval_sample.json") + _read("scripts/prep_model_inputs.py")
    assert "no FHE or MPC stage runs on the satellite page" not in text


# ---- G3 --------------------------------------------------------------------------------------------------------------

def test_the_hpmpc_benchmark_checks_what_it_reports() -> None:
    """The script's guards, each written because its failure happened: a stale executable reported as a GPU result, a
    GPU result with no GPU, and the unlicensed nested submodule."""
    script = _read("scripts/bench_hpmpc_gpu.sh")
    subprocess.run(["bash", "-n", str(REPO / "scripts/bench_hpmpc_gpu.sh")], check=True)
    assert "submodule update --init nn/PIGEON" in script and "--recursive" not in script
    assert 'die "nn/Pygeon is populated; it carries no licence"' in script
    assert "has a compiler error" in script and "left a stale run-P" in script and "a stale executable" in script
    assert 'run "$W/test57_hidden.log" "CUDA_VISIBLE_DEVICES="' in script
    assert not (REPO / "scripts/build_gpu_mpc.sh").exists()
    rec = json.loads(_read("docs/measurements/hpmpc-gpu-2026-11-06.json"))
    assert rec["checks"]["test57_gpu_hidden"] == "matrix multiplication failed"
    gpu = {r["n"]: r for r in rec["results"] if r["device"] == "gpu"}
    cpu = {r["n"]: r for r in rec["results"] if r["device"] == "cpu"}
    for n in set(gpu) & set(cpu):  # the same protocol: the same volume sent at the same N
        assert {x["p0_sent"] for x in gpu[n]["runs"]} == {x["p0_sent"] for x in cpu[n]["runs"]}, n


def test_the_hpmpc_convolution_is_checked_before_it_is_timed() -> None:
    """G3, the convolution: hpmpc has no correctness test for it, so the script inserts one into function 33, runs it on
    every path against a plain convolution and NumPy, requires both GPU paths to fail it with the GPU hidden, and takes
    it out again before any run is timed. The record must show all of that, and the same volume sent on every path."""
    script = _read("scripts/bench_hpmpc_gpu.sh")
    assert "QSH CONV CHECK" in script and "conv_reference.txt" in script, "the convolution check is gone"
    assert 'run "$W/convcheck_${g}_hidden.log" "CUDA_VISIBLE_DEVICES="' in script
    taken_out = script.index('git -C hpmpc checkout -q -- "${BENCH#hpmpc/}"')
    assert script.index("QSH CONV CHECK") < taken_out < script.index("measure_conv() {"), (
        "the check must be taken out of the clone before the convolution is timed")
    assert "Testing|QSH_CONV_CHECK" in script, "a timed run must refuse the check's output"
    rec = json.loads(_read("docs/measurements/hpmpc-gpu-conv-2026-11-06.json"))
    chk = rec["checks"]
    assert chk["reference"]["outputs"] == 576 and chk["reference"]["sum"] == 1036170
    for path in ("cpu", "gpu_gemm", "gpu_conv"):
        assert chk[path]["parties_reproducing"] == 3, f"{path} did not reproduce the convolution on every party"
    for path in ("gpu_gemm_hidden", "gpu_conv_hidden"):
        assert chk[path]["parties_reproducing"] == 0, f"{path}: the GPU path reproduced it with the GPU hidden"
    assert chk["sent_volume_unequal_at"] == []
    cells = {(r["function"], r["n"]) for r in rec["results"]}
    for f, n in cells:
        paths = {r["path"] for r in rec["results"] if (r["function"], r["n"]) == (f, n)}
        assert paths == {"cpu", "gpu_gemm", "gpu_conv"}, (f, n, paths)
    assert {f for f, _ in cells} == {33, 34, 35}
