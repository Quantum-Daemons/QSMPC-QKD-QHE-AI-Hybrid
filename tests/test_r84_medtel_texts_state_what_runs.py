"""The medical, telecom and fibre-QKD texts say what runs (Round 84).

WHY THIS EXISTS (scratchpad audit83 page-medical-telecom.md rows 1-5, 8-13; private-inference-vs-
research.md row 5)

- The three BreastMNIST cards said Pawlowska et al. "found 235 duplicates (~19% of 780)". The letter
  says 155 images are copies, 19% of the collection; 235 is the count of items in its duplicate
  appendix, 30% of 780. And "70 axilla images that are not breast tissue" is the letter's "other
  structures than the breast", which it lists as suspected axilla images. MedMNIST's own notice, that
  its dataset is not intended for clinical use, appeared nowhere a medical model is described.
- The ResNet-50 entry called MedMNIST v2's table entry "the published finding", and the ResNet-50 and
  5G-NIDD cards said "the encrypted path runs a distilled student" though nothing is distilled from
  either (the template line is `scripts/publish_hf.py`'s; the notes now say so per model).
- Telecom: the canvas said "the IDS classifies the link's telemetry", and the blurb "a QML
  intrusion-detection plane"; the IDS trains and scores on synthetic rows it generates in the tab.
- OptiQKD: the tooltip said it raises the secret-key rate, and the canvas called its figure a
  "surrogate SKR"; nothing it finds is applied, and it scores the run's own bound.
- The fibre chart plotted the asymptotic rate without saying so, and no fibre text said the finite
  key takes the bases at even odds, or that Rusca et al.'s Table I distances carry 6 dB of receiver
  loss where the model has the detectors' 3.
- The WebGPU tooltip blamed WebGPU for an 18216 ms ResNet-18 run that ran the int8 graph, whose
  quantised nodes have no WebGPU kernel and ran on the CPU.
- The intensity comments in defaults.py called 0.399/0.098 "the optimum" without naming its objective,
  and still measured the pair against "the shipped 2e7 block".

Every assertion fails on the baseline. The browser modules are run in node where they compute text
(`captionOf`, `canvasRow`, the canvas labels, `securityFacts`); the rest is read from source.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
SRC = WEB / "src"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"


# --------------------------------------------------------------------------------------------- #
# The registry: what the medical and 5G-NIDD cards print
# --------------------------------------------------------------------------------------------- #

def _spec(key: str):
    from qsh.models.registry import specs

    return next(s for s in specs() if s.key == key)


def test_the_breastmnist_note_counts_the_letters_copies_as_the_letter_does() -> None:
    from qsh.models.registry import DS_BREASTMNIST

    n = DS_BREASTMNIST.notes
    assert "235 duplicates" not in n and "~19% of 780" not in n, (
        "the note still calls the appendix's 235 items duplicates and puts them at 19%")
    assert "155 images are copies" in n and "19% of its 780" in n and "235" in n, n
    assert "70 axilla images" not in n and "show other structures than the breast" in n
    assert "PMC10293973" in n


@pytest.mark.parametrize("key", ["medical_fl.resnet18", "medical_fl.resnet50", "medical_fl.student"])
def test_every_medical_model_carries_medmnists_clinical_use_notice(key: str) -> None:
    assert "NOT intended for clinical use" in _spec(key).notes, f"{key}: MedMNIST's notice is missing"


def test_the_resnet50_entry_checks_a_table_ordering_not_a_published_finding() -> None:
    s = _spec("medical_fl.resnet50")
    assert "That is the published finding" not in s.metric.baseline_source
    assert "test a published claim" not in s.notes
    assert "benchmark table" in s.metric.baseline_source and "checks that ordering" in s.metric.baseline_source


@pytest.mark.parametrize("key, student", [("medical_fl.resnet50", "medical_fl.student"),
                                          ("telecom_ids.nidd", "telecom_ids.student")])
def test_a_model_nothing_is_distilled_from_says_so(key: str, student: str) -> None:
    notes = _spec(key).notes
    assert re.search(r"no student is distilled from this model", notes, re.I), (
        f"{key}: the card's template says a distilled student runs the encrypted path, and the notes "
        "do not correct it")
    assert student in notes


# --------------------------------------------------------------------------------------------- #
# The canvas, the blurbs, the key row: run in node
# --------------------------------------------------------------------------------------------- #

HARNESS = r"""
import { createRequire } from "node:module";
import { resolve } from "node:path";
const [WEB, OUT] = process.argv.slice(2);
const { build } = createRequire(WEB + "/package.json")("esbuild");
await build({
  stdin: { contents: `export { captionOf, canvasRow, USECASE_SPECS, kindOf } from ${JSON.stringify(resolve(WEB, "src/lib/useCaseFlows"))};
export { USECASE_META } from ${JSON.stringify(resolve(WEB, "src/lib/useCases"))};
export { securityFacts } from ${JSON.stringify(resolve(WEB, "src/lib/clientCompute"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: OUT, logLevel: "error",
  define: { "import.meta.env": '{"BASE_URL":"/"}', "process.env.NODE_ENV": '"production"' },
  plugins: [{ name: "stubs", setup(b) {
    b.onResolve({ filter: /\.wasm\?url$|qsh_mpc\.js$/ }, (a) => ({ path: a.path, namespace: "stub" }));
    b.onLoad({ filter: /.*/, namespace: "stub" }, () => ({ contents: "export default '';", loader: "js" }));
  } }],
});
const m = await import(`file://${OUT}`);
const tel = m.USECASE_SPECS.telecom_ids;
const qkdStep = tel.steps.find((s) => s.stageId === "qkd_key_agreement");
const idsStep = tel.steps.find((s) => s.stageId === "qml_ids");
const runMetrics = { qber: 0.01, qkd_skr_improvement_pct: 60.8, qkd_convergence_evals: 22, qkd_eve_fraction: 0 };
const fibre = { usecase: "telecom_ids", qkd: "bb84-decoy", qkd_protocol: "bb84" };
const fibreMetrics = { secure_key_bits: 5214327, qkd_finite_key_bits: 5214327, qkd_block: "pa_bits",
  qkd_block_s: 5.334, qkd_block_pulses: 5334030000, qkd_finite_key_bound: "Rusca et al. 2018 (one decoy)",
  qkd_eps_sec: 1e-9, qkd_fk_deviation: "tighter", qkd_block_units_for_key: 1, skr_per_pulse: 0.001426711, qber: 0.01 };
console.log(JSON.stringify({
  telecomCaption: m.captionOf("telecom_ids"),
  telecomIdsRow: m.canvasRow("telecom_ids", idsStep, m.kindOf(idsStep, {}), {}, {}),
  telecomQkdLabel: qkdStep.label(runMetrics, { qkdProtocol: "bb84", optiqkd: true }),
  blurbs: { telecom: m.USECASE_META.telecom_ids.blurb, medical: m.USECASE_META.medical_fl.blurb },
  fibreKeyRow: m.securityFacts(fibre, fibreMetrics).find((f) => f.label === "QKD key material"),
}));
"""


def _require_node_or_skip() -> None:
    missing = [name for name, ok in (("web/node_modules/.bin/esbuild", ESBUILD.exists()),
                                     ("node", shutil.which("node") is not None)) if not ok]
    if not missing:
        return
    msg = f"{' and '.join(missing)} absent, so the browser modules cannot run. This is a SKIP, not a pass."
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


@pytest.fixture(scope="module")
def ui(tmp_path_factory) -> dict:
    _require_node_or_skip()
    tmp = tmp_path_factory.mktemp("medtel_texts")
    script = tmp / "harness.mjs"
    script.write_text(HARNESS)
    proc = subprocess.run(["node", str(script), str(WEB), str(tmp / "bundle.mjs")], cwd=WEB,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, "the browser modules did not run:\n" + proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_the_telecom_canvas_says_the_ids_reads_synthetic_rows_not_the_link(ui: dict) -> None:
    cap = ui["telecomCaption"]
    assert "classifies the link's telemetry" not in cap, cap
    assert "synthetic" in cap and "do not feed it" in cap, cap
    row = ui["telecomIdsRow"]["a"]
    assert "classifies the link telemetry" not in row and "synthetic" in row, row


def test_the_blurbs_say_what_runs(ui: dict) -> None:
    tel = ui["blurbs"]["telecom"]
    assert "QML intrusion-detection plane" not in tel, tel
    assert "synthetic telemetry it generates" in tel and "neither the link's QBER nor the Eve control" in tel, tel
    assert "not intended for clinical use" in ui["blurbs"]["medical"]


def test_the_optiqkd_tag_names_the_bound_and_applies_nothing(ui: dict) -> None:
    label = ui["telecomQkdLabel"]
    assert "surrogate" not in label, label
    assert "asymptotic SKR, not applied" in label, label


def test_the_fibre_key_row_states_its_bases_and_its_receiver_loss(ui: dict) -> None:
    src = ui["fibreKeyRow"]["source"]
    assert "pZ = 0.5" in src and "Rusca et al. optimise the bias" in src, src
    assert "3.0 dB" in src and "6 dB" in src, src


# --------------------------------------------------------------------------------------------- #
# Source: the chart, the tooltips, the comments
# --------------------------------------------------------------------------------------------- #

def test_the_fibre_chart_labels_its_asymptotic_lines_and_draws_the_runs_key() -> None:
    src = (SRC / "components" / "QkdCharts.tsx").read_text(encoding="utf-8")
    fig = src[src.index("export function FibreDistanceCharts"):src.index("export function QsdcRangeChart")]
    assert re.search(r"<b>asymptotic</b> \(infinite-key\) bounds", fig), "the chart does not say its rates are asymptotic"
    assert 'name="decoy-state bound, asymptotic"' in fig
    assert 'dataKey="finite"' in fig and "composable finite key" in fig
    assert "pZ = 0.5" in fig and "RUSCA_TABLE_I_RECEIVER_DB" in fig
    assert "const RUSCA_TABLE_I_RECEIVER_DB = 6;" in src


def test_the_tooltips_say_what_optiqkd_and_webgpu_did() -> None:
    src = (SRC / "components" / "PipelineControls.tsx").read_text(encoding="utf-8")
    assert "raise the secret-key rate" not in src, "the OptiQKD tooltip still claims to raise the rate"
    assert "nothing it finds is applied to the run" in src
    gpu = src[src.index('id="sel-onnx-backend"'):src.index('<option value="">WASM (default)</option>')]
    assert "18216 ms on WebGPU with a hardware adapter" not in gpu, "the tooltip still blames WebGPU"
    assert "int8" in gpu and "ConvInteger" in gpu and "DynamicQuantizeLinear" in gpu, gpu
    assert "No fp32 graph was timed on WebGPU" in gpu


def test_the_onnx_header_says_the_slow_run_was_the_int8_graph() -> None:
    src = (SRC / "lib" / "onnxInference.ts").read_text(encoding="utf-8")
    head = src[:src.index("import type * as OrtTypes")]
    assert "INT8 graph" in head and "ConvInteger" in head, head
    assert "was NOT established here" not in src


def test_the_intensity_comments_name_each_optimums_objective() -> None:
    src = (REPO / "src" / "qsh" / "config" / "defaults.py").read_text(encoding="utf-8")
    assert "the shipped 2e7 block clears either threshold" not in src
    assert "MINIMISES THE PULSES A POSITIVE KEY NEEDS" in src
    assert "5,214,327 bits in 5.33e9 pulses against 4,977,637 in 5.25e9 at 0.399/0.098" in src
    assert "3.31e6 pulses (3,312,035)" in src and "3.60e6 (3,595,158)" in src


@pytest.mark.parametrize("rel", ["web/src/lib/clientFHE.ts", "web/src/components/QkdCharts.tsx",
                                 "web/src/components/PipelineControls.tsx", "web/src/lib/useCases.ts",
                                 "web/src/lib/useCaseFlows.ts", "web/src/lib/onnxInference.ts",
                                 "src/qsh/config/defaults.py", "src/qsh/models/registry.py",
                                 "web/src/lib/clientCompute.ts"])
def test_the_lines_this_round_wrote_are_ascii(rel: str) -> None:
    """The house rule, on the lines that carry this round's text (older lines keep their glyphs).

    UI strings in .ts/.tsx may keep the existing middle-dot separator, as the rule allows."""
    phrases = ("decryptOutputs", "CKKS_FLOOD_BITS", "encryptedDense", "RUSCA_TABLE_I_RECEIVER_DB",
               "nothing it finds is applied", "synthetic attack telemetry", "not intended for clinical use",
               "NOT intended for clinical use", "MINIMISES THE PULSES", "INT8 graph",
               "No student is distilled", "NO STUDENT IS DISTILLED", "studentNotRun", "qkd_tuned_finite_key",
               "asymptotic SKR, not applied", "ConvInteger and DynamicQuantizeLinear nodes")
    lines = (REPO / rel).read_text(encoding="utf-8").splitlines()
    hits = [(i + 1, ln) for i, ln in enumerate(lines) if any(p in ln for p in phrases)]
    assert hits, f"{rel}: none of this round's lines found"
    allowed = {"\u00b7"} if rel.endswith((".ts", ".tsx")) else set()
    bad = [(i, ln) for i, ln in hits if any(ord(ch) > 127 and ch not in allowed for ch in ln)]
    assert not bad, bad
