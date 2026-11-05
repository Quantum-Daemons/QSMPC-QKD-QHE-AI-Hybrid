"""The Overview, Value, Use Cases and Validations pages claim what the sources and the code support.

WHAT WAS WRONG (Round 84; scratchpad audit83/page-overview-value-validations.md and
private-inference-vs-research.md), and the primary source each correction rests on:

1. Overview said the ML-KEM + X25519 layer "implements" what BSI recommends. BSI TR-02102-1 v2026-01
   recommends ML-KEM-768/1024 (Table 2.7) and the Brainpool curves (Table B.3) and never mentions
   25519; TR-02102-2 v2026-01 leaves x25519 out of its TLS 1.3 groups (Table 10). BSI's TR-02102 page,
   updated 2026-10-02 after RFC 10024, says SecP256r1MLKEM768 and SecP384r1MLKEM1024 will be recommended
   in the next TR-02102-2 and may already be used, and that X25519MLKEM768 may also be used.
2. "0 server-side compute" (Overview KPI, Value's novelty sentence) was unconditional; a full
   deployment sends every run to its server.
3. Value: "HNDL is active" (EO 14412 sec. 1 calls it a risk); the value table omitted the PSI key
   both banks hold, the session key the aggregator holds, the classical IDS and its silence on HNDL;
   "not a present-day mandate" missed the GDPR duty the CJEU confirmed in C-203/22 (27 Feb 2025).
4. Overview named "Rate-1 QFHE transport", a homomorphic scheme the code does not run, and called the
   comparison stage by a third name.
5. The satellite card printed "event-wise F0.5 0.0000 vs published 0.968" with none of its registry
   entry's context; the Use Cases table put "+ QML-IDS" in its quantum-channel column.
6. NetKeyCards called cacophony's Noise vector "official" and stated the IKEv2 ML-KEM status undated.
7. The regulatory clock cited EO 14412 sec. 4(b) alone for an obligation that reaches agencies
   through OMB M-26-15, and lacked M-26-15 sec. 4's TLS 1.3 date (2030-01-02, under EO 14306).
8. The benchmark promised an "accuracy" no browser row fills, and headed the full deployment's
   agreement-with-plaintext figure "accuracy".
9. The QCNN log lines said "this reports what was measured" over figures read from the weights file.

HOW. The pages are rendered to HTML in node, from the real TSX, once for each build: the client-only
build the public demo ships (`VITE_CLIENT_ONLY=1`) and a full build. Text checks on source strip the
comments first, because the comments recording each correction quote the wording they replaced.
"""

from __future__ import annotations

import html
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
CONFIG = SRC / "lib" / "config.defaults.json"

# ----------------------------------------------------------------------------------------------------
# Rendering the real pages in node, in both builds
# ----------------------------------------------------------------------------------------------------

#: A full deployment's live benchmark, in the shape `QuantumComparisonStage` assembles: the MPC-FHE
#: row as `classical_mpc_fhe.run_hybrid` returns it for the run's one input (`extra.samples` 1), and
#: the comparison row, which reports no accuracy.
FULL_BENCHMARK = {
    "workload": "linear", "model_size": "lightweight", "live": True, "advantage": None,
    "results": [
        {"path": "MPC-FHE", "backend": "tenseal + mpyc-shamir", "workload": "linear",
         "model_size": "lightweight", "latency_ms": 25.9, "throughput_ips": 38.6, "accuracy": 1.0,
         "ciphertext_plaintext_ratio": 1456.0, "security_model": "computational (RLWE) + secret-sharing MPC",
         "fhe_linear_ms": 20.1, "mpc_nonlinear_ms": 5.8, "mpc_comm_kb": 3.2, "device": "cpu",
         "projected": False, "extra": {"samples": 1}},
        {"path": "QSMPC-FHE", "backend": "qotp + clifford+t", "workload": "linear", "model_size": "lightweight",
         "latency_ms": 1.2, "accuracy": None, "t_gate_count": 56, "qubits": 8,
         "security_model": "information-theoretic as a scheme (QOTP)", "projected": False, "extra": {}},
    ],
}

_HARNESS = r"""
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const [WEB, OUT, CLIENT_ONLY, BENCH] = process.argv.slice(2);
const esbuild = await import(pathToFileURL(join(WEB, "node_modules/esbuild/lib/main.js")).href);
const src = (p) => JSON.stringify(join(WEB, p));
const env = CLIENT_ONLY === "1" ? '{"BASE_URL":"/","VITE_CLIENT_ONLY":"1"}' : '{"BASE_URL":"/"}';
const bundle = OUT + ".bundle.mjs";
await esbuild.build({
  stdin: {
    contents: [
      `export { default as Overview } from ${src("src/pages/Overview")};`,
      `export { default as Value } from ${src("src/pages/Value")};`,
      `export { default as UseCases } from ${src("src/pages/UseCases")};`,
      `export { default as Validations } from ${src("src/pages/Validations")};`,
      `export { default as NetKeyCards } from ${src("src/components/NetKeyCards")};`,
      `export { default as BenchmarkPanel } from ${src("src/components/BenchmarkPanel")};`,
      `export { CLOCK } from ${src("src/lib/regulatoryClock")};`,
      `export { createElement } from "react";`,
      `export { renderToStaticMarkup } from "react-dom/server.browser";`,
      `export { MemoryRouter } from "react-router";`,
    ].join("\n"),
    resolveDir: WEB, loader: "tsx",
  },
  bundle: true, format: "esm", platform: "node", outfile: bundle, logLevel: "error", jsx: "automatic",
  define: { "import.meta.env": env, "process.env.NODE_ENV": '"production"' },
  plugins: [{ name: "stub-unresolvable", setup(b) {
    b.onResolve({ filter: /qsh_mpc\.js$|\?url$|\?worker$/ }, (a) => ({ path: a.path, namespace: "stub" }));
    b.onLoad({ filter: /.*/, namespace: "stub" }, () => ({ contents: "export default '';", loader: "js" }));
  } }],
});
const m = await import(pathToFileURL(bundle).href);
const page = (C, props = null) => m.renderToStaticMarkup(m.createElement(m.MemoryRouter, null, m.createElement(C, props)));
writeFileSync(OUT, JSON.stringify({
  overview: page(m.Overview), value: page(m.Value), useCases: page(m.UseCases),
  validations: page(m.Validations), netKeyCards: page(m.NetKeyCards),
  benchEmpty: page(m.BenchmarkPanel, { result: null }),
  benchFull: page(m.BenchmarkPanel, { result: JSON.parse(BENCH) }),
  clock: m.CLOCK,
}));
"""


def _require_node_or_skip() -> None:
    missing = [n for n, ok in (("web/node_modules/.bin/esbuild", (WEB / "node_modules" / ".bin" / "esbuild").exists()),
                               ("node", shutil.which("node") is not None)) if not ok]
    if not missing:
        return
    msg = f"{' and '.join(missing)} absent; run `npm ci` in web/. This is a SKIP, not a pass."
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


def _text(markup: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", markup))).strip()


@pytest.fixture(scope="module")
def rendered(tmp_path_factory) -> dict[str, dict]:
    """{"client": {...}, "full": {...}}: every page's HTML in each build, plus the clock's rows."""
    _require_node_or_skip()
    d = tmp_path_factory.mktemp("r84-pages")
    script = d / "harness.mjs"
    script.write_text(_HARNESS)
    out: dict[str, dict] = {}
    for build, flag in (("client", "1"), ("full", "0")):
        target = d / f"{build}.json"
        r = subprocess.run(["node", str(script), str(WEB), str(target), flag, json.dumps(FULL_BENCHMARK)],
                           cwd=WEB, capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, f"rendering the {build} build failed:\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}"
        out[build] = json.loads(target.read_text())
    return out


def _code(rel: str) -> str:
    """Source with comments blanked (TS block, line and JSX comments), whitespace collapsed."""
    text = (SRC / rel).read_text()
    text = re.sub(r"/\*[\s\S]*?\*/", " ", text)
    text = re.sub(r"(?<![:\"'`])//[^\n]*", " ", text)
    return re.sub(r"\s+", " ", text)


# ----------------------------------------------------------------------------------------------------
# 1. BSI
# ----------------------------------------------------------------------------------------------------

def test_overview_credits_bsi_with_the_ml_kem_half_only(rendered) -> None:
    for build in ("client", "full"):
        t = _text(rendered[build]["overview"])
        assert "which the ML-KEM + X25519 layer implements" not in t, "the layer is said to implement BSI's recommendation again"
        assert "only its ML-KEM half is a BSI-recommended mechanism (Table 2.7)" in t
        assert "X25519 is neither among BSI's recommended curves (the Brainpool curves, Table B.3)" in t
        assert "TLS 1.3 groups (TR-02102-2 v2026-01, Table 10)" in t
        assert "SecP256r1MLKEM768 and SecP384r1MLKEM1024 will be recommended in the next TR-02102-2" in t
        assert "X25519MLKEM768 can also be used" in t
        # What BSI does say, unchanged: QKD not recommended, hybrid recommended.
        assert '"is currently not ready for use from a security point of view"' in t
        assert "hybrid key agreement combining a quantum-safe and a classical mechanism (section 2.2)" in t


# ----------------------------------------------------------------------------------------------------
# 2. Zero server compute is a fact of one build
# ----------------------------------------------------------------------------------------------------

def test_zero_server_compute_is_stated_for_the_client_only_build(rendered) -> None:
    client, full = _text(rendered["client"]["overview"]), _text(rendered["full"]["overview"])
    assert "0 server-side compute in this demo: CKKS, garbled-circuit MPC and the model inference all run in your tab" in client
    assert "server-side compute in this demo" not in full, "a full deployment is told it does no server compute"
    assert "0 server-side compute in the client-only build the public demo ships" in full
    assert "This full deployment is not that build: its runs go to its server" in full

    v_client, v_full = _text(rendered["client"]["value"]), _text(rendered["full"]["value"])
    claim = "chained end to end in one browser tab with no server-side compute, in the client-only build the public demo ships"
    assert claim + "." in v_client
    assert claim + " (this full deployment runs its pipeline on its server instead)." in v_full


def test_the_novelty_sentence_still_reads_as_one_sentence_in_three_files() -> None:
    """The Value copy keeps the chain wording the README and the report carry (the existing gate's subject)."""
    v = _code("pages/Value.tsx")
    m = re.search(r"PQC key establishment(.{0,160}?)secure multiparty computation", v)
    assert m and "a simulated QKD link (it yields a key length, not key bytes)" in m.group(1)


# ----------------------------------------------------------------------------------------------------
# 3. Value page
# ----------------------------------------------------------------------------------------------------

def test_hndl_is_a_risk_not_an_observed_campaign(rendered) -> None:
    t = _text(rendered["client"]["value"])
    assert "HNDL is a present risk" in t
    assert "HNDL is active" not in t and "HNDL is active" not in _code("pages/Value.tsx")


def test_the_value_table_states_the_limits_the_code_has(rendered) -> None:
    """The PSI's limit is the OPRF PSI's since Round 84 (psi): semi-honest and not post-quantum. The
    keyed hash whose key both banks held, which the row stated before, no longer runs."""
    t = _text(rendered["client"]["value"])
    assert "QML-IDS could reduce" not in t, "the classical classifier is credited to QML again"
    assert "previously couldn't be combined" not in t
    assert "The PSI is an HMAC" not in t and "OPRF-based protocol" not in t, "the retired keyed-hash PSI is described again"
    for needle in (
        "The PSI is semi-honest and not post-quantum: it runs RFC 9497's OPRF over ristretto255",
        "Each pair of federated clients agrees its mask seed by X25519, so the aggregator cannot remove a mask alone, "
        "though every client runs in this tab and there is no dropout recovery or authentication of the public keys",
        "The classifier is classical: the page runs a softmax and the published random forest",
        "It does not address HNDL, which is passive recording of classical traffic",
        "this demo finds that QSDC cannot close over a LEO downlink",
    ):
        assert needle in t, f"the value table lost: {needle!r}"


def test_the_quantum_classifiers_still_lose_to_the_forest() -> None:
    """The table says the quantum-kernel SVC and the QNN lose to the forest; the registry must agree."""
    models = json.loads(CONFIG.read_text())["models"]["telecom_ids"]
    a = next(m for m in models if m["key"] == "telecom_ids.qml")["achieved"]
    assert a["quantum_beats_classical"] is False
    assert max(a["quantum_svc_macro_f1"], a["quantum_qnn_macro_f1"]) < a["classical_rf_macro_f1"]


def test_explanations_are_owed_today_and_the_signature_is_not(rendered) -> None:
    t = _text(rendered["client"]["value"])
    assert "not a present-day mandate" not in t, "the GDPR duty to explain automated decisions is denied again"
    assert "Explanations of automated decisions are already owed today" in t
    assert "Art. 15(1)(h) with Art. 22, as the CJEU held in C-203/22, Dun & Bradstreet Austria, 27 February 2025" in t
    assert "Neither requires the explanation to be signed" in t


# ----------------------------------------------------------------------------------------------------
# 4. Overview names
# ----------------------------------------------------------------------------------------------------

def test_overview_names_no_scheme_the_code_does_not_run(rendered) -> None:
    t = _text(rendered["client"]["overview"])
    assert "Rate-1 QFHE transport" not in t
    assert "Rate-1 QFHE is implemented" not in t
    assert "a QOTP sent with its pad under a KEM (transport only, not a homomorphic scheme)" in t
    assert "so it is not the Rate-1 QFHE scheme (Chardouvelis, Doettling and Malavolta, TCC 2021)" in t


def test_overview_calls_the_comparison_stage_by_the_one_name() -> None:
    """The run log, the benchmark row, the use-case page and the orchestration graph say QSMPC-FHE."""
    code = _code("pages/Overview.tsx")
    assert "comparison circuit" not in code and "QHE comparison" not in code
    assert code.count("the QSMPC-FHE comparison, an untrained QOTP + Clifford+T circuit") == 2


# ----------------------------------------------------------------------------------------------------
# 5. Use Cases
# ----------------------------------------------------------------------------------------------------

def _telemanom() -> dict:
    models = json.loads(CONFIG.read_text())["models"]["satellite_comms"]
    return next(m for m in models if m["key"] == "satellite_comms.telemanom")


def test_the_satellite_card_carries_its_registry_context(rendered) -> None:
    t = _text(rendered["client"]["useCases"])
    spec = _telemanom()
    n = spec["achieved"]["n_events_scored"]
    assert "event-wise F0.5 0.0000 vs published 0.968" not in t, "the bare juxtaposition is back"
    assert f"event-wise F0.5 0.0000 over {n} scored events \u00b7 published 0.968, a different measurement" in t
    assert ("Not a like-for-like pair: ESA-ADB's 0.968 is scored on 29 anomaly events (65 annotated events in "
            "all), with another split, resampling and precision correction.") in t
    assert "two earlier, identically configured trainings, before a scaler fix, scored 0.714 and 0.455" in t
    # Cards with a like-for-like baseline keep "vs published".
    assert "vs published 0.901" in t, "the medical card lost its ordinary comparison"


def test_every_number_in_the_card_sentence_is_the_registry_entrys() -> None:
    code = (SRC / "pages" / "UseCases.tsx").read_text()
    m = re.search(r'"satellite_comms\.telemanom":\s*((?:"[^"]*"\s*\+?\s*)+),', code)
    assert m, "the satellite card's sentence moved; re-derive this test"
    sentence = "".join(re.findall(r'"([^"]*)"', m.group(1)))
    spec = _telemanom()
    note, source = spec["achieved"]["baseline_gap_note"], spec["metric"]["baseline_source"]
    assert spec["metric"]["comparison_caveat"], "the registry no longer flags the pair as not like-for-like"
    assert spec["metric"]["baseline"] == 0.968 and "0.968" in sentence
    assert "65 annotated events of which 29 are anomalies" in source
    assert "29 anomaly events" in sentence and "65 annotated events" in sentence
    assert "0.714286" in note and "0.454545" in note and "identical-config retrains" in note
    assert "0.714 and 0.455" in sentence
    numbers = set(re.findall(r"\d+(?:\.\d+)?", sentence))
    assert numbers == {"0.968", "29", "65", "0.714", "0.455"}, f"an unpinned number entered the sentence: {numbers}"


def test_the_quantum_channel_column_names_no_classifier(rendered) -> None:
    page = rendered["client"]["useCases"]
    table = page[page.index("<table"):page.index("</table>")]
    cells = [_text(c) for c in re.findall(r"<tr>(.*?)</tr>", table, re.S)]
    telecom = next(c for c in cells if c.startswith("Telecom"))
    assert "QML-IDS" not in telecom and telecom.endswith("fibre BB84/E91 simulated"), telecom
    assert "QML-IDS" not in _code("pages/UseCases.tsx")
    assert "latency \u00b7 accuracy" not in _text(page), "the benchmark is promised an accuracy no browser row fills"


# ----------------------------------------------------------------------------------------------------
# 6. Network key cards
# ----------------------------------------------------------------------------------------------------

def test_the_noise_vector_is_named_for_what_it_is(rendered) -> None:
    t = _text(rendered["client"]["netKeyCards"])
    assert "official cacophony" not in t and "official cacophony" not in (SRC / "components" / "NetKeyCards.tsx").read_text()
    assert "checked on every run against the vector cacophony publishes (a Haskell implementation of Noise" in t
    # The repository's own provenance record says the same.
    noise = json.loads((SRC / "lib" / "netkeys" / "vectors" / "noise_ikpsk2.json").read_text())["source"]
    assert noise["publisher"].startswith("cacophony, a Haskell implementation")
    assert noise["listedBy"].endswith("noise_wiki/wiki/Test-vectors")


def test_the_ikev2_status_line_carries_the_date_it_was_checked(rendered) -> None:
    ike = (SRC / "lib" / "netkeys" / "ikev2.ts").read_text()
    checked = re.search(r'MLKEM_IKEV2_STATUS = \{[^}]*?checked: "(\d{4}-\d{2}-\d{2})"', ike, re.S)
    assert checked, "MLKEM_IKEV2_STATUS.checked moved; re-derive this test"
    card = (SRC / "components" / "NetKeyCards.tsx").read_text()
    copy = re.search(r'const MLKEM_STATUS_CHECKED = "(\d{4}-\d{2}-\d{2})";', card)
    assert copy and copy.group(1) == checked.group(1), (
        f"the card's date ({copy and copy.group(1)}) is not the module's ({checked.group(1)}): a re-check "
        "updates both")
    t = _text(rendered["client"]["netKeyCards"])
    assert f"As checked on {checked.group(1)}, ML-KEM in IKEv2 is not yet an RFC" in t


# ----------------------------------------------------------------------------------------------------
# 7. Regulatory clock
# ----------------------------------------------------------------------------------------------------

def test_the_agency_dates_name_the_memorandum_that_carries_them(rendered) -> None:
    clock = rendered["client"]["clock"]
    rows = {(r["lane"], r["when"]): r for r in clock}
    for when, phase in (("2030-12-31", "Phase 3"), ("2031-12-31", "Phase 4")):
        r = rows[("US federal", when)]
        assert r["instrument"] == f"EO 14412, sec. 4(b), via OMB M-26-15 ({phase})", r["instrument"]
    tls = rows.get(("US federal", "2030-01-02"))
    assert tls, "the clock has no TLS 1.3 row (OMB M-26-15 sec. 4, under EO 14306)"
    assert "M-26-15, sec. 4" in tls["instrument"] and "EO 14306" in tls["instrument"]
    assert tls["firmness"] == "requirement" and "TLS 1.3" in tls["what"] and "not NSS" in tls["binds"]
    ts = [r["t"] for r in clock]
    assert ts == sorted(ts), "the rows left time order, which the label alternation per lane relies on"
    # Rendered on the Value page, in the table and on the timeline, and the card above it agrees.
    t = _text(rendered["client"]["value"])
    assert "2030-01-02 OMB M-26-15, sec. 4, under EO 14306" in t and "TLS 1.3 support" in t
    assert "US: EO 14412, through OMB M-26-15, binds federal high value assets" in t


# ----------------------------------------------------------------------------------------------------
# 8. Benchmark labels
# ----------------------------------------------------------------------------------------------------

def test_the_benchmark_promises_no_accuracy_and_labels_agreement(rendered) -> None:
    empty = _text(rendered["client"]["benchEmpty"])
    assert "the benchmark fills in from that live run" in empty and "accuracy" not in empty
    full = rendered["full"]["benchFull"]
    assert "<th>accuracy</th>" not in full, "the agreement-with-plaintext column is headed 'accuracy' again"
    assert "<th>matches plaintext</th>" in full
    t = _text(full)
    assert "1 of 1" in t, "the MPC-FHE row's agreement is not shown as a count of its inputs"
    assert "it is agreement, not accuracy against true labels" in t


def test_the_full_deployment_field_is_agreement_with_plaintext() -> None:
    """What the label rests on: `run_hybrid` counts decisions that match the plaintext model's."""
    src = (REPO / "src" / "qsh" / "benchmark" / "classical_mpc_fhe.py").read_text()
    body = src[src.index("def run_hybrid"):]
    body = body[:body.index("\ndef ") if "\ndef " in body else len(body)]
    assert 'correct += int(r["decision_matches_plaintext"])' in body
    assert "accuracy=round(correct / n, 4)" in body and '"samples": n' in body
    stages = (REPO / "src" / "qsh" / "orchestration" / "stages.py").read_text()
    assert stages.count("n_samples=1)") >= 4, "the live run no longer benchmarks the run's one input"


# ----------------------------------------------------------------------------------------------------
# 9. QCNN log lines
# ----------------------------------------------------------------------------------------------------

def test_the_qcnn_log_says_its_figures_are_recorded_not_re_scored() -> None:
    # Log lines are template literals concatenated across source lines; join them before matching.
    code = re.sub(r"`\s*\+\s*`", "", _code("lib/clientCompute.ts"))
    assert "this reports what was measured" not in code
    assert "Reported because it is the measured result" not in code
    assert code.count("recorded at training time") == 3, (
        "each QCNN figure line (telecom verdict, telecom fallback, finance/medical) must say its "
        "figures were recorded at training time")
    assert code.count("not re-scored in this tab") == 2 and "this tab does not re-score them" in code
