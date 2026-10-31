"""The full deployment's QSDC stage names a statistics-limited abort, and its CKKS telemetry check runs
(Round 84).

WHY THIS EXISTS. `quantum_secure_direct_comm` has told three aborts apart since Round 55: the link
cannot close even without Eve, the block is too short for its check carriers to certify an error
rate (`statistics_limited`), or the link would close and the block certifies it, so the loss is
Eve's. The satellite stage emitted two of the three, and logged the statistics-limited abort - every
abort at the 74 bytes of telemetry the demo sends, wherever the link would close - as "this link
closes without Eve, so the loss is attributable". And `TelemetryAnomalyStage` was gated on a
delivery: QSDC withholds the demo's telemetry at every setting, so the CKKS check README.md describes
never ran, although it scores a window of the bundled telemetry, not the delivered payload, which is
why the browser's plaintext twin was moved out of its delivery branch long ago. The canvas's QSDC
arrow had no state for the third abort either, and read "QSDC direct telemetry" for it.

WHAT IS ASSERTED
  - At 1 km with no fixed loss and no eavesdropper, the stage reports `qsdc_statistics_limited` 1 and
    logs the abort as statistics-limited, not as an attributable loss; on the default link it is
    link-limited and not statistics-limited; with a 4 MB block under attack the attributable branch
    is the one that runs.
  - The CKKS check runs on the default link, out of view and under the PQC-only profile, and says it
    scores the bundled window.
  - The canvas's QSDC step draws the statistics-limited abort as withheld, not as a delivery.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from qsh.orchestration.events import EventType
from qsh.usecases import satellite_comms

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
SHORT_CLEAN = {"sat_distance_km": 1, "sat_atmospheric_db": 0.0, "sat_pointing_db": 0.0, "sat_eve": 0.0}


def _run(cfg: dict) -> tuple[dict, dict, dict]:
    pipe, ctx = satellite_comms.build({"seed": 1, **cfg})
    pipe.run(ctx, out_dir="results/")
    logs: dict[str, list[str]] = {}
    for e in pipe.bus.history:
        if e.type == EventType.LOG:
            logs.setdefault(e.stage, []).append(e.message)
    return pipe.bus.metrics(), ctx.artifacts, logs


def test_a_statistics_limited_abort_is_named_and_logged_as_such() -> None:
    m, art, logs = _run(SHORT_CLEAN)
    assert art["telemetry_delivered"] is False
    assert m["qsdc_statistics_limited"] == 1 and m["qsdc_link_budget_limited"] == 0 and m["qsdc_eavesdrop_detected"] == 0
    log = " ".join(logs["qsdc_telemetry"])
    assert "statistics-limited" in log and "check carriers" in log
    assert "attributable" not in log and "says nothing about the channel or an eavesdropper" in log


def test_the_default_link_is_link_limited_not_statistics_limited() -> None:
    m, _, logs = _run({})
    assert m["qsdc_link_budget_limited"] == 1 and m["qsdc_statistics_limited"] == 0
    assert "NOT AVAILABLE" in " ".join(logs["qsdc_telemetry"])


def test_the_attributable_branch_is_the_eavesdroppers() -> None:
    """A block long enough to certify an error rate, on a link that closes without her."""
    m, _, logs = _run({**SHORT_CLEAN, "sat_eve": 0.4, "telemetry": "S" * 4_194_304})
    assert m["qsdc_eavesdrop_detected"] == 1 and m["qsdc_statistics_limited"] == 0
    assert "the loss is attributable" in " ".join(logs["qsdc_telemetry"])


@pytest.mark.parametrize("cfg", [{}, {"sat_distance_km": 3000}, {"qkd": "pqc-only"}],
                         ids=["default link", "out of view", "PQC-only"])
def test_the_ckks_telemetry_check_runs_whatever_qsdc_did(cfg) -> None:
    m, art, logs = _run(cfg)
    assert art.get("telemetry_delivered") is not True
    assert m["telemetry_anomaly_detected"] in (0, 1) and art["telemetry_anomaly"] == m["telemetry_anomaly_detected"]
    assert art["explanation"]["method"] and len(art["explanation"]["top_features"]) == 3
    log = " ".join(logs["telemetry_anomaly"])
    assert "(CKKS) on a window of the bundled SMAP/MSL telemetry, not the QSDC payload" in log
    assert "skipped" not in log


def test_the_docstrings_describe_the_stage_that_runs() -> None:
    doc = satellite_comms.TelemetryAnomalyStage.__doc__
    assert "runs whether or not QSDC delivered" in doc and "After QSDC delivers" not in doc
    assert "withholds them" in satellite_comms.__doc__ and "A CKKS anomaly check scores a window" in satellite_comms.__doc__


@pytest.fixture(scope="module")
def flows() -> dict:
    if not ESBUILD.exists():
        pytest.skip("web/node_modules/.bin/esbuild is absent; run `npm ci` in web/")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        bundle, runner = d / "bundle.mjs", d / "run.mjs"
        entry = f"export * from {json.dumps(str(WEB / 'src/lib/useCaseFlows'))};\n"
        subprocess.run([str(ESBUILD), "--bundle", "--loader=ts", "--format=esm", "--platform=node",
                        '--define:import.meta.env={"BASE_URL":"/"}', f"--outfile={bundle}", "--log-level=error"],
                       cwd=WEB, check=True, capture_output=True, input=entry.encode())
        runner.write_text(f"import * as m from {json.dumps(bundle.as_uri())};\n" + """
const step = m.USECASE_SPECS.satellite_comms.steps.find((s) => s.stageId === "qsdc_telemetry");
const cases = {
  statistics: { qsdc_statistics_limited: 1, qsdc_link_budget_limited: 0, qsdc_eavesdrop_detected: 0, qsdc_secrecy_capacity: -1.85 },
  link: { qsdc_statistics_limited: 0, qsdc_link_budget_limited: 1, qsdc_eavesdrop_detected: 0, qsdc_secrecy_capacity: -532.5,
          channel_loss_db: 32.2, qsdc_eve_gain_g: 532.7 },
  delivered: { qsdc_link_budget_limited: 0, qsdc_eavesdrop_detected: 0, qsdc_secrecy_capacity: 0.4 },
};
console.log(JSON.stringify({
  ...Object.fromEntries(Object.entries(cases).map(([k, mm]) => [k, { label: step.label(mm, {}), ok: step.ok(mm, {}) }])),
  caption: m.captionOf("satellite_comms", {}),
}));
""")
        return json.loads(subprocess.run(["node", str(runner)], check=True, capture_output=True, text=True).stdout)


def test_the_canvas_draws_the_statistics_limited_abort_as_withheld(flows) -> None:
    s = flows["statistics"]
    assert s["ok"] is False and "statistics-limited" in s["label"] and "telemetry withheld" in s["label"]
    assert "QSDC direct telemetry" not in s["label"]
    assert flows["link"]["ok"] is False and "link-budget limited" in flows["link"]["label"]
    assert flows["delivered"]["ok"] is True and flows["delivered"]["label"].startswith("QSDC direct telemetry")


def test_the_canvas_caption_does_not_send_a_key_down_the_quantum_channel(flows) -> None:
    """In the browser the QKD stage yields a key length and no key bytes; the session key is the PQC
    handshake's (the Overview's key-source row)."""
    c = flows["caption"]
    assert "the key comes down the quantum channel" not in c
    assert "a QKD key length is modelled for the downlink (in this tab the session key is the PQC handshake's)" in c
