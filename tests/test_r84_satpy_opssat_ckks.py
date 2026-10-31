"""The full deployment's satellite CKKS stage scores a bundled OPS-SAT-AD segment with the browser's model.

WHY THIS EXISTS (Round 84)

The README and the report's flowchart say "the full deployment also scores a bundled OPS-SAT-AD
segment under CKKS, whatever QSDC did". It did not. `TelemetryAnomalyStage` scored a window of
`data/satellite/smap_msl_telemetry.tsv`, NASA SMAP/MSL telemetry for which no provider grants a data
licence (DATA_LICENSES.md section 2; the registry's `DS_SMAP_MSL`), with a regression fitted in the
pipeline on that TSV and a synthetic fallback for when it was absent, while the browser's twin had
already moved to OPS-SAT-AD segments (CC BY 4.0) and a demo regression shipped with them in
`web/public/data/satellite/opssat_segments.json`. Two languages, two corpora, two models, one
sentence describing neither correctly.

WHAT IS ASSERTED

  - The SMAP/MSL TSV is gone from the tree, no source file names it and its loader is gone, `make
    data` lists the OPS-SAT-AD payload in its place, and the registry keeps `DS_SMAP_MSL` only as the
    record of why it is not shipped (attached to no model, not publishable).
  - The payload the Python loader reads is the browser's, byte for byte (the copy under `data/` is
    the one the API image ships), and the loader returns its model, rows and reference point, raising
    rather than substituting anything when the file is missing or malformed.
  - The stage scores one of the payload's held-out segments with the payload's regression under
    CKKS (TenSEAL), and reports the dataset, the segment, its label, the browser's score, the CKKS
    error and whether the encrypted verdict is the plaintext one; the SHAP reference point is the
    payload's E[x] over the fitting segments, recomputed here from the benchmark's test partition.
  - The logs name OPS-SAT-AD and the model in the payload's words, and never SMAP/MSL; without the
    payload the stage says so and scores nothing.

Every test fails at the base commit (aa293ab): the TSV is there, `load_opssat_segments` does not
exist, the stage scores 48-wide SMAP/MSL windows and logs "SMAP/MSL".
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
WEB_PAYLOAD = REPO / "web" / "public" / "data" / "satellite" / "opssat_segments.json"
DATA_PAYLOAD = REPO / "data" / "satellite" / "opssat_segments.json"
TSV = REPO / "data" / "satellite" / "smap_msl_telemetry.tsv"
OPSSAT_TEST_PARTITION = REPO / "models" / "satellite_opssat" / "eval_sample.json"


def _web() -> dict:
    return json.loads(WEB_PAYLOAD.read_text())


def _browser_margin(d: dict, k: int) -> float:
    """`realSatelliteAnomaly`'s margin for held-out position k: (W1 - W0) . x + (b1 - b0)."""
    W, b = np.asarray(d["model"]["W"], dtype=float), np.asarray(d["model"]["b"], dtype=float)
    return float((W[1] - W[0]) @ np.asarray(d["segments"][k], dtype=float) + (b[1] - b[0]))


def _run_stage(cfg: dict):
    """Build the satellite pipeline and run its telemetry stage on its own: it reads nothing the QKD
    and QSDC stages write, which is the point of it running whatever QSDC did."""
    from qsh.orchestration.crypto_agility import default_registry
    from qsh.orchestration.events import EventBus, EventType
    from qsh.usecases import satellite_comms

    _, ctx = satellite_comms.build(cfg)
    bus = EventBus()
    satellite_comms.TelemetryAnomalyStage().run(ctx, bus, default_registry())
    metrics = {e.data["name"]: e.data["value"] for e in bus.history if e.type == EventType.METRIC}
    logs = " ".join(e.message for e in bus.history if e.type == EventType.LOG)
    return ctx, metrics, logs


# --------------------------------------------------------------------------------------------- #
#  the SMAP/MSL file and every path to it
# --------------------------------------------------------------------------------------------- #

def _path_literals(name: str) -> list[str]:
    """Python files under src/ and api/ holding `name` as a path: a string constant that is the file
    name or ends in it with no space, as a `Path / "..."` join or a path string has it. Prose that
    mentions the file, as the registry's record of it does, is not a way to read it."""
    hits = []
    for root in ("src", "api"):
        for p in (REPO / root).rglob("*.py"):
            for node in ast.walk(ast.parse(p.read_text())):
                if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                        and node.value.endswith(name) and " " not in node.value):
                    hits.append(f"{p.relative_to(REPO)}:{node.lineno}")
    return hits


def test_the_smap_msl_tsv_is_gone_and_no_source_reads_it() -> None:
    assert not TSV.exists(), (
        "data/satellite/smap_msl_telemetry.tsv (NASA SMAP/MSL, no data licence from any provider) is "
        "still in the tree")
    readers = _path_literals("smap_msl_telemetry.tsv")
    assert not readers, f"source files still build a path to the SMAP/MSL TSV: {readers}"
    assert _path_literals("opssat_segments.json"), "the scan finds nothing, so it proves nothing"
    from qsh.data import loaders

    assert not hasattr(loaders, "load_satellite_telemetry"), "the SMAP/MSL loader is still importable"


def test_make_data_lists_the_opssat_payload_in_its_place() -> None:
    from qsh.data import download

    bundled = {label: path for label, path in download._bundled()}
    assert not [label for label in bundled if "smap" in label.lower() or "msl" in label.lower()]
    sat = {label: path for label, path in bundled.items() if label.startswith("satellite")}
    assert {p.relative_to(REPO).as_posix() for p in sat.values()} == {"data/satellite/opssat_segments.json"}
    assert all(p.exists() for p in sat.values())


def test_the_registry_keeps_smap_msl_as_a_record_nothing_loads() -> None:
    from qsh.models import registry

    ds = registry.DS_SMAP_MSL
    assert ds.licence == "no licence granted for the data"
    assert ds.publishable is False, "data with no licence marked publishable"
    assert "removed in Round 84" in ds.notes and "Retained as the bundled browser sample" not in ds.notes
    assert all(spec.dataset is not ds for spec in registry._SPECS), "a model spec carries DS_SMAP_MSL"


# --------------------------------------------------------------------------------------------- #
#  one payload, two languages
# --------------------------------------------------------------------------------------------- #

def test_the_python_payload_is_the_browsers_byte_for_byte() -> None:
    from qsh.data.loaders import OPSSAT_SEGMENTS

    assert OPSSAT_SEGMENTS == DATA_PAYLOAD
    assert DATA_PAYLOAD.read_bytes() == WEB_PAYLOAD.read_bytes(), (
        "data/satellite/opssat_segments.json is not web/public/data/satellite/opssat_segments.json: the "
        "full deployment would score a different model or different rows from the browser. Regenerate "
        "the web file with scripts/data/build_heldout_payloads.py --only satellite and copy it.")


def test_the_loader_returns_the_payloads_model_rows_and_reference_point() -> None:
    from qsh.data.loaders import load_opssat_segments

    d, p = _web(), load_opssat_segments()
    assert p.names == tuple(d["names"]) and len(p.names) == 19
    np.testing.assert_array_equal(p.weight, np.asarray(d["model"]["W"], dtype=float))
    np.testing.assert_array_equal(p.bias, np.asarray(d["model"]["b"], dtype=float))
    np.testing.assert_array_equal(p.segments, np.asarray(d["segments"], dtype=float))
    assert p.segments.shape == (159, 19)
    assert p.labels.tolist() == d["labels"] and p.row_index.tolist() == d["row_index"]
    np.testing.assert_array_equal(p.baseline, np.asarray(d["baseline"], dtype=float))
    assert p.baseline_note == d["baseline_note"] and p.model_note == d["real_model"]
    assert p.licence == "CC-BY-4.0" and "OPS-SAT" in p.provenance and "SMAP" not in p.provenance


def test_a_missing_or_malformed_payload_raises_rather_than_falling_back(tmp_path: Path) -> None:
    from qsh.data.loaders import load_opssat_segments

    with pytest.raises(FileNotFoundError):
        load_opssat_segments(tmp_path / "absent.json")
    d = _web()
    d["segments"] = [row[:18] for row in d["segments"]]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="segments"):
        load_opssat_segments(bad)


# --------------------------------------------------------------------------------------------- #
#  the stage
# --------------------------------------------------------------------------------------------- #

def test_the_stage_scores_a_held_out_segment_with_the_browsers_model_under_ckks() -> None:
    d = _web()
    ctx, m, _ = _run_stage({"seed": 1})
    k = ctx.data["telemetry_segment"]
    np.testing.assert_array_equal(ctx.data["features"], np.asarray(d["segments"][k], dtype=float))
    browser_score = 1.0 / (1.0 + math.exp(-_browser_margin(d, k)))      # realSatelliteAnomaly
    assert m["telemetry_anomaly_dataset"] == "OPS-SAT-AD"
    assert m["telemetry_anomaly_segment"] == d["row_index"][k]
    assert m["telemetry_anomaly_label"] == d["labels"][k]
    assert m["telemetry_anomaly_detected"] == int(browser_score > 0.5) == ctx.artifacts["telemetry_anomaly"]
    assert abs(m["telemetry_anomaly_score"] - browser_score) < 1e-4      # four decimals, CKKS error
    assert m["telemetry_anomaly_ckks_matches_plaintext"] == 1
    assert 0.0 < m["telemetry_anomaly_ckks_max_abs_error"] < 1e-4
    case = ctx.artifacts["telemetry_anomaly_case"]
    assert "tenseal" in case["ckks_backend"], case["ckks_backend"]
    assert case["dataset"] == "OPS-SAT-AD" and case["segment_position"] == k
    assert case["licence"] == "CC-BY-4.0" and case["licence_url"] == d["licence_url"]
    assert case["attribution"] == d["attribution"] and case["model"] == d["real_model"]


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_the_segment_is_drawn_from_the_runs_seed_by_the_browsers_rule(seed: int) -> None:
    """k = floor(u n) mod n, as `realSatelliteAnomaly` draws it, on the run's own seeded generator."""
    from qsh.usecases import satellite_comms

    _, ctx = satellite_comms.build({"seed": seed})
    n = len(_web()["segments"])
    assert ctx.data["telemetry_segment"] == int(np.random.default_rng(seed).random() * n) % n


def test_a_run_can_pin_its_segment_and_an_out_of_range_one_is_refused() -> None:
    from qsh.usecases import satellite_comms

    ctx, m, _ = _run_stage({"seed": 1, "telemetry_segment": 5})
    assert ctx.data["telemetry_segment"] == 5 and m["telemetry_anomaly_segment"] == _web()["row_index"][5]
    with pytest.raises(ValueError, match="159 held-out segments"):
        satellite_comms.build({"telemetry_segment": 159})


def test_the_shap_reference_point_is_the_fitting_segments_mean() -> None:
    """The payload's E[x] over the 370 fitting segments, recomputed from the test partition, not the
    mean of the held-out rows being explained, which is a different point."""
    d = _web()
    ctx, _, _ = _run_stage({"seed": 1})
    exp, k = ctx.artifacts["explanation"], ctx.data["telemetry_segment"]
    src = json.loads(OPSSAT_TEST_PARTITION.read_text())
    X = np.asarray(src["samples"], dtype=float)
    fit = np.array(sorted(set(range(len(X))) - set(d["row_index"])))
    mu, sd = X[fit].mean(axis=0), X[fit].std(axis=0)
    base = np.asarray(d["baseline"], dtype=float)
    np.testing.assert_allclose(base, ((X[fit] - mu) / sd).mean(axis=0), atol=1e-8)
    assert np.abs(np.asarray(d["segments"], dtype=float).mean(axis=0) - base).max() > 0.05

    W, b = np.asarray(d["model"]["W"], dtype=float), np.asarray(d["model"]["b"], dtype=float)
    w_eff, x = W[1] - W[0], np.asarray(d["segments"][k], dtype=float)
    phi = w_eff * (x - base)
    top = sorted(range(len(phi)), key=lambda i: -abs(phi[i]))[:3]
    assert [t["feature"] for t in exp["top_features"]] == [d["names"][i] for i in top]
    assert all(abs(t["contribution"] - phi[i]) < 1e-6 for t, i in zip(exp["top_features"], top, strict=True))
    assert exp["base_value"] == pytest.approx(float(w_eff @ base + b[1] - b[0]), abs=1e-6)
    assert exp["margin"] == pytest.approx(_browser_margin(d, k), abs=1e-6)
    assert exp["baseline"] == d["baseline_note"] and exp["efficiency_gap"] < 1e-9
    # The side artifacts the page's SHAP panel reads, with the browser's values for this step.
    browser_provenance = "demo OPS-SAT-AD regression (not the published model)"
    cc = (REPO / "web" / "src" / "lib" / "clientCompute.ts").read_text()
    assert f'artifacts.explanation_provenance = "{browser_provenance}"' in cc, "the browser's label moved"
    assert ctx.artifacts["explanation_provenance"] == browser_provenance
    assert ctx.artifacts["explanation_baseline"] == d["baseline_note"]
    assert ctx.artifacts["explanation_is_of_reported_class"] is True


def test_the_log_names_ops_sat_ad_the_segment_and_the_model() -> None:
    d = _web()
    _, m, log = _run_stage({"seed": 1})
    assert f"on OPS-SAT-AD segment {m['telemetry_anomaly_segment']} of the benchmark's test partition" in log
    assert "not the QSDC payload" in log and "so the verdict is the plaintext one" in log
    assert d["real_model"] in log, "the log does not say what the model is in the payload's words"
    assert "SMAP" not in log and "MSL" not in log


def test_without_the_payload_the_stage_says_so_and_scores_nothing(monkeypatch, tmp_path: Path) -> None:
    from qsh.data import loaders

    monkeypatch.setattr(loaders, "OPSSAT_SEGMENTS", tmp_path / "absent.json")
    ctx, m, log = _run_stage({"seed": 1})
    assert "anomaly check skipped: the bundled OPS-SAT-AD payload is missing" in log
    assert not [name for name in m if name.startswith("telemetry_anomaly")]
    assert "telemetry_anomaly" not in ctx.artifacts and "explanation" not in ctx.artifacts


def test_the_full_run_signs_the_segment_with_its_credit(tmp_path: Path) -> None:
    """Through the whole pipeline: the run's signed summary carries the case and its CC BY credit."""
    from qsh.usecases import satellite_comms

    pipe, ctx = satellite_comms.build({"seed": 1})
    pipe.run(ctx, out_dir=str(tmp_path))
    signed = ctx.artifacts["summary"]
    case = signed["artifacts"]["telemetry_anomaly_case"]
    assert case["dataset"] == "OPS-SAT-AD" and case["matches_plaintext"] is True
    assert case["attribution"] == _web()["attribution"]
    assert signed["metrics"]["telemetry_anomaly_dataset"] == "OPS-SAT-AD"
    assert ctx.artifacts["result_signature"]["verified"] is True


def test_no_script_writes_the_smap_msl_tsv() -> None:
    """A maintainer's rerun of the Kaggle prep script must not bring the unlicensed TSV back."""
    prep = (REPO / "scripts" / "prep_real_kaggle_data.py").read_text()
    assert '/ "smap_msl_telemetry.tsv"' not in prep and "def prep_satellite" not in prep
