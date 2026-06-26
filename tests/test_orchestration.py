"""Tests for the E2E orchestration engine: pipeline run, stop, crypto-agility, artifacts."""

import json

from qsh.orchestration.crypto_agility import default_registry
from qsh.orchestration.events import EventType
from qsh.orchestration.pipeline import (
    Pipeline,
    PipelineState,
    build_pipeline,
    standard_stages,
)
from qsh.orchestration.stages import Context


def test_demo_pipeline_runs_to_completion(tmp_path):
    pipe, ctx = build_pipeline("demo")
    pipe.run(ctx, out_dir=tmp_path)
    assert pipe.state is PipelineState.DONE
    m = pipe.bus.metrics()
    assert m["session_key_bits"] == 256
    assert m["psi_intersection_size"] >= 1
    assert "qber" in m
    # artifacts written
    assert list(tmp_path.glob("*.summary.json"))
    assert list(tmp_path.glob("*.metrics.csv"))
    assert list(tmp_path.glob("*.events.jsonl"))


def test_pipeline_stop_is_cooperative():
    pipe, ctx = build_pipeline("demo")
    pipe.request_stop()
    pipe.run(ctx)
    assert pipe.state is PipelineState.STOPPED
    # stopped before any stage completed
    assert not any(e.type is EventType.STAGE_END for e in pipe.bus.history)


def test_pipeline_restart():
    pipe, ctx = build_pipeline("demo")
    pipe.request_stop()
    pipe.run(ctx)
    assert pipe.state is PipelineState.STOPPED
    pipe.reset()
    _, ctx2 = build_pipeline("demo")
    pipe.run(ctx2)
    assert pipe.state is PipelineState.DONE


def test_crypto_agility_pqc_only_profile():
    reg = default_registry()
    assert "pqc-only" in reg.options("qkd")
    ctx = Context(usecase="demo", config={"qkd": "pqc-only", "qsmpc_op": "psi",
                                           "inference": "ckks"})
    ctx.data["set_a"] = ["a", "b", "c"]
    ctx.data["set_b"] = ["b", "c", "d"]
    pipe = Pipeline("pqc-only", standard_stages(), registry=reg)
    pipe.run(ctx)
    assert pipe.state is PipelineState.DONE
    assert ctx.metrics.get("qber") is None  # QKD skipped


def test_event_log_is_serializable():
    pipe, ctx = build_pipeline("demo")
    pipe.run(ctx)
    for e in pipe.bus.history:
        json.loads(e.to_json())  # every event round-trips through JSON
