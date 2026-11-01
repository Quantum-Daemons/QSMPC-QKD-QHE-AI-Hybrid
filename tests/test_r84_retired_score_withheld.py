"""The full deployment withholds the retired quantum-advantage score, as the browser does.

WHY THIS EXISTS (Round 84)

The 0-100 "quantum advantage" composite was retired from display on 2026-10-23: its inputs divided
a CKKS byte ratio by a QOTP qubit ratio, and a real encrypted inference by a simulated untrained
circuit. The browser has returned `advantage: null` since then
(tests/test_the_retired_comparisons_stay_retired.py). The full deployment did not: the comparison
stage still computed the score into `benchmark_result`, which rides into the ML-DSA-signed run
summary, and `/api/benchmark` returned it, with a comment saying the change was deferred. Round 83's
records put it in the defects round, which is this one.

WHAT IS ASSERTED

- A finance run, which has the measured MPC-FHE row the score used to be computed from, carries
  `advantage` None in its benchmark result and in the signed summary.
- `/api/benchmark` returns `advantage` None and no longer calls `compute_advantage`. Checked in the
  source, because the endpoint runs the OpenFHE benchmark, which this suite does not start.

On the base commit (de1111c) both fail: the run's score is a dict with a positive `score`, and the
endpoint returns `compute_advantage(results)`.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_a_run_carries_no_score_in_its_benchmark_or_its_signed_summary() -> None:
    from qsh.orchestration.pipeline import build_pipeline

    pipe, ctx = build_pipeline("finance_aml", {"model_size": "medium", "seed": 0})
    pipe.run(ctx)
    br = ctx.artifacts["benchmark_result"]
    assert {r["path"] for r in br["results"]} == {"MPC-FHE", "QSMPC-FHE"}, (
        "the run produced no MPC-FHE row, so the check below proves nothing")
    assert "advantage" in br and br["advantage"] is None
    signed = ctx.artifacts["summary"]["artifacts"]["benchmark_result"]
    assert signed["advantage"] is None


def test_the_benchmark_endpoint_returns_none() -> None:
    tree = ast.parse((REPO / "api" / "main.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "benchmark")
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    assert "compute_advantage" not in names, "the endpoint still computes the retired score"
    ret = next(n for n in ast.walk(fn) if isinstance(n, ast.Return))
    assert isinstance(ret.value, ast.Dict)
    entries = {k.value: v for k, v in zip(ret.value.keys, ret.value.values) if isinstance(k, ast.Constant)}
    assert isinstance(entries.get("advantage"), ast.Constant) and entries["advantage"].value is None
