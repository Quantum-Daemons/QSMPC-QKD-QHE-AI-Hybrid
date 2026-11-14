"""The satellite student's CKKS runs in a Web Worker beside the page, not on the page's thread (Round 86).

Measured in a browser tab on 2026-11-07, from a client-only build. The student's verification puts all 8,640 forecasts
through CKKS: 135 ciphertexts, about 21 s.
- On the page's thread: 138 long tasks of about 160 ms, 21.8 s of blocked input over a 24.5 s run.
- In the worker: 4 long tasks totalling 0.57 s, the longest 186 ms. The result was unchanged: max |decrypted -
  plaintext| 1.43e-9, 0 of 8,640 flags changed, and the published flag digest matched.

node has no Web Worker, so these hold the wiring that the measurement depends on:
- the page calls the off-thread entry point;
- the worker runs the same function, so `clientFHE.ts` (frozen for the poster) is not copied or bypassed;
- Vite can bundle the worker;
- a worker that fails is not retried on the page's thread, where the failure would hide behind 21 s of blocked input.

The CKKS itself is checked against the real node-seal by tests/test_r86_satellite_student_ckks_in_node.py.
"""

from __future__ import annotations

import re
from pathlib import Path

LIB = Path(__file__).resolve().parents[1] / "web" / "src" / "lib"
VITE = Path(__file__).resolve().parents[1] / "web" / "vite.config.ts"


def _off_thread_body() -> str:
    src = (LIB / "forecastStudent.ts").read_text()
    start = src.index("export async function evaluateForecastStudentOffThread")
    return src[start:src.index("/** The published weights", start)]


def test_the_page_evaluates_the_student_through_the_off_thread_entry() -> None:
    src = (LIB / "onnxInference.ts").read_text()
    assert 'await evaluateForecastStudentOffThread(f, w, { ckks: "all" })' in src
    assert not re.search(r"\bevaluateForecastStudent\(", src), "the page calls the CKKS on its own thread again"


def test_the_worker_runs_the_same_function_and_vite_can_bundle_it() -> None:
    worker = (LIB / "forecastStudent.worker.ts").read_text()
    assert 'import { evaluateForecastStudent' in worker and "await evaluateForecastStudent(file, weights, { ckks })" in worker
    assert 'from "./clientFHE"' not in worker and "encryptedLinearBatch(" not in worker, (
        "the worker carries a second copy of the CKKS")
    # Vite recognises a worker only in this exact form, and the worker reaches a dynamic import (node-seal), which the
    # default IIFE worker format cannot split.
    assert 'new Worker(new URL("./forecastStudent.worker.ts", import.meta.url), { type: "module" })' in _off_thread_body()
    # ES workers, whatever else the worker build is given (Round 86: a plugin that traces the workers' packages for the
    # notices file).
    assert re.search(r'worker:\s*\{\s*format:\s*"es"\s*[,}]', VITE.read_text()), "the worker would not bundle"


def test_a_failed_worker_is_reported_not_retried_on_the_page_thread() -> None:
    body = _off_thread_body()
    # The one in-thread call is the branch for an environment without a Worker (node); the browser path has none.
    assert body.count("evaluateForecastStudent(") == 1, "a second, in-thread call would be a fallback"
    first = body.index("evaluateForecastStudent(")
    assert first < body.index("new Worker(") and 'typeof Worker === "undefined"' in body[:first]
    assert "reject(new Error" in body and "worker.terminate()" in body
