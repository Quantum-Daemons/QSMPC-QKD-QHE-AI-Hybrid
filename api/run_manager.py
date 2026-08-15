"""In-process manager for orchestration runs (background threads + event queues).

Each run executes a :class:`~qsh.orchestration.pipeline.Pipeline` in a worker thread;
its events are mirrored into a thread-safe queue the WebSocket endpoint drains and
streams to the SPA. Supports start / stop / restart and artifact retrieval -- the
control surface the SPA's E2E Orchestration page drives.
"""

from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from qsh.orchestration.events import Event
from qsh.orchestration.pipeline import Pipeline, build_pipeline

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


@dataclass
class Run:
    run_id: str
    usecase: str
    config: dict
    pipeline: Pipeline
    ctx: object
    queue: queue.Queue[Event] = field(default_factory=queue.Queue)
    thread: threading.Thread | None = None

    @property
    def state(self) -> str:
        return self.pipeline.state.value

    def metrics(self) -> dict:
        return self.pipeline.bus.metrics()

    def artifacts(self) -> dict:
        return getattr(self.ctx, "artifacts", {})


class RunManager:
    """Tracks active/completed orchestration runs."""

    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}

    def list_usecases(self) -> list[dict]:
        return [
            {"id": "finance_aml", "title": "Finance — cross-bank AML / PSI",
             "industry": "finance", "depth": "deep"},
            {"id": "medical_fl", "title": "Medical — federated learning + private inference",
             "industry": "healthcare", "depth": "deep"},
            {"id": "genomics", "title": "Genomics — confidential analysis",
             "industry": "genomics", "depth": "scaffold"},
            {"id": "telecom_ids", "title": "Telecom/Defense — QKD+PQC + QML-IDS",
             "industry": "telecom", "depth": "scaffold"},
            {"id": "satellite_comms", "title": "Satellite — free-space QKD + QSDC telemetry",
             "industry": "satellite", "depth": "scaffold"},
            {"id": "demo", "title": "Generic demo (PQC→QKD→QSMPC→FHE)",
             "industry": "generic", "depth": "demo"},
        ]

    def start(self, usecase: str, config: dict | None = None,
              start_paused: bool = False) -> str:
        run_id = uuid.uuid4().hex[:12]
        pipe, ctx = build_pipeline(usecase, config or {})
        run = Run(run_id, usecase, config or {}, pipe, ctx)
        pipe.bus.subscribe(run.queue.put)         # mirror events into the queue
        if start_paused:                          # begin paused so Step can advance stage-by-stage
            pipe.request_pause()
        self._runs[run_id] = run

        def worker():
            try:
                pipe.run(ctx, out_dir=RESULTS_DIR)
            finally:
                run.queue.put(Event(type=_SENTINEL, message="__end__"))

        run.thread = threading.Thread(target=worker, daemon=True)
        run.thread.start()
        return run_id

    def stop(self, run_id: str) -> bool:
        run = self._runs.get(run_id)
        if not run:
            return False
        run.pipeline.request_stop()
        return True

    def pause(self, run_id: str) -> bool:
        run = self._runs.get(run_id)
        if not run:
            return False
        run.pipeline.request_pause()
        return True

    def resume(self, run_id: str) -> bool:
        run = self._runs.get(run_id)
        if not run:
            return False
        run.pipeline.request_resume()
        return True

    def step(self, run_id: str) -> bool:
        run = self._runs.get(run_id)
        if not run:
            return False
        run.pipeline.request_step()
        return True

    def restart(self, run_id: str) -> str | None:
        run = self._runs.get(run_id)
        if not run:
            return None
        return self.start(run.usecase, run.config)

    def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    def status(self, run_id: str) -> dict | None:
        run = self._runs.get(run_id)
        if not run:
            return None
        return {
            "run_id": run_id, "usecase": run.usecase, "state": run.state,
            "metrics": run.metrics(), "artifacts": run.artifacts(),
        }


# sentinel event type for "stream finished"
from qsh.orchestration.events import EventType as _EventType  # noqa: E402

_SENTINEL = _EventType.PIPELINE_END
