"""The E2E orchestration pipeline executor.

Runs a sequence of :class:`~qsh.orchestration.stages.Stage` objects with live event
emission and cooperative *start / stop / restart* control (so the API and SPA can
drive it). Produces three downloadable artifacts per run: a JSONL event log, a
summary JSON, and a metrics CSV.
"""

from __future__ import annotations

import csv
import json
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .crypto_agility import CryptoRegistry, default_registry
from .events import Event, EventBus, EventType
from ..monitoring.exporter import record_qber, record_run, record_stage
from .stages import (
    Context,
    ExplainabilityStage,
    PQCHandshakeStage,
    PrivateInferenceStage,
    QKDKeyAgreementStage,
    QSMPCComputeStage,
    QuantumComparisonStage,
    ResultStage,
    SignatureStage,
    Stage,
)


def _set_event() -> threading.Event:
    """A threading.Event that starts in the 'set' (running, not paused) state."""
    e = threading.Event()
    e.set()
    return e


class PipelineState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    DONE = "done"
    ERROR = "error"


@dataclass
class Pipeline:
    """An ordered list of stages with cooperative stop/restart control."""

    name: str
    stages: list[Stage]
    registry: CryptoRegistry = field(default_factory=default_registry)
    bus: EventBus = field(default_factory=EventBus)
    state: PipelineState = PipelineState.IDLE
    _stop: bool = False
    _resume_gate: threading.Event = field(default_factory=lambda: _set_event())
    _step_budget: int = -1   # -1 = run freely; >=0 = stages allowed before auto-pause

    def request_stop(self) -> None:
        """Ask the pipeline to halt after the current stage (cooperative)."""
        self._stop = True
        self._resume_gate.set()   # unblock a paused worker so it can observe the stop

    def request_pause(self) -> None:
        """Pause after the current stage finishes; keep all state.

        Emits the REQUEST. The run is still executing a stage at this point - the gate in `run()`
        announces `PIPELINE_PAUSE_EFFECTIVE` once it is genuinely blocked. Until Round 41 this method
        emitted nothing and the gate emitted `PIPELINE_PAUSE`, so a consumer that distinguishes the
        two saw only the effect under the request's name and never left its "pausing" state.
        """
        self._step_budget = 0
        self._resume_gate.clear()
        self.bus.emit(Event(EventType.PIPELINE_PAUSE, self.name, "pause requested"))

    def request_resume(self) -> None:
        """Resume free execution from where it paused."""
        self._step_budget = -1
        self._resume_gate.set()

    def request_step(self) -> None:
        """Execute exactly one more stage, then pause again."""
        self._step_budget = 1
        self._resume_gate.set()

    def reset(self) -> None:
        """Reset control + event history so the pipeline can be restarted."""
        self._stop = False
        self._step_budget = -1
        self._resume_gate.set()
        self.state = PipelineState.IDLE
        self.bus = EventBus()

    def run(self, ctx: Context, *, out_dir: str | Path | None = None,
            step_delay: float | None = None) -> Context:
        """Execute all stages, honouring stop / pause / resume / step between stages."""
        if step_delay is None:
            from ..config import get_config

            step_delay = get_config().orchestration.step_delay
        self.state = PipelineState.RUNNING
        self.bus.emit(Event(EventType.PIPELINE_START, self.name,
                            f"starting {self.name}", {"usecase": ctx.usecase,
                                                      "stages": [s.name for s in self.stages]}))
        try:
            for stage in self.stages:
                # gate: block here while paused (step_budget == 0). Resume/step releases it.
                if self._step_budget == 0 and not self._stop:
                    self.state = PipelineState.PAUSED
                    # The EFFECT, not the request. `request_pause()` announced the request; this
                    # fires only once the worker is actually about to block, which is what the UI
                    # needs before it may enable Resume and Step. This line said PIPELINE_PAUSE
                    # until Round 41 - the right moment under the wrong name.
                    self.bus.emit(Event(EventType.PIPELINE_PAUSE_EFFECTIVE, self.name, "paused"))
                    self._resume_gate.wait()
                    if not self._stop:
                        self.state = PipelineState.RUNNING
                        self.bus.emit(Event(EventType.PIPELINE_RESUME, self.name, "resumed"))
                if self._stop:
                    self.state = PipelineState.STOPPED
                    self.bus.emit(Event(EventType.PIPELINE_STOP, self.name, "stopped by request"))
                    break
                self.bus.emit(Event(EventType.STAGE_START, stage.name, f"-> {stage.name}"))
                t0 = time.monotonic()
                try:
                    stage.run(ctx, self.bus, self.registry)
                except Exception as exc:  # surface, don't crash the whole run
                    self.state = PipelineState.ERROR
                    self.bus.emit(Event(EventType.STAGE_ERROR, stage.name, str(exc),
                                        {"error": type(exc).__name__}))
                    raise
                dt = round((time.monotonic() - t0) * 1000, 2)
                record_stage(stage.name, dt)
                self.bus.metric(f"{stage.name}_ms", dt, stage.name)
                self.bus.emit(Event(EventType.STAGE_END, stage.name, f"{stage.name} done",
                                    {"ms": dt}))
                if self._step_budget > 0:      # step mode: consume budget, re-pause at 0
                    self._step_budget -= 1
                    if self._step_budget == 0:
                        self._resume_gate.clear()
                if step_delay:
                    time.sleep(step_delay)
            else:
                self.state = PipelineState.DONE
                self.bus.emit(Event(EventType.PIPELINE_END, self.name, "complete",
                                    {"metrics": self.bus.metrics()}))
        finally:
            # The exporter defined four metrics and three recorders and had ZERO call sites for
            # three rounds, so /metrics served a valid, permanently empty scrape and the Grafana
            # dashboard charted it. These are the call sites. `finally`, so a run that raised is
            # counted in the state it ended in rather than vanishing from the counter.
            record_run(ctx.usecase, self.state.value if hasattr(self.state, "value") else str(self.state))
            qber = self.bus.metrics().get("qber")
            if isinstance(qber, (int, float)):
                record_qber(float(qber))
            if out_dir:
                self._persist(ctx, out_dir)
        return ctx

    def _persist(self, ctx: Context, out_dir: str | Path) -> dict:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{ctx.usecase}_{int(self.bus.history[-1].t * 1000) if self.bus.history else 0}"
        log_path = out / f"{stem}.events.jsonl"
        self.bus.write_jsonl(log_path)
        summary = ctx.artifacts.get("summary", {"usecase": ctx.usecase,
                                                 "metrics": self.bus.metrics()})
        (out / f"{stem}.summary.json").write_text(json.dumps(summary, indent=2, default=str))
        metrics = self.bus.metrics()
        with (out / f"{stem}.metrics.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in metrics.items():
                w.writerow([k, v])
        for p, kind in (("events.jsonl", "log"), ("summary.json", "json"), ("metrics.csv", "csv")):
            self.bus.artifact(str(out / f"{stem}.{p}"), kind, "result")
        _prune_runs(out)
        return {"dir": str(out), "stem": stem}


def standard_stages() -> list[Stage]:
    """The canonical PQC -> QKD -> QSMPC -> private-inference -> result chain."""
    return [
        PQCHandshakeStage(),
        QKDKeyAgreementStage(),
        QSMPCComputeStage(),
        PrivateInferenceStage(),
        ExplainabilityStage(),
        QuantumComparisonStage(),     # unify: assemble the benchmark from this live run
        ResultStage(),
        SignatureStage(),
    ]


def build_pipeline(usecase: str, config: dict | None = None) -> tuple[Pipeline, Context]:
    """Build a (pipeline, context) for ``usecase``.

    Dispatches to ``qsh.usecases.<usecase>.build`` when present (Phase 4 use cases);
    otherwise returns a generic demo wired with synthetic data.
    """
    config = dict(config or {})
    try:
        import importlib

        mod = importlib.import_module(f"qsh.usecases.{usecase}")
        if hasattr(mod, "build"):
            return mod.build(config)
    except ModuleNotFoundError:
        pass
    # generic demo
    ctx = Context(usecase=usecase, config={"qsmpc_op": "psi", "inference": "ckks", **config})
    ctx.data["set_a"] = [f"acct{i}" for i in range(50)] + ["fraud-x", "fraud-y"]
    ctx.data["set_b"] = [f"acct{i}" for i in range(40, 60)] + ["fraud-x", "fraud-y"]
    return Pipeline(name=f"{usecase}-pipeline", stages=standard_stages()), ctx


def main(argv: list[str] | None = None) -> int:
    import argparse

    from rich.console import Console

    parser = argparse.ArgumentParser(description="Run a QSH E2E orchestration pipeline")
    parser.add_argument("--usecase", default="demo",
                        help="demo | finance_aml | medical_fl | genomics | telecom_ids | "
                             "satellite_comms")
    parser.add_argument("--out", default="results/", help="output directory for artifacts")
    parser.add_argument("--kem", default=None, help="override KEM provider")
    parser.add_argument("--qkd", default=None, help="qkd provider (bb84-decoy | pqc-only)")
    parser.add_argument("--fhe", default=None, help="fhe backend (tenseal-cpu | openfhe-cpu)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    console = Console()
    cfg = {k: v for k, v in (("kem", args.kem), ("qkd", args.qkd), ("fhe", args.fhe)) if v}
    pipe, ctx = build_pipeline(args.usecase, cfg)

    if not args.quiet:
        def printer(e: Event) -> None:
            colour = {"stage_start": "bold cyan", "metric": "green",
                      "stage_error": "bold red", "pipeline_end": "bold green"}.get(e.type.value)
            tag = f"[{colour}]{e.type.value}[/]" if colour else e.type.value
            console.print(f"[{e.t:6.3f}s] {tag} {e.stage} :: {e.message}")
        pipe.bus.subscribe(printer)

    pipe.run(ctx, out_dir=args.out)
    console.print(f"\n[bold]state[/]: {pipe.state.value}  "
                  f"[bold]metrics[/]: {json.dumps(pipe.bus.metrics(), default=str)}")
    return 0 if pipe.state in (PipelineState.DONE, PipelineState.STOPPED) else 1


if __name__ == "__main__":
    raise SystemExit(main())


def _prune_runs(out: Path) -> int:
    """Keep the newest ``orchestration.keep_runs`` runs in ``out``, delete the rest.

    Every run writes three files here and nothing ever removed them, so the directory grew for three
    consecutive rounds - 953 files, then 1133, then 1421 - on the VPS as well as locally. It is
    gitignored, which is exactly why it kept being easy to defer rather than easy to notice.

    Grouped by run stem rather than by file, so a run is retained or removed whole: half a run on
    disk is worse than none, because its metrics.csv would still parse.

    Returns the number of files removed. Never raises: a demo run must not fail because a stale
    artefact could not be unlinked.
    """
    from ..config import get_config

    try:
        keep = int(get_config().orchestration.keep_runs)
    except Exception:
        return 0
    if keep <= 0:
        return 0

    runs: dict[str, list[Path]] = {}
    for f in out.glob("*.*"):
        stem = f.name.split(".")[0]
        if stem:
            runs.setdefault(stem, []).append(f)
    if len(runs) <= keep:
        return 0

    def newest(files: list[Path]) -> float:
        try:
            return max(x.stat().st_mtime for x in files)
        except OSError:
            return 0.0

    removed = 0
    for _, files in sorted(runs.items(), key=lambda kv: newest(kv[1]))[:len(runs) - keep]:
        for f in files:
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    return removed
