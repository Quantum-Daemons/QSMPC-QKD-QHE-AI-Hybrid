"""Structured event bus for the orchestration pipeline.

Every stage emits :class:`Event` records to an :class:`EventBus`. Subscribers
include the CLI logger, a JSONL run-log writer, and (in the API) a WebSocket fan-out
that streams live progress to the SPA. Events carry enough structure to drive the
React-Flow DAG animation and the downloadable run artifacts.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable


class EventType(str, Enum):
    PIPELINE_START = "pipeline_start"
    PIPELINE_END = "pipeline_end"
    PIPELINE_STOP = "pipeline_stop"
    # Two events, because "pause was requested" and "the run has actually stopped" are different
    # facts and the UI shows different states for them ("pausing" vs "paused"). Only the second
    # enables Resume and Step. The browser gate has emitted both since Round 24
    # (`web/src/lib/clientCompute.ts`); this side had only PIPELINE_PAUSE, emitted from INSIDE the
    # gate - semantically the EFFECT, under the name the browser uses for the REQUEST. So against
    # this transport the UI reached "pausing" and stayed there: Resume and Step were disabled
    # forever and Start-stepwise was unusable from its first stage.
    PIPELINE_PAUSE = "pipeline_pause"                      # requested; a stage may still be running
    PIPELINE_PAUSE_EFFECTIVE = "pipeline_pause_effective"  # the worker is now blocked at the gate
    PIPELINE_RESUME = "pipeline_resume"
    STAGE_START = "stage_start"
    STAGE_END = "stage_end"
    STAGE_ERROR = "stage_error"
    LOG = "log"
    METRIC = "metric"
    ARTIFACT = "artifact"


@dataclass
class Event:
    """A single structured pipeline event."""

    type: EventType
    stage: str = ""
    message: str = ""
    data: dict = field(default_factory=dict)
    t: float = 0.0  # set on emit; relative seconds when a clock is provided

    def to_json(self) -> str:
        d = asdict(self)
        d["type"] = self.type.value
        return json.dumps(d, default=str)


class EventBus:
    """Synchronous publish/subscribe with an in-memory log.

    Subscribers are simple callables. The bus also retains the full event history so
    a late subscriber (or a run-record writer) can replay it. ``clock`` lets the
    caller stamp monotonic relative times (kept out of the dataclass for resumability).
    """

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._subs: list[Callable[[Event], None]] = []
        self.history: list[Event] = []
        self._t0 = (clock or time.monotonic)()
        self._clock = clock or time.monotonic

    def subscribe(self, fn: Callable[[Event], None]) -> Callable[[], None]:
        self._subs.append(fn)
        return lambda: self._subs.remove(fn)

    def emit(self, event: Event) -> None:
        event.t = round(self._clock() - self._t0, 4)
        self.history.append(event)
        for fn in list(self._subs):
            fn(event)

    # convenience emitters -------------------------------------------------
    def log(self, message: str, stage: str = "", **data) -> None:
        self.emit(Event(EventType.LOG, stage, message, data))

    def metric(self, name: str, value, stage: str = "", **data) -> None:
        self.emit(Event(EventType.METRIC, stage, name, {"name": name, "value": value, **data}))

    def artifact(self, path: str, kind: str, stage: str = "") -> None:
        self.emit(Event(EventType.ARTIFACT, stage, path, {"path": path, "kind": kind}))

    # persistence ----------------------------------------------------------
    def write_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for e in self.history:
                f.write(e.to_json() + "\n")
        return path

    def metrics(self) -> dict:
        """Collected metric events as a flat {name: value} dict."""
        return {e.data["name"]: e.data["value"] for e in self.history
                if e.type == EventType.METRIC}
