"""Pause is two events, and both transports must mean the same thing by the same name.

WHY THIS EXISTS

The browser's run gate emits TWO events (`web/src/lib/clientCompute.ts`):

    pipeline_pause             the REQUEST - the button was pressed
    pipeline_pause_effective   the run is actually sitting at the gate

and `usePipelineRun.ts` maps them to `"pausing"` and `"paused"`. Only `"paused"` enables **Resume**
and **Step**.

The Python pipeline emitted `PIPELINE_PAUSE` from **inside the gate**, at the moment it actually
blocked - semantically the browser's `pipeline_pause_effective`, under the browser's name for the
request. `request_pause()` emitted nothing at all. So against the backend transport the UI reached
`"pausing"` and stopped there: `paused` stayed false forever, Resume and Step were permanently
disabled, and Start-stepwise was unusable from its first stage. Only Stop and Restart still worked.

One scheme, two implementations, the same name meaning different things - the class this repository
has now shipped in the hybrid-KEM combiner, the ITS combiner, the federated mask and the QCNN
metrics. This is its fifth appearance and its first in a state machine.

WHY THE CHECKLIST DID NOT CATCH IT

Rows `OR7` and `OR8` assert exactly this behaviour and were verified **on the deployed page**, which
is client-only - the one transport where both events exist. A check that does not say which side of
a boundary it looked at reports on one side and reads as reporting on both. That is the whole subject
of this round, and this test is the instance in code.

WHY IT ASSERTS THE ORDER AND NOT JUST THE PRESENCE

A pipeline that emits both events at the same moment satisfies "both exist" while telling the UI the
run stopped before it had. The request must precede the effect, and the effect must arrive only once
the worker is genuinely blocked.
"""

from __future__ import annotations

import re
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qsh.orchestration.events import EventBus, EventType  # noqa: E402
from qsh.orchestration.pipeline import Pipeline, PipelineState  # noqa: E402
from qsh.orchestration.stages import Context  # noqa: E402

TS = REPO / "web" / "src" / "lib" / "usePipelineRun.ts"


class _SlowStage:
    """A stage slow enough that a pause lands between stages rather than after the run."""

    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, ctx, bus, reg) -> None:  # noqa: D102 - test double
        time.sleep(0.05)


def _pipeline(n: int = 6):
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe(lambda e: seen.append(e.type.value if hasattr(e.type, "value") else str(e.type)))
    p = Pipeline(name="pause-test", stages=[_SlowStage(f"s{i}") for i in range(n)], bus=bus)
    return p, seen


def test_the_vocabulary_exists_on_the_python_side() -> None:
    """Both names must be defined, or the assertions below pass for the wrong reason."""
    names = {e.value for e in EventType}
    assert "pipeline_pause" in names
    assert "pipeline_pause_effective" in names, (
        "the Python pipeline has no `pipeline_pause_effective`. The browser needs it to reach "
        "\"paused\", and \"paused\" is what enables Resume and Step - so against this transport a "
        "paused run can never be resumed or stepped."
    )


def test_the_request_precedes_the_effect_and_both_are_emitted() -> None:
    p, seen = _pipeline()
    t = threading.Thread(target=lambda: p.run(Context()), daemon=True)
    t.start()
    time.sleep(0.08)
    p.request_pause()

    deadline = time.time() + 5.0
    while time.time() < deadline and "pipeline_pause_effective" not in seen:
        time.sleep(0.02)

    assert "pipeline_pause" in seen, "request_pause() emitted nothing; the UI cannot show 'pausing'"
    assert "pipeline_pause_effective" in seen, (
        "the run never reported reaching the gate, so the UI stays on 'pausing' forever and Resume "
        "and Step stay disabled - the defect this test exists for"
    )
    assert seen.index("pipeline_pause") < seen.index("pipeline_pause_effective"), (
        "the effect was announced before the request. A run that says it stopped before it had is "
        "the badge-lies-about-the-state defect the browser gate was written to remove."
    )
    assert p.state is PipelineState.PAUSED
    p.request_stop()
    t.join(timeout=5)


def test_resume_after_an_effective_pause_actually_continues() -> None:
    """The point of the two events is that Resume becomes reachable; prove the run goes on."""
    p, seen = _pipeline()
    t = threading.Thread(target=lambda: p.run(Context()), daemon=True)
    t.start()
    time.sleep(0.08)
    p.request_pause()
    deadline = time.time() + 5.0
    while time.time() < deadline and "pipeline_pause_effective" not in seen:
        time.sleep(0.02)
    before = seen.count("stage_end")
    p.request_resume()
    t.join(timeout=10)
    assert seen.count("stage_end") > before, "the run did not continue after resume"
    assert "pipeline_end" in seen


def test_the_browser_still_reads_the_same_two_names() -> None:
    """If the TypeScript side is renamed, this fix is half-applied again.

    The Python and browser pipelines are one scheme in two languages, and every previous instance of
    this class was a change that reached one of them. Reading the other side here is cheap.
    """
    src = TS.read_text()
    assert re.search(r'e\.type === "pipeline_pause"', src), "the browser no longer reads the request"
    assert re.search(r'e\.type === "pipeline_pause_effective"', src), (
        "the browser no longer reads `pipeline_pause_effective`; the two transports have diverged "
        "again, in the direction this test cannot see from Python alone"
    )
