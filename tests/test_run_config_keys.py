"""Every config key the browser run READS is a key the browser run EMITS.

WHY THIS EXISTS. `clientCompute`'s security facts choose the channel model with
``config?.usecase === "satellite_comms" ? FREE_SPACE : FIBER``, and ``buildConfig`` never put
``usecase`` in the config object. So the key was permanently ``undefined``, the ternary permanently
took the fibre branch, and the satellite page printed a decoy-state tagged fraction of 7.26 computed
over 50 km of FIBRE - against roughly 125 for its own free-space downlink - using two parameters that
page does not even render. Labelled ``measured``.

Nothing failed. There is no error to raise when an optional-chained read returns ``undefined``; it
just silently takes the other branch, and the number that comes out is plausible. The same shape had
already happened once before and is documented in ``makeDefaultParams``: ``sat_distance_km``,
``sat_eve`` and ``sat_pulses`` were read by two stages "since the page was written" while no default
or control supplied them, so Eve sat at zero and the eavesdrop check the page advertises could not
fire from any user action.

Twice is a pattern, and it is mechanisable: the reads and the writes are both greppable, so compare
them rather than re-noticing by eye. This test parses the TypeScript instead of importing it, which
``tests/test_api_config.py`` and ``tests/test_qcnn_usecases.py`` already set the precedent for -
adding a JS test runner for one set-difference is the worse trade.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMPUTE = REPO / "web" / "src" / "lib" / "clientCompute.ts"
RUNHOOK = REPO / "web" / "src" / "lib" / "usePipelineRun.ts"

#: Keys read from the run config that are deliberately NOT emitted by `buildConfig`, each with the
#: reason. A key belongs here only when its absence is a decision; anything else is the `usecase`
#: bug again. Keeping the reasons in the test means the next person meets them at the point of
#: failure rather than having to reconstruct why the list has holes.
INTENTIONALLY_ABSENT = {
    # The model-size selector was removed deliberately (see PipelineControls) and the read defaults to
    # "medium". The consequence is cosmetic but real: the benchmark panel renders a permanent "medium"
    # tag and a size column that can no longer change.
    "model_size",
    # No combiner control exists on any run surface, so `combiner: config.combiner ?? null` in the
    # signed payload always records null. The Overview page's claim that the combiner is selectable at
    # configuration time is about the FULL deployment, not the browser demo.
    "combiner",
}


def _read_keys() -> set[str]:
    """Keys the run reads off its config object, from `config.x` / `config?.x`, in the pipeline and in
    the satellite link builder it calls (`satLink.ts`, moved out of the pipeline in Round 77)."""
    text = COMPUTE.read_text() + "\n" + (COMPUTE.parent / "satLink.ts").read_text()
    # Strip line comments first: two of the matches in this file are prose about `config.federated`
    # and the `config.ts` module, not property reads, and a comment is not a dependency.
    code = "\n".join(re.sub(r"//.*$", "", ln) for ln in text.splitlines())
    return set(re.findall(r"config\??\.([a-z_][a-z0-9_]*)", code))


def _emitted_keys() -> set[str]:
    """Keys `buildConfig` can put on the object: literals, conditionals, and the params spread."""
    text = RUNHOOK.read_text()

    body = re.search(r"const buildConfig = \(\) => \{(.*?)\n  \};", text, re.S)
    assert body, "buildConfig not found - did it move or change shape?"
    b = body.group(1)

    keys: set[str] = set()
    # `const c: Record<string, unknown> = { usecase, kem, ... , ...nums };`
    lit = re.search(r"const c: Record<string, unknown> = \{([^}]*)\}", b)
    assert lit, "the config literal in buildConfig not found"
    for part in lit.group(1).split(","):
        part = part.strip()
        if not part or part.startswith("..."):
            continue
        keys.add(part.split(":")[0].strip())
    # `if (...) c.foo = ...`
    keys |= set(re.findall(r"\bc\.([a-z_][a-z0-9_]*)\s*=", b))

    # `...nums` is `params` minus the three destructured out of it, and `params` is makeDefaultParams().
    defaults = re.search(r"export const makeDefaultParams = \(\) => \(\{(.*?)\n\}\);", text, re.S)
    assert defaults, "makeDefaultParams not found"
    d = "\n".join(re.sub(r"//.*$", "", ln) for ln in defaults.group(1).splitlines())
    keys |= set(re.findall(r"^\s{2}([a-z_][a-z0-9_]*):", d, re.M))
    return keys


def test_every_read_key_is_emitted_or_explicitly_excepted() -> None:
    read, emitted = _read_keys(), _emitted_keys()
    missing = read - emitted - INTENTIONALLY_ABSENT
    assert not missing, (
        "these keys are READ from the run config and never emitted by buildConfig, so each is "
        f"permanently undefined at run time: {sorted(missing)}.\n"
        "An optional-chained read of a missing key does not raise - it takes the other branch and "
        "produces a plausible number. That is how the satellite page came to report a fibre-channel "
        "decoy figure. Either emit the key or add it to INTENTIONALLY_ABSENT with the reason."
    )


def test_the_satellite_channel_reads_the_satellite_parameters() -> None:
    """The channel MODEL and the parameters fed to it must both follow the use case.

    Round 29 emitted `usecase`, so the model switched to free-space on the satellite page - and left
    the parameter reads on `qkd_distance_km` / `qkd_eve`, the FIBRE page's controls. Those keys are
    never absent, so the satellite page evaluated the free-space model at 50 km of fibre distance. The
    printed tagged fraction moved from 7.26 to 2.89 instead of to 124.77, and the three controls that
    page renders still could not move it. A half-corrected number is harder to notice than an
    uncorrected one, because it moved.
    """
    read, emitted = _read_keys(), _emitted_keys()
    for key in ("sat_distance_km", "sat_eve"):
        assert key in read, (
            f"{key} is no longer read by clientCompute. If the satellite channel stopped using its "
            "own parameters, that is the regression this test exists for."
        )
        assert key in emitted, f"{key} is read but not emitted by buildConfig"


def test_usecase_is_emitted_because_the_channel_model_depends_on_it() -> None:
    """The specific regression, pinned separately so a refactor cannot quietly drop it."""
    assert "usecase" in _emitted_keys(), (
        "buildConfig must emit `usecase`: clientCompute selects the free-space channel model with "
        "`config?.usecase === \"satellite_comms\"`, and without the key every use case is scored on "
        "the fibre channel."
    )
    assert "usecase" in _read_keys(), "the read moved; update this test to follow it"


def test_the_exception_list_does_not_rot() -> None:
    """An excepted key that is no longer read should leave the list, or it hides the next real gap."""
    stale = INTENTIONALLY_ABSENT - _read_keys()
    assert not stale, (
        f"INTENTIONALLY_ABSENT lists keys nothing reads any more: {sorted(stale)}. Remove them - a "
        "stale exception is an exception that can mask a genuine missing key later."
    )
