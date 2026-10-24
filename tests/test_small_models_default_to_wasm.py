"""The small in-browser models run on WASM unless the visitor selects WebGPU, and every size a visitor
chooses by is computed from recorded bytes.

WHY THIS EXISTS

Round 59 timed the in-browser agent against the inference library it upgraded and found a defect older
than both versions. The agent took WebGPU whenever a GPU adapter existed. On the machine measured (an
NVIDIA RTX PRO 6000 adapter, Chrome 152 on Linux) that path ran distilGPT-2 at 750 to 1140 ms per token
against about 9 ms on WASM, no first run finished inside the panel's 120 s limit, and in the inference
stage one ResNet-18 session.run took 18216 ms on onnx-webgpu. The model's label said "tiny, ~80 MB";
WASM fetches 236.7 MB and WebGPU 327.8 MB, and no file the library loads is 80 MB.

Round 60 made WASM the default for both tiers and WebGPU a selection. This gate holds the properties
that change rests on, because each can be undone by a one-line edit that looks harmless:

1. WASM is the default in the config (Python and the exported JSON), and neither tier derives its
   backend from the adapter's existence alone.
2. Every Transformers.js model is pinned to a commit and records the dtype and the bytes of the weights
   file for each backend, and the loader passes `dtype` and `revision` to the pipeline - so the bytes
   describe the file that is fetched, whatever the library's per-device default becomes.
3. The size a visitor sees is computed from those bytes. A label carries no size.
4. The agent's time limit measures a STALL, not the total: a first run's download that is still
   arriving is progress. A total limit reported working runs as timed out - 313 MB took 277 s on one
   connection measured.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from qsh.config.defaults import WebConfig

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web" / "src"


def _ts(rel: str) -> str:
    """TypeScript source with comments stripped: the comments recording the change quote the old code."""
    src = (WEB / rel).read_text()
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    src = re.sub(r"\{/\*.*?\*/\}", " ", src, flags=re.S)
    return re.sub(r"(?<![:\"'`])//[^\n]*", " ", src)


def _function_body(src: str, signature: str) -> str:
    """The body of the first function whose declaration contains `signature`, braces balanced."""
    start = src.find(signature)
    assert start >= 0, f"{signature!r} not found - did the function move or get renamed?"
    open_at = src.index("{", src.index(")", start))
    depth, i = 0, open_at
    while i < len(src):
        depth += {"{": 1, "}": -1}.get(src[i], 0)
        i += 1
        if depth == 0:
            break
    return src[open_at:i]


def test_the_default_backend_is_wasm_in_both_configs() -> None:
    py = WebConfig()
    exported = json.loads((WEB / "lib" / "config.defaults.json").read_text())["web"]
    for name in ("tfjs_default_device", "onnx_default_backend"):
        assert getattr(py, name) == "wasm", f"WebConfig.{name} is {getattr(py, name)!r}, not 'wasm'"
        assert exported[name] == "wasm", f"config.defaults.json web.{name} is {exported[name]!r}"


def test_neither_tier_derives_its_backend_from_the_adapter_alone() -> None:
    agent = _function_body(_ts("lib/clientAI.ts"), "async function runTransformersJs(")
    device = re.search(r"const device\b[^;]*;", agent)
    assert device, "runTransformersJs no longer computes `device` - re-derive this check"
    assert 'requested === "webgpu"' in device.group(0), (
        f"the agent's device is decided without the visitor's request: {device.group(0)!r}. An adapter "
        "existing is not a reason to take WebGPU; on the machine measured it was the slow path.")

    onnx = _ts("lib/onnxInference.ts")
    gpu = re.search(r"const gpuAvailable\s*=[^;]*;", onnx)
    assert gpu, "onnxInference no longer computes `gpuAvailable` - re-derive this check"
    assert 'backend === "webgpu"' in gpu.group(0), (
        f"the ONNX tier takes WebGPU without a request: {gpu.group(0)!r}")
    assert re.search(r"backend: OnnxBackend = DEFAULT_ONNX_BACKEND", onnx), (
        "loadModel's backend no longer defaults to the configured one")
    # Every caller that runs a published model must forward a backend, or a WebGPU selection would
    # silently run on the default.
    assert "loadModel(usecase, key, opts.backend ?? DEFAULT_ONNX_BACKEND)" in onnx
    assert "loadModel(usecase, target.spec.key, backend)" in onnx

    compute = _ts("lib/clientCompute.ts")
    assert "backend: onnxBackendChoice" in compute, "runPublishedModel is called without the run's backend"
    assert "evaluatePublishedModel(usecase, 0.05, key, onnxBackendChoice)" in compute, (
        "the whole-sample verification runs without the run's backend")
    assert re.search(r'config\.onnx_backend === "webgpu" \? "webgpu" : DEFAULT_ONNX_BACKEND', compute), (
        "the run's backend is not derived from `onnx_backend` with the configured default")


_SIZE = re.compile(r"\d\s*(?:k|M|G)i?B\b|~\s*\d|\btiny\b", re.I)
_SHA = re.compile(r"^[0-9a-f]{40}$")


def test_every_tfjs_model_is_pinned_and_records_both_backends() -> None:
    models = WebConfig().tfjs_models
    assert models, "no Transformers.js models configured"
    for m in models:
        mid = m.get("id")
        assert _SHA.match(str(m.get("revision", ""))), f"{mid}: not pinned to a 40-char commit"
        for dev in ("wasm", "webgpu"):
            assert m.get("dtype", {}).get(dev), f"{mid}: no dtype recorded for {dev}"
            b = m.get("bytes", {}).get(dev)
            assert isinstance(b, int) and b > 0, f"{mid}: no byte count recorded for {dev}"
        assert not _SIZE.search(m.get("label", "")), (
            f"{mid}: the label {m['label']!r} carries a size. Sizes are computed from `bytes`; a typed "
            "one is how 'tiny, ~80 MB' shipped beside a 236.7 MB download.")


def test_the_loader_fetches_the_file_the_bytes_describe() -> None:
    agent = _function_body(_ts("lib/clientAI.ts"), "async function runTransformersJs(")
    call = re.search(r"lib\.pipeline\(\s*\"text-generation\"[^;]*;", agent, re.S)
    assert call, "the Transformers.js pipeline call is gone - re-derive this check"
    for arg in ("dtype: spec?.dtype[device]", "revision: spec?.revision"):
        assert arg in call.group(0), (
            f"the pipeline call no longer passes `{arg}`, so the recorded bytes may describe a different "
            "file from the one fetched")


def test_the_picker_computes_the_size_it_shows() -> None:
    label = _function_body(_ts("lib/clientAI.ts"), "export function tfjsModelLabel(")
    assert "m.bytes[device]" in label, "tfjsModelLabel no longer reads the recorded bytes"
    panel = _ts("components/AgenticAIPanel.tsx")
    assert "tfjsModelLabel(m, device)" in panel, (
        "the agent panel's Transformers.js options no longer use the computed label")


def test_the_timeout_measures_a_stall_not_the_download() -> None:
    src = _ts("lib/clientAI.ts")
    agent = _function_body(src, "export async function runAgent(")
    assert re.search(r"performance\.now\(\) - lastProgress < AGENT_TIMEOUT_MS", agent), (
        "runAgent's timeout no longer measures time since the last progress event")
    assert not re.search(r"setTimeout\([^;]*AGENT_TIMEOUT_MS\)", agent), (
        "runAgent is back to a limit on TOTAL time, which a first run's download can exhaust on its own")
    loader = _function_body(src, "async function runTransformersJs(")
    assert "progress_callback" in loader, (
        "the Transformers.js pipeline reports no progress, so a download would count as a stall")
