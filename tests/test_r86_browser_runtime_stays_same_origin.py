"""A run of the Agentic AI panel reaches this site and Hugging Face, and nothing else.

WHY THIS EXISTS (Round 86). A real run on the wire fetched ONNX Runtime's wasm from cdn.jsdelivr.net, which no page
named, because Transformers.js points ONNX Runtime there unless told otherwise. WebLLM fetched its model libraries from
a repository with no licence file, and its weights from `resolve/main`. The panel also offered a model whose licence
asks for notices nothing showed, and labelled one download 376 MB when it is 206 MB. The wire is measured by hand
(checklist CS12), since a run downloads hundreds of megabytes; these hold what the source can get wrong.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = json.loads((REPO / "web/src/lib/config.defaults.json").read_text(encoding="utf-8"))["web"]
LIB_DIR = REPO / "web/public" / WEB["webllm_lib_dir"]
CLIENT_AI = (REPO / "web/src/lib/clientAI.ts").read_text(encoding="utf-8")


def _manifest() -> dict:
    return json.loads((LIB_DIR / "MANIFEST.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("model", WEB["webllm_models"], ids=lambda m: m["id"])
def test_each_webllm_model_is_pinned_and_its_library_is_the_one_served(model: dict) -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", model["revision"]), "weights must not move under a page's label"
    assert model["bytes"] > 0, "a picker label is computed from this"
    data = (LIB_DIR / model["lib"]).read_bytes()
    assert model["lib_sri"] == "sha256-" + base64.b64encode(hashlib.sha256(data).digest()).decode()
    entry = _manifest()["libs"][model["lib"]]
    assert entry["sha256"] == hashlib.sha256(data).hexdigest() and entry["bytes"] == len(data)
    assert entry["model_id"] == model["id"] and entry["weights_revision"] == model["revision"]


def test_the_manifest_names_the_runtime_the_page_loads_and_the_commit_the_script_builds() -> None:
    package = json.loads((REPO / "web/package.json").read_text(encoding="utf-8"))
    version = package["dependencies"]["@mlc-ai/web-llm"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), "exact: a library is built against one runtime"
    assert _manifest()["runtime"] == f"@mlc-ai/web-llm@{version}"
    assert WEB["webllm_lib_dir"].endswith(version)
    script = (REPO / "scripts/build_webllm_libs.sh").read_text(encoding="utf-8")
    assert re.search(r"^MLC_LLM_COMMIT=(\w+)$", script, re.M).group(1) == _manifest()["built_with"]["mlc-llm"]
    assert re.search(r"^EMSDK_VERSION=(\S+)$", script, re.M).group(1) == _manifest()["built_with"]["emscripten"]


def test_nothing_is_served_that_no_model_uses() -> None:
    served = {p.name for p in LIB_DIR.glob("*.wasm")}
    assert served == {m["lib"] for m in WEB["webllm_models"]} == set(_manifest()["libs"])


def test_the_notices_ship_with_the_libraries() -> None:
    text = (REPO / "web/public/licenses/webllm-libs.txt").read_text(encoding="utf-8")
    for needed in ("MLC LLM", "Apache TVM", "Apache License", "Emscripten", "DLPack"):
        assert needed in text, needed


@pytest.mark.parametrize("model", WEB["webllm_models"] + WEB["tfjs_models"], ids=lambda m: m["id"])
def test_every_model_on_the_panel_has_its_licence_recorded(model: dict) -> None:
    licences = (REPO / "DATA_LICENSES.md").read_text(encoding="utf-8")
    section = licences[licences.index("## 5b."):licences.index("## 6. OSS")]
    repo_id = model["id"] if "/" in model["id"] else f"mlc-ai/{model['id']}"
    family = re.sub(r"-q4f(16|32)_1-MLC$", "", repo_id)
    assert family in section, f"{repo_id} is offered and not in DATA_LICENSES.md section 5b"


def test_no_model_carries_a_licence_the_page_does_not_honour() -> None:
    for model in WEB["webllm_models"] + WEB["tfjs_models"]:
        assert "llama" not in model["id"].lower(), "the Llama licence asks for notices: DATA_LICENSES.md section 5b"


def test_transformers_js_is_pointed_at_this_sites_wasm_before_any_model_loads() -> None:
    for build in ("asyncify.mjs", "asyncify.wasm", "threaded.mjs", "threaded.wasm"):
        assert f"ort-wasm-simd-threaded.{build}?url" in CLIENT_AI or f"ort-wasm-simd-{build}?url" in CLIENT_AI, build
    assert CLIENT_AI.index("useSameOriginOrtWasm(lib);") < CLIENT_AI.index("lib.pipeline("), "it reads wasmPaths on load"


def test_webllm_is_given_its_record_and_never_its_prebuilt_list() -> None:
    assert "appConfig: { model_list: [record] }" in CLIENT_AI
    assert "resolve/${spec.revision}/" in CLIENT_AI, "the library would read resolve/main"
    assert "prebuiltAppConfig" not in CLIENT_AI and "CreateMLCEngine" not in CLIENT_AI
    assert "integrity: { model_lib: spec.lib_sri }" in CLIENT_AI


def test_no_source_names_the_library_host_without_a_licence() -> None:
    for path in (REPO / "web/src").rglob("*.ts*"):
        assert "binary-mlc-llm-libs" not in path.read_text(encoding="utf-8"), path


def test_cancel_reaches_both_engines() -> None:
    assert "stopping_criteria: stop" in CLIENT_AI and "lib.env.fetch = (url" in CLIENT_AI, "Transformers.js"
    assert "engine.unload?.()" in CLIENT_AI and "engine.interruptGenerate?.()" in CLIENT_AI, "WebLLM"
    assert CLIENT_AI.count("signal?.throwIfAborted();") == 2, "a cancel during the import has no listener to reach"
    panel = (REPO / "web/src/components/AgenticAIPanel.tsx").read_text(encoding="utf-8")
    assert "cancelRun.current?.abort()" in panel and "signal: ctl.signal" in panel


def test_a_lost_request_resumes_and_anything_else_is_reported() -> None:
    """Chrome's ERR_NETWORK_CHANGED ended a 2 GB download at the first failed shard, with the rest cached."""
    assert WEB["webllm_load_attempts"] > 1
    body = CLIENT_AI[CLIENT_AI.index("async function loadWebLlmResuming("):CLIENT_AI.index("async function runWebLLM(")]
    assert "signal?.aborted" in body, "a cancel is not a lost request"
    assert "failed to fetch|network error" in body, "only a lost request is retried"


def test_a_run_releases_what_it_loaded() -> None:
    """Each run opened a session or an engine and none was released, so memory grew with every run."""
    assert "await gen.dispose?.();" in CLIENT_AI, "Transformers.js"
    assert "loadedWebLlm.engine.unload()" in CLIENT_AI, "WebLLM: one engine at a time"


def test_a_software_adapter_is_named_by_the_spec_first() -> None:
    onnx = (REPO / "web/src/lib/onnxInference.ts").read_text(encoding="utf-8")
    assert "info.isFallbackAdapter === true" in onnx
    assert 'createContext({ deviceType' not in onnx, "MLDeviceType left the WebNN spec on 2026-10-21"
