"""ZeroGPU Space: the T3 tier — models too large to run under encryption or in a browser tab.

WHAT THIS IS FOR
    The browser tier serves int8 ONNX under 12 MB and the encrypted tier serves models small enough
    for CKKS to carry. Neither can host a model of real size. This Space is the third tier: it runs
    the full-precision industry models on a GPU and returns a result the demo labels REMOTE, so a
    visitor can see what the plaintext model actually does without the VPS needing a GPU.

WHY GRADIO AND WHY ZEROGPU
    ZeroGPU is the only free GPU on the Hub, and it is Gradio-only. It allocates an NVIDIA RTX Pro
    6000 Blackwell per @spaces.GPU call and releases it after. Two constraints follow and both are
    load-bearing:
      * Models must be moved to CUDA at module level, not inside the decorated function. A PyTorch
        CUDA emulation layer is active outside @spaces.GPU, so this works, and doing it lazily is
        markedly slower.
      * The quota is per *visitor*, not per owner: 5 minutes/day for a signed-in free user, 2 for an
        anonymous one. A public demo therefore costs its visitors' quota, not ours — but it also
        means a heavy default would exhaust a visitor in a couple of clicks. Keep calls short.

ELIGIBILITY
    Hosting a ZeroGPU Space needs an account "in good standing": verified email AND older than 30
    days. Daemons-Q was created 2026-08-02, so this Space becomes hostable around 2026-09-01. Until
    then the demo's capability probe simply reports the tier as unavailable and falls back — which is
    why the probe exists rather than the demo assuming the Space is there.

DEPLOY
    hf repo create Daemons-Q/qsh-inference --repo-type space --space_sdk gradio
    hf upload Daemons-Q/qsh-inference spaces/qsh-inference . --repo-type space
    # then set hardware to ZeroGPU in the Space settings
"""

from __future__ import annotations

import json
import os
import time

import gradio as gr
import numpy as np

try:  # only present on the Space; absent locally, which is fine
    import spaces
except ImportError:  # pragma: no cover - local dev
    class _Shim:
        @staticmethod
        def GPU(*args, **kwargs):
            def deco(fn):
                return fn
            return deco if not args or callable(args[0]) is False else args[0]
    spaces = _Shim()  # type: ignore[assignment]

HF_NAMESPACE = "Daemons-Q"

#: Registry mirror. Kept as a literal here because a Space is a separate deployment unit and must
#: not import the repository; the source of truth remains src/qsh/models/registry.py, and the
#: publish step is what keeps the two aligned.
MODELS = {
    "medical (BreastMNIST, ResNet-18)": {
        "repo": f"{HF_NAMESPACE}/qsh-medical-breastmnist-resnet18",
        "file": "onnx/model.onnx",
        "shape": [1, 1, 28, 28],
        "classes": ["benign/normal", "malignant"],
        "metric": "AUC 0.888 measured; 0.901 published baseline (MedMNIST v2)",
    },
    "finance (AMLworld, GFP + XGBoost)": {
        "repo": f"{HF_NAMESPACE}/qsh-finance-aml-gbm",
        "file": "onnx/model.onnx",
        "shape": [1, 26],
        "classes": ["not laundering", "laundering"],
        "metric": "minority-class F1 0.081 measured; 0.632 published (see the model card for why)",
    },
}

_sessions: dict[str, object] = {}


def _session(key: str):
    """Lazily fetch and open an ONNX session, pinned to the revision the registry recorded."""
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    if key in _sessions:
        return _sessions[key]
    spec = MODELS[key]
    path = hf_hub_download(repo_id=spec["repo"], filename=spec["file"],
                           token=os.environ.get("HF_TOKEN"))
    _sessions[key] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return _sessions[key]


@spaces.GPU(duration=30)
def infer(model_key: str, raw: str):
    """Run one input through the full-precision model and report what actually happened."""
    started = time.time()
    spec = MODELS[model_key]
    try:
        values = json.loads(raw) if raw.strip() else []
    except json.JSONDecodeError as exc:
        return {"error": f"input is not valid JSON: {exc}"}

    n = int(np.prod(spec["shape"]))
    arr = np.array(values, dtype=np.float32).ravel()
    if arr.size == 0:
        arr = np.zeros(n, dtype=np.float32)
    elif arr.size != n:
        return {"error": f"expected {n} values for shape {spec['shape']}, got {arr.size}"}

    sess = _session(model_key)
    out = sess.run(None, {sess.get_inputs()[0].name: arr.reshape(spec["shape"])})
    scores = np.asarray(out[-1]).ravel().astype(float)
    idx = int(np.argmax(scores))

    return {
        "prediction": spec["classes"][idx] if idx < len(spec["classes"]) else str(idx),
        "scores": [round(float(s), 6) for s in scores],
        "model": spec["repo"],
        "precision": "fp32",
        # The demo renders this as REMOTE. The latency includes network and queue time, so it is not
        # comparable with the in-browser MEASURED numbers and the label says so.
        "provenance": "REMOTE (hosted GPU Space)",
        "latency_ms": round((time.time() - started) * 1000, 1),
        "measured_metric": spec["metric"],
    }


with gr.Blocks(title="QSH — hosted inference (T3)") as demo:
    gr.Markdown(
        "# QSH hosted inference\n"
        "The **plaintext** industry models at full precision, for the tier that cannot run in a "
        "browser tab or under homomorphic encryption.\n\n"
        "Results returned here are labelled **REMOTE** in the demo: the reported latency includes "
        "network and queue time and is not comparable with the in-browser measurements. Metrics "
        "shown are what this project measured, with the published baseline beside them — including "
        "where we fall short of it."
    )
    with gr.Row():
        model = gr.Dropdown(list(MODELS), value=list(MODELS)[0], label="model")
        payload = gr.Textbox(label="input (JSON array; empty = zeros)", value="[]", lines=3)
    run = gr.Button("Run", variant="primary")
    result = gr.JSON(label="result")
    run.click(infer, inputs=[model, payload], outputs=result)

if __name__ == "__main__":
    demo.launch()
