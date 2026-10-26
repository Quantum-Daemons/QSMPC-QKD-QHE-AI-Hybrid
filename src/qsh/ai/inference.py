"""Inference + ONNX export (for the in-browser onnxruntime-web demo).

The public demo runs *trained-model inference only* on the client. We export the
PyTorch models to ONNX so the SPA can run them with onnxruntime-web (WASM/WebGPU)
with zero server compute.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .models import device


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Return class predictions for a batch of inputs."""
    dev = device()
    model.eval().to(dev)
    xb = torch.tensor(np.asarray(X, dtype=np.float32)).to(dev)
    return model(xb).argmax(1).cpu().numpy()


def export_onnx(model: nn.Module, example_input: np.ndarray, path: str | Path,
                *, input_name: str = "input", output_name: str = "logits") -> Path:
    """Export ``model`` to ONNX at ``path`` for onnxruntime-web.

    TRACED ON AT LEAST TWO ROWS (Round 71). torch>=2.9 exports through torch.export, which
    specialises any dimension whose example size is 0 or 1. Given one row, the Telemanom LSTM came out
    with input ``[batch, 250, 6]`` and output ``[1, 6]``: the graph still computed every row
    correctly, but onnxruntime warned "Expected shape from model of {1,6} does not match actual shape
    of {128,6}" on every satellite run, because the browser scores 128 windows at once. The ResNets
    exported through the same call kept ``[batch, 2]``, which is why one row looked safe. The
    example only drives tracing, so a one-row example is repeated, and a graph whose output batch
    is still fixed is refused rather than published.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    example = np.asarray(example_input, dtype=np.float32)
    if example.shape[0] < 2:
        example = np.concatenate([example, example], axis=0)
    dummy = torch.tensor(example)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=[input_name], output_names=[output_name],
        dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
        opset_version=17,
    )
    from ..models.export import fixed_batch_outputs

    fixed = fixed_batch_outputs(path)
    if fixed:
        raise RuntimeError(f"{path}: the input batch axis is symbolic but output(s) {fixed} declare a "
                           "fixed batch, so onnxruntime warns on every batched call. Not publishing it.")
    return path


def verify_onnx(path: str | Path, model: nn.Module, example_input: np.ndarray,
                *, atol: float = 1e-4) -> bool:
    """Check ONNX outputs match the PyTorch model (parity for the web demo)."""
    import onnxruntime as ort

    model.eval().cpu()
    with torch.no_grad():
        ref = model(torch.tensor(np.asarray(example_input, dtype=np.float32))).numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: np.asarray(example_input, np.float32)})[0]
    return bool(np.allclose(ref, out, atol=atol))
