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
    """Export ``model`` to ONNX at ``path`` for onnxruntime-web."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    dummy = torch.tensor(np.asarray(example_input, dtype=np.float32))
    torch.onnx.export(
        model, dummy, str(path),
        input_names=[input_name], output_names=[output_name],
        dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
        opset_version=17,
    )
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
