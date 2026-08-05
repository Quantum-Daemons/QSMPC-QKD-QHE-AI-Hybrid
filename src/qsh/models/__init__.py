"""Trained-model lifecycle: registry -> train -> export -> distil -> publish.

Before Round 17 this project trained every model from scratch inside each pipeline run and threw it
away when the process exited: there was no checkpoint, no ONNX artefact, no fine-tuning and no model
registry anywhere in the tree. This package is the durable half of that gap.

  :mod:`qsh.models.registry`  what each use case's models are, what data they came from, what licence
                              that data carries, and which published baseline they are measured against.
  :mod:`qsh.models.train`     a real training loop (AMP, scheduling, early stopping, seeded, checkpointed)
                              that records a manifest of exactly what produced each artefact.
  :mod:`qsh.models.export`    torch/sklearn/xgboost -> ONNX fp32 -> int8, with a numeric parity gate.
  :mod:`qsh.models.distil`    teacher (plaintext, big) -> student (encrypted, small), measuring the
                              agreement and metric cost that the encryption actually buys.
"""

from __future__ import annotations

from .registry import (  # noqa: F401
    HF_NAMESPACE,
    ArtifactSpec,
    DatasetSpec,
    MetricSpec,
    ModelSpec,
    achieved,
    for_usecase,
    frontend_models,
    get,
    specs,
)

__all__ = [
    "HF_NAMESPACE",
    "ArtifactSpec",
    "DatasetSpec",
    "MetricSpec",
    "ModelSpec",
    "achieved",
    "for_usecase",
    "frontend_models",
    "get",
    "specs",
]
