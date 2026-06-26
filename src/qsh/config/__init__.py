"""Configuration loading + access.

``get_config()`` returns a process-wide :class:`QSHConfig` built from the dataclass
defaults, optionally overlaid by ``config.yaml`` (repo root or ``$QSH_CONFIG``).
The WebUI sends per-run overrides in the orchestration ``config`` dict, which take
precedence over these defaults at run time.
"""

from __future__ import annotations

import os
from dataclasses import fields, is_dataclass
from pathlib import Path

from .defaults import (  # noqa: F401  (re-exported for convenience)
    BenchmarkConfig,
    CryptoConfig,
    DataConfig,
    FederatedConfig,
    FHEConfig,
    IDSConfig,
    OrchestrationConfig,
    QHEConfig,
    QKDConfig,
    QSHConfig,
    QSMPCConfig,
    QuantumAdvantageConfig,
    SecurityScoringConfig,
    TrainConfig,
)

_CONFIG: QSHConfig | None = None


def _overlay(obj, data: dict) -> None:
    """Recursively overlay a plain dict onto a dataclass instance (in place)."""
    if not is_dataclass(obj) or not isinstance(data, dict):
        return
    names = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key not in names:
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _overlay(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(obj, key, tuple(value))
        else:
            setattr(obj, key, value)


def load_config(path: str | Path | None = None) -> QSHConfig:
    """Build a config from defaults, overlaying ``config.yaml`` if present."""
    cfg = QSHConfig()
    candidate = path or os.environ.get("QSH_CONFIG")
    if candidate is None:
        default_path = Path(__file__).resolve().parents[3] / "config.yaml"
        candidate = default_path if default_path.exists() else None
    if candidate and Path(candidate).exists():
        try:
            import yaml

            with open(candidate) as f:
                data = yaml.safe_load(f) or {}
            _overlay(cfg, data)
        except Exception:
            pass  # malformed config => fall back to defaults
    return cfg


def get_config(reload: bool = False) -> QSHConfig:
    """Return the process-wide config (cached)."""
    global _CONFIG
    if _CONFIG is None or reload:
        _CONFIG = load_config()
    return _CONFIG


def frontend_config() -> dict:
    """The config subset the SPA needs: the benchmark cost model + tier architectures + workload
    multipliers + the medical-CNN shape + security/advantage weights + demo data sizes.

    Single source of truth for the frontend: served at ``/api/config`` (full mode) and baked into
    ``web/src/lib/config.defaults.json`` by ``scripts/export_config.py`` (demo mode). The backend reads
    the same dataclasses directly, so the demo's cost model can never drift from the backend.
    """
    from dataclasses import asdict

    c = get_config()
    b = c.benchmark
    return {
        "benchmark": {
            "tier_architectures": b.tier_architectures,
            "generic_architecture": b.generic_architecture,
            "workload_multipliers": b.workload_multipliers,
            "cost_per_op": b.cost_per_op,
            "medical_cnn": b.medical_cnn,
            "demo_feature_dim": b.demo_feature_dim,
            "hybrid_samples": b.hybrid_samples,
            "quantum_qubits": b.quantum_qubits,
            "quantum_samples": b.quantum_samples,
        },
        "security_scoring": asdict(c.security_scoring),
        "quantum_advantage": asdict(c.quantum_advantage),
        "data": {
            "aml_shared_accounts": c.data.aml_shared_accounts,
            "aml_bank_a_accounts": c.data.aml_bank_a_accounts,
            "aml_bank_b_accounts": c.data.aml_bank_b_accounts,
            "genomics_biobanks": c.data.genomics_biobanks,
            "genomics_allele_min": c.data.genomics_allele_min,
            "genomics_allele_max": c.data.genomics_allele_max,
        },
        "qkd": {"pulses": c.qkd.pulses, "distance_km": c.qkd.distance_km},
        "web": asdict(c.web),
    }
