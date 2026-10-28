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
        # The QSMPC field parameters. Absent until Round 20, so the browser had no idea what the
        # backend's party count or threshold were and its own secure-sum used an unrelated modulus
        # while the diagram above it said "d-qudit GHZ". `prime: null` means "derive from n as the
        # construction requires" (d prime, n <= d <= 2n), which is the honest default.
        "qsmpc": {"prime": c.qsmpc.prime, "n_parties": c.qsmpc.n_parties,
                  "threshold": c.qsmpc.threshold},
        # The federated hyperparameters. Absent until Round 24, which meant the browser COULD NOT
        # read them even in principle: it ran 6 rounds at lr 0.15 while this config said 8 at 0.05,
        # and nothing could reconcile the two because the dataclass never reached the frontend. The
        # divergence is defensible - a federated round has to finish inside a page interaction - but
        # it has to be a deliberate override of a value the browser can see, not two unrelated
        # numbers. `browser_rounds` states the override explicitly rather than hiding it in a
        # literal.
        "federated": {
            "rounds": c.federated.rounds,
            "local_epochs": c.federated.local_epochs,
            "lr": c.federated.lr,
            "hospitals": c.federated.hospitals,
            "test_split": c.federated.test_split,
            "secure_aggregation": c.federated.secure_aggregation,
            "browser_rounds": c.federated.browser_rounds,
            "browser_lr": c.federated.browser_lr,
            "browser_override_reason":
                "the browser runs a lighter schedule so a federated round completes inside a page "
                "interaction; the server-side values above are what the full deployment trains with",
        },
        # The CKKS parameter set. Absent until Round 38, which meant the ring degree the browser ran
        # was a literal in `clientFHE.ts`, the ring degree the server ran was a default argument in
        # `fhe/backends.py`, and the ring degree the report PRINTED was a literal inside
        # `report/build_data.py` - three copies, and `FHEConfig.poly_modulus_degree` read by none of
        # them. They agreed only because no one had changed one. A ring change would have left the
        # published PDF asserting a parameter set the code no longer used.
        #
        # Only the parameters that genuinely govern BOTH sides are exported. `backend`, `use_gpu`,
        # `max_level` and the OpenFHE knobs describe the server's context and would be false of the
        # browser, which runs node-seal on WASM CPU with no GPU path - exporting them would replace
        # a duplicated number with a wrong one.
        "fhe": {
            "poly_modulus_degree": c.fhe.poly_modulus_degree,
            "coeff_mod_bit_sizes": list(c.fhe.coeff_mod_bit_sizes),
            "global_scale_bits": c.fhe.global_scale_bits,
            # Derived here so no consumer divides by two on its own: CKKS packs conjugate pairs, so
            # a degree-N ring carries N/2 usable slots. Three prose sites had this multiplied out.
            "slots": c.fhe.poly_modulus_degree // 2,
        },
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
            "aml_transactions": c.data.aml_transactions,
            "fraud_rate": c.data.fraud_rate,
            "aml_shared_accounts": c.data.aml_shared_accounts,
            "aml_bank_a_accounts": c.data.aml_bank_a_accounts,
            "aml_bank_b_accounts": c.data.aml_bank_b_accounts,
            "genomics_n": c.data.genomics_n,
            "genomics_n_snps": c.data.genomics_n_snps,
            "genomics_biobanks": c.data.genomics_biobanks,
            "genomics_allele_min": c.data.genomics_allele_min,
            "genomics_allele_max": c.data.genomics_allele_max,
        },
        "qkd": asdict(c.qkd),          # full QKD + OptiQKD knobs (editable in the WebUI Parameters panel)
        # The link profiles (Round 77): every downlink and inter-satellite-link template, values with
        # their quotes, so the page can offer them and say where each number comes from. A file added
        # under src/qsh/config/profiles/ reaches the browser through here with no code change.
        "link_profiles": _link_profiles(),
        # Published measurements the models are checked against and the page draws (Round 78): one
        # JSON file each under src/qsh/config/validation/, read by the tests and the browser alike.
        "validation": _validation_sets(),
        "qrng": asdict(c.qrng),        # QRNG bits / source / post-processing / health alpha
        "ids": asdict(c.ids),          # QML-IDS sample sizes (incl. the in-browser demo IDS)
        "crypto": asdict(c.crypto),    # crypto-agility defaults incl. hybrid_combiner_mode (PQC+QKD combiner)
        "web": asdict(c.web),
        # Per-use-case model registry: repo ids, pinned revisions, artefact paths, dataset licences,
        # published baselines and the metrics actually measured. The SPA renders model names, numbers
        # and download URLs from this, so none of them can be hard-coded in TypeScript and none can
        # drift from what the training runs produced.
        "models": _model_registry(),
        # Provenance of the four samples bundled INTO the demo, which is not the same thing as the
        # corpora the published models were trained on (finance trains on AMLworld, the browser ships
        # PaySim). The run log prints these verbatim, so they live here rather than in TypeScript:
        # the same claim previously existed in three places and they drifted.
        "demo_samples": _demo_samples(),
    }


def _validation_sets() -> dict:
    import json

    folder = Path(__file__).resolve().parent / "validation"
    return {f.stem: json.loads(f.read_text()) for f in sorted(folder.glob("*.json"))}


def _link_profiles() -> dict:
    from .profiles import frontend_profiles

    return frontend_profiles()


def _demo_samples() -> dict:
    """Bundled-sample provenance from the registry, or an empty mapping in the slim demo image."""
    try:
        from ..models.registry import BUNDLED_DEMO_SAMPLES
    except Exception:  # pragma: no cover - slim image only
        return {}
    return {
        usecase: {"provenance": text, "licence": ds.licence, "url": ds.url,
                  "licence_checked": ds.licence_checked}
        for usecase, (text, ds) in BUNDLED_DEMO_SAMPLES.items()
    }


def _model_registry() -> dict:
    """The model registry, or an empty mapping if the models package is unavailable.

    Kept import-guarded so ``frontend_config()`` still works in the slim demo image, which ships
    neither torch nor the training stack. An empty registry makes the SPA say "no models registered"
    rather than inventing one.
    """
    try:
        from ..models.registry import frontend_models
    except Exception:  # pragma: no cover - slim image only
        return {}
    return frontend_models()
