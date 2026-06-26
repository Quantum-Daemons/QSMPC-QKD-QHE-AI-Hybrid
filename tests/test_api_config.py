"""The /api/config endpoint serves the single-source config the SPA consumes (Round 10).

It must expose the benchmark cost model + tier architectures + workload multipliers + security/
advantage weights, and match the baked web/src/lib/config.defaults.json (so the demo's cost model
cannot drift from the backend).
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from api.main import app_config
from qsh.config import frontend_config


def test_api_config_serves_frontend_config():
    data = app_config()                                   # the /api/config handler
    # the sections the frontend depends on
    for key in ("benchmark", "security_scoring", "quantum_advantage", "data", "qkd", "web"):
        assert key in data
    b = data["benchmark"]
    assert b["tier_architectures"]["medical_fl"]["hidden"]["medium"]  # per-use-case arch present
    assert b["cost_per_op"]["fhe_ms_per_dot"] > 0
    assert b["medical_cnn"]["img_size"] > 0                            # genuine-CNN config present
    # it equals frontend_config() (the same source the export script writes), JSON-round-trippable
    assert json.loads(json.dumps(data)) == json.loads(json.dumps(frontend_config()))


def test_baked_json_matches_backend_config():
    """The committed web/src/lib/config.defaults.json (demo single-source) must equal frontend_config();
    regenerate with `python scripts/export_config.py` after editing config/defaults.py."""
    baked = json.loads((Path(__file__).resolve().parents[1] / "web" / "src" / "lib" /
                        "config.defaults.json").read_text())
    baked.pop("_generated_by", None)
    assert baked == json.loads(json.dumps(frontend_config()))
