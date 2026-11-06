"""What the round's readers found that a run could get wrong, held where the code could get it wrong again.

WHY THIS EXISTS (Round 86). Nine readers went through the plans, orders and records against the code, and the functional
findings still live on main were these. Each test names the defect it holds; each was seen to fail with the defect put back.

- The MPC selector offered MP-SPDZ and hpmpc, which no stage ran: choosing either changed nothing a run computed. The
  family now names only what runs, and the browser's offline mirror of the options names the same.
- The eavesdropper flag on the Validations page read the retired security score's threshold, in both tiers, while the
  QKD stage aborts at its own; the MCP telemetry tool had a literal of its own. All three read `qkd.pp_threshold`.
- The qubit range of the comparison circuit was the literals 2 and 14 at three sites, and the large tier's calibration
  sample count a literal 4.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_every_mpc_option_is_the_backend_a_run_executes() -> None:
    """A registered MPC option must be what the run's benchmark row says ran, or the selector changes nothing."""
    from qsh.orchestration.crypto_agility import default_registry
    from qsh.orchestration.pipeline import build_pipeline

    options = default_registry().options("mpc")
    assert options, "the mpc family is empty"
    for option in options:
        pipe, ctx = build_pipeline("finance_aml", {"model_size": "lightweight", "seed": 0, "mpc": option})
        pipe.run(ctx)
        ran = ctx.artifacts.get("mpc_nonlinear_backend") or ""
        assert ran.endswith(option), f"mpc={option!r} was offered, and the run's row says {ran!r} ran"


def test_the_static_demo_offers_what_the_registry_offers() -> None:
    """`DEMO_CRYPTO` is the offline mirror of /api/crypto-options; a name in one and not the other is a drifted twin.

    Options named for a GPU are left out: the registry offers them only on a machine where they load.
    """
    from qsh.orchestration.crypto_agility import default_registry

    hook = (REPO / "web" / "src" / "lib" / "usePipelineRun.ts").read_text(encoding="utf-8")
    block = hook[hook.index("export const DEMO_CRYPTO"):]
    block = block[:block.index("defaults:")]
    reg = default_registry()
    for family in ("kem", "qkd", "fhe", "mpc", "sig"):
        m = re.search(rf"{family}: \[([^\]]*)\]", block)
        assert m, f"DEMO_CRYPTO has no {family} list"
        mirrored = re.findall(r'"([^"]+)"', m.group(1))
        registered = [o for o in reg.options(family) if not o.endswith("-gpu")]
        assert mirrored == registered, f"{family}: the browser mirror says {mirrored}, the registry {registered}"


def test_the_eavesdropper_flag_and_the_mcp_tool_read_the_abort_threshold() -> None:
    from qsh.config import get_config
    from qsh.mcp.server import dispatch_tool

    t = get_config().qkd.pp_threshold
    just_over, just_under = round(t + 0.005, 4), round(t - 0.005, 4)
    assert dispatch_tool("inspect_qkd_telemetry", {"qber": just_over})["eavesdropper_detected"] is True
    assert dispatch_tool("inspect_qkd_telemetry", {"qber": just_under})["eavesdropper_detected"] is False
    assert dispatch_tool("inspect_qkd_telemetry", {"qber": just_under})["threshold"] == t
    service = (REPO / "api" / "inference_service.py").read_text(encoding="utf-8")
    assert '"eavesdropper_likely": r.qber > get_config().qkd.pp_threshold' in service
    api = (REPO / "web" / "src" / "lib" / "api.ts").read_text(encoding="utf-8")
    assert "eavesdropper_likely: r.qber > qkdCfg().pp_threshold" in api
    # The code, not the comments, which say what it read before.
    code = re.sub(r"#[^\n]*", "", service) + re.sub(r"//[^\n]*", "", api)
    assert "qber_secure_threshold" not in code, "a flag still reads the retired score's threshold"


def test_the_qubit_range_and_the_calibration_count_come_from_config() -> None:
    from qsh.config import frontend_config, get_config

    b = get_config().benchmark
    exported = frontend_config()["benchmark"]
    assert (exported["quantum_qubits_min"], exported["quantum_qubits_max"]) == (b.quantum_qubits_min, b.quantum_qubits_max)
    compute = (REPO / "web" / "src" / "lib" / "clientCompute.ts").read_text(encoding="utf-8")
    panel = (REPO / "web" / "src" / "components" / "ParameterPanel.tsx").read_text(encoding="utf-8")
    assert not re.search(r"Math\.min\(\s*14\s*,", compute), "a literal qubit cap is back in clientCompute.ts"
    assert '"qubits", 1, 2,' not in panel and "capped at 14" not in panel, "the Parameters field holds literals again"
    assert not re.search(r"qsmpcFheRow\([^)]*,\s*6\s*,", compute), "a literal sample count is back in a quantum row"
    runner = (REPO / "src" / "qsh" / "benchmark" / "runner.py").read_text(encoding="utf-8")
    assert "min(n_hybrid, 4)" not in runner and "large_calibration_samples" in runner


def test_the_retired_security_score_stays_retired(tmp_path) -> None:
    """No row, signed summary, export or config carries the 0-100 security score or its points (Round 86, owner's choice).

    It was retired from the pages in Round 22 and still computed in both tiers, signed into every run summary and written
    into the benchmark downloads. A number described as indefensible was shipping inside the records meant to be checked.
    """
    import json

    from qsh.benchmark.runner import run_benchmark
    from qsh.config import frontend_config
    from qsh.orchestration.pipeline import build_pipeline

    assert not (REPO / "src" / "qsh" / "benchmark" / "security.py").exists()
    assert "security_scoring" not in frontend_config()
    for row in run_benchmark("linear", out_dir=tmp_path, n_samples=4,
                             include_openfhe=False, model_size="lightweight"):
        assert not hasattr(row, "security_score"), row
        assert "security_components" not in (row.extra or {}), row.backend
    pipe, ctx = build_pipeline("finance_aml", {"model_size": "lightweight", "seed": 0})
    pipe.run(ctx)
    signed = json.dumps(ctx.artifacts["summary"], default=str)
    for word in ("security_score", "security_components", "security_summary"):
        assert word not in signed, f"the signed run summary carries {word}"
    code = re.sub(r"//[^\n]*", "", (REPO / "web" / "src" / "lib" / "clientCompute.ts").read_text(encoding="utf-8"))
    assert "security_score" not in code and "function securityScore" not in code
    assert "securityScoringCfg" not in (REPO / "web" / "src" / "lib" / "config.ts").read_text(encoding="utf-8")


def test_a_missing_medmnist_set_is_an_error_unless_a_test_opts_in(monkeypatch) -> None:
    """The loader fell back to synthetic images with a warning only the server log showed (Round 86, a reader)."""
    import pytest

    from qsh.data.loaders import load_medmnist

    monkeypatch.delenv("QSH_ALLOW_SYNTHETIC_IMAGES", raising=False)
    with pytest.raises(RuntimeError, match="could not be loaded"):
        load_medmnist("no-such-mnist")
    monkeypatch.setenv("QSH_ALLOW_SYNTHETIC_IMAGES", "1")
    X, y = load_medmnist("no-such-mnist")
    assert X.ndim == 4 and len(X) == len(y) > 0


def test_the_medical_cnn_path_reads_the_configured_corpus_and_names_its_class(monkeypatch) -> None:
    """The full deployment's CNN path defaulted to PneumoniaMNIST, while the config, the published student and the browser
    are BreastMNIST; and its result was a bare index, whose meaning flips between the two corpora."""
    import numpy as np

    from qsh.config import get_config
    from qsh.data import loaders
    from qsh.usecases import medical_fl

    asked = []

    def fake_load(name=None, split="train"):
        asked.append(name)
        rng = np.random.default_rng(0)
        return rng.uniform(size=(64, 1, 28, 28)).astype(np.float32), rng.integers(0, 2, 64)

    monkeypatch.setattr(loaders, "load_medmnist", fake_load)
    pipe, ctx = medical_fl.build({"dataset": "medmnist", "img_size": 10, "medmnist_n": 64})
    assert asked == [get_config().benchmark.medical_cnn["medmnist_name"]] == ["breastmnist"], asked
    assert ctx.data["class_labels"] == {0: "malignant", 1: "normal, benign"}
    # And the full deployment's page sends the dataset that selects this path.
    hook = (REPO / "web" / "src" / "lib" / "usePipelineRun.ts").read_text(encoding="utf-8")
    assert 'if (!DEMO && usecase === "medical_fl") c.dataset = "medmnist";' in hook
    image = (REPO / "docker" / "Dockerfile.api").read_text(encoding="utf-8")
    assert "BreastMNIST(split='train', download=True)" in image and "PneumoniaMNIST" not in image.split("RUN python -c")[1]
