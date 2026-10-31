"""The Python genomics pipeline: full score under CKKS against a stated cut, a sum that cannot wrap.

WHY THIS EXISTS (Round 84, scratchpad audit83/page-finance-genomics.md, findings 1, 2, 8, 13, 14)

The Python tier always evaluated all 313 variants, but against the 75th percentile of its cohort under
the label "the clinical risk cut", which no source supports, and it reported agreement over ONE row,
the top scorer, whose flag cannot come out "no". The secure sum's field is the inputs' range since
the Round 84 draft; on the maintainer's full panel the four cohorts carry 75, 78, 94 and 76 effect
alleles (323), which the paper's GF(5) had returned as 3, and that case is pinned here. The MCP
allele-pooling tool named a "d-qudit GHZ secure summation" that its 2^31-1 field never simulates, and
returned a reduced count for inputs that sum past it. The benchmark row priced a "genomic risk MLP".

WHAT IS ASSERTED

  - the cut is the 90th percentile of the loader's own scores, the same percentile as the browser
    and as the Models card's global cut;
  - every panel individual is scored under CKKS, with every score within 1e-6 of the clear and the
    high-risk flag agreeing on every row, and the flag comes out both ways;
  - the packed SEAL path and the one-row-per-ciphertext fallback compute the same scores;
  - the 323 case pools to 323, in the stage and through `build`, and a field that cannot hold it
    is refused;
  - the MCP tool names what ran and refuses counts that could wrap;
  - the genomics benchmark row has no hidden layer at any size.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
COUNTS_323 = [75, 78, 94, 76]


@pytest.fixture(scope="module")
def demo() -> dict:
    pytest.importorskip("tenseal")
    from qsh.usecases import genomics

    pipe, ctx = genomics.build({"seed": 1})
    pipe.run(ctx)
    logs = [e.message for e in pipe.bus.history if e.type.value == "log" and e.stage == "private_inference"]
    return {"ctx": ctx, "metrics": pipe.bus.metrics(), "logs": logs}


def _cohort():
    from qsh.config import get_config
    from qsh.data.loaders import load_genomic_prs

    d = get_config().data
    x, w, *_ = load_genomic_prs(n_individuals=max(d.genomics_n // 3, 200), seed=1,
                                name=d.genomics_prs_score)
    return x, w


def test_one_percentile_in_both_languages_and_on_the_models_card() -> None:
    import importlib.util

    from qsh.usecases.genomics import HIGH_RISK_PERCENTILE

    spec = importlib.util.spec_from_file_location("gp", REPO / "scripts" / "train" / "genomics_prs.py")
    gp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gp)
    ts = (REPO / "web" / "src" / "lib" / "realData.ts").read_text()
    browser = re.search(r"export const PRS_HIGH_RISK_PERCENTILE = (\d+(?:\.\d+)?);", ts)
    assert browser, "realData.ts declares no PRS_HIGH_RISK_PERCENTILE"
    assert HIGH_RISK_PERCENTILE == gp.HIGH_RISK_PERCENTILE == float(browser.group(1)) == 90.0


def test_the_cut_is_the_stated_percentile_of_the_panel(demo: dict) -> None:
    from qsh.usecases.genomics import HIGH_RISK_PERCENTILE

    x, w = _cohort()
    cut = float(np.percentile(x @ w, HIGH_RISK_PERCENTILE))
    prs = demo["ctx"].data["_prs"]
    assert prs["threshold"] == round(cut, 4) and prs["cut_population"] == len(x)
    assert prs["high_risk_percentile"] == HIGH_RISK_PERCENTILE
    assert demo["metrics"]["prs_high_risk_cut"] == round(cut, 6)
    model = demo["ctx"].data["linear_model"]
    assert np.allclose(model.bias, [0.0, -cut]) and not model.weight[0].any()
    assert np.array_equal(model.weight[1], w), "the encrypted model is not the full weight vector"


def test_every_panel_individual_is_scored_and_the_flag_agrees(demo: dict) -> None:
    x, w = _cohort()
    scores = x @ w
    flagged = int(np.sum(scores > demo["ctx"].artifacts["prs_panel"]["cut"]))
    p = demo["ctx"].artifacts["prs_panel"]
    assert p["rows"] == len(x) and p["flags_agree"] == len(x), p
    assert 0 < p["max_abs_error"] < 1e-6
    assert p["flagged_encrypted"] == p["flagged_plaintext"] == flagged
    assert 0 < flagged < len(x), "the cut flags nobody or everybody, so the flag cannot test anything"
    assert p["ciphertexts"] == -(-len(x) // p["rows_per_ciphertext"])
    m = demo["metrics"]
    assert m["prs_panel_flag_agreement"] == 1.0 and m["prs_subject_flags_agree"] == 1


def test_the_log_says_what_is_encrypted_and_what_is_decrypted(demo: dict) -> None:
    line = next((m for m in demo["logs"] if m.startswith("full polygenic score under CKKS")), None)
    assert line, "the Python run does not describe its full-score evaluation"
    for part in ("Encrypted: each person's 313 effect-allele dosages", "In plaintext:",
                 "Decrypted: the result ciphertext", "90th percentile of the plaintext score"):
        assert part in line, f"{part!r} missing from: {line}"
    assert not any("sign test" in m for m in demo["logs"])


def test_the_packed_path_and_the_fallback_agree() -> None:
    pytest.importorskip("tenseal")
    from qsh.fhe.backends import get_backend
    from qsh.usecases.genomics import encrypted_scores

    x, w = _cohort()
    x = x[:20]
    be = get_backend("tenseal")

    class Plain:
        """The same backend without SEAL's evaluator: one person per ciphertext."""
        def __init__(self, inner):
            self.inner = inner

        def encrypt(self, v):
            return self.inner.encrypt(v)

        def dot_plain(self, ct, v):
            return self.inner.dot_plain(ct, v)

        def decrypt(self, ct):
            return self.inner.decrypt(ct)

    packed, single = encrypted_scores(be, x, w), encrypted_scores(Plain(be), x, w)
    assert packed["packed"] and not single["packed"] and single["ciphertexts"] == len(x)
    assert packed["ciphertexts"] == -(-len(x) // packed["rows_per_ciphertext"])
    assert np.allclose(packed["encrypted"], x @ w, atol=1e-6)
    assert np.allclose(single["encrypted"], x @ w, atol=1e-4)


def test_the_full_panel_case_pools_to_323() -> None:
    from qsh.qsmpc.secure_summation import QSMS, field_for_total

    # The defect, as arithmetic: the paper's field for four parties is GF(5), and 323 mod 5 is 3.
    assert QSMS(n=4, t=3).run(COUNTS_323, seed=0).total == 3
    p = field_for_total(2 * 400, 4)
    assert p == 809
    for seed in range(5):
        assert QSMS(n=4, t=3, prime=p).run(COUNTS_323, seed=seed).total == 323


def test_the_full_panel_case_through_build_and_the_stage(monkeypatch) -> None:
    """`build` on a 400-person cohort whose first column splits as the full panel's does."""
    from qsh.data import loaders
    from qsh.orchestration.crypto_agility import CryptoRegistry
    from qsh.orchestration.events import EventBus
    from qsh.orchestration.stages import QSMPCComputeStage
    from qsh.usecases import genomics

    keys, eff, w = loaders.load_prs("breast_cancer_prs313")
    rng = np.random.default_rng(84)
    x = rng.integers(0, 3, size=(400, len(w))).astype(float)
    for k, total in enumerate(COUNTS_323):                 # 100 people per cohort, `total` alleles
        col = np.zeros(100)
        col[: total // 2] = 2
        col[total // 2: total // 2 + total % 2] = 1
        x[k * 100:(k + 1) * 100, 0] = rng.permutation(col)
    monkeypatch.setattr(loaders, "load_genomic_prs",
                        lambda **_kw: (x, np.asarray(w, dtype=float), keys, eff, x.mean(axis=0) / 2))
    _, ctx = genomics.build({"seed": 1})
    assert ctx.data["secrets"] == COUNTS_323 and ctx.config["prime"] == 809
    ctx.data["session_key"] = b"\x02" * 32
    QSMPCComputeStage().run(ctx, EventBus(), CryptoRegistry())
    assert ctx.artifacts["secure_sum"] == 323 == ctx.data["_ground_truth_total"]
    with pytest.raises(ValueError, match="would wrap"):
        genomics.build({"seed": 1, "prime": 797})            # prime, and below the 800 the cohort can reach


def test_the_mcp_tool_names_what_ran_and_cannot_wrap() -> None:
    from qsh.mcp.server import pool_allele_counts

    r = pool_allele_counts(COUNTS_323)
    assert r["pooled_allele_count"] == 323
    assert r["ghz_statevector_simulated"] is False
    assert "analytic sampler" in r["method"] and "GHZ statevector simulated" not in r["method"], r["method"]
    for bad in ([-1, 4], [2**31, 1]):
        with pytest.raises(ValueError, match="cannot wrap"):
            pool_allele_counts(bad)


def test_the_genomics_benchmark_row_is_not_an_mlp() -> None:
    from qsh.benchmark.runner import usecase_arch_layers
    from qsh.config import get_config

    arch = get_config().benchmark.tier_architectures["genomics"]
    assert "MLP" not in arch["label"] and all(h == [] for h in arch["hidden"].values()), arch
    for size in ("medium", "large"):
        layers, label, params = usecase_arch_layers("genomics", size, n_in=313)
        assert [lw.shape for lw, _ in layers] == [(2, 313)] and "MLP" not in label
