"""Round 84: the finance ceiling cites MEGA's published table, and three finance sentences stop
asserting what is not so.

WHY THIS EXISTS

1. The ceiling cited MEGA's arXiv v2 Table 2 (MEGA-PNA 74.01, with MEGA-GenAgg 74.88, MEGA(PNA)-GIN
   74.69 and Multi-PNA 68.16 beside it). The paper's published version, v3 of 2026-08-31 ("Published
   in Transactions on Machine Learning Research (08/2026)"), reports minority-class F1 in Table 16:
   MEGA-PNA is unchanged at 74.01 +/- 1.55, MEGA-GenAgg is 74.48 +/- 0.84, Multi-PNA 67.35 +/- 2.89,
   there is no MEGA(PNA)-GIN row, and the best AML Small HI row is Multi-FraudGT at 75.81 +/- 0.75.
   `ceiling` is "best credible published number", so it is that row.
2. "a tree ensemble is not FHE-feasible" is false as stated: Effendi and Chattopadhyay ran XGBoost
   with graph features on AMLworld HI-Small samples under TFHE (SPACE 2024, LNCS,
   doi:10.1007/978-3-031-80408-3_6; arXiv:2411.02926). Trees are not CKKS-native, which is the
   true part.
3. The finance agentic prompt asked why the agent "must ... run over a quantum-safe MCP channel
   rather than sharing the blacklists", presupposing what the answer should state, while the page
   said the keyed-hash PSI then in use let either bank recover the other's list over a small ID space.
   The prompt names the OPRF PSI that replaced it in Round 84 (tests/test_r84_oprf_psi.py).

Each test fails at the base commit (de1111c) on the text it checks.
"""

from __future__ import annotations

import re
from pathlib import Path

from qsh.models import registry

REPO = Path(__file__).resolve().parents[1]
FINANCE_KEYS = ("finance_aml.gnn", "finance_aml.gbm", "finance_aml.student")


def test_the_ceiling_is_the_best_small_hi_row_of_the_published_table() -> None:
    m = registry.M_AML
    assert m.ceiling == 0.7581
    src = m.ceiling_source
    for frag in ("Multi-FraudGT", "Table 16", "TMLR (08/2026)", "arXiv:2412.00241v3",
                 "75.81 +/- 0.75", "MEGA-PNA scores 74.01 +/- 1.55", "MEGA-GenAgg 74.48 +/- 0.84",
                 "Multi-PNA 67.35 +/- 2.89"):
        assert frag in src, (frag, src)
    for stale in ("v2", "74.88", "74.69", "MEGA(PNA)-GIN", "68.16", "accepted for publication"):
        assert stale not in src, (stale, src)
    # The ceiling is read from the source text, not kept apart from it.
    assert f"{m.ceiling * 100:.2f}" in src


def test_every_finance_model_exports_the_new_ceiling_and_the_gnn_note_agrees() -> None:
    models = {s["key"]: s for s in registry.frontend_models()["finance_aml"]}
    for key in FINANCE_KEYS:
        assert models[key]["metric"]["ceiling"] == 0.7581, key
        assert models[key]["metric"]["ceiling_source"] == registry.M_AML.ceiling_source
    gnn = models["finance_aml.gnn"]["notes"]
    assert "67.35 +/- 2.89" in gnn and "the MEGA rows above it are" not in gnn
    student = models["finance_aml.student"]["notes"]
    assert "8-feature" not in student and "four-feature linear model" in student
    # The teacher's card says which model the page encrypts instead, from its own note, so a card
    # template that defers to the notes has a true sentence to defer to.
    gbm = models["finance_aml.gbm"]["notes"]
    assert "It is not evaluated under encryption" in gbm and "finance_aml.student" in gbm


def test_the_finance_text_written_here_is_plain_ascii() -> None:
    texts = [registry.M_AML.ceiling_source, registry.DS_PAYSIM.notes, registry.PAYSIM_BALANCE_WARNING,
             *registry.PAYSIM_MODEL_FEATURES.values(), *registry.PAYSIM_EXCLUDED_FEATURES.values(),
             *(s.notes for s in registry.specs() if s.key in FINANCE_KEYS)]
    bad = [t for t in texts if not t.isascii()]
    assert not bad, bad


def test_the_tree_sentence_says_ckks_and_cites_the_tfhe_result() -> None:
    src = (REPO / "src" / "qsh" / "usecases" / "finance_aml" / "__init__.py").read_text()
    flat = re.sub(r"\s+", " ", src)
    assert "not FHE-feasible" not in flat and "trees = secure comparisons" not in flat
    assert "not CKKS-native" in flat and "Under TFHE trees do run" in flat
    assert "doi:10.1007/978-3-031-80408-3_6" in flat and "arXiv:2411.02926" in flat
    assert "15,038 to 165,917 times slower than plaintext" in flat


def test_the_finance_agentic_prompt_asks_rather_than_asserts() -> None:
    uc = (REPO / "web" / "src" / "lib" / "useCases.ts").read_text()
    m = re.search(r'id: "finance_aml".*?agenticPrompt: "([^"]*)"', uc, re.S)
    assert m, "finance agenticPrompt not found"
    prompt = m.group(1)
    assert "rather than sharing the blacklists" not in prompt and "must" not in prompt
    # The construction the page runs, by name: the OPRF PSI since Round 84 (psi), the keyed hash before.
    assert "OPRF-based PSI" in prompt and "keyed-hash" not in prompt and "what is hidden, and from whom" in prompt
