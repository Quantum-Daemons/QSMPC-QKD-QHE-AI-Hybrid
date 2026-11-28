"""Which model each page encrypts is declared once, and every text that names it is held to the declaration (Round 86).

The owner's choice of 2026-11-28, after an inventory of every page found that finance's cards went on naming the
linear CKKS student as the page's encrypted model for three days after F7b's square network had replaced it on both
of the page's encrypted paths: the use-case card's "encrypted student 115 params", the Benchmark row's label, the
private-inference figure's fit note, the Models panel's note, the SHAP footer and the canvas's lane, while the page
fetched the linear student's weights and applied them to 8,192 rows for nothing. Telecom's 5G-NIDD path has run its
square network since batch 12, in place of batch 6's linear student.

`web/src/lib/pageEncryptedModels.ts` declares each page's encrypted models; this holds:
1. every site-file network's parameter count to the file the page fetches;
2. each replaced model to being replaced: no code path runs it, and no text the page renders says it is encrypted on
   the page, while the registry notes say when it was and what replaced it;
3. the texts whose words depend on the model's kind (the SHAP footer, the canvas lane, the Benchmark card, the figure's
   fit note) to the kind each page encrypts.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
LIB = WEB / "src" / "lib"
DECL = (LIB / "pageEncryptedModels.ts").read_text()

sys.path.insert(0, str(REPO / "src"))
from qsh.models import registry  # noqa: E402

FILES = {"finance_aml.square_network": WEB / "public" / "data" / "finance" / "square_student_weights.json",
         "telecom_ids.square_network": WEB / "public" / "data" / "telecom_nidd" / "square_network.json"}


def _site_entries() -> dict[str, dict]:
    out = {}
    for m in re.finditer(r'key: "([a-z_.]+)", label: "([^"]+)", params: (\d+), kind: "([a-z ]+)",\s*'
                         r'where: "([^"]+)", replaced: "([a-z_.]+)"', DECL):
        out[m.group(1)] = {"label": m.group(2), "params": int(m.group(3)), "kind": m.group(4), "where": m.group(5),
                           "replaced": m.group(6)}
    return out


def _params(w: dict) -> int:
    return len(w["w1"]) * len(w["w1"][0]) + len(w["b1"]) + len(w["w2"]) * len(w["w2"][0]) + len(w["b2"])


def test_every_site_file_network_is_declared_at_its_files_size() -> None:
    entries = _site_entries()
    assert set(entries) == set(FILES), entries
    for key, path in FILES.items():
        assert entries[key]["params"] == _params(json.loads(path.read_text())), key
        assert entries[key]["kind"] == "square network", key


def test_a_replaced_model_runs_on_no_path_the_declaration_replaced() -> None:
    compute = (LIB / "clientCompute.ts").read_text()
    real = (LIB / "realData.ts").read_text()
    # Finance: both encrypted paths read the square network; the held-out corpus no longer reads the linear weights.
    finance = real[real.index("export async function heldOutCorpus("):]
    finance = finance[:finance.index("if (usecase === \"telecom_ids\")")]
    assert "squareHeldOut(usecase" in finance and "publishedStudentHeldOut" not in finance
    # Telecom's 5G-NIDD stage reads the square network and nothing else.
    nidd = compute[compute.index("async function scoreNiddStudent("):compute.index("async function scoreNiddFederated(")]
    assert "niddSquareHeldOut()" in nidd and "niddStudentHeldOut" not in nidd and "encryptedHeldOutAgreement(" not in nidd


def test_no_note_says_a_replaced_model_is_encrypted_on_the_page() -> None:
    student, gbm = registry.get("finance_aml.student"), registry.get("finance_aml.gbm")
    assert "Encrypted on the public page since" not in student.notes
    assert "until F7b (2026-11-25)" in student.notes and "square network" in student.notes
    assert "the public page encrypts finance_aml.student" not in gbm.notes and "square network" in gbm.notes
    nidd = registry.get("telecom_ids.nidd")
    assert "with a linear model trained on the labels" not in nidd.notes and "square network" in nidd.notes
    fed = registry.M_NIDD_FED
    assert "page's CKKS student" not in fed.baseline_source and "page's CKKS student" not in fed.companion_note


def test_the_kind_dependent_words_follow_the_kind() -> None:
    shap = (WEB / "src" / "components" / "ShapPanel.tsx").read_text()
    assert 'explanation?.method === "exact-quadratic-shap"' in shap and "exact Shapley values of the quadratic network" in shap
    flows = (LIB / "useCaseFlows.ts").read_text()
    assert 'usecase === "finance_aml" ? "CKKS: evaluates the square network"' in flows
    assert 'usecase === "finance_aml" ? "MPC-FHE hybrid: the square network under CKKS end to end' in flows
    bench = (WEB / "src" / "components" / "BenchmarkPanel.tsx").read_text()
    assert "Hybrid split - linear layers to FHE" not in bench and "FHE-linear:" not in bench
    real = (LIB / "realData.ts").read_text()
    assert "F7b's square network was fitted on the earlier fit window" in real
    # The card and the cost row read the declaration, not the registry's first encrypted student.
    cards = (WEB / "src" / "pages" / "UseCases.tsx").read_text()
    # Every model the page encrypts (batch 13: telecom's card named its QKD-layer student only, after its 5G-NIDD figure
    # began to encrypt the square network).
    assert "pageEncryptedModels(u.id)" in cards and "studentModel(u.id)" not in cards and "primaryEncryptedModel(u.id)" not in cards
    compute = (LIB / "clientCompute.ts").read_text()
    assert "const student = usecase ? primaryEncryptedModel(usecase) : null;" in compute
