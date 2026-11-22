"""DATA_LICENSES.md states what the repository holds, and the QKD-attack corpus claim stays corrected (Round 86, 2026-11-22).

WHY THIS EXISTS

The register was checked at every source it names on 2026-11-22, and every licence held. What had gone stale was its own
bookkeeping: it counted "All thirteen Hub repos, seven teachers, four encrypted students" when the registry publishes eighteen,
ten and six; it listed four students and "the five Hub students'" weights; it named the MCP filesystem server's 2.1.0 when the
image locks 2.2.0. Nothing checked any of these against the repository. The same pass found that the correction of 2026-11-08,
"No public real QKD-attack capture was found" in place of a claim that no commercial-OK real corpus exists, had reached the
registry, the bundled files and the telecom cards but not the telecom QML record's `framing`, the trainer constant that writes
it, the 7-wire telecom QCNN's served weights and its card, a reference note, or the register's own section 3.

WHAT IS ASSERTED

- The counts and the student list in DATA_LICENSES.md are the registry's, and the student weights it names are tracked.
- Each submodule's pinned commit in the register is the gitlink the repository records.
- The MCP filesystem server's version in the register is the one `docker/mcp-fs/` locks and `qsh.mcp.fs_interop` starts.
- No served file, card, record or trainer constant says that no commercial-OK real QKD-attack corpus exists.
- Seeded: the sentences as they stood are refused.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
#: The two finance models published beside the teachers and the students, each with held-out rows in its weights.
NEITHER = {"finance_aml.mlp_large", "finance_aml.gnn_compact"}
#: The QKD-attack claim in each wording it shipped in, the network-attack one before it, and the finance cards'
#: "no commercial-OK corpus of real laundering transactions exists", which a real GPL-3.0 bank set contradicts
#: (references/aml-real-data-2026-11.md; reworded on the owner's choice of 2026-11-22).
RETIRED = re.compile(r"commercial-OK (real (QKD-|network-)?attack|real QKD|corpus of real (QKD|laundering))", re.I)
#: The finance sentence the owner chose, which the registry note, the cards and the two documents carry.
FOUND = ("No real laundering-labelled corpus under a permissive licence was found (2026-11-22): Elliptic and Elliptic2 "
         "are CC BY-NC-ND 4.0, and the one real bank set found is GPL-3.0 with differentially private PCA features "
         "(arXiv:2507.08835)")


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _published() -> tuple[set[str], set[str], set[str]]:
    from qsh.models import registry

    published = {s.key for s in registry.specs() if s.hf_repo}
    students = {k for k in published if k.endswith("student")}
    return published, students, published - students - NEITHER


def count_problems(register: str) -> list[str]:
    published, students, teachers = _published()
    flat = _flat(register)
    want = (f"All {WORDS[len(published)]} Hub repos, {WORDS[len(teachers)]} teachers, {WORDS[len(students)]} encrypted "
            "students, the finance MLP and the compact GNN, are public")
    problems = [] if want in flat else [f"the register does not say {want!r}"]
    if f"the {WORDS[len(students)]} Hub students'" not in flat:
        problems.append(f"the tracked student weights are not counted as {WORDS[len(students)]}")
    if f"all {WORDS[len(published)]} repos ship one" not in flat:
        problems.append("the evaluation samples are not counted per published repo")
    metrics = json.loads((REPO / "models" / "metrics.json").read_text(encoding="utf-8"))
    bullet = flat[flat.index("The encrypted-tier students are published too"):][:700]
    for key in sorted(students):
        params = metrics[key].get("student_params") or metrics[key].get("n_parameters")
        if f"`{key}` ({params}" not in bullet:
            problems.append(f"the students' list lacks `{key}` ({params})")
    return problems


def test_the_register_counts_what_the_registry_publishes() -> None:
    assert count_problems((REPO / "DATA_LICENSES.md").read_text(encoding="utf-8")) == []
    tracked = set(subprocess.run(["git", "ls-files", "models/*/weights.json"], cwd=REPO, capture_output=True, text=True,
                                 check=True).stdout.split())
    from qsh.models import registry

    _, students, _ = _published()
    for key in students:
        assert f"{registry.ARTIFACT_DIRS[key]}/weights.json" in tracked, f"{key}'s weights.json is not tracked"


def test_the_submodule_pins_are_the_gitlinks() -> None:
    register = (REPO / "DATA_LICENSES.md").read_text(encoding="utf-8")
    for path in ("submodules/HEonGPU", "submodules/hpmpc"):
        line = subprocess.run(["git", "ls-tree", "HEAD", path], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.split()
        assert line and line[1] == "commit", f"{path} is not a submodule at HEAD"
        row = next(r for r in register.splitlines() if r.startswith(f"| `{path}`"))
        assert f"`{line[2][:12]}`" in row, f"{path}: the register's pin is not the gitlink {line[2][:12]}"


def test_the_mcp_server_version_is_the_one_the_image_locks() -> None:
    lock = json.loads((REPO / "docker" / "mcp-fs" / "package-lock.json").read_text(encoding="utf-8"))
    locked = lock["packages"]["node_modules/@modelcontextprotocol/server-filesystem"]["version"]
    register = _flat((REPO / "DATA_LICENSES.md").read_text(encoding="utf-8"))
    m = re.search(r"npm (\d+\.\d+\.\d+), the version `docker/mcp-fs/` locks", register)
    assert m and m.group(1) == locked, f"the register names {m and m.group(1)}; the image locks {locked}"
    assert f"server-filesystem@{locked}" in (REPO / "src" / "qsh" / "mcp" / "fs_interop.py").read_text(encoding="utf-8")


def _strings(node) -> list[str]:
    if isinstance(node, dict):
        return [s for v in node.values() for s in _strings(v)]
    if isinstance(node, list):
        return [s for v in node for s in _strings(v)]
    return [node] if isinstance(node, str) else []


def _trainer_constants(path: Path) -> list[str]:
    """Module-level string constants of a trainer, read without importing it (the trainers import their ML stacks)."""
    out = []
    for stmt in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            out.append(stmt.value.value)
    return out


def retired_claims(texts: dict[str, list[str]]) -> list[str]:
    return [f"{rel}: ...{s[max(0, m.start() - 60):m.end() + 40]}..." for rel, strings in texts.items()
            for s in strings for m in [RETIRED.search(s)] if m]


def _shipped_texts() -> dict[str, list[str]]:
    texts: dict[str, list[str]] = {}
    for rel in ("models/metrics.json", "web/src/lib/config.defaults.json"):
        texts[rel] = _strings(json.loads((REPO / rel).read_text(encoding="utf-8")))
    for p in sorted((REPO / "web" / "public" / "data").rglob("*.json")):
        texts[str(p.relative_to(REPO))] = _strings(json.loads(p.read_text(encoding="utf-8")))
    for p in sorted((REPO / "models").glob("*/README.md")):
        texts[str(p.relative_to(REPO))] = [_flat(p.read_text(encoding="utf-8"))]
    for p in sorted((REPO / "scripts" / "train").glob("*.py")):
        texts[str(p.relative_to(REPO))] = _trainer_constants(p)
    return texts


def test_no_shipped_text_says_no_commercial_ok_real_corpus_exists() -> None:
    texts = _shipped_texts()
    assert len(texts) > 40, f"only {len(texts)} files read; the scan is looking at nothing"
    assert retired_claims(texts) == []
    register = _flat((REPO / "DATA_LICENSES.md").read_text(encoding="utf-8"))
    assert "no commercial-OK corpus of real QKD-layer attacks exists" not in register
    assert "no public corpus of real QKD-layer attacks was found" in register


def test_the_finance_sentence_is_the_one_chosen_everywhere_it_stands() -> None:
    from qsh.models import registry

    places = {"the registry note": registry.get("finance_aml.gbm").dataset.notes,
              "DATA_LICENSES.md": (REPO / "DATA_LICENSES.md").read_text(encoding="utf-8"),
              "docs/models-and-benchmarks.md": (REPO / "docs" / "models-and-benchmarks.md").read_text(encoding="utf-8")}
    for key in sorted(k for k in registry.ARTIFACT_DIRS if k.startswith("finance_aml.")):
        places[key] = (REPO / registry.ARTIFACT_DIRS[key] / "README.md").read_text(encoding="utf-8")
    missing = [where for where, text in places.items() if FOUND not in _flat(text)]
    assert not missing, f"the chosen finance sentence is missing from {missing}"


def test_the_checks_refuse_the_sentences_as_they_stood() -> None:
    as_they_stood = {
        "web/public/data/telecom/qcnn_weights_q7.json": [
            "8 classes = 7 attack types + normal traffic. No commercial-OK real attack corpus exists, and the synthetic "
            "origin is why no result here is a performance claim about real links."],
        "models/metrics.json": ["The data is synthetic because no commercial-OK real QKD-attack corpus exists. "],
        "models/finance_aml_gbm/README.md": ["Chosen for its calibration to real transaction statistics and its "
                                             "CDLA-Sharing terms, not because it is real data - no commercial-OK corpus "
                                             "of real laundering transactions exists."],
    }
    assert len(retired_claims(as_they_stood)) == 3
    register = (REPO / "DATA_LICENSES.md").read_text(encoding="utf-8")
    stale = register.replace(
        _flat_find(register, "All eighteen"), "All thirteen Hub\n  repos, seven teachers, four encrypted students")
    assert count_problems(stale), "the count as it stood was not refused"


def _flat_find(text: str, start: str) -> str:
    """The register's own text from `start` through "encrypted students", line breaks and indentation included."""
    i = text.index(start)
    return text[i:text.index("encrypted students", i) + len("encrypted students")]
