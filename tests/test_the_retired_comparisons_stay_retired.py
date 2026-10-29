"""Two comparisons were retired from every surface on 2026-10-23, and this keeps them retired.

WHAT WAS RETIRED, AND WHY

1. The Overview card "~1.32 Rate-1 QFHE ciphertext/plaintext ratio vs ~1483x for classical CKKS".
   Both numbers were configuration constants and they were not the same quantity: 1.32 divided by the
   bytes of a CLASSICALLY SIMULATED 2^n-amplitude state vector (a quantum ciphertext has no size in
   bytes), and was not in the Rate-1 QFHE paper it was attributed to (Chardouvelis, Doettling,
   Malavolta, TCC 2021, which reports no concrete figure); 1483 was CKKS ciphertext bytes over about 34
   inputs - slot utilisation, not CKKS (Cheon, Kim, Kim, Song, ASIACRYPT 2017). Classical FHE reaches
   rate 1 too (Gentry and Halevi, TCC 2019; Brakerski, Doettling, Garg, Malavolta, TCC 2019).

2. The "quantum advantage" score. It printed "10/10 (dominant)" and "Quantum path is ~6146x more
   communication-compact and 5794x faster" on /orchestration and every inference page, labelled a
   diagram bar, and was fed to the in-browser LLM as run context. It divided the same mismatched byte
   and qubit ratios, divided a real CKKS + garbled-circuit inference latency by an 8-qubit simulation of
   an untrained circuit (5794x, 8163x and 9610x across three runs), and multiplied by 1.5 for a
   security label the run does not achieve. The Python module is KEPT, as the 0-100 composite was in
   Round 22; nothing renders, logs or prompts with it.

WHY A SOURCE-LEVEL GATE

Each surface was a separate place the same number reached a visitor - a card, a log line, a panel, a
diagram label, a prompt - and this project's most-repeated defect is a fix reaching one copy of
several. The checks below enumerate every surface the retirement covered. Comments are stripped before
matching, because the comments recording WHY these were retired name the very identifiers being
watched, and a gate that its own documentation satisfies tests nothing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web" / "src"


def _strip_ts(src: str) -> str:
    """Remove `/* ... */` (including JSX `{/* */}`) and `//` comments, block comments first."""
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    return re.sub(r"(?<![:\"'`])//[^\n]*", " ", src)


def _strip_py(src: str) -> str:
    """Remove `#` comments and triple-quoted strings (docstrings carry the retirement's history)."""
    src = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', " ", src)
    return re.sub(r"#[^\n]*", " ", src)


def _ts(rel: str) -> str:
    return _strip_ts((WEB / rel).read_text())


def test_the_overview_card_is_gone_and_reads_no_ratio() -> None:
    src = _ts("pages/Overview.tsx")
    assert "ct_pt_ratio" not in src, (
        "Overview.tsx reads a ciphertext/plaintext ratio again. The card it fed compared a simulated "
        "state vector's bytes with a CKKS byte ratio at under 1% slot use - two different quantities.")
    assert "Rate-1 QFHE ciphertext/plaintext ratio" not in src


def test_the_constant_with_the_false_provenance_is_not_configured() -> None:
    cfg = json.loads((WEB / "lib" / "config.defaults.json").read_text())
    cost = cfg["benchmark"]["cost_per_op"]
    assert "qsmpc_ct_pt_ratio" not in cost, (
        "`qsmpc_ct_pt_ratio` is back in the baked config. It was described as the Rate-1 QFHE "
        "'literature value', and the paper reports no concrete figure.")
    defaults = _strip_py((REPO / "src" / "qsh" / "config" / "defaults.py").read_text())
    assert "qsmpc_ct_pt_ratio" not in defaults


def test_nothing_calls_the_retired_advantage_function() -> None:
    """Defined once, called from nowhere - in the browser."""
    hits = []
    for path in sorted(WEB.rglob("*.ts*")):
        src = _strip_ts(path.read_text())
        calls = re.findall(r"\bquantumAdvantage\s*\(", src)
        defs = re.findall(r"function\s+quantumAdvantage\s*\(", src)
        if len(calls) - len(defs) > 0:
            hits.append(f"{path.relative_to(REPO)}: {len(calls) - len(defs)} call(s)")
    assert not hits, (
        "the retired quantum-advantage score is computed for display again:\n  " + "\n  ".join(hits))


def test_no_surface_renders_logs_or_prompts_with_the_score() -> None:
    """Every surface the score reached, enumerated - not just the panel that drew it."""
    surfaces = {
        "lib/clientCompute.ts": [r'metric\(\s*stage\s*,\s*"quantum_advantage_score"',
                                 r"quantum advantage \$\{"],
        "components/BenchmarkPanel.tsx": [r"\bresult\?\.advantage\b", r"\badvantage\.(score|tier|interpretation)\b"],
        "components/AgenticAIPanel.tsx": [r"\bbenchmark\?\.advantage\b", r"quantum advantage \$\{"],
        "lib/useCaseFlows.ts": [r'"quantum_advantage_score"'],
        "components/UseCaseCanvas.tsx": [r'"quantum_advantage_score"'],
        "pages/UseCases.tsx": [r"quantum\s+advantage"],
    }
    found = []
    for rel, pats in surfaces.items():
        src = _ts(rel)
        for pat in pats:
            if re.search(pat, src):
                found.append(f"{rel}: /{pat}/")
    assert not found, (
        "the retired quantum-advantage score reaches a visitor again:\n  " + "\n  ".join(found)
        + "\n\nIt divided quantities that are not the same on the two sides. If a comparison is wanted, "
        "build one whose two sides measure the same thing, and say so on the page.")


def test_the_python_pipeline_does_not_emit_it() -> None:
    src = _strip_py((REPO / "src" / "qsh" / "orchestration" / "stages.py").read_text())
    assert not re.search(r'bus\.metric\(\s*"quantum_advantage_score"', src), (
        "stages.py emits `quantum_advantage_score` again; in a full deployment that metric and its "
        "log line are what a visitor reads.")
    assert "quantum advantage {" not in src and "quantum advantage" not in src


def test_the_quantum_row_ct_pt_is_never_rendered() -> None:
    """The table column was the last place the two ratios still sat side by side."""
    src = _ts("components/BenchmarkPanel.tsx")
    assert re.search(r'path\s*===\s*"QSMPC-FHE"\s*\?\s*null', src), (
        "BenchmarkPanel no longer withholds the QSMPC-FHE ct/pt. A qubit-count ratio (or a simulated "
        "state vector's byte ratio) printed beside a CKKS byte ratio is the retired comparison again.")
    assert re.search(r"\{ctpt\(r\)\s*\?\?", src), "the table cell must go through the same rule"


def test_the_agent_prompt_follows_the_use_case() -> None:
    """The SPA defect found in the same survey: `useState(defaultPrompt)` reads its prop once.

    After Finance -> Medical in the nav the medical page still asked the AML question. A hard load
    showed the right prompt, which is why every check that started from a fresh URL passed.
    """
    src = _ts("components/AgenticAIPanel.tsx")
    m = re.search(r"useEffect\(\s*\(\)\s*=>\s*\{(?P<body>[^}]*setPrompt\([^}]*)\}\s*,\s*\[(?P<deps>[^\]]*)\]", src)
    assert m, "AgenticAIPanel has no effect that resets the prompt when the use case changes"
    deps = m.group("deps")
    assert "usecase" in deps and "defaultPrompt" in deps, f"the reset effect depends on [{deps}]"


def test_the_readme_no_longer_makes_the_headline_claims() -> None:
    # The README and the documents its long sections moved to in Round 80.
    readme = "\n".join((REPO / f).read_text() for f in (
        "README.md", "docs/use-cases.md", "docs/findings.md", "docs/models-and-benchmarks.md",
        "docs/development.md"))
    for phrase in ("more\ncommunication-compact than classical CKKS", "more communication-compact than classical CKKS",
                   "information-theoretic (vs computational) security"):
        assert phrase not in readme, f"README makes the retired claim again: {phrase!r}"
    assert not re.search(r"1\.32x?\*{0,2} figure quoted elsewhere is the \*lit", readme), (
        "README attributes 1.32 to the literature again; the Rate-1 QFHE paper reports no figure")
