"""The lending page's FHEVM view and table are drawn from Zama's documentation and use none of Zama's code (Round 86).

The owner's choice of 2026-11-21: a cited comparison and a modelled flow in the canvas, within the terms Zama's code is offered on.
Zama's libraries are free under BSD-3-Clause-Clear "only for development, research, prototyping, and experimentation", commercial use
needs Zama's patent licence, and the licence grants no patent rights (zama-ai/fhevm, the README's FAQ and LICENSE, read 2026-11-21). So
the project takes no Zama package and contacts no Zama host, and this holds it there:

  * the HCU figures in `web/src/lib/fhevmModel.ts` are Zama's table as read on 2026-11-21 (docs.zama.org, the HCU page);
  * no package file in the repository takes a Zama package (npm: `@zama-fhe/*`, `@fhevm/*`, `fhevm*`, `tfhe`, `node-tfhe`; Python:
    `concrete-*`, `tfhe*`);
  * no Content-Security-Policy the site serves names a Zama host, so the page could not reach one if code tried;
  * a Zama URL appears in the web code only in the model's list of sources, which the page shows as links and never fetches.

The canvas side (every modelled step placed, its own legend, the caption, the counts taken from the run) is section 10 of
`web/scripts/check_use_case_canvas.mjs`.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "web" / "src" / "lib" / "fhevmModel.ts"
#: Zama's HCU table and devnet limits, read at docs.zama.org/protocol/solidity-guides/development-guide/hcu on 2026-11-21.
ZAMA_HCU = {"euint64": {"mulScalar": 365_000, "add": 162_000, "lt": 146_000},
            "euint128": {"mulScalar": 696_000, "add": 259_000, "lt": 215_000}}
ZAMA_LIMITS = {"perTransaction": 20_000_000, "sequentialPerTransaction": 5_000_000}
ZAMA_NPM = re.compile(r"^(@zama-fhe/|@fhevm/|fhevm|tfhe$|node-tfhe$|tfhe-)")
ZAMA_PY = re.compile(r"^(concrete-|concrete_|tfhe)", re.IGNORECASE)


def _tracked(*patterns: str) -> list[Path]:
    out = subprocess.run(["git", "ls-files", *patterns], cwd=REPO, check=True, capture_output=True, text=True).stdout
    return [REPO / line for line in out.splitlines() if line]


def _int(s: str) -> int:
    return int(s.replace("_", ""))


def test_the_hcu_figures_are_zamas_table() -> None:
    text = MODEL.read_text(encoding="utf-8")
    for width, ops in ZAMA_HCU.items():
        m = re.search(rf"{width}: \{{ mulScalar: ([\d_]+), add: ([\d_]+), lt: ([\d_]+) \}}", text)
        assert m, f"no HCU row for {width} in {MODEL.name}"
        assert {"mulScalar": _int(m[1]), "add": _int(m[2]), "lt": _int(m[3])} == ops, width
    m = re.search(r"HCU_LIMIT = \{ perTransaction: ([\d_]+), sequentialPerTransaction: ([\d_]+) \}", text)
    assert m and {"perTransaction": _int(m[1]), "sequentialPerTransaction": _int(m[2])} == ZAMA_LIMITS


def test_one_check_costs_what_the_table_adds_up_to() -> None:
    """Three multiplications by a public scalar, an addition and a comparison: the figures the page prints."""
    for width, h in ZAMA_HCU.items():
        total = 3 * h["mulScalar"] + h["add"] + h["lt"]
        sequential = h["mulScalar"] + h["add"] + h["lt"]
        assert sequential <= ZAMA_LIMITS["sequentialPerTransaction"]
        assert (width, total, ZAMA_LIMITS["perTransaction"] // total) in {("euint64", 1_403_000, 14), ("euint128", 2_562_000, 7)}


def test_no_package_file_takes_a_zama_package() -> None:
    found = []
    for path in _tracked("*package.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        for field in ("dependencies", "devDependencies", "optionalDependencies", "peerDependencies"):
            found += [f"{path.relative_to(REPO)}: {n}" for n in (data.get(field) or {}) if ZAMA_NPM.match(n)]
    for path in _tracked("*package-lock.json"):
        for key in (json.loads(path.read_text(encoding="utf-8")).get("packages") or {}):
            name = key.rsplit("node_modules/", 1)[-1]
            if key and ZAMA_NPM.match(name):
                found.append(f"{path.relative_to(REPO)}: {name}")
    for path in _tracked("requirements*.txt", "pyproject.toml"):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip().strip('"').strip("'").strip(",")
            if line and ZAMA_PY.match(line):
                found.append(f"{path.relative_to(REPO)}: {line}")
    assert found == []


def test_no_served_policy_lets_the_page_reach_zama() -> None:
    policies = [p for p in _tracked("docker/env/Caddyfile*", "web/public/_headers")
                if "Content-Security-Policy" in p.read_text(encoding="utf-8")]
    assert len(policies) >= 3, policies
    for path in policies:
        for line in path.read_text(encoding="utf-8").splitlines():
            if "Content-Security-Policy" in line or "connect-src" in line:
                assert "zama" not in line.lower(), f"{path.relative_to(REPO)}: {line.strip()[:120]}"


def test_a_zama_url_is_only_a_source_the_page_links_to() -> None:
    hits = {}
    for path in _tracked("web/src/*.ts", "web/src/*.tsx"):
        urls = re.findall(r"https?://[A-Za-z0-9.-]*zama[A-Za-z0-9./#_-]*", path.read_text(encoding="utf-8"), re.IGNORECASE)
        if urls:
            hits[path.relative_to(REPO).as_posix()] = urls
    assert set(hits) <= {"web/src/lib/fhevmModel.ts"}, hits
    text = MODEL.read_text(encoding="utf-8")
    sources = text[text.index("export const FHEVM_SOURCES"):text.index("} as const;", text.index("export const FHEVM_SOURCES"))]
    assert all(u in sources for u in hits.get("web/src/lib/fhevmModel.ts", [])), "a Zama URL outside FHEVM_SOURCES"
    assert not re.search(r"\b(fetch|import)\s*\(\s*FHEVM_SOURCES", "\n".join(p.read_text(encoding="utf-8") for p in _tracked("web/src/*.ts", "web/src/*.tsx")))


@pytest.mark.parametrize("rel, needles", [
    ("web/src/components/FhevmBeside.tsx", ['kind="modelled"', "No Zama code runs in this page", "FHEVM_READ"]),
    ("web/src/pages/UseCasePage.tsx", [">This run</button>", ">FHEVM, modelled</button>", 'kind="modelled"']),
])
def test_the_card_and_the_view_say_they_are_modelled(rel: str, needles: list[str]) -> None:
    text = (REPO / rel).read_text(encoding="utf-8")
    assert [n for n in needles if n not in text] == []
