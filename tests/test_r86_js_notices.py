"""The npm packages bundled into the page carry their notices (Round 86).

MIT, ISC, BSD and Apache-2.0 ask that their notice travel with copies of the code. Minification keeps only the comments a
package marks for it, and a built page carried React's, Three.js's and ONNX Runtime's and no other; the wasm crates and
the WebLLM libraries had notice files and the 51 npm packages in the JavaScript did not. `web/vite.config.ts` now writes
`licenses/js-bundle.txt` from the modules the bundle contains, the workers' included, and `web/scripts/check_js_notices.mjs`
reads the built file in both pipelines. Held here:

1. the config traces the page's chunks and the workers' and writes the file, and the four packages that ship no licence
   file take theirs from their repositories;
2. the gate is in both pipelines and the sidebar links the file;
3. the gate passes a complete file and fails one that drops a package, a declared licence or a licence text.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
GATE = WEB / "scripts" / "check_js_notices.mjs"
REQUIRED = ["react", "react-dom", "@noble/post-quantum", "@noble/curves", "@noble/hashes", "node-seal",
            "onnxruntime-web", "recharts", "@xyflow/react", "three", "html-to-image", "gif.js", "react-router"]
BAR = "=" * 78


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def test_the_config_traces_the_page_and_the_workers_and_writes_the_file() -> None:
    cfg = _read("web/vite.config.ts")
    assert 'fileName: "licenses/js-bundle.txt"' in cfg
    assert "thirdPartyNotices(true)" in cfg and "plugins: () => [thirdPartyNotices(false)]" in cfg, \
        "the page's plugin writes the file and the workers' plugin only traces"
    for pkg in ("gif.js", "onnxruntime-common", "onnxruntime-web", "victory-vendor"):
        assert f'"{pkg}"' in cfg, f"{pkg} ships no licence file and has no notice from its repository"


def test_the_gate_runs_in_both_pipelines_and_the_sidebar_links_the_file() -> None:
    assert '"check:js-notices": "node scripts/check_js_notices.mjs"' in _read("web/package.json")
    assert "npm run --silent check:js-notices" in _read("scripts/ci_local.sh")
    assert "run: npm run check:js-notices" in _read(".github/workflows/ci.yml")
    assert "licenses/js-bundle.txt" in _read("web/src/App.tsx")


def _entry(name: str, licence: str = "MIT", text: str = "x" * 200) -> str:
    return f"{BAR}\n{name} 1.0.0\nDeclared licence: {licence}\nFrom its LICENSE:\n\n{text}\n"


def _run(tmp_path: Path, body: str) -> subprocess.CompletedProcess:
    f = tmp_path / "js-bundle.txt"
    f.write_text("Third-party notices\n\n" + body)
    return subprocess.run(["node", str(GATE), str(f)], capture_output=True, text=True, timeout=60)


@pytest.fixture
def node_or_skip() -> None:
    if shutil.which("node") is None:
        pytest.skip("node absent, so the gate cannot run here. This is a SKIP, not a pass.")


def test_the_gate_passes_a_complete_file(tmp_path: Path, node_or_skip) -> None:
    names = REQUIRED + [f"pkg-{i}" for i in range(40 - len(REQUIRED))]
    r = _run(tmp_path, "".join(_entry(n) for n in names))
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("fault", ["a package dropped", "no licence declared", "no licence text"])
def test_the_gate_fails_a_file_with_a_fault(tmp_path: Path, node_or_skip, fault: str) -> None:
    names = REQUIRED + [f"pkg-{i}" for i in range(40 - len(REQUIRED))]
    entries = {n: _entry(n) for n in names}
    if fault == "a package dropped":
        del entries["node-seal"]
        entries["pkg-extra"] = _entry("pkg-extra")
    elif fault == "no licence declared":
        entries["recharts"] = _entry("recharts", licence="none declared")
    else:
        entries["three"] = _entry("three", text="")
    r = _run(tmp_path, "".join(entries.values()))
    assert r.returncode == 1, (fault, r.stdout, r.stderr)
