"""A module's `__main__` guard is its last statement, and the pipeline CLI runs its default use case (Round 86).

WHY THIS EXISTS (the documentation audit of 2026-11-22)

`python -m qsh.orchestration.pipeline`, whose `--usecase` defaults to `demo`, stopped with
`NameError: name '_prune_runs' is not defined`: the module's `if __name__ == "__main__": raise SystemExit(main())`
stood above the definition of `_prune_runs`, which the run calls when it writes its results, so the call came
before Python had read the definition. `--usecase finance_aml`, the command the documents quote, completed, because
its pipeline is built through the imported module, where the definition exists; the default builds it in the
half-read `__main__`. The guard now ends the module.

WHAT IS ASSERTED

- In every tracked module under src/, scripts/ and api/, nothing but docstrings follows the `__main__` guard.
- Seeded: the guard above a definition, as it stood, is found.
- The default CLI run completes into a temporary folder and writes its summary.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _after_the_guard(source: str) -> list[int]:
    """Line numbers of the statements after a module's `__main__` guard (docstrings aside)."""
    body = ast.parse(source).body
    for i, node in enumerate(body):
        test = getattr(node, "test", None)
        if isinstance(node, ast.If) and isinstance(test, ast.Compare) and getattr(test.left, "id", None) == "__name__":
            return [n.lineno for n in body[i + 1:] if not isinstance(n, ast.Expr)]
    return []


def test_no_module_has_code_after_its_main_guard() -> None:
    files = subprocess.run(["git", "ls-files", "src/*.py", "scripts/*.py", "api/*.py"], cwd=REPO, check=True,
                           capture_output=True, text=True).stdout.split()
    assert len(files) > 50, files
    late = {f: lines for f in files if (lines := _after_the_guard((REPO / f).read_text(encoding="utf-8")))}
    assert late == {}, f"code after a __main__ guard runs after main(): {late}"


def test_the_check_finds_the_guard_as_it_stood() -> None:
    as_it_stood = ('def main():\n    return _prune()\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n\n\n'
                   'def _prune():\n    return 0\n')
    assert _after_the_guard(as_it_stood) == [9]


def test_the_pipeline_cli_runs_its_default_use_case(tmp_path: Path) -> None:
    # The tree under test, not whichever checkout the environment installed in editable mode.
    path = os.pathsep.join([str(REPO / "src"), str(REPO), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONPATH": path}
    proc = subprocess.run([sys.executable, "-m", "qsh.orchestration.pipeline", "--out", str(tmp_path)], cwd=REPO,
                          env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "NameError" not in proc.stderr
    assert list(tmp_path.glob("*summary.json")), sorted(p.name for p in tmp_path.iterdir())
