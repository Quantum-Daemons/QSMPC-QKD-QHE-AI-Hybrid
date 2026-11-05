"""`make check-secrets` reads file CONTENTS, not only file names.

WHY THIS EXISTS (2026-10-29)

The check used to look at tracked file NAMES only, and passed for months while three things leaked
through contents: the demo's host name in a compose comment and the verification log, a local home
path (whose user name is the deploy login) inside two tracked LaTeX build files, and the private
notes named in three prose documents. They were removed from the whole git history the same day,
and `scripts/check_secrets.sh` gained content checks. This seeds each leak in a throwaway repository
and requires the check to fail on it, and to pass on the clean repository and on the placeholder.

The seeded strings are built at run time, so this file itself carries no home path for the real
check to find.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_secrets.sh"
HOME = "/" + "home" + "/"


def _repo(tmp_path: Path, files: dict[str, str], env_demo: str | None = None) -> subprocess.CompletedProcess[str]:
    if shutil.which("git") is None:
        pytest.skip("git absent")
    (tmp_path / "scripts").mkdir()
    shutil.copy(SCRIPT, tmp_path / "scripts" / "check_secrets.sh")
    for name, text in {"README.md": "a public readme\n", **files}.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    if env_demo is not None:
        (tmp_path / ".env.demo").write_text(env_demo)
    run = lambda *a: subprocess.run(a, cwd=tmp_path, capture_output=True, text=True)  # noqa: E731
    run("git", "init", "-q")
    run("git", "add", "README.md", "scripts", *files)
    return subprocess.run(["bash", "scripts/check_secrets.sh"], cwd=tmp_path, capture_output=True, text=True)


def _fresh(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.mkdir()
    return path


def test_a_clean_repository_passes(tmp_path: Path) -> None:
    r = _repo(tmp_path, {"docs/a.md": "see " + HOME + "user/project for the layout\n"})
    assert r.returncode == 0, r.stdout


def test_an_absolute_home_path_fails(tmp_path: Path) -> None:
    r = _repo(tmp_path, {"report/out/x.fls": "PWD " + HOME + "alice/build\n"})
    assert r.returncode == 1 and "absolute home path" in r.stdout and "report/out/x.fls" in r.stdout, r.stdout


def test_naming_a_local_only_file_fails_and_is_not_printed(tmp_path: Path) -> None:
    """The names come from the git-ignored env file, so this repository needs to carry none of them."""
    r = _repo(tmp_path, {"docs/b.md": "Background for `LOCAL-ONLY-NOTE.md` only.\n"},
              env_demo="QSH_PRIVATE_NAMES=LOCAL-ONLY-NOTE\n")
    assert r.returncode == 1 and "value from .env.demo appears in tracked files: docs/b.md" in r.stdout, r.stdout
    assert "LOCAL-ONLY-NOTE" not in r.stdout, r.stdout


def test_an_unexpected_markdown_file_at_the_root_fails(tmp_path: Path) -> None:
    r = _repo(_fresh(tmp_path, "a"), {"PLAN.md": "kept locally\n", "docs/fine.md": "a document\n"})
    assert r.returncode == 1 and "unexpected Markdown file is tracked at the root: PLAN.md" in r.stdout, r.stdout
    r = _repo(_fresh(tmp_path, "b"), {"SECURITY.md": "report here\n", "CONTRIBUTING.md": "how\n", "DATA_LICENSES.md": "x\n"})
    assert r.returncode == 0, r.stdout


def test_the_gitignore_keeps_other_root_markdown_local_without_naming_it() -> None:
    repo = Path(__file__).resolve().parents[1]
    run = lambda *paths: subprocess.run(["git", "check-ignore", "--no-index", "-q", *paths], cwd=repo).returncode  # noqa: E731
    assert run("SOME-LOCAL-NOTE.md") == 0, "a Markdown file at the root is not ignored by default"
    for kept in ("README.md", "DATA_LICENSES.md", "SECURITY.md", "CONTRIBUTING.md", "CHANGELOG.md", "docs/x.md"):
        assert run(kept) == 1, f"{kept} would be ignored"


def test_a_deploy_value_from_the_env_file_fails_and_is_not_printed(tmp_path: Path) -> None:
    box = "demo-box.example.test"
    r = _repo(tmp_path, {"docs/c.md": f"Observed on https://{box}/usecase\n"},
              env_demo=f"QSH_DEMO_URL=https://{box}\nQSH_DEMO_SSH=deployer@{box}\n")
    assert r.returncode == 1 and "value from .env.demo appears in tracked files: docs/c.md" in r.stdout, r.stdout
    assert box not in r.stdout and "deployer" not in r.stdout, r.stdout


def test_a_sibling_host_on_the_demo_domain_fails_and_is_not_printed(tmp_path: Path) -> None:
    """A page once linked a sibling site under the demo's own registered domain, and the sibling resolved
    to the demo's address (2026-10-30): the domain names the box even when the demo's host does not appear."""
    r = _repo(tmp_path, {"web/src/x.tsx": "const NETWORK = 'https://sibling.example.test/';\n"},
              env_demo="QSH_DEMO_URL=https://demo-box.example.test\nQSH_DEMO_SSH=deployer@demo-box.example.test\n")
    assert r.returncode == 1 and "value from .env.demo appears in tracked files: web/src/x.tsx" in r.stdout, r.stdout
    assert "example.test" not in r.stdout and "sibling" not in r.stdout, r.stdout


def test_one_named_public_host_may_be_linked_and_nothing_else_is_excused(tmp_path: Path) -> None:
    """The business page sits under the demo's registered domain and does not resolve to the demo's box;
    the owner chose to allow exactly that host (2026-11-05). The host list stays in the env file."""
    box, page, env = "demo.example.test", "page.example.test", ""
    env = f"QSH_DEMO_URL=https://{box}\nQSH_DEMO_SSH=deployer@{box}\nQSH_PUBLIC_HOSTS={page}\n"
    ok = _repo(_fresh(tmp_path, "ok"), {"docs/c.md": f"Business page: https://{page}/\n"}, env_demo=env)
    assert ok.returncode == 0, ok.stdout
    for i, leak in enumerate((f"https://{box}/", "https://other.example.test/",
                              f"https://{page}/ and https://other.example.test/")):
        r = _repo(_fresh(tmp_path, str(i)), {"docs/c.md": f"See {leak}\n"}, env_demo=env)
        assert r.returncode == 1 and "docs/c.md" in r.stdout, (leak, r.stdout)
        assert "example.test" not in r.stdout, r.stdout



def test_a_generic_second_level_does_not_widen_the_domain(tmp_path: Path) -> None:
    """Under co.jp the registered domain is three labels: another company's co.jp site is not the demo's."""
    r = _repo(tmp_path, {"docs/e.md": "Compare https://other.co.jp/ for context.\n"},
              env_demo="QSH_DEMO_URL=https://demo.ourfirm.co.jp\nQSH_DEMO_SSH=deployer@demo.ourfirm.co.jp\n")
    assert r.returncode == 0, r.stdout
