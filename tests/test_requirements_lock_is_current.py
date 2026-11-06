"""What CI and the API image install is what a lock names, and the lock is regenerated when its inputs change.

WHY THIS EXISTS (Round 86, the owner's choice). `pip install -r requirements-*.txt` resolves floors against the day's index, so two
builds of one commit could install different code, and CI could pass on versions the image never saw. The filesystem MCP server
had the same gap through `npx`, which pins one package and resolves the rest. The floors stay; a lock records what they resolved to.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

REPO = Path(__file__).resolve().parents[1]
INPUTS = ["requirements-core.txt", "requirements-api.txt", "requirements-fhe.txt", "requirements-ai.txt"]
LOCK = (REPO / "requirements.lock").read_text(encoding="utf-8")


def _digest(root: Path) -> str:
    return hashlib.sha256(b"".join((root / name).read_bytes() for name in INPUTS)).hexdigest()


def _locked() -> dict[str, str]:
    return {m[0].lower().replace("_", "-"): m[1] for m in re.findall(r"^([A-Za-z0-9_.-]+)==(\S+)", LOCK, re.M)}


def test_the_lock_was_generated_from_these_inputs() -> None:
    assert f"# inputs sha256: {_digest(REPO)}" in LOCK, "an input changed: run scripts/lock_requirements.sh"


def test_the_digest_would_notice_a_changed_input(tmp_path: Path) -> None:
    """The seeded negative: one floor moved, and the recorded digest no longer matches."""
    for name in INPUTS:
        (tmp_path / name).write_bytes((REPO / name).read_bytes())
    (tmp_path / INPUTS[0]).write_text((tmp_path / INPUTS[0]).read_text(encoding="utf-8") + "\nnumpy>=99\n", encoding="utf-8")
    assert f"# inputs sha256: {_digest(tmp_path)}" not in LOCK


def test_every_locked_package_has_hashes() -> None:
    blocks = re.split(r"^(?=[A-Za-z0-9_.-]+==)", LOCK, flags=re.M)[1:]
    assert len(blocks) == len(_locked()) > 50
    for block in blocks:
        assert "--hash=sha256:" in block, block.splitlines()[0]


def test_each_floor_is_met_by_what_is_locked() -> None:
    locked = _locked()
    for name in INPUTS:
        for line in (REPO / name).read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if not line or line.startswith("-"):
                continue
            req = Requirement(line)
            if req.marker is not None and not req.marker.evaluate({"sys_platform": "linux", "python_version": "3.12"}):
                continue
            key = req.name.lower().replace("_", "-")
            assert key in locked, f"{req.name} in {name} is not in the lock"
            public = Version(locked[key]).public
            assert req.specifier.contains(public, prereleases=True), f"{req.name}: {locked[key]} against {req.specifier}"


@pytest.mark.parametrize("path", [".github/workflows/ci.yml", "docker/Dockerfile.api"])
def test_ci_and_the_image_install_from_the_lock_with_its_hashes(path: str) -> None:
    text = (REPO / path).read_text(encoding="utf-8")
    assert "--require-hashes" in text and "-r requirements.lock" in text, path
    installs = [ln for ln in text.splitlines() if re.search(r"pip install .*-r requirements-", ln) and not ln.lstrip().startswith("#")]
    assert not installs, f"{path} still resolves floors: {installs}"
    assert "pip install --no-deps -e ." in text, f"{path}: installing the package must not resolve anything again"


def test_the_image_starts_the_locked_filesystem_server() -> None:
    from qsh.mcp import fs_interop

    package = json.loads((REPO / "docker/mcp-fs/package.json").read_text(encoding="utf-8"))
    lock = json.loads((REPO / "docker/mcp-fs/package-lock.json").read_text(encoding="utf-8"))
    name, version = fs_interop.FILESYSTEM_SERVER.rsplit("@", 1)
    assert package["dependencies"] == {name: version}, "the image and npx would run different servers"
    assert lock["packages"][f"node_modules/{name}"]["version"] == version
    image = (REPO / "docker/Dockerfile.api").read_text(encoding="utf-8")
    assert "npm ci --omit=dev --prefix /opt/mcp-fs" in image and "ENV QSH_MCP_FS_BIN=" in image


def test_the_audit_reads_the_lock_and_runs_on_a_change_to_it() -> None:
    script = (REPO / "scripts/audit_deps.sh").read_text(encoding="utf-8")
    assert "-r requirements.lock" in script and "cd docker/mcp-fs && npm audit" in script
    workflow = (REPO / ".github/workflows/audit.yml").read_text(encoding="utf-8")
    assert workflow.count('"requirements.lock"') == 2 and "audit_deps.sh python" in workflow


@pytest.mark.parametrize("path", ["docker/Dockerfile.api", "docker/Dockerfile.web", "docker/docker-compose.yml"])
def test_every_base_image_is_pinned_by_digest(path: str) -> None:
    """`caddy:2-alpine` and `prom/prometheus:latest` were different images from one build to the next."""
    text = (REPO / path).read_text(encoding="utf-8")
    images = re.findall(r"^FROM\s+(\S+)", text, re.M) + re.findall(r"^\s+image:\s+(\S+)", text, re.M)
    floating = [i for i in images if i != "scratch" and "@sha256:" not in i]
    assert not floating, f"{path}: {floating}"


def test_the_web_image_installs_its_lockfile() -> None:
    text = (REPO / "docker/Dockerfile.web").read_text(encoding="utf-8")
    assert "RUN npm ci" in text and "RUN npm install" not in text, "npm install may rewrite the lock it was given"

