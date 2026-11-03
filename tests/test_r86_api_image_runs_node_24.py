"""The API image runs a supported Node, pinned, and the filesystem server it starts can run on it.

WHY THIS EXISTS (Round 86, the round's carried item)

The full deployment's filesystem-MCP interop (`qsh.mcp.fs_interop`) starts the official
`@modelcontextprotocol/server-filesystem` with `npx`. The API image took Node from Debian trixie's
`nodejs` package, which is Node 20, past its upstream end of life on 2026-04-30; and the server's 2.x
line, npm's newest since 2026-10-19, declares `engines: node >= 24`. Round 85 therefore pinned the
server to 1.0.0. Round 86 copies Node 24.22.2 LTS out of the official image, pinned by digest, and
moves the server to 2.1.0, checked inside the built image.

WHAT IS ASSERTED

- The image has a Node stage from the official image at an exact 24.x version, pinned by digest, on
  the same Debian release as its Python base, and copies only the binary and npm from it.
- Debian's `nodejs` and `npm` are no longer installed.
- The filesystem server is pinned at an exact 2.x version, and the image's Node meets its engine floor.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO / "docker" / "Dockerfile.api"

#: The engine floor the server's 2.x line declares (`engines.node` in 2.0.0 and 2.1.0's package.json).
SERVER_2X_NODE_FLOOR = 24


def _text() -> str:
    return DOCKERFILE.read_text()


def test_node_comes_from_the_official_image_pinned_by_digest() -> None:
    m = re.search(r"^FROM node:(\d+)\.(\d+)\.(\d+)-(\w+)-slim@sha256:[0-9a-f]{64} AS node$", _text(), flags=re.M)
    assert m, "the API image has no Node stage from the official image at an exact version and digest"
    major, release = int(m.group(1)), m.group(4)
    assert major >= SERVER_2X_NODE_FLOOR, f"Node {major} is below the filesystem server's floor"
    base = re.search(r"^FROM python:[\d.]+-slim-(\w+)$", _text(), flags=re.M)
    assert base and base.group(1) == release, (
        f"the Node stage is {release} and the Python base {base and base.group(1)}: the copied binary "
        "needs the glibc of the release it was built on")
    for line in ("COPY --from=node /usr/local/bin/node /usr/local/bin/node",
                 "COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules",
                 "ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx",
                 # Node's licence travels with the binary (npm's is inside its node_modules).
                 "COPY --from=node /usr/local/LICENSE /usr/local/share/doc/node/LICENSE"):
        assert line in _text(), f"the image no longer {line!r}"


def test_debian_node_is_not_installed() -> None:
    apt = re.findall(r"apt-get install[^&]*", _text(), flags=re.S)
    assert apt, "no apt-get install found; the parse has stopped working"
    for block in apt:
        assert not re.search(r"(?<![\w-])(nodejs|npm)(?![\w-])", block), f"Debian's Node is installed again: {block}"


def test_the_filesystem_server_runs_on_the_images_node() -> None:
    from qsh.mcp import fs_interop

    m = re.fullmatch(r"@modelcontextprotocol/server-filesystem@(\d+)\.(\d+)\.(\d+)", fs_interop.FILESYSTEM_SERVER)
    assert m, fs_interop.FILESYSTEM_SERVER
    node = int(re.search(r"^FROM node:(\d+)\.", _text(), flags=re.M).group(1))
    if int(m.group(1)) >= 2:
        assert node >= SERVER_2X_NODE_FLOOR, f"server {m.group(0)} needs Node {SERVER_2X_NODE_FLOOR}+, image has {node}"
    assert int(m.group(1)) >= 2, "the server is back on the 1.x line, which the image no longer needs"
