"""A missing file is a 404; only an app route falls back to the single-page shell.

WHY THIS EXISTS

Both Caddyfiles served the app with `try_files {path} /index.html`, which answers EVERY missing path
with index.html and a 200, `/assets/` included. Measured on the deploy in Round 74: a request for
`/assets/definitely-missing-file.js` returned 200 `text/html`. So a page loaded before a deploy, asking
for a chunk hash that no longer exists, received HTML, and the browser reported a MIME error instead
of a 404. A missing data file failed the same way, as a JSON parse error. Either way the error hides
what is missing.

It was found while chasing a console 404 that turned out to be an old entry in a long-used tab. The
first explanation, a source map, was wrong precisely BECAUSE of this: the map path could not 404,
since nothing could.

WHAT IS ASSERTED

Both Caddyfiles (the client-only demo's and the full deployment's) rewrite to index.html only for a
path with no file extension that is not an existing file, and neither keeps the catch-all. The
behaviour was measured on a local container of the same image: routes, including unknown ones,
still get the app, so the app's own not-found page still answers a typo; existing files keep their
types; a missing chunk, data file, `favicon.ico` and `robots.txt` return 404; the page stays
cross-origin isolated.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CADDYFILES = [REPO / "docker" / "env" / "Caddyfile.client-only", REPO / "docker" / "env" / "Caddyfile"]


def _default_handle(text: str) -> str:
    """The body of the matcher-less `handle { ... }` block, the one every unmatched path reaches."""
    m = re.search(r"^\thandle \{\n(.*?)^\t\}", text, re.S | re.M)
    assert m, "no matcher-less handle block found - did the Caddyfile's layout change?"
    return m.group(1)


@pytest.mark.parametrize("path", CADDYFILES, ids=lambda p: p.name)
def test_no_catch_all_rewrites_a_missing_file_to_the_app(path: Path) -> None:
    text = "\n".join(line for line in path.read_text().splitlines() if not line.lstrip().startswith("#"))
    assert "try_files {path} /index.html" not in text, (
        f"{path.name}: `try_files {{path}} /index.html` answers every missing path, assets included, "
        "with the app shell and a 200")


@pytest.mark.parametrize("path", CADDYFILES, ids=lambda p: p.name)
def test_only_an_extensionless_route_gets_the_app_shell(path: Path) -> None:
    body = _default_handle(path.read_text())
    route = re.search(r"@route \{\n(.*?)\n\t\t\}", body, re.S)
    assert route, f"{path.name}: no @route matcher in the default handle"
    conds = route.group(1)
    assert re.search(r"not path_regexp \\\.\[A-Za-z0-9\]\+\$", conds), (
        f"{path.name}: the route matcher no longer excludes paths with a file extension")
    assert re.search(r"^\s*not file\s*$", conds, re.M), (
        f"{path.name}: the route matcher no longer excludes existing files")
    assert re.search(r"^\s*rewrite @route /index\.html\s*$", body, re.M), (
        f"{path.name}: the fallback to index.html is not limited to @route")
    assert re.search(r"^\s*file_server\s*$", body, re.M), f"{path.name}: no file_server"


def test_the_client_only_demo_still_answers_the_api_with_a_404() -> None:
    """The client-only image has no backend; the route rewrite must not swallow /api into the app."""
    text = CADDYFILES[0].read_text()
    assert re.search(r"handle /api/\* \{\n\t\trespond .* 404\n\t\}", text), "the /api 404 is gone"
