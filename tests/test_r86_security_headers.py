"""The site's security headers, as both Caddyfiles set them (Round 86, 2026-11-20, the owner's choice of an enforced CSP).

WHY THIS EXISTS. Until 2026-11-20 the demo sent COOP, COEP and CORP and nothing else: no HSTS, no X-Content-Type-Options
and no Content-Security-Policy, on either host it ran on. The owner chose to enforce a policy at once, not report-only,
so what the policy allows is a decision with a consequence, and it is held here rather than remembered:

- HSTS for a year, for this host only: other hosts share the registered domain, so no includeSubDomains or preload;
- the app's policy allows scripts and workers from this site only (with wasm compilation, never 'unsafe-eval' or
  'unsafe-inline' for scripts), and fetches to this site and to Hugging Face, which serves the pinned model files and
  redirects to its CDN;
- the walkthrough, a static marimo export with inline scripts and a chart library that compiles expressions, has its
  own looser policy, and only under /walkthrough/.

The behaviour in a browser is checked by `web/scripts/check_csp.mjs`, under a real Caddy with this file.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CADDYFILES = [REPO / "docker" / "env" / "Caddyfile.client-only", REPO / "docker" / "env" / "Caddyfile"]
#: The hosts a full walk of every page and run contacted besides this site (2026-11-20): huggingface.co, which
#: redirects a pinned file to its CDN (us.aws.cdn.hf.co on that walk).
FETCH_HOSTS = {"'self'", "https://huggingface.co", "https://*.huggingface.co", "https://*.hf.co"}


#: The two clauses the walkthrough's placement turns on (2026-11-20, the owner's choice of an origin of its own):
#: QSH_WALKTHROUGH_ORIGIN unset and a /walkthrough/ path, or that setting naming the request's own host and port.
_O = '"{$QSH_WALKTHROUGH_ORIGIN}"'
UNSPLIT = f"{_O} == \"\" && path('/walkthrough', '/walkthrough/*')"
ON_WALK = f'{_O} != "" && {_O}.endsWith("//" + {{http.request.hostport}})'


def _matcher(text: str, name: str) -> str:
    m = re.search(rf"^\t@{name} expression `([^`]+)`$", text, re.M)
    assert m, f"no expression matcher @{name}"
    return m.group(1)


def _policy(text: str, matcher: str) -> dict[str, list[str]]:
    m = re.search(rf'^\theader @{matcher} >Content-Security-Policy "([^"]+)"$', text, re.M)
    assert m, f"no enforced Content-Security-Policy for @{matcher}"
    out: dict[str, list[str]] = {}
    for part in m.group(1).split(";"):
        words = part.split()
        if words:
            out[words[0]] = words[1:]
    return out


@pytest.mark.parametrize("path", CADDYFILES, ids=lambda p: p.name)
def test_hsts_nosniff_and_the_referrer_policy(path: Path) -> None:
    text = path.read_text()
    hsts = re.findall(r"^\t+>?Strict-Transport-Security (.+)$", text, re.M)
    # Exactly this value wherever it is set (the top-level block and, since 2026-11-20, handle_errors): includeSubDomains
    # or preload would reach the other hosts on the domain.
    assert hsts and set(hsts) == {'"max-age=31536000"'}, f"{path.name}: HSTS for a year, this host only, not {hsts}"
    assert '\t\t>X-Content-Type-Options "nosniff"\n' in text, path.name
    assert '\t\t>Referrer-Policy "strict-origin-when-cross-origin"\n' in text, path.name
    # The owner chose an enforced policy. A Report-Only header is only ever a stricter candidate beside it, sent before a
    # tightening (docs/deployment.md); none is pending since style-src 'self' was enforced on 2026-11-20.
    assert text.count(">Content-Security-Policy-Report-Only ") == 0, f"{path.name}: a Report-Only header with no candidate pending"
    # No Server header (2026-11-20), dropped in the same deferred block as HSTS, so on the same responses (below).
    block = text[text.index("\theader {\n\t\t>Strict-Transport-Security"):]
    assert "\t\t-Server\n" in block[:block.index("\t}\n")], f"{path.name}: the Server header is not dropped"


@pytest.mark.parametrize("path", CADDYFILES, ids=lambda p: p.name)
def test_the_app_policy_is_strict_for_scripts_and_names_every_host_it_fetches(path: Path) -> None:
    text = path.read_text()
    assert _matcher(text, "app_csp") == f"!({UNSPLIT}) && !({ON_WALK})", f"{path.name}: the app policy covers all but the walkthrough"
    p = _policy(text, "app_csp")
    assert p["default-src"] == ["'self'"]
    assert p["script-src"] == ["'self'", "'wasm-unsafe-eval'"], p["script-src"]
    assert set(p["connect-src"]) == FETCH_HOSTS, p["connect-src"]
    assert p["worker-src"] == ["'self'", "blob:"] and p["object-src"] == ["'none'"] and p["base-uri"] == ["'self'"]
    assert p["frame-ancestors"] == ["'self'"] and p["form-action"] == ["'self'"]


@pytest.mark.parametrize("path", CADDYFILES, ids=lambda p: p.name)
def test_the_walkthrough_policy_is_scoped_to_the_walkthrough(path: Path) -> None:
    text = path.read_text()
    assert _matcher(text, "walkthrough_csp") == f"({UNSPLIT}) || ({ON_WALK})", path.name
    p = _policy(text, "walkthrough_csp")
    assert p["connect-src"] == ["'self'"], "the walkthrough loads nothing from another origin"
    assert "'unsafe-eval'" in p["script-src"] and "'unsafe-inline'" in p["script-src"]
    assert p["object-src"] == ["'none'"]


def test_both_caddyfiles_carry_the_same_headers() -> None:
    def block(path: Path) -> str:
        t = path.read_text()
        return t[t.index("\t# Security headers"):t.index("\t@wasm path *.wasm")]
    assert block(CADDYFILES[0]) == block(CADDYFILES[1])


@pytest.mark.parametrize("path", CADDYFILES, ids=lambda p: p.name)
def test_the_headers_are_top_level(path: Path) -> None:
    """Outside every handle, so the routes rewritten to index.html, the files, and the /api and /metrics 404s all carry them.

    A missing file's 404 is file_server's error, and Caddy writes error responses past the deferred header blocks: it
    carried none of them and did carry `Server: Caddy` (measured 2026-11-20). handle_errors gives it the same static
    headers (the test below).
    """
    text = path.read_text()
    assert text.index("header @app_csp") < text.index("\thandle {"), path.name


def test_the_static_host_file_carries_the_same_headers() -> None:
    """web/public/_headers serves the build on a static host (Cloudflare Pages), so it must send what Caddy sends."""
    static = (REPO / "web" / "public" / "_headers").read_text()
    caddy = CADDYFILES[0].read_text()
    for matcher, rule in (("app_csp", "/*"), ("walkthrough_csp", "/walkthrough/*")):
        value = re.search(rf'^\theader @{matcher} >Content-Security-Policy "([^"]+)"$', caddy, re.M).group(1)
        block = static[static.index(f"\n{rule}\n"):]
        block = block[:block.find("\n\n") if "\n\n" in block else len(block)]
        assert f"  Content-Security-Policy: {value}\n" in block + "\n", f"_headers {rule}: not Caddy's policy"
    assert "/walkthrough/*\n  ! Content-Security-Policy\n" in static, "the app policy would also apply to the walkthrough"
    for line in ("Strict-Transport-Security: max-age=31536000", "X-Content-Type-Options: nosniff",
                 "Referrer-Policy: strict-origin-when-cross-origin"):
        assert f"  {line}\n" in static, line


@pytest.mark.parametrize("path", CADDYFILES, ids=lambda p: p.name)
def test_the_app_policy_allows_no_inline_style(path: Path) -> None:
    """2026-11-20, under the owner's Report-Only-first rule: the candidate with style-src 'self' reported nothing on the
    deployed site's every page, tab, run and action, nor under both agent engines in a browser with WebGPU, once the
    loading placeholder's style attributes had moved to /placeholder.css. It is the enforced policy now, here and in
    web/public/_headers, and no Report-Only header remains."""
    assert _policy(path.read_text(), "app_csp")["style-src"] == ["'self'"]
    static = (REPO / "web" / "public" / "_headers").read_text()
    assert "Report-Only" not in static


@pytest.mark.parametrize("path", CADDYFILES, ids=lambda p: p.name)
def test_error_responses_carry_the_same_headers(path: Path) -> None:
    """2026-11-20: Caddy's own error responses (a missing file's 404) get the six static headers the top-level blocks
    set, the same values, no Server header, and the same empty body with the error's status. No policy: no body."""
    text = path.read_text()
    # The full tier sets the isolation headers without `>` (it is the edge); the client-only one defers them.
    top = set(re.findall(r'^\t\t>?([A-Za-z-]+ "[^"]+")$', text, re.M))
    assert len(top) == 6, sorted(top)
    m = re.search(r"^\thandle_errors \{\n(.*?)^\t\}\n", text, re.M | re.S)
    assert m, f"{path.name}: no handle_errors block"
    inner = m.group(1)
    errs = set(re.findall(r'^\t\t\t>?([A-Za-z-]+ "[^"]+")$', inner, re.M))
    assert errs == top, f"{path.name}: {sorted(errs ^ top)}"
    assert "\t\t\t-Server\n" in inner and '\t\trespond "" {err.status_code}\n' in inner
    assert "Content-Security-Policy" not in inner


@pytest.mark.parametrize("path", CADDYFILES, ids=lambda p: p.name)
def test_the_walkthrough_moves_to_its_own_origin_on_one_setting(path: Path) -> None:
    """2026-11-20, the owner's choice: with QSH_WALKTHROUGH_ORIGIN set, the walkthrough is served at the root of that
    origin and nothing of the app is, and /walkthrough/ on any other host answers 301 there, path and query kept; unset,
    nothing changes. The two policies' matchers are complements built from the same two clauses (tested above), so
    the app's origin never sends the walkthrough's policy, not even on the redirect."""
    text = path.read_text()
    assert _matcher(text, "walk_origin") == ON_WALK
    off_walk = f'{_O} != "" && !{_O}.endsWith("//" + {{http.request.hostport}})'
    assert _matcher(text, "walk_moved") == off_walk + " && path('/walkthrough', '/walkthrough/*')"
    assert "\thandle @walk_origin {\n\t\troot * /srv/walkthrough\n\t\tfile_server\n\t}\n" in text
    # In a route, so the prefix goes before the redirect is written: in a plain handle, Caddy's directive order runs
    # redir first, and the 301 kept /walkthrough/ in its target (measured 2026-11-20).
    assert ("\thandle @walk_moved {\n\t\troute {\n\t\t\turi strip_prefix /walkthrough\n"
            "\t\t\tredir {$QSH_WALKTHROUGH_ORIGIN}{uri} 301\n\t\t}\n\t}\n") in text


def test_both_caddyfiles_place_the_walkthrough_the_same_way() -> None:
    def handles(path: Path) -> str:
        t = path.read_text()
        start = t.index("\t# The walkthrough's own origin: its files at the root, and nothing of the app.\n")
        return t[start:t.index("\t}\n\t}\n", t.index("\thandle @walk_moved {")) + 6]
    assert handles(CADDYFILES[0]) == handles(CADDYFILES[1])

