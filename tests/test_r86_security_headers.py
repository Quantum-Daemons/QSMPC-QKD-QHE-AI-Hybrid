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
    # Exactly one, and exactly this value: includeSubDomains or preload would reach the other hosts on the domain.
    assert hsts == ['"max-age=31536000"'], f"{path.name}: HSTS for a year, this host only, not {hsts}"
    assert '\t\t>X-Content-Type-Options "nosniff"\n' in text, path.name
    assert '\t\t>Referrer-Policy "strict-origin-when-cross-origin"\n' in text, path.name
    assert "Content-Security-Policy-Report-Only" not in text, f"{path.name}: the owner chose an enforced policy"


@pytest.mark.parametrize("path", CADDYFILES, ids=lambda p: p.name)
def test_the_app_policy_is_strict_for_scripts_and_names_every_host_it_fetches(path: Path) -> None:
    text = path.read_text()
    assert "\t@app_csp not path /walkthrough /walkthrough/*\n" in text, f"{path.name}: the app policy covers all but the walkthrough"
    p = _policy(text, "app_csp")
    assert p["default-src"] == ["'self'"]
    assert p["script-src"] == ["'self'", "'wasm-unsafe-eval'"], p["script-src"]
    assert set(p["connect-src"]) == FETCH_HOSTS, p["connect-src"]
    assert p["worker-src"] == ["'self'", "blob:"] and p["object-src"] == ["'none'"] and p["base-uri"] == ["'self'"]
    assert p["frame-ancestors"] == ["'self'"] and p["form-action"] == ["'self'"]


@pytest.mark.parametrize("path", CADDYFILES, ids=lambda p: p.name)
def test_the_walkthrough_policy_is_scoped_to_the_walkthrough(path: Path) -> None:
    text = path.read_text()
    assert "\t@walkthrough_csp path /walkthrough /walkthrough/*\n" in text, path.name
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
    """Outside every handle, so the routes rewritten to index.html, the files and the 404s all carry them."""
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
