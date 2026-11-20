"""`scripts/deploy_demo.sh` verifies the box it deployed, not whatever the public name resolves to.

WHY THIS EXISTS

On 2026-11-19 the public names pointed at a temporary host while the usual host's provider ran maintenance, and the
usual host was to be redeployed before the names moved back (checklist DP18). The script read "200, serving <bundle>"
through the public URL alone, so a deploy to the usual host would have been verified against the temporary host: a
PASS describing another machine. It now reads the public name through a resolve pinned to the box's own address (the
one ssh connects to, from `ssh -G`, or QSH_DEMO_ADDRESS), fails on that read, and reports the name as DNS resolves it
beside it, as a note when the two differ.

These tests run the script with `ssh` and `curl` replaced by stubs, so no host is contacted.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "deploy_demo.sh"
ADDRESS = "192.0.2.10"  # TEST-NET-1 (RFC 5737): documentation only
NAME = "qsh.example.test"

SSH_STUB = """#!/usr/bin/env bash
if [ "$1" = "-G" ]; then printf 'user stub\\nhostname %s\\nport 22\\n' "$STUB_SSH_HOSTNAME"; exit 0; fi
cmd="${@: -1}"
case "$cmd" in
  *"rev-parse --short HEAD"*) echo abc1234 ;;
  *"rev-parse --abbrev-ref HEAD"*) echo main ;;
  *"docker inspect"*) echo "proxy-net " ;;
  *) : ;;
esac
"""

CURL_STUB = """#!/usr/bin/env bash
mode=public; resolve=""; want_code=0; dump=0; url=""
while [ $# -gt 0 ]; do
  case "$1" in
    --resolve) resolve="$2"; mode=box; shift ;;
    -w) want_code=1; shift ;;
    -D) dump=1; shift ;;
    -o) shift ;;
    http*) url="$1" ;;
  esac
  shift
done
echo "$mode $resolve $dump" >> "$STUB_LOG"
if [ "$dump" = 1 ]; then
  if [ -n "${STUB_WALK_HOST:-}" ] && [[ "$url" == *"//$STUB_WALK_HOST"* ]]; then
    case "$url" in *"/usecase/"*) printf '%b' "$STUB_WALKAPP_HEADERS" ;; *) printf '%b' "$STUB_WALKORIGIN_HEADERS" ;; esac
    exit 0
  fi
  case "$url" in *"/walkthrough/"*) printf '%b' "$STUB_WALK_HEADERS" ;; *"-missing-"*) printf '%b' "$STUB_MISSING_HEADERS" ;;
    *) printf '%b' "$STUB_APP_HEADERS" ;; esac
  exit 0
fi
if [ "$mode" = box ]; then code="$STUB_BOX_CODE"; bundle="$STUB_BOX_BUNDLE"; else code="$STUB_PUB_CODE"; bundle="$STUB_PUB_BUNDLE"; fi
if [ "$want_code" = 1 ]; then printf '%s' "$code"; else printf '<script type="module" src="%s"></script>' "$bundle"; fi
"""

CADDYFILE = (REPO / "docker" / "env" / "Caddyfile.client-only").read_text()


def _caddy(pattern: str) -> str:
    m = re.search(pattern, CADDYFILE, re.M)
    assert m, pattern
    return m.group(1)


def _headers(drop: str = "", change: tuple[str, str] | None = None) -> tuple[str, str]:
    """The app's and the walkthrough's response headers as the Caddyfile writes them, with one dropped or changed."""
    app = {"Content-Security-Policy": _caddy(r'^\theader @app_csp >Content-Security-Policy "([^"]+)"$')}
    # A Report-Only candidate only while one is pending (none since 2026-11-20).
    ro = re.search(r'^\theader @app_csp >Content-Security-Policy-Report-Only "([^"]+)"$', CADDYFILE, re.M)
    if ro:
        app["Content-Security-Policy-Report-Only"] = ro.group(1)
    for name in ("Strict-Transport-Security", "X-Content-Type-Options", "Referrer-Policy", "Cross-Origin-Opener-Policy",
                 "Cross-Origin-Embedder-Policy", "Cross-Origin-Resource-Policy"):
        app[name] = _caddy(rf'^\t\t>{name} "([^"]+)"$')
    walk = {"Content-Security-Policy": _caddy(r'^\theader @walkthrough_csp >Content-Security-Policy "([^"]+)"$')}
    # The walkthrough gets the static headers too, which the deploy checks there as well (2026-11-20).
    walk.update({name: app[name] for name in STATIC})
    app.pop(drop, None)
    if change:
        app[change[0]] = change[1]
    fmt = lambda h: "HTTP/2 200\\r\\n" + "".join(f"{k.lower()}: {v}\\r\\n" for k, v in h.items()) + "\\r\\n"
    return fmt(app), fmt(walk)


STATIC = ("Strict-Transport-Security", "X-Content-Type-Options", "Referrer-Policy", "Cross-Origin-Opener-Policy",
          "Cross-Origin-Embedder-Policy", "Cross-Origin-Resource-Policy")


def _missing(drop: str = "", status: str = "404") -> str:
    """A missing file's response: Caddy's error response, which handle_errors gives the static headers, no policy."""
    h = {name: _caddy(rf'^\t\t>{name} "([^"]+)"$') for name in STATIC}
    h.pop(drop, None)
    return f"HTTP/2 {status}\\r\\n" + "".join(f"{k.lower()}: {v}\\r\\n" for k, v in h.items()) + "\\r\\n"


def _run(tmp_path: Path, box=("200", "/assets/index-NEW.js"), public=("200", "/assets/index-NEW.js"),
         address_override: str | None = None, headers: tuple[str, str] | None = None,
         missing: str | None = None, extra: dict[str, str] | None = None) -> tuple[subprocess.CompletedProcess, list[str]]:
    stubs = tmp_path / "bin"
    stubs.mkdir(exist_ok=True)
    for name, body in (("ssh", SSH_STUB), ("curl", CURL_STUB)):
        f = stubs / name
        f.write_text(body)
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "curl.log"
    log.write_text("")
    env = {
        "PATH": f"{stubs}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "QSH_DEMO_SSH": "deploy@demo-alias",
        "QSH_DEMO_URL": f"https://{NAME}",
        "QSH_DEMO_DIR": "/srv/qsh",
        "PROXY_NETWORK": "proxy-net",
        "STUB_SSH_HOSTNAME": ADDRESS,
        "STUB_LOG": str(log),
        "STUB_BOX_CODE": box[0], "STUB_BOX_BUNDLE": box[1],
        "STUB_PUB_CODE": public[0], "STUB_PUB_BUNDLE": public[1],
    }
    env["STUB_APP_HEADERS"], env["STUB_WALK_HEADERS"] = headers or _headers()
    env["STUB_MISSING_HEADERS"] = missing or _missing()
    env.update(extra or {})
    if address_override:
        env["QSH_DEMO_ADDRESS"] = address_override
    res = subprocess.run(["bash", str(SCRIPT), "--return-to-main"], capture_output=True, text=True, env=env,
                         timeout=60)
    return res, log.read_text().split("\n")


def test_it_reads_the_box_through_a_resolve_pinned_to_the_address_ssh_uses(tmp_path: Path) -> None:
    res, calls = _run(tmp_path)
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    assert "the box serves the public name: 200, /assets/index-NEW.js" in out
    assert "gets the same build" in out
    assert any(c.startswith(f"box {NAME}:443:{ADDRESS} ") for c in calls), calls
    assert any(c.startswith("public") for c in calls), "the name as DNS resolves it is read too"
    assert ADDRESS not in out, "the box's address must never be printed"


def test_a_name_that_reaches_another_machine_is_a_note_not_a_pass_or_a_failure(tmp_path: Path) -> None:
    res, _ = _run(tmp_path, public=("200", "/assets/index-OLD.js"))
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    assert "the box serves the public name: 200, /assets/index-NEW.js" in out
    assert "NOTE  the public name, as DNS resolves it here, gets 200 /assets/index-OLD.js" in out
    assert "/assets/index-NEW.js (DP18)" in out
    assert "gets the same build" not in out


def test_a_broken_box_fails_even_when_the_public_name_answers(tmp_path: Path) -> None:
    """The failure the old check could not see: the public name answered from another machine."""
    res, _ = _run(tmp_path, box=("502", ""), public=("200", "/assets/index-OLD.js"))
    assert res.returncode == 1
    assert "the box returned 502 for the public name after returning to main" in res.stderr


def test_an_explicit_address_overrides_the_one_ssh_uses(tmp_path: Path) -> None:
    res, calls = _run(tmp_path, address_override="198.51.100.7")  # TEST-NET-2
    assert res.returncode == 0, res.stdout + res.stderr
    assert any(c.startswith(f"box {NAME}:443:198.51.100.7 ") for c in calls), calls
    assert "198.51.100.7" not in res.stdout + res.stderr


def test_both_halves_verify_the_same_way() -> None:
    src = SCRIPT.read_text()
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    assert code.count('verify_served "') == 2, "the deploy half and the return half must both read the box"
    assert 'curl -s -o /dev/null -w \'%{http_code}\' "$QSH_DEMO_URL' not in code.split("verify_served()")[0], (
        "no unpinned public read may decide a PASS outside verify_served")



def test_the_security_headers_are_read_on_the_box_and_match_the_caddyfile(tmp_path: Path) -> None:
    """2026-11-20, the owner's choice: every deploy asserts the headers, through the same pinned resolve."""
    res, calls = _run(tmp_path)
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    assert "the box sends the Caddyfile's security headers" in out
    # the app, the walkthrough and a missing file, each pinned
    assert sum(1 for c in calls if c == f"box {NAME}:443:{ADDRESS} 1") == 3, calls


def test_a_missing_policy_fails(tmp_path: Path) -> None:
    res, _ = _run(tmp_path, headers=_headers(drop="Content-Security-Policy"))
    assert res.returncode == 1 and "Content-Security-Policy for the app is not the Caddyfile's" in res.stderr


def test_a_changed_policy_fails(tmp_path: Path) -> None:
    looser = _caddy(r'^\theader @app_csp >Content-Security-Policy "([^"]+)"$').replace("'wasm-unsafe-eval'", "'unsafe-eval'")
    res, _ = _run(tmp_path, headers=_headers(change=("Content-Security-Policy", looser)))
    assert res.returncode == 1 and "is not the Caddyfile's" in res.stderr


def test_a_report_only_header_the_caddyfile_does_not_set_fails(tmp_path: Path) -> None:
    """With no candidate pending, a Report-Only header from the box comes from an older image: a stale deploy."""
    assert "Content-Security-Policy-Report-Only" not in CADDYFILE
    res, _ = _run(tmp_path, headers=_headers(change=("Content-Security-Policy-Report-Only", "default-src 'self'")))
    assert res.returncode == 1 and "that the Caddyfile in this checkout does not set" in res.stderr


def test_a_missing_hsts_fails(tmp_path: Path) -> None:
    res, _ = _run(tmp_path, headers=_headers(drop="Strict-Transport-Security"))
    assert res.returncode == 1 and "does not send Strict-Transport-Security" in res.stderr


def test_a_missing_files_404_must_carry_the_headers(tmp_path: Path) -> None:
    """2026-11-20: a missing file's 404 is Caddy's error response, which carried none of them until handle_errors."""
    res, _ = _run(tmp_path, missing=_missing(drop="Strict-Transport-Security"))
    assert res.returncode == 1 and "a missing file's 404 does not carry Strict-Transport-Security" in res.stderr


def test_a_missing_file_must_be_a_404(tmp_path: Path) -> None:
    """The SPA shell answers app routes only (Round 74); a file that is not there is a 404, not index.html."""
    res, _ = _run(tmp_path, missing=_missing(status="200"))
    assert res.returncode == 1 and "a missing file did not get a 404" in res.stderr


def test_a_server_header_is_a_note_not_a_failure(tmp_path: Path) -> None:
    """The Caddyfile drops Server; one that arrives was added by a proxy in front, whose configuration is not here."""
    res, _ = _run(tmp_path)
    assert res.returncode == 0 and "NOTE" not in res.stdout.replace("NOTE  the public name", ""), res.stdout
    res, _ = _run(tmp_path, headers=_headers(change=("Server", "Caddy")))
    assert res.returncode == 0, res.stdout + res.stderr
    assert 'NOTE  the box sends "Server: Caddy", which this Caddyfile drops' in res.stdout


def test_both_halves_verify_the_headers() -> None:
    code = "\n".join(line for line in SCRIPT.read_text().splitlines() if not line.lstrip().startswith("#"))
    assert code.count('verify_headers "') == 2, "the deploy half and the return half must both read the headers"


WALK = "https://walk.example"


def _split(moved_status: str = "301", moved_csp: str = "app_csp", walk_app_status: str = "404") -> dict[str, str]:
    """The responses of a box whose walkthrough is on its own origin (2026-11-20, the owner's choice)."""
    app_csp = _caddy(r'^\theader @app_csp >Content-Security-Policy "([^"]+)"$')
    walk_csp = _caddy(r'^\theader @walkthrough_csp >Content-Security-Policy "([^"]+)"$')
    static = {name: _caddy(rf'^\t\t>{name} "([^"]+)"$') for name in STATIC}
    fmt = lambda status, h: f"HTTP/2 {status}\\r\\n" + "".join(f"{k.lower()}: {v}\\r\\n" for k, v in h.items()) + "\\r\\n"
    moved = {"Location": f"{WALK}/?cb=1", "Content-Security-Policy": app_csp if moved_csp == "app_csp" else walk_csp, **static}
    if moved_status != "301":
        moved = {"Content-Security-Policy": walk_csp, **static}
    return {
        "QSH_WALKTHROUGH_ORIGIN": WALK, "STUB_WALK_HOST": "walk.example",
        "STUB_WALK_HEADERS": fmt(moved_status, moved),
        "STUB_WALKORIGIN_HEADERS": fmt("200", {"Content-Security-Policy": walk_csp, **static}),
        "STUB_WALKAPP_HEADERS": fmt(walk_app_status, static),
    }


def test_the_walkthrough_on_its_own_origin_is_read_there(tmp_path: Path) -> None:
    res, calls = _run(tmp_path, extra=_split())
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    assert "the walkthrough is on its own origin, under its policy" in out
    # its root and an app page, each through the box's address (the return half; the deploy half reads them too)
    assert sum(1 for c in calls if c == f"box walk.example:443:{ADDRESS} 1") == 2, calls


def test_a_walkthrough_still_on_the_app_origin_fails_the_split(tmp_path: Path) -> None:
    res, _ = _run(tmp_path, extra=_split(moved_status="200"))
    assert res.returncode == 1 and "the app's /walkthrough/ did not answer 301" in res.stderr


def test_a_redirect_with_the_walkthrough_policy_fails(tmp_path: Path) -> None:
    res, _ = _run(tmp_path, extra=_split(moved_csp="walkthrough_csp"))
    assert res.returncode == 1 and "does not carry the app's policy" in res.stderr


def test_the_app_answering_on_the_walkthrough_origin_fails(tmp_path: Path) -> None:
    res, _ = _run(tmp_path, extra=_split(walk_app_status="200"))
    assert res.returncode == 1 and "the walkthrough's origin answers an app page" in res.stderr


def test_a_walkthrough_origin_with_a_path_is_refused(tmp_path: Path) -> None:
    res, _ = _run(tmp_path, extra={**_split(), "QSH_WALKTHROUGH_ORIGIN": WALK + "/walkthrough"})
    assert res.returncode == 1 and "QSH_WALKTHROUGH_ORIGIN must be a scheme, a host and an optional port" in res.stderr


def test_the_setting_reaches_the_container() -> None:
    code = SCRIPT.read_text()
    assert 'COMPOSE="PROXY_NETWORK=$PROXY_NETWORK QSH_WALKTHROUGH_ORIGIN=$WALK_ORIGIN docker compose' in code
    compose = (REPO / "docker" / "docker-compose.demo.yml").read_text()
    assert 'QSH_WALKTHROUGH_ORIGIN: "${QSH_WALKTHROUGH_ORIGIN:-}"' in compose

