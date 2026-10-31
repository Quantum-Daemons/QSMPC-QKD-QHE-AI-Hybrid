"""The BB84 slider survives an aborted run, and a failed validation gives its button back.

WHAT WAS WRONG (Round 84, scratchpad audit83/page-overview-value-validations.md, finding 3)

1. In the full deployment, `/api/demo/qkd` answered HTTP 500 for every intercept fraction from 0.45 up
   at the page's 50 km. Above the 0.11 post-processing threshold `run_bb84` aborts and returns
   `block_pulses=None`, and `api/inference_service.py` formatted it with `f"{None:.6g}"`, a TypeError.
   The browser twin in `api.ts` maps the absent block to 0 (`?? 0`), so only the server branch broke,
   and the existing tests drove the endpoint at Eve = 0 alone.
2. On /validations each server-reaching button ran `setBusy(x); set(await call()); setBusy("")`, so a
   call that threw skipped the reset and left the button disabled until a reload. After the 500 above,
   "Run BB84" never came back.

WHAT IS ASSERTED

- `qkd_demo` and the FastAPI route answer every fraction from 0.45 to 1.0, with the payload shape of a
  successful run and the abort's values (no key, block 0, eavesdropper flagged).
- `runValidation`, the page's one runner, is driven in node with a task that rejects: the busy flag is
  set and then cleared, the failure is recorded, and the call itself resolves. A succeeding task
  clears a previous failure.
- Every server-reaching call on the page goes through that runner, and no handler resets `busy` itself.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
VALIDATIONS = WEB / "src" / "pages" / "Validations.tsx"

#: The page's slider moves in steps of 0.1; 0.45 is the first fraction that aborts at 50 km (QBER
#: 0.109 at 0.40 still passes the 0.11 threshold). Both kinds of value are covered.
ABORTING = [0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _api():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from api.inference_service import qkd_demo

    return qkd_demo


def _distance_km() -> float:
    """The distance the page sends: `qkdCfg().distance_km`, i.e. `QKDConfig.distance_km`."""
    from qsh.config import get_config

    return float(get_config().qkd.distance_km)


@pytest.mark.parametrize("eve", ABORTING)
def test_the_demo_answers_an_aborted_run_instead_of_raising(eve: float) -> None:
    from qsh.config import get_config

    qkd_demo = _api()
    km = _distance_km()
    ok = qkd_demo(km, 0.0)
    r = qkd_demo(km, eve)                      # raised TypeError before the fix
    assert set(r) == set(ok), f"the aborted payload has another shape: {sorted(set(r) ^ set(ok))}"
    threshold = get_config().security_scoring.qber_secure_threshold
    assert r["qber"] > threshold, f"Eve {eve} no longer aborts at {km} km (QBER {r['qber']}); re-derive ABORTING"
    assert r["eavesdropper_likely"] is True
    assert r["secure_key_bits"] == 0 and r["skr_per_pulse"] == 0.0
    # 0, as the browser twin emits it (`Number((r.blockPulses ?? 0).toPrecision(6))`): the page then
    # leaves out "over a block of ... pulses", since no finite key was computed on any block.
    assert r["block_pulses"] == 0.0 and isinstance(r["block_pulses"], float)
    assert ok["block_pulses"] > 0, "the non-aborted run lost its block; the comparison above is empty"


def test_the_http_route_answers_200_across_the_slider() -> None:
    """Through the FastAPI handler, as the page calls it - no server process needed."""
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from api.main import app

    client = TestClient(app, raise_server_exceptions=False)
    km = _distance_km()
    for eve in [round(0.1 * i, 1) for i in range(11)] + [0.45]:
        resp = client.get("/api/demo/qkd", params={"distance_km": km, "eve": eve})
        assert resp.status_code == 200, f"Eve {eve}: HTTP {resp.status_code}: {resp.text[:300]}"
        body = resp.json()
        assert isinstance(body["block_pulses"], (int, float)), body


def test_the_browser_twin_maps_the_absent_block_the_same_way() -> None:
    """Both branches of `demoQkd` emit 0 for a run with no block, so the page reads one value."""
    api = (WEB / "src" / "lib" / "api.ts").read_text()
    assert "block_pulses: Number((r.blockPulses ?? 0).toPrecision(6))" in api, (
        "the browser branch no longer maps an absent block to 0; the server branch copies that rule")


# ----------------------------------------------------------------------------------------------------
# The runner, driven in node
# ----------------------------------------------------------------------------------------------------

_HARNESS = r"""
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const [WEB, OUT] = process.argv.slice(2);
const esbuild = await import(pathToFileURL(join(WEB, "node_modules/esbuild/lib/main.js")).href);
await esbuild.build({
  stdin: { contents: `export * as page from ${JSON.stringify(join(WEB, "src/pages/Validations"))};`,
           resolveDir: WEB, loader: "tsx" },
  bundle: true, format: "esm", platform: "node", outfile: OUT, logLevel: "error", jsx: "automatic",
  define: { "import.meta.env": '{"BASE_URL":"/"}', "process.env.NODE_ENV": '"production"' },
  plugins: [{ name: "stub-unresolvable", setup(b) {
    b.onResolve({ filter: /qsh_mpc\.js$|\?url$|\?worker$/ }, (a) => ({ path: a.path, namespace: "stub" }));
    b.onLoad({ filter: /.*/, namespace: "stub" }, () => ({ contents: "export default '';", loader: "js" }));
  } }],
});
const { page } = await import(pathToFileURL(OUT).href);
const out = { hasRunner: typeof page.runValidation === "function" };
if (out.hasRunner) {
  const log = [];
  const setBusy = (w) => log.push(["busy", w]);
  const setFailure = (w, msg) => log.push(["failure", w, msg]);
  let rejected = false;
  try {
    await page.runValidation("qkd", async () => { throw new Error("/api/demo/qkd?distance_km=50&eve=0.5 -> 500"); },
                             setBusy, setFailure);
  } catch { rejected = true; }
  out.failed = { log, rejected };
  const log2 = [];
  await page.runValidation("qkd", async () => {}, (w) => log2.push(["busy", w]),
                           (w, msg) => log2.push(["failure", w, msg]));
  out.succeeded = { log: log2 };
}
console.log(JSON.stringify(out));
"""


def _require_node_or_skip() -> None:
    missing = [n for n, ok in (("web/node_modules/.bin/esbuild", (WEB / "node_modules" / ".bin" / "esbuild").exists()),
                               ("node", shutil.which("node") is not None)) if not ok]
    if not missing:
        return
    msg = f"{' and '.join(missing)} absent; run `npm ci` in web/. This is a SKIP, not a pass."
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


def test_a_failed_run_gives_the_button_back_and_says_why(tmp_path: Path) -> None:
    _require_node_or_skip()
    script = tmp_path / "harness.mjs"
    script.write_text(_HARNESS)
    r = subprocess.run(["node", str(script), str(WEB), str(tmp_path / "bundle.mjs")], cwd=WEB,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"the harness failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["hasRunner"], "Validations.tsx exports no `runValidation`; the buttons have no shared runner"

    log, rejected = out["failed"]["log"], out["failed"]["rejected"]
    busy = [e[1] for e in log if e[0] == "busy"]
    assert busy == ["qkd", ""], f"busy went {busy}: after a failed run the button must come back"
    assert log[-1] == ["busy", ""], f"the reset is not the last thing the runner does: {log}"
    failures = [e[1:] for e in log if e[0] == "failure" and e[2] is not None]
    assert failures == [["qkd", "/api/demo/qkd?distance_km=50&eve=0.5 -> 500"]], (
        f"the failure was not recorded for the card to show: {log}")
    assert not rejected, "the runner re-threw, which leaves an unhandled rejection in an event handler"

    ok = out["succeeded"]["log"]
    assert [e[1] for e in ok if e[0] == "busy"] == ["qkd", ""]
    assert ["failure", "qkd", None] in ok and not [e for e in ok if e[0] == "failure" and e[2]], (
        "a successful run must clear the previous failure of that validation, and record none")


def _code(path: Path) -> str:
    """Comments blanked, so a comment quoting the old pattern cannot satisfy or trip the check."""
    text = path.read_text()
    return re.sub(r"/\*[\s\S]*?\*/|(?<![:\"'`])//[^\n]*", lambda m: "\n" * m.group(0).count("\n"), text)


def test_every_server_call_on_the_page_goes_through_the_runner() -> None:
    code = _code(VALIDATIONS)
    start = code.index("export async function runValidation")
    end = code.index("\n}\n", start)
    runner, rest = code[start:end], code[:start] + code[end:]
    assert "finally" in runner and 'setBusy("")' in runner, "the runner no longer resets busy in a finally"
    resets = [m.start() for m in re.finditer(r"setBusy\(", rest)]
    assert not resets, (
        f"{len(resets)} handler(s) on /validations call setBusy themselves; a throw between the two "
        "calls leaves the button disabled. Route them through runValidation.")
    calls = re.findall(r"await api\.(\w+)\(", code)
    assert sorted(set(calls)) == ["demoMcp", "demoPqc", "demoQkd"], f"server calls on the page: {calls}"
    for name in ("pqc", "qkd", "mcp"):
        assert re.search(rf'run\("{name}",\s*async', code), f"the {name} button does not use the runner"
        assert f"<RunFailed message={{failures.{name}}} />" in code, f"the {name} card shows no failure"
