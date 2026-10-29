"""A failed request for a published model's evaluation sample is not reported as the model having none.

WHY THIS EXISTS (Round 80)

The browser fetches each published model's evaluation sample from the Hugging Face Hub. The fetch
returned `null` for a 404 and for every failure alike: offline, a timeout, an outage, a rate limit, a
body that would not parse. The page then stated that the published model "ships no eval_sample.json
at that revision", and the verification logged "no published evaluation sample". Both are claims
about the artefact made with no answer from the Hub. The null was cached, so the claim stood for the
rest of the session. Found by scanning every page with the network cut, which is how CI scans them.

`web/scripts/check_sample_fetch.mjs` bundles the real module with esbuild, replaces `fetch`, and
asserts that each kind of failure is reported as a failed fetch, with its reason, by both readers,
and asked again. A 404 must still read as no sample, and be kept. This file runs it from the Python
suite, the job where CI sets `QSH_REQUIRE_BROWSER_GATES=1`, so the check cannot pass by being skipped.
The seeded negative is the module before the fix: run against it, the check fails on every kind.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
SCRIPT = WEB / "scripts" / "check_sample_fetch.mjs"
MODULE = WEB / "src" / "lib" / "onnxInference.ts"


def _require_browser_or_skip() -> None:
    """Skip without the toolchain, unless the environment says a skip is a failure (CI does)."""
    missing = [name for name, ok in (("web/node_modules/.bin/esbuild", ESBUILD.exists()),
                                     ("node", shutil.which("node") is not None)) if not ok]
    if not missing:
        return
    msg = (f"{' and '.join(missing)} absent, so the sample-fetch check cannot run. Run `npm ci` in "
           "web/. This is a SKIP, not a pass.")
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


def _run(module: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if module is not None:
        env["CHECK_MODULE"] = str(module.with_suffix(""))
    return subprocess.run(["node", str(SCRIPT)], cwd=WEB, capture_output=True, text=True,
                          timeout=300, env=env)


def test_a_failed_fetch_is_reported_as_one_and_asked_again() -> None:
    _require_browser_or_skip()
    r = _run()
    assert r.returncode == 0, f"check:sample-fetch failed:\n{r.stdout}\n{r.stderr}"
    assert "check:sample-fetch OK" in r.stdout, r.stdout


def test_the_check_fails_on_the_conflation_it_exists_for() -> None:
    """The seeded negative: the fetch as it was before Round 80, every failure read as `null`."""
    _require_browser_or_skip()
    src = MODULE.read_text()
    old_fetch = src[src.index("function evalSample("):src.index("/** Said wherever a sample")]
    conflated = (
        "function evalSample(modelUrl: string): Promise<SampleFetch> {\n"
        "  const url = sampleUrlFor(modelUrl);\n"
        "  let hit = _sampleCache.get(url);\n"
        "  if (!hit) {\n"
        "    hit = (async (): Promise<SampleFetch> => {\n"
        "      try {\n"
        "        const res = await fetch(url, { mode: \"cors\" });\n"
        "        if (!res.ok) return { kind: \"absent\" };\n"
        "        return { kind: \"ok\", file: (await res.json()) as ModelInputFile };\n"
        "      } catch {\n"
        "        return { kind: \"absent\" };\n"
        "      }\n"
        "    })();\n"
        "    _sampleCache.set(url, hit);\n"
        "  }\n"
        "  return hit;\n"
        "}\n\n"
    )
    seeded = MODULE.with_name("zz_seeded_onnxInference.ts")
    try:
        seeded.write_text(src.replace(old_fetch, conflated))
        r = _run(seeded)
    finally:
        seeded.unlink(missing_ok=True)
    out = r.stdout + r.stderr
    assert r.returncode == 1, f"the conflated fetch passed the check:\n{out}"
    for kind in ("a network error", "a timeout", "an HTTP 503", "a body that does not parse"):
        assert f"{kind}: the one-row run reported a failed request as the model shipping no sample" in out, out
