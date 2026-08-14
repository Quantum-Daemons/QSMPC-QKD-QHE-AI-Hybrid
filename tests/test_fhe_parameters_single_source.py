"""The CKKS parameter set has to be one number, and the number that runs has to be the documented one.

WHY THIS EXISTS

Before Round 38 the ring degree was written in four places and read from none of them:

  - ``src/qsh/config/defaults.py`` held ``FHEConfig.poly_modulus_degree = 8192``, which had NO
    reader anywhere in ``src/``. It was a config field that governed nothing.
  - ``src/qsh/fhe/backends.py`` built its TenSEAL context from default ARGUMENTS holding the same
    numbers. That is what the backend actually ran.
  - ``web/src/lib/clientFHE.ts`` held its own ``POLY_MODULUS_DEGREE = 8192``. That is what the
    browser actually ran.
  - ``report/build_data.py`` wrote ``"CKKS, ring degree 8192"`` and ``"4096 slots"`` as literals
    INSIDE the generator, so the published PDF asserted a parameter set nothing consulted.

Four copies agreeing only because nobody had yet changed one. The failure mode is not that they
drift by accident - it is that changing the ring is a five-site edit where missing one produces a
document describing a system that does not exist.

``test_baked_json_matches_backend_config`` in test_api_config.py covers the exported JSON. This
covers the consumers: it asserts each one DERIVES the parameters instead of restating them.

AND THE TRAP THAT CAUGHT THE AUTHOR

``load_config()`` overlays ``config.yaml`` onto the dataclass defaults, so ``config.yaml`` is what
runs. Raising the ring in ``defaults.py`` alone changed precisely nothing - the export script,
the report and the browser all kept emitting the old value, silently, with a zero exit status. The
last test here keeps the documented default and the running value in step, so that edit cannot be a
no-op again.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def test_browser_reads_the_ring_from_config() -> None:
    """`clientFHE.ts` must derive its CKKS parameters, not restate them."""
    text = (REPO / "web" / "src" / "lib" / "clientFHE.ts").read_text()
    for name, key in (("POLY_MODULUS_DEGREE", "poly_modulus_degree"),
                      ("COEFF_MOD_BITS", "coeff_mod_bit_sizes")):
        m = re.search(rf"^const {name} = (.+?);", text, re.M)
        assert m, f"{name} is no longer declared in clientFHE.ts - did it move?"
        rhs = m.group(1)
        assert f"fheCfg().{key}" in rhs, (
            f"clientFHE.ts sets {name} to `{rhs}` instead of reading `fheCfg().{key}`. The browser "
            "then runs a parameter set nothing else can see, which is how the ring degree came to "
            "exist in four independent copies."
        )


def test_report_derives_the_crypto_table() -> None:
    """The generator must not hardcode the parameter set it is describing."""
    src = (REPO / "report" / "build_data.py").read_text()
    m = re.search(r"def crypto_table\(\).*?\n(?=\ndef )", src, re.S)
    assert m, "crypto_table() not found in report/build_data.py"
    body = m.group(0)
    # Strip comments: this function's own comment explains the defect using the old literals.
    code = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
    assert "poly_modulus_degree" in code, (
        "crypto_table() does not mention poly_modulus_degree, so its FHE row is not derived from "
        "the config. A generator that hardcodes the ring publishes the old one the moment the code "
        "changes - and the PDF is the artefact a reader trusts most."
    )
    stale = re.findall(r'"[^"]*\b(?:8192|16384|4096)\b[^"]*"', code)
    assert not stale, (
        f"crypto_table() still carries the parameter set as a literal: {stale}. Interpolate it from "
        "the config instead."
    )


def test_tenseal_backend_reads_the_config() -> None:
    """The server-side context must come from the config too, not from default arguments."""
    src = (REPO / "src" / "qsh" / "fhe" / "backends.py").read_text()
    m = re.search(r"class TenSEALBackend.*?\n    def encrypt", src, re.S)
    assert m, "TenSEALBackend.__init__ not found"
    init = m.group(0)
    code = "\n".join(ln for ln in init.splitlines() if not ln.lstrip().startswith("#"))
    assert "get_config()" in code, (
        "TenSEALBackend does not read the config. It used to default poly_modulus_degree, "
        "coeff_mod_bit_sizes and global_scale to literals in its signature, which meant "
        "FHEConfig.poly_modulus_degree had no reader anywhere in src/."
    )
    assert not re.search(r"poly_modulus_degree: int = \d", code), (
        "TenSEALBackend still defaults poly_modulus_degree to a literal in its signature."
    )


@pytest.mark.parametrize("key", ["poly_modulus_degree", "coeff_mod_bit_sizes", "global_scale_bits"])
def test_running_value_matches_the_documented_default(key: str) -> None:
    """`config.yaml` is what runs; `defaults.py` is what a reader reads. They must agree.

    Not a style rule. Raising the ring in `defaults.py` alone produced no change anywhere - the
    export script rewrote the JSON with the OLD value and exited 0, the report regenerated with the
    old value and exited 0, and the browser bundle rebuilt with the old value. Nothing failed; the
    edit was simply inert. An override here is legitimate in principle, but it has to be deliberate
    enough to require editing this test.
    """
    import yaml

    from qsh.config.defaults import FHEConfig

    yaml_path = REPO / "config.yaml"
    if not yaml_path.exists():                       # defaults are then the running value
        pytest.skip("no config.yaml; the dataclass defaults are what run")
    data = (yaml.safe_load(yaml_path.read_text()) or {}).get("fhe") or {}
    if key not in data:                              # not overridden: the default is what runs
        return
    documented = getattr(FHEConfig(), key)
    running = data[key]
    if isinstance(documented, tuple):
        running = tuple(running)
    assert running == documented, (
        f"config.yaml sets fhe.{key} to {running!r} while FHEConfig documents {documented!r}. "
        "config.yaml is overlaid onto the defaults, so the running value is the yaml's - editing "
        "defaults.py alone changes nothing and reports no error while doing it."
    )
