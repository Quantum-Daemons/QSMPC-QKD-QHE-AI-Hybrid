"""A configuration default must not be published under the key a measurement fills.

WHY THIS EXISTS

`clientCompute.ts` emitted `costPerOp().demo_max_abs_error` - the literal `4.6e-07` from
`config.defaults.json` - as the run metric `fhe_max_abs_error`, on the branch where node-seal fails
to load. The measured branch fills the SAME key from a real decryption, and the recorded measured
values are `6.20e-13` and `1.93e-9`.

The only thing separating the two was a log line beside the metric. `downloadMetricsCsv` exports
`{metric, value}` pairs and no log, so the CSV a reviewer opens showed a config default that is
indistinguishable from a measurement, under a key `docs/threat-model.md` describes as the computed
CKKS approximation error.

The direction is worth recording because it cuts against the usual pattern: `4.6e-07` is three to
four orders WORSE than measured, so this understated precision rather than flattering it. It is still
a constant wearing a measurement's name.

WHAT IS ASSERTED

That no metric key which the measured path fills is ALSO filled from `costPerOp()` or another config
read. Asserted on the source, because the two branches emit the same key and a value alone cannot
say where it came from - which is the whole defect.

WHAT IS DELIBERATELY NOT ASSERTED

Not that representative figures are wrong to publish. They are useful and this project publishes
several. Only that they carry a name saying what they are, and that the run states which tier
produced them - the `published_model_status: "failed"` convention this file already follows.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLIENT = REPO / "web" / "src" / "lib" / "clientCompute.ts"

#: Keys the MEASURED path fills from a real computation. A config read published under any of these
#: is the defect. Listed explicitly rather than inferred, so adding a key is a decision.
MEASURED_KEYS = (
    "fhe_max_abs_error",
    "fhe_conv_max_abs_error",
    "encrypted_batch_max_abs_error",
    "encrypted_student_ms",
    "published_inference_ms",
    # Round 68: the fallback branch filled these three from the cost model's row for as long as the
    # first key above was fixed, because only that key was listed here.
    "fhe_linear_ms",
    "mpc_nonlinear_ms",
    "mpc_comm_kb",
)

#: Expressions that read configuration rather than measuring anything. `representative.` and
#: `split.` are the cost model's row (`mpcFheRow` with no measurement) under the names the fallback
#: branch has used for it.
CONFIG_READS = (r"costPerOp\(\)", r"cfg\(\)\.", r"qkdCfg\(\)\.", r"idsCfg\(\)\.",
                r"\brepresentative\.", r"\bsplit\.")


def test_no_measured_metric_key_is_emitted_from_a_config_read():
    """The property: one key, one provenance."""
    src = CLIENT.read_text()
    bad = []
    for m in re.finditer(r'metric\(\s*stage\s*,\s*"([a-z0-9_]+)"\s*,\s*([^)]*)\)', src):
        key, value = m.group(1), m.group(2)
        if key not in MEASURED_KEYS:
            continue
        if any(re.search(pat, value) for pat in CONFIG_READS):
            line = src[: m.start()].count("\n") + 1
            bad.append(f"clientCompute.ts:{line} emits `{key}` from `{value.strip()[:60]}`")
    assert not bad, (
        "these run metrics are filled from configuration on some branch and from a measurement on "
        "another, under one key:\n  " + "\n  ".join(bad)
        + "\n\nA value cannot carry its own provenance, and the CSV export drops the log line that "
        "was the only distinguisher. Rename the representative one - `<key>_representative` - and "
        "emit a tier status beside it, the way `published_model_status: \"failed\"` already works."
    )


def test_the_scan_has_a_subject():
    """Anti-tautology. If no metric call matches the pattern, the gate above proves nothing."""
    src = CLIENT.read_text()
    calls = re.findall(r'metric\(\s*stage\s*,\s*"([a-z0-9_]+)"', src)
    assert len(calls) > 40, (
        f"only {len(calls)} `metric(stage, ...)` calls parsed, which is far below what this file "
        "contains - the emit helper was renamed and this scan is looking at nothing.")
    covered = [k for k in MEASURED_KEYS if k in calls]
    assert len(covered) >= 3, (
        f"only {covered} of the declared measured keys are still emitted; re-derive MEASURED_KEYS "
        "rather than letting the gate pass on keys that no longer exist.")


def test_the_representative_figure_is_named_as_one_and_declares_its_tier():
    """The fallback branch must say what it is, not merely avoid the measured key."""
    src = CLIENT.read_text()
    assert "fhe_max_abs_error_representative" in src, (
        "the CKKS-unavailable branch no longer publishes a representative error under a "
        "representative name. If it stopped publishing one entirely that is fine - delete this "
        "assertion. If it went back to the measured key, that is the defect.")
    assert re.search(r'metric\(\s*stage\s*,\s*"encrypted_tier"', src), (
        "the CKKS-unavailable branch emits no `encrypted_tier` status. Renaming the metric is half "
        "the fix; a reader also has to be able to tell from the metrics alone that the encrypted "
        "tier did not run, without relying on a log line the CSV export does not carry.")
