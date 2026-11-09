#!/usr/bin/env python
"""Publish trained model artefacts to the Hugging Face Hub.

Defaults to PRIVATE repositories, and that default is not timidity. A public Hub repo grants every
user "a perpetual, irrevocable, worldwide, royalty-free, non-exclusive license" over its contents,
so a licence mistake cannot be walked back - you cannot unpublish your way out of it. The free tier
includes 100 GB of private storage precisely so artefacts can be staged, audited and only then
flipped. ``--public`` requires the registry to mark the training dataset publishable AND the
per-score licence audit to have passed, and it refuses otherwise.

API notes, current as of 2026-08:
  * ``huggingface-cli`` was REMOVED in huggingface_hub v1.0.0 (2025-10-24). The CLI is ``hf``.
  * ``upload_large_folder`` is deprecated; ``upload_folder`` is Xet-streamed, auto-splits into
    multiple commits and resumes by re-running the same call.
  * ``HF_XET_HIGH_PERFORMANCE=1`` replaces the removed ``hf_transfer``.

Usage:
    python scripts/publish_hf.py --dry-run            # show what would be pushed
    python scripts/publish_hf.py                      # push private
    python scripts/publish_hf.py --key medical_fl.resnet18 --public
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qsh.models import registry  # noqa: E402

#: Local artefact directory per model key. Written by the scripts/train/* scripts.
#: Single source of truth is the registry -- it needs the same mapping to read export reports.
ARTIFACT_DIRS = registry.ARTIFACT_DIRS

#: Never upload these even if they land in an artefact directory: fixtures are for the local export
#: gate and the browser audit, and raw training data must not leave the machine.
DENY = ("*.csv", "*.tsv", "*.npy", "*.parquet", "*.pkl")


def token() -> str:
    """Read the fine-grained token. Env wins so CI can inject one without a file on disk."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    path = Path.home() / ".huggingface" / "huggingface.json"
    if not path.exists():
        raise SystemExit(f"no HF token: set HF_TOKEN or create {path}")
    return json.loads(path.read_text())["key"]


def model_card(spec: registry.ModelSpec, achieved: dict | None) -> str:
    """A model card built from the registry, so the card cannot drift from what the code believes."""
    m = spec.metric
    tags = ["quantum-safe", "privacy-preserving", spec.usecase.replace("_", "-")]
    if any(a.fmt == "onnx" for a in spec.artifacts):
        tags += ["onnx", "onnxruntime-web"]

    front = [
        "---",
        *_hub_licence(spec.dataset.licence),
        "library_name: onnx" if "onnx" in tags else "library_name: transformers",
        f"pipeline_tag: {_pipeline_tag(spec)}",
        "tags:",
        *[f"  - {t}" for t in tags],
        "---",
        "",
    ]

    body = [
        f"# {spec.label}",
        "",
        f"{spec.architecture}",
        "",
        f"Part of [QSMPC-QKD-QHE-AI-Hybrid]({REPO_URL}),",
        "a quantum-safe orchestration demo."]
    if spec.runs_encrypted:
        # "and the demo runs it that way" was true of some students only. The public page encrypts the medical CNN
        # student and, on held-out rows, the finance, telecom and OPS-SAT students (finance since Round 84, when it
        # stopped encrypting a bundled PaySim model in its place), and the Telemanom student under CKKS nowhere.
        # The registry notes say what runs, per model, and the card points at them.
        body += [f"This is the **encrypted-tier** model for the `{spec.usecase}` use case: a",
                 "distilled student small enough to evaluate under CKKS. Whether the public demo",
                 "encrypts THIS model differs by use case; the notes below say what actually runs.",
                 "It is not the plaintext teacher, and its numbers should be read against that",
                 "teacher rather than against the published literature. Its agreement with the",
                 "teacher was measured offline, in plaintext."]
    else:
        # "the encrypted path runs a distilled student" was false for most teachers: nothing is
        # distilled from the ResNet-50 or the 5G-NIDD model, and on the satellite page the CKKS check
        # (Round 86) scores the OPS-SAT student, not the Telemanom teacher's own student.
        body += [f"This is the **plaintext** model for the `{spec.usecase}` use case. No code path",
                 "evaluates this model under encryption; what each use case encrypts, if anything,",
                 "is stated in the notes below."]
    body += [
        "",
        "## Measured performance",
        "",
    ]

    if achieved:
        body += ["| metric | value |", "|---|---|"]
        for k, v in sorted(achieved.items()):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                body.append(f"| `{k}` | {v} |")
        body.append("")
    else:
        body += ["Not yet trained. No metrics are reported, and none should be inferred from the",
                 "published baselines below.", ""]

    body += [
        "## Published baselines this is measured against",
        "",
        f"- **Target metric**: {m.display}",
    ]
    if m.baseline is not None:
        body.append(f"- **Baseline to beat**: {m.baseline} - {m.baseline_source}")
    else:
        body.append(f"- **Baseline**: {m.baseline_source}")
    if m.ceiling is not None:
        body.append(f"- **Published ceiling**: {m.ceiling} - {m.ceiling_source}")
    # Only a companion this record carries. Five cards named one their own table did not contain: every student
    # inherits its teacher's spec and records agreement instead, and the Telemanom spec named a key no record has
    # (Round 86, a reader).
    if m.companion and achieved and m.companion in achieved:
        why = m.companion_note or ("reported together because the aggregate figure can look healthy while the class "
                                   "that matters is not")
        body.append(f"- **Companion metric shown alongside**: `{m.companion}` - {why}.")
    body.append("")

    if achieved and achieved.get("baseline_gap_note"):
        body += ["### Gap to the published baseline", "", achieved["baseline_gap_note"], ""]

    body += _calibration_section((achieved or {}).get("calibration"))
    body += _esa_scores_section((achieved or {}).get("esa_scores"), m.baseline,
                                (achieved or {}).get("published_lightweight_f0_5"))

    body += [
        "## Training data",
        "",
        f"- **Dataset**: {spec.dataset.name}",
        f"- **Licence**: {spec.dataset.licence}",
        f"- **Source**: {spec.dataset.url} (licence read {spec.dataset.licence_checked})",
    ]
    # A CC BY model card without a credit line does not satisfy the licence it declares in its own
    # front matter, and this card is the copy a Hub visitor reads. Emitted from the registry so it
    # cannot drift from what the browser and the eval sample say.
    if spec.dataset.attribution:
        body.append(f"- **Attribution ({spec.dataset.licence})**: {spec.dataset.attribution}")
    if spec.dataset.notes:
        body += ["", spec.dataset.notes]
    body.append("")

    if spec.notes:
        body += ["## Notes and limitations", "", spec.notes, ""]

    body += [
        "## Honest scope",
        "",
        "This model is published as part of a research proof of concept, not as a production system.",
        "Numbers above are what this repository measured on the split described, with the code in",
        "`scripts/train/`. Where a figure is carried from the literature rather than measured here,",
        "it is labelled as such.",
        "",
    ]
    return "\n".join(front + body)


#: An inline math span written the repository's way, `$...$`, outside any `$$` display block.
_INLINE_MATH = re.compile(r"(?<![$\\])\$(?!\$)([^$\n]+?)(?<!\\)\$(?!\$)")


def _hub_inline_math(text: str) -> str:
    r"""Inline math as the Hub renders it (Round 86).

    The Hub typesets the card's source with KaTeX before it parses the Markdown, and its model-card guide gives
    ``$$ ... $$`` for display math and ``\\(...\\)`` for inline math. A single-dollar span is not math there: the first
    calibration card published showed ``$q = \sigma(...)$`` as raw text, backslashes and all, while its display
    formula rendered.
    """
    return _INLINE_MATH.sub(lambda m: r"\\(" + m.group(1) + r"\\)", text)


def _esa_scores_section(esa: dict | None, published: float | None, published_all: float | None) -> list[str]:
    """The satellite forecaster's arms and floors as a table, so the replication is readable on the Hub (Round 86).

    The card's prose names `esa_scores`, which lives in the repository's record; this puts the cells beside the
    published figures on the card a Hub visitor reads.
    """
    if not isinstance(esa, dict):
        return []
    arms = (("plain_zero_order_hold", "plain zero-order hold"),
            ("restored_annotated_samples", "ESA-ADB's prepared series, which reads the labels"))
    #: The two cells the card names: this project's result, and the replication of the published figure.
    named = {("plain_zero_order_hold", "validation_floor"): "this project's result",
             ("restored_annotated_samples", "paper_floor"): "replication"}
    runs = (("validation_floor", "floor from the validation months"), ("paper_floor", "the paper's floor"),
            ("no_floor", "no floor"), ("agreement", "no floor, channels agreeing"))
    rows = []
    for arm, arm_label in arms:
        for run, run_label in runs:
            r = (esa.get(arm) or {}).get(run)
            if not isinstance(r, dict) or not isinstance(r.get("corrected_event_wise_f0_5"), (int, float)):
                continue
            every = (r.get("all_events") or {}).get("corrected_event_wise_f0_5")
            value = f"{r['corrected_event_wise_f0_5']:.3f}"
            if (arm, run) in named:
                value = f"**{value}** ({named[(arm, run)]})"
            setting = f"k = {r['k']} of 6, from the validation months" if "k" in r else r["floor"]
            rows.append(f"| {arm_label} | {run_label} ({setting}) | {value} | "
                        f"{r['tp']} / {r['fp']} / {r['fn']} | "
                        f"{'-' if every is None else format(every, '.3f')} |")
    if not rows:
        return []
    pub = "-" if published is None else f"{published:.3f}"
    pub_all = "-" if published_all is None else f"{published_all:.3f}"
    return ["### ESA-ADB's scores, both input series", "",
            "Corrected event-wise F0.5 over the test half's anomalies, and over all events but communication gaps "
            "(ESA-ADB's Table 2 selection). One model; the two series differ only in ESA-ADB's restoration of "
            "annotated samples.", "",
            "| input series | floor | anomalies | tp / fp / fn | all events |", "|---|---|---|---|---|",
            *rows,
            f"| published Telemanom-ESA-Pruned (not measured here) | 0.05/7 | {pub} | | {pub_all} |", ""]


def _calibration_section(cal: dict | None) -> list[str]:
    """The record's Platt map, as the page applies it before it reads a score against a threshold (Round 86).

    The finance page draws decision curves through these maps, and until Round 86 no card printed one: a Hub reader had
    the graph and not the function that turns its output into the risks the page plots. Numbers go in a "quantity" table
    so `tests/test_a_model_card_states_what_its_weights_state.py` checks each against the record, and in fixed notation,
    since that gate reads `4.505e-05` as the integer 5.
    """
    if not cal or cal.get("method") != "platt":
        return []
    fx = lambda v: f"{v:.8f}".rstrip("0").rstrip(".") if abs(v) < 1e-3 else repr(v)  # noqa: E731
    eps_exp = round(math.log10(cal["logit_clip_eps"]))
    arg = (r"a\,\operatorname{logit}\big(\operatorname{clip}(q,\ \varepsilon,\ 1-\varepsilon)\big) + b"
           if cal["on_logit"] else r"a\,q + b")
    corp_raw, corp_cal = cal.get("corp_raw") or {}, cal.get("corp_calibrated") or {}
    lines = [
        "## Calibration map",
        "",
        "The demo page reads this model's score $q$ as a risk through this map before it compares it with a "
        "threshold probability, as its decision curve does. Fitted on the "
        f"{cal['fitted_on'].replace('; ', ' and ')}.",
        "",
        # The map and its definitions in two blocks: on one line they set 736 px in the Hub's 589 px column, and the
        # display does not scroll, so the clip constant ran off the card (Round 86, measured on the published card).
        rf"$$p = \frac{{1}}{{1 + \exp\!\big({arg}\big)}}$$",
        "",
        *([rf"$$\operatorname{{logit}}(q) = \log\frac{{q}}{{1-q}},\qquad \varepsilon = 10^{{{eps_exp}}}$$", ""]
          if cal["on_logit"] else []),
    ]
    if cal.get("argument"):
        lines += [f"Here $q$ is {cal['argument']}.", ""]
    lines += [
        "| quantity | value |",
        "|---|---|",
        f"| `a` | {fx(cal['a'])} |",
        f"| `b` | {fx(cal['b'])} |",
        f"| fitted on | {cal['n_calibration']} validation rows, {cal['positives_calibration']} positive |",
    ]
    if corp_cal.get("n"):
        lines.append(f"| scored on | {corp_cal['n']} test rows, {corp_cal['positives']} positive |")
    for name, raw, calibrated in (("Brier score", cal.get("brier_raw"), cal.get("brier_calibrated")),
                                  ("log-loss", cal.get("log_loss_raw"), cal.get("log_loss_calibrated")),
                                  ("CORP miscalibration (MCB)", corp_raw.get("mcb"), corp_cal.get("mcb"))):
        if raw is not None and calibrated is not None:
            lines.append(f"| {name}, raw to calibrated | {fx(raw)} to {fx(calibrated)} |")
    if corp_cal.get("unc") is not None:
        lines.append(f"| base-rate Brier score (CORP UNC) | {fx(corp_cal['unc'])} |")
    lines.append("")
    if corp_cal.get("unc") and cal.get("brier_calibrated") is not None:
        # Derived from the two rows above, in prose: the skill is not a recorded field, and the gate checks the table.
        skill = 1 - cal["brier_calibrated"] / corp_cal["unc"]
        lines += [r"Brier skill against the base rate, $1 - \mathrm{BS}/\mathrm{UNC}$, is " + f"{skill:.4f}" + (
            ": negative, so as a probability forecast the calibrated score does worse than giving every row the "
            "window's rate. A decision curve asks a different question, what acting at each threshold pays."
            if skill < 0 else "."), ""]
    if cal.get("raw_is"):
        lines += [f"Raw is {cal['raw_is'][0].lower()}{cal['raw_is'][1:]}", ""]
    if cal.get("why_not_isotonic"):
        lines += [f"Why Platt and not isotonic: {cal['why_not_isotonic']}", ""]
    return [_hub_inline_math(line) for line in lines]


def _licence_slug(licence: str) -> str:
    return licence.split(";")[0].strip().lower().replace(" ", "-")[:40] or "unknown"


REPO_URL = "https://github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid"

#: Licences the Hub lists by identifier: a card naming one of these uses it directly, so the Hub
#: indexes the repo under that licence instead of "other" (Round 84; Hub model-card specification).
_HUB_LICENCE_IDS = {"CDLA-Sharing-1.0": "cdla-sharing-1.0", "CC-BY-4.0": "cc-by-4.0",
                    "Apache-2.0": "apache-2.0"}

#: For the rest, `license: other` takes a name and a link (the Hub's documented practice).
_LICENCE_LINKS = {
    "CC-BY-3.0-IGO": "https://creativecommons.org/licenses/by/3.0/igo/",
    "CC-BY-4.0 asserted by MedMNIST": "https://creativecommons.org/licenses/by/4.0/",
}


def _hub_licence(licence: str) -> list[str]:
    """The card's licence front matter: a Hub identifier where one exists, else name and link."""
    if licence in _HUB_LICENCE_IDS:
        return [f"license: {_HUB_LICENCE_IDS[licence]}"]
    head = licence.split(";")[0].strip()
    link = _LICENCE_LINKS.get(head, f"{REPO_URL}/blob/main/DATA_LICENSES.md")
    return ["license: other", f"license_name: {_licence_slug(licence)}", f"license_link: {link}"]


def _pipeline_tag(spec: registry.ModelSpec) -> str:
    """The Hub task tag, decided by what the model DOES rather than by which industry it serves.

    Keying this off the use case alone published `time-series-forecasting` for the two OPS-SAT
    models, which are segment CLASSIFIERS - and the registry says so twice over, in
    `architecture` and in the fact that they are scored on AUC-PR. That tag is what the Hub
    filters and what the widget offers, so it is a claim about the model, not decoration. The
    registry's own metric name is the discriminator: an event-wise F0.5 score belongs to a
    forecaster, AUC-PR to a classifier.
    """
    if spec.usecase == "medical_fl":
        return "image-classification"
    if spec.usecase == "satellite_comms":
        # The corrected name since Round 86, the uncorrected one before; both are the forecaster's event-wise score.
        return ("time-series-forecasting" if spec.metric.name in ("event_wise_f0_5", "corrected_event_wise_f0_5")
                else "tabular-classification")
    return "tabular-classification"


def stale_remote_files(remote: set[str], local: set[str]) -> list[str]:
    """Files at the repo head that this upload would leave behind: on the Hub, not in the folder.

    `upload_folder` adds and replaces and never deletes, so a file the local directory stopped
    producing stays at the head of the repo indefinitely. Round 71 found one: ResNet-50's repo still
    served an int8 graph quantised on 2026-08-09 from weights since replaced, while the
    export_report.json beside it said int8 had been withheld and fp32 only was served. The browser
    reads the report and never fetched it; anyone downloading from the Hub got another model's int8.
    """
    return sorted(remote - local - {".gitattributes"})


def publish(spec: registry.ModelSpec, *, private: bool, dry_run: bool, api,
            delete_stale: bool = False) -> str | None:
    if not spec.hf_repo:
        print(f"  {spec.key}: no hf_repo in the registry - skipping (publication intentionally withheld)")
        return None

    local = Path(ARTIFACT_DIRS.get(spec.key, ""))
    if not local or not local.exists():
        print(f"  {spec.key}: no artefacts at {local} - train it first")
        return None

    if not private and not spec.dataset.publishable:
        raise SystemExit(
            f"refusing to publish {spec.key} publicly: its training data ({spec.dataset.name}, "
            f"{spec.dataset.licence}) is not marked publishable in the registry. The Hub's public "
            "grant is irrevocable, so this check is not overridable from the command line."
        )

    achieved = registry.achieved(spec.key)
    card = model_card(spec, achieved)
    (local / "README.md").write_text(card)

    # List exactly what upload_folder will send. Printing the whole directory and relying on
    # ignore_patterns to filter it later would make a dry run that shows training rows being
    # published look identical to one that does not - the wrong thing to be vague about.
    from fnmatch import fnmatch

    all_files = sorted(p for p in local.rglob("*") if p.is_file())
    files = [p for p in all_files
             if not any(fnmatch(p.name, pat) for pat in DENY)]
    excluded = [p for p in all_files if p not in files]
    total = sum(p.stat().st_size for p in files)
    print(f"  {spec.key} -> {spec.hf_repo_id} ({'private' if private else 'PUBLIC'})")
    for p in files:
        print(f"      {p.relative_to(local)}  {p.stat().st_size/1e6:.2f} MB")
    for p in excluded:
        print(f"      [excluded] {p.relative_to(local)}  (matches the deny list; stays local)")
    print(f"      total to upload {total/1e6:.2f} MB")

    # Checked BEFORE any upload, dry runs included, so a stale file is reported rather than shipped.
    try:
        remote = set(api.list_repo_files(spec.hf_repo_id, repo_type="model"))
    except Exception:                                  # noqa: BLE001 - a repo not created yet
        remote = set()
    stale = stale_remote_files(remote, {str(p.relative_to(local)) for p in files})
    for f in stale:
        print(f"      [stale on the Hub] {f}  (not produced locally; "
              f"{'deleted in this commit' if delete_stale else 'left in place'})")
    if stale and not delete_stale and not dry_run:
        raise SystemExit(
            f"refusing to publish {spec.key}: the head of {spec.hf_repo_id} holds {stale}, which this "
            "folder no longer produces, and upload_folder would leave them beside the new artefacts. "
            "Re-run with --delete-stale to remove them in the same commit (earlier revisions keep "
            "them), or restore them locally if they are meant to stay.")

    if dry_run:
        return None

    api.create_repo(spec.hf_repo_id, repo_type="model", private=private, exist_ok=True)
    # create_repo(exist_ok=True) does NOT change the visibility of a repo that already exists, so a
    # repo first published privately stayed private and --public reported success while changing
    # nothing. Set it explicitly. (update_repo_visibility was removed from huggingface_hub;
    # update_repo_settings is the current call.)
    if not dry_run:
        try:
            api.update_repo_settings(repo_id=spec.hf_repo_id, repo_type="model", private=private)
        except Exception as exc:                       # noqa: BLE001 - surfaced, never swallowed
            raise SystemExit(
                f"could not set visibility on {spec.hf_repo_id}: {exc}\n"
                "The upload may have succeeded while the repo stayed private -- check the Hub "
                "before assuming this model is reachable by a visitor."
            ) from exc
    api.upload_folder(
        folder_path=str(local),
        repo_id=spec.hf_repo_id,
        repo_type="model",
        ignore_patterns=list(DENY),
        # Exact paths, never a wildcard: only what the check above named and printed is removed.
        delete_patterns=stale if delete_stale and stale else None,
        commit_message=f"{spec.label}: artefacts + measured metrics",
    )
    info = api.model_info(spec.hf_repo_id)
    print(f"      pushed, revision {info.sha}")
    return info.sha


def _refuse_transfer_learned_weights(key: str) -> None:
    """Refuse to publish weights derived from a corpus this project may not redistribute.

    DATA_LICENSES.md and docs/findings.md (the README's text until Round 80) both state that
    `medical_fl.resnet18` is trained from random
    initialisation, because ImageNet's terms bind the Database to non-commercial research and say
    nothing about redistributing a model trained on it - so ImageNet-derived weights cannot be
    published here. `scripts/train/medical_breastmnist.py` implements that as `--from-scratch`.

    It is not the default. So training without the flag and publishing produced exactly the state
    Round 24c found: two documents promising random initialisation while the artefact on the Hub
    recorded `pretrained: ImageNet1K_V1`. Nothing checked, and the divergence survived several
    rounds. A licence claim that only a human remembers to honour is not a control.

    The manifest already records the answer, so this reads it and stops.
    """
    spec = registry.get(key)
    if not spec.hf_repo or not spec.dataset.publishable:
        return
    manifest = Path(ARTIFACT_DIRS.get(key, "")) / "manifest.json"
    if not manifest.exists():
        return
    pretrained = (json.loads(manifest.read_text()) or {}).get("pretrained")
    if pretrained and str(pretrained).lower() not in {"none", "false", ""}:
        raise SystemExit(
            f"REFUSING to publish {key}: its manifest records pretrained={pretrained!r}, so these "
            f"weights are derived from a corpus whose terms this project has not cleared for "
            f"redistribution (see DATA_LICENSES.md). Retrain with --from-scratch and republish. "
            f"If the policy has genuinely changed, change DATA_LICENSES.md and docs/findings.md in the "
            f"same commit -- do not let the artefact and the licence note disagree again."
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", action="append", help="model key(s); default = everything trained")
    ap.add_argument("--public", action="store_true", help="publish publicly (irreversible grant)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--delete-stale", action="store_true",
                    help="remove files at the repo head that the local folder no longer produces")
    args = ap.parse_args()

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    from huggingface_hub import HfApi

    api = HfApi(token=token())
    who = api.whoami()
    print(f"authenticated as {who['name']} (pro={who.get('isPro', False)})")
    if who["name"] != registry.HF_NAMESPACE:
        raise SystemExit(f"token belongs to {who['name']}, expected {registry.HF_NAMESPACE}")

    keys = args.key or [s.key for s in registry.specs() if s.hf_repo]
    print(f"\npublishing {len(keys)} model(s), private={not args.public}\n")

    # Licence gate BEFORE any upload, so a mixed batch cannot push half of itself and then stop.
    for key in keys:
        _refuse_transfer_learned_weights(key)

    revisions = {}
    for key in keys:
        sha = publish(registry.get(key), private=not args.public, dry_run=args.dry_run, api=api,
                      delete_stale=args.delete_stale)
        if sha:
            revisions[key] = sha

    if revisions:
        # Pin the revision in metrics.json so the web app fetches an immutable commit rather than
        # whatever `main` happens to be when a visitor loads the page.
        from qsh.models.train import record_metrics

        for key, sha in revisions.items():
            payload = dict(registry.achieved(key) or {})
            payload["hf_revision"] = sha
            # DROP the invalidation marker. `record_metrics` sets it when a retrain lands without a
            # revision, and nothing cleared it afterwards - so a re-publish left the entry asserting
            # both "pinned at X" and "X was invalidated by a retrain". On finance_aml.student the two
            # were the SAME sha, because re-uploading identical bytes returns the existing commit.
            # The registry prefers `hf_revision`, so the contradiction was invisible at runtime and
            # visible only in the record, which is where this project has been bitten before.
            payload.pop("hf_revision_invalidated_by_retrain", None)
            record_metrics(key, payload)
        print(f"\nrecorded {len(revisions)} pinned revision(s) in models/metrics.json")
        # The browser refuses a file with no digest for its pin, so a moved pin takes the model off the page until this runs.
        print("NEXT: python scripts/pin_hub_hashes.py, then scripts/export_config.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
