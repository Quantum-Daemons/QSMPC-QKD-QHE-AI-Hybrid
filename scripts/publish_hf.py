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
import os
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
        "license: other",
        f"license_name: {_licence_slug(spec.dataset.licence)}",
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
        "Part of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/thedaemon-wizard/QSMPC-QKD-QHE-AI-Hybrid),",
        "a quantum-safe orchestration demo."]
    if spec.runs_encrypted:
        body += [f"This is the **encrypted-tier** model for the `{spec.usecase}` use case: it is a",
                 "distilled student small enough to evaluate under real CKKS, and the demo runs it",
                 "that way. It is not the plaintext teacher, and its numbers should be read against",
                 "that teacher rather than against the published literature."]
    else:
        body += [f"This is the **plaintext** model for the `{spec.usecase}` use case; the encrypted",
                 "path runs a distilled student, not this model."]
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
    if m.companion:
        body.append(f"- **Companion metric shown alongside**: `{m.companion}` - reported together "
                    "because the aggregate figure can look healthy while the class that matters is not.")
    body.append("")

    if achieved and achieved.get("baseline_gap_note"):
        body += ["### Gap to the published baseline", "", achieved["baseline_gap_note"], ""]

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


def _licence_slug(licence: str) -> str:
    return licence.split(";")[0].strip().lower().replace(" ", "-")[:40] or "unknown"


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
        return ("time-series-forecasting" if spec.metric.name == "event_wise_f0_5"
                else "tabular-classification")
    return "tabular-classification"


def publish(spec: registry.ModelSpec, *, private: bool, dry_run: bool, api) -> str | None:
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
        commit_message=f"{spec.label}: artefacts + measured metrics",
    )
    info = api.model_info(spec.hf_repo_id)
    print(f"      pushed, revision {info.sha}")
    return info.sha


def _refuse_transfer_learned_weights(key: str) -> None:
    """Refuse to publish weights derived from a corpus this project may not redistribute.

    DATA_LICENSES.md and the README both state that `medical_fl.resnet18` is trained from random
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
            f"If the policy has genuinely changed, change DATA_LICENSES.md and the README in the "
            f"same commit -- do not let the artefact and the licence note disagree again."
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", action="append", help="model key(s); default = everything trained")
    ap.add_argument("--public", action="store_true", help="publish publicly (irreversible grant)")
    ap.add_argument("--dry-run", action="store_true")
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
        sha = publish(registry.get(key), private=not args.public, dry_run=args.dry_run, api=api)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
