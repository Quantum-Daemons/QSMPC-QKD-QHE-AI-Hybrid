#!/usr/bin/env python
"""Publish trained model artefacts to the Hugging Face Hub.

Defaults to PRIVATE repositories, and that default is not timidity. A public Hub repo grants every
user "a perpetual, irrevocable, worldwide, royalty-free, non-exclusive license" over its contents,
so a licence mistake cannot be walked back — you cannot unpublish your way out of it. The free tier
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
ARTIFACT_DIRS = {
    "finance_aml.gbm": "models/finance_aml_gbm",
    "finance_aml.gnn": "models/finance_aml_gnn",
    "medical_fl.resnet18": "models/medical_breastmnist_resnet18",
    "medical_fl.densenet": "models/medical_camelyon17_densenet121",
    "genomics.prs": "models/genomics_prs",
    "satellite_comms.telemanom": "models/satellite_telemanom",
    "telecom_ids.qml": "models/telecom_qml_ids",
}

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
        "a quantum-safe orchestration demo. This is the **plaintext** model for the",
        f"`{spec.usecase}` use case"
        + (" and it runs under real CKKS encryption." if spec.runs_encrypted
           else "; the encrypted path runs a distilled student, not this model."),
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
        body.append(f"- **Baseline to beat**: {m.baseline} — {m.baseline_source}")
    else:
        body.append(f"- **Baseline**: {m.baseline_source}")
    if m.ceiling is not None:
        body.append(f"- **Published ceiling**: {m.ceiling} — {m.ceiling_source}")
    if m.companion:
        body.append(f"- **Companion metric shown alongside**: `{m.companion}` — reported together "
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
    return {"medical_fl": "image-classification",
            "satellite_comms": "time-series-forecasting"}.get(spec.usecase, "tabular-classification")


def publish(spec: registry.ModelSpec, *, private: bool, dry_run: bool, api) -> str | None:
    if not spec.hf_repo:
        print(f"  {spec.key}: no hf_repo in the registry — skipping (publication intentionally withheld)")
        return None

    local = Path(ARTIFACT_DIRS.get(spec.key, ""))
    if not local or not local.exists():
        print(f"  {spec.key}: no artefacts at {local} — train it first")
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
    # published look identical to one that does not — the wrong thing to be vague about.
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
            record_metrics(key, payload)
        print(f"\nrecorded {len(revisions)} pinned revision(s) in models/metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
