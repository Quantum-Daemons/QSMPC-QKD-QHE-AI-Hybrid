#!/usr/bin/env python3
"""What the LIDC-IDRI stream verified, in aggregate: the owner's requirement of 2026-11-24 that every dataset fetched
be complete, undamaged and genuine.

Reads the stream's per-patient records (`OUT/done/<patient>.json`, written by scripts/data/lidc_stream.py only after
every file of every series matched the MD5 TCIA's API lists with it and the series' file count matched TCIA's
`ImageCount`), the records the split was drawn from (scripts/data/lidc_records.py), TCIA's series listing, and the
TotalSegmentator cache, and checks that nothing is missing: every recorded series fetched and verified, every patient
done, every cluster cut, every training series segmented. Writes aggregate figures only; no image and no per-patient
row.

    python scripts/data/lidc_stream_summary.py OUT -o docs/measurements/lidc-stream-<date>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()
    records_file = args.out / "records.json"
    records = json.loads(records_file.read_text())["records"]
    listing = {s["SeriesInstanceUID"]: s for s in json.loads((args.out / "tcia_getSeries_LIDC-IDRI.json").read_text())}
    done = {p.stem: json.loads(p.read_text()) for p in sorted((args.out / "done").glob("*.json"))}
    want_series = {r["series"] for r in records}
    want_patients = {r["patient"] for r in records}
    got_series = {s["series"]: s for d in done.values() for s in d["series"]}
    problems = []
    if set(done) != want_patients:
        problems.append(f"patients: {len(want_patients - set(done))} not done, {len(set(done) - want_patients)} unexpected")
    if set(got_series) != want_series:
        problems.append(f"series: {len(want_series - set(got_series))} not fetched, {len(set(got_series) - want_series)} unexpected")
    for uid, s in got_series.items():
        if s["md5_checked"] != s["files"] or s["files"] != s["image_count"] or s["image_count"] != int(listing[uid]["ImageCount"]):
            problems.append(f"series {uid[-12:]}: files {s['files']}, MD5-checked {s['md5_checked']}, ImageCount {s['image_count']}")
    crops = sum(d["crops"] for d in done.values())
    if crops != len(records):
        problems.append(f"{crops} crops cut against {len(records)} cluster records")
    train = [d for d in done.values() if d["train"]]
    segmented = [t for d in train for t in d.get("totalseg", [])]
    errors = [t for t in segmented if "error" in t]
    if errors:
        problems.append(f"{len(errors)} TotalSegmentator runs failed")
    summary = {
        "what": "LIDC-IDRI (TCIA, CC BY 3.0) as streamed through the system disk and verified file by file",
        "records": {"file": str(records_file.name), "sha256": hashlib.sha256(records_file.read_bytes()).hexdigest(),
                    "clusters": len(records), "series": len(want_series), "patients": len(want_patients)},
        "fetched": {"patients": len(done), "series": len(got_series), "files": sum(s["files"] for s in got_series.values()),
                    "files_md5_checked": sum(s["md5_checked"] for s in got_series.values()),
                    "bytes": sum(s["zip_bytes"] for s in got_series.values())},
        "cut": {"crops_28mm": crops, "training_patients": len(train),
                "hu64_cubes": sum(d.get("hu64", 0) for d in train)},
        "totalsegmentator": {"series_segmented": len(segmented), "failed": len(errors),
                             "empty_masks": sum(1 for t in segmented if t.get("components") == 0)},
        "complete": not problems, "problems": problems[:20],
        "date": time.strftime("%Y-%m-%d"),
    }
    args.output.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
