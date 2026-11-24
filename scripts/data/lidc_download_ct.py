#!/usr/bin/env python3
"""Download LIDC-IDRI's CT series from TCIA's public REST API, verified (owner's approval of 2026-11-24).

The collection: LIDC-IDRI on The Cancer Imaging Archive, CC BY 3.0, used under TCIA's Data Usage Policies and
Restrictions (no attempt to identify or contact participants; full data citation with its DOI,
https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX). Only the 1,018 CT series are fetched (128.4 GB by TCIA's listing);
the radiographs and the later SEG and SR series are not needed for nodule crops.

For each series: GET getImageWithMD5Hash, which returns a zip of the DICOM files and a CSV of their MD5 hashes;
every file's MD5 is checked before the series is extracted to <OUT>/LIDC-IDRI/<PatientID>/<StudyUID>/<SeriesUID>/
(the layout pylidc reads) and marked complete. A series already marked complete is skipped, so a stopped run
resumes. Standard library only.

    python3 scripts/data/lidc_download_ct.py <OUT> [--limit N] [--workers N]

Its `get` and `md5_table` are also what scripts/data/lidc_stream.py fetches and checks with.
"""

from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import io
import json
import sys
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
UA = {"User-Agent": "lidc-ct-fetch/1.0 (research; python-urllib)"}


def get(url: str, timeout: int = 600) -> bytes:
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001 - network errors are retried, then raised
            if attempt == 4:
                raise
            wait = 15 * (attempt + 1)
            print(f"  retry in {wait}s after {type(e).__name__}: {e}", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def md5_table(z: zipfile.ZipFile) -> dict[str, str]:
    """The zip's MD5 list, whatever the header calls its two columns."""
    name = next((n for n in z.namelist() if n.lower().endswith(".csv") and "md5" in n.lower()), None)
    if name is None:
        raise ValueError("the zip carries no MD5 list")
    rows = list(csv.reader(io.StringIO(z.read(name).decode("utf-8", "replace"))))
    out = {}
    for row in rows[1:]:
        if len(row) >= 2:
            fname, digest = row[0].strip(), row[1].strip().lower()
            out[Path(fname).name] = digest
    return out


def main() -> int:
    out = Path(sys.argv[1])
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    listing = json.loads((out / "tcia_getSeries_LIDC-IDRI_20261124.json").read_text())
    series = sorted((s for s in listing if s.get("Modality") == "CT"), key=lambda s: (s["PatientID"], s["SeriesInstanceUID"]))
    if limit:
        series = series[:limit]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 1
    state = {"bytes": 0, "t0": time.time(), "ok": 0, "bad": 0}
    lock = threading.Lock()

    def one(k: int, s: dict) -> None:
        uid = s["SeriesInstanceUID"]
        dest = out / "LIDC-IDRI" / s["PatientID"] / s["StudyInstanceUID"] / uid
        if (dest / ".complete").exists():
            return
        data = get(f"{API}/getImageWithMD5Hash?SeriesInstanceUID={urllib.parse.quote(uid)}")
        z = zipfile.ZipFile(io.BytesIO(data))
        table = md5_table(z)
        dicoms = [n for n in z.namelist() if n.lower().endswith(".dcm")]
        bad = [n for n in dicoms if hashlib.md5(z.read(n)).hexdigest() != table.get(Path(n).name)]
        if bad or len(dicoms) != int(s.get("ImageCount", len(dicoms))):
            with lock:
                state["bad"] += 1
                print(f"[{k}/{len(series)}] {s['PatientID']} {uid}: {len(bad)} MD5 mismatches, "
                      f"{len(dicoms)} files of {s.get('ImageCount')}; NOT extracted", flush=True)
            return
        dest.mkdir(parents=True, exist_ok=True)
        for n in dicoms:
            (dest / Path(n).name).write_bytes(z.read(n))
        (dest / ".complete").write_text(json.dumps({"files": len(dicoms), "zip_bytes": len(data),
                                                    "md5_checked": len(dicoms), "at": time.strftime("%Y-%m-%dT%H:%M:%S")}))
        with lock:
            state["bytes"] += len(data)
            state["ok"] += 1
            rate = state["bytes"] / max(1.0, time.time() - state["t0"]) / 1e6
            print(f"[{k}/{len(series)}] {s['PatientID']} {len(dicoms)} files {len(data)/1e6:.1f} MB ok "
                  f"({rate:.1f} MB/s, {state['ok']} done this run)", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, k, s) for k, s in enumerate(series, 1)]
        for f in concurrent.futures.as_completed(futures):
            exc = f.exception()
            if exc is not None:
                with lock:
                    state["bad"] += 1
                    print(f"  a series failed: {type(exc).__name__}: {exc}", flush=True)
    print(f"finished: {state['ok']} series fetched this run, {state['bad']} not extracted", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
