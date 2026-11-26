#!/usr/bin/env python3
"""The 3D-divergence predictor's inputs, exactly as pre-registered (Round 86, batch 6; option D1).

`references/genomics-3d-divergence-preregistration.md` fixed everything below before it ran: the split of the 2,457
individuals (the test set holds the 30 the genomics payload ships), the twelve windows (the divergent windows with the
largest spread of divergence over the fit individuals, none overlapping another), ten variants a window (by their own
in-silico effect, never by association with the target), and each variant's 1000 Genomes genotypes read through
Ensembl as a carrier indicator, with nothing filled in.

Reads `$QSH_DATA_ROOT/raw/akita-1000g-zenodo13900918/` (Zenodo 13900918, CC BY 4.0: `anc_window_spearman.csv` and
`divergent_windows.zip`), caches each Ensembl response beside them (`ensembl_cache/`), and writes
`$QSH_DATA_ROOT/artifacts/akita_d1_inputs.npz` (C: carrier indicators, Y: divergence, part: 0 fit, 1 validation, 2
test) with its `.json` (the individuals, windows, variants, declarations and counts). Local only: the matrix holds
2,457 individuals' genotypes, which DATA_LICENSES section 4 does not justify committing.

Usage: QSH_DATA_ROOT=<data root> python -I scripts/data/akita_d1_inputs.py
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import io
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from qsh.models import datasets  # noqa: E402

PREREG = REPO / "references" / "genomics-3d-divergence-preregistration.md"
PREREG_SHA256 = "2e93f4fab32d8f2852c7b5be59284460e3d5488533adc2cd491e0261222884c2"  # with its two addenda (empty cells; NA effects)
WINDOW = 1_048_576            # 1 Mb windows sliding by half
N_WINDOWS = 12
PER_WINDOW = 10
PREFETCH = 14                 # candidates a window fetched ahead, a few more than the ten kept
SHARE = 0.15


def _fetcher():
    spec = importlib.util.spec_from_file_location("fetch_g1k_dosages", REPO / "scripts" / "data" / "fetch_g1k_dosages.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def overlap(chrom: str, pos: int, cache: Path, retries: int = 6) -> list[dict]:
    """Ensembl's variants at one GRCh38 position (`/overlap/region`), cached like the fetcher's records."""
    path = cache / f"overlap_{chrom}_{pos}.json.gz"
    if path.exists():
        return json.loads(gzip.decompress(path.read_bytes()))
    url = f"https://rest.ensembl.org/overlap/region/human/{chrom}:{pos}-{pos}?feature=variation"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Content-Type": "application/json", "User-Agent": "qsh-provenance/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:     # noqa: S310 - fixed Ensembl URL
                body = r.read()
            break
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    cache.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(body))
    time.sleep(0.3)
    return json.loads(body)


def ref_alt_ok(hit: dict, v: dict) -> bool:
    return int(hit.get("end", -1)) == int(v["pos"]) and v["ref"] in hit.get("alleles", []) and v["alt"] in hit.get("alleles", [])


def stratified(pool: list[int], labels: list[str], k: int, rng: np.random.Generator) -> list[int]:
    """`k` of `pool`, each super-population's share by largest remainder, drawn within it."""
    groups: dict[str, list[int]] = {}
    for i in pool:
        groups.setdefault(labels[i], []).append(i)
    names = sorted(groups)
    exact = {g: k * len(groups[g]) / len(pool) for g in names}
    take = {g: int(exact[g]) for g in names}
    for g in sorted(names, key=lambda g: (-(exact[g] - take[g]), g))[: k - sum(take.values())]:
        take[g] += 1
    out: list[int] = []
    for g in names:
        out += sorted(rng.choice(groups[g], size=take[g], replace=False).tolist())
    return sorted(out)


def resolve_snv(chrom: str, pos: int, ref: str, alt: str, record: dict, ids: list[str], f) -> tuple[dict | None, str]:
    """Carrier indicators for one forward-strand GRCh38 SNV, or the reason there are none; nothing is filled in."""
    if "error" in record or "http_error" in record:
        return None, f"Ensembl returned no record ({record.get('error') or record.get('http_error')})"
    maps = [m for m in f.autosomal_mappings(record) if int(m.get("start", -1)) == pos and str(m.get("seq_region_name")) == chrom]
    if len(maps) != 1:
        return None, f"no single GRCh38 mapping at {chrom}:{pos}"
    alleles = maps[0]["allele_string"].split("/")
    if ref not in alleles or alt not in alleles:
        return None, f"alleles {ref}/{alt} are not Ensembl's {maps[0]['allele_string']}"
    calls = {g["sample"][len(f.G1K_PREFIX):]: g["genotype"].replace("/", "|").split("|")
             for g in record.get("genotypes", []) if str(g.get("sample", "")).startswith(f.G1K_PREFIX)}
    have = [s for s in ids if s in calls]
    if len(have) < len(ids):
        return None, f"Ensembl returns 1000 Genomes phase 3 genotypes for {len(have)} of the {len(ids)} individuals"
    bad = next((s for s in ids if len(calls[s]) != 2 or not set(calls[s]) <= set(alleles)), None)
    if bad:
        return None, f"genotype {'|'.join(calls[bad])} of {bad} is not made of Ensembl's alleles"
    carrier = {s: int(alt in calls[s]) for s in ids}
    if len(set(carrier.values())) == 1:
        return None, "every individual has the same carrier status: a constant column"
    return carrier, f"SNV {ref}>{alt} on the forward strand; {sum(carrier.values())} of {len(ids)} carry {alt}"


def main() -> int:
    started = time.time()
    got = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if got != PREREG_SHA256:
        raise SystemExit(f"{PREREG.name} is not the committed pre-registration ({got})")
    raw = datasets.raw("akita-1000g-zenodo13900918")
    cache = raw / "ensembl_cache"
    f = _fetcher()

    # ---- individuals and the split ---------------------------------------------------------------------------
    with (raw / "anc_window_spearman.csv").open() as fh:
        r = csv.reader(fh)
        sup = next(r)[2:]
        next(r)
        next(r)
        ids = next(r)[2:]
        next(r)
        divergent_rows: dict[tuple[str, int], np.ndarray] = {}
        with zipfile.ZipFile(raw / "divergent_windows.zip") as z:
            exp = list(csv.DictReader(io.TextIOWrapper(z.open("divergent_windows/divergent_windows_exp_distributions.txt"))))
        divergent = {(e["chr"], int(e["windowStartPos"])) for e in exp}
        for row in r:
            key = (row[0], int(row[1]))
            if key in divergent:
                # An empty cell is no value (the addendum): NaN, left out of every figure, never filled in.
                divergent_rows[key] = 1.0 - np.asarray([float(v) if v != "" else np.nan for v in row[2:]], dtype=np.float64)
    n = len(ids)
    assert len(divergent_rows) == len(divergent) == 392, (len(divergent_rows), len(divergent))
    payload = json.loads((REPO / "web" / "public" / "data" / "genomics" / "pgs000783_breast_cancer.json").read_text())
    shipped = [ids.index(s) for s in payload["sample_ids"]]
    k = int(round(SHARE * n))
    rng = np.random.default_rng(0)
    rest = [i for i in range(n) if i not in set(shipped)]
    test = sorted(shipped + stratified(rest, sup, k - len(shipped), rng))
    rest = [i for i in rest if i not in set(test)]
    val = stratified(rest, sup, k, rng)
    part = np.zeros(n, dtype=np.int8)
    part[val] = 1
    part[test] = 2
    fit = np.flatnonzero(part == 0)

    # ---- the twelve windows ----------------------------------------------------------------------------------
    spread = {key: float(np.nanstd(y[fit])) for key, y in divergent_rows.items()}
    chosen: list[tuple[str, int]] = []
    for key in sorted(spread, key=lambda kk: (-spread[kk], kk[0], kk[1])):
        if any(c[0] == key[0] and abs(c[1] - key[1]) < WINDOW for c in chosen):
            continue
        chosen.append(key)
        if len(chosen) == N_WINDOWS:
            break
    Y = np.stack([divergent_rows[key] for key in chosen], axis=1)
    print(f"{n} individuals: fit {len(fit)}, validation {len(val)}, test {len(test)}; windows {chosen}", flush=True)

    # ---- ten variants a window -------------------------------------------------------------------------------
    by_window: dict[tuple[str, int], list[dict]] = {key: [] for key in chosen}
    no_effect: dict[tuple[str, int], int] = {}
    with zipfile.ZipFile(raw / "divergent_windows.zip") as z:
        for v in csv.DictReader(io.TextIOWrapper(z.open("divergent_windows/divergent_windows_10AF_variants.txt")),
                                delimiter="\t"):
            key = (v["chr"], int(v["window"]))
            if key in by_window:
                if v["1-spearman"] in ("NA", ""):          # addendum 2: no effect, no rank
                    no_effect[key] = no_effect.get(key, 0) + 1
                    continue
                by_window[key].append(v)
    # Warm the cache four requests at a time (Ensembl's genotype endpoint answers in tens of seconds a record, and
    # allows 15 requests a second): the first PREFETCH candidates of each window, in rank order. The choice below is
    # still made one variant at a time in rank order, from the cache, so the prefetch changes no variant chosen.
    from concurrent.futures import ThreadPoolExecutor

    wanted = []
    for key in chosen:
        ranked = sorted(by_window[key], key=lambda v: (-float(v["1-spearman"]), int(v["pos"])))
        for v in [v for v in ranked if len(v["ref"]) == 1 and len(v["alt"]) == 1][:PREFETCH]:
            chrom, pos = v["chr"].removeprefix("chr"), int(v["pos"])
            hits = [h for h in overlap(chrom, pos, cache) if str(h.get("id", "")).startswith("rs")
                    and int(h.get("start", -1)) == pos and ref_alt_ok(h, v)]
            if len(hits) == 1:
                wanted.append(hits[0]["id"])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda rid: f.get_record(rid, cache), wanted))
    print(f"prefetched {len(wanted)} records", flush=True)

    variants, declared, columns = [], [], []
    for w_index, key in enumerate(chosen):
        ranked = sorted(by_window[key], key=lambda v: (-float(v["1-spearman"]), int(v["pos"])))
        kept = 0
        for v in ranked:
            if kept == PER_WINDOW:
                break
            chrom, pos, ref, alt = v["chr"].removeprefix("chr"), int(v["pos"]), v["ref"], v["alt"]
            if len(ref) != 1 or len(alt) != 1:
                declared.append({"window": w_index, "chr": chrom, "pos": pos, "reason": "not an SNV"})
                continue
            hits = [h for h in overlap(chrom, pos, cache) if str(h.get("id", "")).startswith("rs")
                    and int(h.get("start", -1)) == pos and int(h.get("end", -1)) == pos
                    and ref in h.get("alleles", []) and alt in h.get("alleles", [])]
            if len(hits) != 1:
                declared.append({"window": w_index, "chr": chrom, "pos": pos,
                                 "reason": f"{len(hits)} rsIDs at this position with these alleles"})
                continue
            rsid = hits[0]["id"]
            carrier, how = resolve_snv(chrom, pos, ref, alt, f.get_record(rsid, cache), ids, f)
            if carrier is None:
                declared.append({"window": w_index, "chr": chrom, "pos": pos, "rsid": rsid, "reason": how})
                continue
            columns.append(np.asarray([carrier[s] for s in ids], dtype=np.int8))
            variants.append({"window": w_index, "rsid": rsid, "chr": chrom, "pos": pos, "ref": ref, "alt": alt,
                             "in_silico_1_minus_spearman": float(v["1-spearman"]), "how": how})
            kept += 1
            print(f"  window {w_index} {key}: {rsid} {how}", flush=True)
        if kept < PER_WINDOW:
            print(f"  window {w_index}: only {kept} variants resolved", flush=True)
    C = np.stack(columns, axis=1)
    out = datasets.artifacts("akita_d1_inputs.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, C=C, Y=Y, part=part)
    meta = {
        "preregistration": {"file": str(PREREG.relative_to(REPO)), "sha256": PREREG_SHA256},
        "source": {"zenodo": "10.5281/zenodo.13900918", "licence": "CC-BY-4.0",
                   "files": ["anc_window_spearman.csv", "divergent_windows.zip"]},
        "ids": ids, "super_population": sup, "shipped": payload["sample_ids"],
        "counts": {"individuals": n, "fit": int(len(fit)), "validation": int(len(val)), "test": int(len(test)),
                   "variants": len(variants), "declared": len(declared)},
        "windows": [{"chr": c, "start": s, "fit_sd": round(spread[(c, s)], 10), "variants_without_an_effect": no_effect.get((c, s), 0),
                     "no_value": [ids[i] for i in np.flatnonzero(np.isnan(divergent_rows[(c, s)]))]} for c, s in chosen],
        "variants": variants, "declared": declared,
        "wall_clock_s": round(time.time() - started, 1),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {out} ({C.shape[1]} variants) in {meta['wall_clock_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
