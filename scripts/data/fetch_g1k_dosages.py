#!/usr/bin/env python
"""Fetch real 1000 Genomes phase-3 effect-allele dosages for an rsID-keyed PGS Catalog score.

Maintainer-only provenance script, in the same spirit as ``scripts/prep_real_kaggle_data.py``: it is
not runtime code and nothing in the pipeline calls it. It writes, for the score it is given:

  * the full matrix, every one of the 2,504 phase-3 individuals (``data/genomics/<dosage file>``,
    gitignored: DATA_LICENSES.md section 4 justifies committing a slice, not the matrix);
  * the committed CI slice beside it (``<stem>_slice.tsv``): the first 48 AFR and the first 49 EUR
    individuals of the panel in sample-ID order (all ACB and GBR), 97 people;

and ``scripts/fetch_gnomad_af.py`` reads the same records for the overlay's gnomAD frequencies.

SOURCE. Ensembl REST, ``https://rest.ensembl.org/variation/human/{rsid}?genotypes=1;pops=1`` (GRCh38).
Its ``genotypes`` list holds the 1000 Genomes phase 3 genotype of each phase-3 individual, named
``1000GENOMES:phase_3:<sample>``; the same list also carries Gambian Genome Variation Project (GGVP)
samples, which are dropped. The population of each sample comes from the bundled
``data/genomics/g1k_panel.tsv`` (IGSR's phase-3 panel file), so no other service is read. Cite the
genotypes as The 1000 Genomes Project Consortium, *A global reference for human genetic variation*,
Nature 526:68-74 (2015), doi:10.1038/nature15393. IGSR attaches no licence instrument to them
(DATA_LICENSES.md section 4).

NOTHING IS FILLED IN. The fetcher this replaces summed effect-allele matches per sample and wrote
``.get(sample, 0)`` for any sample it had no call for, so an allele it could not match, or a variant
Ensembl returned no 1000 Genomes calls for, became a column of zeros: the retired PGS000001 dosage file
carries five constant columns that way (rs1432679 is 0 in all 400 individuals against a gnomAD
non-Finnish European frequency of 0.56). Here a variant is scored only if every one of these holds,
and otherwise it is written to the file header as ``#unscored<TAB>rsid<TAB>reason`` and left out:

  1. Ensembl maps the rsID (or the current name it was merged into) to exactly one GRCh38 autosome;
  2. every individual of the panel has a 1000 Genomes genotype, made of Ensembl's alleles for it;
  3. the score's two alleles are found among the alleles those genotypes carry - for an SNV as
     written or as their complements (the other strand); for an indel by its length change, because
     Ensembl writes indels expanded across their repeat (the score's VCF-style ``A``/``AG`` is
     ``GGGG``/``GGGGG``);
  4. exactly one such reading puts the effect allele's frequency over the panel's 503 European
     individuals within ``AF_TOLERANCE`` of the score's own published ``allelefrequency_effect`` (its
     UK Biobank estimate); an A/T or C/G SNV whose two readings both pass is read as written, on the
     forward strand, and the header says so;
  5. the column is not constant across the panel.

``loaders.load_genomic_prs`` and the trainer accept a score variant without a dosage column only if
the file declares it ``#unscored``, and refuse any other gap.

Usage:
    python scripts/data/fetch_g1k_dosages.py                       # the active score
    python scripts/data/fetch_g1k_dosages.py --cache /tmp/ensembl  # keep / reuse raw responses
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENSEMBL = "https://rest.ensembl.org"

#: Ensembl asks for at most 15 requests per second. The genotype endpoint is slow (tens of seconds a
#: record when it is busy), so this is far inside that.
REQUEST_INTERVAL = 0.3

#: How far the effect allele's frequency over the panel's 503 European individuals may sit from the
#: score's published one before a reading is rejected. 0.10 is the score's own authors' criterion:
#: building PGS000072 they "excluded any [variant] not in a 1000 Genomes reference population or with
#: minor allele frequencies (MAF) that differed by >0.10" (Graff et al., Nat Commun 12:970, 2021,
#: Methods), and they dropped A/T and C/G variants with MAF >= 0.45, so a palindromic variant's two
#: readings sit at least 0.1 apart. A strand or allele error moves the frequency to one minus itself.
#: On PGS000072 the largest difference among the 160 SNVs whose two alleles alone fix the strand is
#: 0.069 (rs2180341); this script prints it on every run.
AF_TOLERANCE = 0.1

#: The CI slice: the first individuals of each group in sample-ID order. AFR and EUR because the
#: finding is the gap between them, and at least 20 each because the trainer reports a group only at
#: n >= 20.
SLICE_GROUPS = (("AFR", 48), ("EUR", 49))

AUTOSOMES = {str(i) for i in range(1, 23)}
COMPLEMENT = str.maketrans("ACGT", "TGCA")
G1K_PREFIX = "1000GENOMES:phase_3:"


def get_record(rsid: str, cache: Path | None = None, retries: int = 6) -> dict:
    """Ensembl's variation record with genotypes and population frequencies, cached when asked.

    A cached error response is refetched, not reused: a server-side 500 is not a fact about the
    variant. A 400 is (Ensembl answers 400 for an rsID it does not know) and is returned as such.
    """
    path = cache / f"{rsid}.json.gz" if cache else None
    if path and path.exists():
        rec = json.loads(gzip.decompress(path.read_bytes()))
        if "http_error" not in rec:
            return rec
    url = f"{ENSEMBL}/variation/human/{rsid}?genotypes=1;pops=1"
    body = b""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Content-Type": "application/json",
                                                       "User-Agent": "qsh-provenance/1.0"})
            with urllib.request.urlopen(req, timeout=180) as r:   # noqa: S310 - fixed Ensembl URL
                body = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 400:
                body = json.dumps({"error": f"HTTP 400: {e.read().decode(errors='replace')[:200]}"}).encode()
                break
            if attempt == retries - 1:
                raise
            time.sleep(int(e.headers.get("Retry-After", 2 ** attempt)))
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(body))
    time.sleep(REQUEST_INTERVAL)
    return json.loads(body)


def read_score(path: Path) -> list[dict[str, str]]:
    """Rows of a PGS Catalog scoring file that carries an ``rsID`` column."""
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if header is None:
            header = parts
            continue
        rows.append(dict(zip(header, parts, strict=True)))
    for col in ("rsID", "effect_allele", "other_allele", "allelefrequency_effect"):
        assert header and col in header, f"{path.name} lacks the '{col}' column this fetcher keys on"
    return rows


def read_panel(path: Path) -> dict[str, tuple[str, str]]:
    """sample -> (population, super population), from the bundled phase-3 panel."""
    out: dict[str, tuple[str, str]] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#") or line.startswith("sample"):
            continue
        f = line.split("\t")
        out[f[0]] = (f[1], f[2])
    return out


def _strip_anchor(ea: str, oa: str) -> tuple[str, str]:
    """A VCF-style indel without its shared first base: ``A``/``AG`` -> ``""``/``G``."""
    if len(ea) != len(oa) and ea[:1] == oa[:1]:
        return ea[1:], oa[1:]
    return ea, oa


def _is_insertion_of(longer: str, shorter: str, seq: str) -> bool:
    """True if inserting a rotation of ``seq`` somewhere in ``shorter`` gives ``longer``.

    The rotation is what lets a VCF-anchored repeat indel match Ensembl's expanded form: the same
    ``GT`` deletion sits at different offsets in ``GGT``/``G`` and ``GTGTG``/``GTG``.
    """
    if len(longer) - len(shorter) != len(seq) or not seq:
        return False
    rotations = {seq[i:] + seq[:i] for i in range(len(seq))}
    return any(longer[:i] + longer[i + len(seq):] == shorter
               for i in range(len(shorter) + 1) if longer[i:i + len(seq)] in rotations)


def autosomal_mappings(record: dict) -> list[dict]:
    """The record's GRCh38 mappings to an autosome (patches and other contigs excluded)."""
    return [m for m in record.get("mappings", [])
            if m.get("assembly_name") == "GRCh38" and str(m.get("seq_region_name")) in AUTOSOMES]


def resolve(row: dict[str, str], record: dict, panel: dict[str, tuple[str, str]]
            ) -> tuple[str, dict[str, int], str] | tuple[None, None, str]:
    """(effect allele as Ensembl writes it, its count per panel individual, how), or a reason.

    No individual's count is ever defaulted: a missing or foreign genotype rejects the variant.
    """
    if "error" in record or "http_error" in record:
        return None, None, f"Ensembl returned no record ({record.get('error') or record.get('http_error')})"
    maps = autosomal_mappings(record)
    if len(maps) != 1:
        why = record.get("failed") or f"{len(maps)} GRCh38 autosomal mappings"
        return None, None, f"Ensembl maps it to no single GRCh38 autosomal site ({why})"
    alleles = maps[0]["allele_string"].split("/")
    calls = {g["sample"][len(G1K_PREFIX):]: g["genotype"].replace("/", "|").split("|")
             for g in record.get("genotypes", []) if str(g.get("sample", "")).startswith(G1K_PREFIX)}
    have = [s for s in panel if s in calls]
    if len(have) < len(panel):
        return None, None, (f"Ensembl returns 1000 Genomes phase 3 genotypes for {len(have)} of the "
                            f"{len(panel)} individuals")
    bad = next((s for s in panel if len(calls[s]) != 2 or not set(calls[s]) <= set(alleles)), None)
    if bad:
        return None, None, f"genotype {'|'.join(calls[bad])} of {bad} is not made of Ensembl's alleles"
    observed = {a for s in panel for a in calls[s]}
    eur = [s for s in panel if panel[s][1] == "EUR"]

    def eur_freq(allele: str) -> float:
        return sum(calls[s].count(allele) for s in eur) / (2 * len(eur))

    ea, oa = row["effect_allele"], row["other_allele"]
    published = float(row["allelefrequency_effect"])
    snv = len(ea) == 1 and len(oa) == 1
    palindromic = snv and {ea, oa} in ({"A", "T"}, {"C", "G"})
    # Every reading of the score's two alleles among the alleles the genotypes carry.
    readings: list[tuple[str, str, str]] = []
    if snv:
        if {ea, oa} <= observed:
            readings.append((ea, oa, "SNV, alleles as written"))
        comp = (ea.translate(COMPLEMENT), oa.translate(COMPLEMENT))
        if set(comp) <= observed:
            readings.append((comp[0], comp[1], "SNV, alleles read on the other strand"))
    else:
        e_core, o_core = _strip_anchor(ea, oa)
        if e_core and o_core:
            return None, None, f"indel {ea}/{oa} is not one anchored insertion or deletion"
        seq, effect_longer = (e_core, True) if e_core else (o_core, False)
        for e_ens in sorted(observed):
            for o_ens in sorted(observed):
                e_s, o_s = e_ens.replace("-", ""), o_ens.replace("-", "")
                longer, shorter = (e_s, o_s) if effect_longer else (o_s, e_s)
                if e_ens != o_ens and _is_insertion_of(longer, shorter, seq):
                    readings.append((e_ens, o_ens, f"indel {ea}/{oa} read as Ensembl's {e_ens}/{o_ens}"))
    if not readings:
        return None, None, (f"alleles {ea}/{oa} are not among those the 1000 Genomes genotypes carry "
                            f"({'/'.join(sorted(observed))})")
    fits = [(e, how, f) for e, _o, how in readings
            if abs((f := eur_freq(e)) - published) <= AF_TOLERANCE]
    if not fits:
        seen = ", ".join(f"{e}: {eur_freq(e):.4f}" for e, _o, _h in readings)
        return None, None, (f"no reading puts the effect allele's frequency over the {len(eur)} EUR "
                            f"individuals within {AF_TOLERANCE} of the published {published} ({seen})")
    named = record.get("name", row["rsID"])
    merged = f"; Ensembl's current name is {named}" if named != row["rsID"] else ""
    if len(fits) > 1 and not (palindromic and fits[0][1] == "SNV, alleles as written"):
        return None, None, f"{len(fits)} readings of {ea}/{oa} fit the published frequency"
    effect, how, f = fits[0]
    if len(fits) > 1:
        how += "; A/T or C/G and both strands fit, so read as written on the forward strand"
    counts = {s: calls[s].count(effect) for s in panel}
    if len(set(counts.values())) == 1:
        return None, None, f"every individual carries {next(iter(counts.values()))} copies: a constant column"
    return effect, counts, f"{how} (EUR {f:.4f}, published {published}){merged}"


def main() -> int:
    sys.path.insert(0, str(REPO / "src"))
    from qsh.data.loaders import _PRS_FILES, ACTIVE_PRS  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score", choices=sorted(_PRS_FILES), default=ACTIVE_PRS)
    ap.add_argument("--panel", type=Path, default=REPO / "data" / "genomics" / "g1k_panel.tsv")
    ap.add_argument("--cache", type=Path, default=None, help="directory for the raw Ensembl records")
    ap.add_argument("--out", type=Path, default=None, help="defaults to the score's dosage file")
    a = ap.parse_args()

    score_file, dosage_file = _PRS_FILES[a.score]
    score_path = REPO / "data" / "genomics" / score_file
    out = a.out or REPO / "data" / "genomics" / dosage_file
    rows = read_score(score_path)
    panel = read_panel(a.panel)
    samples = sorted(panel)
    print(f"{a.score}: {len(rows)} variants from {score_path.name}; panel {len(samples)} individuals")

    columns: dict[str, dict[str, int]] = {}
    unscored: dict[str, str] = {}
    resolved: dict[str, str] = {}
    gaps: list[tuple[float, str]] = []
    for i, row in enumerate(rows, 1):
        rsid = row["rsID"]
        record = get_record(rsid, a.cache)
        effect, col, how = resolve(row, record, {s: panel[s] for s in samples})
        if col is None:
            unscored[rsid] = how
            print(f"  [{i:3d}/{len(rows)}] {rsid}: UNSCORED - {how}", file=sys.stderr)
            continue
        columns[rsid], resolved[rsid] = col, how
        # The calibration figure for AF_TOLERANCE: SNVs whose two alleles alone fix the strand.
        ea, oa = row["effect_allele"], row["other_allele"]
        if len(ea) == len(oa) == 1 and {ea, oa} not in ({"A", "T"}, {"C", "G"}):
            eur = [s for s in samples if panel[s][1] == "EUR"]
            gaps.append((abs(sum(col[s] for s in eur) / (2 * len(eur))
                             - float(row["allelefrequency_effect"])), rsid))
        print(f"  [{i:3d}/{len(rows)}] {rsid}: {how}")

    keys = [r["rsID"] for r in rows if r["rsID"] in columns]
    assert keys, "no variant resolved - refusing to write an empty dosage table"
    print(f"\nscored {len(keys)} of {len(rows)}; unscored {len(unscored)}")
    if gaps:
        top, top_rsid = max(gaps)
        print(f"largest |1000 Genomes EUR - published| effect-allele frequency over {len(gaps)} SNVs whose "
              f"two alleles fix the strand: {top:.4f} ({top_rsid})")

    def write(path: Path, chosen: list[str], title: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            fh.write(f"# {title}\n")
            fh.write(f"# Real 1000 Genomes phase 3 effect-allele dosages (0/1/2) for the {score_file} "
                     f"variants, keyed by the scoring file's rsID; {len(chosen)} of {len(samples)} "
                     f"individuals x {len(keys)} of its {len(rows)} variants.\n")
            fh.write("# Source: Ensembl REST /variation/human/{rsid}?genotypes=1;pops=1 (GRCh38), "
                     "1000GENOMES:phase_3 samples only; populations from data/genomics/g1k_panel.tsv. "
                     "Written by scripts/data/fetch_g1k_dosages.py, which fills in no genotype.\n")
            fh.write("# Cite: The 1000 Genomes Project Consortium, A global reference for human genetic "
                     "variation, Nature 526:68-74 (2015), doi:10.1038/nature15393. IGSR attaches no "
                     "licence instrument to these genotypes; see DATA_LICENSES.md section 4.\n")
            for rsid in (r["rsID"] for r in rows):
                if rsid in unscored:
                    fh.write(f"#unscored\t{rsid}\t{unscored[rsid]}\n")
            # Every reading other than a plain SNV as written: indels, the other strand, A/T or C/G
            # read by convention, and merged rsIDs.
            for rsid in keys:
                how = resolved[rsid]
                if not how.startswith("SNV, alleles as written (") or "current name" in how:
                    fh.write(f"#resolved\t{rsid}\t{how}\n")
            fh.write("\t".join(["sample", "population", "super_population", *keys]) + "\n")
            for s in chosen:
                pop, sup = panel[s]
                fh.write("\t".join([s, pop, sup, *(str(columns[k][s]) for k in keys)]) + "\n")
        print(f"wrote {path}: {len(chosen)} individuals x {len(keys)} variants")

    write(out, samples, f"Full panel for {score_file}: local and gitignored (DATA_LICENSES.md section 4).")
    chosen: list[str] = []
    for group, n in SLICE_GROUPS:
        chosen += [s for s in samples if panel[s][1] == group][:n]
    slice_path = out.with_name(f"{out.stem}_slice.tsv")
    write(slice_path, chosen, "CI slice, committed: the first " + " and the first ".join(
        f"{n} {g}" for g, n in SLICE_GROUPS) + " individuals of the panel in sample-ID order, so CI "
          "reads real genotypes. Not a source for published figures: models/metrics.json comes from "
          "the full panel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
