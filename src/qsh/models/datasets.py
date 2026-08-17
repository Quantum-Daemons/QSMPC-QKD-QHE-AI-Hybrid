"""Where the full-scale training corpora live, and how to load them.

The repository ships tiny curated samples under ``data/`` so the demo and the tests run offline.
Those are not what the T2 models are trained on. Full training reads from ``$QSH_DATA_ROOT``, which
defaults to a path on the large volume rather than the system disk — HI-Small alone is 476 MB and
ESA-Mission1 is 3.8 GB, and the root filesystem has well under 200 GB free.

Nothing here downloads anything implicitly. A missing corpus raises with the exact command that
fetches it, because a training script that silently falls back to the 4,000-row demo sample and
reports a metric would be worse than one that fails.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Where full-scale corpora live. Set ``QSH_DATA_ROOT`` to a large volume before training: HI-Small
#: is 476 MB and ESA-Mission1 is 3.8 GB, so a system disk with under 200 GB free is the wrong home
#: for it. The default stays inside the repo so a fresh clone works without configuration, and the
#: path is deliberately ASCII — mount points are frequently non-ASCII and this module is a public
#: file, which must not embed one.
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "data-full"


def data_root() -> Path:
    return Path(os.environ.get("QSH_DATA_ROOT", DEFAULT_DATA_ROOT))


def raw(*parts: str) -> Path:
    return data_root().joinpath("raw", *parts)


def artifacts(*parts: str) -> Path:
    p = data_root().joinpath("artifacts", *parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def require(path: Path, how: str) -> Path:
    """Return ``path`` or raise with the exact command that produces it."""
    if not path.exists():
        raise FileNotFoundError(
            f"missing corpus: {path}\n"
            f"fetch it with:\n    {how}\n"
            "Training will not substitute the bundled demo sample: a metric measured on 4,000 curated "
            "rows is not the metric this model claims."
        )
    return path


# --------------------------------------------------------------------------- #
#  Finance — IBM AMLworld HI-Small (CDLA-Sharing-1.0)
# --------------------------------------------------------------------------- #
AMLWORLD_FETCH = (
    "kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml "
    "-f HI-Small_Trans.csv -p \"$QSH_DATA_ROOT/raw/amlworld\" --unzip"
)


def amlworld_transactions() -> Path:
    return require(raw("amlworld", "HI-Small_Trans.csv"), AMLWORLD_FETCH)


def load_amlworld(nrows: int | None = None):
    """Load HI-Small as a dataframe with the column names normalised.

    The CSV has two columns literally named ``Account`` (sender and receiver); pandas mangles the
    second to ``Account.1``. Renaming them here keeps every downstream reference unambiguous.
    """
    import pandas as pd

    df = pd.read_csv(
        amlworld_transactions(),
        nrows=nrows,
        dtype={
            "From Bank": "str", "To Bank": "str",
            "Account": "str", "Account.1": "str",
            "Receiving Currency": "category", "Payment Currency": "category",
            "Payment Format": "category", "Is Laundering": "int8",
        },
        parse_dates=["Timestamp"],
    )
    return df.rename(columns={
        "From Bank": "from_bank", "Account": "from_account",
        "To Bank": "to_bank", "Account.1": "to_account",
        "Amount Received": "amount_received", "Receiving Currency": "recv_currency",
        "Amount Paid": "amount_paid", "Payment Currency": "pay_currency",
        "Payment Format": "payment_format", "Is Laundering": "label",
        "Timestamp": "timestamp",
    })


# --------------------------------------------------------------------------- #
#  Satellite — ESA Anomaly Dataset, Mission1 (CC-BY-3.0-IGO)
# --------------------------------------------------------------------------- #
ESA_FETCH = (
    "curl -L -o \"$QSH_DATA_ROOT/raw/esa/ESA-Mission1.zip\" "
    "'https://zenodo.org/records/12528696/files/ESA-Mission1.zip?download=1' && "
    "unzip -d \"$QSH_DATA_ROOT/raw/esa\" \"$QSH_DATA_ROOT/raw/esa/ESA-Mission1.zip\""
)

#: ESA-ADB's "lightweight" subset for Mission1. Reported separately from the full channel set
#: because the same method scores event-wise F0.5 of 0.786 here and 0.061 on all channels.
ESA_MISSION1_LIGHTWEIGHT = ("channel_41", "channel_42", "channel_43",
                            "channel_44", "channel_45", "channel_46")


def esa_mission1_dir() -> Path:
    return require(raw("esa", "ESA-Mission1"), ESA_FETCH)


#: Classes a trusted ESA-ADB channel pickle is allowed to construct. Anything else aborts the load.
_PICKLE_ALLOWLIST = {
    ("builtins", "slice"),
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.numeric", "_frombuffer"),
    ("pandas._libs.arrays", "__pyx_unpickle_NDArrayBacked"),
    ("pandas._libs.internals", "_unpickle_block"),
    ("pandas.core.arrays.datetimes", "DatetimeArray"),
    ("pandas.core.frame", "DataFrame"),
    ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.core.indexes.datetimes", "DatetimeIndex"),
    ("pandas.core.indexes.datetimes", "_new_DatetimeIndex"),
    ("pandas.core.indexes.range", "RangeIndex"),
    ("pandas.core.internals.managers", "BlockManager"),
    ("pandas.core.series", "Series"),
}


class _RestrictedUnpickler(__import__("pickle").Unpickler):
    """Unpickler that refuses to import anything outside :data:`_PICKLE_ALLOWLIST`.

    ESA-ADB distributes each channel as a pickled pandas DataFrame, which is the upstream format and
    not something we can choose. Unpickling arbitrary downloaded bytes executes whatever the stream
    asks for, so ``pandas.read_pickle`` on a 3.8 GB archive from the internet is a genuine remote
    code execution surface — the fact that this particular archive is benign is not a property the
    code can rely on.

    ``find_class`` is the single choke point every GLOBAL/STACK_GLOBAL opcode passes through, so
    allowlisting there makes the load safe by construction rather than by prior inspection. The
    allowlist was derived by parsing the opcodes with ``pickletools.genops``, which reads the stream
    without executing it.
    """

    def find_class(self, module: str, name: str):
        if (module, name) in _PICKLE_ALLOWLIST:
            return super().find_class(module, name)
        raise __import__("pickle").UnpicklingError(
            f"blocked pickle import {module}.{name!r}: not in the ESA channel allowlist. "
            "If upstream changed its serialisation, re-derive the allowlist with pickletools.genops "
            "and confirm the new entries are pandas/numpy data classes before adding them."
        )


def read_esa_channel(path: Path):
    """Load one ESA-ADB channel DataFrame through the restricted unpickler."""
    with open(path, "rb") as f:
        return _RestrictedUnpickler(f).load()


def load_esa_channels(names=ESA_MISSION1_LIGHTWEIGHT):
    """Load the named Mission1 channels, unzipping any that are still compressed.

    Returns a single DataFrame indexed by timestamp with one column per channel, forward-filled onto
    the union of their timestamps: ESA telemetry is irregularly sampled per channel, and a detector
    needs them on a common axis.
    """
    import pandas as pd

    base = esa_mission1_dir() / "channels"
    frames = {}
    for name in names:
        raw_file = base / name
        if not raw_file.exists():
            archive = base / f"{name}.zip"
            require(archive, ESA_FETCH)
            import zipfile

            with zipfile.ZipFile(archive) as z:
                z.extract(name, path=base)
        df = read_esa_channel(raw_file)
        series = df.iloc[:, 0] if hasattr(df, "iloc") else df
        frames[name] = pd.Series(series).astype("float32")
    out = pd.DataFrame(frames).sort_index()
    return out.ffill().bfill()


def load_esa_labels(names=ESA_MISSION1_LIGHTWEIGHT, *, anomalies_only: bool = True):
    """Anomaly intervals for the named channels.

    ``anomalies_only`` keeps only rows whose ``anomaly_types.csv`` Category is ``Anomaly``. ESA-ADB
    deliberately also annotates 690 *rare nominal events*, which are unusual but not faults. Scoring
    those as positives would both inflate the apparent anomaly rate and punish a detector for
    correctly ignoring them.
    """
    import pandas as pd

    root = esa_mission1_dir()
    labels = pd.read_csv(root / "labels.csv", parse_dates=["StartTime", "EndTime"])
    labels = labels[labels["Channel"].isin(list(names))]
    if anomalies_only:
        types = pd.read_csv(root / "anomaly_types.csv")
        keep = set(types.loc[types["Category"] == "Anomaly", "ID"])
        labels = labels[labels["ID"].isin(keep)]
    return labels.reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  Medical — MedMNIST v2 BreastMNIST (CC-BY-4.0), fetched by the medmnist package
# --------------------------------------------------------------------------- #
def breastmnist(split: str, *, size: int = 28):
    """BreastMNIST split as ``(images, labels)``, cached under ``$QSH_DATA_ROOT/raw/medmnist``.

    medmnist refuses to create its own ``root`` and raises "Failed to setup the default root
    directory", so the directory is created here before handing the path over.
    """
    from medmnist import BreastMNIST

    root = data_root() / "raw" / "medmnist"
    root.mkdir(parents=True, exist_ok=True)
    ds = BreastMNIST(split=split, download=True, size=size, root=str(root))
    return ds.imgs, ds.labels.ravel().astype(int)


# --------------------------------------------------------------------------- #
#  Telecom — Engelen-corrected CIC-IDS2017 (research use; NOT redistributed)
# --------------------------------------------------------------------------- #
CICIDS_FETCH = (
    "# Research use only, with citation to Engelen, Rimmer & Joosen (WTMC 2021) and\n"
    "# Sharafaldin et al. (2018). Neither the corpus nor weights derived from it are redistributed.\n"
    "#   see https://downloads.distrinet-research.be/WTMC2021\n"
    "# place the extracted CSVs under \"$QSH_DATA_ROOT/raw/cicids2017-fixed/\""
)


def cicids2017_fixed_dir() -> Path:
    return require(raw("cicids2017-fixed"), CICIDS_FETCH)


# --------------------------------------------------------------------------- #
#  Satellite — OPS-SAT-AD (CC-BY-4.0)
# --------------------------------------------------------------------------- #
# The companion to ESA-ADB and the cleaner licence of the two: CC BY 4.0 against ESA-ADB's
# CC BY 3.0 IGO, and 507 KB against 3.6 GB. It is also a different TASK -- 2,123 pre-segmented
# telemetry fragments with per-segment statistical features and a binary anomaly label, against
# ESA-ADB's continuous multichannel series -- so its result is not comparable with the Telemanom
# forecaster's, and neither number should be presented as the other's baseline.
#
# Ruszczak et al., "The OPS-SAT benchmark for detecting anomalies in satellite telemetry",
# Scientific Data 12 (2025), s41597-025-05035-3. Code MIT at kplabs-pl/OPS-SAT-AD.
OPSSAT_FETCH = (
    "curl -L -o \"$QSH_DATA_ROOT/raw/opssat/dataset.csv\" "
    # A concept DOI cannot go here: this is a direct FILE fetch and needs a specific record. The
    # record 12588359 is superseded (both it and 15108715 share concept DOI 10.5281/zenodo.12588358),
    # so if this download starts 404ing, resolve the concept DOI to the current record and update the
    # id rather than assuming the dataset moved.
    "'https://zenodo.org/records/12588359/files/dataset.csv?download=1'"
)

#: The 19 per-segment features the published benchmark computes. Order is fixed here because it is
#: the ONNX input contract: the browser feeds a row in this order and a silent reordering would
#: produce a confident wrong answer rather than an error.
OPSSAT_FEATURES = (
    "sampling", "duration", "len", "mean", "var", "std", "kurtosis", "skew",
    "n_peaks", "smooth10_n_peaks", "smooth20_n_peaks", "diff_peaks", "diff2_peaks",
    "diff_var", "diff2_var", "gaps_squared", "len_weighted", "var_div_duration", "var_div_len",
)


def opssat_csv() -> Path:
    return require(raw("opssat", "dataset.csv"), OPSSAT_FETCH)


def opssat_ad(split: str = "train"):
    """Return ``(X, y)`` for the benchmark's own train/test split.

    The split is the dataset's ``train`` column, not a fresh random one: the published scores are
    reported on that partition, so re-splitting would make the comparison meaningless while still
    looking like a comparison.
    """
    import numpy as np
    import pandas as pd

    df = pd.read_csv(opssat_csv())
    want = 1 if split == "train" else 0
    part = df[df["train"] == want]
    x = part[list(OPSSAT_FEATURES)].to_numpy(dtype=np.float64)
    y = part["anomaly"].to_numpy(dtype=np.int64)
    return x, y


# --------------------------------------------------------------------------- #
#  Telecom (network layer) - 5G-NIDD (CC-BY-4.0)
# --------------------------------------------------------------------------- #
# The corpus this project rejected until Round 47 on a mirror that carries no licence string. The
# authors' repository of record is Fairdata/Metax, where the record reads access_type Open,
# license CC-BY-4.0 and restriction_grounds []. IEEE DataPort is a mirror and its subscription wall
# is not the licence.
#
# It is a DIFFERENT TASK from `telecom_ids.qml` on the same page: 9-class network-layer flow
# classification on real 5G traffic against 8-class QKD-layer attack classification on synthetic
# telemetry. Zero overlap in label space or feature space, so neither score is the other's baseline.
#
# Siriwardhana et al., "Descriptor: 5G Wireless Network Intrusion Detection Dataset (5G-NIDD)",
# IEEE Data Descriptions, doi:10.1109/IEEEDATA.2025.3592888.
#: The fetch is a TWO-STEP TOKEN DANCE and both steps must be one pipeline. Two facts a reader will
#: otherwise rediscover the hard way, both measured at source:
#:
#:   * the authorize endpoint lives on ``etsin.fairdata.fi``, not on ``download.fairdata.fi``.
#:     ``https://download.fairdata.fi/authorize`` is a 404 from nginx - that endpoint does not
#:     exist - and calling ``download?dataset=..&file=..`` without a token returns 400
#:     ``{'token': ['Missing data for required field.']}``;
#:   * the URL that comes back carries a JWT with an ``exp`` claim. Split across two shell commands
#:     run minutes apart, the second fails opaquely, so this is deliberately one pipeline.
#:
#: ``BTS1_BTS2_fields_preserved.zip`` IS THE FILE TO FETCH, and this changed in Round 50. The
#: release also offers ``Combined.zip``, which is what this loader read until then: same 1,215,890
#: rows, but 52 columns against 113. Every one of those 52 is present and informationally identical
#: in the wide export - equal distinct-value counts on all of them - so ``Combined.csv`` is a strict
#: column SUBSET, and one of the columns it omits, ``IdleTime``, is worth 22 points of accuracy on
#: its own. Measured: 0.9829 here against 0.7609 on the narrow export, and removing ``IdleTime``
#: alone from this file reproduces the narrow figure to within 0.001. See
#: ``references/5g-nidd-metric-profile.md`` and the `idle_time_ablation` block that
#: ``scripts/train/telecom_nidd.py`` re-measures every run.
#:
#: The dataset's advertised 7,985,550,132 B across 14 files is real but misleading: 12 of those are
#: raw PCAP captures (BS1.zip 2.4 GB, BS2.zip 1.6 GB) and per-attack splits. The single file needed
#: here is 46,527,823 B compressed and 472,438,689 B as CSV.
NIDD_FETCH = (
    "mkdir -p \"$QSH_DATA_ROOT/raw/5g-nidd/fields_preserved\" && "
    "curl -sS -X POST -H 'Content-Type: application/json' "
    "-d '{\"cr_id\":\"9d13ef28-2ca7-44b0-9950-225359afac65\","
    "\"file\":\"/BTS1_BTS2_fields_preserved.zip\"}' "
    "https://etsin.fairdata.fi/api/download/authorize "
    "| python3 -c 'import json,sys;print(json.load(sys.stdin)[\"url\"])' "
    "| xargs curl -L -o \"$QSH_DATA_ROOT/raw/5g-nidd/fields_preserved.zip\" && "
    "unzip -o -d \"$QSH_DATA_ROOT/raw/5g-nidd/fields_preserved\" "
    "\"$QSH_DATA_ROOT/raw/5g-nidd/fields_preserved.zip\""
)

#: Columns that hold exactly ONE STATE in this release, counting "missing" as a state, and are
#: therefore dropped. 30 of them are 100% null and 9 hold a single non-null value on every row.
#: (Four more all-null columns - `SrcMac`, `DstMac`, `SrcOui`, `DstOui` - are listed under host
#: fingerprints instead, because they would be dropped for that reason even if upstream filled them.)
#:
#: THE SInt*/DInt*/JIT* COLUMNS ARE IN HERE AND THAT SURPRISES PEOPLE. Argus computes inter-packet
#: time, inter-packet distribution and jitter, and their names read like the most discriminative
#: features a flow record could carry - but this release ships them EMPTY, 1,215,890 nulls each. Do
#: not reason about what they would contribute; they contribute nothing because they are not there.
#:
#: The rule is deliberately "one state INCLUDING null", not "one distinct non-null value". `sVid`,
#: `dVid`, `sVlan` and `dVlan` each hold one value (610.0 / `0x0262`) over 9.42% and 0.17% of rows
#: respectively and are null elsewhere, so they carry a real two-state present/absent signal and are
#: KEPT as features. They were dropped under the narrow export's stricter rule; that changed here,
#: and it is worth essentially nothing - best one-vs-rest AUC 0.6207 for `sVlan` and 0.5020 for
#: `dVlan` - so the change is a consistency fix rather than a source of the improvement.
NIDD_CONSTANT_VALUED: tuple[str, ...] = (
    "StdDev", "TotAppByte", "SAppBytes", "DAppBytes", "PCRatio",
    "Retrans", "SrcRetra", "DstRetra", "pRetran",
)
NIDD_ALL_NULL: tuple[str, ...] = (
    "sCo", "dCo", "sMpls", "dMpls", "sAS", "dAS", "iAS", "NStrok", "sNStrok", "dNStrok",
    "SIntPkt", "SIntDist", "SIntPktAct", "SIntActDist", "SIntPktIdl", "SIntIdlDist",
    "DIntPkt", "DIntDist", "DIntPktAct", "DIntActDist", "DIntPktIdl",
    "SrcJitAct", "DstJitter", "DstJitAct", "srcUdata", "dstUdata",
    "sVpri", "dVpri", "SRange", "ERange",
)
NIDD_SINGLE_STATE: tuple[str, ...] = NIDD_CONSTANT_VALUED + NIDD_ALL_NULL

_EMPTY = "empty in this release: 100% null on all 1,215,890 rows, so it carries no state at all"
_ZERO = "constant: a single value on all 1,215,890 rows, so it carries no state at all"

#: Every column removed from the 113, with the reason, as DATA rather than as a code comment - so
#: `feature_spec.json` can publish the reasons and a reader can check them against the corpus.
#:
#: `Unnamed: 0` IS THE ENTRY THAT MATTERS MOST and omitting it is the mistake this file has now seen
#: made twice. It is the CSV's own row index; leaving it in takes accuracy to 0.9997, which looks
#: like a result and is a leak. The first wide fit attempted in Round 50 named `SrcId`, `Rank`, `Seq`
#: and `Offset` but not `Unnamed: 0` and was briefly recorded as confirmation of the published
#: figures. `scripts/train/telecom_nidd.py` refits with all four restored every run so the number is
#: measured rather than remembered.
NIDD_DROPPED: dict[str, str] = {
    # ---- the target and its two leaks ----------------------------------------------------------
    "Label": "target leak: the binary collapse of `Attack Type` (Benign vs Malicious)",
    "Attack Type": "the target",
    "Attack Tool": (
        "target leak: 6 tools over 9 classes, many-to-one in both directions (Hping3 -> ICMPFlood/"
        "SYNFlood/UDPFlood, Nmap -> SYNScan/TCPConnectScan/UDPScan, SlowrateDoS <- Slowloris+"
        "Torshammer). Alone it scores accuracy 0.9615, macro F1 0.5017"),
    # ---- row identifiers: where a row sits in a file, not what the traffic did ------------------
    "Unnamed: 0": (
        "row identifier: the CSV's own index column, 728,316 distinct values. Restoring it alone "
        "takes accuracy from 0.9829 to 0.9997 - it is the single most dangerous column in the file"),
    "SrcId": "row identifier: Argus source identifier, and constant `0.0.0.0` on every row here",
    "Rank": "row identifier: Argus record rank, 467,717 distinct integer values",
    "Seq": "row identifier: Argus flow sequence number, 137,210 distinct integer values",
    "Offset": "row identifier: byte offset into the capture, 1,118,981 distinct integer values",
    "AutoId": "row identifier: Argus auto-increment id, and constant `0` on every row here",
    "Trans": "row identifier: Argus aggregation count, and constant `1` on every row here",
    "StartTime": (
        "row identifier: wall-clock capture timestamp. The attack classes were generated in "
        "sequence, so absolute time separates them by when the capture ran, not by what it saw"),
    "LastTime": "row identifier: wall-clock capture timestamp, same objection as `StartTime`",
    # ---- host fingerprints: which machine, not which behaviour ---------------------------------
    "SrcAddr": (
        "host fingerprint: 121 distinct source addresses on a closed testbed, so the attacker hosts "
        "are enumerable and the model would learn the lab rather than the attack"),
    "DstAddr": "host fingerprint: 170 distinct destination addresses on the same closed testbed",
    "SrcMac": f"host fingerprint (hardware address); also {_EMPTY}",
    "DstMac": f"host fingerprint (hardware address); also {_EMPTY}",
    "SrcOui": f"host fingerprint (MAC vendor prefix); also {_EMPTY}",
    "DstOui": f"host fingerprint (MAC vendor prefix); also {_EMPTY}",
    "Sport": (
        "host fingerprint: 65,550 distinct source ports. Ephemeral port RANGES differ per tool, so "
        "this is a fingerprint of the generator, not a property of the traffic"),
    "Dport": (
        "host fingerprint: 3,668 distinct destination ports, which name the targeted service "
        "directly and each attack class here targeted its own service"),
    "sIpId": "host fingerprint: IP identification field, 65,535 distinct values, per-host counter",
    "dIpId": "host fingerprint: IP identification field, 63,775 distinct values, per-host counter",
    # ---- six columns, one duration --------------------------------------------------------------
    "RunTime": "duplicate of `Dur`, bit-identical on all 1,215,890 rows",
    "Mean": "duplicate of `Dur`, bit-identical on all 1,215,890 rows",
    "Sum": "duplicate of `Dur`, bit-identical on all 1,215,890 rows",
    "Min": "duplicate of `Dur`, bit-identical on all 1,215,890 rows",
    "Max": "duplicate of `Dur`, bit-identical on all 1,215,890 rows",
    # ---- the 39 single-state columns, assembled from the tuples above so the two cannot drift ----
    **{c: _ZERO for c in NIDD_CONSTANT_VALUED},
    **{c: _EMPTY for c in NIDD_ALL_NULL},
}

#: The four dropped row counters, kept as a name so the leak counterfactual in the trainer is
#: reproducible rather than a hand-typed list that could drift from NIDD_DROPPED. `Unnamed: 0` is
#: first because it is the one that does the damage on its own.
NIDD_IDENTIFIERS = ("Unnamed: 0", "Rank", "Seq", "Offset")

#: The 47 usable features, in corpus order: 113 - 3 targets - 9 identifiers - 10 host fingerprints
#: - 5 duplicate durations - 39 single-state columns. Order is fixed here because it is the ONNX
#: input contract: the browser feeds a row in this order and a silent reordering would produce a
#: confident wrong answer rather than an error.
#:
#: `IdleTime` IS THE COLUMN THIS MODEL WAS REFITTED FOR. It is absent from `Combined.csv`, it takes
#: 17 distinct integer values so it cannot be a row counter, and removing it alone from this list
#: drops accuracy from 0.9829 to 0.7616 - the entire gap against the narrow export. It scores
#: one-vs-rest AUC 0.9574 on `SlowrateDoS`, which is what a legitimately dominant feature looks like
#: on an attack whose definition is holding a connection idle.
NIDD_FEATURES = (
    "Flgs", "Dur", "IdleTime", "Proto", "sTos", "dTos", "sDSb", "dDSb", "sTtl", "dTtl",
    "sHops", "dHops", "Cause", "TotPkts", "SrcPkts", "DstPkts", "TotBytes", "SrcBytes",
    "DstBytes", "sMeanPktSz", "dMeanPktSz", "Load", "SrcLoad", "DstLoad", "Loss", "SrcLoss",
    "DstLoss", "pLoss", "SrcGap", "DstGap", "Rate", "SrcRate", "DstRate", "Dir", "State",
    "SrcWin", "DstWin", "sVlan", "dVlan", "sVid", "dVid", "SrcTCPBase", "DstTCPBase",
    "TcpRtt", "SynAck", "AckDat", "TcpOpt",
)

#: The six features `Combined.csv` does not carry, named so the trainer's ablation is reproducible.
#: `IdleTime` is worth the whole difference between the two exports and the other five are worth
#: 0.0006 accuracy between them: dropping all six costs 0.2219 against 0.2213 for `IdleTime` alone.
#:
#: THAT IS REDUNDANCY, NOT EMPTINESS, and the distinction is worth keeping straight because the
#: obvious shorthand is wrong. Individually two of the five are strongly discriminative - `Dir`
#: scores one-vs-rest AUC 0.9811 on `ICMPFlood` and `TcpOpt` 0.9433 on `SlowrateDoS`, both above
#: this project's 0.90 leak-check threshold - and `Flgs` (0.6136), `sVlan` (0.6207) and `dVlan`
#: (0.5020) are weak. What the ablation shows is that whatever the strong two carry is already
#: recoverable from the other 41 columns, so adding them changes nothing. Do not restate this as
#: "the other five carry no signal": measured, two of them do.
NIDD_ABSENT_FROM_COMBINED = ("Flgs", "IdleTime", "Dir", "sVlan", "dVlan", "TcpOpt")

#: The 9 classes, sorted, matching the integer label encoding used everywhere downstream.
NIDD_CLASSES = (
    "Benign", "HTTPFlood", "ICMPFlood", "SYNFlood", "SYNScan",
    "SlowrateDoS", "TCPConnectScan", "UDPFlood", "UDPScan",
)

#: The PUBLISHED ordinal vocabulary for the ten categorical columns, as a literal.
#:
#: This is a literal and not a per-run `astype("category")` for two independent reasons. First,
#: `export_xgboost` converts through `FloatTensorType([None, n_features])`, so an XGBoost
#: `enable_categorical=True` model cannot survive ONNX export at all - the browser gets floats.
#: Second, a vocabulary re-derived per run would silently remap codes the moment a row order or a
#: subsample changed, and the ONNX graph carries thresholds against the OLD codes: the failure is a
#: confident wrong class, not an error. Measured cost of encoding rather than using the category
#: dtype on the narrow export: macro F1 0.926183 encoded against 0.926086 native, i.e. nothing.
#:
#: Values are sorted, so the mapping is reconstructible from the corpus by anyone. An unseen or null
#: value encodes to -1.0, the same sentinel `nan_to_num` uses for missing numerics - `dDSb` is
#: 77.56% null and `sDSb` 0.02%, so "missing" is a real and frequent state here, not an error case.
#:
#: THE WHITESPACE IN `Flgs`, `Dir` AND `TcpOpt` IS PART OF THE VALUE AND IS DELIBERATELY KEPT. Argus
#: writes those three as fixed-width flag fields, so the cell really is `" e s      "` and not
#: `"e s"`. Stripping would be lossless on this release - no two values collide once stripped - but
#: it would put a transformation between the published CSV and the published vocabulary, and the
#: contract this file offers a consumer is "look the raw cell up in the list". One rule, no
#: preprocessing.
NIDD_CATEGORICAL: dict[str, tuple[str, ...]] = {
    "Flgs": (" *        ", " *    V   ", " *    f   ", " e        ", " e    f   ", " e &      ",
             " e *      ", " e d      ", " e g      ", " e i      ", " e r      ", " e s      ",
             " eU       "),
    "Proto": ("arp", "icmp", "ipv6-icmp", "llc", "lldp", "sctp", "tcp", "udp"),
    "sDSb": ("39", "4", "52", "54", "af11", "af12", "af41", "cs0", "cs4", "cs6", "cs7", "ef"),
    "dDSb": ("af11", "af12", "cs0", "cs1", "cs4", "ef"),
    "Cause": ("Shutdown", "Start", "Status"),
    "Dir": ("   ->", "   ?>", "  <-", "  <->", "  <?", "  <?>", "  who"),
    "State": ("ACC", "CON", "ECO", "FIN", "INT", "NRS", "REQ", "RSP", "RST", "TST", "URP"),
    # One value plus a null state. Kept as features, unlike the 39 single-state columns, because
    # present-versus-absent is two states and `sVlan` is present on 9.42% of rows. Worth almost
    # nothing in practice - best one-vs-rest AUC 0.6207 and 0.5020 - and kept for the consistency of
    # the rule rather than for a score.
    "sVlan": ("0x0262",),
    "dVlan": ("0x0262",),
    "TcpOpt": ("      T", "   S", "   S  T", "M", "Mws   T", "MwsS", "MwsS  T"),
}

#: The sentinel a missing numeric or an out-of-vocabulary categorical encodes to. One value for both
#: so a consumer has one rule to implement rather than two.
NIDD_MISSING = -1.0


def nidd_fields_preserved_csv() -> Path:
    return require(raw("5g-nidd", "fields_preserved", "BTS1_BTS2_fields_preserved.csv"), NIDD_FETCH)


def load_5g_nidd(nrows: int | None = None, *, with_identifiers: bool = False):
    """Return ``(X float32, y int64, feature_names, class_names)`` for the WIDE 5G-NIDD export.

    Reads ``BTS1_BTS2_fields_preserved.csv`` (113 columns), not ``Combined.csv`` (52). Same rows,
    same nine classes, same distribution - `Combined.csv` is a strict column subset - but it omits
    ``IdleTime``, and that one column is worth 22 accuracy points here. Round 50 changed this.

    ``with_identifiers`` restores the four dropped row counters as extra trailing columns. It is a
    DIAGNOSTIC, not a training option: it exists so `scripts/train/telecom_nidd.py` can fit the leak
    counterfactual through this same loader and the same encoding, rather than reimplementing the
    drop list beside it and reporting a difference that is partly its own. Nothing exported to ONNX
    may be fitted with it on - the shipped model takes 47 columns.

    THE TRAPS ARE ASSERTED, NOT ASSUMED. Every drop rests on a property of the published file, and
    an upstream re-release that FIXES one of them must fail loudly here rather than silently discard
    real features. The assertions cost one pass over columns that are being dropped anyway.
    """
    import numpy as np
    import pandas as pd

    df = pd.read_csv(nidd_fields_preserved_csv(), nrows=nrows, low_memory=False)

    missing = [c for c in (*NIDD_FEATURES, "Attack Type", *NIDD_IDENTIFIERS, *NIDD_SINGLE_STATE)
               if c not in df.columns]
    if missing:
        raise SystemExit(
            f"BTS1_BTS2_fields_preserved.csv is missing {missing} - the published file has 113 "
            f"columns and this one has {df.shape[1]}. IF THE MISSING NAMES LOOK LIKE THE WHOLE "
            f"FEATURE SET, the likely cause is that Combined.csv (52 columns) is sitting at this "
            f"path: that is the NARROW export this model was refitted away from in Round 50 and it "
            f"does not carry `IdleTime`. Re-fetch rather than working around the difference:\n"
            f"    {NIDD_FETCH}")

    # Trap 1. Six columns, one feature: Dur == RunTime == Mean == Sum == Min == Max, bit-identical.
    for dup in ("RunTime", "Mean", "Sum", "Min", "Max"):
        delta = float((df["Dur"] - df[dup]).abs().max())
        assert delta == 0.0, (
            f"`{dup}` is no longer bit-identical to `Dur` (max|diff| = {delta:.3e}). This corpus "
            f"ships six columns holding one duration; if upstream has changed that, {dup} is now a "
            f"real feature and NIDD_FEATURES must grow rather than keep dropping it.")

    # Trap 2. The 39 single-state columns hold ONE state each, counting null as a state. This is the
    # trap most likely to fire on a re-release, because 30 of them are Argus statistics that a fixed
    # export would fill in - the SIntPkt/DIntPkt inter-packet times and the jitter columns above all.
    # If that happens they are real features and this list must shrink, not keep discarding them.
    for const in NIDD_SINGLE_STATE:
        n_states = int(df[const].nunique(dropna=False))
        assert n_states <= 1, (
            f"`{const}` now takes {n_states} states, so it is no longer the single-state column this "
            f"loader drops it for. Move it into NIDD_FEATURES and re-measure before publishing "
            f"anything trained without it.")

    # Trap 3. `IdleTime` is the reason this loader reads the wide export at all: dropping it alone
    # costs 0.9829 -> 0.7616 accuracy. A re-release that flattened it would silently return this
    # model to the narrow export's score while every label still said 47 features.
    if nrows is None:
        n_idle = int(df["IdleTime"].nunique(dropna=False))
        assert n_idle > 1, (
            "`IdleTime` is single-state in this file, which means it is NOT the column measured in "
            "references/5g-nidd-metric-profile.md and this fit will score about 0.76, not 0.98. "
            "Check that the CSV at the path above is the fields-preserved export.")

    y_names = df["Attack Type"].astype(str)
    unknown = sorted(set(y_names.unique()) - set(NIDD_CLASSES))
    assert not unknown, (
        f"`Attack Type` carries classes NIDD_CLASSES does not list: {unknown}. The class order is the "
        f"published label encoding and the ONNX output contract, so it must be updated deliberately.")
    code = {name: i for i, name in enumerate(NIDD_CLASSES)}
    y = y_names.map(code).to_numpy(dtype=np.int64)

    names = list(NIDD_FEATURES) + (list(NIDD_IDENTIFIERS) if with_identifiers else [])
    frame = df[names].copy()

    # `include="str"` rather than `include=object`. On pandas 3.0.5 both still select all ten
    # columns, but `object` emits Pandas4Warning "This behavior is deprecated and will be removed in
    # a future version. Explicitly pass 'str'" - so this is the spelling that keeps working, and the
    # assertion below is what catches the day it stops selecting anything.
    str_cols = list(frame.select_dtypes(include="str").columns)
    assert set(str_cols) == set(NIDD_CATEGORICAL), (
        f"string columns {str_cols} do not match the published vocabulary keys "
        f"{sorted(NIDD_CATEGORICAL)}. A column that silently became numeric would be ordinal-encoded "
        f"by neither path and reach the model as NaN.")
    for col, vocab in NIDD_CATEGORICAL.items():
        # Trap 4. An unseen value encodes to the SAME sentinel as a missing one, which is the right
        # contract for a consumer and the wrong silence at training time: a stale vocabulary would
        # quietly turn a whole category into "missing". Checked on full reads only, since a --nrows
        # head legitimately sees fewer values than the vocabulary lists, never more.
        if nrows is None:
            unseen = sorted(set(df[col].dropna().astype(str).unique()) - set(vocab))
            assert not unseen, (
                f"`{col}` carries values NIDD_CATEGORICAL does not list: {unseen}. They would encode "
                f"to {NIDD_MISSING} alongside genuine nulls. Extend the vocabulary by APPENDING - "
                f"reordering it remaps codes the exported ONNX thresholds were fitted against.")
        index = {v: float(i) for i, v in enumerate(vocab)}
        frame[col] = frame[col].astype(str).map(index).astype("float64")

    x = np.nan_to_num(frame.to_numpy(dtype=np.float32), nan=NIDD_MISSING,
                      posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
    return x, y, names, list(NIDD_CLASSES)
