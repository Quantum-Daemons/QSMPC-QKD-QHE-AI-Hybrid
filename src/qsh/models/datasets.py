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
