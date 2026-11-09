#!/usr/bin/env python
"""Window statistics from the raw 5G-NIDD captures, measured across base stations.

WHAT THIS ASKS
    The flow model (`scripts/train/telecom_nidd.py`, 46 Argus columns) has two weak spots across base
    stations: SYNFlood trained on the second station scores F1 0.22 on the first, and without its TTL and
    hop columns UDPScan scores 0.05. Does anything computed from the packets themselves do better?

THE UNIT
    A conversation, an unordered pair of addresses, at a one-second tick. A row is written for each
    second in which the pair exchanged a packet, and it reads only packets at or before that tick, over
    windows of one and ten seconds, so a detector could compute it live. Direction is relative to the
    address that sent the pair's first packet. Statistics over several time windows per channel are the
    idea of Kitsune's AfterImage (Mirsky et al., NDSS 2018), here as plain sliding windows.

THE LABELS follow the dataset's own rule, which `attacker_pairs` reads and checks from its per-attack
    files: in each attack session one address pair is the attack and everything else is Benign. The second
    base station's attacker in the first station's UDP-flood capture is left out, since those packets are
    the second capture's (see references/5g-nidd-metric-profile.md).

THE FILES come from the dataset's record, doi:10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361 (CC BY 4.0), and
    are not part of this repository: BS1_GTP_removed.zip, "BS2 GTP removed.zip", BS1_each_attack_csv.zip and
    BS2_each_attack_csv.zip, unpacked under $QSH_DATA_ROOT/raw/5g-nidd/pcap as bs1/, bs2/, each_attack_csv_bs1/
    and each_attack_csv_bs2/.

Usage:
    QSH_DATA_ROOT=$PWD/data-full python scripts/research/nidd_window_features.py [--seeds 5]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import socket
import struct
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "references" / "data" / "5g-nidd-window-features.json"

VICTIM = "10.41.150.68"
#: The second station's attacker, whose packets are also in the first station's UDP-flood capture.
COPIED_SOURCE = "10.155.15.7"
COPIED_CAPTURE = "UDPflood_BS1"

CLASSES = ["Benign", "HTTPFlood", "ICMPFlood", "SYNFlood", "SYNScan", "SlowrateDoS", "TCPConnectScan", "UDPFlood", "UDPScan"]

#: Capture name to the per-attack CSV that labels it.
LABEL_FILE = {
    "Goldeneye_BS1": "Goldeneye1", "ICMPflood_BS1": "ICMPFlood1", "SSH_BS1": "SSH1", "SYNflood_BS1": "SYNFlood1",
    "SYNScan_BS1": "SYNScan1", "Slowloris_BS1": "Slowloris1", "TCPConnect_BS1": "TCPConnect1", "Torshammer_BS1": "Torshammer1",
    "UDPflood_BS1": "UDPFlood1", "UDPScan_BS1": "UDPScan1", "Goldeneye_BS2": "Goldeneye2", "ICMPflood_BS2": "ICMPFlood2",
    "SSH_BS2": "SSH2", "SYNflood_BS2": "SYNFlood2", "SYNscan_BS2": "SYNScan2", "Slowloris_BS2": "Slowloris2",
    "TCPConnect_BS2": "TCPConnect2", "Torshammer_BS2": "Torshammer2", "UDPflood_BS2": "UDPFlood2", "UDPScan_BS2": "UDPScan2",
}

SCALES = (1, 10)
STATISTICS = ("n_f", "n_b", "bytes_f", "bytes_b", "len_mean_f", "len_std_f", "len_mean_b", "dports_f", "sports_f", "tcp_f", "udp_f", "icmp_f",
              "syn_only_f", "ack_f", "psh_f", "rst_f", "fin_f", "synack_b", "rst_b", "icmp_b", "iat_mean_f", "iat_std_f", "win_mean_f",
              "ttl_mean_f", "ttl_std_f", "ttl_kinds_f")
FEATURES = tuple(f"{name}_w{w}" for w in SCALES for name in STATISTICS)


def read_pcapng(path: str) -> dict[str, np.ndarray]:
    """Columns for each IPv4 packet in a pcapng file, with no third-party parser.

    Reads Section Header, Interface Description and Enhanced Packet blocks, honours `if_tsresol`, and
    unwraps VLAN tags. Ports are zero for anything but TCP and UDP; flags and window for anything but TCP.
    """
    with open(path, "rb") as handle:
        data = handle.read()
    size, position, endian, interfaces = len(data), 0, "<", []
    rows: dict[str, list] = {k: [] for k in ("ts", "src", "dst", "proto", "ttl", "flags", "sport", "dport", "iplen", "win")}
    while position + 12 <= size:
        block = struct.unpack_from(endian + "I", data, position)[0]
        if block == 0x0A0D0D0A:
            endian = "<" if data[position + 8:position + 12] == b"\x4d\x3c\x2b\x1a" else ">"
            interfaces = []
        length = struct.unpack_from(endian + "I", data, position + 4)[0]
        if length < 12 or position + length > size:
            break
        if block == 1:
            link = struct.unpack_from(endian + "H", data, position + 8)[0]
            resolution, option = 1e-6, position + 16
            while option + 4 <= position + length - 4:
                code, width = struct.unpack_from(endian + "HH", data, option)
                if code == 0:
                    break
                if code == 9:
                    value = data[option + 4]
                    resolution = 2.0 ** -(value & 0x7F) if value & 0x80 else 10.0 ** -value
                option += 4 + width + ((4 - width % 4) % 4)
            interfaces.append((link, resolution))
        elif block == 6:
            interface, high, low = struct.unpack_from(endian + "III", data, position + 8)
            link, resolution = interfaces[interface]
            frame = position + 28
            if link == 1:
                ethertype, ip = (data[frame + 12] << 8) | data[frame + 13], frame + 14
                while ethertype in (0x8100, 0x88A8):
                    ethertype, ip = (data[ip + 2] << 8) | data[ip + 3], ip + 4
            elif link in (101, 228):
                ethertype, ip = 0x0800, frame
            elif link == 113:
                ethertype, ip = (data[frame + 14] << 8) | data[frame + 15], frame + 16
            else:
                ethertype, ip = 0, frame
            if ethertype == 0x0800 and data[ip] >> 4 == 4:
                protocol, transport = data[ip + 9], ip + (data[ip] & 15) * 4
                sport = dport = flags = window = 0
                if protocol in (6, 17):
                    sport, dport = (data[transport] << 8) | data[transport + 1], (data[transport + 2] << 8) | data[transport + 3]
                if protocol == 6:
                    flags, window = data[transport + 13], (data[transport + 14] << 8) | data[transport + 15]
                rows["ts"].append(((high << 32) | low) * resolution)
                rows["src"].append(int.from_bytes(data[ip + 12:ip + 16], "big"))
                rows["dst"].append(int.from_bytes(data[ip + 16:ip + 20], "big"))
                rows["proto"].append(protocol)
                rows["ttl"].append(data[ip + 8])
                rows["flags"].append(flags)
                rows["sport"].append(sport)
                rows["dport"].append(dport)
                rows["iplen"].append((data[ip + 2] << 8) | data[ip + 3])
                rows["win"].append(window)
        position += length
    kinds = {"ts": np.float64, "src": np.uint32, "dst": np.uint32, "proto": np.uint8, "ttl": np.uint8, "flags": np.uint8,
             "sport": np.uint16, "dport": np.uint16, "iplen": np.uint16, "win": np.uint16}
    return {k: np.array(v, dtype=kinds[k]) for k, v in rows.items()}


def _ip(text: str) -> int:
    return struct.unpack(">I", socket.inet_aton(text))[0]


def _pair(a: int, b: int) -> int:
    return (min(a, b) << 32) | max(a, b)


def _statistics(ts, direction, proto, flags, sport, dport, length, window, ttl) -> list[float]:
    forward, backward = direction == 0, direction == 1
    n_f, n_b = int(forward.sum()), int(backward.sum())
    row = [n_f, n_b, float(length[forward].sum()), float(length[backward].sum())]
    if n_f:
        size_f, time_f, proto_f, flag_f, ttl_f = length[forward], ts[forward], proto[forward], flags[forward], ttl[forward]
        gaps = np.diff(time_f) if n_f > 1 else np.zeros(1)
        row += [float(size_f.mean()), float(size_f.std())]
    else:
        row += [0.0, 0.0]
    row.append(float(length[backward].mean()) if n_b else 0.0)
    if n_f:
        row += [len(np.unique(dport[forward])), len(np.unique(sport[forward])), float((proto_f == 6).mean()), float((proto_f == 17).mean()),
                float((proto_f == 1).mean()), float(((flag_f & 0x12) == 0x02).mean()), float(((flag_f & 0x10) != 0).mean()),
                float(((flag_f & 0x08) != 0).mean()), float(((flag_f & 0x04) != 0).mean()), float(((flag_f & 0x01) != 0).mean())]
    else:
        row += [0, 0] + [0.0] * 8
    if n_b:
        flag_b, proto_b = flags[backward], proto[backward]
        row += [float(((flag_b & 0x12) == 0x12).mean()), float(((flag_b & 0x04) != 0).mean()), float((proto_b == 1).mean())]
    else:
        row += [0.0] * 3
    if n_f:
        row += [float(gaps.mean()), float(gaps.std()), float(window[forward].mean()), float(ttl_f.mean()), float(ttl_f.std()), len(np.unique(ttl_f))]
    else:
        row += [0.0] * 5 + [0]
    return row


def conversation_rows(packets: dict[str, np.ndarray], attack_pair: int | None, attack_class: int, skip_pair: int | None = None):
    """(features, labels, ticks): one row per conversation per active second, from packets at or before the tick."""
    order = np.argsort(packets["ts"], kind="stable")
    ts = packets["ts"][order]
    src, dst = packets["src"][order], packets["dst"][order]
    proto, flags, sport, dport = (packets[k][order] for k in ("proto", "flags", "sport", "dport"))
    length, window, ttl = (packets[k][order].astype(float) for k in ("iplen", "win", "ttl"))
    low, high = np.minimum(src, dst).astype(np.uint64), np.maximum(src, dst).astype(np.uint64)
    pairs = (low << np.uint64(32)) | high
    bucket = np.floor(ts - ts[0]).astype(np.int64)
    x, y, ticks = [], [], []
    unique, inverse = np.unique(pairs, return_inverse=True)
    for group, pair in enumerate(unique.tolist()):
        if pair == skip_pair:
            continue
        idx = np.flatnonzero(inverse == group)
        direction = (src[idx] != src[idx[0]]).astype(np.int8)
        seconds = bucket[idx]
        label = attack_class if pair == attack_pair else 0
        for tick in np.unique(seconds).tolist():
            row: list[float] = []
            for width in SCALES:
                lo, hi = np.searchsorted(seconds, tick - width + 1, "left"), np.searchsorted(seconds, tick, "right")
                chosen = idx[lo:hi]
                row += _statistics(ts[chosen], direction[lo:hi], proto[chosen], flags[chosen], sport[chosen], dport[chosen],
                                   length[chosen], window[chosen], ttl[chosen])
            x.append(row)
            y.append(label)
            ticks.append(tick)
    return np.array(x, dtype=np.float32).reshape(-1, len(FEATURES)), np.array(y, dtype=np.int64), np.array(ticks, dtype=np.int64)


def attacker_pairs(root: str) -> dict[str, tuple[str, int] | None]:
    """Read each capture's attacker and class from the dataset's per-attack files, and hold the labelling rule to account."""
    import pandas as pd

    out: dict[str, tuple[str, int] | None] = {}
    folders = {"BS1": sorted(glob.glob(os.path.join(root, "each_attack_csv_bs1", "*")))[0], "BS2": sorted(glob.glob(os.path.join(root, "each_attack_csv_bs2", "*")))[0]}
    for capture, stem in LABEL_FILE.items():
        path = os.path.join(folders[capture[-3:]], f"{stem}.csv")
        # One of the files spells this header differently.
        label = next(c for c in pd.read_csv(path, nrows=0).columns if c.strip().lower().replace("_", " ") in ("attack type", "attacktype"))
        frame = pd.read_csv(path, usecols=["SrcAddr", "DstAddr", label], low_memory=False).rename(columns={label: "Attack Type"})
        attacks = frame[frame["Attack Type"] != "Benign"]
        pairs = set(map(tuple, attacks[["SrcAddr", "DstAddr", "Attack Type"]].drop_duplicates().to_numpy()))
        assert len(pairs) <= 1, f"{stem}: {sorted(pairs)}: more than one attack pair, so the labelling rule is not what this script assumes"
        if pairs:
            source, destination, kind = next(iter(pairs))
            assert destination == VICTIM, f"{stem}: the attack is against {destination}"
            same = frame[(frame["SrcAddr"] == source) & (frame["DstAddr"] == destination)]
            assert (same["Attack Type"] != "Benign").all(), f"{stem}: the attack pair also has flows labelled Benign"
            out[capture] = (source, CLASSES.index(kind))
        else:
            out[capture] = None
    return out


def _one(job):
    capture, path, attack = job
    packets = read_pcapng(path)
    victim = _ip(VICTIM)
    pair, kind = None, 0
    if attack:
        pair, kind = _pair(_ip(attack[0]), victim), attack[1]
    skip = _pair(_ip(COPIED_SOURCE), victim) if capture == COPIED_CAPTURE else None
    x, y, _ = conversation_rows(packets, pair, kind, skip)
    return capture, x, y


def _evaluate(x, y, station, columns, seeds):
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, f1_score

    result: dict = {}
    for name, (a, b) in {"first_to_second": (0, 1), "second_to_first": (1, 0)}.items():
        train, test = station == a, station == b
        accuracy, macro, per = [], [], []
        for seed in range(seeds):
            model = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                                      device="cuda", n_jobs=6, random_state=seed)
            model.fit(x[train][:, columns], y[train])
            model.set_params(device="cpu")
            predicted = model.predict(x[test][:, columns])
            accuracy.append(accuracy_score(y[test], predicted))
            macro.append(f1_score(y[test], predicted, average="macro"))
            per.append(f1_score(y[test], predicted, average=None, labels=list(range(len(CLASSES)))))
        per_mean, per_std = np.mean(per, axis=0), np.std(per, axis=0)
        result[name] = {
            "accuracy_mean": round(float(np.mean(accuracy)), 6), "accuracy_std": round(float(np.std(accuracy)), 6),
            "macro_f1_mean": round(float(np.mean(macro)), 6), "macro_f1_std": round(float(np.std(macro)), 6),
            "per_class_f1_mean": {c: round(float(v), 4) for c, v in zip(CLASSES, per_mean, strict=True)},
            "per_class_f1_std": {c: round(float(v), 4) for c, v in zip(CLASSES, per_std, strict=True)},
        }
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5, help="booster seeds; the data split is the base station, so it does not vary")
    args = ap.parse_args()
    data_root = os.environ.get("QSH_DATA_ROOT")
    if not data_root:
        raise SystemExit("set QSH_DATA_ROOT; see the docstring for the files this reads")
    root = os.path.join(data_root, "raw", "5g-nidd", "pcap")
    started = time.time()
    attacks = attacker_pairs(root)
    jobs = []
    for capture in sorted(LABEL_FILE):
        folder = "bs1" if capture.endswith("BS1") else "bs2"
        jobs.append((capture, os.path.join(root, folder, f"{capture}_nogtp.pcapng"), attacks[capture]))
    with Pool(4) as pool:
        done = pool.map(_one, jobs, chunksize=1)
    x = np.nan_to_num(np.vstack([d[1] for d in done]))
    y = np.concatenate([d[2] for d in done])
    station = np.concatenate([np.full(len(d[2]), int(d[0].endswith("BS2")), dtype=np.int8) for d in done])
    every = list(range(len(FEATURES)))
    no_ttl = [i for i, n in enumerate(FEATURES) if not n.startswith("ttl")]
    print(f"rows {len(y):,} from {len(done)} captures in {time.time() - started:.0f} s")
    result = {
        "unit": "a conversation at a one-second tick, over windows of 1 and 10 seconds, from packets at or before the tick",
        "rows_per_class": {c: int(v) for c, v in zip(CLASSES, np.bincount(y, minlength=len(CLASSES)), strict=True)},
        "rows_first_station": int((station == 0).sum()),
        "rows_second_station": int((station == 1).sum()),
        "features": list(FEATURES),
        "seeds": args.seeds,
        "all_window_features": _evaluate(x, y, station, every, args.seeds),
        "without_ttl": _evaluate(x, y, station, no_ttl, args.seeds),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for tag in ("all_window_features", "without_ttl"):
        for direction, r in result[tag].items():
            print(f"{tag:22s} {direction:16s} accuracy {r['accuracy_mean']:.4f} macro F1 {r['macro_f1_mean']:.4f} +/- {r['macro_f1_std']:.4f}  {r['per_class_f1_mean']}")
    print(f"wrote {OUT} in {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
