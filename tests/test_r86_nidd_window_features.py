"""The window study's parser and its causality, and that its record is what its write-up quotes (Round 86)."""

from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "research"))

import nidd_window_features as W  # noqa: E402


def _block(kind: int, body: bytes) -> bytes:
    length = 12 + len(body)
    return struct.pack("<II", kind, length) + body + struct.pack("<I", length)


def _frame(src: str, dst: str, *, vlan: bool = False, proto: int = 17, sport: int = 1000, dport: int = 80, ttl: int = 64, flags: int = 0, window: int = 0) -> bytes:
    tag = b"\x81\x00\x00\x05" if vlan else b""
    transport = struct.pack(">HH", sport, dport) + (b"\0" * 8 + bytes([0x50, flags]) + struct.pack(">H", window) + b"\0" * 4 if proto == 6 else b"\0" * 4)
    ip = bytes([0x45, 0]) + struct.pack(">H", 20 + len(transport)) + b"\0" * 4 + bytes([ttl, proto]) + b"\0\0" + bytes(map(int, src.split("."))) + bytes(map(int, dst.split(".")))
    return b"\0" * 12 + tag + b"\x08\x00" + ip + transport


def _packet(ts_us: int, frame: bytes) -> bytes:
    padding = b"\0" * ((4 - len(frame) % 4) % 4)
    return _block(6, struct.pack("<IIIII", 0, ts_us >> 32, ts_us & 0xFFFFFFFF, len(frame), len(frame)) + frame + padding)


def _capture(path: Path, packets: list[bytes]) -> None:
    section = _block(0x0A0D0D0A, b"\x4d\x3c\x2b\x1a\x01\x00\x00\x00" + b"\xff" * 8)
    interface = _block(1, struct.pack("<HHI", 1, 0, 65535))
    path.write_bytes(section + interface + b"".join(packets))


def test_the_reader_returns_the_fields_of_each_packet(tmp_path: Path) -> None:
    path = tmp_path / "a.pcapng"
    _capture(path, [_packet(1_000_000, _frame("10.0.0.1", "10.0.0.2", ttl=63, dport=53)),
                    _packet(2_500_000, _frame("10.0.0.2", "10.0.0.1", vlan=True, proto=6, sport=443, dport=9, flags=0x12, window=777, ttl=64))])
    got = W.read_pcapng(str(path))
    assert got["ts"].tolist() == pytest.approx([1.0, 2.5])
    assert got["proto"].tolist() == [17, 6] and got["ttl"].tolist() == [63, 64] and got["dport"].tolist() == [53, 9]
    assert got["flags"].tolist() == [0, 0x12] and got["win"].tolist() == [0, 777] and got["sport"].tolist() == [1000, 443]


def _stream(n: int) -> dict[str, np.ndarray]:
    ts = np.arange(n, dtype=np.float64) * 0.4
    return {"ts": ts, "src": np.full(n, 1, np.uint32), "dst": np.full(n, 2, np.uint32), "proto": np.full(n, 17, np.uint8), "ttl": np.full(n, 64, np.uint8),
            "flags": np.zeros(n, np.uint8), "sport": np.arange(n, dtype=np.uint16), "dport": np.arange(n, dtype=np.uint16) * 3, "iplen": np.full(n, 60, np.uint16), "win": np.zeros(n, np.uint16)}


def test_a_row_reads_nothing_after_its_tick() -> None:
    full = W.conversation_rows(_stream(60), None, 0)
    # Each cut lands inside a second, so a row that read ahead would see a part-filled next second.
    for last in (11, 26, 46):
        cut = W.conversation_rows(_stream(last), None, 0)
        keep = cut[2] < cut[2].max()  # the final tick of a truncated stream is still filling
        assert np.array_equal(cut[0][keep], full[0][: keep.sum()]), "a row changed when later packets were removed"


def test_the_ten_second_window_counts_what_it_says() -> None:
    x, _, ticks = W.conversation_rows(_stream(60), None, 0)
    # packets every 0.4 s: 2 or 3 per second, so at most 25 in ten seconds, and n_f in the one-second window is at most 3
    assert x[:, W.FEATURES.index("n_f_w1")].max() == 3
    assert x[:, W.FEATURES.index("n_f_w10")].max() == 25
    assert x[0, W.FEATURES.index("n_f_w10")] == x[0, W.FEATURES.index("n_f_w1")] and ticks[0] == 0


def test_the_attack_pair_is_labelled_and_the_copied_pair_is_left_out() -> None:
    s = _stream(10)
    pair = W._pair(1, 2)
    assert set(W.conversation_rows(s, pair, 5)[1].tolist()) == {5}
    assert set(W.conversation_rows(s, None, 5)[1].tolist()) == {0}
    assert len(W.conversation_rows(s, None, 0, skip_pair=pair)[1]) == 0


def test_the_write_up_quotes_the_record() -> None:
    record = json.loads((REPO / "references" / "data" / "5g-nidd-window-features.json").read_text())
    text = (REPO / "references" / "5g-nidd-window-features.md").read_text()
    without = record["without_ttl"]
    every = record["all_window_features"]
    for value in (every["first_to_second"]["macro_f1_mean"], every["second_to_first"]["macro_f1_mean"], without["first_to_second"]["macro_f1_mean"],
                  without["second_to_first"]["macro_f1_mean"], without["first_to_second"]["per_class_f1_mean"]["UDPScan"], without["second_to_first"]["per_class_f1_mean"]["UDPScan"],
                  without["second_to_first"]["per_class_f1_mean"]["SYNFlood"], without["second_to_first"]["per_class_f1_mean"]["SYNScan"]):
        assert f"{value:.4f}" in text, f"{value:.4f} is in the record and not in the write-up"
    assert re.search(r"104,605 rows", text) and sum(record["rows_per_class"].values()) == 104_605
