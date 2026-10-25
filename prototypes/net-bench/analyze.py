"""Summarises a network-bench grid into results/summary.json and prints the comparison table.

Inputs, all written by the run: results/grid-party0.json and grid-party1.json (one per tab) and
results/relay-tcp.json (the relays' TCP_INFO per room). A run's completion time is the LATER of the two
parties' times, since the protocol is finished only when both have their output; bytes and messages are
the sum of both parties' sends. Run with any Python 3.9+, no dependencies.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent / "results"


def load(name: str):
    return json.loads((HERE / name).read_text())


def main() -> None:
    p0, p1 = load("grid-party0.json"), load("grid-party1.json")
    tcp = {(r["room"], r["direction"]): r for r in load("relay-tcp.json")}
    by_tag1 = {r["cfg"]["tag"]: r for r in p1["results"]}
    summary = []
    for r0 in p0["results"]:
        cfg, r1 = r0["cfg"], by_tag1[r0["cfg"]["tag"]]
        row = {"tag": cfg["tag"], "n": cfg["n"], "link": cfg["link"], "applied": (r0.get("applied") or {}).get("applied"),
               "calibration": {k: r0["calib"][k] for k in ("rttMs", "rttMinMs", "rttMaxMs", "goodputMbps", "bulkBytes")},
               "engines": {}}
        for engine in cfg["engines"]:
            name = "emp-wasm" if engine == "emp" else "polytune"
            a = [x for x in r0["runs"] if x["engine"] == name]
            b = [x for x in r1["runs"] if x["engine"] == name]
            done = [max(x["ms"], y["ms"]) for x, y in zip(a, b)]
            rooms = [f"{cfg['tag']}-{engine}-{i}" for i in range(len(a))]
            stats = [tcp[(room, d)] for room in rooms for d in ("ab", "ba") if (room, d) in tcp]
            row["engines"][name] = {
                "runs": len(done),
                "correct": sum(x["correct"] and y["correct"] for x, y in zip(a, b)),
                "completion_ms": {"median": statistics.median(done), "min": min(done), "max": max(done), "all": done},
                "bytes": a[0]["bytesOut"] + b[0]["bytesOut"],
                "messages": a[0]["msgsOut"] + b[0]["msgsOut"],
                "relay_tcp": {
                    "max_cwnd_segments": max(s["max_cwnd"] for s in stats),
                    "srtt_ms_min": min(s["min_rtt_us"] for s in stats if s["min_rtt_us"] is not None) / 1e3,
                    "srtt_ms_max": max(s["max_rtt_us"] for s in stats) / 1e3,
                    "retransmissions": sum(s["retrans"] for s in stats),
                },
            }
        e, p = row["engines"]["emp-wasm"], row["engines"]["polytune"]
        row["polytune_speedup"] = e["completion_ms"]["median"] / p["completion_ms"]["median"]
        summary.append(row)
    out = {"measuredAt": p0.get("measuredAt"), "browser": p0.get("ua"), "configs": summary}
    (HERE / "summary.json").write_text(json.dumps(out, indent=1) + "\n")

    print(f"{'n':>2} {'rtt':>4} {'Mbit':>4} {'loss':>5} | {'cal rtt':>7} {'cal Mbps':>8} | "
          f"{'emp ms':>7} {'poly ms':>7} {'speedup':>7} | retrans e/p")
    for r in summary:
        e, p, c, ln = r["engines"]["emp-wasm"], r["engines"]["polytune"], r["calibration"], r["link"]
        print(f"{r['n']:>2} {ln['rttMs']:>4} {ln['rateMbit']:>4} {ln['lossPct']:>5} | {c['rttMs']:>7.1f} "
              f"{c['goodputMbps']:>8.1f} | {e['completion_ms']['median']:>7.0f} {p['completion_ms']['median']:>7.0f} "
              f"{r['polytune_speedup']:>7.2f} | {e['relay_tcp']['retransmissions']}/{p['relay_tcp']['retransmissions']}")
    bad = [(r["tag"], k) for r in summary for k, v in r["engines"].items() if v["correct"] != v["runs"]]
    print("all runs correct" if not bad else f"INCORRECT: {bad}")


def web_copy(summary_path: Path, out_path: Path) -> None:
    """The trimmed copy the Validations page draws: every number it shows, nothing it does not.

    Written into web/, which must not refer to this directory, so it carries the figures and no paths;
    tests/test_mpc_net_bench_copy_matches_the_record.py checks it against results/summary.json.
    """
    s = json.loads(summary_path.read_text())
    rows = []
    for r in s["configs"]:
        def eng(e):
            return {"medianMs": round(e["completion_ms"]["median"], 1), "minMs": round(e["completion_ms"]["min"], 1),
                    "maxMs": round(e["completion_ms"]["max"], 1), "bytes": e["bytes"], "messages": e["messages"],
                    "retransmissions": e["relay_tcp"]["retransmissions"], "correct": e["correct"], "runs": e["runs"]}
        rows.append({"n": r["n"], "rttMs": r["link"]["rttMs"], "rateMbit": r["link"]["rateMbit"],
                     "lossPct": r["link"]["lossPct"], "measuredRttMs": round(r["calibration"]["rttMs"], 1),
                     "measuredGoodputMbps": round(r["calibration"]["goodputMbps"], 1),
                     "empWasm": eng(r["engines"]["emp-wasm"]), "polytune": eng(r["engines"]["polytune"]),
                     "ratio": round(r["polytune_speedup"], 2)})
    out_path.write_text(json.dumps({"measuredAt": s["measuredAt"], "browser": s["browser"], "configs": rows},
                                   indent=1) + "\n")


if __name__ == "__main__":
    import sys
    main()
    if "--web" in sys.argv:
        web = Path(__file__).resolve().parents[2] / "web" / "src" / "lib" / "mpcNetBench.json"
        web_copy(HERE / "summary.json", web)
        print(f"wrote {web.relative_to(web.parents[3])}")
