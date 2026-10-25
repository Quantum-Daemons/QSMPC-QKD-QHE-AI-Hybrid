# Network bench: the secure argmax under two engines, over a shaped link

A measurement harness, not part of the demo. It runs the demo's 2-party secure argmax
(`web/src/lib/clientMPC.ts`, as it was before Round 66) under emp-wasm, which the demo shipped until
Round 65, and under Polytune (`prototypes/polytune-argmax`), with the two parties in two browser tabs
connected over real WebSockets through relays that netem shapes. The Validations page draws the
recorded results from a copy, `web/src/lib/mpcNetBench.json`, which
`tests/test_mpc_net_bench_copy_matches_the_record.py` holds equal to `results/summary.json`.

## How the link is built

- **Two relays, one per direction.** `relay/relay.py` forwards one room's sender to its receiver.
  Party 0 sends through the A-to-B relay and receives through the B-to-A one, so each direction has its
  own bottleneck, as on a full-duplex link. Each forward is awaited, so backpressure reaches the
  sender through TCP.
- **netem on each relay's own interface.** `control.py` applies
  `netem delay <RTT/2> rate <Mbit/s> loss <percent>` (default 1000-packet queue) to both relays. The
  browsers' real TCP connections pass through it, so slow start, the congestion window and loss
  recovery all happen. Each relay logs the kernel's TCP_INFO per room (largest congestion window,
  smoothed RTT range, retransmissions) to `results/relay-tcp.json`.
- **Two sites, two processes.** Party 0 is opened on `http://127.0.0.1:4320`, party 1 on
  `http://localhost:4320`, which the browser treats as different sites and runs in separate processes.
- **No compression.** The relay disables permessage-deflate. The first smoke test measured 297 Mbit/s
  through a 10 Mbit/s link because its repeated-byte payload compressed to almost nothing.
- **Calibrated before use.** Each configuration starts with an echo test (RTT) and a bulk transfer of
  random bytes (goodput) through the same relays, so the link is measured, not assumed.

## Run it

The host needs Docker and the kernel's netem module (`sudo modprobe sch_netem`). Containers get
`NET_ADMIN` for their own network namespace only, and every port binds to 127.0.0.1.

```bash
cd prototypes/polytune-argmax && ./build.sh
```

```bash
cd prototypes/net-bench/relay && docker build -t qsh-netem-relay:2 .
```

```bash
docker run -d --name qsh-netem-ab --cap-add NET_ADMIN -p 127.0.0.1:8765:8765 qsh-netem-relay:2
```

```bash
docker run -d --name qsh-netem-ba --cap-add NET_ADMIN -p 127.0.0.1:8766:8765 qsh-netem-relay:2
```

```bash
cd prototypes/net-bench && python3 control.py
```

```bash
cd prototypes/net-bench/bench && npm install && npm run build && npm run preview
```

Open `http://localhost:4320/?party=1`, then `http://127.0.0.1:4320/?party=0`, and call
`window.runGrid(grid)` in both with the same grid (party 1 first). Party 0 sets each link through the
control server; the relay's ready signal keeps the tabs in step. Save each tab's results with a POST
to `http://127.0.0.1:8779/save?name=grid-party0` (and `grid-party1`), collect the relays' logs into
`results/relay-tcp.json`, then:

```bash
python3 analyze.py --web
```

Remove the relays when done:

```bash
docker rm -f qsh-netem-ab qsh-netem-ba
```

## Results (2026-10-25, Chrome 152, one machine)

Median completion time of three runs (the later of the two parties); all 120 runs returned the right
index.

| classes | RTT | Mbit/s | loss | emp-wasm | Polytune | faster |
|---|---|---|---|---|---|---|
| 2 | 1 ms | 10 | 0 to 1% | 935 to 953 ms | 853 to 856 ms | Polytune, 1.09 to 1.11x |
| 2 | 1 ms | 100 | 0 to 1% | 764 to 779 ms | 234 to 246 ms | Polytune, 3.10 to 3.33x |
| 2 | 10 ms | 10 | 0 to 1% | 1153 to 1366 ms | 1019 to 1031 ms | Polytune, 1.13 to 1.32x |
| 2 | 10 ms | 100 | 0 to 1% | 999 to 1001 ms | 404 to 409 ms | Polytune, 2.44 to 2.48x |
| 2 | 50 ms | 10 | 0 to 1% | 2070 to 2129 ms | 1832 to 1874 ms | Polytune, 1.13 to 1.14x |
| 2 | 50 ms | 100 | 0 to 1% | 1952 to 2152 ms | 1205 to 1518 ms | Polytune, 1.42 to 1.63x |
| 10 | 10 ms | 100 | 0 | 1158 ms | 1698 ms | emp-wasm, 1.47x |
| 10 | 50 ms | 10 | 0 | 3038 ms | 8363 ms | emp-wasm, 2.75x |

Traffic per run: at 2 classes 214,861 bytes in 43 messages (emp-wasm) against 1,018,898 bytes in 69
(Polytune); at 10 classes 1,241,896 bytes in 55 against 7,184,705 in 195. Calibration matched the
settings within about 1.7 ms of RTT and at 90 to 95% of the shaped rate, except where real TCP limits
it: 1% loss at 10 ms on 100 Mbit/s (33.8 Mbit/s) and at 50 ms (7.9 Mbit/s on 10, 10.4 on 100), and slow
start at 50 ms on 100 Mbit/s (76 Mbit/s for an 8 MB transfer). Full figures: `results/summary.json`.

The demo's secure argmax is the 2-class case on every page that runs it (finance, medical, genomics).

## Not covered

- Other machines, browsers and real wide-area paths. One machine, one browser build.
- The demo as shipped: its two parties share one tab and exchange messages in memory.
- Relay hops: each direction crosses one relay, so a protocol message pays two TCP hops.
