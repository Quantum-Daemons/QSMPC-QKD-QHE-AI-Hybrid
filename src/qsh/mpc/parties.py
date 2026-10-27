"""Run an MPyC computation with several parties, each its own OS process, over localhost TCP.

WHY THIS EXISTS

Until Round 75 the non-linear half of the hybrid ran MPyC in-process with its command line cleared,
which leaves MPyC at its defaults: ONE party (m = 1), threshold t = 0, security parameter 30. A
degree-0 Shamir "share" of a value is the value, so the "real Shamir secure comparison" README
described computed in the clear and exchanged no bytes at all - the communication figure was an
estimate multiplied out from a per-comparison constant because there was nothing to measure.

Here each party is a separate process started with MPyC's own flags (`-M m -I i -T t -B base -K k`),
connected over loopback TCP. The roles follow the hybrid's masked conversions
(:mod:`qsh.mpc.secret_inference`):
- party 0 is the CLIENT: it holds the CKKS key and inputs what it decrypted, a masked value;
- party 1 is the SERVER: it inputs its additive share of the same value (minus its mask) and, where
  the result goes back under FHE, a fresh mask for the output;
- party 2 is a HELPER that inputs nothing. Shamir with t = 1 needs a third party for an honest
  majority, and it sees only shares.
The input is the SUM of the two parties' shares, so neither one's file holds the value, and the
result is opened to party 0 alone: re-masked by the server's output mask, or, for the decision,
the class index and nothing else. Every party reports the bytes it actually sent.

What it is not:
- The parties share one host, so the timings are LAN-free loopback figures.
- Semi-honest: Shamir with t = 1 of 3 hides the inputs from any one party that follows the protocol.
  A client and helper that collude learn the server's masks, and so the values.

Measured per secure comparison (Round 75, m = 3, t = 1, k = 40, 64-bit fixed point, no gmpy2): about
15 KB across the three parties, the same on every run, and about 2.2 ms of protocol time on loopback.
Each session also pays about 0.25 s to start its processes, reported apart from the protocol time.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

#: The package root a party process must import `qsh` from.
_SRC = Path(__file__).resolve().parents[2]

#: The parties' roles, as the job files and the hybrid's conversions use them.
CLIENT, SERVER = 0, 1


def _ephemeral_range() -> tuple[int, int]:
    """The kernel's ephemeral port range, from which OUTGOING connections take their ports."""
    try:
        lo, hi = (int(v) for v in Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split())
        return lo, hi
    except (OSError, ValueError):
        return 32768, 60999                          # the Linux default


def _free_base_port(m: int) -> int:
    """A base port b with b .. b + m - 1 all free on loopback (MPyC listens on b + party id).

    Drawn from BELOW the kernel's ephemeral range, which outgoing connections take their source ports
    from. A base found by binding port 0 lies inside it, so an outgoing connection could hold a port a
    party was about to listen on: that party's bind failed with EADDRINUSE and the others waited for
    it. Measured in Round 75: 5 of 300 sessions failed that way, every one of them party 1 on base + 1,
    the even port that connect() draws from. From below the range: 0 of 300.
    """
    import random
    lo, _ = _ephemeral_range()
    top = max(10_000 + m + 1, lo - m - 1)
    for _ in range(256):
        base = random.randrange(10_000, top)
        free = True
        for port in range(base, base + m):
            with socket.socket() as t:
                try:
                    t.bind(("127.0.0.1", port))
                except OSError:
                    free = False
                    break
        if free:
            return base
    raise RuntimeError("found no run of free loopback ports for the MPC parties")


#: Concurrent sessions would draw their bases independently, and two that overlapped could connect
#: one run's party to another run's listener. Sessions are serialised instead: a thread lock inside
#: one process, and an advisory file lock across processes where the platform has one.
_LOCK = threading.Lock()
_LOCK_FILE = Path(tempfile.gettempdir()) / "qsh-mpc-parties.lock"

#: Attempts when a party loses its port between the check and its own bind (EADDRINUSE).
_ATTEMPTS = 3


@contextmanager
def _one_session_at_a_time():
    with _LOCK:
        try:
            import fcntl
            f = open(_LOCK_FILE, "a")                    # noqa: SIM115 - held for the whole session
        except (ImportError, OSError):                   # no flock (Windows), or another user's file
            yield
            return
        with f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


class _PortTaken(RuntimeError):
    """A party could not bind its port: something took it after it was checked free."""


def _floats(name: str, values, n: int | None) -> list[float]:
    out = [float(v) for v in values]
    if n is not None and len(out) != n:
        raise ValueError(f"{name} has {len(out)} values, expected {n}")
    return out


def run_parties(op: str, values: list[float] | None = None, *, windows: list[list[float]] | None = None,
                server_share: list[float] | None = None, out_mask: list[float] | None = None,
                m: int = 3, threshold: int | None = None, sec_param: int = 40, bits: int = 64,
                timeout_s: float = 60.0) -> dict:
    """Run one secure operation (``relu``, ``argmax`` or ``maxpool``) with ``m`` party processes.

    ``values`` (or, for ``maxpool``, ``windows``) is the CLIENT's input. ``server_share`` is the
    server's additive share of the same values, so the computation runs on their sum; ``out_mask``
    is added to a ReLU or max-pool result before it is opened. The result is opened to the client,
    party 0, and to no one else.

    Returns that result with what the run measured: party count, threshold, security parameter,
    total and per-party bytes sent, the protocol time inside the parties (``mpc_ms``) and the wall
    time including process start-up (``wall_ms``). Raises rather than falling back to an in-process
    run, because the fallback is the defect this replaces.
    """
    if m < 2:
        raise ValueError(f"m = {m}: a secret-sharing computation needs at least two parties")
    t = (m - 1) // 2 if threshold is None else int(threshold)
    if not 0 <= 2 * t < m:
        raise ValueError(f"threshold {t} with {m} parties: Shamir's honest majority needs t < m/2")
    if op == "maxpool":
        if windows is None:
            raise ValueError("maxpool needs windows")
        client = [float(v) for w in windows for v in w]
        shape = {"sizes": [len(w) for w in windows]}
        n_out = len(windows)
    elif op in ("relu", "argmax"):
        if values is None:
            raise ValueError(f"{op} needs values")
        client = _floats("values", values, None)
        shape = {}
        n_out = len(client) if op == "relu" else None
    else:
        raise ValueError(f"unknown MPC op {op!r}")
    if (server_share is not None or out_mask is not None) and m < 3:
        raise ValueError("a server share needs a third party: with two, the server and client hold "
                         "every share between them")
    jobs = {CLIENT: {"share": client}}
    server = {}
    if server_share is not None:
        server["share"] = _floats("server_share", server_share, len(client))
    if out_mask is not None:
        if n_out is None:
            raise ValueError("argmax opens a class index, which has no mask")
        server["out_mask"] = _floats("out_mask", out_mask, n_out)
    if server:
        jobs[SERVER] = server
    common = {"op": op, "bits": bits, "n": len(client), **shape,
              "inputs_from": [CLIENT] + ([SERVER] if "share" in server else []),
              "out_masked": "out_mask" in server}

    with _one_session_at_a_time():
        for attempt in range(1, _ATTEMPTS + 1):
            try:
                run = _session(common, jobs, m, t, sec_param, timeout_s)
            except _PortTaken:
                if attempt == _ATTEMPTS:
                    raise
                continue
            run["attempts"] = attempt
            return run
    raise AssertionError("unreachable")


def _session(common: dict, jobs: dict, m: int, t: int, sec_param: int, timeout_s: float) -> dict:
    """One attempt: start the parties on a fresh base port and collect what each of them reports."""
    with tempfile.TemporaryDirectory(prefix="qsh-mpc-") as tmp:
        d = Path(tmp)
        # One job file per party: the shape everyone needs, plus that party's own inputs, if any.
        for i in range(m):
            (d / f"job_{i}.json").write_text(json.dumps({**common, **jobs.get(i, {})}))
        base = _free_base_port(m)
        env_base = {**os.environ, "PYTHONPATH": os.pathsep.join(
            [str(_SRC)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])}
        procs = []
        t0 = time.perf_counter()
        for i in range(m):
            env = {**env_base, "QSH_MPC_JOB": str(d / f"job_{i}.json"),
                   "QSH_MPC_OUT": str(d / f"out_{i}.json")}
            cmd = [sys.executable, "-m", "qsh.mpc._party", "-M", str(m), "-I", str(i), "-T", str(t),
                   "-B", str(base), "-K", str(sec_param), "--no-log"]
            # stderr goes to a file, not a pipe: an unread pipe that fills up blocks the party.
            with open(d / f"err_{i}.txt", "wb") as err:
                procs.append(subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=err))
        # Watch every party at once. One that fails leaves the others waiting for it forever, so the
        # first non-zero exit ends the run instead of the timeout.
        deadline = time.monotonic() + timeout_s
        timed_out = False
        while True:
            codes = [p.poll() for p in procs]            # every party, every pass: any() would stop early
            if None not in codes:
                break
            timed_out = time.monotonic() > deadline
            if timed_out or any(c not in (None, 0) for c in codes):
                for p in procs:
                    if p.poll() is None:
                        p.kill()
                for p in procs:
                    p.wait()
                break
            time.sleep(0.005)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        if timed_out or any(p.returncode != 0 for p in procs):
            why = f"did not finish within {timeout_s} s" if timed_out else "failed"
            tails = [(d / f"err_{i}.txt").read_text(errors="replace").strip()[-600:] or "-"
                     for i in range(m)]
            lost = any("address already in use" in tail.lower() for tail in tails)
            raise (_PortTaken if lost else RuntimeError)(
                f"MPC parties {why} (ports {base}..{base + m - 1}): " + " | ".join(
                    f"party {i} exit {p.returncode}: {tail}"
                    for i, (p, tail) in enumerate(zip(procs, tails, strict=True))))
        outs = [json.loads((d / f"out_{i}.json").read_text()) for i in range(m)]

    if outs[CLIENT]["result"] is None:
        raise RuntimeError("the client was not given the result")
    leaked = [i for i, o in enumerate(outs) if i != CLIENT and o["result"] is not None]
    if leaked:
        raise RuntimeError(f"the result was opened to parties {leaked} as well as the client")
    per_party = [int(o["bytes_sent"]) for o in outs]
    return {
        "result": outs[CLIENT]["result"], "parties": m, "threshold": t, "sec_param": sec_param,
        "bytes_sent": sum(per_party), "per_party_bytes": per_party,
        "wall_ms": wall_ms, "mpc_ms": max(float(o["elapsed_ms"]) for o in outs),
    }
