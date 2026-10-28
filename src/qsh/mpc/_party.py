"""One MPyC party process, started by :func:`qsh.mpc.parties.run_parties` - never imported by the app.

MPyC reads its own flags (``-M -I -T -B -K --no-log``) from ``sys.argv`` when it is imported, which
is why this runs as a separate program rather than inside the caller. The job comes in through
``QSH_MPC_JOB``, one file per party: the shape everyone needs, plus this party's own inputs. The
client's file carries its masked values, the server's its share of them and its output mask, and
the helper's nothing, so the inputs reach the other parties only as Shamir shares. The result is
opened to party 0, the client, alone.
"""

from __future__ import annotations

import json
import os
import time


def main() -> None:
    job = json.loads(open(os.environ["QSH_MPC_JOB"]).read())
    from mpyc.runtime import mpc  # parses the party flags from sys.argv

    async def run() -> None:
        await mpc.start()
        sec = mpc.SecFxp(int(job["bits"]))              # local set-up, before the clock starts
        # start() returns when THIS party's links are up, which can be one 0.1 s connect retry before
        # another's: a quarter of sessions measured ~90 ms of that skew as protocol time. One exchange
        # with every party first, so all clocks start once everyone is ready.
        await mpc.transfer(mpc.pid)
        before = {peer.pid: peer.protocol.nbytes_sent for peer in mpc.parties if peer.pid != mpc.pid}
        t0 = time.perf_counter()
        n = int(job["n"])

        def shared(values, sender: int, count: int):
            """Secret-share `values` from `sender`; every other party passes placeholders."""
            mine = values if mpc.pid == sender else None
            return mpc.input([sec(float(v)) for v in (mine or [0.0] * count)], senders=sender)

        # The input is the sum of the senders' additive shares: the client's masked value and the
        # server's share, the negated mask. Neither party's file holds the value itself.
        xs = None
        for sender in job["inputs_from"]:
            part = shared(job.get("share"), sender, n)
            xs = part if xs is None else [a + b for a, b in zip(xs, part, strict=True)]
        op = job["op"]
        if op == "relu":
            out = [(x >= 0) * x for x in xs]
        elif op == "maxpool":
            out, k = [], 0
            for size in job["sizes"]:
                out.append(mpc.max(xs[k:k + size]))
                k += size
        elif op == "argmax":
            out = None
        else:
            raise ValueError(f"unknown MPC op {op!r}")
        if out is None:
            # The decision: the class index, to the client only. Nothing else is opened.
            index = await mpc.output(mpc.argmax(xs)[0], receivers=[0])
            result = None if index is None else int(index)
        else:
            if job["out_masked"]:
                # Re-masked by the server before it is opened: the client receives a + s and the
                # server keeps -s, an additive sharing of a for the next layer under FHE.
                mask = shared(job.get("out_mask"), 1, len(out))
                out = [a + s for a, s in zip(out, mask, strict=True)]
            opened = await mpc.output(out, receivers=[0])
            result = None if opened[0] is None else [float(v) for v in opened]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        sent = sum(peer.protocol.nbytes_sent - before[peer.pid] for peer in mpc.parties if peer.pid != mpc.pid)
        await mpc.shutdown()
        with open(os.environ["QSH_MPC_OUT"], "w") as f:
            json.dump({"result": result, "bytes_sent": int(sent), "elapsed_ms": elapsed_ms}, f)

    mpc.run(run())


if __name__ == "__main__":
    main()
