"""One direction of the network benchmark's link: forwards a room's sender to its receiver.

A client connects to ws://<relay>/<room>/send or /<room>/recv. Once both ends of a room have joined,
each gets a text frame "ready", and every binary frame from the sender is forwarded to the receiver
unchanged. Each forward is awaited before the next frame is read, so a slow receiver pushes back on
the sender through TCP rather than piling up here.

netem shapes this container's eth0, which carries the forwarded data to the receiver and the ACKs to
the sender, so the receiver's TCP connection meets the emulated delay, bandwidth and loss and the
kernel's congestion control reacts to them. To show that it does, the relay samples TCP_INFO on the
receiver's socket after every forwarded frame and prints one JSON line per room when it closes:
smoothed RTT, the largest congestion window reached, and retransmissions.
"""

import asyncio
import json
import socket
import struct
import time

import websockets

ROOMS: dict[str, dict] = {}


def tcp_info(ws) -> dict | None:
    """The kernel's TCP_INFO for this connection: the same counters `ss -ti` prints."""
    sock = ws.transport.get_extra_info("socket")
    if sock is None:
        return None
    raw = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 104)
    if len(raw) < 104:
        return None
    u8 = struct.unpack_from("8B", raw, 0)
    u32 = struct.unpack_from("24I", raw, 8)
    return {
        "ca_state": u8[1],
        "rtt_us": u32[15],
        "rttvar_us": u32[16],
        "snd_ssthresh": u32[17],
        "snd_cwnd": u32[18],
        "snd_mss": u32[2],
        "total_retrans": u32[23],
    }


async def handler(ws):
    parts = ws.request.path.strip("/").split("/")
    if len(parts) != 2 or parts[1] not in ("send", "recv"):
        await ws.close(code=1008, reason="path must be /<room>/send or /<room>/recv")
        return
    room_id, role = parts
    room = ROOMS.setdefault(room_id, {"ready": asyncio.Event(), "stats": {
        "frames": 0, "bytes": 0, "max_cwnd": 0, "min_rtt_us": None, "max_rtt_us": 0,
        "retrans": 0, "samples": 0, "t0": None, "t1": None}})
    room[role] = ws
    if "send" in room and "recv" in room and not room["ready"].is_set():
        room["ready"].set()
        await asyncio.gather(room["send"].send("ready"), room["recv"].send("ready"))
    try:
        if role == "send":
            await room["ready"].wait()
            st = room["stats"]
            async for msg in ws:
                if isinstance(msg, str):
                    continue
                st["t0"] = st["t0"] or time.time()
                await room["recv"].send(msg)
                st["frames"] += 1
                st["bytes"] += len(msg)
                st["t1"] = time.time()
                info = tcp_info(room["recv"])
                if info:
                    st["samples"] += 1
                    st["max_cwnd"] = max(st["max_cwnd"], info["snd_cwnd"])
                    st["max_rtt_us"] = max(st["max_rtt_us"], info["rtt_us"])
                    st["min_rtt_us"] = info["rtt_us"] if st["min_rtt_us"] is None else min(st["min_rtt_us"], info["rtt_us"])
                    st["retrans"] = info["total_retrans"]
                    st["last"] = info
        else:
            await ws.wait_closed()
    finally:
        room.pop(role, None)
        if "send" not in room and "recv" not in room and ROOMS.pop(room_id, None) is not None:
            print(json.dumps({"room": room_id, **room["stats"]}), flush=True)


async def main():
    # compression=None: websockets negotiates permessage-deflate by default, and the first smoke test
    # "measured" 297 Mbit/s through a 10 Mbit/s link because its repeated-byte payload shrank to almost
    # nothing on the wire. MPC messages are close to random and would not compress, but a relay that
    # compresses distorts every calibration and adds CPU work neither protocol would see on a link.
    async with websockets.serve(handler, "0.0.0.0", 8765, max_size=None, ping_interval=None,
                                compression=None):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
