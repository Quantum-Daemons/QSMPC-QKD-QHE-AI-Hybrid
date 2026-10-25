// One party's link to the other: frames OUT through one relay and frames IN through the other, so each
// direction crosses its own netem-shaped interface, as on a full-duplex link. Both engines use exactly
// this code; only the protocol above it differs.
export async function connect(relays, room) {
  const out = new WebSocket(`${relays.out}/${room}/send`);
  const inn = new WebSocket(`${relays.in}/${room}/recv`);
  out.binaryType = inn.binaryType = "arraybuffer";
  const queue = [];
  const waiters = [];
  let bytesOut = 0;
  let msgsOut = 0;
  await new Promise((resolve, reject) => {
    let ready = 0;
    const onText = (data) => { if (data === "ready" && ++ready === 2) resolve(); };
    out.onmessage = (ev) => { if (typeof ev.data === "string") onText(ev.data); };
    inn.onmessage = (ev) => {
      if (typeof ev.data === "string") return onText(ev.data);
      const u8 = new Uint8Array(ev.data);
      if (waiters.length) waiters.shift()(u8); else queue.push(u8);
    };
    out.onerror = inn.onerror = () => reject(new Error("websocket error"));
  });
  return {
    // A copy: Polytune hands over a view into wasm memory, which the next call may overwrite.
    send(data) { bytesOut += data.length; msgsOut += 1; out.send(data.slice()); },
    recv() { return queue.length ? Promise.resolve(queue.shift()) : new Promise((r) => waiters.push(r)); },
    stats() { return { bytesOut, msgsOut }; },
    close() { out.close(); inn.close(); },
  };
}
