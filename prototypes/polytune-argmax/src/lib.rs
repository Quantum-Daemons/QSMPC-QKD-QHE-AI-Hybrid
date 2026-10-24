//! The demo's secure argmax on Polytune, with both parties in one browser tab.
//!
//! `web/src/lib/clientMPC.ts` computes the argmax of N logits that are additively shared between two
//! parties (`q_k = a_k + b_k - BIG`) under emp-wasm's authenticated garbling, and outputs only the
//! index. This crate computes the same function with Polytune (WRK17b multi-party authenticated
//! garbling, malicious security against all but one party), so the two can be measured side by side.
//!
//! What is the same: the inputs, the masking, the output, and both parties sharing one tab over an
//! in-memory transport. What differs: Garble's integers are 32-bit where the summon circuit uses a
//! 24-bit boolify width, so this side does slightly more arithmetic per comparison.

use std::{cell::Cell, rc::Rc};

use futures::{StreamExt, channel::mpsc, future::join, lock::Mutex};
use polytune::{
    channel::Channel,
    garble_lang::{
        CircuitKind, CompileOptions, compile_with_options,
        literal::Literal,
        token::{SignedNumType, UnsignedNumType},
    },
    mpc,
};
use wasm_bindgen::prelude::*;
use web_time::Instant;

/// The additive-share mask, as in clientMPC.ts.
pub const BIG: i64 = 1 << 16;

/// Garble source for the argmax of `n` shared values, unrolled the way clientMPC.ts builds its
/// summon circuit: `n - 1` comparisons, the index of the first maximum wins.
pub fn argmax_source(n: usize) -> String {
    let mut s = format!("pub fn main(a: [i32; {n}], b: [i32; {n}]) -> u8 {{\n");
    s += &format!("    let mut best: i32 = a[0usize] + b[0usize] - {BIG}i32;\n");
    s += "    let mut arg: u8 = 0u8;\n";
    for k in 1..n {
        s += &format!("    let q{k}: i32 = a[{k}usize] + b[{k}usize] - {BIG}i32;\n");
        s += &format!("    if q{k} > best {{\n        best = q{k};\n        arg = {k}u8;\n    }}\n");
    }
    s += "    arg\n}\n";
    s
}

/// An in-memory channel between parties in one thread. Polytune's own `SimpleChannel` is compiled
/// out on wasm32 (it relies on tokio's timer), so a browser needs its own.
pub struct MemChannel {
    tx: Vec<Option<mpsc::UnboundedSender<Vec<u8>>>>,
    rx: Vec<Option<Mutex<mpsc::UnboundedReceiver<Vec<u8>>>>>,
    sent: Rc<Cell<u64>>,
    /// Messages, not only bytes: over a real network each one costs a latency, which an in-tab
    /// measurement cannot show, so the count is what lets the timing be projected onto a link.
    msgs: Rc<Cell<u64>>,
}

impl MemChannel {
    /// Two connected endpoints, and the shared counts of bytes and messages either one sends.
    pub fn pair() -> (MemChannel, MemChannel, Rc<Cell<u64>>, Rc<Cell<u64>>) {
        let (sent, msgs) = (Rc::new(Cell::new(0)), Rc::new(Cell::new(0)));
        let (tx01, rx01) = mpsc::unbounded();
        let (tx10, rx10) = mpsc::unbounded();
        let p0 = MemChannel { tx: vec![None, Some(tx01)], rx: vec![None, Some(Mutex::new(rx10))],
            sent: sent.clone(), msgs: msgs.clone() };
        let p1 = MemChannel { tx: vec![Some(tx10), None], rx: vec![Some(Mutex::new(rx01)), None],
            sent: sent.clone(), msgs: msgs.clone() };
        (p0, p1, sent, msgs)
    }
}

impl Channel for MemChannel {
    type SendError = String;
    type RecvError = String;

    async fn send_bytes_to(&self, party: usize, data: Vec<u8>, _phase: &str) -> Result<(), String> {
        self.sent.set(self.sent.get() + data.len() as u64);
        self.msgs.set(self.msgs.get() + 1);
        self.tx
            .get(party)
            .and_then(|t| t.as_ref())
            .ok_or_else(|| format!("no channel to party {party}"))?
            .unbounded_send(data)
            .map_err(|e| e.to_string())
    }

    async fn recv_bytes_from(&self, party: usize, _phase: &str) -> Result<Vec<u8>, String> {
        let mut rx = self
            .rx
            .get(party)
            .and_then(|r| r.as_ref())
            .ok_or_else(|| format!("no channel from party {party}"))?
            .lock()
            .await;
        rx.next().await.ok_or_else(|| format!("party {party} closed the channel"))
    }
}

/// One run: the index, how long compiling and the protocol took, and what the two parties exchanged.
#[derive(Debug, Clone, Copy)]
pub struct Outcome {
    pub arg: u32,
    pub compile_ms: f64,
    pub mpc_ms: f64,
    pub bytes: u64,
    pub messages: u64,
    pub and_ops: u64,
}

/// Runs the secure argmax with party 0 holding `a` and party 1 holding `b`.
pub async fn secure_argmax_shares(a: &[i32], b: &[i32]) -> Result<Outcome, String> {
    if a.len() != b.len() || a.len() < 2 {
        return Err(format!("need two equal-length share vectors of at least 2, got {} and {}", a.len(), b.len()));
    }
    let code = argmax_source(a.len());
    let t0 = Instant::now();
    let prg = compile_with_options(&code, CompileOptions { circuit_kind: CircuitKind::Register, ..Default::default() })
        .map_err(|e| e.prettify(&code))?;
    let circuit = prg.circuit.unwrap_register_ref();
    let compile_ms = t0.elapsed().as_secs_f64() * 1e3;

    let lit = |v: &[i32]| Literal::Array(v.iter().map(|&x| Literal::NumSigned(x as i64, SignedNumType::I32)).collect());
    let in0 = prg.literal_arg(0, lit(a)).map_err(|e| format!("party 0 input: {e}"))?.as_bits();
    let in1 = prg.literal_arg(1, lit(b)).map_err(|e| format!("party 1 input: {e}"))?.as_bits();

    let (ch0, ch1, sent, msgs) = MemChannel::pair();
    let p_out = [0usize, 1usize];
    let t1 = Instant::now();
    let (o0, o1) = join(
        mpc(&ch0, circuit, &in0, 0, 0, &p_out, None),
        mpc(&ch1, circuit, &in1, 0, 1, &p_out, None),
    )
    .await;
    let mpc_ms = t1.elapsed().as_secs_f64() * 1e3;
    let (o0, o1) = (o0.map_err(|e| format!("party 0: {e}"))?, o1.map_err(|e| format!("party 1: {e}"))?);
    if o0 != o1 {
        return Err("the two parties disagree on the output".into());
    }
    let arg = match prg.parse_output(&o0).map_err(|e| format!("output: {e}"))? {
        Literal::NumUnsigned(v, UnsignedNumType::U8) => v as u32,
        other => return Err(format!("expected a u8 index, got {other}")),
    };
    Ok(Outcome { arg, compile_ms, mpc_ms, bytes: sent.get(), messages: msgs.get(), and_ops: circuit.and_ops as u64 })
}

/// Browser entry: the same shares clientMPC.ts would build, and the measured outcome.
#[wasm_bindgen]
pub async fn secure_argmax(a: Vec<i32>, b: Vec<i32>) -> Result<JsValue, JsValue> {
    let r = secure_argmax_shares(&a, &b).await.map_err(|e| JsValue::from_str(&e))?;
    let o = js_sys::Object::new();
    for (k, v) in [
        ("arg", r.arg as f64),
        ("compileMs", r.compile_ms),
        ("mpcMs", r.mpc_ms),
        ("bytes", r.bytes as f64),
        ("messages", r.messages as f64),
        ("andOps", r.and_ops as f64),
    ] {
        js_sys::Reflect::set(&o, &JsValue::from_str(k), &JsValue::from_f64(v))?;
    }
    Ok(o.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shares clientMPC.ts builds: q = clamp(round((logit + 8) * 64), 0, 1024), a uniform in
    /// [0, BIG), b = q - a + BIG.
    fn shares(q: &[i64], a: &[i64]) -> (Vec<i32>, Vec<i32>) {
        (a.iter().map(|&x| x as i32).collect(), q.iter().zip(a).map(|(&q, &a)| (q - a + BIG) as i32).collect())
    }

    #[test]
    fn matches_the_plaintext_argmax() {
        let cases: [(&[i64], &[i64], u32); 4] = [
            (&[300, 700], &[12_345, 60_001], 1),
            (&[900, 100], &[0, 65_535], 0),
            (&[512, 512], &[7, 9], 0),
            (&[10, 1024, 400], &[1, 2, 3], 1),
        ];
        for (q, a, want) in cases {
            let (sa, sb) = shares(q, a);
            let r = futures::executor::block_on(secure_argmax_shares(&sa, &sb)).expect("mpc runs");
            assert_eq!(r.arg, want, "q = {q:?}");
            assert!(r.bytes > 0);
        }
    }
}
