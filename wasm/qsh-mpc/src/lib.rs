//! The demo's secure argmax, on Polytune, with both parties in one browser tab.
//!
//! `web/src/lib/clientMPC.ts` hands this the two parties' additive shares of N logits modulo 2^32
//! (`q_k = a_k + b_k mod 2^32`, `a_k` uniform over 32 bits) and gets back only the index of the
//! largest, computed under Polytune's WRK17b multi-party authenticated garbling (secure against
//! malicious adversaries controlling all but one party). Both parties run in this module over an
//! in-memory channel: the party separation is notional, as it was under emp-wasm, while the protocol,
//! its messages and its timing are real.
//!
//! Replaced emp-wasm + mpc-framework + summon-ts in Round 65.
//!
//! SHARING MODULO 2^32, ROUND 66. Until then the shares were `q_k = a_k + b_k - 2^16` with `a_k`
//! bounded to 2^16, which hides a value only statistically (two values up to 1024 apart are
//! distinguishable with advantage up to 1024 / 2^16 = 1/64). Modulo 2^32 each share alone is uniform
//! whatever the value. Garble panics on integer overflow rather than wrapping, so the circuit adds in
//! 64 bits, where two 32-bit values cannot overflow, and casts back to 32: a narrowing cast keeps the
//! low bits, which is the reduction mod 2^32.

use std::{cell::Cell, rc::Rc};

use futures::{StreamExt, channel::mpsc, future::join, lock::Mutex};
use polytune::{
    channel::Channel,
    garble_lang::{
        CircuitKind, CompileOptions, compile_with_options,
        literal::Literal,
        token::UnsignedNumType,
    },
    mpc,
};
use wasm_bindgen::prelude::*;
use web_time::Instant;

/// Garble source for the argmax of `n` values shared modulo 2^32, unrolled: `n - 1` comparisons, the
/// index of the first maximum wins.
pub fn argmax_source(n: usize) -> String {
    let mut s = format!("pub fn main(a: [u32; {n}], b: [u32; {n}]) -> u8 {{\n");
    s += "    let mut best: u32 = ((a[0usize] as u64) + (b[0usize] as u64)) as u32;\n";
    s += "    let mut arg: u8 = 0u8;\n";
    for k in 1..n {
        s += &format!("    let q{k}: u32 = ((a[{k}usize] as u64) + (b[{k}usize] as u64)) as u32;\n");
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
pub async fn secure_argmax_shares(a: &[u32], b: &[u32]) -> Result<Outcome, String> {
    if a.len() != b.len() || a.len() < 2 {
        return Err(format!("need two equal-length share vectors of at least 2, got {} and {}", a.len(), b.len()));
    }
    // The index comes back as a u8. Checked here, before anything is compiled, rather than left to
    // fail as an out-of-range literal inside the generated Garble source (Round 68).
    if a.len() > 255 {
        return Err(format!("at most 255 values: the argmax index is a u8, got {}", a.len()));
    }
    let code = argmax_source(a.len());
    let t0 = Instant::now();
    let prg = compile_with_options(&code, CompileOptions { circuit_kind: CircuitKind::Register, ..Default::default() })
        .map_err(|e| e.prettify(&code))?;
    let circuit = prg.circuit.unwrap_register_ref();
    let compile_ms = t0.elapsed().as_secs_f64() * 1e3;

    let lit = |v: &[u32]| Literal::Array(v.iter().map(|&x| Literal::NumUnsigned(x as u64, UnsignedNumType::U32)).collect());
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
pub async fn secure_argmax(a: Vec<u32>, b: Vec<u32>) -> Result<JsValue, JsValue> {
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

    /// The shares clientMPC.ts builds: q = clamp(round((logit + 8) * 64), 0, 1024), a uniform over
    /// 32 bits, b = q - a mod 2^32.
    fn shares(q: &[u32], a: &[u32]) -> (Vec<u32>, Vec<u32>) {
        (a.to_vec(), q.iter().zip(a).map(|(&q, &a)| q.wrapping_sub(a)).collect())
    }

    /// Masks across the whole 32-bit range, so most pairs wrap: a + b exceeds 2^32 whenever a > q.
    /// Taken in 32 bits, Garble would panic on exactly these; the test fails if the circuit does.
    #[test]
    fn matches_the_plaintext_argmax() {
        let cases: [(&[u32], &[u32], u32); 6] = [
            (&[300, 700], &[12_345, 4_000_000_001], 1),
            (&[900, 100], &[0, u32::MAX], 0),
            (&[512, 512], &[7, 9], 0),
            (&[10, 1024, 400], &[1, 2_147_483_648, 3], 1),
            (&[0, 1], &[u32::MAX, u32::MAX], 1),
            (&[1024, 0, 1023], &[u32::MAX - 5, 3_141_592_653, 1], 0),
        ];
        for (q, a, want) in cases {
            let (sa, sb) = shares(q, a);
            let r = futures::executor::block_on(secure_argmax_shares(&sa, &sb)).expect("mpc runs");
            assert_eq!(r.arg, want, "q = {q:?}");
            assert!(r.bytes > 0);
        }
    }

    /// Inputs the circuit cannot take are refused before anything is compiled or sent.
    #[test]
    fn refuses_inputs_it_cannot_index() {
        let run = |n: usize, m: usize| futures::executor::block_on(secure_argmax_shares(&vec![0; n], &vec![0; m]));
        assert!(run(1, 1).unwrap_err().contains("at least 2"));
        assert!(run(3, 2).unwrap_err().contains("equal-length"));
        assert!(run(256, 256).unwrap_err().contains("at most 255"));
    }

    /// The circuit size and traffic the register records, at the demo's 2 classes and at 10. Not part
    /// of the build: `cargo test --release -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn report_circuit_size() {
        for n in [2usize, 10] {
            let q: Vec<u32> = (0..n as u32).map(|k| (k * 97) % 1025).collect();
            let a: Vec<u32> = (0..n as u32).map(|k| k.wrapping_mul(2_654_435_761)).collect();
            let (sa, sb) = shares(&q, &a);
            let r = futures::executor::block_on(secure_argmax_shares(&sa, &sb)).expect("mpc runs");
            println!("n={n} and_ops={} bytes={} messages={}", r.and_ops, r.bytes, r.messages);
        }
    }
}
