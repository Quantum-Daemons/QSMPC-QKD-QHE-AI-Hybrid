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

use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    rc::Rc,
};

use futures::{StreamExt, channel::mpsc, future::{join, join_all}, lock::Mutex};
use polytune::{
    channel::Channel,
    garble_lang::{
        CircuitKind, CompileOptions, GarbleProgram, compile_with_options,
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

    /// `n` fully connected endpoints, and the shared counts, for a protocol with more than two parties.
    pub fn mesh(n: usize) -> (Vec<MemChannel>, Rc<Cell<u64>>, Rc<Cell<u64>>) {
        let (sent, msgs) = (Rc::new(Cell::new(0)), Rc::new(Cell::new(0)));
        let mut txs: Vec<Vec<_>> = (0..n).map(|_| (0..n).map(|_| None).collect()).collect();
        let mut rxs: Vec<Vec<_>> = (0..n).map(|_| (0..n).map(|_| None).collect()).collect();
        for a in 0..n {
            for b in 0..n {
                if a != b {
                    let (t, r) = mpsc::unbounded();
                    txs[a][b] = Some(t);
                    rxs[b][a] = Some(Mutex::new(r));
                }
            }
        }
        let ends = txs.into_iter().zip(rxs)
            .map(|(tx, rx)| MemChannel { tx, rx, sent: sent.clone(), msgs: msgs.clone() })
            .collect();
        (ends, sent, msgs)
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

/// Garble source for a batch of `n` ReLUs, bit by bit, so an adder costs one AND gate per bit where Garble's own `+` costs three.
///
/// Party 0 holds `y[i]`, party 1 holds `r[i]` and `s[i]`, all least significant bit first. `x = y + r mod 2^l`, read as signed; `z` is `x` where
/// `x >= 0` and 0 otherwise; the output is `z + s`, over the integers, in `b + 1` bits. Over the integers because the result goes back into CKKS
/// as a real number, which a value reduced mod 2^b cannot do. 86 AND gates per ReLU at l = 20 and b = 48.
pub fn relu_source(n: usize, l: usize, b: usize) -> String {
    let msb = l - 1;
    let out = b + 1;
    format!(
        "fn relu1(y: [bool; {l}], r: [bool; {l}], s: [bool; {b}]) -> [bool; {out}] {{
    let mut x = [false; {l}];
    let mut c = false;
    for i in 0usize..{l}usize {{
        x[i] = y[i] ^ r[i] ^ c;
        c = ((y[i] ^ c) & (r[i] ^ c)) ^ c;
    }}
    let keep = !x[{msb}usize];
    let mut z = [false; {b}];
    for i in 0usize..{msb}usize {{
        z[i] = x[i] & keep;
    }}
    let mut o = [false; {out}];
    let mut k = false;
    for i in 0usize..{b}usize {{
        o[i] = z[i] ^ s[i] ^ k;
        k = ((z[i] ^ k) & (s[i] ^ k)) ^ k;
    }}
    o[{b}usize] = k;
    o
}}
pub fn main(y: [[bool; {l}]; {n}], rs: [([bool; {l}], [bool; {b}]); {n}]) -> [[bool; {out}]; {n}] {{
    let mut o = [[false; {out}]; {n}];
    for i in 0usize..{n}usize {{
        let (r, s) = rs[i];
        o[i] = relu1(y[i], r, s);
    }}
    o
}}
"
    )
}

thread_local! {
    /// One compiled program per (n, l, b). A page runs fourteen batches of one size, and compiling is the slow part: 0.9 s against 0.4 s for one run,
    /// natively, at 64 ReLUs, and it grows faster than the circuit does (75 s at 800).
    static RELU_PROGRAMS: RefCell<HashMap<(usize, usize, usize), Rc<GarbleProgram>>> = RefCell::new(HashMap::new());
}

fn relu_program(n: usize, l: usize, b: usize) -> Result<(Rc<GarbleProgram>, f64), String> {
    if let Some(p) = RELU_PROGRAMS.with(|m| m.borrow().get(&(n, l, b)).cloned()) {
        return Ok((p, 0.0));
    }
    let code = relu_source(n, l, b);
    let t0 = Instant::now();
    let prg = compile_with_options(&code, CompileOptions { circuit_kind: CircuitKind::Register, ..Default::default() })
        .map_err(|e| e.prettify(&code))?;
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let prg = Rc::new(prg);
    RELU_PROGRAMS.with(|m| m.borrow_mut().insert((n, l, b), prg.clone()));
    Ok((prg, ms))
}

fn bits(v: u64, width: usize) -> Literal {
    Literal::Array((0..width).map(|i| if (v >> i) & 1 == 1 { Literal::True } else { Literal::False }).collect())
}

/// One batch of ReLUs: what party 0 receives, and what the batch cost.
#[derive(Debug, Clone)]
pub struct ReluOutcome {
    pub out: Vec<u64>,
    pub compile_ms: f64,
    pub mpc_ms: f64,
    pub bytes: u64,
    pub messages: u64,
    pub and_ops: u64,
}

/// A batch of ReLUs under Polytune. Only party 0 receives the outputs, each masked by `s`, which only party 1 knows.
pub async fn secure_relu_shares(y: &[u32], r: &[u32], s: &[u64], l: usize, b: usize) -> Result<ReluOutcome, String> {
    let n = y.len();
    if n == 0 || r.len() != n || s.len() != n {
        return Err(format!("need three equal, non-empty vectors, got {n}, {} and {}", r.len(), s.len()));
    }
    if !(2..=32).contains(&l) || b < l || b > 52 {
        return Err(format!("need 2 <= l <= 32 and l <= b <= 52 (a double holds an output exactly below 2^53), got l = {l} and b = {b}"));
    }
    let ring = 1u64 << l;
    if y.iter().chain(r).any(|&v| (v as u64) >= ring) || s.iter().any(|&v| v >= (1u64 << b)) {
        return Err("a share is out of range for its ring".into());
    }
    let (prg, compile_ms) = relu_program(n, l, b)?;
    let circuit = prg.circuit.unwrap_register_ref();
    let lit0 = Literal::Array(y.iter().map(|&v| bits(v as u64, l)).collect());
    let lit1 = Literal::Array(r.iter().zip(s).map(|(&a, &c)| Literal::Tuple(vec![bits(a as u64, l), bits(c, b)])).collect());
    let in0 = prg.literal_arg(0, lit0).map_err(|e| format!("party 0 input: {e}"))?.as_bits();
    let in1 = prg.literal_arg(1, lit1).map_err(|e| format!("party 1 input: {e}"))?.as_bits();

    let (ch0, ch1, sent, msgs) = MemChannel::pair();
    let p_out = [0usize];
    let t1 = Instant::now();
    let (o0, o1) = join(mpc(&ch0, circuit, &in0, 0, 0, &p_out, None), mpc(&ch1, circuit, &in1, 0, 1, &p_out, None)).await;
    let mpc_ms = t1.elapsed().as_secs_f64() * 1e3;
    let (o0, o1) = (o0.map_err(|e| format!("party 0: {e}"))?, o1.map_err(|e| format!("party 1: {e}"))?);
    if !o1.is_empty() {
        return Err(format!("party 1 is not an output party and it received {} bits", o1.len()));
    }
    let Literal::Array(rows) = prg.parse_output(&o0).map_err(|e| format!("output: {e}"))? else {
        return Err("expected an array of outputs".into());
    };
    let mut out = Vec::with_capacity(n);
    for row in rows {
        let Literal::Array(row_bits) = row else {
            return Err("expected an array of bits".into());
        };
        out.push(row_bits.iter().enumerate().fold(0u64, |v, (i, bit)| v | ((matches!(bit, Literal::True) as u64) << i)));
    }
    Ok(ReluOutcome { out, compile_ms, mpc_ms, bytes: sent.get(), messages: msgs.get(), and_ops: circuit.and_ops as u64 })
}

/// Browser entry for one batch of ReLUs.
#[wasm_bindgen]
pub async fn secure_relu(y: Vec<u32>, r: Vec<u32>, s: Vec<u64>, l: usize, b: usize) -> Result<JsValue, JsValue> {
    let o = secure_relu_shares(&y, &r, &s, l, b).await.map_err(|e| JsValue::from_str(&e))?;
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(&obj, &JsValue::from_str("out"), &js_sys::BigUint64Array::from(&o.out[..]))?;
    for (k, v) in [
        ("compileMs", o.compile_ms),
        ("mpcMs", o.mpc_ms),
        ("bytes", o.bytes as f64),
        ("messages", o.messages as f64),
        ("andOps", o.and_ops as f64),
    ] {
        js_sys::Reflect::set(&obj, &JsValue::from_str(k), &JsValue::from_f64(v))?;
    }
    Ok(obj.into())
}

/// Most parties and IDs a page may ask for: the circuit's size is quadratic in `k` and the protocol's cost in `n`.
pub const PSI_MAX_PARTIES: usize = 5;
pub const PSI_MAX_IDS: usize = 16;

/// Garble source for the intersection of `n` parties' sets of `k` 32-bit IDs. Every element of party 0's set is compared with every element of
/// the others', and the output is that element where all the others hold it and 0 where any does not, so an ID must never be 0.
pub fn psi_source(n: usize, k: usize) -> String {
    let params: Vec<String> = (0..n).map(|p| format!("s{p}: [u32; {k}]")).collect();
    let mut s = format!("pub fn main({}) -> [u32; {k}] {{\n", params.join(", "));
    for i in 0..k {
        s += &format!("    let x{i} = s0[{i}usize];\n    let mut all{i} = true;\n");
        for p in 1..n {
            s += &format!("    let mut f{i}_{p} = false;\n");
            for l in 0..k {
                s += &format!("    if x{i} == s{p}[{l}usize] {{ f{i}_{p} = true; }}\n");
            }
            s += &format!("    if !f{i}_{p} {{ all{i} = false; }}\n");
        }
        s += &format!("    let y{i} = if all{i} {{ x{i} }} else {{ 0u32 }};\n");
    }
    let ys: Vec<String> = (0..k).map(|i| format!("y{i}")).collect();
    s += &format!("    [{}]\n}}\n", ys.join(", "));
    s
}

/// One n-party PSI: the output for party 0's positions, how long it took and what the parties exchanged.
#[derive(Debug, Clone)]
pub struct PsiOutcome {
    pub ids: Vec<u32>,
    pub compile_ms: f64,
    pub mpc_ms: f64,
    pub bytes: u64,
    pub messages: u64,
    pub and_ops: u64,
}

/// Runs the intersection with one party per set, every party receiving the output, under Polytune's WRK17b protocol.
pub async fn secure_psi_ids(sets: &[Vec<u32>]) -> Result<PsiOutcome, String> {
    let n = sets.len();
    if !(2..=PSI_MAX_PARTIES).contains(&n) {
        return Err(format!("need between 2 and {PSI_MAX_PARTIES} parties, got {n}"));
    }
    let k = sets[0].len();
    if k == 0 || k > PSI_MAX_IDS || sets.iter().any(|s| s.len() != k) {
        return Err(format!("each party needs the same number of IDs, between 1 and {PSI_MAX_IDS}"));
    }
    if sets.iter().flatten().any(|&id| id == 0) {
        return Err("an ID must not be 0, which the circuit's output uses for no match".into());
    }
    let code = psi_source(n, k);
    let t0 = Instant::now();
    let prg = compile_with_options(&code, CompileOptions { circuit_kind: CircuitKind::Register, ..Default::default() })
        .map_err(|e| e.prettify(&code))?;
    let circuit = prg.circuit.unwrap_register_ref();
    let compile_ms = t0.elapsed().as_secs_f64() * 1e3;

    let lit = |v: &[u32]| Literal::Array(v.iter().map(|&x| Literal::NumUnsigned(x as u64, UnsignedNumType::U32)).collect());
    let mut inputs = Vec::with_capacity(n);
    for (p, set) in sets.iter().enumerate() {
        inputs.push(prg.literal_arg(p, lit(set)).map_err(|e| format!("party {p} input: {e}"))?.as_bits());
    }
    let (ends, sent, msgs) = MemChannel::mesh(n);
    let p_out: Vec<usize> = (0..n).collect();
    let t1 = Instant::now();
    let runs = ends.iter().zip(&inputs).enumerate().map(|(p, (ch, input))| mpc(ch, circuit, input, 0, p, &p_out, None));
    let outputs = join_all(runs).await;
    let mpc_ms = t1.elapsed().as_secs_f64() * 1e3;

    let mut first: Option<Vec<bool>> = None;
    for (p, o) in outputs.into_iter().enumerate() {
        let o = o.map_err(|e| format!("party {p}: {e}"))?;
        match &first {
            None => first = Some(o),
            Some(f) if *f != o => return Err(format!("party {p} disagrees with party 0 on the output")),
            Some(_) => {}
        }
    }
    let bits = first.ok_or("no output")?;
    let ids = match prg.parse_output(&bits).map_err(|e| format!("output: {e}"))? {
        Literal::Array(items) => items
            .into_iter()
            .map(|l| match l {
                Literal::NumUnsigned(v, UnsignedNumType::U32) => Ok(v as u32),
                other => Err(format!("expected a u32, got {other}")),
            })
            .collect::<Result<Vec<u32>, String>>()?,
        other => return Err(format!("expected an array, got {other}")),
    };
    Ok(PsiOutcome { ids, compile_ms, mpc_ms, bytes: sent.get(), messages: msgs.get(), and_ops: circuit.and_ops as u64 })
}

/// Browser entry: `n` sets of `k` IDs, concatenated.
#[wasm_bindgen]
pub async fn secure_psi(flat: Vec<u32>, n: usize, k: usize) -> Result<JsValue, JsValue> {
    if n == 0 || k == 0 || flat.len() != n * k {
        return Err(JsValue::from_str(&format!("expected {n} sets of {k} IDs, got {} IDs", flat.len())));
    }
    let sets: Vec<Vec<u32>> = flat.chunks(k).map(|c| c.to_vec()).collect();
    let r = secure_psi_ids(&sets).await.map_err(|e| JsValue::from_str(&e))?;
    let o = js_sys::Object::new();
    js_sys::Reflect::set(&o, &JsValue::from_str("ids"), &js_sys::Uint32Array::from(r.ids.as_slice()).into())?;
    for (key, v) in [
        ("compileMs", r.compile_ms),
        ("mpcMs", r.mpc_ms),
        ("bytes", r.bytes as f64),
        ("messages", r.messages as f64),
        ("andOps", r.and_ops as f64),
    ] {
        js_sys::Reflect::set(&o, &JsValue::from_str(key), &JsValue::from_f64(v))?;
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

    struct XorShift(u64);
    impl XorShift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// Values at the sign boundary and at the ring's edges, with masks across the whole ring, so most reconstructions wrap.
    fn relu_case(n: usize, l: usize, b: usize) -> (Vec<i64>, Vec<u32>, Vec<u32>, Vec<u64>) {
        let ring = 1u64 << l;
        let half = 1i64 << (l - 1);
        let mut g = XorShift(0x9E37_79B9_7F4A_7C15);
        let (mut x, mut y, mut r, mut s) = (vec![], vec![], vec![], vec![]);
        for i in 0..n {
            let v = match i % 6 {
                0 => -1,
                1 => 0,
                2 => 1,
                3 => half - 1,
                4 => -half,
                _ => (g.next() % ring) as i64 - half,
            };
            let m = g.next() & (ring - 1);
            x.push(v);
            y.push((v as u64).wrapping_add(m) as u32 & (ring - 1) as u32);
            r.push(m.wrapping_neg() as u32 & (ring - 1) as u32);
            s.push(g.next() & ((1u64 << b) - 1));
        }
        (x, y, r, s)
    }

    /// The circuit against the plaintext ReLU, over the integers: the output is relu(x) + s exactly, including past 2^b.
    #[test]
    fn relu_matches_the_plaintext() {
        for (l, b) in [(20, 48), (20, 42), (8, 8)] {
            let (x, y, r, s) = relu_case(18, l, b);
            let o = futures::executor::block_on(secure_relu_shares(&y, &r, &s, l, b)).expect("mpc runs");
            for i in 0..x.len() {
                assert_eq!(o.out[i], x[i].max(0) as u64 + s[i], "l = {l}, b = {b}, i = {i}, x = {}", x[i]);
            }
            assert!(o.bytes > 0);
        }
    }

    /// The AND count the design rests on: one per adder bit and one per bit gated by the sign.
    #[test]
    fn relu_costs_what_the_design_says() {
        let (_, y, r, s) = relu_case(4, 20, 48);
        let o = futures::executor::block_on(secure_relu_shares(&y, &r, &s, 20, 48)).expect("mpc runs");
        assert_eq!(o.and_ops, 4 * (19 + 19 + 48));
    }

    /// Inputs the circuit cannot take are refused before anything is compiled or sent.
    #[test]
    fn relu_refuses_what_it_cannot_take() {
        let run = |y: &[u32], r: &[u32], s: &[u64], l, b| futures::executor::block_on(secure_relu_shares(y, r, s, l, b));
        assert!(run(&[], &[], &[], 20, 48).unwrap_err().contains("non-empty"));
        assert!(run(&[0], &[0, 1], &[0], 20, 48).unwrap_err().contains("equal"));
        assert!(run(&[0], &[0], &[0], 20, 53).unwrap_err().contains("below 2^53"));
        assert!(run(&[1 << 20], &[0], &[0], 20, 48).unwrap_err().contains("out of range"));
        assert!(run(&[0], &[0], &[1 << 48], 20, 48).unwrap_err().contains("out of range"));
    }

    fn psi_case(n: usize, k: usize, shared: usize) -> (Vec<Vec<u32>>, Vec<u32>) {
        let common: Vec<u32> = (0..shared as u32).map(|j| 1000 + j).collect();
        let sets: Vec<Vec<u32>> = (0..n as u32)
            .map(|p| {
                let mut s = common.clone();
                s.extend((shared as u32..k as u32).map(|j| 2000 + 100 * p + j));
                s
            })
            .collect();
        (sets, common)
    }

    #[test]
    fn psi_matches_the_plain_intersection() {
        for (n, k, shared) in [(2, 4, 1), (3, 4, 2), (4, 3, 1), (5, 2, 1)] {
            let (sets, common) = psi_case(n, k, shared);
            let r = futures::executor::block_on(secure_psi_ids(&sets)).unwrap();
            let mut got: Vec<u32> = r.ids.iter().copied().filter(|&x| x != 0).collect();
            got.sort_unstable();
            assert_eq!(got, common, "n={n} k={k}");
            assert_eq!(r.ids.len(), k, "one output per element of party 0's set");
        }
    }

    #[test]
    fn psi_finds_nothing_when_one_party_holds_none_of_them() {
        let (mut sets, _) = psi_case(3, 3, 2);
        sets[2] = vec![7001, 7002, 7003];
        let r = futures::executor::block_on(secure_psi_ids(&sets)).unwrap();
        assert!(r.ids.iter().all(|&x| x == 0), "{:?}", r.ids);
    }

    #[test]
    fn psi_costs_what_the_design_says() {
        for (n, k) in [(3usize, 4usize), (3, 8), (5, 8)] {
            let code = psi_source(n, k);
            let prg = compile_with_options(&code, CompileOptions { circuit_kind: CircuitKind::Register, ..Default::default() }).unwrap();
            let expected = k * ((n - 1) * k * 31 + (n - 1) * (k - 1) + (n - 2) + 32);
            assert_eq!(prg.circuit.unwrap_register_ref().and_ops as usize, expected, "n={n} k={k}");
        }
    }

    #[test]
    fn psi_refuses_what_it_cannot_take() {
        assert!(futures::executor::block_on(secure_psi_ids(&[vec![1, 2]])).is_err(), "one party");
        assert!(futures::executor::block_on(secure_psi_ids(&vec![vec![1]; 6])).is_err(), "six parties");
        assert!(futures::executor::block_on(secure_psi_ids(&[vec![1, 2], vec![3]])).is_err(), "unequal sets");
        assert!(futures::executor::block_on(secure_psi_ids(&[vec![1, 0], vec![3, 4]])).is_err(), "an ID of 0");
        assert!(futures::executor::block_on(secure_psi_ids(&[vec![1; 17], vec![2; 17]])).is_err(), "too many IDs");
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
