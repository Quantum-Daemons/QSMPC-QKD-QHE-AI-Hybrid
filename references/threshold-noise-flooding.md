# Noise flooding in threshold decryption: what was read, and what the module does

Read on 2026-11-13 for the medical page's threshold CKKS and BGV rows, the genomics card's threshold sum and the DeFi
committee to come, all on Lattigo v6.2.0 in `wasm/qsh-fedthresh`. A research agent read the papers (several from
archived copies of the ePrint PDFs, which returned 403 directly) and the libraries' source at the versions named;
"read here" marks what was read again for this file, and the rest is agent-read. Arithmetic marked [calc] is this
project's, for N = 8192, four parties, error deviation 3.2 and bound 19.2, ternary secrets.

## The smudging lemma

| source | distribution | distance to a shift by x | unit |
|---|---|---|---|
| Asharov, Jain, Lopez-Alt, Tromer, Vaikuntanathan and Wichs, ePrint 2011/613 (EUROCRYPT 2012), Lemma 2.1 | uniform on $`[-B_2, B_2]`$ | $`B_1/B_2`$ for $`\|x\| \le B_1`$ | one integer |
| Colin de Verdiere, Passelegue and Stehle, ePrint 2026/031, Lemma 2.1 | uniform | $`2NB/B_{SM}`$ | a ring element |
| Mouchet, Troncoso-Pastoriza, Bossuat and Hubaux, ePrint 2020/304, section IV.E | discrete Gaussian, variance $`2^{\lambda}`$ times the ciphertext's | argued per coefficient, treating the noise as Gaussian | one coefficient |
| Canonne, Kamath and Steinke, arXiv:2004.00010, Prop. 5 (with Pinsker) [calc] | exact discrete Gaussian | at most $`\|x\|_2/(2\sigma)`$ | a ring element |

For a uniform on $`[-F, F]`$ the distance is exactly $`1 - \prod_i (1 - |x_i|/(2F+1)) \le \|x\|_1/(2F+1)`$, which is
what the module sizes $`F`$ by.

## The bound

- Li and Micciancio, ePrint 2020/1533 (EUROCRYPT 2021): one decrypted CKKS output can recover the key; repeated
  decryptions average unbiased noise out, so each ciphertext is decrypted once.
- Li, Micciancio, Schultz and Sorrell, ePrint 2022/816: flooding against a worst-case static estimate; noise sized to
  the actual ciphertext's error is insecure.
- Guo, Nabokov, Suvanto and Johansson, USENIX Security 2024 (read here, the abstract): key recovery from one shared
  OpenFHE decryption when the flooding rests on non-worst-case noise estimates; they call for worst-case estimates.
  Alexandru et al., ePrint 2024/203, answer that the attacks ran a circuit other than the one estimated, and that the
  worst case should range over inputs and circuits.
- Cheon, Choe, Passelegue, Stehle and Suvanto, ePrint 2024/127 (CCS 2024): exact schemes need a decryption-failure
  probability near $`2^{-128}`$, and a partial decryption counts as a decryption query.
- The worst case for this page, one encryption under the collective key decrypting with noise
  $`u e_{pk} + e_0 + e_1 s`$ (Mouchet et al., eq. 8): $`B = k(2Nkb + b)`$ for $`k`$ parties, $`2^{22.26}`$ for four
  [calc]. A conditional Gaussian bound, worst over the ternary values at failure probability $`2^{-41}`$, would be
  about $`2^{14.74}`$ and the average case $`2^{13.45}`$ [calc], which is where the measured $`2^{12.3}`$ to
  $`2^{12.8}`$ sat.

## How libraries flood

| library | where | what |
|---|---|---|
| OpenFHE v1.6.1 | `src/pke/lib/schemerns/rns-multiparty.cpp` | BFV/BGV: a uniform ring element; CKKS: a Gaussian sized from an empirical noise estimate |
| Lattigo v6.2.0 | `multiparty/keyswitch_sk.go` | a discrete Gaussian only; the deviation passed is combined with the fresh noise and the bound set to six deviations; `SECURITY.md` asks for exponential flooding and forbids retries |
| Zama threshold-fhe | `src/execution/small_execution/prss.rs` | the sum of two uniform terms in $`\pm 2^{40} B`$ (40 bits) |
| fhe.rs | `crates/fhe/src/mbfv/secret_key_switch.rs` | the ordinary error variance, with a TODO to make it exponential |

## What the module does (since 2026-11-13)

Each share's flooding is drawn in `wasm/qsh-fedthresh/agg/agg.go`: in every coefficient an integer uniform on
$`[-F, F]`$, $`F = 2^f - 1`$, from `crypto/rand`, reduced exactly into every limb; Lattigo's own share noise stays at
its fresh level. $`f`$ is the least with $`NB/(2F+1) \le 2^{-\lambda}`$ over a whole share, for the analytic worst
case $`B`$: 51, 75 and 163 at 16, 40 and 128 bits for four parties. The CKKS rows run at scales $`2^{80}`$ and
$`2^{100}`$; the BGV row in a 200-bit $`Q`$, inside the 202 bits of the HomomorphicEncryption.org standard's
post-quantum table at $`N = 8192`$, where the four bounded floods cannot exceed $`Q/(2t)`$, so it cannot fail to
decrypt. Shares under one key add up: a run of $`r`$ decryptions keeps $`\lambda - \log_2 r`$ bits over the key's
lifetime, printed on the page.

## Not used, and why

- A Gaussian against a conditional bound saves 6 to 8 bits, holds only for semi-honest parties and fresh ciphertexts,
  and adds its failure probability to the distance.
- An exact discrete Gaussian (Canonne, Kamath and Steinke, Algorithm 3, or Karney, arXiv:1303.6257) saves about 5.7
  bits of deviation and needs a big-integer sampler per coefficient.
- Renyi-divergence analyses that allow polynomial flooding (Boudgoust and Scholl, ePrint 2023/016) have an attack
  reported against them (Passelegue and Stehle, ePrint 2024/1984), and simulation security is said to need
  super-polynomial flooding.
