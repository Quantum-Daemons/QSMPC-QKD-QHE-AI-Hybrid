# Which homomorphic scheme, and why not a CKKS/BFV hybrid
### Round 24g, 2026-08-10. A decision with the hardness result behind it.

The question asked was whether a CKKS + BFV **hybrid** exists that this project should adopt, and if
not, whether an integer scoring system under BFV is the right alternative. The answer to the first
is a clear no, and it is no for a stronger reason than "not implemented yet".

## CKKS to BFV scheme switching: research-only, and believed hard

**There is a peer-reviewed hardness result.** Eldefrawy, Genise and Manohar, *"On the Hardness of
Scheme-Switching Between SIMD FHE Schemes"*, PQCrypto 2023 (ePrint 2023/988):

> "we provide strong evidence that homomorphic switching between BGV/BFV and CKKS is as hard as
> bootstrapping... one cannot hope to obtain significant improvements to homomorphic
> scheme-switching without also significantly improving the state-of-the-art for bootstrapping."

The problem was posed by Halevi and Shoup at Eurocrypt 2015 and remains open. So this is not a gap
waiting on engineering effort.

**The framework that specified it never implemented it.** CHIMERA (ePrint 2018/758, J. Math.
Cryptology 2020) defines bridges between TFHE, B/FV and HEAAN, but there is no direct B/FV-to-HEAAN
bridge - BFV routes *through* TFHE - and the paper states plainly that the BFV bridges were left
unimplemented: *"we implemented the TFHE/HEAAN bridges, and we left the implementation of the
TFHE-B/FV bridges as a future work."* The reference code has not been touched since 2018-09-07 and
carries no licence file, so it could not be used here even if it worked.

**Everything that does work switches CKKS to FHEW/TFHE, which is a different thing.** PEGASUS
(IEEE S&P 2021) is CKKS-to-FHEW. OpenFHE added scheme switching in v1.1.0 and it is
`EvalCKKStoFHEW` / `EvalFHEWtoCKKS`; searching its source for `CKKStoBFV`, `BFVtoCKKS`,
`BGVtoCKKS` or `CKKStoBGV` returns zero hits for each.

**And none of it is reachable from a browser.** Microsoft SEAL implements no scheme switching at
all, so `node-seal` cannot expose it. The only browser path to OpenFHE is `openfhe-wasm`, whose
CMake binds `pke` and `core` and **not** `binfhe` - so `EvalCKKStoFHEW` is absent there too. It is
also self-described work-in-progress with no multithreading, while OpenFHE itself has moved on
several minor versions.

For scale, the nearest working system costs 40 s to 8.4 minutes per switch on native multithreaded
CPU at ring degree 2^16, against this project's 8192.

**ISO/IEC 28033-5** is titled *Mechanisms for scheme switching* and is the laggard of that series at
Working Draft 20.60 (2025-08-16), while Part 3 (CKKS) is at FDIS 50.00. The standards process agrees
with the literature about maturity.

## What the goal actually needs: two contexts, not one hybrid

The intended use is an integer-coefficient scoring system - coefficients in {-10..10} from the
sparse integer program of Chi and Birbil (arXiv:2604.04241) - evaluated under encryption. That needs
a **second, independent BFV context**, not a conversion between schemes. Nothing is ever switched:
the CKKS models run in a CKKS context, the scorecard runs in a BFV context, and the two never
exchange a ciphertext.

That is available today with zero new dependencies. `node-seal` 7.0.0 already exposes `bfv`, `bgv`
and `ckks` plus `BatchEncoder`, verified directly against the installed package.

| | CKKS (current) | BFV (scorecard) |
|---|---|---|
| arithmetic | approximate, over reals | **exact, over integers** |
| slots at N=8192 | 4096 complex | **8192 integer** |
| rescaling / scale management | required | none |
| plaintext-encrypted agreement | to a precision bound | **bit-exact** |

The exactness is the point, and it is a better demonstration than the approximate path rather than
merely a different one: an encrypted score that reproduces the plaintext score *exactly* removes a
whole class of question about whether the encrypted tier is faithful. It also composes with the
operating-point work already done, because the integer program optimises net benefit directly and
the resulting model is a single packed dot product by construction.

## Decision

- **No scheme switching.** Research-only, believed as hard as bootstrapping, unimplemented for this
  scheme pair, and unreachable from a browser regardless.
- **Add BFV as a second, independent context** for an integer scoring system. No conversion, no new
  dependency.
- **Keep CKKS** for the existing real-valued students. It remains the right scheme for a dot product
  over reals, and nothing here argues otherwise.

## What would change this

If `openfhe-wasm` ever binds `binfhe`, CKKS-to-FHEW switching becomes browser-reachable and worth
re-examining - for large-precision sign evaluation and small-precision lookup tables, which is what
the OpenFHE maintainers say the capability is actually for. That is a different use case from this
one and would not be a CKKS/BFV hybrid.
