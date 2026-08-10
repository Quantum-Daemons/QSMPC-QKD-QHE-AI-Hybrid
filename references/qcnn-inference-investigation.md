# The QCNN inference backend returned a coin flip, and the page called it a prediction
### Round 25, 2026-08-10

Selecting **Inference = QCNN (QHE)** on the finance, medical or genomics page routed the surfaced
class through `qotpQcnnInfer` and reported it as `primary_inference_class`. It carries no information
about the input.

## Measured

Same eight-feature vector, twelve different run seeds:

| seed | <Z0> | class |
|---|---|---|
| 0 | +0.0928 | 1 |
| 1 | -0.0024 | 0 |
| 2 | +0.0129 | 1 |
| 3 | +0.0016 | 1 |
| 4 | -0.0854 | 0 |
| 5 | -0.0618 | 0 |

Over 24 random feature vectors x 8 seeds each:

| quantity | value |
|---|---|
| vectors whose class FLIPS across seeds | **24 / 24 = 100%** |
| mean \|<Z0>\| with the QOTP mask | 0.0761 (max 0.2013) |
| mean \|<Z0>\| with the mask DISABLED | 0.0748 (max 0.2096) |
| unmasked class balance | 0.50 |
| correlation, unmasked class vs mean(feature) | **-0.070** |

## Two causes, and the obvious one is the smaller

**The QOTP mask is applied and never undone.** `qotpQcnnInfer` applies a random Pauli X/Z per qubit,
then runs the circuit, then reads <Z0> off the *masked* state. An X on qubit 0 flips the sign of
<Z0>, so the class inherits a coin flip from the seed. That is the intuitive explanation and it is
not the main one.

**The circuit has no trained parameters, and scrambles.** Disabling the mask entirely barely moves
the numbers - 0.0748 against 0.0761 - because seven fixed rounds of H, a CNOT ring and T on every
qubit drive the state toward uniform regardless of the input. <Z0> lands within +/-0.21 of zero for
every vector tested, and its sign is numerical noise. A correlation of -0.070 against the simplest
possible summary of the input is the whole story: **the readout is independent of the data.**

## Why nothing caught it

The project already knew the neighbouring row was empty. `clientCompute.ts` labels the benchmark
entry `"QOTP + N-qubit QCNN circuit (no trained weights)"` and states "There is nothing here to be
accurate ABOUT", and the browser checklist records that an invented `accuracy: 1.0` was removed from
exactly that row.

That scrutiny stopped at the benchmark row. The **inference selector** presents the same circuit's
output as `primary_inference_class`, beside the CKKS prediction, with a log line reading
`inference backend = QCNN (QHE): class=<c> (56 T-gates on 8 qubits, state-vector sim)` - every word
of which is true, and none of which says the number is meaningless. The T-gate count is genuinely
counted. The simulation is genuinely a state-vector simulation. The class is genuinely computed. It
just is not a prediction.

**The lesson is narrower than "check your models" and worth stating exactly:** a component honestly
labelled in one place will be trusted everywhere else it appears. The caveat lived in the benchmark
row's comment; the selector inherited the code and not the caveat.

## What a fix has to separate

QOTP hiding and QCNN prediction are two different demonstrations and were fused into one function.

- **Hiding** is real and worth showing: under a random Pauli mask the server's view is maximally
  mixed. That claim needs no trained weights and no meaningful readout.
- **Prediction** needs trainable parameters and an unmasked readout.

Undoing a QOTP mask *through* the circuit is not a detail that can be waved away: Pauli frames
propagate through Clifford gates but `T X T-dagger` is not Pauli, which is precisely the difficulty
Ortega & Martin-Delgado's scheme exists to handle. So the honest split is to keep the masked circuit
as the hiding demonstration and to give the prediction path its own unmasked, parameterised circuit,
rather than to claim a correction this repository does not implement.
