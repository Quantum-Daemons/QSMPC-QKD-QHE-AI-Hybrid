/**
 * What actually ran, as a badge beside the stage it describes.
 *
 * Four kinds, because three were being collapsed into two. "real" means the computation the label
 * names executed in the visitor's tab (ML-KEM, CKKS, a published model). "notional" means the protocol
 * and its bytes are real but the parties are not separate - the garbled-circuit 2PC runs both sides
 * over an in-memory transport in one tab. "simulated" means an analytic or state-vector model stands in
 * for hardware (QKD, QSDC, the quantum comparison). "seeded" is the quantum one-time pad: a real mask
 * from a seeded, non-cryptographic generator, so its structure is measured and its secrecy is not.
 */
export type StageKind = "real" | "notional" | "simulated" | "seeded";

const LABEL: Record<StageKind, string> = {
  real: "real, in your tab",
  notional: "real protocol, notional parties",
  simulated: "simulated",
  seeded: "simulated, seeded pad",
};

const TITLE: Record<StageKind, string> = {
  real: "The computation this label names executed in your browser tab.",
  notional: "The protocol, its bytes and its wall-clock are real; both parties run in this one tab over "
    + "an in-memory transport, so the party separation is not.",
  simulated: "An analytic or state-vector model stands in for quantum hardware; no QKD device or "
    + "quantum computer exists in this project.",
  seeded: "The one-time-pad mask is real but drawn from a seeded, non-cryptographic generator, so its "
    + "structure and cost are measured and its secrecy is not.",
};

export default function StageBadge({ kind, label }: { kind: StageKind; label?: string }) {
  return <span className={`stage-badge ${kind}`} title={TITLE[kind]}>{label ?? LABEL[kind]}</span>;
}
