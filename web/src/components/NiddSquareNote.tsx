/**
 * The caption under the telecom page's 5G-NIDD figure since Round 86's batch 12: what replaced what, and the
 * pre-registered run's figures, read from the network's file (`data/telecom_nidd/square_network.json`), so that no
 * figure here is typed by hand.
 */
import type { TelecomSquare } from "../lib/telecomSquareData";

export interface NiddSquareShown { figures: TelecomSquare["figures"]; set: string; record: string; split: string }

const f4 = (v: number) => v.toFixed(4);

export function NiddSquareNote({ square }: { square?: NiddSquareShown }) {
  if (!square) return null;
  const f = square.figures;
  const b = f.browser[square.set];
  const split = square.split.replace(/ \(references\/[^)]*\)$/, "");
  return (
    <p className="small muted" style={{ marginTop: 6 }}>
      This network replaced the linear student here on the owner&apos;s choice of 2026-11-28, after its pre-registered run
      met every criterion: on the same flows&apos; split, five new seeds scored a test FLamby metric of{" "}
      {f4(f.test_flamby_mean)} on average (lowest {f4(f.test_flamby_min)}) against the linear student&apos;s{" "}
      {f4(f.control_test_flamby)}, the margins chosen after an earlier run of this recipe had been seen
      {b ? <>; in a Web Worker the slowest of three runs under {square.set} took {(b.slowest_total_ms / 1000).toFixed(1)} s</> : null}{" "}
      (<code>{square.record}</code>). The agreement is between the encrypted and the plaintext network on the same flows and
      does not depend on the split; the network&apos;s accuracy does. These flows are {split}.
    </p>
  );
}
