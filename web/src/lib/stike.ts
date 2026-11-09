/**
 * One-way quasi-QSDC (STIKE), the satellite telemetry's path when two-way QSDC cannot close (Round 86).
 * Twin of `src/qsh/qkd/stike.py`, which carries the reasoning and the sources: Pan et al., Sci. Adv. 11,
 * eadt4627 (2025). Eq. 3 is the BB84 key rate, so it comes from the QKD stage's own `decoySecureFraction`.
 */
import { qkdCfg } from "./config";
import { channelLossDb, decoySecureFraction, eDetForBound, h2, receiverOpticsDb, rng, sourceOf, transmittance,
  type ChannelParams } from "./quantumSim";

/** Bob keeps a detected pulse only when his basis matches Alice's. */
const SIFT = 0.5;

export interface StikeRates {
  /** Eq. 3, per signal pulse: the key the link distils. */
  secrecy: number;
  /** Eq. 4, per signal pulse: what the link can carry reliably. */
  reliable: number;
  /** q Q_mu: the chance a signal pulse reaches Bob in the matching basis. */
  retained: number;
  gain: number;
  qber: number;
  singlePhotonError: number;
}

export function stikeRates(ch: ChannelParams, fEC: number = qkdCfg().error_correction_eff): StikeRates {
  const eta = transmittance(ch);
  const src = sourceOf(ch);
  const y0 = 2 * ch.darkCount;
  const r = decoySecureFraction(eta, src.mu, src.nu, eDetForBound(ch), y0, fEC, SIFT, src.vacuum);
  const gain = y0 + 1 - Math.exp(-eta * src.mu);
  return {
    secrecy: r.perPulse, reliable: Math.max(0, SIFT * gain * (1 - fEC * h2(r.eMu))), retained: SIFT * gain,
    gain, qber: r.eMu, singlePhotonError: r.e1,
  };
}

export type StikeStatus = "delivered" | "no_secure_rate" | "pool_too_small" | "undecodable";

export interface StikeResult {
  status: StikeStatus;
  rates: StikeRates;
  messageBits: number;
  signalPulses: number;
  pulsesSent: number;
  seconds: number;
  keyInFlightBits: number;
  keyConsumedBits: number;
  keyReturnedBits: number;
  keyRegeneratedBits: number;
  poolBeforeBits: number;
  poolAfterBits: number;
  wrongBits: number;
  erasedBits: number;
}

/** How many signal pulses carry one message bit, for `votesPerBit` retained pulses on average. */
export const pulsesPerBit = (rates: StikeRates, votesPerBit: number): number =>
  rates.retained > 0 ? Math.ceil(votesPerBit / rates.retained) : 0;

/** Send `messageBytes` one way over the channel statistics `rates` describes. The pad and the mask cancel exactly, so
 *  what is drawn is which pulses Bob retains and which of those flip. Retained pulses are stepped through by their
 *  geometric gaps, so the cost follows what is retained and not the pulses sent. */
export function stikeSend(messageBytes: number, rates: StikeRates, o: {
  pSignal: number; sourceRateHz: number; poolBits: number; votesPerBit: number; seed?: number;
}): StikeResult {
  const messageBits = messageBytes * 8;
  const perBit = pulsesPerBit(rates, o.votesPerBit);
  const signalPulses = messageBits * perBit;
  const pulsesSent = o.pSignal > 0 ? Math.ceil(signalPulses / o.pSignal) : 0;
  const seconds = o.sourceRateHz > 0 ? pulsesSent / o.sourceRateHz : Infinity;
  const outcome = (status: StikeStatus, consumed = 0, regenerated = 0, wrong = 0, erased = 0, inFlight = 0): StikeResult => ({
    status, rates, messageBits, signalPulses, pulsesSent, seconds, keyInFlightBits: inFlight, keyConsumedBits: consumed,
    keyReturnedBits: inFlight - consumed, keyRegeneratedBits: regenerated, poolBeforeBits: o.poolBits,
    poolAfterBits: o.poolBits - consumed + regenerated, wrongBits: wrong, erasedBits: erased,
  });
  if (rates.secrecy <= 0) return outcome("no_secure_rate");
  if (signalPulses > o.poolBits) return outcome("pool_too_small");

  const u = rng(o.seed ?? 0);
  const logMiss = Math.log1p(-rates.retained);
  let consumed = 0, wrong = 0, erased = 0;
  for (let bit = 0; bit < messageBits; bit++) {
    let position = 0, votes = 0, errors = 0;
    for (;;) {
      position += 1 + Math.floor(Math.log1p(-u()) / logMiss);
      if (position > perBit) break;
      votes++;
      if (u() < rates.qber) errors++;
    }
    consumed += votes;
    if (votes === 0) erased++;
    else if (2 * errors >= votes) wrong++;
  }
  const regenerated = Math.floor(rates.secrecy * signalPulses);
  return outcome(erased || wrong ? "undecodable" : "delivered", consumed, regenerated, wrong, erased, signalPulses);
}

const MAX_PLOT_LOSS_DB = 70;

export interface QsdcCurveRow {
  lossDb: number;
  oneWaySecrecy: number | null;
  oneWayReliable: number | null;
  twoWay: number | null;
  /** Set on the rows that carry a published measurement, and only there. */
  measured?: number;
  source?: string;
}

/** Both bounds against the loss from the transmitter to the detector's output, on this link's source, error floor and
 *  station. The two-way bound's g leaves the station's optics out, as Qi et al.'s did, and counts the detector.
 *  `alsoAtDb` adds samples at those losses, in place of the grid sample nearest each, so a reader hovering a published
 *  point reads that point and the bounds at the same loss. */
export function qsdcCurves(ch: ChannelParams, fEC: number = qkdCfg().error_correction_eff, alsoAtDb: readonly number[] = []): {
  rows: QsdcCurveRow[]; linkLossDb: number; twoWayClosesToDb: number | null; lastPositiveDb: number;
} {
  const src = sourceOf(ch);
  const y0 = 2 * ch.darkCount;
  const opticsDb = receiverOpticsDb(ch);
  const detectorDb = -10 * Math.log10(ch.detectorEff);
  const rows: QsdcCurveRow[] = [];
  let twoWayClosesToDb: number | null = null;
  let lastPositiveDb = opticsDb + detectorDb;
  const start = opticsDb + detectorDb;
  const extra = alsoAtDb.filter((x) => x >= start && x <= MAX_PLOT_LOSS_DB);
  const grid: number[] = [];
  for (let x = start; x <= MAX_PLOT_LOSS_DB; x += 0.5) if (!extra.some((e) => Math.abs(x - e) < 0.25)) grid.push(x);
  for (const lossDb of [...grid, ...extra].sort((a, b) => a - b)) {
    const eta = 10 ** (-lossDb / 10);
    const r = decoySecureFraction(eta, src.mu, src.nu, eDetForBound(ch), y0, fEC, SIFT, src.vacuum);
    const gain = y0 + 1 - Math.exp(-eta * src.mu);
    const reliable = SIFT * gain * (1 - fEC * h2(r.eMu));
    const g = 10 ** ((lossDb - opticsDb) / 10);
    const twoWay = gain * (1 - h2(r.eMu) - g * h2(Math.min(0.5, 2 * r.eMu)));
    if (twoWay > 0) twoWayClosesToDb = lossDb;
    if (r.perPulse > 0 || twoWay > 0) lastPositiveDb = lossDb;
    rows.push({ lossDb, oneWaySecrecy: r.perPulse > 0 ? r.perPulse : null, oneWayReliable: reliable > 0 ? reliable : null, twoWay: twoWay > 0 ? twoWay : null });
  }
  return { rows: rows.filter((r) => r.lossDb <= lastPositiveDb + 5 || extra.includes(r.lossDb)), linkLossDb: channelLossDb(ch) + detectorDb, twoWayClosesToDb, lastPositiveDb };
}
