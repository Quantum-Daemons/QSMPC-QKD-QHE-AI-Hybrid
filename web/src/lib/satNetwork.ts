// THE SATELLITE NETWORK BEYOND ONE STATION'S VIEW (Round 77, the user's request).
//
// A downlink has no link past the horizon - 2,571 km of slant range at a 500 km orbit - and none
// below the 10 degree minimum elevation, 1,692 km. Two published answers bridge it, and this module
// models both with the same channel and bound a Downlink run uses (`bb84Rate`), integrated over the
// geometry rather than evaluated at one range:
//
//   - A TRUSTED SATELLITE RELAY (Liao et al., Phys. Rev. Lett. 120, 030501 (2018), arXiv:1801.04418):
//     the satellite makes a key with each station on its own pass, "performs bitwise exclusive OR
//     operations between the two keys and relays the result to one of the ground stations" - 7,600 km,
//     Xinglong to Graz.
//   - A CONSTELLATION WITH INTER-SATELLITE LINKS (Vergoossen et al., arXiv:1903.07845 (2019)): a
//     "string-of-pearls configuration of satellites in the same plane", keys re-distributed across
//     it by quantum links between neighbours, every satellite a trusted node.
//
// And under both, the PASS: one satellite crossing one station's sky, checked against Liao et al.
// 2017's own pass of 19 December 2016 (tests/test_satellite_network_is_checked.py).
//
// What it leaves out, and says so on the page: Earth's rotation during a pass (about 0.35 km/s at
// Xinglong against the satellite's 7.6), weather, the Sun's beta angle beyond the one it is given,
// and finite-key effects - the key is the asymptotic bound.
//
// ONE KEY CONVENTION, per pulse sent (Round 78, the user's choice). `bb84Rate` counts the key the
// signal pulses carry - half of Micius's, which sent "probabilities of 50%, 25%, and 25%" - so a pass's
// key is comparable with a measured one. In Round 77 this module carried a second, signal-share figure
// beside the Downlink tab's every-pulse rate; the rate itself changed instead.

import { linkProfiles, profileValues, qkdCfg } from "./config";
import { bb84Rate, clickMix, EARTH_RADIUS_KM, elevationDeg, type ChannelParams } from "./quantumSim";

/** Earth's gravitational parameter in km^3/s^2 (WGS 84). */
export const GM_EARTH_KM3_S2 = 398600.4418;
const DEG = Math.PI / 180;

/** The period of a circular orbit. 94.5 minutes at 500 km; Liao et al. 2018: Micius "circles Earth
 *  every 94 minutes". */
export function orbitalPeriodS(altitudeKm: number): number {
  return 2 * Math.PI * Math.sqrt((EARTH_RADIUS_KM + altitudeKm) ** 3 / GM_EARTH_KM3_S2);
}

/** The Earth central angle between a station and the sub-satellite point when the satellite is at
 *  `elevation` degrees: acos((R+g)/(R+h) cos e) - e. */
export function centralAngleRad(elevation: number, orbitKm: number, stationKm = 0): number {
  const a = EARTH_RADIUS_KM + stationKm, b = EARTH_RADIUS_KM + orbitKm, e = elevation * DEG;
  return Math.acos((a / b) * Math.cos(e)) - e;
}

/** The slant range at an elevation: sqrt(a^2 sin^2 e + b^2 - a^2) - a sin e. 1,692 km at 10 degrees
 *  for 500 km over Xinglong; 2,571 km at the horizon. The inverse of `elevationDeg`. */
export function rangeAtElevationKm(elevation: number, orbitKm: number, stationKm = 0): number {
  const a = EARTH_RADIUS_KM + stationKm, b = EARTH_RADIUS_KM + orbitKm, s = Math.sin(elevation * DEG);
  return Math.sqrt(a * a * s * s + b * b - a * a) - a * s;
}

const geometry = (link: ChannelParams) => {
  const fs = link.freeSpace!;
  const q = qkdCfg();
  return {
    orbitKm: fs.orbitAltitudeKm ?? q.free_space_orbit_altitude_km,
    stationKm: fs.stationAltitudeKm ?? q.free_space_station_altitude_km,
    minElevation: fs.minElevationDeg ?? q.free_space_min_elevation_deg,
    rateHz: Number(link.sourceRateHz ?? q.free_space_source_rate_hz),
  };
};

export interface PassSample {
  t: number; rangeKm: number; elevationDeg: number;
  /** bit/s: the asymptotic bound per pulse sent times the clock (the Downlink tab's rate at this range). */
  keyRate: number;
  /** bit/s of SIFTED key counted as Liao counted it: matching bases on SIGNAL states only. */
  siftedRate: number;
  /** detections per second over every state sent: signal, decoy and vacuum. */
  detectionRate: number;
}

export interface PassResult {
  maxElevationDeg: number;
  /** The elevation the window opens and closes at: the link's minimum unless a window is given. */
  windowElevationDeg: number;
  durationS: number; minRangeKm: number; maxRangeKm: number;
  keyBits: number; siftedBits: number; detections: number;
  samples: PassSample[];
}

/**
 * One pass of a circular orbit over one station, culminating at `maxElevation` degrees, integrated
 * (trapezoid rule) while the satellite is above `windowElevation` (default: the link's minimum).
 *
 * The geometry is the spherical right triangle of the ground track: the station sits a central angle
 * psi0 off it, and t seconds from culmination the satellite is psi(t) from the station with
 * cos psi = cos psi0 cos(omega t). The range follows from the triangle of radii, the elevation from
 * the range, and every sample runs the Downlink tab's own `bb84Rate` at that range.
 */
export function simulatePass(link: ChannelParams, maxElevation: number,
                             opts: { windowElevationDeg?: number; stepS?: number } = {}): PassResult {
  const g = geometry(link);
  const window = Math.max(opts.windowElevationDeg ?? g.minElevation, g.minElevation);
  const empty: PassResult = { maxElevationDeg: maxElevation, windowElevationDeg: window, durationS: 0,
    minRangeKm: NaN, maxRangeKm: NaN, keyBits: 0, siftedBits: 0, detections: 0, samples: [] };
  if (!(maxElevation >= window) || maxElevation > 90) return empty;
  const a = EARTH_RADIUS_KM + g.stationKm, b = EARTH_RADIUS_KM + g.orbitKm;
  const w = (2 * Math.PI) / orbitalPeriodS(g.orbitKm);
  const psi0 = centralAngleRad(maxElevation, g.orbitKm, g.stationKm);
  const psiW = centralAngleRad(window, g.orbitKm, g.stationKm);
  const half = Math.acos(Math.min(1, Math.cos(psiW) / Math.cos(psi0))) / w;
  const n = Math.max(2, Math.ceil((2 * half) / (opts.stepS ?? 1)));
  const h = (2 * half) / n;
  const samples: PassSample[] = [];
  let keyBits = 0, siftedBits = 0, detections = 0;
  for (let i = 0; i <= n; i++) {
    const t = -half + i * h;
    const r = Math.sqrt(a * a + b * b - 2 * a * b * Math.cos(psi0) * Math.cos(w * t));
    const el = elevationDeg(r, g.orbitKm, g.stationKm) ?? 0;
    // Inside the window by construction; its two ends sit on the window elevation to rounding, and
    // the window is never below the link's minimum, so the cut is lifted here rather than letting a
    // last-digit difference zero an end sample.
    const ch = { ...link, distanceKm: r, freeSpace: { ...link.freeSpace!, minElevationDeg: 0 } };
    const rate = bb84Rate(ch);
    const sample: PassSample = {
      t, rangeKm: r, elevationDeg: el,
      keyRate: rate.perPulse * g.rateHz,
      siftedRate: 0.5 * rate.src.pSignal * rate.qMu * rate.cDt * g.rateHz,
      detectionRate: clickMix(rate.src, rate.qMu, rate.qNu, rate.y0) * rate.cDt * g.rateHz,
    };
    samples.push(sample);
    const wt = i === 0 || i === n ? 0.5 * h : h;
    keyBits += wt * sample.keyRate;
    siftedBits += wt * sample.siftedRate;
    detections += wt * sample.detectionRate;
  }
  return { maxElevationDeg: maxElevation, windowElevationDeg: window, durationS: 2 * half,
    minRangeKm: rangeAtElevationKm(maxElevation, g.orbitKm, g.stationKm),
    maxRangeKm: rangeAtElevationKm(window, g.orbitKm, g.stationKm),
    keyBits, siftedBits, detections, samples };
}

/** Liao et al. 2017's pass of 19 December 2016, as the paper states it: "the minimal (maximal)
 *  separation of 645 km (1200 km). Within a duration of 273 s for the QKD data collection, the ground
 *  station collected 3,551,136 detection events, and thereof 1,671,072 bits of sifted keys", and a
 *  "secure final key of 300,939 bits" (finite-key, failure probability 1e-9). */
export const LIAO_2017_PASS = {
  minRangeKm: 645, maxRangeKm: 1200, durationS: 273, detections: 3_551_136, siftedBits: 1_671_072,
  finalKeyBits: 300_939,
};

/** That pass through this model: culminating at the elevation 645 km implies, counted while the
 *  range is under 1200 km. */
export function liaoPass(link: ChannelParams): PassResult {
  const g = geometry(link);
  const top = elevationDeg(LIAO_2017_PASS.minRangeKm, g.orbitKm, g.stationKm)!;
  const edge = elevationDeg(LIAO_2017_PASS.maxRangeKm, g.orbitKm, g.stationKm)!;
  return simulatePass(link, top, { windowElevationDeg: edge, stepS: 0.25 });
}

// --------------------------------------------------------------------------------------------- //
//  The trusted relay
// --------------------------------------------------------------------------------------------- //

export interface Station { id: string; name: string; latDeg: number; lonDeg: number; altitudeKm: number }

/** Liao et al. 2018: "Xinglong (near Beijing, 40 deg 23' 45.12'' N, 117 deg 34' 38.85'' E, altitude of
 *  890 m), Nanshan (near Urumqi, 43 deg 28' 31.66'' N, 87 deg 10' 36.07'' E, altitude of 2028 m), and
 *  Graz (47 deg 4' 1.72'' N, 15 deg 29' 35.92'' E, altitude of 490m)". */
export const LIAO_2018_STATIONS: Station[] = [
  { id: "xinglong", name: "Xinglong", latDeg: 40 + 23 / 60 + 45.12 / 3600, lonDeg: 117 + 34 / 60 + 38.85 / 3600, altitudeKm: 0.89 },
  { id: "nanshan", name: "Nanshan", latDeg: 43 + 28 / 60 + 31.66 / 3600, lonDeg: 87 + 10 / 60 + 36.07 / 3600, altitudeKm: 2.028 },
  { id: "graz", name: "Graz", latDeg: 47 + 4 / 60 + 1.72 / 3600, lonDeg: 15 + 29 / 60 + 35.92 / 3600, altitudeKm: 0.49 },
];

/** Great-circle distance on the mean sphere (haversine). Liao et al. 2018: "The distances from
 *  Xinglong to Nanshan and Graz are 2500 km and 7600 km". */
export function groundDistanceKm(p: Station, q: Station): number {
  const f1 = p.latDeg * DEG, f2 = q.latDeg * DEG, dl = (q.lonDeg - p.lonDeg) * DEG;
  const x = Math.sin((f2 - f1) / 2) ** 2 + Math.cos(f1) * Math.cos(f2) * Math.sin(dl / 2) ** 2;
  return 2 * EARTH_RADIUS_KM * Math.asin(Math.min(1, Math.sqrt(x)));
}

/** The widest ground separation one satellite can see at once, both stations at or above the minimum
 *  elevation: twice the ground radius of its footprint. 3,122 km at 500 km and 10 degrees. */
export function sharedViewMaxKm(link: ChannelParams): number {
  const g = geometry(link);
  return 2 * EARTH_RADIUS_KM * centralAngleRad(g.minElevation, g.orbitKm, g.stationKm);
}

export interface RelayResult {
  a: PassResult; b: PassResult;
  /** min(K_A, K_B): the XOR pads one key with the other, so the shorter one bounds it. */
  keyBits: number;
  groundKm: number;
  /** Whether one satellite could see both stations at once - if so, no store-and-forward is needed. */
  sharedView: boolean;
  /** Hours between the two passes: a sun-synchronous orbit crosses each station at the same LOCAL
   *  time ("Each night starting at around 0:50 AM local time", Liao 2018), so the second key exists
   *  the longitude difference / 15 deg per hour later, held on board until then. */
  waitH: number;
  /** Pass pairs to accumulate the 100 kB Liao et al. 2018 relayed ("we establish a 100 kB secure key
   *  between Xinglong and Graz"), at this model's asymptotic key. */
  passPairsFor100kB: number;
}

const atStation = (link: ChannelParams, s: Station): ChannelParams =>
  ({ ...link, freeSpace: { ...link.freeSpace!, stationAltitudeKm: s.altitudeKm } });

export function trustedRelay(link: ChannelParams, a: Station, b: Station, maxElevA: number,
                             maxElevB: number): RelayResult {
  const pa = simulatePass(atStation(link, a), maxElevA);
  const pb = simulatePass(atStation(link, b), maxElevB);
  const keyBits = Math.min(pa.keyBits, pb.keyBits);
  const groundKm = groundDistanceKm(a, b);
  let dl = Math.abs(a.lonDeg - b.lonDeg) % 360;
  if (dl > 180) dl = 360 - dl;
  return { a: pa, b: pb, keyBits, groundKm, sharedView: groundKm <= sharedViewMaxKm(link),
    waitH: dl / 15, passPairsFor100kB: keyBits > 0 ? Math.ceil(800_000 / keyBits) : Infinity };
}

/** The relay's XOR, on real random bytes: the satellite broadcasts K_A xor K_B, and station A, holding
 *  K_A, recovers K_B. The broadcast alone is K_B one-time-padded with K_A, so it tells a listener
 *  nothing; the SATELLITE holds both keys, which is what "trusted" means. */
export function xorRelay(kA: Uint8Array, kB: Uint8Array): { broadcast: Uint8Array; recovered: Uint8Array } {
  const n = Math.min(kA.length, kB.length);
  const broadcast = new Uint8Array(n), recovered = new Uint8Array(n);
  for (let i = 0; i < n; i++) { broadcast[i] = kA[i] ^ kB[i]; recovered[i] = broadcast[i] ^ kA[i]; }
  return { broadcast, recovered };
}

// --------------------------------------------------------------------------------------------- //
//  The constellation: one plane, neighbours linked
// --------------------------------------------------------------------------------------------- //

export interface IslSpec {
  txDivergenceUrad: number; rxApertureM: number; receiverOpticsEff: number; pointingDb: number;
  detectorEff: number; darkCount: number; misalignment: number; sourceRateHz: number; deadTimeS: number;
  muSignal: number; muDecoy: number; pSignal: number; pDecoy: number; vacuumDecoy: boolean;
  clearanceKm: number; eclipseOnly: boolean;
}

/** An inter-satellite terminal pair from a profile under src/qsh/config/profiles/isl/. */
export function islFromProfile(id = String(qkdCfg().isl_profile)): IslSpec {
  const v = profileValues("isl", id) ?? profileValues("isl", Object.keys(linkProfiles("isl"))[0])!;
  return {
    txDivergenceUrad: Number(v.tx_divergence_urad), rxApertureM: Number(v.rx_aperture_m),
    receiverOpticsEff: Number(v.receiver_optics_efficiency), pointingDb: Number(v.pointing_db),
    detectorEff: Number(v.detector_efficiency), darkCount: Number(v.dark_count_prob),
    misalignment: Number(v.misalignment), sourceRateHz: Number(v.source_rate_hz),
    deadTimeS: Number(v.detector_dead_time_s), muSignal: Number(v.mu_signal), muDecoy: Number(v.mu_decoy),
    pSignal: Number(v.p_signal), pDecoy: Number(v.p_decoy), vacuumDecoy: Boolean(v.vacuum_decoy),
    clearanceKm: Number(v.clearance_km), eclipseOnly: Boolean(v.eclipse_only),
  };
}

/** The ISL as a channel the BB84 model runs: the downlink's diffraction law between two terminals,
 *  no atmosphere (the line of sight passes above `clearanceKm`). */
export function islChannel(spec: IslSpec, distanceKm: number): ChannelParams {
  return {
    distanceKm, lossDbPerKm: 0, detectorEff: spec.detectorEff, darkCount: spec.darkCount,
    misalignment: spec.misalignment, eveFraction: 0,
    freeSpace: { beamDivergenceUrad: spec.txDivergenceUrad, rxApertureM: spec.rxApertureM,
      atmosphericDb: 0, pointingDb: spec.pointingDb, receiverOpticsEff: spec.receiverOpticsEff,
      interSatellite: true },
    sourceRateHz: spec.sourceRateHz, deadTimeS: spec.deadTimeS,
    muSignal: spec.muSignal, muDecoy: spec.muDecoy, pSignal: spec.pSignal, pDecoy: spec.pDecoy,
    vacuumDecoy: spec.vacuumDecoy,
  };
}

/** Fewest satellites in one plane whose neighbours see each other above `clearanceKm`: the chord
 *  between neighbours dips to (R+h) cos(pi/N) - R, so N >= pi / acos((R + c) / (R + h)). */
export function minSatellitesForLineOfSight(altitudeKm: number, clearanceKm: number): number {
  const ratio = (EARTH_RADIUS_KM + clearanceKm) / (EARTH_RADIUS_KM + altitudeKm);
  if (ratio >= 1) return Infinity;
  return Math.ceil(Math.PI / Math.acos(ratio) - 1e-12);
}

/** The fraction of an orbit a satellite spends in Earth's (cylindrical) shadow at Sun beta angle
 *  `betaDeg`: its in-plane half-angle cos(alpha) = sqrt(1 - (R/(R+h))^2) / cos(beta), none when that
 *  exceeds 1. 0.378 at 500 km and beta = 0, 35.7 minutes of 94.5. A link between neighbours 2 pi / N
 *  apart needs both in shadow, which leaves 2 alpha - 2 pi / N of the orbit. */
export function eclipseHalfAngleRad(altitudeKm: number, betaDeg = 0): number {
  const c = Math.sqrt(1 - (EARTH_RADIUS_KM / (EARTH_RADIUS_KM + altitudeKm)) ** 2) / Math.cos(betaDeg * DEG);
  return c >= 1 ? 0 : Math.acos(c);
}

export interface ConstellationResult {
  n: number; altitudeKm: number; periodS: number;
  neighbourKm: number;
  /** How far above the ground the line of sight between neighbours passes. */
  chordMinAltitudeKm: number;
  losOk: boolean;
  minSatellites: number;
  islLossDb: number;
  /** The bound per pulse sent, as the Downlink tab's. */
  islPerPulse: number;
  /** bit/s: `islPerPulse` times the clock. */
  islBitPerS: number;
  eclipseFraction: number; pairEclipseFraction: number;
  /** Seconds per orbit each neighbour link can run: in the pair's shared shadow, or always. */
  islSecondsPerOrbit: number;
  islKeyPerOrbitBits: number; islKeyPerDayBits: number;
  /** Hops from a satellite to the one opposite it: the trusted nodes a key crosses. */
  hopsAcross: number;
}

/**
 * A string-of-pearls plane of `n` equally spaced satellites at `altitudeKm`, each linked to its two
 * neighbours by the terminal pair `spec`. Every satellite is a trusted node: a key re-distributed k
 * hops is known to k + 1 satellites. The key a link makes per orbit is the BB84 bound over the ISL
 * channel times the clock and the time the link runs (eclipse only, by Vergoossen et al.'s
 * assumption, unless the spec says otherwise).
 */
export function stringOfPearls(n: number, altitudeKm: number, spec: IslSpec,
                               betaDeg = 0): ConstellationResult {
  const b = EARTH_RADIUS_KM + altitudeKm;
  const periodS = orbitalPeriodS(altitudeKm);
  const neighbourKm = 2 * b * Math.sin(Math.PI / n);
  const chordMinAltitudeKm = b * Math.cos(Math.PI / n) - EARTH_RADIUS_KM;
  const losOk = chordMinAltitudeKm >= spec.clearanceKm;
  const ch = islChannel(spec, neighbourKm);
  const rate = bb84Rate(ch);
  const alpha = eclipseHalfAngleRad(altitudeKm, betaDeg);
  const eclipseFraction = alpha / Math.PI;
  const pairEclipseFraction = Math.max(0, 2 * alpha - (2 * Math.PI) / n) / (2 * Math.PI);
  const islSecondsPerOrbit = losOk ? (spec.eclipseOnly ? pairEclipseFraction : 1) * periodS : 0;
  const islBitPerS = losOk ? rate.perPulse * spec.sourceRateHz : 0;
  const islKeyPerOrbitBits = islBitPerS * islSecondsPerOrbit;
  return {
    n, altitudeKm, periodS, neighbourKm, chordMinAltitudeKm, losOk,
    minSatellites: minSatellitesForLineOfSight(altitudeKm, spec.clearanceKm),
    islLossDb: losOk ? -10 * Math.log10(Math.max(rate.eta, 1e-300) / spec.detectorEff) : Infinity,
    islPerPulse: losOk ? rate.perPulse : 0, islBitPerS,
    eclipseFraction, pairEclipseFraction, islSecondsPerOrbit,
    islKeyPerOrbitBits, islKeyPerDayBits: islKeyPerOrbitBits * (86_400 / periodS),
    hopsAcross: Math.floor(n / 2),
  };
}
