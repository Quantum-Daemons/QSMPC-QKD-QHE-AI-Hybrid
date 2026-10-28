import { lazy, Suspense } from "react";

/**
 * The 3D views behind React.lazy, so three.js (the largest dependency in the bundle) is fetched only on
 * a page that draws one - never on Overview, Value, Validations or the index.
 */
const QcnnScene = lazy(() => import("./Scene3D").then((m) => ({ default: m.QcnnScene })));
const QotpScene = lazy(() => import("./Scene3D").then((m) => ({ default: m.QotpScene })));
const SatScene = lazy(() => import("./Scene3D").then((m) => ({ default: m.SatScene })));
const SatNetworkScene = lazy(() => import("./Scene3D").then((m) => ({ default: m.SatNetworkScene })));

const loading = <p className="muted small">Loading the 3D view…</p>;

export function Qcnn3D(props: { features: number[]; seed: number }) {
  return <Suspense fallback={loading}><QcnnScene {...props} /></Suspense>;
}
export function Qotp3D(props: { x: number }) {
  return <Suspense fallback={loading}><QotpScene {...props} /></Suspense>;
}
export function Sat3D(props: { slantKm: number; orbitKm: number; minElevationDeg: number; stationKm?: number }) {
  return <Suspense fallback={loading}><SatScene {...props} /></Suspense>;
}
export function SatNetwork3D(props: { orbitKm: number; n: number; clearanceKm: number; minElevationDeg: number;
                                      shadowHalfAngleDeg: number }) {
  return <Suspense fallback={loading}><SatNetworkScene {...props} /></Suspense>;
}
