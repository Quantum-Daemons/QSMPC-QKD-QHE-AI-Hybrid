import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import { blochVectors, EARTH_RADIUS_KM, elevationDeg, qotpQcnnInfer } from "../lib/quantumSim";
import { centralAngleRad } from "../lib/satNetwork";

/**
 * The 3D views, loaded only on the pages that draw them (see `Lazy3D.tsx`) - three.js is the largest
 * dependency here and most pages never need it.
 *
 * EACH VIEW SAYS WHAT ITS DEPTH CARRIES. The comparison-circuit view (`QcnnScene`, an untrained QOTP +
 * Clifford+T circuit, not the trained QCNN) plots the reduced Bloch vectors the circuit actually
 * produces: T gates move them off the x-z plane and entanglement shortens them, which a 2D plot
 * of the encoding plane cannot show. The one-time-pad sphere is an illustration: the pad's four states
 * lie on one circle. The satellite views draw the model's own geometry since Round 77 - the orbit, the
 * elevation a range implies, the minimum-elevation arc, the constellation's spacing and shadow - where
 * the downlink view used to be an illustration of a model that had no elevation at all.
 */

function cssColor(name: string, fallback: string): THREE.Color {
  const v = typeof document !== "undefined"
    ? getComputedStyle(document.documentElement).getPropertyValue(name).trim() : "";
  try { return new THREE.Color(v || fallback); } catch { return new THREE.Color(fallback); }
}

function webglAvailable(): boolean {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch { return false; }
}

/** Where the camera starts and what it orbits. The default frames a unit sphere from a corner; the
 *  satellite views pass their own, because the part of their geometry that matters is a few percent of
 *  the Earth's radius above it (Round 77, seen on the deployed page). */
type View = { position: [number, number, number]; target: [number, number, number]; fov?: number };

/** A canvas with a renderer, a camera and orbit controls; `build` adds the objects and may return a
 *  per-frame update. Renders on demand (controls change, resize) rather than in a loop. */
function useScene(build: (scene: THREE.Scene) => void, deps: unknown[], view?: View) {
  const mount = useRef<HTMLDivElement>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    const el = mount.current;
    if (!el) return;
    if (!webglAvailable()) { setFailed(true); return; }
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    const width = el.clientWidth || 480, height = 300;
    renderer.setSize(width, height);
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, width / height, 0.01, 100);
    camera.position.set(...(view?.position ?? [2.6, 1.8, 2.6]));
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = false;
    if (view) {
      if (view.fov) { camera.fov = view.fov; camera.updateProjectionMatrix(); }
      controls.target.set(...view.target); controls.update();
    }
    scene.add(new THREE.AmbientLight(0xffffff, 0.9));
    const dir = new THREE.DirectionalLight(0xffffff, 0.8); dir.position.set(3, 4, 2); scene.add(dir);
    build(scene);
    const render = () => renderer.render(scene, camera);
    controls.addEventListener("change", render);
    // A ResizeObserver rather than window "resize": a scene inside a closed <details> mounts at zero
    // width, and only an observer on the element sees it become visible when the fold opens.
    const onResize = () => {
      const w = el.clientWidth;
      if (!w) return;
      renderer.setSize(w, height); camera.aspect = w / height; camera.updateProjectionMatrix(); render();
    };
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(onResize) : null;
    ro?.observe(el);
    render();
    return () => {
      ro?.disconnect();
      controls.dispose();
      scene.traverse((o) => {
        const m = o as THREE.Mesh;
        m.geometry?.dispose?.();
        const mat = m.material as THREE.Material | THREE.Material[] | undefined;
        if (Array.isArray(mat)) mat.forEach((x) => x.dispose()); else mat?.dispose?.();
      });
      renderer.dispose();
      el.removeChild(renderer.domElement);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return { mount, failed };
}

function addBlochSphere(scene: THREE.Scene) {
  const grid = cssColor("--border", "#243049");
  const sphere = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 16),
    new THREE.MeshBasicMaterial({ color: grid, wireframe: true, transparent: true, opacity: 0.35 }));
  scene.add(sphere);
  // Axes: x (red), y (green), z (blue) - three.js puts +y up, so the Bloch z axis is drawn on +y.
  const axis = (to: THREE.Vector3, color: number) => {
    const g = new THREE.BufferGeometry().setFromPoints([to.clone().multiplyScalar(-1.2), to.clone().multiplyScalar(1.2)]);
    scene.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color })));
  };
  axis(new THREE.Vector3(1, 0, 0), 0xf56565);   // Bloch x
  axis(new THREE.Vector3(0, 0, -1), 0x36d399);  // Bloch y
  axis(new THREE.Vector3(0, 1, 0), 0x4f9dff);   // Bloch z
}

/** Bloch (x, y, z) -> three.js scene coordinates (z up). */
const toScene = ([x, y, z]: [number, number, number]) => new THREE.Vector3(x, z, -y);

function Fallback({ what, instead = "the table below carries the same numbers" }: { what: string; instead?: string }) {
  return <p className="muted small">This browser has no WebGL, so the 3D view of {what} is not drawn; {instead}.</p>;
}

const QUBIT_COLORS = [0xb072ff, 0x4f9dff, 0x36d399, 0xf6b73c, 0xf56565, 0xe6edf6, 0x97a3b6, 0xff7ad9];

/** The comparison circuit's per-qubit reduced Bloch vectors through the real circuit - the 3D view with
 *  content. Named for the QCNN in code; it is the untrained QOTP + Clifford+T circuit, not a QCNN. */
export function QcnnScene({ features, seed }: { features: number[]; seed: number }) {
  const snaps = useMemo(() => {
    const out: { label: string; v: [number, number, number][] }[] = [];
    qotpQcnnInfer(features, seed, 8, (label, re, im) => out.push({ label, v: blochVectors(re, im, 8) }));
    return out;
  }, [features, seed]);
  const [stage, setStage] = useState(snaps.length - 1);
  const current = snaps[Math.min(stage, snaps.length - 1)];
  const { mount, failed } = useScene((scene) => {
    addBlochSphere(scene);
    current.v.forEach((vec, q) => {
      const v = toScene(vec);
      const len = v.length();
      if (len < 1e-6) return;
      scene.add(new THREE.ArrowHelper(v.clone().normalize(), new THREE.Vector3(0, 0, 0), len, QUBIT_COLORS[q], 0.08, 0.05));
    });
  }, [current]);
  return (
    <figure className="chart-figure" style={{ margin: 0 }}>
      <figcaption className="small muted">
        <b>Inside the QSMPC-FHE comparison circuit (an untrained QOTP + Clifford+T circuit, state-vector
        simulation, 8 qubits): one arrow per qubit.</b> After encoding and the pad every arrow has length 1
        and lies in the x-z plane; through the Clifford+T rounds the T gates move them off that plane (a y
        component appears) and entanglement shortens them. This is the circuit the QSMPC-FHE row runs, drawn
        at 8 qubits on one illustrative input; its pad is seeded and never undone, so it predicts nothing. It
        is not the trained QCNN that can decide a page&apos;s class, which runs unencrypted and is not drawn here.
      </figcaption>
      <div ref={mount} role="img" aria-label={`Reduced Bloch vectors of 8 qubits, ${current.label}`} style={{ width: "100%", minHeight: failed ? 0 : 300 }} />
      {failed && <Fallback what="the comparison circuit's state" />}
      <label className="small muted" htmlFor="qcnn-stage">Stage: <b>{current.label}</b></label>
      <input id="qcnn-stage" type="range" min={0} max={snaps.length - 1} step={1} value={stage}
        onChange={(e) => setStage(Number(e.target.value))} style={{ width: "100%" }} aria-label="Circuit stage" />
      {/* Its own scroller, with narrower cells (Round 80), so nine columns never widen the card. The
          figure measured 4 px too wide at 1280 px; that was the slider above (see styles.css). */}
      <div style={{ overflowX: "auto", maxWidth: "100%" }}>
      <table className="facts-table small" style={{ ["--cell-pad-x" as string]: "5px" }}>
        <thead><tr><th>stage</th>{current.v.map((_, q) => <th key={q}>|r| q{q}</th>)}</tr></thead>
        <tbody>
          {snaps.map((s) => (
            <tr key={s.label} style={s.label === current.label ? { color: "var(--text)" } : undefined}>
              <td>{s.label}</td>
              {s.v.map(([x, y, z], q) => <td key={q} className="mono">{Math.hypot(x, y, z).toFixed(2)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
      </div>
    </figure>
  );
}

/** The quantum one-time pad on one qubit - an ILLUSTRATION, labelled as one. */
export function QotpScene({ x }: { x: number }) {
  const th = x * Math.PI;
  const pts: { v: [number, number, number]; label: string }[] = [
    { v: [Math.sin(th), 0, Math.cos(th)], label: "key 00 (I)" },
    { v: [Math.sin(th), 0, -Math.cos(th)], label: "key 10 (X)" },
    { v: [-Math.sin(th), 0, Math.cos(th)], label: "key 01 (Z)" },
    { v: [-Math.sin(th), 0, -Math.cos(th)], label: "key 11 (XZ)" },
  ];
  const { mount, failed } = useScene((scene) => {
    addBlochSphere(scene);
    pts.forEach((p, i) => {
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.06, 16, 8), new THREE.MeshStandardMaterial({ color: QUBIT_COLORS[i] }));
      m.position.copy(toScene(p.v)); scene.add(m);
    });
    const c = new THREE.Mesh(new THREE.SphereGeometry(0.07, 16, 8), new THREE.MeshStandardMaterial({ color: 0xffffff }));
    scene.add(c);   // the average over a uniform key: the centre, i.e. the maximally mixed state
  }, [x]);
  return (
    <figure className="chart-figure" style={{ margin: 0 }}>
      <figcaption className="small muted">
        <b>The quantum one-time pad on one qubit (illustration).</b> The four keys map the encoded state to four
        points; averaged over a uniform key they give the centre - the maximally mixed state, which is why the
        pad hides the input <i>as a scheme</i>. The four points lie on one circle, so this 3D view adds no
        information beyond a 2D drawing, and this demo's pad is seeded, so no run here is hidden that way.
      </figcaption>
      <div ref={mount} role="img" aria-label="Four one-time-pad keys on a Bloch sphere and their average at the centre" style={{ width: "100%", minHeight: failed ? 0 : 300 }} />
      {failed && <Fallback what="the one-time pad" />}
      <p className="small muted">{pts.map((p) => `${p.label}: (${p.v.map((c) => c.toFixed(2)).join(", ")})`).join(" · ")} · average: (0, 0, 0)</p>
    </figure>
  );
}

/** An arc as a thin tube, for the arcs a reader has to find: WebGL draws every line one pixel wide. */
function tubeArc(radius: number, a0: number, a1: number, color: number, thickness: number): THREE.Mesh {
  const pts: THREE.Vector3[] = [];
  for (let i = 0; i <= 64; i++) {
    const a = a0 + ((a1 - a0) * i) / 64;
    pts.push(new THREE.Vector3(radius * Math.sin(a), radius * Math.cos(a), 0));
  }
  return new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 96, thickness, 6, false),
    new THREE.MeshBasicMaterial({ color }));
}

/** A circle of `radius` in the x-y plane (the drawing's orbit plane), from angle a0 to a1 (radians,
 *  measured from +y, the station's zenith). */
function arc(radius: number, a0: number, a1: number, color: number, width = 1, segments = 96): THREE.Line {
  const pts: THREE.Vector3[] = [];
  for (let i = 0; i <= segments; i++) {
    const a = a0 + ((a1 - a0) * i) / segments;
    pts.push(new THREE.Vector3(radius * Math.sin(a), radius * Math.cos(a), 0));
  }
  return new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color, linewidth: width }));
}

/**
 * The satellite downlink geometry - the model's own since Round 77. The slant range and the orbit's
 * altitude fix the elevation, the air mass the atmosphere scales by, and whether there is a link at
 * all, so the drawing is the geometry the run computed with rather than an illustration of a range.
 */
export function SatScene({ slantKm, orbitKm, minElevationDeg, stationKm = 0 }: {
  slantKm: number; orbitKm: number; minElevationDeg: number; stationKm?: number;
}) {
  // The model's own geometry (`elevationDeg`, `centralAngleRad`) for the run's station, so the caption
  // reads the elevation the run reports. The station's few hundred metres are not drawn.
  const R = EARTH_RADIUS_KM;
  const b = R + orbitKm;
  const elOrNull = elevationDeg(slantKm, orbitKm, stationKm);
  const visible = elOrNull !== null;
  const elDeg = elOrNull ?? 0;
  const inView = visible && elDeg >= minElevationDeg - 1e-9;
  // Central angles (from the station's zenith) of the satellite and of the minimum-elevation edge.
  const psiSat = visible ? centralAngleRad(elDeg, orbitKm, stationKm) : Math.acos(R / b) + 0.15;
  const psiMin = centralAngleRad(minElevationDeg, orbitKm, stationKm);
  const { mount, failed } = useScene((scene) => {
    const s = 1 / R;                                         // scene units: Earth radius = 1
    const earth = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24),
      new THREE.MeshStandardMaterial({ color: 0x1d4ed8, roughness: 0.9, transparent: true, opacity: 0.85 }));
    scene.add(earth);
    const r = b * s;
    scene.add(arc(r, -Math.PI, Math.PI, 0x5b6b8c));                 // the orbit
    scene.add(tubeArc(r, -psiMin, psiMin, 0x36d399, 0.0035));        // where the station sees it above the minimum
    const gs = new THREE.Vector3(0, 1, 0);                           // ground station at the drawing's "north pole"
    const sat = new THREE.Vector3(r * Math.sin(psiSat), r * Math.cos(psiSat), 0);
    const dot = (p: THREE.Vector3, color: number, rad: number) => {
      const m = new THREE.Mesh(new THREE.SphereGeometry(rad, 16, 8), new THREE.MeshStandardMaterial({ color }));
      m.position.copy(p); scene.add(m);
    };
    dot(gs, 0x36d399, 0.012); dot(sat, inView ? 0xf6b73c : 0xf56565, 0.015);
    if (visible) {
      scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([gs, sat]),
        new THREE.LineBasicMaterial({ color: 0xb072ff })));
    }
  }, [slantKm, orbitKm, minElevationDeg, stationKm],
  // Face-on to the orbit plane, over the station: the orbit is 8 percent of the Earth's radius up, so
  // a whole-Earth view left the pass a sliver at the top of the sphere.
  { position: [0.12, 1.08, 0.78], target: [0, 1.03, 0] });
  return (
    <figure className="chart-figure" style={{ margin: 0 }}>
      <figcaption className="small muted">
        <b>Downlink geometry.</b> The ground station (green), the {orbitKm} km orbit (grey), the arc of it the station
        sees above {minElevationDeg} degrees (green), and the satellite at this run&apos;s slant range of
        {" "}{visible
          ? <>{slantKm} km, at {elDeg.toFixed(1)} degrees elevation{inView ? "" : <> - below the minimum, so there is no link (red)</>}.</>
          : <>{slantKm} km: past the horizon, so there is no line of sight (red, drawn beyond the edge of view).</>}
        {" "}The model uses this geometry: the elevation sets the air mass the atmosphere scales by.
      </figcaption>
      <div ref={mount} role="img" aria-label={`Ground station and a satellite at ${slantKm} km slant range`} style={{ width: "100%", minHeight: failed ? 0 : 300 }} />
      {failed && <Fallback what="the downlink" instead="the caption above states the same geometry" />}
    </figure>
  );
}

/**
 * The Network tab's geometry (Round 77): one orbital plane of `n` satellites linked to their
 * neighbours, the shell the links must clear, one station's visible arc and Earth's shadow. Radii,
 * spacing, clearance, visible arc and shadow arc are the model's; the station's place on the plane
 * and the Sun's direction are chosen for the drawing.
 */
export function SatNetworkScene({ orbitKm, n, clearanceKm, minElevationDeg, shadowHalfAngleDeg }: {
  orbitKm: number; n: number; clearanceKm: number; minElevationDeg: number; shadowHalfAngleDeg: number;
}) {
  const R = EARTH_RADIUS_KM;
  const b = R + orbitKm;
  const chordLow = b * Math.cos(Math.PI / n) - R;
  const losOk = chordLow >= clearanceKm;
  const psiMin = centralAngleRad(minElevationDeg, orbitKm);
  const alpha = (shadowHalfAngleDeg * Math.PI) / 180;
  const { mount, failed } = useScene((scene) => {
    const s = 1 / R;
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24),
      new THREE.MeshStandardMaterial({ color: 0x1d4ed8, roughness: 0.9, transparent: true, opacity: 0.85 })));
    // The shell a link's line of sight must stay above.
    scene.add(new THREE.Mesh(new THREE.SphereGeometry((R + clearanceKm) * s, 32, 16),
      new THREE.MeshBasicMaterial({ color: 0x97a3b6, wireframe: true, transparent: true, opacity: 0.12 })));
    const r = b * s;
    scene.add(arc(r, -Math.PI, Math.PI, 0x5b6b8c));
    // Earth's shadow on the plane, centred opposite the Sun (drawn to the -y side, the station's
    // night), as a band just outside the orbit.
    if (alpha > 0) scene.add(tubeArc(r * 1.035, Math.PI - alpha, Math.PI + alpha, 0x7c8aa8, 0.008));
    // A station on the night side and the arc of the orbit it sees above the minimum elevation.
    const gs = new THREE.Vector3(0, -1, 0);
    scene.add(tubeArc(r, Math.PI - psiMin, Math.PI + psiMin, 0x36d399, 0.006));
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.025, 16, 8), new THREE.MeshStandardMaterial({ color: 0x36d399 }));
    m.position.copy(gs); scene.add(m);
    const sats: THREE.Vector3[] = [];
    for (let i = 0; i < n; i++) {
      const a = Math.PI + (2 * Math.PI * i) / n;
      sats.push(new THREE.Vector3(r * Math.sin(a), r * Math.cos(a), 0));
    }
    sats.forEach((p, i) => {
      const dot = new THREE.Mesh(new THREE.SphereGeometry(0.022, 12, 6), new THREE.MeshStandardMaterial({ color: 0xf6b73c }));
      dot.position.copy(p); scene.add(dot);
      const q = sats[(i + 1) % n];
      scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([p, q]),
        new THREE.LineBasicMaterial({ color: losOk ? 0xb072ff : 0xf56565 })));
    });
  }, [orbitKm, n, clearanceKm, minElevationDeg, shadowHalfAngleDeg],
  // Face-on to the plane, so every satellite, link, the shadow and the station's arc are in view; from
  // the default corner the ring was edge-on and half of it behind the Earth.
  // A long lens from farther out, near-orthographic: at 500 km the ring is 8 percent outside the
  // Earth, and a close perspective camera hid its far side behind the limb.
  { position: [0.3, 0.2, 8.5], target: [0, 0, 0], fov: 17 });
  return (
    <figure className="chart-figure" style={{ margin: 0 }}>
      <figcaption className="small muted">
        <b>One plane of {n} satellites at {orbitKm} km.</b> Neighbour links pass {Math.round(chordLow)} km above the
        ground at their lowest, {losOk ? <>clear of the {clearanceKm} km shell (purple)</> : <>below the {clearanceKm} km shell, so they are blocked (red)</>};
        {" "}the arc a station sees above {minElevationDeg} degrees (green); and Earth&apos;s shadow on the orbit (grey band),
        where a link that needs darkness can run. The station&apos;s place and the Sun&apos;s direction are chosen for the
        drawing; the radii, spacing, clearance and arcs are the model&apos;s.
      </figcaption>
      <div ref={mount} role="img" aria-label={`A plane of ${n} satellites at ${orbitKm} km with inter-satellite links`} style={{ width: "100%", minHeight: failed ? 0 : 300 }} />
      {failed && <Fallback what="the network" instead="the caption and the tables above carry the same numbers" />}
    </figure>
  );
}
