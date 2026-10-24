import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import { blochVectors, qotpQcnnInfer } from "../lib/quantumSim";

/**
 * The 3D views, loaded only on the pages that draw them (see `Lazy3D.tsx`) - three.js is the largest
 * dependency here and most pages never need it.
 *
 * ONE OF THE THREE CARRIES INFORMATION AND TWO DO NOT, AND EACH SAYS WHICH. The QCNN view plots the
 * reduced Bloch vectors the circuit actually produces: T gates move them off the x-z plane and
 * entanglement shortens them, which a 2D plot of the encoding plane cannot show. The one-time-pad sphere
 * and the satellite geometry were added on request as illustrations: the pad's four states lie on one
 * circle, and the channel model uses slant range alone, so their third dimension carries nothing the
 * page does not already say. Their captions state that rather than letting depth imply precision.
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

/** A canvas with a renderer, a camera and orbit controls; `build` adds the objects and may return a
 *  per-frame update. Renders on demand (controls change, resize) rather than in a loop. */
function useScene(build: (scene: THREE.Scene) => void, deps: unknown[]) {
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
    camera.position.set(2.6, 1.8, 2.6);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = false;
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

function Fallback({ what }: { what: string }) {
  return <p className="muted small">This browser has no WebGL, so the 3D view of {what} is not drawn; the table below carries the same numbers.</p>;
}

const QUBIT_COLORS = [0xb072ff, 0x4f9dff, 0x36d399, 0xf6b73c, 0xf56565, 0xe6edf6, 0x97a3b6, 0xff7ad9];

/** The QCNN's per-qubit reduced Bloch vectors through the real circuit - the 3D view with content. */
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
        <b>Inside the QCNN (state-vector simulation, 8 qubits): one arrow per qubit.</b> After encoding and the
        pad every arrow has length 1 and lies in the x-z plane; through the Clifford+T rounds the T gates move
        them off that plane (a y component appears) and entanglement shortens them. This is the circuit the
        QSMPC-FHE row runs, on one illustrative input; its pad is seeded.
      </figcaption>
      <div ref={mount} role="img" aria-label={`Reduced Bloch vectors of 8 qubits, ${current.label}`} style={{ width: "100%", minHeight: 300 }} />
      {failed && <Fallback what="the QCNN state" />}
      <label className="small muted" htmlFor="qcnn-stage">Stage: <b>{current.label}</b></label>
      <input id="qcnn-stage" type="range" min={0} max={snaps.length - 1} step={1} value={stage}
        onChange={(e) => setStage(Number(e.target.value))} style={{ width: "100%" }} aria-label="Circuit stage" />
      <table className="facts-table small">
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
      <div ref={mount} role="img" aria-label="Four one-time-pad keys on a Bloch sphere and their average at the centre" style={{ width: "100%", minHeight: 300 }} />
      {failed && <Fallback what="the one-time pad" />}
      <p className="small muted">{pts.map((p) => `${p.label}: (${p.v.map((c) => c.toFixed(2)).join(", ")})`).join(" · ")} · average: (0, 0, 0)</p>
    </figure>
  );
}

/** The satellite downlink geometry - an ILLUSTRATION: the model uses slant range only. */
export function SatScene({ slantKm }: { slantKm: number }) {
  const R = 6371;
  const h = Math.min(500, Math.max(1, slantKm));            // drawing altitude, never above the range
  const sinEl = ((R + h) ** 2 - R * R - slantKm ** 2) / (2 * R * slantKm);
  const el = Math.asin(Math.max(-1, Math.min(1, sinEl)));
  const { mount, failed } = useScene((scene) => {
    const s = 1 / R;                                         // scene units: Earth radius = 1
    const earth = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24),
      new THREE.MeshStandardMaterial({ color: 0x1d4ed8, roughness: 0.9, transparent: true, opacity: 0.85 }));
    scene.add(earth);
    const gs = new THREE.Vector3(0, 1, 0);                   // ground station at the "north pole" of the drawing
    const sat = new THREE.Vector3(Math.cos(el) * slantKm * s, 1 + Math.sin(el) * slantKm * s, 0);
    const dot = (p: THREE.Vector3, color: number, r: number) => {
      const m = new THREE.Mesh(new THREE.SphereGeometry(r, 16, 8), new THREE.MeshStandardMaterial({ color }));
      m.position.copy(p); scene.add(m);
    };
    dot(gs, 0x36d399, 0.025); dot(sat, 0xf6b73c, 0.03);
    scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([gs, sat]),
      new THREE.LineDashedMaterial({ color: 0xb072ff, dashSize: 0.02, gapSize: 0.015 })).computeLineDistances());
  }, [slantKm]);
  return (
    <figure className="chart-figure" style={{ margin: 0 }}>
      <figcaption className="small muted">
        <b>Downlink geometry (illustration).</b> The ground station (green), a satellite (amber) at this page&apos;s
        slant range of {slantKm} km, and the line of sight. The channel model uses the slant range alone - it has
        no altitude, elevation or pass - so the drawing assumes a {h} km altitude, which puts the satellite at
        {" "}{(el * 180 / Math.PI).toFixed(1)} degrees elevation. This 3D view adds no information to the model.
      </figcaption>
      <div ref={mount} role="img" aria-label={`Ground station and satellite at ${slantKm} km slant range`} style={{ width: "100%", minHeight: 300 }} />
      {failed && <Fallback what="the downlink" />}
    </figure>
  );
}
