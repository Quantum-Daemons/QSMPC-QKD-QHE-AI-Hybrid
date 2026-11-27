#!/usr/bin/env python3
"""D1 v2's feasibility gate, its first half: AkitaV2 ported to PyTorch, checked against TensorFlow on the CPU (Round 86,
batch 8; the owner's choices of 2026-11-26 and 2026-11-27: the port check only, downloads asked one by one, nothing
published until the weights' authors answer the licence inquiry).

The draft pre-registration's gate (outside the repository until it is frozen): the PyTorch port equals TensorFlow 2.21
on the CPU within 1e-4, the largest absolute difference of the map, on eight sequences, four windows with the reference
and one haplotype each. Three steps, two interpreters, so PyTorch is not installed beside TensorFlow:

    python -I scripts/train/akita_v2_port_check.py sequences   # the project's venv: GRCh38 (Ensembl REST) + D1 v1's variants
    <TF venv>/python scripts/train/akita_v2_port_check.py reference   # TF 2.21 + basenji: the graph, its weights, the outputs
    python -I scripts/train/akita_v2_port_check.py compare     # the project's venv: the PyTorch port on the same inputs

Everything is written to WORK, outside the repository. The model is AkitaV2's human head, fold 0 (gs://basenji_hic/3-2021/
models/f0c0/train/model0_best.h5); `reference` checks its size and MD5 before anything reads it.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO.parent / "tools_external"
WORK = TOOLS / "akita_v2_port_check"
MODEL = TOOLS / "akita_v2" / "f0c0" / "model0_best.h5"
PARAMS = TOOLS / "akita_v2" / "f0c0" / "params.json"
MODEL_BYTES, MODEL_MD5_B64 = 12_854_344, "2bQ1V9RK52LwnDS5CNtD+A=="
SEQ_LEN, WINDOW, PAD = 1_310_720, 1_048_576, 131_072
TOL = 1e-4
D1_RECORD = REPO / "docs" / "measurements" / "genomics-3d-divergence-2026-11-26.json"


def one_hot(seq: str) -> np.ndarray:
    lut = np.full(256, -1, dtype=np.int64)
    for i, b in enumerate("ACGT"):
        lut[ord(b)] = i
        lut[ord(b.lower())] = i
    idx = lut[np.frombuffer(seq.encode(), dtype=np.uint8)]
    out = np.zeros((len(seq), 4), dtype=np.float32)
    ok = idx >= 0
    out[np.flatnonzero(ok), idx[ok]] = 1.0          # N and other codes stay all zero, as basenji's dna_1hot leaves them
    return out


def stage_sequences() -> int:
    """Four of D1 v1's windows: the GRCh38 reference of the 1,310,720 bp input centred on each, and one haplotype, the
    reference with the alternate alleles of the first individual of D1 v1's inputs who carries any in the window."""
    sys.path.insert(0, str(REPO / "src"))
    from qsh.models import datasets

    d1 = json.loads(D1_RECORD.read_text())
    meta = json.loads(datasets.artifacts("akita_d1_inputs.json").read_text())
    C = np.load(datasets.artifacts("akita_d1_inputs.npz"))["C"]
    WORK.mkdir(parents=True, exist_ok=True)
    seqs, info = [], []
    for w, win in enumerate(d1["windows"][:4]):
        chrom, start = str(win["chr"]).removeprefix("chr"), int(win["start"])
        lo = start - PAD                                   # 0-based start of the input interval
        url = f"https://rest.ensembl.org/sequence/region/human/{chrom}:{lo + 1}..{lo + SEQ_LEN}:1?content-type=text/plain"
        req = urllib.request.Request(url, headers={"User-Agent": "sequence-fetch", "Content-Type": "text/plain"})
        with urllib.request.urlopen(req, timeout=300) as r:
            ref = r.read().decode().strip()
        if len(ref) != SEQ_LEN:
            raise SystemExit(f"window {w}: Ensembl returned {len(ref)} bases, not {SEQ_LEN}")
        cols = [j for j, v in enumerate(meta["variants"]) if v["window"] == w and len(v["ref"]) == 1 and len(v["alt"]) == 1]
        person = next((i for i in range(C.shape[0]) if C[i, cols].max(initial=0) > 0), None)
        hap = list(ref)
        applied = mismatched = 0
        for j in cols:
            if person is None or C[person, j] <= 0:
                continue
            v = meta["variants"][j]
            k = int(v["pos"]) - lo - 1
            if 0 <= k < SEQ_LEN and ref[k].upper() == v["ref"].upper():
                hap[k] = v["alt"]
                applied += 1
            else:
                mismatched += 1
        seqs += [ref, "".join(hap)]
        info.append({"window": w, "chr": chrom, "window_start": start, "input": [lo, lo + SEQ_LEN],
                     "ensembl_sha256": hashlib.sha256(ref.encode()).hexdigest(), "n_bases": int(sum(b in "ACGTacgt" for b in ref)),
                     "variants_applied": applied, "ref_mismatches": mismatched})
        print(f"window {w} {chrom}:{start}: {applied} alternate alleles applied, {mismatched} not matching", flush=True)
    np.save(WORK / "inputs.npy", np.stack([one_hot(s) for s in seqs]))
    (WORK / "sequences.json").write_text(json.dumps({"windows": info, "order": "reference then haplotype, per window"}, indent=1))
    return 0


def stage_reference() -> int:
    """In the TF venv: the graph as a list of layers (type, configuration, inputs), its weights, and TF's outputs."""
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    if MODEL.stat().st_size != MODEL_BYTES or base64.b64encode(hashlib.md5(MODEL.read_bytes()).digest()).decode() != MODEL_MD5_B64:
        raise SystemExit("the AkitaV2 model is not the one the bucket lists: run nothing")
    sys.path.insert(0, str(TOOLS / "basenji"))
    import tensorflow as tf
    from basenji import seqnn

    m = seqnn.SeqNN(json.loads(PARAMS.read_text())["model"])
    m.restore(str(MODEL), head_i=0)
    model = m.model
    keep = ("kernel_size", "dilation_rate", "strides", "padding", "use_bias", "pool_size", "epsilon", "cropping",
            "diagonal_offset", "operation", "additive", "units", "activation", "axis")
    graph, weights = [], {}
    for layer in model.layers:
        cfg = layer.get_config()
        node = layer._inbound_nodes[0]
        inputs = [[il.name, int(ti)] for il, _ni, ti, _t in node.iterate_inbound()]
        entry = {"name": layer.name, "class": type(layer).__name__, "inputs": inputs,
                 "config": {k: cfg[k] for k in keep if k in cfg}}
        if type(layer).__name__ == "SqueezeExcite":
            entry["config"].update({"additive": layer.additive, "norm": layer.norm is not None,
                                    "two_d": layer.one_or_two == "two"})
            if layer.norm is not None:
                entry["config"]["epsilon"] = float(layer.norm.epsilon)
        for i, wv in enumerate(layer.get_weights()):
            weights[f"{layer.name}/{i}"] = wv
        graph.append(entry)
    x = np.load(WORK / "inputs.npy")
    outs, times = [], []
    for i in range(len(x)):
        t0 = time.time()
        outs.append(model(x[i:i + 1], training=False).numpy()[0])
        times.append(time.time() - t0)
    np.savez(WORK / "weights.npz", **weights)
    np.save(WORK / "tf_outputs.npy", np.stack(outs))
    (WORK / "graph.json").write_text(json.dumps({"layers": graph, "output": model.output.name if hasattr(model.output, "name") else None,
                                                 "tensorflow": tf.__version__, "seconds_per_sequence": times}, indent=1))
    print(f"exported {len(graph)} layers, {len(weights)} weight arrays; TF {np.median(times):.1f} s a sequence", flush=True)
    return 0


def stage_compare() -> int:
    """In the project's venv: each layer by its TF definition, in torch, on the CPU, in float32."""
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    graph = json.loads((WORK / "graph.json").read_text())["layers"]
    W = np.load(WORK / "weights.npz")
    w = lambda name, i: torch.from_numpy(np.ascontiguousarray(W[f"{name}/{i}"]))  # noqa: E731

    def same_pad(k: int, d: int) -> tuple[int, int]:
        total = d * (k - 1)
        return total // 2, total - total // 2                # TF's 'same': the extra on the right

    def run(x: np.ndarray) -> np.ndarray:
        vals: dict[str, list] = {}
        for L in graph:
            name, cls, cfg = L["name"], L["class"], L["config"]
            ins = [vals[n][i] for n, i in L["inputs"]]
            if cls == "InputLayer":
                out = [torch.from_numpy(x)[None]]
            elif cls == "StochasticReverseComplement":
                out = [ins[0], False]                        # inference: the sequence unchanged, reverse False
            elif cls in ("StochasticShift", "Dropout"):
                out = [ins[0]]
            elif cls == "SwitchReverseTriu":
                if ins[1] is not False:
                    raise SystemExit("a reversed input reached SwitchReverseTriu; inference never reverses")
                out = [ins[0]]
            elif cls == "ReLU":
                out = [F.relu(ins[0])]
            elif cls == "Conv1D":
                k, d = cfg["kernel_size"][0], cfg["dilation_rate"][0]
                h = ins[0].permute(0, 2, 1)                 # (B, C, L)
                h = F.pad(h, same_pad(k, d))
                y = F.conv1d(h, w(name, 0).permute(2, 1, 0), w(name, 1) if cfg.get("use_bias") else None, dilation=d)
                out = [y.permute(0, 2, 1)]
            elif cls == "Conv2D":
                k, d = cfg["kernel_size"][0], cfg["dilation_rate"][0]
                h = ins[0].permute(0, 3, 1, 2)
                p = same_pad(k, d)
                h = F.pad(h, (p[0], p[1], p[0], p[1]))
                y = F.conv2d(h, w(name, 0).permute(3, 2, 0, 1), w(name, 1) if cfg.get("use_bias") else None, dilation=d)
                out = [y.permute(0, 2, 3, 1)]
            elif cls == "BatchNormalization":
                g, b, mu, var = (w(name, i) for i in range(4))
                out = [(ins[0] - mu) / torch.sqrt(var + cfg["epsilon"]) * g + b]
            elif cls == "MaxPooling1D":
                pool = cfg["pool_size"][0]
                if ins[0].shape[1] % pool:
                    raise SystemExit(f"{name}: a length not divisible by the pool, which 'same' would pad")
                out = [F.max_pool1d(ins[0].permute(0, 2, 1), pool).permute(0, 2, 1)]
            elif cls == "Add":
                out = [sum(ins[1:], ins[0])]
            elif cls == "OneToTwo":
                if cfg.get("operation", "mean") != "mean":
                    raise SystemExit(f"{name}: operation {cfg.get('operation')} is not ported")
                v = ins[0]
                out = [(v[:, None, :, :] + v[:, :, None, :]) / 2]
            elif cls == "Symmetrize2D":
                out = [(ins[0] + ins[0].transpose(1, 2)) / 2]
            elif cls == "SqueezeExcite":
                v = F.relu(ins[0])
                sq = v.mean(dim=(1, 2)) if cfg["two_d"] else v.mean(dim=1)
                ex = F.relu(sq @ w(name, 0) + w(name, 1))
                ex = ex @ w(name, 2) + w(name, 3)
                if cfg["norm"]:
                    g, b, mu, var = (w(name, i) for i in range(4, 8))
                    ex = (ex - mu) / torch.sqrt(var + cfg["epsilon"]) * g + b
                ex = ex[:, None, None, :] if cfg["two_d"] else ex[:, None, :]
                out = [v + ex if cfg["additive"] else v * torch.sigmoid(ex)]
            elif cls == "Cropping2D":
                (t, bt), (lf, rt) = cfg["cropping"]
                hgt, wid = ins[0].shape[1], ins[0].shape[2]
                out = [ins[0][:, t:hgt - bt, lf:wid - rt, :]]
            elif cls == "UpperTri":
                n, c = ins[0].shape[1], ins[0].shape[3]
                iu = np.triu_indices(n, cfg["diagonal_offset"])
                idx = torch.from_numpy((iu[0] + n * iu[1]).astype(np.int64))
                out = [ins[0].reshape(1, n * n, c)[:, idx, :]]
            elif cls == "Dense":
                y = ins[0] @ w(name, 0) + (w(name, 1) if cfg.get("use_bias", True) else 0)
                if cfg.get("activation") not in (None, "linear"):
                    raise SystemExit(f"{name}: activation {cfg.get('activation')} is not ported")
                out = [y]
            else:
                raise SystemExit(f"{name}: layer {cls} is not ported")
            vals[name] = out
        return vals[graph[-1]["name"]][0][0].numpy()

    x = np.load(WORK / "inputs.npy")
    tf_out = np.load(WORK / "tf_outputs.npy")
    rows = []
    with torch.no_grad():
        for i in range(len(x)):
            t0 = time.time()
            y = run(x[i])
            diff = float(np.max(np.abs(y - tf_out[i])))
            rows.append({"sequence": i, "max_abs_difference": diff, "max_abs_value": float(np.max(np.abs(tf_out[i]))),
                         "seconds": round(time.time() - t0, 2)})
            print(f"sequence {i}: max |torch - TF| {diff:.3e} (largest |value| {rows[-1]['max_abs_value']:.3f})", flush=True)
    worst = max(r["max_abs_difference"] for r in rows)
    result = {"tolerance": TOL, "worst": worst, "criterion_met": bool(worst <= TOL), "sequences": rows,
              "torch": torch.__version__, "device": "cpu"}
    (WORK / "port_check.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"worst {worst:.3e} against {TOL}: {'met' if worst <= TOL else 'NOT met'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("sequences", "reference", "compare"))
    a = ap.parse_args()
    return {"sequences": stage_sequences, "reference": stage_reference, "compare": stage_compare}[a.stage]()


if __name__ == "__main__":
    raise SystemExit(main())
