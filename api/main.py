"""FastAPI backend for the QSH SPA.

Exposes:
  REST  /api/health, /api/usecases, /api/run/{usecase}, /api/stop/{id},
        /api/restart/{id}, /api/runs/{id}, /api/benchmark, /api/infer/*,
        /api/download/{file}
  WS    /api/ws/{id}   -- live pipeline event stream (drives the SPA DAG)

Heavy compute is server-side only in the full local deployment; the public demo
serves the static SPA and uses these endpoints sparingly (client does the rest).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .inference_service import pqc_handshake_demo, predict_fraud, qkd_demo
from .run_manager import RESULTS_DIR, RunManager

app = FastAPI(title="QSMPC-QKD-QHE-AI-Hybrid API", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.middleware("http")
async def cross_origin_isolation(request, call_next):
    """COOP/COEP -> cross-origin isolation for SharedArrayBuffer / WebGPU / threaded WASM."""
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    return response
manager = RunManager()
WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"


class RunConfig(BaseModel):
    config: dict = {}
    start_paused: bool = False


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "qsh", "version": "0.1.0"}


@app.get("/metrics")
def metrics():
    from fastapi import Response

    from qsh.monitoring.exporter import render

    body, content_type = render()
    return Response(content=body, media_type=content_type)


@app.get("/api/usecases")
def usecases() -> list[dict]:
    return manager.list_usecases()


@app.post("/api/run/{usecase}")
def run(usecase: str, body: RunConfig | None = None) -> dict:
    cfg = body.config if body else {}
    paused = body.start_paused if body else False
    run_id = manager.start(usecase, cfg, start_paused=paused)
    return {"run_id": run_id, "usecase": usecase}


@app.post("/api/stop/{run_id}")
def stop(run_id: str) -> dict:
    return {"stopped": manager.stop(run_id)}


@app.post("/api/pause/{run_id}")
def pause(run_id: str) -> dict:
    return {"paused": manager.pause(run_id)}


@app.post("/api/resume/{run_id}")
def resume(run_id: str) -> dict:
    return {"resumed": manager.resume(run_id)}


@app.post("/api/step/{run_id}")
def step(run_id: str) -> dict:
    return {"stepped": manager.step(run_id)}


@app.get("/api/crypto-options")
def crypto_options() -> dict:
    """Expose the crypto-agility registry: swappable providers per family."""
    from qsh.orchestration.crypto_agility import default_registry

    reg = default_registry()
    fams = ["kem", "qkd", "fhe", "mpc", "sig"]
    return {
        "families": {f: reg.options(f) for f in fams},
        "defaults": {f: reg.default(f) for f in fams},
    }


@app.get("/api/config")
def app_config() -> dict:
    """Single source of truth for the SPA: benchmark cost model, tier architectures, workload
    multipliers, the medical-CNN shape, security/advantage weights, demo data sizes. The frontend
    fetches this (full mode) instead of duplicating the literals; the demo bakes the same JSON."""
    from qsh.config import frontend_config

    return frontend_config()


@app.post("/api/restart/{run_id}")
def restart(run_id: str) -> dict:
    new_id = manager.restart(run_id)
    return {"run_id": new_id} if new_id else JSONResponse({"error": "unknown run"}, 404)


@app.get("/api/runs/{run_id}")
def run_status(run_id: str):
    st = manager.status(run_id)
    return st or JSONResponse({"error": "unknown run"}, 404)


@app.websocket("/api/ws/{run_id}")
async def ws(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
    run = manager.get(run_id)
    if not run:
        await websocket.send_json({"type": "error", "message": "unknown run"})
        await websocket.close()
        return
    try:
        # replay any already-emitted events, then stream live
        replayed = 0
        while True:
            hist = run.pipeline.bus.history
            while replayed < len(hist):
                await websocket.send_text(hist[replayed].to_json())
                replayed += 1
            # terminal once the run finished AND every event has been sent
            if run.state in ("done", "stopped", "error") and replayed >= len(run.pipeline.bus.history):
                await websocket.send_json({"type": "stream_end", "state": run.state})
                break
            await asyncio.sleep(0.15)
    except WebSocketDisconnect:
        pass


@app.get("/api/benchmark")
def benchmark(workload: str = "linear", samples: int = 20, model_size: str = "medium") -> dict:
    from qsh.benchmark.runner import compute_advantage, run_benchmark

    results = run_benchmark(workload, out_dir=RESULTS_DIR, n_samples=samples,
                            include_openfhe=True, model_size=model_size)
    return {"workload": workload, "model_size": model_size,
            "results": [r.to_row() for r in results], "advantage": compute_advantage(results)}


@app.post("/api/alice-bob")
def alice_bob(body: RunConfig | None = None) -> dict:
    """Two-party Alice<->Bob E2E: PQC handshake -> QKD -> QOT/PSI -> private inference."""
    from qsh.benchmark.alice_bob_e2e import run_alice_bob_e2e

    cfg = body.config if body else {}
    return run_alice_bob_e2e(seed=int(cfg.get("seed", 0)),
                             n_shared=int(cfg.get("shared", 8)))


class FraudReq(BaseModel):
    features: list[float]


@app.post("/api/infer/fraud")
def infer_fraud(req: FraudReq) -> dict:
    return predict_fraud(req.features)


@app.get("/api/demo/pqc")
def demo_pqc() -> dict:
    return pqc_handshake_demo()


@app.get("/api/demo/qkd")
def demo_qkd(distance_km: float = 50.0, eve: float = 0.0) -> dict:
    return qkd_demo(distance_km, eve)


@app.get("/api/demo/mcp")
def demo_mcp(qkd: bool = False) -> dict:
    """Quantum-safe MCP agent-to-agent call (PQC+QKD session, ML-DSA-signed, AEAD)."""
    from qsh.pqc.mcp_bridge import mcp_demo

    qkd_key = (b"q" * 32) if qkd else None
    return mcp_demo(qkd_key=qkd_key)


class McpCallReq(BaseModel):
    name: str
    arguments: dict = {}
    qkd: bool = False


class McpFsReq(BaseModel):
    directory: str = "."
    tool: str = "list_directory"
    arguments: dict | None = None
    qkd: bool = False


@app.get("/api/mcp/tools")
def mcp_tools() -> dict:
    """List the real MCP server's 5 per-industry tools (full deployment; the demo uses the client mirror)."""
    from qsh.mcp.server import list_tools

    return {"server": "qsh-quantum-safe", "tools": list_tools()}


@app.post("/api/mcp/call")
def mcp_call(req: McpCallReq) -> dict:
    """Invoke a real MCP tool over the quantum-safe channel (PQC+QKD+ML-DSA+AEAD) → signed real result."""
    from qsh.pqc.mcp_bridge import Agent, secure_agent_call

    qkd_key = (b"q" * 32) if req.qkd else None
    caller, callee = Agent.create("api-agent"), Agent.create(f"{req.name}-silo")
    try:
        return secure_agent_call(caller, callee, "tools/call",
                                 {"name": req.name, "arguments": req.arguments}, qkd_key=qkd_key)
    finally:
        caller.free()
        callee.free()


@app.post("/api/mcp/filesystem")
def mcp_filesystem(req: McpFsReq) -> dict:
    """Interop: call the OFFICIAL filesystem MCP server (Node) + carry its result over the quantum-safe layer."""
    from qsh.mcp.fs_interop import node_available, secure_filesystem_interop

    if not node_available():
        return JSONResponse({"error": "Node/npx not available — filesystem-MCP interop needs Node"}, 503)
    qkd_key = (b"q" * 32) if req.qkd else None
    try:
        return secure_filesystem_interop(req.directory, req.tool, req.arguments, qkd_key=qkd_key)
    except Exception as exc:                             # surface the cause (no silent fallback)
        return JSONResponse({"error": str(exc)}, 500)


@app.get("/api/download/{filename}")
def download(filename: str):
    safe = Path(filename).name                       # prevent path traversal
    path = RESULTS_DIR / safe
    if not path.exists():
        return JSONResponse({"error": "not found"}, 404)
    return FileResponse(path, filename=safe)


@app.get("/api/results")
def list_results() -> list[dict]:
    if not RESULTS_DIR.exists():
        return []
    return [{"name": p.name, "bytes": p.stat().st_size}
            for p in sorted(RESULTS_DIR.glob("*")) if p.is_file() and p.suffix in
            (".json", ".csv", ".jsonl")]


# Serve the built SPA with history-API (SPA) fallback so deep links / refreshes work.
# Mount hashed assets directly; everything else that isn't a real file -> index.html.
if WEB_DIST.exists():
    if (WEB_DIST / "assets").exists():
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str):
        if full_path.startswith("api/"):
            return JSONResponse({"error": "not found"}, 404)
        candidate = (WEB_DIST / full_path)
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(WEB_DIST / "index.html")  # SPA route -> client router

