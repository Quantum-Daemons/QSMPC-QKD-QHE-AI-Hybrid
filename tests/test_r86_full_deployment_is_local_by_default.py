"""The full deployment is local by default, and its API refuses DNS-rebound requests (Round 86).

WHY THIS EXISTS (the owner's choice of 2026-11-22)

The documentation audit of 2026-11-22 found that `docker/docker-compose.yml` bound the API's own port to loopback
while publishing the web service, which forwards `/api/*` and `/metrics` to the API without authentication, on every
interface, with Prometheus, Grafana and node-exporter beside it. And nothing checked a request's `Host` or `Origin`,
so a web page the operator opened could reach the API through DNS rebinding even on localhost. Docker's
port-publishing docs say publishing is insecure by default; the MCP specification (2026-07-28, Streamable HTTP) says a
server MUST validate `Origin` against DNS rebinding, answering 403, and SHOULD bind to localhost when local.

WHAT IS ASSERTED

- Every published port in `docker/docker-compose.yml` binds to 127.0.0.1, directly or through `QSH_BIND_ADDRESS`
  defaulting to it, and the API service receives `QSH_ALLOWED_HOSTS` and `QSH_CORS_ORIGINS`.
- The API answers its own names and refuses others: a foreign `Host` is 400, a foreign or `null` `Origin` 403, on HTTP
  and on the WebSocket handshake; Prometheus's `api:8000` and the loopback names pass, and so does an origin named in
  `QSH_CORS_ORIGINS`.
- The allowlists parse as documented. Seeded: the compose mapping as it stood is refused.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest
import yaml

starlette = pytest.importorskip("starlette")
from starlette.applications import Starlette  # noqa: E402
from starlette.responses import PlainTextResponse  # noqa: E402
from starlette.routing import Route, WebSocketRoute  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from api.host_guard import DEFAULT_HOSTS, HostOriginGuard, allowed_hosts_from_env, allowed_origins_from_env  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
COMPOSE = REPO / "docker" / "docker-compose.yml"
LOCAL = re.compile(r"^(127\.0\.0\.1|\$\{QSH_BIND_ADDRESS:-127\.0\.0\.1\}):")


def mapping_problems(services: dict) -> list[str]:
    problems = []
    for name, svc in services.items():
        for port in svc.get("ports") or []:
            if not isinstance(port, str) or not LOCAL.match(port):
                problems.append(f"{name}: {port!r} is published beyond 127.0.0.1 by default")
    return problems


def test_every_published_port_is_local_by_default() -> None:
    services = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]
    published = [p for s in services.values() for p in (s.get("ports") or [])]
    assert len(published) >= 5, published
    assert mapping_problems(services) == []
    env = services["api"].get("environment") or []
    assert any(str(e).startswith("QSH_ALLOWED_HOSTS=") for e in env), env
    assert any(str(e).startswith("QSH_CORS_ORIGINS=") for e in env), env


def test_the_compose_check_refuses_the_mapping_as_it_stood() -> None:
    assert mapping_problems({"web": {"ports": ["${WEB_PORT:-8080}:80"]}})


def test_the_allowlists_parse_as_documented() -> None:
    assert allowed_hosts_from_env({}) == frozenset(DEFAULT_HOSTS) == frozenset({"localhost", "127.0.0.1", "::1", "api"})
    assert allowed_hosts_from_env({"QSH_ALLOWED_HOSTS": " Box.Lan:8080, [::1] ,10.0.0.5"}) == {"box.lan", "::1", "10.0.0.5"}
    assert allowed_origins_from_env({"QSH_CORS_ORIGINS": "http://localhost:5173/, https://A.example"}) == {
        "http://localhost:5173", "https://a.example"}


def _client(hosts=DEFAULT_HOSTS, origins=(), host="localhost:8000") -> TestClient:
    async def ok(request):
        return PlainTextResponse("ok")

    async def ws(websocket):
        await websocket.accept()
        await websocket.send_text("ok")
        await websocket.close()

    app = Starlette(routes=[Route("/api/health", ok), Route("/metrics", ok), WebSocketRoute("/api/ws/x", ws)])
    return TestClient(HostOriginGuard(app, hosts=hosts, origins=origins), base_url=f"http://{host}")


@pytest.mark.parametrize("host", ["localhost:8080", "127.0.0.1:8000", "api:8000"])
def test_the_api_answers_its_own_names(host: str) -> None:
    assert _client(host=host).get("/api/health").status_code == 200


def _asgi_status(host: str, origin: str | None = None) -> int:
    """One request straight into the guard: Starlette's test client cannot send a bracketed IPv6 Host."""
    async def ok(request):
        return PlainTextResponse("ok")

    guard = HostOriginGuard(Starlette(routes=[Route("/api/health", ok)]), hosts=DEFAULT_HOSTS)
    headers = [(b"host", host.encode())] + ([(b"origin", origin.encode())] if origin else [])
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET", "scheme": "http",
             "path": "/api/health", "raw_path": b"/api/health", "root_path": "", "query_string": b"",
             "headers": headers, "server": ("::1", 8000), "client": ("::1", 50000)}
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(guard(scope, receive, send))
    return next(m["status"] for m in sent if m["type"] == "http.response.start")


def test_an_ipv6_loopback_host_passes_and_a_foreign_one_does_not() -> None:
    assert _asgi_status("[::1]:8000") == 200
    assert _asgi_status("[::1]:8000", origin="http://[::1]:8080") == 200
    assert _asgi_status("[2001:db8::1]:8000") == 400


def test_a_rebound_host_is_refused() -> None:
    r = _client(host="attacker.example:8080").get("/api/health")
    assert r.status_code == 400 and "Host" in r.json()["error"]


@pytest.mark.parametrize("origin, status", [
    ("http://localhost:8080", 200), ("http://127.0.0.1:5173", 200), ("http://attacker.example", 403),
    ("null", 403), ("file://", 403), ("http://localhost.attacker.example", 403),
])
def test_an_origin_must_be_one_the_deployment_serves(origin: str, status: int) -> None:
    assert _client().get("/api/health", headers={"Origin": origin}).status_code == status


def test_a_named_cross_origin_passes() -> None:
    c = _client(origins={"https://lab.example"})
    assert c.get("/api/health", headers={"Origin": "https://lab.example"}).status_code == 200
    assert c.get("/api/health", headers={"Origin": "https://other.example"}).status_code == 403


def test_the_websocket_handshake_is_checked_too() -> None:
    # The test client sends Host: testserver on a WebSocket whatever its base URL, so each handshake names its host.
    good = {"host": "localhost:8000", "origin": "http://localhost:8080"}
    with _client().websocket_connect("/api/ws/x", headers=good) as ws:
        assert ws.receive_text() == "ok"
    for bad in ({**good, "origin": "http://attacker.example"}, {**good, "host": "attacker.example:8000"}):
        with pytest.raises(WebSocketDisconnect) as refused:
            with _client().websocket_connect("/api/ws/x", headers=bad):
                pass
        assert refused.value.code == 1008, bad


def test_the_real_app_carries_the_guard() -> None:
    from fastapi.testclient import TestClient as FastAPIClient

    from api.main import app

    assert any(m.cls is HostOriginGuard for m in app.user_middleware)
    assert FastAPIClient(app, base_url="http://localhost").get("/api/health").status_code == 200
    assert FastAPIClient(app, base_url="http://attacker.example").get("/api/health").status_code == 400
    r = FastAPIClient(app, base_url="http://localhost").get("/api/health", headers={"Origin": "http://attacker.example"})
    assert r.status_code == 403
