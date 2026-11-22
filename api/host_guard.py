"""The full deployment's API answers only the host names it is configured for, and only its own origins (Round 86).

WHY THIS EXISTS (the owner's choice of 2026-11-22)

The API runs tools and pipelines for whoever can reach it, unauthenticated. Binding its ports to loopback keeps the
network out, but not a web page the operator happens to open: through DNS rebinding a hostile site makes its own host
name resolve to 127.0.0.1, and the browser then sends that site's requests to the local API as same-origin requests,
which CORS does not stop. The MCP specification (revision 2026-07-28, Streamable HTTP, "Security Warning") requires a
server to validate the `Origin` header against exactly this and to answer an invalid one with 403, and says a local
server should bind to localhost only.

So every HTTP request and WebSocket handshake is checked before the app sees it:

- its `Host` must name one of `QSH_ALLOWED_HOSTS` (comma-separated, ports ignored; by default the loopback names and
  `api`, the compose service name Prometheus scrapes), else 400;
- an `Origin`, when the browser sends one, must be http or https on one of those hosts, or one of `QSH_CORS_ORIGINS`,
  else 403. `null` is refused.

A request with no `Origin` (curl, Prometheus, the healthcheck) is checked on its `Host` alone.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from urllib.parse import urlsplit

DEFAULT_HOSTS = ("localhost", "127.0.0.1", "::1", "api")


def _names(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def allowed_hosts_from_env(environ: dict[str, str] | None = None) -> frozenset[str]:
    """The host names this deployment answers to: `QSH_ALLOWED_HOSTS`, or the defaults when it is unset or empty."""
    env = os.environ if environ is None else environ
    names = _names(env.get("QSH_ALLOWED_HOSTS", ""))
    return frozenset(_hostname(n) or n.lower() for n in (names or DEFAULT_HOSTS))


def allowed_origins_from_env(environ: dict[str, str] | None = None) -> frozenset[str]:
    """Origins named for cross-origin use (`QSH_CORS_ORIGINS`), normalised to scheme://host[:port]."""
    env = os.environ if environ is None else environ
    return frozenset(o.rstrip("/").lower() for o in _names(env.get("QSH_CORS_ORIGINS", "")))


def _hostname(host_header: str) -> str:
    """The host name of a `Host` header or an authority, without its port; IPv6 without brackets; lower case."""
    h = host_header.strip().lower()
    if h.startswith("["):
        end = h.find("]")
        return h[1:end] if end > 0 else ""
    if h.count(":") == 1:
        h = h.split(":", 1)[0]
    return h


class HostOriginGuard:
    """Pure ASGI, so it covers WebSocket handshakes too, which Starlette's HTTP middleware does not."""

    def __init__(self, app, hosts: Iterable[str], origins: Iterable[str] = ()):
        self.app = app
        self.hosts = frozenset(hosts)
        self.origins = frozenset(origins)

    def origin_ok(self, origin: str) -> bool:
        o = origin.strip().rstrip("/").lower()
        if o in self.origins:
            return True
        parts = urlsplit(o)
        return parts.scheme in ("http", "https") and bool(parts.hostname) and parts.hostname in self.hosts

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers") or []}
        if _hostname(headers.get("host", "")) not in self.hosts:
            await self._refuse(scope, send, 400, "the Host header names no host this deployment answers to")
            return
        origin = headers.get("origin")
        if origin is not None and not self.origin_ok(origin):
            await self._refuse(scope, send, 403, "the Origin header names a site this deployment does not serve")
            return
        await self.app(scope, receive, send)

    @staticmethod
    async def _refuse(scope, send, status: int, reason: str) -> None:
        if scope["type"] == "websocket":
            # Closing before accepting answers the handshake with HTTP 403.
            await send({"type": "websocket.close", "code": 1008, "reason": reason})
            return
        body = json.dumps({"error": reason}).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})
