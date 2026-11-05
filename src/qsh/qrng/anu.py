"""ANU Quantum Numbers (AQN): an opt-in, online hardware entropy source, keyed from the environment.

The ANU QRNG's numbers "are generated in real-time in our lab by measuring the quantum fluctuations
of the vacuum", and the AQN API serves them at ``https://api.quantumnumbers.anu.edu.au`` with the
key in an ``x-api-key`` header. The AQN documentation's own form of a request:
``curl -X GET -H "x-api-key:${API_KEY}" "${API_URL}?length=[array length]&type=[data type]&size=[block size]"``,
the length "between 1-1024", type 'uint8' for "integers between 0-255", and the block size "only
needed for 'hex8' and 'hex16' data types". (Round 84: the keyless legacy API this module called until
then refused connections on 2026-10-31.)

THE KEY COMES FROM THE ENVIRONMENT AND NOWHERE ELSE: ``QSH_ANU_API_KEY``. No file in this repository
holds one, no function takes one as an argument, so no caller can write one into code, and no
message or exception includes it. Without it the source reports itself unavailable
(:func:`anu_status`, :class:`AnuUnavailable`) before any network call, and nothing else changes: the
offline state-vector simulation in :mod:`~qsh.qrng.simulator` is the default, and the source every
pipeline stage uses. Asked for explicitly with no key, the online source raises rather than
substituting simulated bits, which is the no-silent-fallback rule ``generate_quantum_bits`` applies
to an unreachable host.

THE FREE PLAN'S LIMITS, from the AQN pricing page, where it is called Trial ($0 per request; read in a
browser 2026-11-05): "100 requests per month" and "1 request per second". Requests leave this process at least a second apart, and a call that would need more
requests than the monthly allowance is refused before it sends one; the paid plan (AWS Marketplace)
raises both, which ``max_requests`` lets its holder say. The month's count is the service's to keep:
this module cannot see requests made elsewhere with the same key.

NOT A CRYPTOGRAPHIC SOURCE AS USED HERE: the numbers cross the internet, and the service says so
itself: "For high security cryptographic applications, hardware units are essential". Use is subject
to the AQN terms; attribute the ANU QRNG service when using its output. NEVER used by CI or tests,
which mock the HTTP call.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

import numpy as np

__all__ = ["ANU_API_KEY_ENV", "ANU_ENDPOINT", "AnuStatus", "AnuUnavailable", "anu_status",
           "fetch_anu_bits"]

#: The AQN API (the documentation's ``API_URL``).
ANU_ENDPOINT = "https://api.quantumnumbers.anu.edu.au"
#: The one place the key is read from.
ANU_API_KEY_ENV = "QSH_ANU_API_KEY"
_ANU_MAX_LEN = 1024                    # "the length of the array to return. Must be between 1-1024"
FREE_PLAN_REQUESTS_PER_MONTH = 100     # the free plan: "100 requests per month"
MIN_REQUEST_INTERVAL_S = 1.0           # the free plan: "1 request per second"

# The spacing is per process, so concurrent callers share it. Swapped by the tests for a fake clock.
_clock = time.monotonic
_sleep = time.sleep
_lock = threading.Lock()
_last_request: float | None = None


class AnuUnavailable(RuntimeError):
    """The online source cannot be used: no AQN key in the environment. Raised before any request."""


@dataclass(frozen=True)
class AnuStatus:
    """Whether the online source can be used, and why not when it cannot. Never carries the key."""

    available: bool
    reason: str
    endpoint: str = ANU_ENDPOINT


def _api_key() -> str | None:
    key = os.environ.get(ANU_API_KEY_ENV, "").strip()
    return key or None


def anu_status() -> AnuStatus:
    """The online source's availability, read from the environment alone: no request is made."""
    if _api_key() is None:
        return AnuStatus(False, f"ANU QRNG unavailable: no AQN API key in {ANU_API_KEY_ENV}. The AQN API "
                                f"({ANU_ENDPOINT}) requires one; the offline simulated source ('sim') "
                                f"stays the default and is what every pipeline stage uses.")
    return AnuStatus(True, f"an AQN API key is set in {ANU_API_KEY_ENV}; requests are spaced "
                           f"{MIN_REQUEST_INTERVAL_S:g} s apart, the free plan's rate")


def _wait_for_slot() -> None:
    """Hold the caller until a second has passed since this process's last request."""
    global _last_request
    with _lock:
        now = _clock()
        if _last_request is not None and now - _last_request < MIN_REQUEST_INTERVAL_S:
            _sleep(MIN_REQUEST_INTERVAL_S - (now - _last_request))
            now = _clock()
        _last_request = now


def _request(length: int, key: str, timeout: float) -> list[int]:
    query = urllib.parse.urlencode({"length": length, "type": "uint8"})
    req = urllib.request.Request(f"{ANU_ENDPOINT}?{query}",
                                 headers={"x-api-key": key, "Accept": "application/json"})
    _wait_for_slot()
    with urllib.request.urlopen(req, timeout=timeout) as resp:      # noqa: S310 (opt-in, fixed host)
        data = json.loads(resp.read().decode())
    if not isinstance(data, dict) or data.get("success") is False or not isinstance(data.get("data"), list):
        detail = data.get("message") if isinstance(data, dict) else None
        raise RuntimeError("ANU QRNG API returned no numbers" + (f": {detail}" if detail else ""))
    values = data["data"]
    if len(values) != length or not all(isinstance(v, int) and 0 <= v <= 255 for v in values):
        raise RuntimeError(f"ANU QRNG API returned {len(values)} values, not {length} uint8 values")
    return values


def fetch_anu_bits(n_bits: int, *, timeout: float = 5.0,
                   max_requests: int = FREE_PLAN_REQUESTS_PER_MONTH) -> np.ndarray:
    """Fetch ``n_bits`` hardware quantum-random bits from the AQN API (opt-in, online).

    Raises :class:`AnuUnavailable` when no key is set, and ``ValueError`` when the bits would take
    more than ``max_requests`` requests (by default the free plan's monthly allowance), both before
    any request is sent.
    """
    key = _api_key()
    if key is None:
        raise AnuUnavailable(anu_status().reason)
    n_bytes = (int(n_bits) + 7) // 8
    needed = -(-n_bytes // _ANU_MAX_LEN)
    if needed > max_requests:
        raise ValueError(f"{n_bits} bits take {needed} AQN requests of {_ANU_MAX_LEN} bytes, more than the "
                         f"{max_requests} allowed (the free plan allows {FREE_PLAN_REQUESTS_PER_MONTH} "
                         f"a month); no request was sent")
    out: list[int] = []
    while len(out) < n_bytes:
        out.extend(_request(min(_ANU_MAX_LEN, n_bytes - len(out)), key, timeout))
    arr = np.frombuffer(bytes(out[:n_bytes]), dtype=np.uint8)
    return np.unpackbits(arr)[:n_bits].astype(np.int64)
