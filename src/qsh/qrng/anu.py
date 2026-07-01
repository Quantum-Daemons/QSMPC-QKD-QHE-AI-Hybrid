"""ANU QRNG — real hardware quantum entropy from the ANU Quantum Numbers JSON API.

This is an **opt-in, online** source (``qrng.anu.edu.au``). It is NEVER imported at module load,
NEVER used by CI/tests, and only called when the caller passes ``anu_opt_in=True``. Use is subject to
the **ANU Quantum Numbers API terms**; attribute the ANU QRNG service when using its output.

The offline state-vector simulation in :mod:`~qsh.qrng.simulator` is the default entropy source.
"""

from __future__ import annotations

import json
import urllib.request

import numpy as np

_ANU_ENDPOINT = "https://qrng.anu.edu.au/API/jsonI.php"
_ANU_MAX_LEN = 1024          # the API returns up to 1024 uint8 values per request


def fetch_anu_bits(n_bits: int, *, timeout: float = 5.0) -> np.ndarray:
    """Fetch ``n_bits`` real quantum-random bits from the ANU QRNG API (opt-in, online)."""
    n_bytes = (n_bits + 7) // 8
    out: list[int] = []
    while len(out) < n_bytes:
        length = min(_ANU_MAX_LEN, n_bytes - len(out))
        url = f"{_ANU_ENDPOINT}?length={length}&type=uint8"
        with urllib.request.urlopen(url, timeout=timeout) as resp:     # noqa: S310 (opt-in, explicit host)
            data = json.loads(resp.read().decode())
        if not data.get("success"):
            raise RuntimeError("ANU QRNG API returned success=false")
        out.extend(int(x) for x in data["data"])
    arr = np.frombuffer(bytes(out[:n_bytes]), dtype=np.uint8)
    return np.unpackbits(arr)[:n_bits].astype(np.int64)
