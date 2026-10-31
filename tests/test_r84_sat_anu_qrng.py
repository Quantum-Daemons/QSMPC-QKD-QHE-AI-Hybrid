"""The ANU QRNG source calls the current AQN API with a key from the environment (Round 84, the
user's choice: "port with a key from the environment").

WHY THIS EXISTS. `src/qsh/qrng/anu.py` called the ANU's legacy keyless API, which refused connections
on 2026-10-31. The current ANU Quantum Numbers API is `api.quantumnumbers.anu.edu.au`, with the key in
an `x-api-key` header; its free plan allows "100 requests per month" and "1 request per second".

WHAT IS ASSERTED, with the HTTP call mocked and every socket refused, so nothing here touches the
network:
  - without `QSH_ANU_API_KEY` the source reports itself unavailable, and asking for its bits raises
    before any request, through `generate_quantum_bits` too; the simulated source is unchanged;
  - with a key, the request goes to the AQN host with the key in `x-api-key`, uint8, at most 1,024
    a request, and the bits are the response's;
  - requests are at least a second apart, and a call that would need more than the monthly allowance
    sends nothing;
  - the key appears in no message, and the module names no key and no legacy host.
"""

from __future__ import annotations

import io
import json
import socket
import urllib.parse

import numpy as np
import pytest

from qsh.qrng import anu, generate_quantum_bits

FAKE_KEY = "test-key-not-a-real-one-0123456789"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Refuse every socket, so a request the mock does not catch fails here instead of leaving."""
    def refuse(*_a, **_k):
        raise AssertionError("a test tried to open a network connection")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.delenv(anu.ANU_API_KEY_ENV, raising=False)
    monkeypatch.setattr(anu, "_last_request", None)


class _Clock:
    """A fake monotonic clock: `sleep` advances it, so the spacing can be read off exactly."""

    def __init__(self) -> None:
        self.t, self.sleeps = 100.0, []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


@pytest.fixture
def server(monkeypatch):
    """A mocked AQN API: records each request and answers with known bytes."""
    clock, calls = _Clock(), []
    monkeypatch.setattr(anu, "_clock", clock.monotonic)
    monkeypatch.setattr(anu, "_sleep", clock.sleep)

    def urlopen(req, timeout=None):
        url = urllib.parse.urlsplit(req.full_url)
        q = dict(urllib.parse.parse_qsl(url.query))
        calls.append({"host": url.netloc, "query": q, "key": req.get_header("X-api-key"), "t": clock.t})
        n = int(q["length"])
        body = {"success": True, "type": q["type"], "length": n, "data": [(7 * i + len(calls)) % 256 for i in range(n)]}
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(anu.urllib.request, "urlopen", urlopen)
    return calls, clock


def test_without_a_key_the_source_is_unavailable_and_sends_nothing(server) -> None:
    calls, _ = server
    st = anu.anu_status()
    assert st.available is False and anu.ANU_API_KEY_ENV in st.reason and "simulated" in st.reason
    with pytest.raises(anu.AnuUnavailable, match=anu.ANU_API_KEY_ENV):
        anu.fetch_anu_bits(64)
    with pytest.raises(anu.AnuUnavailable):
        generate_quantum_bits(64, source="anu", anu_opt_in=True)
    with pytest.raises(ValueError):                      # the opt-in guard is unchanged
        generate_quantum_bits(64, source="anu", anu_opt_in=False)
    assert calls == []
    # The default source is the simulator, as before, and needs no key.
    r = generate_quantum_bits(256, seed=3)
    assert r.simulated is True and r.source == "sim" and r.n_bits == 256


def test_with_a_key_the_request_is_the_aqn_apis(server, monkeypatch) -> None:
    calls, _ = server
    monkeypatch.setenv(anu.ANU_API_KEY_ENV, FAKE_KEY)
    assert anu.anu_status().available is True
    bits = anu.fetch_anu_bits(8 * 1500)                  # 1,500 bytes: two requests, 1,024 and 476
    assert [c["host"] for c in calls] == ["api.quantumnumbers.anu.edu.au"] * 2
    assert [c["query"] for c in calls] == [{"length": "1024", "type": "uint8"}, {"length": "476", "type": "uint8"}]
    assert all(c["key"] == FAKE_KEY for c in calls)
    first = [(7 * i + 1) % 256 for i in range(1024)]
    assert np.array_equal(bits[:8 * 1024], np.unpackbits(np.array(first, dtype=np.uint8)).astype(np.int64))
    assert bits.size == 12_000 and set(np.unique(bits)) <= {0, 1}


def test_requests_are_a_second_apart_and_the_allowance_is_kept(server, monkeypatch) -> None:
    calls, clock = server
    monkeypatch.setenv(anu.ANU_API_KEY_ENV, FAKE_KEY)
    anu.fetch_anu_bits(8 * 3000)                         # three requests in one call
    anu.fetch_anu_bits(8)                                # and one more call straight after
    gaps = [b["t"] - a["t"] for a, b in zip(calls, calls[1:], strict=False)]
    assert len(calls) == 4 and all(g >= anu.MIN_REQUEST_INTERVAL_S for g in gaps), gaps
    # More than the month's 100 requests in one call: refused before any is sent.
    sent = len(calls)
    with pytest.raises(ValueError, match="no request was sent"):
        anu.fetch_anu_bits(8 * 1024 * 101)
    assert len(calls) == sent
    anu.fetch_anu_bits(8 * 1024 * 2, max_requests=2)     # a paid plan's holder may say otherwise


def test_the_key_appears_in_no_message(server, monkeypatch) -> None:
    calls, _ = server
    monkeypatch.setenv(anu.ANU_API_KEY_ENV, FAKE_KEY)

    def refusing(req, timeout=None):
        return io.BytesIO(json.dumps({"success": False, "message": "Forbidden"}).encode())

    monkeypatch.setattr(anu.urllib.request, "urlopen", refusing)
    with pytest.raises(RuntimeError) as err:
        anu.fetch_anu_bits(64)
    assert "Forbidden" in str(err.value) and FAKE_KEY not in str(err.value)
    assert FAKE_KEY not in repr(anu.anu_status())


def test_the_module_holds_no_key_and_names_no_legacy_host() -> None:
    src = open(anu.__file__).read()
    assert anu.ANU_ENDPOINT == "https://api.quantumnumbers.anu.edu.au"
    assert "jsonI.php" not in src and "qrng.anu.edu.au" not in src
    assert "x-api-key" in src and "os.environ" in src
    # The key is read in one place, from the environment; no function takes it as an argument.
    import inspect
    assert "key" not in inspect.signature(anu.fetch_anu_bits).parameters
