"""Quantum-safe channel carrying MCP messages between two agents.

Addresses the report's gap: "Agentic AI / Model Context Protocol (MCP) agents accessing
multiple data silos lack a protocol that gives quantum-resistance + privacy
simultaneously" (Security Boulevard 2026.02).

Two agents establish a **PQC (+optional QKD) hybrid session** (:mod:`~qsh.pqc.transport`)
and exchange MCP messages, each **ML-DSA-signed** and **AES-256-GCM-sealed**.
This gives, in one protocol: confidentiality (AEAD over a quantum-safe key),
authenticity/non-repudiation (post-quantum signature), and HNDL resistance (ML-KEM) for
agent-to-silo traffic, with no trusted third party. The optional ``qkd_key`` is whatever the
caller passes: the full-deployment API passes the fixed constant ``b"q" * 32``, which adds no
entropy, so in practice the session's strength is ML-KEM-768 + X25519's.

THE MESSAGES (Round 86). Until Round 86 the channel signed ``{"jsonrpc", "method", "params"}`` with no
``id`` and no ``_meta`` and replied with a request-shaped ``"tools/result"``: JSON-RPC notifications,
not MCP. They now follow MCP revision 2026-07-28: a ``tools/call`` request with an id and the required
per-request ``_meta`` fields, and a JSON-RPC response whose result says ``resultType: "complete"``.

THE TRANSPORT. In the terms of that revision's Transports page, this is a custom transport, in one
process. Establishment: one ML-KEM-768 + X25519 session per request. Framing: one AEAD record per
message (associated data ``b"mcp"``) holding ``{"body", "sig", "alg"}``, where ``body`` is the exact
JSON-RPC text the signature covers. Cancellation: none, since a session carries one request and its
response and runs to completion. It serves ``tools/call`` and nothing else, with no ``tools/list``,
``server/discover``, multi round-trip requests or subscriptions, so it is not a complete MCP
implementation. The complete one is the SDK server in :mod:`qsh.mcp.server`, which does not run over
this channel.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from .. import __version__
from .hybrid_kem import HybridKEMRecipient
from .signatures import SignatureKeypair, generate_keypair, verify
from .transport import SecureChannel, establish_initiator, establish_responder

#: The MCP revision these messages follow: https://modelcontextprotocol.io/specification/2026-07-28/basic
MCP_PROTOCOL_VERSION = "2026-07-28"

_META_PROTOCOL = "io.modelcontextprotocol/protocolVersion"
_META_CAPABILITIES = "io.modelcontextprotocol/clientCapabilities"
_META_CLIENT = "io.modelcontextprotocol/clientInfo"
_META_SERVER = "io.modelcontextprotocol/serverInfo"

#: A session carries one request, so one id does: an id must only differ from the sender's other
#: requests still unanswered.
_REQUEST_ID = 1

_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_UNSUPPORTED_VERSION = -32022


@dataclass
class Agent:
    """An MCP agent identity: a hybrid-KEM endpoint + a PQC signing key."""

    name: str
    kem: HybridKEMRecipient
    signer: SignatureKeypair

    @classmethod
    def create(cls, name: str) -> Agent:
        return cls(name=name, kem=HybridKEMRecipient(), signer=generate_keypair())

    @property
    def sig_public(self) -> bytes:
        return self.signer.public_key

    def free(self) -> None:
        self.kem.free()
        self.signer.free()


def mcp_request(client_name: str, method: str, params: dict, *, version: str = __version__) -> dict:
    """A JSON-RPC request carrying the per-request ``_meta`` fields of MCP 2026-07-28."""
    meta = {
        _META_PROTOCOL: MCP_PROTOCOL_VERSION,
        _META_CAPABILITIES: {},                 # an empty object: no optional client capability
        _META_CLIENT: {"name": client_name, "version": version},
    }
    return {"jsonrpc": "2.0", "id": _REQUEST_ID, "method": method, "params": {**params, "_meta": meta}}


def _result_response(request_id: str | int, server_name: str, result: dict) -> dict:
    meta = {_META_SERVER: {"name": server_name, "version": __version__}}
    return {"jsonrpc": "2.0", "id": request_id,
            "result": {"resultType": "complete", **result, "_meta": meta}}


def mcp_tool_result(request_id: str | int, server_name: str, structured: dict, *,
                    is_error: bool = False) -> dict:
    """A ``tools/call`` result holding ``structured``, with the serialized copy the spec recommends."""
    return _result_response(request_id, server_name, {
        "content": [{"type": "text", "text": json.dumps(structured, sort_keys=True)}],
        "structuredContent": structured,
        "isError": is_error,
    })


def mcp_tool_error(request_id: str | int, server_name: str, text: str) -> dict:
    """A tool that ran and failed: a result with ``isError``, so the model can read why and retry."""
    return _result_response(request_id, server_name, {
        "content": [{"type": "text", "text": text}], "isError": True})


def mcp_protocol_error(request_id: str | int | None, code: int, message: str,
                       data: dict | None = None) -> dict:
    """A JSON-RPC error. The id is left out when the request's own id could not be read."""
    error: dict = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    response: dict = {"jsonrpc": "2.0", "error": error}
    if request_id is not None:
        response["id"] = request_id
    return response


def serve_tool_call(message: dict, server_name: str) -> dict:
    """The response a server owes this request under MCP 2026-07-28.

    A request with no usable id or JSON-RPC version gets -32600. One missing a required ``_meta`` field
    is malformed and MUST get -32602, an unknown tool -32602 too, and an unsupported revision -32022
    with the supported list. A tool that runs and raises is a result with ``isError``, not a protocol
    error.
    """
    request_id = message.get("id")
    if (message.get("jsonrpc") != "2.0" or isinstance(request_id, bool)
            or not isinstance(request_id, (str, int))):
        return mcp_protocol_error(None, _INVALID_REQUEST,
                                  "Not a JSON-RPC 2.0 request with a string or integer id")
    params = message.get("params")
    meta = params.get("_meta") if isinstance(params, dict) else None
    missing = [k for k in (_META_PROTOCOL, _META_CAPABILITIES)
               if not isinstance(meta, dict) or k not in meta]
    if missing:
        return mcp_protocol_error(request_id, _INVALID_PARAMS,
                                  f"Missing required _meta field(s): {', '.join(missing)}")
    if meta[_META_PROTOCOL] != MCP_PROTOCOL_VERSION:
        return mcp_protocol_error(request_id, _UNSUPPORTED_VERSION, "Unsupported protocol version",
                                  {"supported": [MCP_PROTOCOL_VERSION], "requested": meta[_META_PROTOCOL]})
    method = message.get("method")
    if method != "tools/call":
        return mcp_protocol_error(request_id, _METHOD_NOT_FOUND, f"Method not found: {method}")

    from ..mcp.server import TOOL_FUNCS, dispatch_tool

    tool = params.get("name")
    if not isinstance(tool, str) or tool not in TOOL_FUNCS:
        return mcp_protocol_error(request_id, _INVALID_PARAMS, f"Unknown tool: {tool}")
    try:
        return mcp_tool_result(request_id, server_name, dispatch_tool(tool, params.get("arguments")))
    except Exception as exc:                           # surface the cause; the channel itself still verified
        return mcp_tool_error(request_id, server_name, str(exc))


def message_body(message: dict) -> str:
    """One message's text: sorted keys, no spaces, UTF-8 left unescaped.

    The browser's ``stableStringify`` writes the same text for messages of strings, integers, booleans
    and nested objects, which `web/scripts/check_mcp_envelope.mjs` holds it to. A verifier needs no
    canonical form, because ``body`` travels as the string that was signed.
    """
    return json.dumps(message, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sign(signer: SignatureKeypair, message: dict) -> bytes:
    """Sign one JSON-RPC message; the signature covers ``body``, the exact text the peer parses."""
    body = message_body(message).encode()
    sig = signer.sign(body)
    return json.dumps({"body": body.decode(), "sig": sig.hex(),
                       "alg": signer.sig_name}).encode()


def _unwrap(raw: bytes, sender_sig_public: bytes) -> tuple[dict, bool]:
    """Verify the signature and return (message, authenticated)."""
    env = json.loads(raw.decode())
    body = env["body"].encode()
    ok = verify(body, bytes.fromhex(env["sig"]), sender_sig_public, env["alg"])
    return json.loads(body.decode()), bool(ok)


def _rejects_tampering(channel: SecureChannel, record: bytes) -> bool:
    """Flip one ciphertext bit: the AEAD must refuse it."""
    bad = bytearray(record)
    bad[-1] ^= 0x01
    try:
        channel.open(bytes(bad), associated_data=b"mcp")
    except Exception:
        return True
    return False


# The two channels share one AES-GCM key, so the response MUST use a nonce range disjoint from the
# request to avoid GCM nonce reuse. The initiator seals from 0 upward; the responder seals its reply from
# this high base, so the (key, nonce) pairs never collide.
_RESPONDER_NONCE_BASE = 1 << 40


@dataclass
class _Exchange:
    """One request and its response over one session, as each side opened and verified them."""

    request: dict
    authentic: bool
    response: dict | None
    response_verified: bool
    tamper_rejected: bool
    response_bytes: int


def _exchange(caller: Agent, callee: Agent, method: str, params: dict,
              respond: Callable[[dict], dict | None], *, qkd_key: bytes | None) -> _Exchange:
    """Session, then the caller's signed + sealed request, then the callee's signed + sealed response.

    ``respond`` runs only on a request whose signature verified; returning ``None`` sends no response.
    """
    # 1. quantum-safe session (ML-KEM-768 + X25519, optionally mixed with a QKD key)
    ct, chan_caller = establish_initiator(callee.kem.public_key, qkd_key=qkd_key)
    chan_callee = establish_responder(callee.kem, ct, qkd_key=qkd_key)
    chan_caller._limit = _RESPONDER_NONCE_BASE     # the caller may never reach the reply's range

    # 2. caller signs (ML-DSA) + seals (AES-GCM) an MCP request
    record = chan_caller.seal(_sign(caller.signer, mcp_request(caller.name, method, params)),
                              associated_data=b"mcp")

    # 3. callee opens + verifies the post-quantum signature, and the AEAD refuses a flipped bit
    request, authentic = _unwrap(chan_callee.open(record, associated_data=b"mcp"), caller.sig_public)
    tamper_rejected = _rejects_tampering(chan_callee, record)

    # 4. callee answers, signed + sealed the same way (disjoint nonce range)
    response, verified, response_bytes = None, False, 0
    answer = respond(request) if authentic else None
    if answer is not None:
        chan_callee._counter = _RESPONDER_NONCE_BASE   # responder reply uses the upper nonce range
        sealed = chan_callee.seal(_sign(callee.signer, answer), associated_data=b"mcp")
        response, verified = _unwrap(chan_caller.open(sealed, associated_data=b"mcp"), callee.sig_public)
        tamper_rejected = tamper_rejected and _rejects_tampering(chan_caller, sealed)
        response_bytes = len(sealed)
    return _Exchange(request, authentic, response, verified, tamper_rejected, response_bytes)


def _outcome(response: dict | None) -> dict | None:
    """What a response says, for the API: the tool's result, or ``{"error": ...}`` when it failed."""
    if response is None:
        return None
    if "error" in response:
        return {"error": response["error"]["message"], "code": response["error"]["code"]}
    result = response["result"]
    if result.get("isError"):
        return {"error": "\n".join(c.get("text", "") for c in result.get("content", []))}
    return result["structuredContent"]


def secure_agent_call(
    caller: Agent, callee: Agent, method: str, params: dict, *,
    qkd_key: bytes | None = None, dispatch: bool = True,
) -> dict:
    """Caller sends a signed tool call to callee over a PQC session (``qkd_key`` mixed in if
    given); callee verifies, **dispatches the real MCP tool**, and seals the genuine result back:
    quantum-safe, authenticated agent-to-silo comms
    that actually runs a tool end-to-end (not just a handshake).

    With ``dispatch=True``, the callee answers the request with :func:`serve_tool_call`: it runs the
    named tool from the real MCP tool registry (:mod:`qsh.mcp.server`) in this process and returns its
    result as an MCP response, signed (ML-DSA) + sealed (AES-GCM) over the same quantum-safe channel;
    the caller verifies the signature on the way back. ``dispatch=False`` stops after the callee has
    verified the request.
    """
    respond = (lambda request: serve_tool_call(request, callee.name)) if dispatch else (lambda request: None)
    ex = _exchange(caller, callee, method, params, respond, qkd_key=qkd_key)
    served = ex.response is not None and "error" not in ex.response
    return {
        "session": f"{caller.name}->{callee.name}",
        "kem": f"{callee.kem.public_key.kem_name}+X25519",   # both halves of the hybrid, as the browser reports
        "qkd_mixed": qkd_key is not None,
        "protocol_version": ex.request["params"]["_meta"][_META_PROTOCOL],
        "request_id": ex.request["id"],
        "method": ex.request.get("method"),
        "tool": ex.request["params"].get("name") if served else None,
        "authenticated": ex.authentic,
        "signature_alg": caller.signer.sig_name,
        "tamper_rejected": ex.tamper_rejected,
        "result": _outcome(ex.response),               # the REAL tool result, end-to-end over the channel
        "result_verified": ex.response_verified,       # callee's ML-DSA signature on the result verified
    }


def secure_carry(payload: dict, *, tool: str = "external_tool", arguments: dict | None = None,
                 qkd_key: bytes | None = None, caller_name: str = "agent",
                 callee_name: str = "external-silo") -> dict:
    """Carry a result from an EXTERNAL MCP server under the quantum-safe channel, as one MCP exchange.

    The external server was reached over its own stdio transport and knows nothing of this channel.
    Here the caller seals the same ``tools/call``, and an identity standing in for that server (created
    for this call, in this process) seals back a result holding what the server said. Each message is
    opened and verified. Returns the payload plus whether both messages were authenticated and whether
    tampering with either is rejected: used by the filesystem-MCP interop to show a real external tool
    result riding the quantum-safe layer, not to show the external server signed anything.
    """
    caller, callee = Agent.create(caller_name), Agent.create(callee_name)
    try:
        ex = _exchange(caller, callee, "tools/call", {"name": tool, "arguments": arguments or {}},
                       lambda request: mcp_tool_result(request["id"], callee.name, payload,
                                                       is_error=bool(payload.get("is_error"))),
                       qkd_key=qkd_key)
        return {
            "kem": f"{callee.kem.public_key.kem_name}+X25519", "qkd_mixed": qkd_key is not None,
            "protocol_version": ex.request["params"]["_meta"][_META_PROTOCOL],
            "authenticated": ex.authentic and ex.response_verified, "signature_alg": caller.signer.sig_name,
            "tamper_rejected": ex.tamper_rejected, "sealed_bytes": ex.response_bytes,
            "payload": ex.response["result"]["structuredContent"] if ex.response else None,
        }
    finally:
        caller.free()
        callee.free()


def mcp_demo(qkd_key: bytes | None = None) -> dict:
    """Self-contained demo: two agents exchange a signed, quantum-safe MCP tool call."""
    a = Agent.create("analyst-agent")
    b = Agent.create("hospital-silo")
    try:
        return secure_agent_call(
            a, b, "tools/call",
            {"name": "query_cohort", "arguments": {"phenotype": "rare-disease-X"}},
            qkd_key=qkd_key,
        )
    finally:
        a.free()
        b.free()
