"""Quantum-safe MCP-style agent-to-agent channel.

Addresses the report's gap: "Agentic AI / Model Context Protocol (MCP) agents accessing
multiple data silos lack a protocol that gives quantum-resistance + privacy
simultaneously" (Security Boulevard 2026.02).

Two agents establish a **PQC (+optional QKD) hybrid session** (:mod:`~qsh.pqc.transport`)
and exchange **ML-DSA-signed**, **AES-256-GCM-sealed** MCP-style tool-call messages.
This gives, in one protocol: confidentiality (AEAD over a quantum-safe key),
authenticity/non-repudiation (post-quantum signature), and HNDL resistance (ML-KEM +
optional QKD key material) for agent↔silo traffic — with no trusted third party.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .hybrid_kem import HybridKEMRecipient
from .signatures import SignatureKeypair, generate_keypair, verify
from .transport import establish_initiator, establish_responder


@dataclass
class Agent:
    """An MCP agent identity: a hybrid-KEM endpoint + a PQC signing key."""

    name: str
    kem: HybridKEMRecipient
    signer: SignatureKeypair

    @classmethod
    def create(cls, name: str) -> "Agent":
        return cls(name=name, kem=HybridKEMRecipient(), signer=generate_keypair())

    @property
    def sig_public(self) -> bytes:
        return self.signer.public_key

    def free(self) -> None:
        self.kem.free()
        self.signer.free()


def _wrap(signer: SignatureKeypair, method: str, params: dict) -> bytes:
    """Build + sign an MCP-style JSON-RPC tool-call message."""
    msg = {"jsonrpc": "2.0", "method": method, "params": params}
    body = json.dumps(msg, sort_keys=True).encode()
    sig = signer.sign(body)
    return json.dumps({"body": body.decode(), "sig": sig.hex(),
                       "alg": signer.sig_name}).encode()


def _unwrap(raw: bytes, sender_sig_public: bytes) -> tuple[dict, bool]:
    """Verify the signature and return (message, authenticated)."""
    env = json.loads(raw.decode())
    body = env["body"].encode()
    ok = verify(body, bytes.fromhex(env["sig"]), sender_sig_public, env["alg"])
    return json.loads(body.decode()), bool(ok)


# The two channels share one AES-GCM key, so the response MUST use a nonce range disjoint from the
# request to avoid GCM nonce reuse. The initiator seals from 0 upward; the responder seals its reply from
# this high base, so the (key, nonce) pairs never collide.
_RESPONDER_NONCE_BASE = 1 << 40


def secure_agent_call(
    caller: Agent, callee: Agent, method: str, params: dict, *,
    qkd_key: bytes | None = None, dispatch: bool = True,
) -> dict:
    """Caller sends a signed tool call to callee over a PQC+QKD session; callee verifies, **dispatches
    the real MCP tool**, and seals the genuine result back — quantum-safe, authenticated agent↔silo comms
    that actually runs a tool end-to-end (not just a handshake).

    With ``dispatch=True`` and ``method == "tools/call"``, the callee runs the named tool from the real
    MCP tool registry (:mod:`qsh.mcp.server`) and returns its result, signed (ML-DSA) + sealed (AES-GCM)
    over the same quantum-safe channel; the caller verifies the signature on the way back.
    """
    # 1. quantum-safe session (ML-KEM-768 + X25519, optionally mixed with a QKD key)
    ct, chan_caller = establish_initiator(callee.kem.public_key, qkd_key=qkd_key)
    chan_callee = establish_responder(callee.kem, ct, qkd_key=qkd_key)

    # 2. caller signs (ML-DSA) + seals (AES-GCM) an MCP tool call
    record = chan_caller.seal(_wrap(caller.signer, method, params), associated_data=b"mcp")

    # 3. callee opens + verifies the post-quantum signature
    opened = chan_callee.open(record, associated_data=b"mcp")
    message, authentic = _unwrap(opened, caller.sig_public)

    # 4. tamper check: flip a ciphertext bit -> AEAD must reject
    tampered_rejected = False
    bad = bytearray(record)
    bad[-1] ^= 0x01
    try:
        chan_callee.open(bytes(bad), associated_data=b"mcp")
    except Exception:
        tampered_rejected = True

    # 5. callee DISPATCHES the real tool + seals the signed result back (disjoint nonce range)
    tool = None
    result: dict | None = None
    result_verified = False
    if dispatch and authentic and message.get("method") == "tools/call":
        tool = message["params"].get("name")
        args = message["params"].get("arguments", {})
        try:
            from ..mcp.server import dispatch_tool
            real = dispatch_tool(tool, args)
        except Exception as exc:                       # surface the cause; the channel itself still verified
            real = {"error": str(exc)}
        chan_callee._counter = _RESPONDER_NONCE_BASE   # responder reply uses the upper nonce range
        reply = chan_callee.seal(_wrap(callee.signer, "tools/result", {"name": tool, "result": real}),
                                 associated_data=b"mcp")
        reply_msg, result_verified = _unwrap(chan_caller.open(reply, associated_data=b"mcp"), callee.sig_public)
        result = reply_msg["params"]["result"]

    return {
        "session": f"{caller.name}->{callee.name}",
        "kem": callee.kem.public_key.kem_name,
        "qkd_mixed": qkd_key is not None,
        "method": message.get("method"),
        "tool": tool,
        "authenticated": authentic,
        "signature_alg": caller.signer.sig_name,
        "tamper_rejected": tampered_rejected,
        "result": result,                              # the REAL tool result, end-to-end over the channel
        "result_verified": result_verified,            # callee's ML-DSA signature on the result verified
    }


def secure_carry(payload: dict, *, qkd_key: bytes | None = None,
                 caller_name: str = "agent", callee_name: str = "external-silo") -> dict:
    """Carry an arbitrary payload (e.g. a result from an EXTERNAL MCP server) under the quantum-safe
    channel: sign (ML-DSA) + seal (AES-GCM) over a PQC(+QKD) session, then open + verify. Returns the
    payload plus whether it was authenticated and whether tampering is rejected — used by the
    filesystem-MCP interop to prove a real external tool result rides the quantum-safe layer.
    """
    caller, callee = Agent.create(caller_name), Agent.create(callee_name)
    try:
        ct, chan_caller = establish_initiator(callee.kem.public_key, qkd_key=qkd_key)
        chan_callee = establish_responder(callee.kem, ct, qkd_key=qkd_key)
        record = chan_caller.seal(_wrap(caller.signer, "mcp/result", payload), associated_data=b"mcp")
        message, authentic = _unwrap(chan_callee.open(record, associated_data=b"mcp"), caller.sig_public)
        tampered_rejected = False
        bad = bytearray(record)
        bad[-1] ^= 0x01
        try:
            chan_callee.open(bytes(bad), associated_data=b"mcp")
        except Exception:
            tampered_rejected = True
        return {
            "kem": callee.kem.public_key.kem_name, "qkd_mixed": qkd_key is not None,
            "authenticated": authentic, "signature_alg": caller.signer.sig_name,
            "tamper_rejected": tampered_rejected, "sealed_bytes": len(record),
            "payload": message.get("params"),
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
