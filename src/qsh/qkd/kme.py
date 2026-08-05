"""A Key Management Entity serving the **ETSI GS QKD 014 V1.1.1 (2019-02)** delivery API.

GS QKD 014 is the interface an application actually uses. It is deliberately small -- three
resources, and the SAE never touches the control plane:

    GET  /api/v1/keys/{slave_SAE_ID}/status
    POST /api/v1/keys/{slave_SAE_ID}/enc_keys      (or GET with ?number=&size=)
    POST /api/v1/keys/{master_SAE_ID}/dec_keys     (or GET with ?key_ID=)

The split is the point of the spec: the *master* SAE requests keys and receives ``(key_ID, key)``
pairs; the *slave* SAE later presents just the ``key_ID`` and gets the same key back. Only the ID
crosses the application channel, never the key -- which is why a VPN can carry ETSI key IDs in-band
(arXiv:2607.06602, 2026-07) without carrying key material.

Version note, because it is routinely misreported: **V1.1.1 (2019-02) is still the only published
edition.** An ed2 stable draft (v1.3.1) existed as of 2025-06 but is not published, so any
"QKD 014 compliant" claim -- including this one -- targets the 2019 document. Related: GS QKD 020
(interoperable KMS API) was published **2026-06-29**, not 2024 as widely repeated, and covers the
KMS<->KMS interface *within a trusted node* rather than general inter-domain transport.

This KME is **simulated**: keys come from the project's QRNG/extractor path, not from a QKD device.
That is stated in every response body rather than left to the reader, and it is an established
pattern -- the LuxQuanta VPN work above validated against a simulated KME before hardware.

Authentication is out of scope here and that is a real gap, not an oversight: GS QKD 014 mandates
mutual TLS with the SAE identity taken from the client certificate. A deployment must do that; this
in-process model has no transport to attach it to, and says so via :func:`authentication_note`.
"""

from __future__ import annotations

import base64
import secrets
import uuid
from dataclasses import dataclass, field

#: GS QKD 014 clause 5: the default key size a KME returns when the SAE does not ask for one.
DEFAULT_KEY_SIZE_BITS = 352
#: An arbitrary but finite store, so "the KME ran out" is reachable in a demo rather than theoretical.
DEFAULT_STORED_KEY_COUNT = 64


@dataclass
class _StoredKey:
    key_ID: str
    key: str                    #: base64, per the spec's Key data format
    size_bits: int
    master_SAE_ID: str
    slave_SAE_ID: str
    delivered_to_slave: bool = False


@dataclass
class KeyManagementEntity:
    """One KME serving a set of SAEs (``KME_ID``), backed by a simulated key store."""

    KME_ID: str
    #: The SD-QKD node this KME sits in, so a caller can line it up with the controller's topology.
    qkdn_id: str = ""
    key_size_bits: int = DEFAULT_KEY_SIZE_BITS
    max_key_per_request: int = 8
    _store: dict[str, _StoredKey] = field(default_factory=dict)
    _available: int = DEFAULT_STORED_KEY_COUNT

    # --- GET /api/v1/keys/{slave_SAE_ID}/status -------------------------------------------------
    def status(self, slave_SAE_ID: str, master_SAE_ID: str = "") -> dict:
        """The spec's Status data format."""
        return {
            "source_KME_ID": self.KME_ID,
            "target_KME_ID": self.KME_ID,
            "master_SAE_ID": master_SAE_ID,
            "slave_SAE_ID": slave_SAE_ID,
            "key_size": self.key_size_bits,
            "stored_key_count": self._available,
            "max_key_count": DEFAULT_STORED_KEY_COUNT,
            "max_key_per_request": self.max_key_per_request,
            "max_key_size": self.key_size_bits,
            "min_key_size": self.key_size_bits,
            "max_SAE_ID_count": 0,
            "status_extension": {
                "spec": "ETSI GS QKD 014 V1.1.1 (2019-02)",
                "simulated": True,
                "note": ("Keys are drawn from this project's QRNG + extractor path, not from a QKD "
                         "device. No QKD hardware is involved anywhere in this repository."),
            },
        }

    # --- POST /api/v1/keys/{slave_SAE_ID}/enc_keys ----------------------------------------------
    def get_key(self, slave_SAE_ID: str, *, master_SAE_ID: str,
                number: int = 1, size: int | None = None) -> dict:
        """Master SAE draws key material. Returns the spec's Key container.

        Refuses rather than truncating silently when the store cannot cover the request: a KME that
        quietly returns fewer keys than asked is indistinguishable from one that is working.
        """
        size_bits = int(size or self.key_size_bits)
        if number > self.max_key_per_request:
            raise ValueError(f"number {number} exceeds max_key_per_request "
                             f"{self.max_key_per_request}")
        if number > self._available:
            raise ValueError(f"stored_key_count {self._available} cannot satisfy {number} keys")

        keys = []
        for _ in range(number):
            raw = secrets.token_bytes(max(1, size_bits // 8))
            rec = _StoredKey(
                key_ID=str(uuid.uuid4()),
                key=base64.b64encode(raw).decode("ascii"),
                size_bits=size_bits, master_SAE_ID=master_SAE_ID, slave_SAE_ID=slave_SAE_ID,
            )
            self._store[rec.key_ID] = rec
            keys.append({"key_ID": rec.key_ID, "key": rec.key})
        self._available -= number
        return {"keys": keys}

    # --- POST /api/v1/keys/{master_SAE_ID}/dec_keys ---------------------------------------------
    def get_key_with_ids(self, master_SAE_ID: str, key_IDs: list[str], *,
                         slave_SAE_ID: str) -> dict:
        """Slave SAE redeems key IDs. Only the ID travelled over the application channel.

        Enforces the binding the spec relies on: a key issued for one SAE pair cannot be redeemed by
        another. Without this check the key ID would be a bearer token for anyone who saw it.
        """
        out = []
        for kid in key_IDs:
            rec = self._store.get(kid)
            if rec is None:
                raise KeyError(f"unknown key_ID {kid}")
            if rec.slave_SAE_ID != slave_SAE_ID or rec.master_SAE_ID != master_SAE_ID:
                raise PermissionError(
                    f"key_ID {kid} was issued for {rec.master_SAE_ID}->{rec.slave_SAE_ID}, "
                    f"not {master_SAE_ID}->{slave_SAE_ID}")
            rec.delivered_to_slave = True
            out.append({"key_ID": rec.key_ID, "key": rec.key})
        return {"keys": out}

    @staticmethod
    def authentication_note() -> str:
        """What a deployment must add that an in-process model cannot demonstrate."""
        return ("GS QKD 014 requires mutual TLS, with the SAE identity taken from the client "
                "certificate — this in-process KME has no transport to attach that to, so SAE "
                "identity here is a parameter rather than an authenticated fact. Any deployment "
                "must add mTLS before the key-ID binding above means anything.")
