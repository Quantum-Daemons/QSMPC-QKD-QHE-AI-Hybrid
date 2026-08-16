"""Crypto-agility registry.

Definition used here (EU NIS Cooperation Group, *A Coordinated Implementation Roadmap for
the Transition to Post-Quantum Cryptography*, 23 June 2025, section 3.2):

    "cryptographic agility (or crypto-agility): The design of cryptographic protocols and
    systems in a modular way that enables replacing the cryptographic components. This
    concept must not be confused with a requirement to negotiate the cipher suite during
    protocol execution; the latter often leads to downgrade attacks if not done very
    carefully."

This registry implements the first half of that definition and deliberately does **not**
implement the second. It maps a *family* (kem, qkd, fhe, mpc, sig) to named providers so a
pipeline can be built with ML-KEM<->hybrid, QKD<->PQC-only, TenSEAL<->OpenFHE (the
linear-FHE half of the hybrid), or MPyC<->MP-SPDZ (the non-linear-MPC half) without touching
stage code -- so a future PQC migration, or the withdrawal of a broken primitive, is a
configuration change rather than a rewrite.

Selection is **configuration-time and operator-driven**, never negotiated:

* the provider set is resolved from configuration / the operator-facing API
  (``GET /api/crypto-options``) and pinned into the run config *before* a run starts;
* there is **no in-band cipher-suite negotiation** between parties anywhere in this
  codebase, and therefore no negotiated-downgrade surface;
* there is **no silent fallback** to a weaker provider: if a selected provider is
  unavailable the run fails loudly (see the repository's no-easy-fallback policy).

Note on the ``qkd`` family: ``pqc-only`` is a deployment profile, not a fallback. Both
``bb84-decoy`` and ``pqc-only`` keep the PQC layer (hybrid ML-KEM + X25519, ML-DSA)
unchanged, so switching between them changes whether QKD material is mixed in -- it does not
weaken the post-quantum guarantee, and a peer cannot induce the switch.

The word "negotiation" does appear in this project as ``qkd_negotiation_ms``: that is a
timing label for the BB84 *key-agreement* step and is unrelated to cipher-suite negotiation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class CryptoRegistry:
    """A registry of pluggable crypto providers, grouped by family."""

    _providers: dict[str, dict[str, Callable]] = field(default_factory=dict)
    _defaults: dict[str, str] = field(default_factory=dict)

    def register(self, family: str, name: str, factory: Callable, *, default: bool = False) -> None:
        self._providers.setdefault(family, {})[name] = factory
        if default or family not in self._defaults:
            self._defaults[family] = name

    def get(self, family: str, name: str | None = None):
        fam = self._providers.get(family)
        if not fam:
            raise KeyError(f"no providers registered for family {family!r}")
        chosen = name or self._defaults[family]
        if chosen not in fam:
            raise KeyError(f"provider {chosen!r} not in family {family!r}: {list(fam)}")
        return fam[chosen]

    def options(self, family: str) -> list[str]:
        return list(self._providers.get(family, {}))

    def default(self, family: str) -> str | None:
        return self._defaults.get(family)


def default_registry() -> CryptoRegistry:
    """Registry pre-populated with the project's providers."""
    reg = CryptoRegistry()

    # KEM family: post-quantum hybrid (default) and a classical-only baseline
    from ..pqc.hybrid_kem import HybridKEMRecipient

    reg.register("kem", "ml-kem-768+x25519",
                 lambda: HybridKEMRecipient("ML-KEM-768"), default=True)
    reg.register("kem", "ml-kem-1024+x25519",
                 lambda: HybridKEMRecipient("ML-KEM-1024"))

    # QKD family: BB84 simulation (default) and a "pqc-only" no-QKD mode
    from ..qkd.bb84 import QKDParams, run_bb84

    # `qkd_params=QKDParams()` IS THE REGISTRATION, not an embellishment on it. This entry is named
    # "bb84-decoy" and until Round 46 it ran `run_bb84(**kw)` with no `qkd_params`, which takes the
    # LEGACY branch at `bb84.py`: uniform bases, transmittance used directly as the detection
    # probability, and `1 - h2(q) - f h2(q)` on the observed QBER. That is the non-decoy GLLP form.
    #
    # Round 45 removed exactly that model from the browser, measured the gap at 4.07x and wrote three
    # documents about it - and left this line, so `QKDStage` went on resolving a provider called
    # "bb84-decoy" that computed 0.008936 bit/pulse against the browser's 0.002355. The name was the
    # claim, and the claim was false.
    #
    # Fixed HERE rather than at the call site, deliberately: `stages.py` never names `run_bb84`, it
    # asks the registry for whatever "bb84-decoy" means. One place decides what the name means. The
    # two surfaces that bypass the registry are fixed separately, and both directions are guarded by
    # `tests/test_the_browser_computes_the_bound_the_page_names.py`.
    reg.register("qkd", "bb84-decoy",
                 lambda **kw: run_bb84(qkd_params=QKDParams(), **kw), default=True)
    reg.register("qkd", "pqc-only", lambda **kw: None)

    # FHE family: commercial-OK CKKS only -- TenSEAL CPU (default, Apache-2.0) and
    # OpenFHE CPU (BSD-2-Clause, bootstrapping). DESILO (desilofhe) was removed: its
    # license is non-commercial (https://fhe.desilo.dev/latest/license/). HEonGPU
    # (Apache-2.0) is the GPU FHE backend exercised live by the benchmark glue
    # (gpu_mpc_adapter); it has no Python binding so it is not a per-run selectable backend.
    from ..fhe.backends import get_backend

    reg.register("fhe", "tenseal-cpu", lambda **kw: get_backend("tenseal", **kw), default=True)
    reg.register("fhe", "openfhe-cpu", lambda **kw: get_backend("openfhe", **kw))

    # MPC family: the *non-linear* half of the hybrid (ReLU/argmax under secret sharing).
    # MPyC (MIT, pure-Python Shamir + secure comparison) is the live engine; MP-SPDZ and
    # hpmpc/PIGEON (vendored) are heavyweight production references.
    reg.register("mpc", "mpyc-shamir", lambda **kw: "mpyc-shamir", default=True)
    reg.register("mpc", "mp-spdz", lambda **kw: "mp-spdz")
    reg.register("mpc", "hpmpc-gpu", lambda **kw: "hpmpc-gpu")

    # signature family
    from ..pqc.signatures import generate_keypair

    reg.register("sig", "ml-dsa-65", lambda: generate_keypair("ML-DSA-65"), default=True)
    reg.register("sig", "ml-dsa-87", lambda: generate_keypair("ML-DSA-87"))
    return reg
