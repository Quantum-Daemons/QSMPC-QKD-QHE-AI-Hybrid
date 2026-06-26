"""Crypto-agility registry.

The report makes crypto-agility a first-class requirement: algorithms must be
swappable at runtime (remote re-key, in-flight algorithm substitution), never
hard-coded. This registry maps a *family* (kem, qkd, fhe, mpc, sig) to named providers so
a pipeline can switch ML-KEM<->hybrid, QKD<->PQC-only, TenSEAL<->OpenFHE (the linear-FHE half
of the hybrid), or MPyC<->MP-SPDZ (the non-linear-MPC half) without touching stage code -- and
so a future PQC migration is a config change.
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
    from ..qkd.bb84 import run_bb84

    reg.register("qkd", "bb84-decoy", lambda **kw: run_bb84(**kw), default=True)
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
