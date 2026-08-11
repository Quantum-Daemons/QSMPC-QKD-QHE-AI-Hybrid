"""Classical homomorphic-encryption backends (CKKS) -- commercial-OK only.

A thin, uniform interface over CKKS FHE so the rest of the codebase (private
inference, the MPC-FHE benchmark) is backend-agnostic:

  * :class:`TenSEALBackend`  -- CPU CKKS via Microsoft SEAL (Apache-2.0; always available, CI default).
  * :class:`OpenFHEBackend`  -- CPU CKKS via OpenFHE (BSD-2-Clause; commercial-OK; supports
    bootstrapping). Live where a compatible libstdc++ is present (Ubuntu 24.04 / the Docker
    image); import-guarded elsewhere.
  * :class:`HeongpuBackend`  -- GPU CKKS via HEonGPU (Apache-2.0). No Python binding, so this is a
    build-gated reference: the GPU FHE numbers come from the benchmark glue (gpu_mpc_adapter).

DESILO's ``desilofhe`` was removed: its license is **non-commercial**
(https://fhe.desilo.dev/latest/license/), incompatible with this project's commercial intent.

Backends expose: ``encrypt(vector) -> ct``, ``decrypt(ct) -> vector``, ``add``, ``mul_plain``,
``dot_plain`` (encrypted-vector · plaintext-vector). That is enough to evaluate the
linear layers of a small CNN/Transformer under encryption (non-linearities are
polynomial-approximated by the caller).
"""

from __future__ import annotations

from typing import Protocol


class CKKSBackend(Protocol):
    """Minimal CKKS interface used across the project."""

    name: str

    def encrypt(self, vector): ...
    def decrypt(self, ct) -> list[float]: ...
    def add(self, ct_a, ct_b): ...
    def mul_plain(self, ct, plain): ...
    def dot_plain(self, ct, plain_vec) -> object: ...


class TenSEALBackend:
    """CPU CKKS backend (Microsoft SEAL via TenSEAL)."""

    name = "tenseal-ckks-cpu"

    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: tuple[int, ...] = (60, 40, 40, 60),
        global_scale: float = 2 ** 40,
    ) -> None:
        import tenseal as ts

        self._ts = ts
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=list(coeff_mod_bit_sizes),
        )
        ctx.global_scale = global_scale
        ctx.generate_galois_keys()
        self.context = ctx

    def encrypt(self, vector):
        return self._ts.ckks_vector(self.context, list(vector))

    def decrypt(self, ct) -> list[float]:
        return ct.decrypt()

    def add(self, ct_a, ct_b):
        return ct_a + ct_b

    def mul_plain(self, ct, plain):
        return ct * plain

    def dot_plain(self, ct, plain_vec):
        return ct.dot(list(plain_vec))


class OpenFHEBackend:
    """CPU CKKS backend via OpenFHE (BSD-2-Clause, commercial-OK; supports bootstrapping).

    Uses the prebuilt ``openfhe`` Python wheel (v1.5.x, OpenFHE 1.5.1). The wheel is built
    against a recent libstdc++ (Ubuntu 24.04 / Debian trixie -- i.e. the Docker image and the
    demo deploy target), so on those platforms this is a *live* backend; on an older host it
    raises a clear ImportError and callers fall back to TenSEAL.

    OpenFHE's CKKS ``EvalBootstrap`` refreshes ciphertext levels for deeper circuits -- this is
    the commercial-OK replacement for the removed DESILO bootstrapping showcase.
    """

    name = "openfhe-ckks-cpu"

    def __init__(self, mult_depth: int = 4, scale_bits: int = 50,
                 batch_size: int = 128, use_bootstrap: bool = False) -> None:
        # batch_size is a power-of-2 >= the model's feature count; shorter vectors are
        # zero-padded (so EvalSum over all slots still yields the exact dot product).
        try:
            import openfhe as ofhe
        except ImportError as e:  # pragma: no cover - optional dep / libstdc++ too old
            raise ImportError(
                "openfhe not importable here (needs a recent libstdc++, e.g. Ubuntu 24.04 / "
                "the Docker image). `pip install openfhe`. Falls back to TenSEAL. See "
                "requirements-fhe.txt."
            ) from e
        self._ofhe = ofhe
        self.bootstrap_enabled = False
        params = ofhe.CCParamsCKKSRNS()
        params.SetMultiplicativeDepth(mult_depth)
        params.SetScalingModSize(scale_bits)
        params.SetBatchSize(batch_size)
        cc = ofhe.GenCryptoContext(params)
        cc.Enable(ofhe.PKESchemeFeature.PKE)
        cc.Enable(ofhe.PKESchemeFeature.KEYSWITCH)
        cc.Enable(ofhe.PKESchemeFeature.LEVELEDSHE)
        cc.Enable(ofhe.PKESchemeFeature.ADVANCEDSHE)   # EvalSum -> dot product
        keys = cc.KeyGen()
        cc.EvalMultKeyGen(keys.secretKey)
        cc.EvalSumKeyGen(keys.secretKey)
        self._cc = cc
        self._keys = keys
        self._batch = batch_size
        if use_bootstrap:
            try:
                cc.Enable(ofhe.PKESchemeFeature.FHE)
                cc.EvalBootstrapSetup([3, 3])
                cc.EvalBootstrapKeyGen(keys.secretKey, batch_size)
                self.bootstrap_enabled = True
            except Exception:                          # pragma: no cover - bootstrap optional
                self.bootstrap_enabled = False

    def _pack(self, vec):
        return self._cc.MakeCKKSPackedPlaintext(list(map(float, vec)))

    def bootstrap(self, ct):
        """Refresh a ciphertext's level (OpenFHE CKKS EvalBootstrap)."""
        if not self.bootstrap_enabled:
            raise RuntimeError("backend not built with use_bootstrap=True")
        return self._cc.EvalBootstrap(ct)

    def encrypt(self, vector):
        return self._cc.Encrypt(self._keys.publicKey, self._pack(vector))

    def decrypt(self, ct) -> list[float]:
        pt = self._cc.Decrypt(ct, self._keys.secretKey)
        pt.SetLength(self._batch)
        return [float(v) for v in pt.GetRealPackedValue()]

    def add(self, ct_a, ct_b):
        return self._cc.EvalAdd(ct_a, ct_b)

    def mul_plain(self, ct, plain):
        return self._cc.EvalMult(ct, self._pack(plain))

    def dot_plain(self, ct, plain_vec):
        prod = self._cc.EvalMult(ct, self._pack(plain_vec))
        return self._cc.EvalSum(prod, self._batch)     # slot-sum -> dot product in slot 0


class HeongpuBackend:
    """Reference adapter for the HEonGPU C++/CUDA CKKS library (crypto-agility slot).

    HEonGPU has no Python API, so this is a build-gated reference: it points at the
    submodule + build script. Wiring a live backend means a pybind11 glue layer over the
    built ``submodules/HEonGPU`` (the GPU MPC-FHE benchmark uses HEonGPU via its own glue).
    """

    name = "heongpu-ckks-gpu"

    def __init__(self, **_kwargs) -> None:
        from pathlib import Path

        built = Path(__file__).resolve().parents[3] / "submodules" / "HEonGPU" / "build"
        raise NotImplementedError(
            "HeongpuBackend is a reference adapter: HEonGPU exposes no Python API. "
            f"Build it first (`bash scripts/build_gpu_mpc.sh`, expects {built}) and add a "
            "pybind11 glue layer. Use 'tenseal' or 'openfhe' for a live FHE backend."
        )


def get_backend(name: str = "tenseal", **kwargs) -> CKKSBackend:
    """Factory. ``name`` in {tenseal, openfhe, heongpu}. Defaults to the CPU backend.

    All three are commercial-OK (TenSEAL Apache-2.0, OpenFHE BSD-2, HEonGPU Apache-2.0).
    """
    name = name.lower()
    if name in ("tenseal", "cpu", "seal"):
        return TenSEALBackend(**kwargs)
    if name in ("openfhe", "ofhe"):
        from ..config import get_config

        f = get_config().fhe                       # central source for the CKKS context params
        kwargs.setdefault("mult_depth", f.openfhe_mult_depth)
        kwargs.setdefault("scale_bits", f.openfhe_scale_bits)
        kwargs.setdefault("batch_size", f.openfhe_batch_size)
        return OpenFHEBackend(**kwargs)
    if name in ("heongpu", "heongpu-gpu", "gpu"):
        return HeongpuBackend(**kwargs)
    raise ValueError(f"unknown FHE backend: {name!r}")
