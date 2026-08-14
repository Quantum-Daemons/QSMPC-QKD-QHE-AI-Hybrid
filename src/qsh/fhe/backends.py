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
        poly_modulus_degree: int | None = None,
        coeff_mod_bit_sizes: tuple[int, ...] | None = None,
        global_scale: float | None = None,
    ) -> None:
        # From the config, not from literals repeated in this signature.
        #
        # These were three default ARGUMENTS holding 8192, (60, 40, 40, 60) and 2**40 - the same
        # values as :class:`FHEConfig`, which nothing read. ``FHEConfig.poly_modulus_degree`` had no
        # reader anywhere in ``src/``: the value that actually built a context was this signature,
        # the value the browser ran was a literal in ``web/src/lib/clientFHE.ts``, and the value the
        # report printed was a literal in ``report/build_data.py``. Three live copies and one dead
        # field, agreeing only because nobody had yet changed one. Raising the ring degree is what
        # exposed that, so the wiring lands before the change rather than after it.
        from ..config import get_config

        fhe = get_config().fhe
        poly_modulus_degree = poly_modulus_degree or fhe.poly_modulus_degree
        coeff_mod_bit_sizes = coeff_mod_bit_sizes or fhe.coeff_mod_bit_sizes
        global_scale = global_scale or 2 ** fhe.global_scale_bits

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
    the commercial-OK replacement for the removed DESILO bootstrapping showcase. Pass
    ``use_bootstrap=True`` to get it; that reconfigures the context (see ``__init__``) and is
    measurably more expensive, so it is off by default.

    Measured with the defaults on the EL9 dev host (OpenFHE 1.5.1, 20-core, ring dimension
    65536 at the standard 128-bit security level): context + key generation 3.0 s, one
    ``EvalBootstrap`` 7.7-9.1 s.

    Bootstrapping is *approximate*, and the error is not reproducible run to run - CKKS
    encryption is randomised, so each bootstrap draws fresh noise. Across four runs of
    ``tests/_openfhe_bootstrap_probe.py`` the same input 0.25 came back as 0.24677, 0.25208,
    0.24918 and 0.25012, and the largest error over all three probe values ranged 8.2e-4 to
    4.5e-3 - between 2.3 and 3.1 correct decimal digits. Stated as a range because an earlier
    version of this docstring quoted one draw ("returns 0.25 as 0.2515") as though it
    characterised the operation; it was a real measurement, and repeating the run does not
    reproduce it. Callers must compare with a tolerance, and a tolerance tighter than about
    5e-3 will be flaky rather than strict.
    """

    name = "openfhe-ckks-cpu"

    def __init__(self, mult_depth: int = 4, scale_bits: int = 50,
                 batch_size: int = 128, use_bootstrap: bool = False,
                 bootstrap_level_budget: tuple[int, int] = (3, 3)) -> None:
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
        params.SetScalingModSize(scale_bits)
        params.SetBatchSize(batch_size)
        # Bootstrapping is NOT a switch you flip on a finished context: both the depth and the
        # scaling technique have to be chosen before GenCryptoContext, and getting either wrong
        # is not a clean error.
        #   depth  - EvalBootstrap *consumes* levels (GetBootstrapDepth([3,3], UNIFORM_TERNARY)
        #            = 20 here), so the context needs those on top of the `mult_depth` levels the
        #            caller wants to compute with afterwards. Calling EvalBootstrapSetup on a
        #            context that lacks them walks off the end of the modulus chain and
        #            SIGSEGVs the interpreter -- no Python exception, nothing to catch.
        #   scaling- EvalBootstrap rejects FIXEDAUTO (the CKKS default) and FLEXIBLEAUTOEXT with
        #            "Degree [10] must be less than or equal to the correction factor [9]".
        #            FLEXIBLEAUTO is what OpenFHE's own bootstrapping examples use.
        # Both are set only on the bootstrap path, so the plain CKKS context is untouched.
        if use_bootstrap:
            skd = ofhe.SecretKeyDist.UNIFORM_TERNARY
            params.SetSecretKeyDist(skd)
            params.SetScalingTechnique(ofhe.ScalingTechnique.FLEXIBLEAUTO)
            boot_depth = ofhe.FHECKKSRNS.GetBootstrapDepth(list(bootstrap_level_budget), skd)
            total_depth = mult_depth + boot_depth
        else:
            total_depth = mult_depth
        params.SetMultiplicativeDepth(total_depth)
        cc = ofhe.GenCryptoContext(params)
        cc.Enable(ofhe.PKESchemeFeature.PKE)
        cc.Enable(ofhe.PKESchemeFeature.KEYSWITCH)
        cc.Enable(ofhe.PKESchemeFeature.LEVELEDSHE)
        cc.Enable(ofhe.PKESchemeFeature.ADVANCEDSHE)   # EvalSum -> dot product
        if use_bootstrap:
            cc.Enable(ofhe.PKESchemeFeature.FHE)
            cc.EvalBootstrapSetup(list(bootstrap_level_budget), [0, 0], batch_size)
        keys = cc.KeyGen()
        cc.EvalMultKeyGen(keys.secretKey)
        cc.EvalSumKeyGen(keys.secretKey)
        self._cc = cc
        self._keys = keys
        self._batch = batch_size
        #: levels the caller asked to be able to compute with (after a bootstrap, if enabled).
        self.mult_depth = mult_depth
        #: levels actually in the modulus chain -- larger than mult_depth when bootstrapping is
        #: on, because EvalBootstrap spends levels of its own. A ciphertext at level
        #: ``total_depth - 1`` has a single RNS tower left and can no longer be bootstrapped.
        self.total_depth = total_depth
        if use_bootstrap:
            # No try/except: a caller that asked for bootstrapping and cannot have it must hear
            # about it. Swallowing this left `bootstrap_enabled = False`, which read to the test
            # suite as "this host has no bootstrapping" and skipped -- so a hard failure in the
            # setup above was indistinguishable from an absent feature. Same no-silent-fallback
            # rule the backend factory follows (docs/crypto-agility.md, requirements-fhe.txt).
            cc.EvalBootstrapKeyGen(keys.secretKey, batch_size)
            self.bootstrap_enabled = True

    def _pack(self, vec):
        return self._cc.MakeCKKSPackedPlaintext(list(map(float, vec)))

    def bootstrap(self, ct):
        """Refresh a ciphertext's level (OpenFHE CKKS EvalBootstrap).

        Returns a ciphertext at level ``total_depth - mult_depth - 1`` regardless of how
        depleted the input was, so this only *gains* levels for an input deeper than that;
        on a fresher ciphertext it is a costly no-op. ``ct`` must have at least two RNS
        towers left (level <= ``total_depth - 2``) -- OpenFHE cannot bootstrap a ciphertext
        on its last tower and raises "Removing last element of DCRTPoly renders it invalid".
        """
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
