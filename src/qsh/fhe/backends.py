"""Classical homomorphic-encryption backends (CKKS) -- commercial-OK only.

A thin, uniform interface over CKKS FHE so the rest of the codebase (private
inference, the MPC-FHE benchmark) is backend-agnostic:

  * :class:`TenSEALBackend`  -- CPU CKKS via Microsoft SEAL (Apache-2.0; always available, CI default).
  * :class:`OpenFHEBackend`  -- CPU CKKS via OpenFHE (BSD-2-Clause; commercial-OK; supports
    bootstrapping). Live where a compatible libstdc++ is present (Ubuntu 24.04 / the Docker
    image); import-guarded elsewhere.
  * :class:`HeongpuBackend`  -- GPU CKKS via HEonGPU (Apache-2.0), through this project's binding
    (`native/heongpu_py`, built by `scripts/build_heongpu.sh`). Live where that module is built and an NVIDIA GPU is present.

DESILO's ``desilofhe`` was removed: its license is **non-commercial**
(https://fhe.desilo.dev/latest/license/), incompatible with this project's commercial intent.

Backends expose: ``encrypt(vector) -> ct``, ``decrypt(ct) -> vector``, ``add``, ``add_plain``
(ciphertext + a plaintext scalar or vector), ``mul_plain``, ``dot_plain`` (encrypted-vector ·
plaintext-vector). That is enough to evaluate the linear layers of a small CNN/Transformer under
encryption. For the hybrid's masked conversions (``qsh.mpc.secret_inference``) they also expose
``slot_count``, ``add_plain_slots`` (a vector over EVERY slot, encrypted under the public key before
it is added) and ``complex_slots`` (whether that vector may have imaginary parts): the server masks
every slot the key holder can decode, not only the one it reads (Round 84).
"""

from __future__ import annotations

import functools
from typing import Protocol

import numpy as np


class CKKSBackend(Protocol):
    """Minimal CKKS interface used across the project."""

    name: str
    #: Slots in one ciphertext: everything a key holder can decode from it.
    slot_count: int
    #: True when ``add_plain_slots`` accepts imaginary parts, so a mask can cover them.
    complex_slots: bool

    def encrypt(self, vector): ...
    def decrypt(self, ct) -> list[float]: ...
    def add(self, ct_a, ct_b): ...
    def add_plain(self, ct, plain): ...
    def add_plain_slots(self, ct, values): ...
    def mul_plain(self, ct, plain): ...
    def dot_plain(self, ct, plain_vec) -> object: ...


class SlotView(list):
    """A decode of EVERY slot: the list holds the real parts, ``imag`` the imaginary parts.

    A list of floats so that a caller reading slot 0 (``view[0]``) is unchanged; the imaginary parts
    ride along because the key holder can always decode them, so whoever accounts for what it sees
    must see them too.
    """

    def __init__(self, values) -> None:
        v = np.asarray(values, dtype=complex)
        super().__init__(v.real.tolist())
        self.imag = v.imag.tolist()


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
        self.slot_count = int(poly_modulus_degree) // 2
        self._seal = None

    #: SEAL's encoder takes complex vectors, so a mask can cover the imaginary parts as well.
    complex_slots = True

    def _sealapi(self):
        """SEAL's own encoder, evaluator, encryptor and decryptor over this context, built once.

        TenSEAL's vector API cannot reach every slot: after ``dot`` a vector has logical size 1, so
        a scalar is broadcast to all 4096 slots and a list must have length 1. SEAL's API can.
        """
        if self._seal is None:
            import tenseal.sealapi as sa

            sc = self.context.seal_context().data
            self._seal = (sa, sa.CKKSEncoder(sc), sa.Evaluator(sc),
                          sa.Encryptor(sc, self.context.public_key().data),
                          sa.Decryptor(sc, self.context.secret_key().data))
        return self._seal

    @staticmethod
    def _raw(ct):
        # A CKKSVector hands out a COPY of its SEAL ciphertext; a raw one is used as it is.
        return ct.ciphertext()[0] if hasattr(ct, "ciphertext") else ct

    def encrypt(self, vector):
        return self._ts.ckks_vector(self.context, list(vector))

    def decrypt(self, ct) -> list[float]:
        """A vector's logical slots; for a ciphertext from ``add_plain_slots``, EVERY slot.

        The second is the key holder's whole decode (a :class:`SlotView`, real parts first), which is
        what it can always compute from a ciphertext it holds the key of.
        """
        if hasattr(ct, "decrypt"):
            return ct.decrypt()
        return SlotView(self.decrypt_slots(ct))

    def decrypt_slots(self, ct) -> list[complex]:
        """Every slot of ``ct`` as a complex number, whatever logical size the vector claims."""
        sa, enc, _, _, dec = self._sealapi()
        pt = sa.Plaintext()
        dec.decrypt(self._raw(ct), pt)
        return enc.decode_complex(pt)

    def add(self, ct_a, ct_b):
        return ct_a + ct_b

    def add_plain(self, ct, plain):
        """``ct`` plus a plaintext scalar (every slot) or a vector of ``ct``'s length."""
        return ct + (float(plain) if isinstance(plain, (int, float)) else [float(v) for v in plain])

    def add_plain_slots(self, ct, values):
        """``ct`` plus a vector over EVERY slot (complex allowed), encrypted first; a SEAL ciphertext.

        Encrypted under the public key, not added as a plaintext, because a plaintext addition
        changes only a ciphertext's first component: the second would reach the key holder exactly
        as the server's arithmetic on the weights left it. A fresh encryption re-randomises it.
        """
        sa, enc, ev, encryptor, _ = self._sealapi()
        raw = self._raw(ct)
        pt, mask, out = sa.Plaintext(), sa.Ciphertext(), sa.Ciphertext()
        enc.encode(np.asarray(values, dtype=complex).tolist(), raw.parms_id(), raw.scale, pt)
        encryptor.encrypt(pt, mask)
        ev.add(raw, mask, out)
        return out

    def mul_plain(self, ct, plain):
        return ct * plain

    def dot_plain(self, ct, plain_vec):
        return ct.dot(list(plain_vec))


class OpenFHEBackend:
    """CPU CKKS backend via OpenFHE (BSD-2-Clause, commercial-OK; supports bootstrapping).

    Uses the prebuilt ``openfhe`` Python wheel (OpenFHE 1.6.1). The wheel is built
    against a recent libstdc++ (Ubuntu 24.04 / Debian trixie -- i.e. the Docker image and the
    demo deploy target), so on those platforms this is a *live* backend; on an older host it
    raises a clear ImportError and callers fall back to TenSEAL.

    OpenFHE's CKKS ``EvalBootstrap`` refreshes ciphertext levels for deeper circuits -- this is
    the commercial-OK replacement for the removed DESILO bootstrapping showcase. Pass
    ``use_bootstrap=True`` to get it; that reconfigures the context (see ``__init__``) and is
    measurably more expensive, so it is off by default.

    Measured with the defaults on the EL9 dev host (OpenFHE 1.6.1 built for EL9 by
    env/build_openfhe_el9.sh, 20 hardware threads, ring dimension 65536 at the standard 128-bit
    security level), four runs on 2026-11-06: context + key generation 1.7-1.8 s, one
    ``EvalBootstrap`` 6.2-7.7 s. On OpenFHE 1.5.1 the same measurement read 3.0 s and 7.7-9.1 s.

    Bootstrapping is *approximate*, and the error is not reproducible run to run - CKKS
    encryption is randomised, so each bootstrap draws fresh noise. Across four runs of
    ``tests/_openfhe_bootstrap_probe.py`` on OpenFHE 1.6.1 the same input 0.25 came back as 0.24984,
    0.25123, 0.25026 and 0.25097, and the largest error over all three probe values ranged 2.0e-3 to
    3.8e-3 - between 2.4 and 2.7 correct decimal digits (on 1.5.1, 8.2e-4 to 4.5e-3, 2.3 to 3.1
    digits). Stated as a range because an earlier
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
            # On `CryptoContext` since OpenFHE 1.6 (openfhe-python v1.6.1.0, src/lib/pke/fbt.cpp); the 1.5 bindings
            # had it on `FHECKKSRNS`, which 1.6 no longer exports. requirements-fhe.txt pins >= 1.6.1.
            boot_depth = ofhe.CryptoContext.GetBootstrapDepth(list(bootstrap_level_budget), skd)
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
        #: The batch is what this context packs and decodes; ``add_plain_slots`` masks all of it.
        self.slot_count = int(batch_size)
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

    def add_plain(self, ct, plain):
        """``ct`` plus a plaintext scalar (every slot) or a packed vector."""
        if isinstance(plain, (int, float)):
            return self._cc.EvalAdd(ct, float(plain))
        return self._cc.EvalAdd(ct, self._pack(plain))

    #: The context is REAL-typed (OpenFHE's default): its decoder reads the imaginary parts as the
    #: approximation error and floods the real parts to match, so a mask there would wreck the
    #: decryption. The hybrid reports that the imaginary parts went unflooded on this backend.
    complex_slots = False

    def add_plain_slots(self, ct, values):
        """``ct`` plus a real vector over every batch slot, encrypted under the public key first.

        Encrypted rather than added as a plaintext so the result is re-randomised, as on TenSEAL.
        """
        v = np.asarray(values, dtype=complex)
        if np.any(v.imag != 0):
            raise ValueError("this OpenFHE context is REAL-typed: a slot vector cannot carry imaginary parts")
        return self._cc.EvalAdd(ct, self._cc.Encrypt(self._keys.publicKey, self._pack(v.real)))

    def mul_plain(self, ct, plain):
        return self._cc.EvalMult(ct, self._pack(plain))

    def dot_plain(self, ct, plain_vec):
        prod = self._cc.EvalMult(ct, self._pack(plain_vec))
        return self._cc.EvalSum(prod, self._batch)     # slot-sum -> dot product in slot 0


class _GpuCiphertext:
    """A HEonGPU ciphertext with the logical length a decrypt returns, which the GPU object does not carry."""

    __slots__ = ("raw", "n")

    def __init__(self, raw, n: int) -> None:
        self.raw, self.n = raw, n


class HeongpuBackend:
    """GPU CKKS backend: HEonGPU through this project's binding (`native/heongpu_py`, Round 86).

    HEonGPU has no Python API of its own. `scripts/build_heongpu.sh` builds the pinned submodule and a pybind11 module over the
    seven operations the protocol above needs; the module is found in `native/heongpu_py/build` or in `QSH_HEONGPU_BUILD`. The
    ring and moduli come from `FHEConfig`, as TenSEAL's do, at 128-bit security (`sec_level_type::sec128`; HEonGPU's own
    benchmarks run with `none`). GPU work is asynchronous, so a timing needs :meth:`synchronize` before the clock is read.
    """

    name = "heongpu-ckks-gpu"
    #: The encoder takes complex vectors, so a mask can cover the imaginary parts as well.
    complex_slots = True
    #: Read by the benchmark, which reports a GPU row as such.
    on_gpu = True

    def __init__(self, poly_modulus_degree: int | None = None, coeff_mod_bit_sizes: tuple[int, ...] | None = None,
                 global_scale_bits: int | None = None) -> None:
        import os
        import sys
        from pathlib import Path

        from ..config import get_config

        build = Path(os.environ.get("QSH_HEONGPU_BUILD", Path(__file__).resolve().parents[3] / "native" / "heongpu_py" / "build"))
        if str(build) not in sys.path:
            sys.path.insert(0, str(build))
        try:
            import qsh_heongpu
        except ImportError as e:
            raise RuntimeError(
                f"the HEonGPU module is not built ({e}); run `bash scripts/build_heongpu.sh`, which needs CUDA 12.8 or later, "
                "CMake 4 and an NVIDIA GPU. Use 'tenseal' or 'openfhe' without one.") from e
        fhe = get_config().fhe
        bits = tuple(coeff_mod_bit_sizes or fhe.coeff_mod_bit_sizes)
        # The last modulus is the special prime key switching uses, as in SEAL's (60, 40, 40, 60).
        self._b = qsh_heongpu.Backend(int(poly_modulus_degree or fhe.poly_modulus_degree), list(bits[:-1]), [bits[-1]],
                                      int(global_scale_bits or fhe.global_scale_bits))
        self.slot_count = int(self._b.slot_count)

    def synchronize(self) -> None:
        self._b.synchronize()

    @property
    def rotations(self) -> int:
        return int(self._b.rotations)

    @staticmethod
    def _reals(plain, n: int) -> np.ndarray:
        """A plaintext operand as the module takes it: a scalar fills the ciphertext's logical slots."""
        return np.full(n, float(plain)) if np.ndim(plain) == 0 else np.asarray(plain, dtype=float).ravel()

    def encrypt(self, vector):
        v = np.asarray(vector, dtype=float).ravel()
        return _GpuCiphertext(self._b.encrypt(v), v.size)

    def decrypt(self, ct) -> list[float]:
        """The logical slots; for a ciphertext from ``add_plain_slots``, EVERY slot, as TenSEALBackend does."""
        slots = self._b.decrypt_slots(ct.raw)
        if ct.n == self.slot_count:
            return SlotView(slots)
        return slots[:ct.n].real.tolist()

    def decrypt_slots(self, ct) -> list[complex]:
        return self._b.decrypt_slots(ct.raw).tolist()

    def add(self, ct_a, ct_b):
        return _GpuCiphertext(self._b.add(ct_a.raw, ct_b.raw), max(ct_a.n, ct_b.n))

    def add_plain(self, ct, plain):
        return _GpuCiphertext(self._b.add_plain(ct.raw, self._reals(plain, ct.n)), ct.n)

    def add_plain_slots(self, ct, values):
        """``ct`` plus a vector over EVERY slot, encrypted under the public key first (see TenSEALBackend)."""
        masked = self._b.add_plain_slots(ct.raw, np.asarray(values, dtype=complex).ravel())
        return _GpuCiphertext(masked, self.slot_count)

    def mul_plain(self, ct, plain):
        return _GpuCiphertext(self._b.mul_plain(ct.raw, self._reals(plain, ct.n)), ct.n)

    def dot_plain(self, ct, plain_vec):
        """The dot product in slot 0: a plaintext multiply, a rescale and log2 rotations (rounded up)."""
        return _GpuCiphertext(self._b.dot_plain(ct.raw, np.asarray(plain_vec, dtype=float).ravel()), 1)

@functools.lru_cache(maxsize=1)
def heongpu_available() -> bool:
    """True where the HEonGPU module is built and loads, so a selector offers only what can run here.

    The probe builds a backend at the configured ring, keys included, so it is made once per process: every pipeline
    builds a registry, and each registry asks.
    """
    try:
        HeongpuBackend()
    except Exception:
        return False
    return True


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
