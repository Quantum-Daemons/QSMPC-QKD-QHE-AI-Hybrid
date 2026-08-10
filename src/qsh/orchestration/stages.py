"""Pipeline stages — the composable steps of the E2E security layer.

Each stage implements ``run(ctx, bus, reg)`` and mutates a shared :class:`Context`
while emitting events. The canonical chain is:

    PQCHandshake -> QKDKeyAgreement -> QSMPCCompute -> QHE/FHEInference -> Result

Stages read their algorithm choices from the :class:`~qsh.orchestration.crypto_agility.
CryptoRegistry`, so the same chain runs PQC-only or PQC+QKD, TenSEAL or OpenFHE, etc.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .crypto_agility import CryptoRegistry
from .events import EventBus


@dataclass
class Context:
    """Mutable bag of data flowing through the pipeline."""

    usecase: str = "demo"
    config: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)        # inputs (sets, features, ...)
    artifacts: dict = field(default_factory=dict)   # results to surface/download
    metrics: dict = field(default_factory=dict)

    def put(self, key: str, value: Any) -> None:
        self.data[key] = value


class Stage(ABC):
    """Base class for a pipeline stage."""

    name: str = "stage"

    @abstractmethod
    def run(self, ctx: Context, bus: EventBus, reg: CryptoRegistry) -> None: ...


# --------------------------------------------------------------------------- #
class PQCHandshakeStage(Stage):
    """Establish a quantum-safe session with a hybrid KEM (PQC-first)."""

    name = "pqc_handshake"

    def run(self, ctx, bus, reg):
        from ..config import get_config
        from ..pqc.transport import establish_initiator, establish_responder
        from ..qrng import generate_quantum_bits

        # Quantum entropy source (simulated state-vector measurement, not hardware) — health + throughput.
        qc = get_config().qrng
        qr = generate_quantum_bits(
            int(ctx.config.get("qrng_bits", qc.bits)), source="sim",
            post_processing=ctx.config.get("qrng_post", qc.post_processing),
            seed=ctx.config.get("seed", 0), health_alpha=qc.health_alpha,
            health_block_size=int(ctx.config.get("qrng_health_block", qc.health_block_size)),
            epsilon_log2=int(ctx.config.get("qrng_eps_log2", qc.security_epsilon_log2)),
            min_entropy_per_raw_bit=float(ctx.config.get("qrng_min_h", qc.min_entropy_per_raw_bit)))
        bus.metric("qrng_health_passed", 1 if qr.health.passed else 0, self.name)
        bus.metric("qrng_throughput_kbps", round(qr.throughput_bits_per_s / 1000, 1), self.name)
        bus.metric("qrng_min_entropy_per_bit", round(qr.health.min_entropy_per_bit, 4), self.name)
        # Name which test family the verdict comes from: `passed` is the SP 800-90B continuous
        # health test (the required one), not the SP 800-22 p-values, which are an output sanity
        # check and are reported separately.
        bus.metric("qrng_sp80090b_passed", 1 if qr.health.passed else 0, self.name)
        bus.metric("qrng_sp80022_passed", 1 if qr.health.sp80022_passed else 0, self.name)
        bus.log(f"QRNG (simulated quantum measurement): {qr.n_bits} bits, "
                f"SP 800-90B health {'PASS' if qr.health.passed else 'FAIL'} "
                f"(repetition-count max run {qr.health.repetition_count_max_run}/"
                f"{qr.health.repetition_count_cutoff}, adaptive-proportion "
                f"{qr.health.adaptive_proportion_max_count}/{qr.health.adaptive_proportion_cutoff}), "
                f"SP 800-22 output check {'pass' if qr.health.sp80022_passed else 'fail'} "
                f"(monobit p={qr.health.monobit_p:.3f}), "
                f"min-entropy {qr.health.min_entropy_per_bit:.3f} bit/bit, "
                f"extractor={qr.post_processing})", self.name)
        for failure in qr.health.failures:
            bus.log(f"QRNG health failure: {failure}", self.name)

        kem_name = ctx.config.get("kem")
        recipient = reg.get("kem", kem_name)()
        bus.log(f"hybrid KEM = {recipient.kem_name} (+X25519)", self.name)
        ct, chan_a = establish_initiator(recipient.public_key,
                                         qkd_key=ctx.data.get("qkd_key"))
        chan_b = establish_responder(recipient, ct, qkd_key=ctx.data.get("qkd_key"))
        ctx.put("channel_initiator", chan_a)
        ctx.put("channel_responder", chan_b)
        ctx.put("kem_recipient", recipient)
        ctx.put("session_key", chan_a.session_key)
        bus.metric("session_key_bits", len(chan_a.session_key) * 8, self.name)
        bus.log("PQC session established (HNDL-resistant)", self.name)


class QKDKeyAgreementStage(Stage):
    """Run BB84 (or skip in pqc-only mode) and fold the QKD key into the session."""

    name = "qkd_key_agreement"

    def run(self, ctx, bus, reg):
        from ..config import get_config

        cfg = get_config().qkd
        qkd_name = ctx.config.get("qkd", reg.default("qkd"))
        provider = reg.get("qkd", qkd_name)
        if qkd_name == "pqc-only":
            bus.log("QKD disabled (PQC-only profile)", self.name)
            ctx.metrics["qber"] = None
            return
        # config defaults, overridable per-run from the WebUI (ctx.config)
        from ..qkd.channel import ChannelParams

        params = ChannelParams(
            distance_km=float(ctx.config.get("qkd_distance_km", cfg.distance_km)),
            misalignment=float(ctx.config.get("qkd_misalignment", cfg.misalignment)),
            eve_intercept_fraction=float(ctx.config.get("qkd_eve", cfg.eve_intercept_fraction)),
        )
        n_pulses = int(ctx.config.get("qkd_pulses", cfg.pulses))
        # optionally seed BB84's basis/bit choices from the (simulated) quantum RNG
        qrng_on = bool(ctx.config.get("qrng_enabled", get_config().qrng.enabled))
        extra = {"entropy": "qrng"} if qrng_on else {}
        result = provider(n_pulses=n_pulses, params=params, seed=ctx.config.get("seed", 0), **extra)
        bus.metric("qber", round(result.qber, 5), self.name)
        bus.metric("skr", round(result.secret_key_rate, 5), self.name)
        bus.metric("secure_key_bits", result.secure_key_bits, self.name)
        ctx.metrics.update(qber=result.qber, skr=result.secret_key_rate)
        if result.secure_key_bits >= 256:
            from ..qkd.bb84 import derive_shared_key

            ctx.put("qkd_key", derive_shared_key(result, 32))
            bus.log(f"QKD key derived (QBER={result.qber:.4f}); mixed into session", self.name)
        else:
            bus.log("insufficient secure key; continuing PQC-only", self.name)


class OptiQKDStage(Stage):
    """AI-driven QKD parameter optimization (OptiQKD, WS1): a measured, gradient-free + surrogate
    search tunes decoy/basis params to raise SKR vs the default baseline on the same simulated channel.

    Gated by ``qkd_optimize`` (default off); opt-in via the WebUI toggle. Simulation, not hardware.
    """

    name = "qkd_optimize"

    def run(self, ctx, bus, reg):
        from ..config import get_config

        cfg = get_config().qkd
        if not ctx.config.get("qkd_optimize", cfg.optimize_enabled):
            return
        from ..qkd.channel import ChannelParams
        from ..qkd.optimize import optimize_qkd

        # optimize the channel the use case actually used (free-space for satellite), else fiber from config
        channel = ctx.data.get("opt_channel") or ctx.data.get("sat_channel") or ChannelParams(
            distance_km=float(ctx.config.get("qkd_distance_km", cfg.distance_km)),
            misalignment=float(ctx.config.get("qkd_misalignment", cfg.misalignment)),
            eve_intercept_fraction=float(ctx.config.get("qkd_eve", cfg.eve_intercept_fraction)),
        )
        res = optimize_qkd(channel,
                           method=ctx.config.get("qkd_opt_method", cfg.optimize_method),
                           n_samples=int(ctx.config.get("qkd_opt_samples", cfg.optimize_n_samples)),
                           n_pulses=int(ctx.config.get("qkd_opt_pulses", cfg.optimize_n_pulses)),
                           protocol=ctx.config.get("qkd_protocol", cfg.protocol),
                           seed=ctx.config.get("seed", 0))
        bus.metric("qkd_skr_improvement_pct", round(res.improvement_pct, 1), self.name)
        bus.metric("qkd_optimized_skr", round(res.best_skr, 6), self.name)
        bus.metric("qkd_baseline_skr", round(res.baseline_skr, 6), self.name)
        bus.metric("qkd_optimized_qber", round(res.best_qber, 5), self.name)
        bus.metric("qkd_convergence_evals", res.convergence_evals, self.name)
        ctx.artifacts["optiqkd"] = {
            "improvement_pct": round(res.improvement_pct, 1), "method": res.method, "n_evals": res.n_evals,
            "best_params": {"mu_signal": round(res.best_params.mu_signal, 3),
                            "basis_prob": round(res.best_params.basis_prob, 3),
                            "pp_threshold": round(res.best_params.pp_threshold, 3)}}
        bus.log(f"OptiQKD ({res.method}): SKR {res.baseline_skr:.2e}->{res.best_skr:.2e} "
                f"(+{res.improvement_pct:.1f}% vs default; basis p={res.best_params.basis_prob:.2f}, "
                f"mu={res.best_params.mu_signal:.2f}; {res.n_evals} evals — simulated, not hardware)", self.name)


class QSMPCComputeStage(Stage):
    """Quantum-secure multiparty computation: PSI or secure summation."""

    name = "qsmpc_compute"

    def run(self, ctx, bus, reg):
        op = ctx.config.get("qsmpc_op", "psi")
        key = ctx.data.get("session_key", b"\x00" * 32)
        if op == "psi":
            from ..qsmpc.qot import multiparty_psi, private_set_intersection

            sets = ctx.data.get("sets")
            if sets and len(sets) > 2:
                res = multiparty_psi(sets, key)
            else:
                a, b = (sets or (ctx.data.get("set_a", []), ctx.data.get("set_b", [])))[:2] \
                    if sets else (ctx.data.get("set_a", []), ctx.data.get("set_b", []))
                res = private_set_intersection(a, b, key)
            ctx.artifacts["psi_intersection"] = list(res.intersection)
            bus.metric("psi_intersection_size", res.intersection_size, self.name)
            bus.log(f"PSI complete: |A∩B|={res.intersection_size} "
                    f"(sizes {res.size_a},{res.size_b})", self.name)
        elif op == "sum":
            from ..qsmpc.secure_summation import quantum_secure_sum

            from ..config import get_config

            from ..qsmpc.secure_summation import QSMS

            secrets = ctx.data.get("secrets", [])
            # `prime` may legitimately be absent now: the config no longer pins one, because the
            # admissible field depends on the party count (d prime, n <= d <= 2n). QSMS derives it.
            prime = ctx.config.get("prime") or get_config().qsmpc.prime
            n = max(len(secrets), 3)
            res = QSMS(n=n, t=min(3, n), prime=prime).run(secrets, seed=ctx.config.get("seed"))
            total = res.total
            ctx.artifacts["secure_sum"] = total
            # The modulus the sum ACTUALLY used, so a caller can reduce a ground truth into the same
            # field instead of guessing from config -- which broke the moment the config stopped
            # pinning one. Whether the honest GHZ statevector ran is recorded for the same reason:
            # it is a property of this run, not of the code.
            ctx.artifacts["qsms_prime"] = res.modulus
            ctx.artifacts["qsms_honest_statevector"] = res.honest_statevector
            bus.metric("secure_sum", total, self.name)
            bus.metric("qsms_prime", res.modulus, self.name)
            bus.metric("qsms_honest_statevector", int(res.honest_statevector), self.name)
            bus.log(f"QSMS secure summation = {total} over GF({res.modulus}) with {len(secrets)} "
                    f"parties, t={min(3, n)}"
                    + (" (exact d-qudit GHZ statevector)" if res.honest_statevector
                       else " (analytic sampler: d**t exceeds the statevector bound)"), self.name)


def _hybrid_bench_row(ctx, reg, x, base_model):
    """Run the faithful linear->FHE + non-linear->MPC hybrid at the configured model size on the
    use case's feature vector, returning the MPC-FHE ``BenchResult`` (the live benchmark row).

    lightweight === the trained linear model (so the SHAP explanation stays faithful); medium/large
    add hidden ReLU layers run under MPyC secret-sharing; large is projected from a medium
    calibration. Reuses the Round-6 benchmark primitives so the live row matches the standalone
    benchmark exactly.
    """
    import numpy as np

    from ..benchmark.classical_mpc_fhe import run_hybrid
    from ..benchmark.runner import _params, _project_large, usecase_arch_layers, usecase_arch_specs

    uc = ctx.usecase
    model_size = ctx.config.get("model_size", "medium")
    workload = ctx.config.get("workload", "linear")
    seed = int(ctx.config.get("seed", 0))
    fhe = ctx.config.get("fhe") or reg.default("fhe") or "tenseal-cpu"
    backend_name = fhe.split("-")[0]
    if backend_name not in ("tenseal", "openfhe"):     # heongpu has no Python binding -> CPU FHE
        backend_name = "tenseal"

    # ---- genuine CNN under FHE+MPC (medical, when a real MedMNIST image is present) ----
    image = ctx.data.get("image")
    if uc == "medical_fl" and image is not None:
        from ..ai.cnn_layers import random_cnn_specs, smallcnn_to_specs

        img = np.asarray(image, dtype=float)
        in_shape = tuple(int(v) for v in img.shape)
        cnn_model = ctx.data.get("cnn_model")
        y1 = np.zeros(1, dtype=int)
        if model_size == "lightweight":                # pixel logistic (fast, faithful SHAP surrogate)
            specs, label, n_params = usecase_arch_specs(
                uc, "lightweight", workload="cnn", seed=seed, base=base_model,
                n_in=int(np.prod(in_shape)), img=in_shape)
            row = run_hybrid(specs, img.reshape(1, -1), y1, backend_name=backend_name,
                             workload="cnn", model_size="lightweight", n_samples=1)
        elif model_size == "large":                    # 2-block CNN, projected from a medium calibration
            med_specs = (smallcnn_to_specs(cnn_model, in_shape) if cnn_model is not None
                         else random_cnn_specs(in_shape, channels=(4,), seed=seed))
            large_specs, label, n_params = usecase_arch_specs(uc, "large", workload="cnn",
                                                              seed=seed, img=in_shape)
            med = run_hybrid(med_specs, img[None], y1, backend_name=backend_name,
                             workload="cnn", model_size="medium", n_samples=1)
            row = _project_large(med, med_specs, large_specs, "cnn")
        else:                                          # medium = the genuine trained 1-block CNN
            specs = (smallcnn_to_specs(cnn_model, in_shape) if cnn_model is not None
                     else random_cnn_specs(in_shape, channels=(4,), seed=seed))
            label = f"diagnostic CNN (MedMNIST {in_shape[1]}x{in_shape[2]}, 1-block) · medium"
            n_params = _params(specs)
            row = run_hybrid(specs, img[None], y1, backend_name=backend_name,
                             workload="cnn", model_size="medium", n_samples=1)
        row.model_arch = label
        row.model_params = n_params
        return row

    # ---- dense path (tabular use cases — unchanged) ----
    x1 = np.atleast_2d(np.asarray(x, dtype=float))
    y1 = np.zeros(len(x1), dtype=int)
    n_in = x1.shape[1]
    if model_size == "large":
        med_layers, _, _ = usecase_arch_layers(uc, "medium", workload=workload, seed=seed, base=base_model, n_in=n_in)
        large_layers, label, n_params = usecase_arch_layers(uc, "large", workload=workload, seed=seed, base=base_model, n_in=n_in)
        med = run_hybrid(med_layers, x1, y1, backend_name=backend_name, workload=workload,
                         model_size="medium", n_samples=1)
        row = _project_large(med, med_layers, large_layers, workload)
    else:
        layers, label, n_params = usecase_arch_layers(uc, model_size, workload=workload, seed=seed, base=base_model, n_in=n_in)
        row = run_hybrid(layers, x1, y1, backend_name=backend_name, workload=workload,
                         model_size=model_size, n_samples=1)
    row.model_arch = label
    row.model_params = n_params
    return row


class PrivateInferenceStage(Stage):
    """QHE/FHE private inference (classical CKKS or quantum QCNN)."""

    name = "private_inference"

    def run(self, ctx, bus, reg):
        mode = ctx.config.get("inference", "ckks")
        if mode == "qcnn":
            import numpy as np

            from ..config import get_config
            from ..qhe.private_inference import private_inference
            from ..qhe.qcnn import QCNN

            from ..qhe.qcnn_trained import infer as qcnn_infer
            from ..qhe.qcnn_trained import load_weights

            qubits = int(ctx.config.get("qubits", get_config().qhe.n_qubits))
            x = ctx.data.get("features", np.zeros(qubits))

            # TRAINED circuit where weights exist. The untrained QCNN is not a classifier: measured
            # over 200 random inputs at 4 and 8 qubits, `predict_proba` never exceeds 0.241, so it
            # never crosses the 0.5 boundary and returns class 0 for EVERY input at every seed, with
            # correlation -0.05 against the input. That is a CONSTANT PREDICTOR - harder to notice
            # than the browser's coin-flip form of the same defect, because it looks stable and
            # reproducible. Round 25 fixed the browser and left this path untouched.
            weights = load_weights(ctx.usecase)
            if weights is not None:
                expectation, cls, _logit = qcnn_infer(x, weights)
                ctx.artifacts["qhe_prediction"] = cls
                ctx.artifacts["qhe_backend"] = "qcnn-trained"
                bus.metric("qhe_expectation_z0", round(expectation, 4), self.name)
                bus.metric("qhe_test_accuracy", weights["test_accuracy"], self.name)
                bus.log(
                    f"QHE private inference (TRAINED QCNN): class={cls} "
                    f"(<Z0>={expectation:.4f}, {weights['layers']}x{weights['n_qubits']} trained "
                    f"angles). Held-out accuracy {weights['test_accuracy']:.4f} against logistic "
                    f"regression {weights['classical_lr_accuracy']:.4f} on the same split - the "
                    f"classical model is ahead, and this reports it rather than implying advantage.",
                    self.name)
            else:
                # No trained circuit here: report the QOTP hiding demonstration, which this module
                # genuinely proves, and decline to surface a class rather than passing off a
                # constant one as a prediction.
                qc = QCNN(n_qubits=qubits, seed=ctx.config.get("seed", 0))
                rep = private_inference(qc, x, verify_hidden=True)
                ctx.artifacts["qhe_backend"] = "untrained-not-a-classifier"
                bus.metric("qhe_t_count", rep.t_count, self.name)
                bus.log(
                    f"QCNN selected but no trained weights ship for {ctx.usecase}: the untrained "
                    f"circuit is a QOTP hiding and T-count demonstration (t_count={rep.t_count}, "
                    f"input_hidden={rep.input_hidden}), NOT a classifier - its probability never "
                    f"crosses 0.5, so it returns the same class for every input. No class is "
                    f"reported from it.", self.name)
        else:
            import numpy as np

            from ..fhe.ckks_inference import LinearModel, encrypted_linear_inference

            model = ctx.data.get("linear_model")
            x = ctx.data.get("features")
            if model is None or x is None:
                rng = np.random.default_rng(0)
                model = LinearModel(rng.normal(size=(2, 8)), rng.normal(size=2))
                x = rng.normal(size=8)
            backend = reg.get("fhe", ctx.config.get("fhe"))()
            out = encrypted_linear_inference(model, np.asarray(x), backend)
            ctx.artifacts["fhe_prediction"] = out["prediction"]
            # record exactly what was inferred so ExplainabilityStage explains the same thing
            ctx.data["linear_model"] = model
            ctx.data["features"] = np.asarray(x)
            bus.metric("fhe_max_abs_error", out["max_abs_error"], self.name)
            bus.log(f"CKKS private inference: class={out['prediction']} "
                    f"(backend={out['backend']}, err={out['max_abs_error']:.2e})", self.name)
            # --- Unified benchmark (Round 7): run the REAL hybrid (linear->FHE + non-linear->MPC)
            # at the selected model size so the live run IS the MPC-FHE benchmark row. lightweight ===
            # the trained linear model (SHAP above explains the same decision); medium/large add
            # hidden ReLU layers under MPyC secret-sharing; large is projected. The split metrics
            # drive both the diagram's FHE-linear/MPC-non-linear steps and the post-run benchmark.
            try:
                row = _hybrid_bench_row(ctx, reg, np.asarray(x), model)
                ctx.data["_mpc_fhe_bench"] = row              # BenchResult object (for advantage)
                ctx.artifacts["mpc_fhe_row"] = row.to_row()
                ctx.artifacts["mpc_nonlinear_backend"] = row.backend
                bus.metric("fhe_linear_ms", row.fhe_linear_ms, self.name)
                bus.metric("mpc_nonlinear_ms", row.mpc_nonlinear_ms, self.name)
                bus.metric("mpc_comm_kb", row.mpc_comm_kb, self.name)
                bus.metric("inference_accuracy", row.accuracy, self.name)
                bus.log(f"hybrid inference [{row.model_size}]: FHE-linear {row.fhe_linear_ms} ms + "
                        f"MPC-non-linear {row.mpc_nonlinear_ms} ms ({row.backend}"
                        f"{', projected' if row.projected else ''})", self.name)
            except Exception as exc:  # keep the live run resilient if MPyC/FHE hiccups
                bus.log(f"hybrid benchmark row skipped: {exc}", self.name)


class ExplainabilityStage(Stage):
    """Explainable AI: exact linear-SHAP feature attributions for the private-inference result.

    Satisfies the report's XAI / audit-trail requirement and prepares for EU AI Act Art. 13 transparency
    (applicable 2027-12-02 for Annex III after the AI Omnibus deferral, not Aug 2026)
    obligation (high-risk credit/AML + healthcare). Faithful (exact for the linear model) and
    cheap; the attributions are written to ``ctx.artifacts["explanation"]`` so the downstream
    :class:`SignatureStage` ML-DSA-signs them -> a tamper-evident, regulator-facing explanation.
    Skips cleanly when the use case has no linear model to explain (e.g. secure-sum, IDS, QSDC).
    """

    name = "explainability"

    def run(self, ctx, bus, reg):
        import numpy as np

        from ..ai.explainability import explain_linear_model

        model = ctx.data.get("linear_model")
        x = ctx.data.get("features")
        if model is None or x is None:
            bus.log("no linear model/features to explain — skipping XAI", self.name)
            return
        names = ctx.data.get("feature_names")
        baseline = ctx.data.get("explain_baseline")
        exp = explain_linear_model(model, np.asarray(x), baseline=baseline,
                                   feature_names=names, top_k=5)
        _r = lambda v: round(float(v), 6)           # noqa: E731 — display rounding only
        ctx.artifacts["explanation"] = {
            "method": exp.method,
            "prediction": exp.prediction,
            "base_value": _r(exp.base_value),
            "margin": _r(exp.margin),
            "top_features": [{"feature": t["feature"], "value": _r(t["value"]),
                              "contribution": _r(t["contribution"])} for t in exp.top],
            "efficiency_gap": round(exp.efficiency_gap(), 9),  # ~0 => exact (SHAP local accuracy)
            # WHICH reference point produced these attributions. efficiency_gap cannot answer it:
            # completeness holds for ANY baseline, so a zero-vector reference on non-zero-mean data
            # passes the self-check while every contribution is measured against a case that does
            # not occur. Recording it is the only way the defect stays visible.
            "baseline": "E[x] (dataset mean)" if baseline is not None else "zero vector (no E[x] supplied)",
        }
        if exp.top:
            bus.metric("explanation_top_contribution", _r(exp.top[0]["contribution"]), self.name)
            bus.log(f"XAI: top driver = {exp.top[0]['feature']} "
                    f"(contribution {exp.top[0]['contribution']:+.3f}); "
                    f"signed for audit", self.name)


class QuantumComparisonStage(Stage):
    """Assemble the live-run benchmark: the measured MPC-FHE row (from PrivateInferenceStage) +
    a QSMPC-FHE comparison row + quantum-advantage -> ``ctx.artifacts['benchmark_result']``.

    This is the Round-7 unification: a single Run animates the live E2E pipeline AND produces the
    benchmark for that very run (the live config's MPC-FHE row uses the run's real measured
    FHE-linear/MPC-non-linear latency). Use cases without a private-inference stage (telecom/
    satellite) still get the QSMPC-FHE security row (advantage stays None -> the panel shows compact).
    """

    name = "quantum_comparison"

    def run(self, ctx, bus, reg):
        import numpy as np

        from ..benchmark.quantum_advantage import quantum_advantage
        from ..benchmark.quantum_qsmpc_fhe import run_quantum

        model_size = ctx.config.get("model_size", "medium")
        workload = ctx.config.get("workload", "linear")
        seed = int(ctx.config.get("seed", 0))
        qubits = int(ctx.config.get("qubits", 8))

        X = ctx.data.get("X")
        if X is None:                                    # finance/genomics: derive from the feature vec
            feats = ctx.data.get("features")
            X = (np.atleast_2d(np.asarray(feats, dtype=float)) if feats is not None
                 else np.random.default_rng(seed).normal(size=(4, qubits)))
        y = ctx.data.get("y")
        if y is None:
            y = np.zeros(len(X), dtype=int)

        # Score this row under the transport the run ACTUALLY used, from the same metrics the
        # classical row is scored against - so the two rows differ by cryptography, not by which
        # arguments each call site happened to supply.
        q = run_quantum(X, y, n_qubits=qubits, n_samples=min(8, len(X)), seed=seed,
                        kem=ctx.config.get("kem"),
                        qkd_key_bits=ctx.metrics.get("secure_key_bits"),
                        qber=ctx.metrics.get("qber"))
        q.model_size = model_size
        q_row = q.to_row()

        c_bench = ctx.data.get("_mpc_fhe_bench")         # live measured MPC-FHE BenchResult (or None)
        c_row = ctx.artifacts.get("mpc_fhe_row")
        results = [r for r in (c_row, q_row) if r]
        adv = quantum_advantage(c_bench, q) if c_bench is not None else None

        ctx.artifacts["benchmark_result"] = {
            "workload": workload, "model_size": model_size,
            "results": results, "advantage": adv, "live": True,
        }
        if adv:
            bus.metric("quantum_advantage_score", adv["score"], self.name)
        bus.log(f"benchmark assembled from the live run: "
                f"{'MPC-FHE (measured) + ' if c_row else ''}QSMPC-FHE comparison"
                + (f"; quantum advantage {adv['score']}/10 ({adv['tier']})" if adv else ""), self.name)


class SignatureStage(Stage):
    """Sign the run summary with the selected post-quantum signature (crypto-agility).

    Demonstrates the registry's `sig` family at runtime: the chosen ML-DSA provider
    signs the canonical summary and the signature is verified, producing a tamper-evident,
    quantum-safe audit record of the run (the report's XAI / audit-trail requirement).
    """

    name = "sign_result"

    def run(self, ctx, bus, reg):
        import json as _json

        from ..pqc.signatures import verify

        summary = ctx.artifacts.get("summary", {"usecase": ctx.usecase})
        payload = _json.dumps(summary, sort_keys=True, default=str).encode()
        sig_name = ctx.config.get("sig", reg.default("sig"))
        kp = reg.get("sig", sig_name)()
        try:
            signature = kp.sign(payload)
            ok = verify(payload, signature, kp.public_key, kp.sig_name)
        finally:
            kp.free()
        ctx.artifacts["result_signature"] = {
            "algorithm": kp.sig_name,
            "signature_hex_prefix": signature.hex()[:24],
            "signature_bytes": len(signature),
            "verified": bool(ok),
        }
        bus.metric("signature_verified", int(ok), self.name)
        bus.log(f"run summary signed with {kp.sig_name} (verified={ok})", self.name)


class ResultStage(Stage):
    """Finalize: stamp a summary artifact."""

    name = "result"

    def run(self, ctx, bus, reg):
        ctx.artifacts["summary"] = {
            "usecase": ctx.usecase,
            "metrics": dict(bus.metrics()),
            "artifacts": {k: v for k, v in ctx.artifacts.items()
                          if k not in ("summary", "result_signature")},
        }
        bus.log("pipeline complete", self.name)
