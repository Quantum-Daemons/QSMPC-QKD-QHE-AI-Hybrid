"""Typed configuration defaults for the whole stack.

Every tunable parameter that was previously a hard-coded literal lives here as a
dataclass field. The values equal the historical defaults, so loading the config
changes no behaviour — it just makes the parameters discoverable, overridable from
``config.yaml``, and adjustable live from the WebUI (the orchestration run config
is merged over these).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class QKDConfig:
    pulses: int = 200_000
    distance_km: float = 50.0
    fiber_loss_db_per_km: float = 0.2
    detector_efficiency: float = 0.25
    dark_count_prob: float = 1e-5
    misalignment: float = 0.015
    eve_intercept_fraction: float = 0.0
    error_correction_eff: float = 1.16
    # --- decoy-state / efficient-BB84 parameters the AI optimizer (qkd/optimize.py) tunes ---
    mu_signal: float = 0.5            # signal mean photon number (decoy-state); ~0.5 is the textbook optimum
    mu_decoy: float = 0.1             # decoy mean photon number
    basis_prob: float = 0.5           # P(Z basis) — biased-basis (efficient) BB84 raises the sift ratio
    pp_threshold: float = 0.11        # post-processing QBER abort threshold
    # --- AI optimizer (OptiQKD, WS1) knobs ---
    optimize_enabled: bool = False
    optimize_method: str = "surrogate"        # surrogate | coordinate | gp
    optimize_n_samples: int = 48
    optimize_n_pulses: int = 20_000           # pulses per evaluation (smaller than the run default, for speed)
    opt_mu_signal_min: float = 0.2
    opt_mu_signal_max: float = 0.9
    opt_basis_prob_min: float = 0.5
    opt_basis_prob_max: float = 0.9
    opt_pp_threshold_min: float = 0.08
    opt_pp_threshold_max: float = 0.12


@dataclass
class QRNGConfig:
    """Quantum random-number generator (entropy source). Default = a local state-vector
    quantum-measurement simulation (honest: simulated, not hardware); optional ANU QRNG API is opt-in."""
    enabled: bool = False             # off by default -> BB84 entropy unchanged unless opted in
    bits: int = 4096
    source: str = "sim"               # "sim" (offline default) | "anu" (opt-in, online hardware)
    post_processing: str = "von_neumann"   # "von_neumann" | "toeplitz" | "none"
    anu_opt_in: bool = False          # must be True to allow the online ANU source (subject to ANU terms)
    health_alpha: float = 0.01        # SP800-22 significance level
    batch_qubits: int = 16            # qubits measured per batch in the state-vector sim


@dataclass
class QSMPCConfig:
    prime: int = 4099          # default field modulus for secure summation
    n_parties: int = 7
    threshold: int = 3
    qot_n: int = 512
    qot_test_fraction: float = 0.25


@dataclass
class QHEConfig:
    n_qubits: int = 8
    sk_eps: float = 1e-3       # Solovay-Kitaev T-gate approximation tolerance


@dataclass
class FHEConfig:
    backend: str = "tenseal"
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: tuple[int, ...] = (60, 40, 40, 60)
    global_scale_bits: int = 40
    max_level: int = 12
    use_gpu: bool = True
    use_bootstrap: bool = False
    # OpenFHE CKKS context params (batch_size is the slot width, was a hard-coded literal)
    openfhe_mult_depth: int = 4
    openfhe_scale_bits: int = 50
    openfhe_batch_size: int = 128


@dataclass
class FederatedConfig:
    rounds: int = 8
    local_epochs: int = 1
    lr: float = 0.05
    batch_size: int = 32
    hospitals: int = 4
    test_split: float = 0.2
    secure_aggregation: bool = True


@dataclass
class TrainConfig:
    epochs: int = 15
    batch: int = 64
    lr: float = 1e-3
    test_split: float = 0.2


@dataclass
class IDSConfig:
    n_per_class: int = 300
    q_n_per_class: int = 100        # smaller sample for the (slower) quantum classifier path
    test_fraction: float = 0.25
    rf_estimators: int = 200
    q_n_qubits: int = 4
    q_n_layers: int = 2
    q_epochs: int = 25
    q_lr: float = 0.1
    demo_n_per_class: int = 120     # in-browser client IDS sample size per class (clientIDS.ts)


@dataclass
class DataConfig:
    n_transactions: int = 4000
    fraud_rate: float = 0.05
    n_genotypes: int = 1000
    n_snps: int = 64
    # transaction distribution parameters (synthetic AML/PaySim-style)
    amount_lognormal_mean: float = 3.5
    amount_fraud_shift: float = 1.2
    prior_poisson_lambda: float = 2.0
    prior_fraud_shift: float = 6.0
    velocity_gamma_shape: float = 2.0
    velocity_fraud_scale_shift: float = 3.0
    # finance_aml use-case set sizes (PSI blacklists)
    aml_transactions: int = 4000
    aml_shared_accounts: int = 12
    aml_bank_a_accounts: int = 300
    aml_bank_b_accounts: int = 280
    # genomics use-case sizes
    genomics_n: int = 1200
    genomics_n_snps: int = 16
    genomics_biobanks: int = 4
    genomics_allele_min: int = 50
    genomics_allele_max: int = 400


@dataclass
class BenchmarkConfig:
    n_samples: int = 30
    quantum_qubits: int = 8
    quantum_samples: int = 20
    hybrid_samples: int = 6            # MPC ReLU/argmax is per-sample; cap to keep the run responsive
    # per-use-case architectures (web-grounded). lightweight === the trained linear model (faithful SHAP);
    # medium/large are representative hidden-layer shapes. The encrypted hybrid preserves the plaintext
    # decision for any weights, so the metric is FHE-linear/MPC-non-linear COST + fidelity, not task SOTA.
    tier_architectures: dict = field(default_factory=lambda: {
        "finance_aml": {"label": "tabular fraud MLP",
                        "hidden": {"lightweight": [], "medium": [32, 16], "large": [128, 64]}},
        "medical_fl": {"label": "diagnostic net (MedMNIST-scale)",
                       "hidden": {"lightweight": [], "medium": [64, 32], "large": [256, 128]}},
        "genomics": {"label": "genomic risk MLP",
                     "hidden": {"lightweight": [], "medium": [64], "large": [256, 128]}},
    })
    generic_architecture: dict = field(default_factory=lambda: {
        "lightweight": [], "medium": [16], "large": [64, 32]})    # standalone benchmark (no use case)
    # workload class -> [cost multiplier, label suffix]: picking CNN/Transformer raises the measured cost
    workload_multipliers: dict = field(default_factory=lambda: {
        "linear": [1.0, ""], "cnn": [2.0, " (CNN-class)"], "transformer": [3.0, " (Transformer-class)"]})
    # representative per-op cost surrogates for the CLIENT demo only (the full backend measures real
    # latency; plaintext-JS wall-clock can't reproduce CKKS cost, so the demo reports these).
    cost_per_op: dict = field(default_factory=lambda: {
        "fhe_ms_per_dot": 12.0, "mpc_ms_per_cmp": 1.6, "mpc_comm_bytes_per_cmp": 256,
        "demo_ct_pt_ratio": 1483.0, "demo_max_abs_error": 4.6e-7,
        "qsmpc_ct_pt_ratio": 1.32, "qsmpc_t_gates": 56, "qsmpc_qubits": 8})
    # the tiny FHE-feasible medical CNN (genuine convolution under FHE+MPC)
    medical_cnn: dict = field(default_factory=lambda: {
        "img_size": 14, "channels": [4], "hidden": 16, "pool": "avg",
        "medmnist_name": "pneumoniamnist", "subset": 512})
    demo_feature_dim: int = 8         # client-demo input feature width (N_IN)


@dataclass
class SecurityScoringConfig:
    """Component weights + thresholds for the 0-100 security score (benchmark/security.py)."""
    hndl_pqc_hybrid: float = 30.0
    qkd_bonus: float = 5.0
    info_theoretic: float = 30.0
    computational: float = 18.0
    # NIST ML-KEM category -> key-strength points
    key_strength_by_category: dict = field(default_factory=lambda: {5: 20.0, 4: 18.0, 3: 16.0, 2: 13.0, 1: 10.0})
    qber_secure_threshold: float = 0.11       # BB84 secure QBER threshold
    qber_danger_threshold: float = 0.20
    eavesdrop_secure_pts: float = 10.0
    eavesdrop_warn_pts: float = 5.0
    leakage_low: float = 0.1                   # PRX ~0.1 bit/weight is strong
    leakage_mid: float = 0.5
    leakage_pts_low: float = 10.0
    leakage_pts_mid: float = 7.0
    leakage_pts_high: float = 3.0
    leakage_pts_default_it: float = 6.0
    leakage_pts_default_comp: float = 4.0


@dataclass
class QuantumAdvantageConfig:
    """Weights + tier boundaries for the 0-10 quantum-advantage score (benchmark/quantum_advantage.py)."""
    w_comm: float = 0.6
    w_latency: float = 0.2
    w_security: float = 0.2
    info_theoretic_multiplier: float = 1.5
    log_scale: float = 3.2
    log_offset: float = 1.0
    tier_dominant: float = 8.0
    tier_strong: float = 5.0
    tier_moderate: float = 2.0


@dataclass
class CryptoConfig:
    kem: str = "ml-kem-768+x25519"
    qkd: str = "bb84-decoy"
    fhe: str = "tenseal-cpu"
    sig: str = "ml-dsa-65"


@dataclass
class OrchestrationConfig:
    step_delay: float = 0.4    # seconds between stages (so the SPA DAG animates visibly)


@dataclass
class WebConfig:
    """Frontend-only knobs (served via /api/config + baked into config.defaults.json) — the in-browser
    AI model lists / generation limits / timeout, plus demo-sim sizes that were inline TS literals."""
    tfjs_models: list = field(default_factory=lambda: [
        {"id": "Xenova/distilgpt2", "label": "distilGPT-2 (tiny, ~80 MB)"},
        {"id": "onnx-community/Qwen2.5-0.5B-Instruct", "label": "Qwen2.5-0.5B-Instruct (ONNX)"}])
    webllm_models: list = field(default_factory=lambda: [
        {"id": "Qwen2.5-0.5B-Instruct-q4f16_1-MLC", "label": "Qwen2.5-0.5B-Instruct (~350 MB)"},
        {"id": "Phi-3-mini-4k-instruct-q4f16_1-MLC", "label": "Phi-3-mini-4k (~2 GB)"}])
    agent_timeout_ms: int = 120_000     # bounds a stuck WebGPU/model load (slow first-run ~60-70 s)
    tfjs_max_new_tokens: int = 96
    webllm_max_tokens: int = 160
    demo_alice_bob_pulses: int = 40_000
    demo_sat_pulses: int = 4_000_000    # free-space BB84 channel uses in the demo satellite run


@dataclass
class QSHConfig:
    """Top-level config aggregating every subsystem."""

    qkd: QKDConfig = field(default_factory=QKDConfig)
    qrng: QRNGConfig = field(default_factory=QRNGConfig)
    qsmpc: QSMPCConfig = field(default_factory=QSMPCConfig)
    qhe: QHEConfig = field(default_factory=QHEConfig)
    fhe: FHEConfig = field(default_factory=FHEConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    ids: IDSConfig = field(default_factory=IDSConfig)
    data: DataConfig = field(default_factory=DataConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    security_scoring: SecurityScoringConfig = field(default_factory=SecurityScoringConfig)
    quantum_advantage: QuantumAdvantageConfig = field(default_factory=QuantumAdvantageConfig)
    crypto: CryptoConfig = field(default_factory=CryptoConfig)
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)
    web: WebConfig = field(default_factory=WebConfig)

    def to_dict(self) -> dict:
        return asdict(self)
