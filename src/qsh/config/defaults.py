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
    # MODELLING CHOICES, and each one departs from a reference this repository already cites and
    # tests against: the GYS set in `qkd/decoy_reference.py` (Gobby, Yuan & Shields, APL 84, 3762
    # (2004), as tabulated in Ma et al., PRA 72, 012326 (2005)), asserted by
    # `tests/test_decoy_reference.py`. They were bare values with no source comment until Round 35,
    # which read as "nobody knows where these came from" when the truth is that a cited alternative
    # sits two modules away and these deliberately differ from it.
    #
    #                          here     GYS      ratio   favours the demo?
    #   fibre loss             0.2      0.21     0.95x   yes, less loss
    #   detector efficiency    0.25     0.045    5.56x   yes, a far better detector
    #   dark count             1e-5     1.7e-6   5.88x   NO, more noise
    #   misalignment           0.015    0.033    0.45x   yes, less error
    #   error correction f_EC  1.16     1.22     0.95x   yes, better error correction
    #
    # FOUR OF FIVE LEAN THE SAME WAY. None is wrong - a 2004 InGaAs APD is not what anyone would
    # model in 2026, and there is no single published detector efficiency to be right about. But the
    # direction is systematic and a reader comparing this demo's key rates against the published
    # curves should know it before they do.
    fiber_loss_db_per_km: float = 0.2         # modelling choice; GYS 0.21
    detector_efficiency: float = 0.25         # modelling choice; GYS 0.045
    dark_count_prob: float = 1e-5             # modelling choice; GYS 1.7e-6
    misalignment: float = 0.015               # modelling choice; GYS 0.033
    eve_intercept_fraction: float = 0.0       # demo control, not a device property
    error_correction_eff: float = 1.16        # modelling choice; GYS 1.22 (Cascade/LDPC overhead)
    # --- free-space (satellite downlink) channel ---
    #
    # These existed only as TypeScript literals in quantumSim.ts's FREE_SPACE, which every satellite
    # call site then spread while overriding just distance and Eve. So four of the six physical
    # constants behind the satellite page could not be changed without editing the source, and the
    # exact same shape had already produced a real bug on the fibre side: config and literal agreed
    # numerically, so editing config.yaml moved the Python model and left the browser on the constant
    # with nothing to indicate it. Fibre lives in config; free space now does too.
    #
    # Values are the ones the TS literals carried, so this is a move rather than a retune.
    free_space_distance_km: float = 800.0      # ground-to-LEO slant range
    free_space_detector_efficiency: float = 0.4
    free_space_dark_count_prob: float = 1e-5
    free_space_misalignment: float = 0.025     # pointing error dominates over optical misalignment
    #
    # A satellite downlink is NOT attenuated per kilometre. It was modelled that way here -
    # `distance_km * 0.035`, a fibre law - which happened to give 28 dB at 800 km, close to the 27 dB
    # Micius reports, and then diverged: 42 dB at 1200 km against a physical ~28, 70 against ~32 at
    # 2000, 175 against ~40 at 5000. The Downlink control has no maximum, so anyone dragging it past
    # about 1000 km was reading numbers with no physical meaning.
    #
    # The dominant term is DIFFRACTION, which scales as 1/r^2: the transmitted beam spreads to
    # theta*r and the ground telescope captures (D_rx / (theta*r))^2 of it. Atmosphere contributes a
    # fixed few dB rather than a per-km rate, because a downlink crosses it once near the ground -
    # 3 to 8 dB at 1200 km in the literature, with pointing under 3 dB.
    #
    # Defaults are Micius's published configuration (Liao et al., Nature 549, 43 (2017);
    # arXiv:1707.00542): ~10 urad far-field divergence, 1.2 m ground aperture.
    free_space_beam_divergence_urad: float = 10.0   # far-field, near-diffraction-limited
    free_space_rx_aperture_m: float = 1.2           # optical ground station telescope diameter
    free_space_atmospheric_db: float = 5.0          # absorption + turbulence, whole path, not per km
    free_space_pointing_db: float = 3.0             # acquisition/tracking residual
    # --- decoy-state / efficient-BB84 parameters the AI optimizer (qkd/optimize.py) tunes ---
    mu_signal: float = 0.5            # signal mean photon number (decoy-state); ~0.5 is the textbook optimum
    mu_decoy: float = 0.1             # decoy mean photon number
    basis_prob: float = 0.5           # P(Z basis) — biased-basis (efficient) BB84 raises the sift ratio
    #: QBER abort threshold, and it is a FIBRE-QKD post-processing figure - not a QSDC one.
    #:
    #: An earlier revision of this comment claimed 0.11 was correct for QSDC because the wiretap
    #: secrecy capacity Cs = 1 - 2h(e) has its root at 0.110028, identical to Shor-Preskill. That is
    #: true only for the LOSS-FREE case. The published QSDC criterion (Qi et al., Light Sci. Appl.
    #: 8, 22 (2019), step 2) is Cs > 0 with
    #:
    #:     Cs = Q_Bob[1 - h(e) - g h(e_chk)],   g = Q_Eve/Q_Bob = 10^(loss_dB/10)
    #:
    #: so the threshold FALLS with channel loss: 0.11 at 0 dB, ~0.048 at the 4.1 dB Qi et al.
    #: actually measured, and ~0.0003 at the ~24.5 dB this project's own free-space downlink
    #: reports. Using 0.11 on a lossy link is wrong in the PERMISSIVE direction - it passes a
    #: channel whose secrecy capacity is already zero. The satellite path therefore computes Cs
    #: directly (clientCompute qsdc_telemetry) rather than comparing against this constant.
    #:
    #: Also worth knowing before quoting any single number: the founding QSDC papers (Long-Liu 2002,
    #: Deng-Long-Liu 2003, DL04 2004) give NO numeric threshold, only "if the error rate is high,
    #: halt"; DL04 - the family every fibre experiment implements - is put at ~5.7% (Wu et al.,
    #: Quantum Eng. 1, e26, 2019); and no standards body has ever specified one, because ETSI,
    #: ITU-T, ISO/IEC and IEEE quantum series are all key DISTRIBUTION.
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
    opt_mu_decoy_min: float = 0.05            # decoy-intensity search range (kept < mu_signal by the model)
    opt_mu_decoy_max: float = 0.20
    # --- protocol selection + E91 (entanglement-based) parameters ---
    protocol: str = "bb84"                    # "bb84" (prepare-measure, decoy-state) | "e91" (entanglement/CHSH)
    # `e91_pairs` was removed in Round 35. It had no reader in Python OR TypeScript - the E91
    # stage sizes itself from `qkd_pulses`, which is the control the Parameters panel actually
    # exposes. An exported config key nothing reads is the config-space form of a control that
    # renders and changes nothing, and this project removes those rather than inventing a
    # reader: wiring it would have created a second, unsurfaced sizing input for one quantity.
    e91_visibility: float = 0.97              # entangled-state visibility -> CHSH S-parameter + QBER (E91)


@dataclass
class QRNGConfig:
    """Quantum random-number generator (entropy source). Default = a local state-vector
    quantum-measurement simulation (honest: simulated, not hardware); optional ANU QRNG API is opt-in."""
    enabled: bool = False             # off by default -> BB84 entropy unchanged unless opted in
    bits: int = 4096
    source: str = "sim"               # "sim" (offline default) | "anu" (opt-in, online hardware)
    post_processing: str = "von_neumann"   # "von_neumann" | "toeplitz" | "seedless" | "none"
    anu_opt_in: bool = False          # must be True to allow the online ANU source (subject to ANU terms)
    health_alpha: float = 0.01        # SP800-22 significance level
    health_block_size: int = 128      # SP800-22 block-frequency block size
    batch_qubits: int = 16            # qubits measured per batch in the state-vector sim
    # --- information-theoretic extractor (Toeplitz + (Quantum) Leftover-Hash-Lemma), Giestinhas & Spiller 2026 ---
    security_epsilon_log2: int = -40      # LHL security parameter: extractor output is 2^this-close to uniform
    min_entropy_per_raw_bit: float = 0.9  # conservative min-entropy per raw simulated-|+> bit (ideal = 1.0)


@dataclass
class QSMPCConfig:
    # Sutradhar's construction works over GF(d) with **d prime and n <= d <= 2n** (arXiv:2501.10083,
    # and the same bound in secure_summation.py's docstring). 4099 satisfied "prime" and "> n" and
    # nothing else: with n=7 the paper wants d in [7, 14], and 4099 is nearly 300x outside it.
    #
    # It was not a harmless overshoot. The honest GHZ statevector is only built when d**t fits the
    # simulator's state bound, and 4099**3 is 68.9 billion against a 4.19 million ceiling - so the
    # shipped default ALWAYS fell through to the analytic sampler and reported
    # honest_statevector=False, while the page above it advertised a d-qudit GHZ state. None means
    # "derive it from n", which lands on the paper's own worked example d=11 for n=7.
    prime: int | None = None
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
    #: The browser's lighter federated schedule. A deliberate override of `rounds` and `lr`, because a
    #: federated round has to finish inside a page interaction - the server-side values above are what
    #: the full deployment trains with.
    #:
    #: These were literals inside `frontend_config()` for three rounds (29, 30, 31), recorded on the
    #: deferred list with a reason and **no condition that would resolve it** - which fails the backlog
    #: gate's first clause as well as its three-round clause. The consequence was concrete rather than
    #: cosmetic: the override could not be tuned the way every value it overrides can, so anyone wanting
    #: a heavier browser schedule had to edit a dict literal in a function that builds a JSON payload.
    #: The gate exists to stop exactly this kind of small thing rolling forward, and the fix is smaller
    #: than the three paragraphs written about deferring it.
    browser_rounds: int = 6
    browser_lr: float = 0.15


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
    #: Which bundled PGS-Catalog score the genomics use case evaluates. A FIELD, not a literal, and
    #: the reason is the defect this replaces: the pipeline read `config.get("prs", "breast_cancer")`
    #: while nothing anywhere set that key, so when Round 31 moved the browser, the trainer, the
    #: registry and the payload to PGS000004 it left the E2E pipeline scoring PGS000001. Two surfaces
    #: then reported different scores for the same use case - exactly what `qsh.data.loaders` warns
    #: "would be worse than either alone".
    #:
    #: A literal default inside a `config.get()` call is invisible to every mechanism this project
    #: uses to keep values honest: it does not appear in the config dump, cannot be overridden per
    #: run, and cannot be compared against what the browser fetched. A value that selects a model
    #: belongs here beside the KEM and the FHE backend. Valid names are the keys of
    #: `qsh.data.loaders._PRS_FILES`; `tests/test_prs_pipeline_parity.py` pins this against the
    #: browser payload the registry points at.
    genomics_prs_score: str = "breast_cancer_prs313"


@dataclass
class BenchmarkConfig:
    n_samples: int = 30
    quantum_qubits: int = 8
    quantum_samples: int = 20
    hybrid_samples: int = 6            # MPC ReLU/argmax is per-sample; cap to keep the run responsive
    mpc_comm_bytes_per_compare: int = 256   # wire bytes per secure comparison (was a literal in mpc/)
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
        # BreastMNIST, not pneumoniamnist: this block describes the shape of the trained
        # medical_fl.student, and that student is distilled from the BreastMNIST teacher.
        "medmnist_name": "breastmnist", "subset": 512})
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
    # PQC+QKD key combiner: "hkdf" = dual-PRF KDF (default, byte-compatible) | "its_toeplitz" =
    # information-theoretic leakage-resilient Toeplitz + Quantum-Leftover-Hash-Lemma combiner
    # (Giestinhas & Spiller, arXiv:2603.26907). Not "secure-if-either" — XOR already gives that.
    hybrid_combiner_mode: str = "hkdf"


@dataclass
class OrchestrationConfig:
    step_delay: float = 0.4    # seconds between stages (so the SPA DAG animates visibly)
    # How many runs to keep in results/. Every run writes three files there - events.jsonl,
    # summary.json and metrics.csv - and nothing ever removed them: the directory went 953 files to
    # 1133 to 1421 across three rounds, on the VPS as well as locally. It is gitignored, which is
    # precisely why it stayed easy to defer for three rounds running.
    #
    # 0 disables pruning. The default keeps enough history to compare a change against the runs
    # around it without keeping every run this project has ever made.
    keep_runs: int = 200


@dataclass
class WebConfig:
    """Frontend-only knobs (served via /api/config + baked into config.defaults.json) — the in-browser
    AI model lists / generation limits / timeout, plus demo-sim sizes that were inline TS literals."""
    tfjs_models: list = field(default_factory=lambda: [
        {"id": "Xenova/distilgpt2", "label": "distilGPT-2 (tiny, ~80 MB)"},
        {"id": "onnx-community/Qwen2.5-0.5B-Instruct", "label": "Qwen2.5-0.5B-Instruct (ONNX)"}])
    # Download sizes are the MEASURED totals from the MLC model repos' blob listings, not estimates.
    # Ordered smallest-first so the cheapest option is the default: a drive-by visitor should not be
    # handed a 2 GB download to see the page work. SmolLM2-360M is the smallest MLC build that still
    # produces usable text. Every q4f16_1 build needs the WebGPU `shader-f16` feature, which is absent
    # on essentially all Qualcomm GPUs, so a q4f32_1 build is listed as the fallback.
    webllm_models: list = field(default_factory=lambda: [
        {"id": "SmolLM2-360M-Instruct-q4f16_1-MLC", "label": "SmolLM2-360M (204 MB, needs shader-f16)",
         "bytes": 204_289_520, "requires": "shader-f16"},
        {"id": "SmolLM2-360M-Instruct-q4f32_1-MLC", "label": "SmolLM2-360M f32 (376 MB, no shader-f16)",
         "bytes": 376_000_000, "requires": ""},
        {"id": "Llama-3.2-1B-Instruct-q4f16_1-MLC", "label": "Llama-3.2-1B (695 MB, needs shader-f16)",
         "bytes": 695_242_752, "requires": "shader-f16"},
        {"id": "Qwen2.5-0.5B-Instruct-q4f16_1-MLC", "label": "Qwen2.5-0.5B (needs shader-f16)",
         "bytes": 0, "requires": "shader-f16"},
        {"id": "Phi-3-mini-4k-instruct-q4f16_1-MLC", "label": "Phi-3-mini-4k (~2 GB, needs shader-f16)",
         "bytes": 2_000_000_000, "requires": "shader-f16"}])
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
