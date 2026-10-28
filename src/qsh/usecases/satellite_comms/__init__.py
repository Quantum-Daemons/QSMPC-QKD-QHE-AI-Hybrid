"""Satellite-communications use case: free-space QKD + QSDC over a LEO downlink.

WHY THIS USE CASE EXISTS, since the line above describes a mechanism and not a problem.
    Operators treat housekeeping telemetry as confidential, and it demonstrably blocks ML. DLR's
    German Space Operations Centre built a synthetic-telemetry pipeline saying so outright - "since
    satellite telemetry data are often strictly confidential, we can share these synthetic data
    easily with our research partners" (CEAS Space Journal, 2025). ESA anonymised missions,
    channels, telecommands and units and scaled the timeline by an undisclosed factor purely to
    publish the benchmark this file consumes. And in 2015 a Space Data Association chairman asked
    for secure computation in plain English: "we still protect our assets without having to know
    what their assets are doing."

    The architectural seam is specific. CCSDS 355.0-B-2 gives confidentiality "over a space link"
    and terminates at the ground station, so telemetry is plaintext for the whole ground segment -
    which is where processing and ML happen. That seam is where encrypted inference sits.

    NOT the reason, and worth stating because it is the first thing anyone reaches for: export
    control does NOT block this. 22 CFR 121.1 USML XV Note 3 explicitly decontrols housekeeping
    telemetry and says encrypting it "does not, in and of itself, cause the telemetry data to become
    subject to the ITAR". The genuine legal hook runs the other way - 22 CFR 120.54(c) states that
    "the ability to access technical data in encrypted form does not constitute the release or
    export", with end-to-end encryption defined as the means of decryption not being provided to any
    third party. Export law already blesses this architecture.

    AND THE HONEST REGISTER, from the same body of evidence. CISA's Space Systems Critical
    Infrastructure Working Group (April 2024) assessed both halves of this stack: homomorphic
    encryption schemes "have yet to be applied to space systems, and their technology readiness level
    remains low", and QKD "has been tested in the space environment, but its technology readiness
    level remains low". That is a US government body describing this as low-TRL in writing. It
    corroborates the gap and it sets the expectation - this is a research demonstrator, not something
    an operator could fly.

    Full workings, including the five motivations to avoid: references/satellite-motivation-fact-check.md


A ground station and a LEO satellite establish a PQC-first hybrid session, distribute a
key over a **free-space BB84** downlink (high loss, atmospheric turbulence, background
counts), then transmit mission telemetry by **Quantum Secure Direct Communication** (QSDC,
report §6) — a direct message transfer with an inline eavesdrop check, which suits a
satellite link where re-keying round-trips are costly. The delivered telemetry is also
sealed in the AES-256-GCM session for an auditable, signed record.

Maps to: WS1 (AI-QKD, here over a free-space channel) + the report's QSDC / QT-QSS section.
Free-space channel and the QSDC threshold are config-driven (WebUI-adjustable via ``config``).

Run:  ``python -m qsh.orchestration.pipeline --usecase satellite_comms``
"""

from __future__ import annotations

from ...orchestration.crypto_agility import CryptoRegistry
from ...orchestration.events import EventBus
from ...orchestration.pipeline import Pipeline
from ...orchestration.stages import (
    Context,
    OptiQKDStage,
    PQCHandshakeStage,
    QuantumComparisonStage,
    ResultStage,
    SignatureStage,
    Stage,
)

# Free-space LEO downlink defaults (overridable per-run from the WebUI via ctx.config).
#
# `sat_loss_db_per_km` is GONE. It was 0.035, a linear rate chosen to give ~28 dB at 800 km, and the
# browser rejected that model rounds ago: it reads plausibly at exactly 800 km beside the ~27 dB
# Micius reports and then diverges badly with range. The channel now carries a FreeSpaceParams and
# `ChannelParams.loss_db()` applies the diffraction law, so both implementations answer 24.48 dB for
# the same nominal link instead of 24.48 and 28.0.
_SAT_DEFAULTS = {
    # None -> QKDConfig.free_space_*: these three were literal copies of the config (0.4, 1e-5,
    # 0.025) until Round 75, so retuning the config moved the browser and left this stage behind.
    # The values are Liao et al. 2017's Micius downlink; see `QKDConfig`.
    "sat_distance_km": None,          # None -> QKDConfig.free_space_distance_km
    "sat_detector_efficiency": None,  # None -> QKDConfig.free_space_detector_efficiency
    "sat_dark_count_prob": None,      # None -> QKDConfig.free_space_dark_count_prob
    "sat_misalignment": None,         # None -> QKDConfig.free_space_misalignment
    "sat_eve": 0.0,                   # intercept-resend fraction (0 = no eavesdropper)
    # 10e6, raised with the decoy-model switch: at 4e6 this downlink yielded 223 secure bits,
    # under the 256 the stage needs to derive a key, so satellite silently stopped mixing QKD.
    # Kept in step with `WebConfig.demo_sat_pulses`; see the note there for the finite-key residual.
    # 20e6 since Round 75: the Micius budget gave 344 secure bits there and 172 at 10e6. Round 77's
    # air-mass atmosphere and Micius's own 0.8 / 0.1 / vacuum source give 671 at 20e6.
    "sat_pulses": 20_000_000,         # channel uses (free-space links need many)
    "sat_opt_pulses": 200_000,        # pulses per OptiQKD evaluation (fewer than a full run, for speed)
    # The two fixed terms of the budget, exposed in the WebUI so the inline eavesdrop check can be
    # demonstrated: they floor total loss at 8 dB by default, which makes every abort attributable to
    # the link rather than to Eve at every setting the page used to offer.
    "sat_atmospheric_db": None,       # None -> the profile's atmospheric_db (at its reference range)
    "sat_pointing_db": None,          # None -> the profile's pointing_db
    # Round 77. The link a run models, by profile id (src/qsh/config/profiles/downlink/); its values
    # are the base every key above overrides. None -> QKDConfig.free_space_profile, Micius 2017.
    "sat_profile": None,
    "sat_min_elevation_deg": None,    # None -> the profile's; no link below it (Liao: 10 degrees)
    "sat_vacuum_decoy": None,         # None -> the profile's; whether the vacuum state is sent
}
_DEFAULT_TELEMETRY = (
    "SAT-OK;orbit=LEO;alt=550km;batt=98%;temp=21C;payload=nominal;ts=T+00:42:17"
)


def _sat_profile(ctx: Context) -> dict:
    """The selected downlink profile's values. The default profile's come from `QKDConfig`, so
    config.yaml can still retune them; any other is read from its JSON file. Twin of
    `clientCompute.ts::satProfile`."""
    from ...config import get_config
    from ...config.profiles import REQUIRED, profile_values

    qkd = get_config().qkd
    pid = ctx.config.get("sat_profile", _SAT_DEFAULTS["sat_profile"]) or qkd.free_space_profile
    if pid == qkd.free_space_profile:
        return {k: getattr(qkd, f"free_space_{k}") for k in REQUIRED["downlink"]}
    return profile_values("downlink", str(pid))


def _sat_channel(ctx: Context):
    from ...qkd.channel import ChannelParams, FreeSpaceParams

    v = _sat_profile(ctx)
    g = lambda k: ctx.config.get(k, _SAT_DEFAULTS[k])  # noqa: E731
    pick = lambda k, fallback: float(fallback if g(k) is None else g(k))  # noqa: E731
    return ChannelParams(
        distance_km=pick("sat_distance_km", v["distance_km"]),
        detector_efficiency=pick("sat_detector_efficiency", v["detector_efficiency"]),
        dark_count_prob=pick("sat_dark_count_prob", v["dark_count_prob"]),
        misalignment=pick("sat_misalignment", v["misalignment"]),
        eve_intercept_fraction=float(g("sat_eve")),
        source_rate_hz=float(v["source_rate_hz"]),
        dead_time_s=float(v["detector_dead_time_s"]),
        free_space=FreeSpaceParams(
            beam_divergence_urad=float(v["beam_divergence_urad"]),
            rx_aperture_m=float(v["rx_aperture_m"]),
            atmospheric_db=pick("sat_atmospheric_db", v["atmospheric_db"]),
            pointing_db=pick("sat_pointing_db", v["pointing_db"]),
            receiver_optics_efficiency=float(v["receiver_optics_efficiency"]),
            atmospheric_ref_km=float(v["atmospheric_ref_km"]),
            orbit_altitude_km=float(v["orbit_altitude_km"]),
            station_altitude_km=float(v["station_altitude_km"]),
            min_elevation_deg=pick("sat_min_elevation_deg", v["min_elevation_deg"]),
        ),
    )


def _sat_source(ctx: Context):
    """The downlink source as `run_bb84` takes it: the profile's intensities, whether it sends the
    vacuum state, and how often each state is sent. Micius: 0.8 / 0.1 / vacuum at 50 / 25 / 25.
    Switching the vacuum off hands its share to the decoy, as `micius_2017_one_decoy` does."""
    from ...qkd.bb84 import QKDParams

    v = _sat_profile(ctx)
    flag = ctx.config.get("sat_vacuum_decoy", _SAT_DEFAULTS["sat_vacuum_decoy"])
    vacuum = bool(v["vacuum_decoy"] if flag is None else flag)
    p_signal = float(v["p_signal"])
    p_decoy = float(v["p_decoy"]) if vacuum else 1.0 - p_signal
    return QKDParams(mu_signal=float(v["mu_signal"]), mu_decoy=float(v["mu_decoy"]),
                     vacuum_decoy=vacuum, p_signal=p_signal, p_decoy=p_decoy)


def _out_of_view(params) -> str | None:
    """Why this downlink is not operated, or None when it is."""
    from ...qkd.channel import in_view, link_elevation_deg

    fs = params.free_space
    if in_view(fs, params.distance_km):
        return None
    e = link_elevation_deg(fs, params.distance_km)
    where = ("beyond the horizon" if e is None
             else f"at {e:.1f} deg elevation, below the {fs.min_elevation_deg:g} deg minimum")
    return (f"no link at {params.distance_km:.0f} km: {where} for a {fs.orbit_altitude_km:g} km "
            f"orbit. A station this far from the satellite needs a relay - the page's Network tab")


class SatelliteQKDStage(Stage):
    """Free-space BB84 over the satellite downlink; derive + fold in the QKD key."""

    name = "satellite_qkd"

    def run(self, ctx: Context, bus: EventBus, reg: CryptoRegistry) -> None:
        from ...qkd.bb84 import derive_shared_key, run_bb84
        from ...qkd.channel import link_elevation_deg

        params = _sat_channel(ctx)
        ctx.put("sat_channel", params)
        source = _sat_source(ctx)
        ctx.put("sat_qkd_params", source)
        n_pulses = int(ctx.config.get("sat_pulses", _SAT_DEFAULTS["sat_pulses"]))
        elevation = link_elevation_deg(params.free_space, params.distance_km)
        if elevation is not None:
            bus.metric("sat_elevation_deg", round(elevation, 2), self.name)
        why = _out_of_view(params)
        bus.metric("sat_in_view", int(why is None), self.name)
        if why is not None:
            # Not an infinite loss on the page: a sentence saying why there is no link (Round 77).
            bus.metric("skr_per_pulse", 0.0, self.name)
            bus.metric("secure_key_bits", 0, self.name)
            bus.metric("qkd_mixed", 0, self.name)
            ctx.metrics.update(skr_per_pulse=0.0)
            bus.log(f"free-space downlink: {why}", self.name)
            return
        loss_db = round(params.loss_db(), 1)
        bus.log(f"free-space downlink: {params.distance_km:.0f} km, ~{loss_db} dB loss, "
                f"eve={params.eve_intercept_fraction:.2f}", self.name)
        # `qkd_params`, for the same reason the registry entry carries `QKDParams()`: the default
        # branch is the NON-decoy GLLP form, and this stage's own use case page labels its rate a
        # decoy-state bound. Since Round 77 it is the downlink's OWN source (`_sat_source`): Micius's
        # 0.8 / 0.1 with the vacuum state, where it ran the fibre page's finite-key optimum.
        res = run_bb84(n_pulses, params, seed=ctx.config.get("seed", 0), qkd_params=source)
        bus.metric("qber", round(res.qber, 5), self.name)
        # `skr_per_pulse`, for the same reason as `orchestration/stages.py`: this is bits per
        # channel use, and `skr` is a spec field name meaning bit/s. The satellite twin is
        # renamed in the same change as the fibre one - fixing one and not the other is the
        # asymmetry this round found on the browser side and would otherwise repeat here.
        # Nine decimals since Round 75, at every site: six kept two significant figures of this
        # downlink's 1.723e-5, and the control-plane panel printed it as 1.700e-5.
        bus.metric("skr_per_pulse", round(res.secret_key_rate, 9), self.name)
        bus.metric("secure_key_bits", res.secure_key_bits, self.name)
        bus.metric("channel_loss_db", loss_db, self.name)
        # AND THIS LINE, which Round 44 did not rename while writing the comment above saying that
        # renaming one and not the other is the asymmetry it had just found. It repeated it three
        # lines down, in its own twin, under its own warning. The fibre twin at
        # `orchestration/stages.py` renamed both writes; this one renamed the bus metric and left
        # `ctx.metrics`. Nothing shipped a wrong number - only `stages.py` reads `ctx.metrics`, and
        # only for `secure_key_bits`/`qber` - but a per-pulse fraction sat in a key called `skr`,
        # which is the exact collision the round's title claimed to have removed.
        ctx.metrics.update(qber=res.qber, skr_per_pulse=res.secret_key_rate)
        if res.secure_key_bits >= 256:
            # Folded in, the same way `QKDKeyAgreementStage` does it - see the note there for why
            # this is a re-derivation rather than a stage reorder. This stage's docstring has said
            # "derive + fold in" since it was written while `qkd_key` had no reader at all: the PQC
            # handshake runs before it and reads a key that is not there yet. Fixing one pipeline
            # and not the other would leave the same defect in one language, twice.
            from ...config import get_config
            from ...pqc.transport import _mix_qkd

            qkd_key = derive_shared_key(res, 32)
            ctx.put("qkd_key", qkd_key)
            chans = [c for c in (ctx.data.get("channel_initiator"),
                                 ctx.data.get("channel_responder")) if c is not None]
            mode = get_config().crypto.hybrid_combiner_mode
            for chan in chans:
                chan.session_key = _mix_qkd(chan.session_key, qkd_key, mode=mode)
            if chans:
                ctx.put("session_key", chans[0].session_key)
                bus.metric("qkd_mixed", 1, self.name)
                bus.log(f"satellite QKD key derived: {res.secure_key_bits} secure bits "
                        f"(QBER={res.qber:.4f}), folded into the session key via the {mode} "
                        f"combiner", self.name)
            else:
                bus.metric("qkd_mixed", 0, self.name)
                bus.log(f"satellite QKD key derived: {res.secure_key_bits} secure bits "
                        f"(QBER={res.qber:.4f}); no PQC channel to fold it into", self.name)
        else:
            bus.metric("qkd_mixed", 0, self.name)
            bus.log(f"insufficient secure key ({res.secure_key_bits} bits) — "
                    f"link too lossy or eavesdropper present", self.name)


class QSDCStage(Stage):
    """Quantum Secure Direct Communication of mission telemetry (inline eavesdrop check)."""

    name = "qsdc_telemetry"

    def run(self, ctx: Context, bus: EventBus, reg: CryptoRegistry) -> None:
        from ...qkd.qsdc import quantum_secure_direct_comm

        params = ctx.data.get("sat_channel")
        telemetry = str(ctx.config.get("telemetry", _DEFAULT_TELEMETRY)).encode()
        why = _out_of_view(params) if params is not None and params.free_space is not None else None
        if why is not None:
            ctx.artifacts["telemetry_delivered"] = False
            bus.metric("qsdc_link_budget_limited", 1, self.name)
            bus.metric("qsdc_eavesdrop_detected", 0, self.name)
            bus.log(f"QSDC NOT AVAILABLE: {why}. Telemetry falls back to the PQC session key.",
                    self.name)
            return
        res = quantum_secure_direct_comm(
            telemetry, params, seed=ctx.config.get("seed", 0))

        bus.metric("qsdc_error_rate", res.error_rate, self.name)
        bus.metric("qsdc_secrecy_capacity", res.secrecy_capacity, self.name)
        bus.metric("qsdc_eve_gain_g", res.eve_gain_g, self.name)
        bus.metric("qsdc_link_budget_limited", int(res.link_budget_limited), self.name)
        bus.metric("qsdc_eavesdrop_detected", int(res.eavesdrop_detected), self.name)
        bus.metric("qsdc_message_bits", res.message_bits, self.name)

        if res.delivered is None:
            ctx.artifacts["telemetry_delivered"] = False
            if res.link_budget_limited:
                bus.log(f"QSDC NOT AVAILABLE on this link — secrecy capacity is not positive at "
                        f"{res.loss_db:.1f} dB (g={res.eve_gain_g:.1f}) even with no eavesdropper. "
                        f"The limit is the link budget, not an attack; error {res.error_rate:.4f}. "
                        f"Telemetry falls back to the PQC session key.", self.name)
            else:
                bus.log(f"QSDC ABORTED: secrecy capacity exhausted at error {res.error_rate:.4f} "
                        f"(g={res.eve_gain_g:.1f}, {res.loss_db:.1f} dB) — this link closes without "
                        f"Eve, so the loss is attributable — telemetry withheld", self.name)
            return

        # also seal the delivered telemetry in the AEAD session for an audit record
        chan = ctx.data.get("channel_initiator")
        sealed = chan.seal(res.delivered) if chan else None
        ctx.artifacts["telemetry_delivered"] = True
        ctx.artifacts["telemetry_preview"] = res.delivered.decode(errors="replace")[:80]
        if sealed is not None:
            ctx.artifacts["telemetry_sealed_bytes"] = len(sealed)
        bus.log(f"QSDC delivered {res.message_bits} bits "
                f"(secrecy capacity {res.secrecy_capacity:.3f} bit/carrier, "
                f"error {res.error_rate:.3f})", self.name)


class TelemetryAnomalyStage(Stage):
    """Private satellite-telemetry anomaly detection on the *encrypted* downlinked telemetry (CKKS).

    After QSDC delivers the mission telemetry, a real (FHE-friendly linear) anomaly detector — trained
    on ESA-ADB-style multi-channel telemetry — scores the encrypted window and returns a signed
    nominal/anomaly verdict, so a relay/ground partner can flag faults without seeing the raw telemetry.
    """

    name = "telemetry_anomaly"

    def run(self, ctx: Context, bus: EventBus, reg: CryptoRegistry) -> None:
        import numpy as np

        from ...ai.explainability import explain_linear_model
        from ...fhe.ckks_inference import encrypted_linear_inference

        model = ctx.data.get("anomaly_model")
        x = ctx.data.get("features")
        if model is None or x is None or not ctx.artifacts.get("telemetry_delivered"):
            bus.log("no delivered telemetry to score (link aborted) — anomaly check skipped", self.name)
            return
        backend = reg.get("fhe", ctx.config.get("fhe"))()
        out = encrypted_linear_inference(model, np.asarray(x), backend)
        ctx.artifacts["telemetry_anomaly"] = int(out["prediction"])
        bus.metric("telemetry_anomaly_detected", int(out["prediction"]), self.name)
        names = ctx.data.get("feature_names")
        exp = explain_linear_model(model, np.asarray(x), baseline=ctx.data.get("explain_baseline"),
                                   feature_names=names, top_k=3)
        ctx.artifacts["explanation"] = {
            "method": exp.method, "prediction": exp.prediction,
            "top_features": [{"feature": t["feature"], "contribution": round(float(t["contribution"]), 6)}
                             for t in exp.top],
        }
        bus.log(f"private telemetry anomaly detection (CKKS): "
                f"{'ANOMALY' if out['prediction'] else 'nominal'} "
                f"(err {out['max_abs_error']:.2e})", self.name)


def _anomaly_model(seed: int = 0):
    """Train the FHE-friendly linear telemetry-anomaly classifier + pick an anomalous window to score."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from ...data.loaders import load_satellite_telemetry
    from ...fhe.ckks_inference import LinearModel

    X, y, names = load_satellite_telemetry(n_windows=600, seed=seed)
    clf = LogisticRegression(max_iter=500, class_weight="balanced").fit(X, y)
    w = np.vstack([np.zeros_like(clf.coef_[0]), clf.coef_[0]])
    b = np.array([0.0, float(clf.intercept_[0])])
    anom = X[(y == 1) & (clf.predict(X) == 1)]
    window = (anom[0] if len(anom) else X[y == 1][0]).astype(float)
    win, ch = 8, len(names)
    feat_names = [f"{names[c]}[t{tt}]" for c in range(ch) for tt in range(win)]
    # E[x] over the telemetry windows. Real NASA SMAP/MSL channels are NOT zero-mean, so a zero
    # baseline made ranks 2 and 3 of the reported top-3 spurious - the attribution was measured
    # against a spacecraft state that never occurs.
    baseline = X.mean(axis=0).astype(float)
    return LinearModel(w.astype(float), b), window, feat_names, baseline


def build(config: dict | None = None) -> tuple[Pipeline, Context]:
    config = dict(config or {})
    ctx = Context(
        usecase="satellite_comms",
        config={"qkd": config.get("qkd", "bb84-decoy"), "inference": "ckks", **config},
    )
    model, window, feat_names, baseline = _anomaly_model(config.get("seed", 0))
    ctx.data["anomaly_model"] = model
    ctx.data["features"] = window
    ctx.data["feature_names"] = feat_names
    ctx.data["explain_baseline"] = baseline
    ctx.config.setdefault("qkd_opt_pulses", _SAT_DEFAULTS["sat_opt_pulses"])   # free-space link needs more pulses
    stages = [
        PQCHandshakeStage(),
        SatelliteQKDStage(),
        OptiQKDStage(),                 # AI-QKD optimization over the free-space channel (opt-in)
        QSDCStage(),
        TelemetryAnomalyStage(),
        QuantumComparisonStage(),
        ResultStage(),
        SignatureStage(),
    ]
    return Pipeline(name="satellite-comms", stages=stages), ctx


def run_demo(config: dict | None = None) -> dict:
    pipe, ctx = build(config)
    pipe.run(ctx, out_dir="results/")
    m = pipe.bus.metrics()
    return {
        "qber": m.get("qber"),
        "secure_key_bits": m.get("secure_key_bits"),
        "channel_loss_db": m.get("channel_loss_db"),
        "telemetry_delivered": ctx.artifacts.get("telemetry_delivered"),
        "qsdc_secrecy_capacity": m.get("qsdc_secrecy_capacity"),
        "eavesdrop_detected": bool(m.get("qsdc_eavesdrop_detected")),
        "telemetry_anomaly": ctx.artifacts.get("telemetry_anomaly"),   # private CKKS anomaly verdict (1=anomaly)
        "explanation": ctx.artifacts.get("explanation"),
        "metrics": m,
    }
