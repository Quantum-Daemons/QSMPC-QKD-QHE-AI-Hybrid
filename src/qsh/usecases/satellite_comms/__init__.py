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

from dataclasses import dataclass

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
    # air-mass atmosphere and Micius's own 0.8 / 0.1 / vacuum source gave 671 at 20e6, and per pulse
    # sent (Round 78) that is 335 - still above the 256 a key needs, and 167 at 10e6, below it.
    # SINCE ROUND 79 THE KEY IS NOT THIS BLOCK'S: it is the finite key of a block of link time (one
    # pass by default, `sat_block` below), and these pulses are the run's sample at its range, which
    # measures the QBER and decides the abort.
    "sat_pulses": 20_000_000,         # sampled channel uses at the range (the QBER and the abort)
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
    "sat_field_stop_urad": None,      # None -> the profile's; 0 = no turbulence coupling (Round 78)
    # Round 79: the block the composable key is computed on - one pass, passes pooled, or seconds at
    # this range - and its security parameter. None -> QKDConfig.free_space_block*, where the choice
    # and its sources are recorded.
    "sat_block": None,
    "sat_block_passes": None,
    "sat_block_seconds": None,
    "sat_eps_sec": None,
    # The satellite's own statistics (the #78 fix): None -> QKDConfig.free_space_finite_key_deviation,
    # Chernoff, not the fibre's union of two, which a pass's small counts make the looser choice.
    "sat_fk_deviation": None,
}
_DEFAULT_TELEMETRY = (
    "SAT-OK;orbit=LEO;alt=550km;batt=98%;temp=21C;payload=nominal;ts=T+00:42:17"
)


def _sat_profile(ctx: Context) -> dict:
    """The selected downlink profile's values. The default profile's come from `QKDConfig`, so
    config.yaml can still retune them; any other is read from its JSON file. Twin of
    `clientCompute.ts::satProfile`."""
    from ...config import get_config
    from ...config.profiles import OPTIONAL, REQUIRED, profile_values

    qkd = get_config().qkd
    pid = ctx.config.get("sat_profile", _SAT_DEFAULTS["sat_profile"]) or qkd.free_space_profile
    optional = {k: getattr(qkd, f"free_space_{k}") for k in OPTIONAL["downlink"]}
    if pid == qkd.free_space_profile:
        return {**optional, **{k: getattr(qkd, f"free_space_{k}") for k in REQUIRED["downlink"]}}
    return {**optional, **profile_values("downlink", str(pid))}


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
            field_stop_urad=pick("sat_field_stop_urad", v["field_stop_urad"]),
            seeing_arcsec=float(v["seeing_arcsec"]),
            wavelength_nm=float(v["wavelength_nm"]),
        ),
    )


def _sat_source(ctx: Context):
    """The downlink source as `run_bb84` takes it: the profile's intensities, whether it sends the
    vacuum state, and how often each state is sent. Micius: 0.8 / 0.1 / vacuum at 50 / 25 / 25.
    Switching the vacuum off hands its share to the decoy, as `micius_2017_one_decoy` does."""
    from ...qkd.bb84 import QKDParams

    v = _sat_profile(ctx)
    flag = ctx.config.get("sat_vacuum_decoy", _SAT_DEFAULTS["sat_vacuum_decoy"])
    p_signal = float(v["p_signal"])
    # A VACUUM THE SOURCE SENDS NO PULSES FOR CANNOT BE MEASURED (Round 79). Switching the vacuum on
    # for a profile whose shares leave it none - `micius_2017_one_decoy`'s 50 / 50 - gave a source that
    # never sent the vacuum and a bound that credited its measured yield; Lim et al.'s finite-key
    # bound refuses that source outright. The switch now holds only where the profile sends the vacuum.
    vacuum = bool(v["vacuum_decoy"] if flag is None else flag) and 1.0 - p_signal - float(v["p_decoy"]) > 1e-12
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


@dataclass(frozen=True)
class _Block:
    """The block a run's composable key is computed on: `copies` of a `unit` pooled into one.

    The unit is one pass for "pass" and "passes" and one second at the run's range for "seconds";
    expected counts are linear in the pulses, so the block's counts are the unit's times `copies`,
    exactly, and `blocks_for_positive_key` can ask how many units a key needs.
    """

    kind: str                       # "pass" | "passes" | "seconds"
    copies: float                   # passes pooled, or seconds held; 1 for one pass
    unit_seconds: float
    unit: tuple                     # (pulses, channel) segments of one unit
    max_elevation_deg: float | None

    @property
    def seconds(self) -> float:
        return self.copies * self.unit_seconds

    @property
    def pulses(self) -> float:
        return self.copies * sum(n for n, _ in self.unit)


def _sat_block(ctx: Context, params) -> _Block:
    """The run's block: one pass peaking at the run's range (default), several pooled, or seconds."""
    from ...config import get_config
    from ...qkd.satellite_pass import simulate_pass, top_of_pass_deg

    q = get_config().qkd
    kind = str(_cfg_or(ctx, "sat_block", q.free_space_block))
    if kind == "seconds":
        secs = max(1e-3, float(_cfg_or(ctx, "sat_block_seconds", q.free_space_block_seconds)))
        return _Block(kind, secs, 1.0, ((params.source_rate_hz, params),), None)
    if kind not in ("pass", "passes"):
        raise ValueError(f"sat_block {kind!r}: one of 'pass', 'passes', 'seconds'")
    top = top_of_pass_deg(params)
    one = simulate_pass(params, top)
    copies = 1 if kind == "pass" else max(1, int(_cfg_or(ctx, "sat_block_passes", q.free_space_block_passes)))
    return _Block(kind, float(copies), one.duration_s, one.segments, top)


def _cfg_or(ctx: Context, key: str, default):
    """The run's value for `key`, or `default` when the run leaves it unset or None."""
    v = ctx.config.get(key, _SAT_DEFAULTS[key])
    return default if v is None else v


def _sat_finite_key_params(ctx: Context, source):
    """The finite-key bound for this run's source: Lim et al. 2014 when it sends the vacuum, Rusca et
    al. 2018's 1-decoy one when it does not. The asymptotic bound's error-correction efficiency, and
    the security parameter the run sets (Liao's 1e-9 by default)."""
    from ...config import get_config
    from ...qkd.finite_key import params_for_source

    q = get_config().qkd
    return params_for_source(source, eps_sec=float(_cfg_or(ctx, "sat_eps_sec", q.free_space_eps_sec)),
                             deviation=str(_cfg_or(ctx, "sat_fk_deviation", q.free_space_finite_key_deviation)))


class SatelliteQKDStage(Stage):
    """Free-space BB84 over the satellite downlink; derive + fold in the QKD key."""

    name = "satellite_qkd"

    def run(self, ctx: Context, bus: EventBus, reg: CryptoRegistry) -> None:
        from ...config import get_config
        from ...qkd.bb84 import derive_block_key, derive_shared_key, run_bb84
        from ...qkd.channel import link_elevation_deg

        # The PQC-only profile has no quantum channel (Round 80). The fibre stage has skipped there for
        # many rounds and the browser's twin of this stage does; this one ran the downlink anyway and
        # reported a key for a channel the profile says is not there.
        if ctx.config.get("qkd", reg.default("qkd")) == "pqc-only":
            bus.log("free-space QKD disabled (PQC-only profile)", self.name)
            return
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
        # The protocol the page selected (Round 80): this stage ran BB84 whatever `qkd_protocol` said,
        # while the browser's twin ran E91. E91 has no decoy bound and no pass block here, so its key
        # is the asymptotic count, as in the browser.
        protocol = str(ctx.config.get("qkd_protocol", get_config().qkd.protocol))
        if protocol == "e91":
            from ...qkd.e91 import run_e91

            res = run_e91(n_pulses, params, visibility=get_config().qkd.e91_visibility,
                          seed=ctx.config.get("seed", 0))
        else:
            res = run_bb84(n_pulses, params, seed=ctx.config.get("seed", 0), qkd_params=source)
        bus.metric("qber", round(res.qber, 5), self.name)
        bus.metric("qkd_decoy_states", "none (entangled source)" if protocol == "e91"
                   else "vacuum + one decoy" if source.vacuum_decoy else "one decoy", self.name)
        if res.chsh_s is not None:
            bus.metric("chsh_s", round(res.chsh_s, 2), self.name)
        # `skr_per_pulse`, for the same reason as `orchestration/stages.py`: this is bits per
        # channel use, and `skr` is a spec field name meaning bit/s. The satellite twin is
        # renamed in the same change as the fibre one - fixing one and not the other is the
        # asymmetry this round found on the browser side and would otherwise repeat here.
        # Nine decimals since Round 75, at every site: six kept two significant figures of this
        # downlink's 1.723e-5, and the control-plane panel printed it as 1.700e-5.
        bus.metric("skr_per_pulse", round(res.secret_key_rate, 9), self.name)
        bus.metric("channel_loss_db", loss_db, self.name)
        if protocol == "e91":
            key_bits = res.secure_key_bits
            block_note = "(asymptotic: no finite-key bound is modelled for an entangled source)"
        else:
            key_bits, block_note = self._block_key(ctx, bus, params, source, res)
        bus.metric("secure_key_bits", key_bits, self.name)
        # AND THIS LINE, which Round 44 did not rename while writing the comment above saying that
        # renaming one and not the other is the asymmetry it had just found. It repeated it three
        # lines down, in its own twin, under its own warning. The fibre twin at
        # `orchestration/stages.py` renamed both writes; this one renamed the bus metric and left
        # `ctx.metrics`. Nothing shipped a wrong number - only `stages.py` reads `ctx.metrics`, and
        # only for `secure_key_bits`/`qber` - but a per-pulse fraction sat in a key called `skr`,
        # which is the exact collision the round's title claimed to have removed.
        ctx.metrics.update(qber=res.qber, skr_per_pulse=res.secret_key_rate)
        if key_bits >= 256:
            # Folded in, the same way `QKDKeyAgreementStage` does it - see the note there for why
            # this is a re-derivation rather than a stage reorder. This stage's docstring has said
            # "derive + fold in" since it was written while `qkd_key` had no reader at all: the PQC
            # handshake runs before it and reads a key that is not there yet. Fixing one pipeline
            # and not the other would leave the same defect in one language, twice.
            from ...pqc.transport import _mix_qkd

            qkd_key = derive_shared_key(res, 32) if protocol == "e91" else derive_block_key(res, key_bits, 32)
            ctx.put("qkd_key", qkd_key)
            chans = [c for c in (ctx.data.get("channel_initiator"),
                                 ctx.data.get("channel_responder")) if c is not None]
            mode = get_config().crypto.hybrid_combiner_mode
            for chan in chans:
                chan.session_key = _mix_qkd(chan.session_key, qkd_key, mode=mode)
            if chans:
                ctx.put("session_key", chans[0].session_key)
                bus.metric("qkd_mixed", 1, self.name)
                bus.log(f"satellite QKD key derived: {key_bits} secure bits {block_note} "
                        f"(QBER={res.qber:.4f}), folded into the session key via the {mode} "
                        f"combiner", self.name)
            else:
                bus.metric("qkd_mixed", 0, self.name)
                bus.log(f"satellite QKD key derived: {key_bits} secure bits {block_note} "
                        f"(QBER={res.qber:.4f}); no PQC channel to fold it into", self.name)
        else:
            bus.metric("qkd_mixed", 0, self.name)
            bus.log(f"insufficient secure key ({key_bits} bits {block_note}) — "
                    f"link too lossy or eavesdropper present", self.name)

    def _block_key(self, ctx: Context, bus: EventBus, params, source, res) -> tuple[int, str]:
        """The block's composable key length, its metrics, and a phrase saying what the block was.

        THE KEY IS THE BLOCK'S FINITE KEY SINCE ROUND 79 (the user's choice). The run above samples a
        fifth of a second at this range, which measures the QBER and decides the abort; until this
        round the key was the asymptotic bound times those pulses, 335 bits, a block no experiment
        distils a key from. Now it is the finite-key bound on the counts of a block of link time -
        one pass peaking at this range by default - and the sampled run's abort still applies to it.
        """
        import math

        from ...qkd.finite_key import (
            DEVIATION_LABELS,
            VacuumDecoyParams,
            block_counts,
            blocks_for_positive_key,
            one_decoy_key_length,
            vacuum_decoy_key_length,
        )

        block = _sat_block(ctx, params)
        fk_params = _sat_finite_key_params(ctx, source)
        vacuum = isinstance(fk_params, VacuumDecoyParams)
        unit_z, unit_x = block_counts(block.unit, fk_params)
        bound = vacuum_decoy_key_length if vacuum else one_decoy_key_length
        fk = bound(unit_z.scaled(block.copies), unit_x.scaled(block.copies), fk_params)
        fk_bits = int(fk.key_bits) if math.isfinite(fk.key_bits) and fk.key_bits > 0.0 else 0
        aborted = res.secure_key_bits == 0 and res.qber > source.pp_threshold
        bus.metric("qkd_block", block.kind, self.name)
        bus.metric("qkd_block_copies", block.copies, self.name)
        # FOUR SIGNIFICANT FIGURES, at every site that emits it (Round 80): the fibre's blocks run from
        # milliseconds to minutes, where one decimal would print a 20 ms block as 0.0 s.
        bus.metric("qkd_block_s", float(f"{block.seconds:.4g}"), self.name)
        bus.metric("qkd_block_pulses", float(f"{block.pulses:.6g}"), self.name)
        if block.max_elevation_deg is not None:
            bus.metric("sat_pass_max_elevation_deg", round(block.max_elevation_deg, 2), self.name)
        bus.metric("qkd_finite_key_bits", fk_bits, self.name)
        bus.metric("qkd_finite_key_bound", "Lim et al. 2014 (vacuum + weak decoy)" if vacuum
                   else "Rusca et al. 2018 (one decoy)", self.name)
        bus.metric("qkd_eps_sec", fk_params.eps_sec, self.name)
        bus.metric("qkd_fk_deviation", fk_params.deviation, self.name)
        bus.metric("qkd_block_units_for_key", blocks_for_positive_key(unit_z, unit_x, fk_params),
                   self.name)
        what = {"pass": "one pass", "passes": f"{block.copies:g} passes pooled",
                "seconds": f"{block.copies:g} s at this range"}[block.kind]
        note = (f"(finite key over {what}, {float(f'{block.seconds:.4g}')} s of link time, "
                f"eps_sec {fk_params.eps_sec:g}, {DEVIATION_LABELS[fk_params.deviation]})")
        return (0 if aborted else fk_bits), note


class QSDCStage(Stage):
    """Quantum Secure Direct Communication of mission telemetry (inline eavesdrop check)."""

    name = "qsdc_telemetry"

    def run(self, ctx: Context, bus: EventBus, reg: CryptoRegistry) -> None:
        from ...qkd.qsdc import quantum_secure_direct_comm

        # Under the PQC-only profile there is no quantum channel to carry the message (Round 80): the
        # free-space stage stops before it stores one, and the browser's twin skips here with the
        # sentence below.
        if ctx.config.get("qkd", reg.default("qkd")) == "pqc-only":
            ctx.artifacts["telemetry_delivered"] = False
            bus.log("QSDC direct telemetry skipped (PQC-only profile): QSDC carries the message on the "
                    "quantum channel itself, so with no quantum channel there is nothing to send it "
                    "over. The payload travels under the PQC session key instead.", self.name)
            return
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
            bus.log("no delivered telemetry to score (QSDC delivered none) — anomaly check skipped", self.name)
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
