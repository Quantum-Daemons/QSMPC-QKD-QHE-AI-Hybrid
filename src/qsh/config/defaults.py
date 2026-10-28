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
    # BLOCK SIZE, and it is a security parameter rather than a speed knob.
    #
    # This was 200_000, giving `nSift = N * Q_mu * 0.5` = **1,244 sifted bits** at the shipped 50 km
    # channel. Round 46 raised it to 2_000_000 (**12,442 sifted bits**) and Round 47 raised it again,
    # to 20_000_000. The threshold Round 46 measured against was wrong twice over, so that is
    # corrected first and the current value is justified against a computed number afterwards.
    #
    # WHAT THIS COMMENT CLAIMED UNTIL ROUND 47. It read: "the positive-key threshold for these device
    # parameters is ~5e3 to 1.1e4 sifted bits (Rusca, Boaron, Grunenfelder, Martin & Zbinden, Appl.
    # Phys. Lett. 112, 171104 (2018), 1-decoy finite-key bound)". Both halves of that sentence failed.
    #
    #   THE ATTRIBUTION. That paper states no positivity threshold anywhere. It compares the 1-decoy
    #   against the 2-decoy protocol across privacy-amplification block sizes n_Z from 1e5 to 1e11,
    #   and its own conclusion is that the 1-decoy approach wins "unless a really big (> 10^11) or
    #   really small (< 10^5) PA block size is applied". A number was attributed to a paper that does
    #   not contain it, and the citation is what made it look checked for three rounds.
    #
    #   THE NUMBER. `src/qsh/qkd/finite_key.py` now implements that paper's own equations, and
    #   evaluated on THIS project's 50 km channel at the intensities shipped until Round 47
    #   (mu = 0.5, nu = 0.1) they put the positive-key threshold near **1e5 sifted bits, about 4e7
    #   pulses** - roughly an order of magnitude above the band stated here. The finite-key optimum
    #   for the intensities lowers the pulse count substantially; that is a separate change, measured
    #   in the note on `mu_signal` below, and it does not move the order of magnitude in sifted bits.
    #   At Round 46's 2_000_000 pulses the single-photon lower bound s^l_{Z,1} of Eq. (4) comes out
    #   NEGATIVE, which means no composable key existed at that block size under ANY setting of the
    #   intensities, their probabilities or the basis bias - searched, not assumed; see
    #   `finite_key.optimise_parameters`, which optimises exactly the variables the paper says it
    #   optimises.
    #
    # THE PUBLISHED FIGURE THAT AGREES WITH THE CORRECTION WAS ALREADY QUOTED HERE, AND IT REFUTED
    # THE BAND IT WAS ATTACHED TO. Lucamarini et al., Opt. Express 21, 24550 (2013), Fig. 2 caption:
    # "1.4x10^5 counts ... is the smallest sample size providing a positive rate" at 50 km. That
    # stood one line below a band 13 to 28 times smaller, labelled "corroborating primary datum". It
    # never corroborated that band; it contradicted it in the next sentence, and it is within an
    # order of magnitude of what the equations compute. It is kept here as the published agreement
    # with the corrected threshold, which is the only thing it was ever evidence for.
    #
    # HOW ACCURATE OUR OWN THRESHOLD IS, stated wherever it is quoted because a computed number
    # deserves its error bar. Validated against Rusca et al.'s published Table I - secret key rate at
    # n_Z = 1e7 for 26 / 46 / 56 / 64 dB - this implementation is CONSERVATIVE by roughly a factor of
    # 2-3 at moderate loss and by more at high loss, even after optimising the intensities and the
    # basis bias the way the paper describes. The threshold it computes is therefore an UPPER bound
    # on what is really required, and it is quoted as an order of magnitude ("~1e5 sifted bits, about
    # 4e7 pulses") rather than to two figures it has not earned. Three claims survive that
    # uncertainty untouched: Round 46's 2_000_000-pulse block carried no composable key, because a
    # factor of 2-3 cannot rescue a NEGATIVE single-photon count; the band stated here was far too
    # small; and the citation was wrong regardless of any numerics. The margin at the CURRENT block
    # size is the one place the 2-3x matters, which is why it is sized below rather than taken at the
    # threshold.
    #
    # WHY IT IS 20_000_000, against a threshold this repository now COMPUTES rather than cites.
    #
    # At the intensities that were shipped (mu = 0.5, nu = 0.1) the threshold is 3.84e7 pulses, and
    # the Python Monte Carlo allocates ~2.0 GB there - too much to put behind an interactive
    # endpoint. But mu ~ 0.5 is the optimum for the ASYMPTOTIC rate, not for the composable key
    # length, and those optima are in different places. Moving the intensities to the finite-key
    # optimum below (mu = 0.3416, nu = 0.1417) brings the threshold down to **1.41e7 pulses**, which
    # 2e7 clears by 1.42x.
    #
    # AND ONLY TWO OF THE FOUR OPTIMISED VARIABLES ARE SHIPPED, which is why the figure is 1.41e7
    # and not the 1.31e7 an earlier revision of this comment claimed. `optimise_parameters` searches
    # the four variables Rusca et al. name - mu, nu, p_mu1 and p_Z - and its solution also sets
    # p_signal = 0.3764 and p_Z = 0.5042. Neither has a shipped counterpart: the asymptotic decoy
    # code takes no intensity probability at all, and `QKDParams.basis_prob` is 0.5. So the
    # threshold this configuration actually reaches is the one computed at p_signal = p_Z = 0.5,
    # and quoting the optimiser's full-solution figure beside a config that ships half of it was
    # the same defect this round is named for, in the document rather than the code.
    #
    # `tests/test_every_block_size_clears_the_band_the_config_names.py` computes the threshold from
    # the shipped config rather than from this comment, so the two cannot drift.
    #
    # Measured cost of the Monte Carlo at 2e7: 0.66 s and ~1.05 GB peak, against 0.08 s and 0.15 GB
    # at 2e6. Free either way in the browser: `bb84()` in `quantumSim.ts` is analytic, not a
    # per-pulse loop, so the block size costs the visitor nothing at all.
    #
    # What the demo publishes at 50 km remains an ASYMPTOTIC decoy bound, labelled as one on every
    # surface. The finite-key length is shown BESIDE it rather than replacing it, because they are
    # different quantities and the page says which is which.
    pulses: int = 20_000_000
    distance_km: float = 50.0
    # SOURCE REPETITION RATE, in Hz. THE ONLY ASSUMED QUANTITY IN THE QKD MODEL, and it is assumed
    # rather than derived because it is a property of a transmitter this project does not own.
    #
    # Everything else here is computed. `secret_key_rate` is bits PER PULSE in all four models
    # (`quantumSim.ts` bb84/e91, `bb84.py`, `e91.py`, `decoy_reference.py`), and `pulses` is a count,
    # so nothing in the simulation has ever had a time axis. ETSI GS QKD 015 defines `skr`/`eskr` as
    # a secret key RATE in bit/s. Round 42 found a session TOTAL sitting in those fields and set them
    # to null, correctly, because bit/s = (bits per pulse) x (pulses per second) and the second
    # factor did not exist.
    #
    # It still does not exist as a measurement. What changed in Round 44 is that it is now an
    # explicit, labelled INPUT instead of an absence: one scalar, editable on the page, defaulting to
    # a published figure, with the UI stating that it describes apparatus this project does not have.
    # That is the honest form - the alternative was to keep a spec field permanently empty because
    # the one number it needed was never asked for.
    #
    # Round 42's own conclusion was broader than its evidence: "No library supplies a repetition rate
    # because none can." Three candidates were surveyed and rejected on three DIFFERENT grounds -
    # TNO-Quantum for being per-pulse, QKDNetSim on GPL licence before its physics was examined,
    # SeQUeNCe for "would still require asserting a clock". Only the first is the ground the
    # generalisation rests on. The correct statement is narrower and survives: a repetition rate is
    # an INPUT to key-rate analysis, stated in a paper's experimental section, not an output of it -
    # which is why asserting one openly is the fix rather than a compromise.
    #
    # DEFAULT: 1 GHz since Round 76, Rusca, Boaron, Grunenfelder, Martin & Zbinden, Appl. Phys. Lett.
    # 112, 171104 (2018), arXiv:1801.03443: "We consider a system working at a repetition rate of
    # 1 GHz". It moved with the rest of the fibre set below, which is that paper's: the clock is part
    # of the detector model now, because the dead time turns a detection rate into lost detections.
    #
    # Until Round 76 it was 100 MHz, borrowed from Park, Kim, Im, Jo, Kim, Choi & Ihn,
    # arXiv:2607.24118 - a 2 km FREE-SPACE link - with the warning that the clock was borrowed and the
    # rest of that experiment was not. This comment also said the fibre default would move "if a fibre
    # demonstration with a stated repetition rate ever enters" the references; Lucamarini et al. 2013
    # (1 GHz over 50 km of fibre) had been cited in this file for the block size all along.
    source_rate_hz: float = 1_000_000_000.0
    # The downlink's clock, Liao et al.'s own: "laser pulses (848.6 nm, 100 MHz, 0.2 ns)" (Nature 549,
    # 43, 2017), quoted in tests/test_the_satellite_channel_is_sourced.py. This comment credited
    # arXiv:2607.24118's 2 km link until Round 76, after Round 75 had already sourced the value from
    # Liao. A second key, because the two links' clocks have different sources.
    #
    # A "2.5 GHz in the 64 Mbps fibre demonstration" figure once sat in `clientSdnController.ts` with
    # no citation. It is Grunenfelder et al., Nature Photonics 17, 422 (2023), arXiv:2210.16126: 64 Mbps
    # over 10.0 km at 2.5 GHz, from the same group as the fibre set below, on ultra-low-loss fibre with
    # 64-85% efficient multi-pixel SNSPDs. Not adopted: its detectors and fibre are not the model's.
    free_space_source_rate_hz: float = 100_000_000.0
    # THE DETECTORS' DEAD TIME, in seconds, the term that makes a GHz clock honest. Rusca et al.:
    # SNSPDs "which have a dead-time tau_DT = 100 ns", with the correction c_dt = 1 / (1 + R P t_DT) of
    # their Eq. (B3) (`qsh.qkd.channel.dead_time_factor`). At 1 GHz on the 50 km default it removes
    # 37% of the detections (c_dt = 0.625). This project modelled no dead time before Round 76; with
    # the old set at 100 MHz the same 100 ns would have cost 2.9%.
    detector_dead_time_s: float = 100e-9
    # Liao et al. state none for the downlink. At its ~1.7e3 detections per second a dead time under a
    # microsecond costs under 0.1%, so none is modelled rather than one invented.
    free_space_detector_dead_time_s: float = 0.0
    # THE FIBRE SET IS RUSCA ET AL. 2018's (Round 76, the user's choice): the stated parameters of the
    # paper whose 1-decoy finite-key analysis `qsh/qkd/finite_key.py` implements, and which
    # `tests/test_finite_key_reproduces_the_published_table.py` already used to check it against that
    # paper's Table I. It is a SIMULATION set - SNSPDs, as the paper models them - not a measured link,
    # and the page says so. Each value is quoted in tests/test_the_fibre_set_is_rusca_2018.py.
    #
    #                          here      Rusca 2018   was (Round 75)   GYS 2004 (Ma et al. 2005)
    #   detector efficiency    0.5       ~50%         0.25             0.045
    #   dark count per gate    1e-8      1e-8         1e-5             Y0 1.7e-6 (total)
    #   error floor            1%        pErr 1%      1.5%             3.3%
    #   dead time              100 ns    100 ns       none             -
    #   repetition rate        1 GHz     1 GHz        100 MHz          2 MHz
    #   fibre loss             0.2 dB/km 0.2 dB/km*   0.2              0.21
    #   error correction f_EC  1.16      -            1.16             1.22**
    #
    # * Table I's distances follow 0.2 dB/km. ** GYS states no f: 1.22 is Ma et al.'s own choice for
    # the GYS link ("the upper bound of f(e) in secure distance for this experiment"); 1.16 is
    # Cascade's efficiency at 1-5% error as tabulated by Lutkenhaus, PRA 61, 052304 (2000),
    # arXiv:quant-ph/9910093, and Rusca et al. state none.
    #
    # WHAT THE OLD SET WAS. "Modelling choices" matching no experiment: against the measured 50 km
    # link this file cites for its block size (Lucamarini et al., Opt. Express 21, 24550 (2013):
    # efficiency 20.5%, dark count 2.1e-5 per gate, QBER 4.26%, 1 GHz), the model detected 2.2x that
    # paper's counts and its rate was 3.1x the paper's BB84 rate. Lucamarini's set was not adopted:
    # it was measured with three intensities, and under this project's 1-decoy finite-key analysis,
    # with the error floor set to reproduce its 4.26% QBER, it needs 7e8 to 2e9 pulses for any key.
    fiber_loss_db_per_km: float = 0.2         # Rusca 2018's Table I distances; GYS 0.21
    detector_efficiency: float = 0.5          # Rusca 2018: "an efficiency (eta_det around 50%)"
    dark_count_prob: float = 1e-8             # Rusca 2018: "DCR of 10 Hz which correspond to pDC = 1e-8"
    misalignment: float = 0.01                # Rusca 2018: "We choose a typical value pErr of 1%"
    eve_intercept_fraction: float = 0.0       # demo control, not a device property
    error_correction_eff: float = 1.16        # Cascade at 1-5% error (Lutkenhaus 2000); Ma et al. use 1.22 for GYS
    # --- free-space (satellite downlink) channel ---
    #
    # These existed only as TypeScript literals in quantumSim.ts's FREE_SPACE, which every satellite
    # call site then spread while overriding just distance and Eve. So four of the six physical
    # constants behind the satellite page could not be changed without editing the source, and the
    # exact same shape had already produced a real bug on the fibre side: config and literal agreed
    # numerically, so editing config.yaml moved the Python model and left the browser on the constant
    # with nothing to indicate it. Fibre lives in config; free space now does too.
    #
    # MICIUS, SOURCED VALUE BY VALUE (Round 75). Until then the set was not "Micius's published
    # configuration" as the comment said: the aperture was another station's, the loss budget had no
    # receiver-optics term (7-8 dB too little loss at 800 km) and the noise was 17-35x Micius's, so the
    # page's ~5.5 kbit/s came from two errors offsetting. Every figure below is now read in Liao et al.,
    # Nature 549, 43 (2017), arXiv:1707.00542 - the Xinglong downlink - unless it says otherwise.
    free_space_distance_km: float = 800.0      # inside Liao's 645-1200 km pass
    free_space_detector_efficiency: float = 0.5           # "efficiency 50%"
    # "The overall optical efficiency including the receiving telescope and the fiber coupling on the
    # ground station is ~16%" - a separate budget line (about 8 dB) that the model did not have.
    free_space_receiver_optics_efficiency: float = 0.16
    # "dark counts <25 Hz", and photons are tagged "within a 2-ns time window": 25 x 2e-9 = 5e-8 per
    # detector per pulse. It was 1e-5, 200x Micius's.
    free_space_dark_count_prob: float = 5e-8
    # Liao's measured QBER "with an average of 1.1%, consistent with the expected error rate due to
    # background noise and polarization visibility". Used as the error floor because this model has no
    # separate background-light term; the polarisation contrast alone (280:1) would be 0.36%. It was
    # 0.025, justified by a 2 km horizontal ground link (arXiv:2607.24118), which is not this channel.
    free_space_misalignment: float = 0.011
    #
    # A satellite downlink is NOT attenuated per kilometre. It was modelled that way here -
    # `distance_km * 0.035`, a fibre law - which diverged from the physics past about 1000 km: 42 dB
    # at 1200 km, 70 at 2000, 175 at 5000. The Downlink control has no maximum.
    #
    # The dominant term is DIFFRACTION, which scales as 1/r^2: the transmitted beam spreads to
    # theta*r and the ground telescope captures (D_rx / (theta*r))^2 of it. Liao: a "near-diffraction-
    # limited far-field divergence of ~10 urad", "the beam diameter expands to about 10 m" at 1200 km,
    # and "the diffraction loss is estimated to be 22 dB at 1200 km" - this law gives 21.6 dB there.
    # Liao: "at 1200 km the loss due to atmospheric absorption and turbulence is in the range of 3 dB
    # to 8 dB, and the loss due to pointing error is less than 3 dB". (A comment here said Micius
    # "reports 27 dB" total; Liao states no total.)
    free_space_beam_divergence_urad: float = 10.0   # Liao: ~10 urad far field
    free_space_rx_aperture_m: float = 1.0           # Liao: Xinglong, "an aperture of 1 m" (1.2 m is Yin 2017's stations)
    # THE ATMOSPHERE SCALES WITH ELEVATION (Round 77, the user's choice). It was a fixed 5 dB at every
    # range, so the model's rate fell only by the inverse-square law along a pass. Liao's own pass did
    # not: "The sifted key rate decrease from ~12 kbit/s at 645 km to ~1 kbit/s at 1200 km, because of
    # the increase of both the physically separated distance and the effective thickness of the
    # atmosphere near the earth at smaller elevation angles" - 10.8 dB, where the fixed model fell 5.4.
    #
    # Now the loss is this value at the reference slant range below, scaled by the optical air mass
    # (Kasten & Young, Appl. Opt. 28, 4735 (1989)) at the elevation the slant range implies for the
    # orbit and station below. Anchoring at a range makes the station's air-pressure correction
    # cancel, since only the ratio of air masses enters.
    #
    # 5.5 dB IS THE MIDPOINT OF LIAO'S 3-8 dB, chosen by the user from a like-for-like comparison with
    # the 19 December 2016 pass. Liao's sifted key counts SIGNAL states only - 1,671,072 of 3,551,136
    # detections is 0.47057, and 0.5 x 0.4/0.425 for 50/25/25 states at 0.8/0.1/0 is 0.47059 - so the
    # check counts the same way (`tests/test_the_satellite_channel_is_sourced.py`):
    #
    #   anchor   detections over the pass    sifted at 645 km      at 1200 km          645 -> 1200 km
    #   5.5 dB   3,517,343  (-1.0%)          10,879 (-0.43 dB)     1,570 (+1.96 dB)    8.41 dB
    #   8.0 dB   2,556,307  (-28.0%)          8,390 (-1.55 dB)       883 (-0.54 dB)    9.78 dB
    #   Liao     3,551,136                   ~12 kbit/s             ~1 kbit/s          10.79 dB
    #
    # The 8 dB upper value ships as its own profile, `micius_2017_upper_atmosphere`. At the 800 km
    # default (35.8 degrees, air mass 1.70) the atmosphere is 3.2 dB.
    free_space_atmospheric_db: float = 5.5          # at the reference range; midpoint of Liao's 3-8 dB at 1200 km
    free_space_atmospheric_ref_km: float = 1200.0   # the slant range Liao's 3-8 dB is stated at
    free_space_orbit_altitude_km: float = 500.0     # Liao: Micius "orbiting at an altitude of ~500 km"
    free_space_station_altitude_km: float = 0.89    # Liao: Xinglong, "altitude 890m"
    # NO LINK BELOW THIS ELEVATION, and none beyond the horizon. Liao: "A single-orbit experiment ends
    # when the satellite reaches 10 deg elevation angle in the other end"; Sidhu et al. (arXiv:2602.11833,
    # 2026) impose the same 10 degrees. At 500 km it is a slant range of 1,692 km; the horizon is 2,571.
    free_space_min_elevation_deg: float = 10.0
    # MICIUS'S OWN SOURCE (Round 77, the user's choice), where the satellite ran the fibre page's
    # finite-key optimum. Liao: "mu_s=0.8, mu_1=0.1, mu_2=0", "sent with probabilities of 50%, 25%,
    # and 25%". Because the vacuum state is sent its yield is measured, so the bound is Ma et al.
    # 2005's Vacuum+Weak estimate (Eqs. 34, 37) rather than the one-decoy Eq. (41) the fibre page
    # runs; `micius_2017_one_decoy` is the same link without the vacuum. The probabilities set the
    # mix of clicks (for the dead time) and the like-for-like pass counts; the asymptotic rate, as on
    # the fibre page, is per pulse at the signal intensity.
    free_space_mu_signal: float = 0.8
    free_space_mu_decoy: float = 0.1
    free_space_p_signal: float = 0.5
    free_space_p_decoy: float = 0.25                # the vacuum takes the remaining 0.25
    free_space_vacuum_decoy: bool = True
    # The link these values describe, by profile id (src/qsh/config/profiles/downlink/), and the
    # inter-satellite terminal pair the Network tab's constellation starts from (profiles/isl/).
    # `tests/test_link_profiles_are_sourced.py` holds the default profile equal to the values above.
    free_space_profile: str = "micius_2017"
    isl_profile: str = "micius_type_2019"
    free_space_pointing_db: float = 3.0             # Liao's upper value, "less than 3 dB"; not scaled
    # --- decoy-state / efficient-BB84 parameters the AI optimizer (qkd/optimize.py) tunes ---
    # THE INTENSITIES ARE NOW THE FINITE-KEY OPTIMUM, NOT THE ASYMPTOTIC ONE, and the difference
    # is the point. mu ~ 0.5 is the textbook optimum for the ASYMPTOTIC decoy rate ON THE GYS LINK
    # (Ma et al.: 0.48 at f = 1.22), and that is what these were. This channel's own asymptotic
    # optimum is elsewhere again (row four below). But the quantity that decides whether this demo
    # distributes a key at all is the composable finite-key length, and its optimum is elsewhere: `finite_key.optimise_parameters`
    # searches exactly the variables Rusca et al. say they optimise - the intensities, their
    # probabilities, and the basis bias - by differential evolution, and lands here.
    #
    # Measured trade, at the shipped 50 km channel, ON THE RUSCA SET OF ROUND 76 with its dead time
    # (asymptotic: Ma et al.'s one-decoy Eq. (41) times the dead-time factor; thresholds:
    # `finite_key.py`, both at p_signal = p_z = 0.5, the shipped setting):
    #
    #     mu=0.5,    nu=0.1      asymptotic 0.003183 bit/pulse   positive-key threshold 1.04e7 pulses
    #     mu=0.342,  nu=0.142    asymptotic 0.002853 bit/pulse   positive-key threshold 5.88e6 pulses
    #     mu=0.319,  nu=0.129    asymptotic 0.002832 bit/pulse   positive-key threshold 5.85e6 pulses
    #     mu=0.51,   nu->0       asymptotic 0.003572 bit/pulse   no finite-key bound
    #
    # THESE RATES ARE PER SIGNAL PULSE, the convention until Round 78. The pages now count per pulse
    # SENT (the user's choice): only the p_signal share of the pulses carries key, so at the shipped
    # p_signal = 0.5 each reads half - the shipped pair's 0.002853 is 0.001427 bit/pulse on the page,
    # 28,534 secure bits in the 2e7-pulse block. The comparison between rows does not change.
    #
    # Row three is this channel's own finite-key optimum, and the shipped pair is within 0.6% of its
    # threshold, so the intensities did not move with the detector set. Row four is the asymptotic
    # optimum: a decoy that weak approximates a vacuum state and leaves the finite-key bound
    # undefined. Without the dead time the shipped pair's asymptotic rate would read 0.004565.
    #
    # ON THE CHANNEL OF ROUND 75 (efficiency 0.25, dark 1e-5, error floor 1.5%, no dead time) the same
    # rows read 0.002318 / 0.001952 bit/pulse with thresholds 3.84e7 / 1.41e7, and its asymptotic
    # optimum was mu = 0.665, nu = 0.055 at 0.002452. The finite-key optimum then cost 15.8% of the
    # mu=0.5 asymptotic rate and bought a 2.7x smaller block (38,410,230 -> 14,055,773); the Python
    # Monte Carlo allocates ~2.0 GB at 3.84e7 pulses against ~1.03 GB at the shipped 2e7.
    #
    # BOTH ROWS AT p_signal = p_z = 0.5. The optimiser's FULL solution also sets p_signal and p_z,
    # which are not config keys here (5.02e6 pulses on the Rusca set, at p_signal = 0.316); see the
    # note on `pulses` above.
    #
    # Chosen by an optimiser, so the value is reproducible rather than tuned by hand:
    #   python -c "from qsh.qkd.finite_key import optimise_parameters; print(optimise_parameters()[0])"
    mu_signal: float = 0.3416         # finite-key optimum (Round 47; within 0.6% on the Rusca set); GYS: ~0.5
    mu_decoy: float = 0.1417          # decoy mean photon number, optimised jointly with the signal
    basis_prob: float = 0.5           # P(Z basis) — biased-basis (efficient) BB84 raises the sift ratio
    #: QBER abort threshold, and it is a FIBRE-QKD post-processing figure - not a QSDC one.
    #:
    #: An earlier revision of this comment claimed 0.11 was correct for QSDC because the wiretap
    #: secrecy capacity Cs = 1 - 2h(e) has its root at 0.110028, identical to Shor-Preskill. That is
    #: true only for the LOSS-FREE case. The published QSDC criterion (Qi et al., Light Sci. Appl.
    #: 8, 22 (2019), step 2) is Cs > 0 with
    #:
    #:     Cs = Q_Bob[1 - h(e) - g h(e_x + e_z)],   g = Q_Eve/Q_Bob = 10^(loss_dB/10)  (Qi Eq. 8)
    #:
    #: so the threshold FALLS with channel loss: 0.0757 at 0 dB, 0.0286 at Qi et al.'s g = 2.57,
    #: and about 5.5e-6 on this project's 800 km downlink (37.0 dB with the detector). Until Round 75
    #: this read 0.11 / ~0.048 / ~0.0003 - the pooled h(e_chk) form, without the detector, on a
    #: downlink budget that had no receiver optics. Using 0.11 on a lossy link is wrong in the PERMISSIVE direction - it passes a
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
    # The CKKS ring degree, and the ONE place it is written. The browser (`web/src/lib/clientFHE.ts`),
    # the TenSEAL backend and the report's crypto table all read it through the config now, so
    # changing it is a line rather than a sweep. Note that `config.yaml` overlays this file, so the
    # value that RUNS is the one there - editing this default alone changes nothing, which is how
    # Round 38's first attempt at raising the ring produced no effect at all.
    #
    # RAISED TO 16384 IN ROUND 38, MEASURED, AND PUT BACK. The case for raising it was the medical
    # student's encrypted convolution: a degree-N ring carries N/2 slots, and one image needs 3136
    # of them (14x14 padded to 16x16, 3x3 kernel -> 196 patches at a 16-slot block), so 4096 slots
    # hold exactly one image and 8192 would hold two.
    #
    # Measured in-browser on the finance page, same machine, fresh tab each time:
    #
    #     keygen              254.4 ms   ->  491.4 ms    (x1.93, two runs each, back to back)
    #     ms per rotation       3.098    ->    6.725     (x2.17)
    #     linear layer         19.25 ms  ->   34.41 ms   (x1.79)
    #     batch, 400 rows      20.59 ms  ->   36.16 ms   (x1.76)
    #     batch precision      25 bits   ->   24 bits
    #
    # So every FHE figure this project publishes roughly doubles, and the keygen sits on the first
    # click, which `clientFHE` already records as feeling like a hang. Against that, the realised
    # benefit measured ZERO: the convolution still packed 196 patches - one image - because nothing
    # asks it for two, and the linear batch's 512 -> 1024 rows per ciphertext went unused because
    # finance offers 400 rows and genomics 30. Paying 2x for capacity no caller reaches is the wrong
    # trade, so the value went back and the wiring stayed.
    #
    # Raising it again is one line HERE and one in config.yaml, once batched convolution exists to
    # spend the slots on. Security is not the constraint either way: coeff_mod_bit_sizes sums to 200
    # bits, under both the 2018 HE standard's 128-bit caps that SEAL's tc128 table encodes (218 at
    # N=8192, 438 at N=16384) and the LOWER caps of the 2024 guidelines, Bossuat et al., IACR CiC
    # 1(4), Table 5.2 (214 and 430 for ternary secrets; read Round 75). The library guard is the 2018
    # table, so a chain between the two would pass it while missing the current guidance.
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
    # THE MPyC TIER'S PARTIES, since Round 75 separate processes over loopback TCP (qsh/mpc/parties.py).
    # It ran as ONE in-process party with MPyC's defaults (m = 1, t = 0, k = 30), so its Shamir
    # shares were the values themselves, nothing crossed a wire, and the communication it reported
    # was 256 B per comparison multiplied out. Measured at m = 3, t = 1, k = 40: 15,120 B, 59x that.
    mpc_parties: int = 3                    # Shamir parties; honest majority needs t < m/2
    mpc_threshold: int = 1                  # corrupted parties tolerated
    mpc_sec_param: int = 40                 # statistical security, as the browser's Polytune tier uses
    # The masks of the hybrid's FHE <-> MPC conversions (qsh/mpc/secret_inference.py), uniform on
    # [-2^b, 2^b). A value of size B behind one is hidden to statistical distance B / 2^b, which each
    # run reports. 30 is the most 64-bit fixed point holds (integer part 2^31) and costs no precision
    # measured through a CKKS round trip: 5e-7 at 2^30 against 8e-6 at 2^36 and 2e-4 at 2^40.
    mpc_mask_bits: int = 30
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
    # The two MPC figures are the MPyC tier's own, measured (Round 75, qsh/mpc/parties.py) because
    # the row they fill is labelled "ckks + mpyc-shamir": 15,120 B per secure comparison across three
    # parties at t = 1, k = 40, the same on every run, and 2.2 ms of protocol time on loopback without
    # gmpy2. They were 256 B and 1.6 ms, from no stated source; the bytes were 59x low.
    # tests/test_the_cost_model_is_the_mpc_tier_measured.py re-measures the bytes.
    cost_per_op: dict = field(default_factory=lambda: {
        "fhe_ms_per_dot": 12.0, "mpc_ms_per_cmp": 2.2, "mpc_comm_bytes_per_cmp": 15120,
        "demo_ct_pt_ratio": 1483.0, "demo_max_abs_error": 4.6e-7,
        # `qsmpc_ct_pt_ratio` (1.32) was REMOVED on 2026-10-23. Its only reader was a static Overview
        # card comparing it with `demo_ct_pt_ratio`, and it was described as the Rate-1 QFHE
        # "literature value" - the paper (Chardouvelis, Doettling, Malavolta, TCC 2021) reports no
        # concrete figure. `demo_ct_pt_ratio` stays: it is the MPC-FHE row's representative ct/pt when
        # no measured ciphertext is available, and it measures slot utilisation, not CKKS itself.
        "qsmpc_t_gates": 56, "qsmpc_qubits": 8})
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
    # Transformers.js models, each PINNED to a commit, with the bytes of the weights file each backend
    # fetches taken from the Hub's file listing at that commit (2026-10-24). The model picker computes
    # the size it shows from `bytes`: the label used to read "tiny, ~80 MB", typed by hand, and no file
    # the library loads is that size - WASM fetches `model_quantized.onnx` (236.7 MB) and WebGPU
    # `model.onnx` (327.8 MB). The 80 MB figure was the old `decoder_model_merged_quantized.onnx`.
    # `dtype` is passed to the pipeline explicitly, so the file `bytes` describes is the file that is
    # fetched whatever the library's per-device default becomes. A label must not carry a size.
    tfjs_models: list = field(default_factory=lambda: [
        {"id": "Xenova/distilgpt2", "label": "distilGPT-2",
         "revision": "a41c10485c18a64b6606729b6a082330cbd8f49e",
         "dtype": {"wasm": "q8", "webgpu": "fp32"},
         "bytes": {"wasm": 236_698_606, "webgpu": 327_825_716}},
        {"id": "onnx-community/Qwen2.5-0.5B-Instruct", "label": "Qwen2.5-0.5B-Instruct",
         "revision": "22942cb7d7ba4cc81bb4673549ca4d4614469b5e",
         "dtype": {"wasm": "q8", "webgpu": "fp32"},
         "bytes": {"wasm": 512_096_557, "webgpu": 1_993_796_793}}])
    # The backend the small models run on unless the visitor picks WebGPU: the Transformers.js agent
    # and the published ONNX models of the inference stage. WASM, because on the machine measured (an
    # NVIDIA RTX PRO 6000 adapter, Chrome 152 on Linux) WebGPU ran distilGPT-2 at 750 to 1140 ms per
    # token against about 9 ms on WASM, and one ResNet-18 session.run took 18216 ms on onnx-webgpu
    # (register, Round 59). Other GPUs and browsers were not measured, which is why WebGPU stays one
    # selection away rather than being removed.
    tfjs_default_device: str = "wasm"
    onnx_default_backend: str = "wasm"
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
    # The agent is abandoned after this long WITHOUT PROGRESS - no download event and no answer - not
    # after this long in total. A first run downloads its weights (236.7 MB for distilGPT-2 on WASM),
    # and until Round 60 a total limit cut off downloads that were still arriving.
    agent_timeout_ms: int = 120_000
    tfjs_max_new_tokens: int = 96
    webllm_max_tokens: int = 160
    # Raised with `QKDConfig.pulses` for the same reason - see the note there. At 40_000 the decoy
    # bound gives **79** secure bits, below the 256 a session key needs, so `/validations` reported a
    # positive bit count beside prose reading "no secure key bits". (94 was the count at Round 46's
    # mu = 0.5 / nu = 0.1; the finite-key optimum this round shipped makes it 79.)
    # Kept EQUAL to `QKDConfig.pulses` and for the same reason: this is the block the two-party
    # panel reports on, so if it sits below the composable threshold the panel publishes an
    # asymptotic rate for a block that carries no key. Round 47 also removed the two Python call
    # sites that hardcoded 40_000 while this key said 400_000 - the fourth time in four rounds that
    # a fix changed one language, and the reason `run_bb84` call sites may no longer spell a
    # literal block size at all.
    demo_alice_bob_pulses: int = 20_000_000
    # 4_000_000 gives **222** secure bits over the 800 km free-space downlink under the decoy bound -
    # below the 256 threshold `SatelliteQKDStage` needs to derive a key, so the satellite use case
    # stopped folding QKD into its session entirely. 10_000_000 gives **555**. (223 and 559 were the
    # counts before this round moved the intensities to the finite-key optimum; both are deterministic
    # across seeds, since the decoy bound is analytic.)
    #
    # AND THE HONEST RESIDUAL, recorded rather than engineered away, and MUCH LARGER THAN THIS
    # COMMENT USED TO SAY. RE-MEASURED at the shipped intensities (mu = 0.3416, nu = 0.1417): at 10e6
    # the downlink yields **~2,440 sifted bits** - a 5-seed mean of 2,442, spread 2,397 to 2,470. The
    # ~3,660 this line carried is the pre-Round-47 figure at mu = 0.5 / nu = 0.1, where the same
    # 5 seeds give a mean of 3,582; the subsection was rewritten in the round that moved the
    # intensities and this line was carried across without re-measurement.
    #
    # This line previously measured that against the "~5e3-1.1e4 finite-key positive-key band" and
    # concluded the gap could be closed with ~30e6 pulses and ~1.5 GB of Python array. Both figures
    # came from a band that Round 47 refuted - see the note on `QKDConfig.pulses` above and
    # `src/qsh/qkd/finite_key.py`. The corrected positive-key threshold is ~1e5 sifted bits and about
    # 4e7 pulses on the 50 km FIBRE channel, so 2,440 sifted bits is short by well over an order of
    # magnitude rather than by the small factor implied here, and the 30e6-pulse fix was never a fix.
    #
    # The free-space threshold is not quoted, because it has not been evaluated: `finite_key.py`
    # takes a fibre `ChannelParams`, and the 800 km downlink's 24.48 dB budget is built elsewhere.
    # Only the DIRECTION is safe to state, and it is the unhelpful one - a lossier link has a lower
    # gain per pulse and a higher QBER at equal statistics, so it needs MORE than the fibre link, not
    # less. The reason is physical and not a setting. The free-space rate is therefore asymptotic-only
    # in a stronger sense than the fibre one, and the register says so rather than the number
    # pretending otherwise.
    #
    # ROUND 75 RAISED IT TO 20e6. The sourced Micius budget (receiver optics, 1 m aperture, Micius's
    # detectors) and the one-decoy bound take the downlink to 1.723e-5 bit/pulse: 172 secure bits at
    # 10e6, under the 256 a key needs, so both the browser and the Python stage would have stopped
    # mixing QKD into the satellite session. 20e6 gives 344, deterministically. The browser's bound is
    # analytic and costs nothing per pulse; the Python Monte Carlo now runs at the fibre default's 2e7.
    #
    # ROUND 77 left it at 20e6. The air-mass atmosphere (3.2 dB at 800 km, where 5 dB was fixed) and
    # Micius's own source with its vacuum state gave 3.356e-5 bit/pulse per SIGNAL pulse. Counted per
    # pulse SENT since Round 78 - Micius sends signals on half its pulses - that is 1.678e-5: 335
    # secure bits at 20e6, and 167 at 10e6, which would not clear the 256, so 20e6 stays.
    demo_sat_pulses: int = 20_000_000   # free-space BB84 channel uses in the demo satellite run


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
