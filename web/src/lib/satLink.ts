// The satellite downlink a run selected, built in one place: the profile it names, then the
// Parameters-panel controls on top. Moved out of clientCompute.ts in Round 77 so the Network tab and
// the cross-language test (tests/test_link_profiles_are_sourced.py) can use it without bundling the
// whole pipeline. Twin of `qsh.usecases.satellite_comms` `_sat_profile` / `_sat_channel` /
// `_sat_source` / `_out_of_view`.
import { profileValues, qkdCfg } from "./config";
import { downlinkFromProfile, FREE_SPACE, inView, linkElevationDeg, sourceRateHz,
  type ChannelParams } from "./quantumSim";

/**
 * The free-space channel this run selected, in one place.
 *
 * `satellite_qkd` and `qsdc_telemetry` each spread `FREE_SPACE` and overrode a subset of its fields
 * inline - four sites, three different subsets. `channelLossDb` was called WITHOUT `sat_eve` (right,
 * loss does not depend on Eve) and `intrinsicQber` WITH it (also right), which is exactly the kind
 * of distinction that survives only while one person holds all four sites in their head. Adding the
 * two link-budget keys would have made it eight overrides across four sites.
 *
 * `eveFraction` stays an explicit argument because the two callers genuinely differ: the counterfactual
 * that decides `link_budget_limited` re-evaluates the SAME link with Eve removed.
 */
export function satChannel(config: Record<string, any>, eveFraction?: number): ChannelParams {
  // THE SELECTED PROFILE IS THE BASE (Round 77): `sat_profile` names a JSON file under
  // src/qsh/config/profiles/downlink/, and every control below overrides it. The default profile is
  // FREE_SPACE itself, built from config, so a config.yaml override still reaches it. Twin of
  // `satellite_comms._sat_profile` / `_sat_channel` / `_sat_source`.
  const id = String(config?.sat_profile ?? qkdCfg().free_space_profile);
  const values = id === qkdCfg().free_space_profile ? null : profileValues("downlink", id);
  const base = values ? downlinkFromProfile(values) : FREE_SPACE;
  const fs = base.freeSpace!;
  // Switching the vacuum off hands its share to the decoy, as `micius_2017_one_decoy` does.
  const vacuum = config?.sat_vacuum_decoy == null ? Boolean(base.vacuumDecoy) : Boolean(config.sat_vacuum_decoy);
  const pSignal = Number(base.pSignal ?? 0.5);
  return {
    ...base,
    distanceKm: Number(config.sat_distance_km ?? base.distanceKm),
    eveFraction: eveFraction ?? Number(config.sat_eve ?? base.eveFraction),
    freeSpace: {
      ...fs,
      atmosphericDb: Number(config.sat_atmospheric_db ?? fs.atmosphericDb),
      pointingDb: Number(config.sat_pointing_db ?? fs.pointingDb),
      minElevationDeg: Number(config.sat_min_elevation_deg ?? fs.minElevationDeg),
      fieldStopUrad: Number(config.sat_field_stop_urad ?? fs.fieldStopUrad ?? 0),
    },
    sourceRateHz: sourceRateHz(config?.sat_source_rate_mhz, Number(base.sourceRateHz)),
    vacuumDecoy: vacuum,
    pDecoy: vacuum ? Number(base.pDecoy) : 1 - pSignal,
  };
}

/** Why a downlink is not operated, or null when it is. Twin of `satellite_comms._out_of_view`. */
export function outOfView(ch: ChannelParams): string | null {
  if (inView(ch)) return null;
  const e = linkElevationDeg(ch);
  const fs = ch.freeSpace!;
  const where = e === null ? "beyond the horizon"
    : `at ${e.toFixed(1)} deg elevation, below the ${fs.minElevationDeg} deg minimum`;
  return `no link at ${Math.round(ch.distanceKm)} km: ${where} for a ${fs.orbitAltitudeKm} km orbit. `
    + "A station this far from the satellite needs a relay - the page's Network tab";
}
