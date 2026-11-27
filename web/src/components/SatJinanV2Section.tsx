// The Jinan-1 pass, second analysis (Round 86, batch 8): references/satellite-jinan1-pass-v2-preregistration.md, run by
// scripts/measure_jinan1_pass_v2.py, whose web/public/data/satellite/jinan1_pass_v2.json this reads. Its rule 3: the card
// states these figures beside the first analysis's, whatever they show. Nothing renders until the file exists.
import { useEffect, useState } from "react";

type Ratio = { model_packets: number; packets_ratio: number; model_key_bits: number; key_ratio: number };
interface V2 {
  preregistration: string; record: string;
  q1_atmosphere: { fitted_efficiency: number; efficiency_se: number; fitted_zenith_loss_db: number; zenith_loss_se_db: number;
                   correlation: number; profile_efficiency: number; profile_zenith_loss_db: number; sd_change_db: number;
                   residual_sd_db: number };
  q2_sifted_bits: { measured_sifted_bits: number; profile: { model_sifted_bits: number; model_over_measured: number;
                                                             lag_s: number | null; rms_after_lag_s: number | null } };
  q3_keys_by_block: { profile: { blocks: { block: number; measured_over_model: number | null }[]; mean_measured_over_model: number | null } };
  q4_other_passes: { passes: ({ date: string; max_elevation_deg: number; published_packets: number; profile: Ratio })[] };
  rules: { "1_profile_takes_the_fitted_pair": boolean; "2_key_model_gains_the_factors": boolean };
}

const sign = (v: number, d = 2) => `${v >= 0 ? "+" : ""}${v.toFixed(d)}`;

export default function SatJinanV2Section() {
  const [d, setD] = useState<V2 | null>(null);
  useEffect(() => {
    let on = true;
    fetch(`${import.meta.env.BASE_URL}data/satellite/jinan1_pass_v2.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then((j: V2 | null) => { if (on && j?.q1_atmosphere) setD(j); })
      .catch(() => undefined);
    return () => { on = false; };
  }, []);
  if (!d) return null;
  const a = d.q1_atmosphere, s = d.q2_sifted_bits.profile, blocks = d.q3_keys_by_block.profile;
  const ratios = blocks.blocks.map((b) => b.measured_over_model).filter((v): v is number => v != null);
  return (
    <div style={{ marginTop: 10 }}>
      <h4 style={{ margin: "6px 0 2px" }}>The second analysis, pre-registered before it ran</h4>
      <ul className="small" style={{ margin: "4px 0 0", paddingLeft: 18 }}>
        <li>The atmosphere: fitting the station efficiency and the zenith atmospheric loss together gives{" "}
          {a.fitted_efficiency.toFixed(4)} (standard error {a.efficiency_se.toFixed(4)}) and {a.fitted_zenith_loss_db.toFixed(3)} dB
          (standard error {a.zenith_loss_se_db.toFixed(3)}), against the profile&apos;s {a.profile_efficiency} and{" "}
          {a.profile_zenith_loss_db.toFixed(3)} dB, the two correlated at {a.correlation.toFixed(2)}. The residual deviation
          moves {sign(a.sd_change_db, 3)} dB from the one-constant fit&apos;s, to {a.residual_sd_db.toFixed(3)} dB;{" "}
          {d.rules["1_profile_takes_the_fitted_pair"] ? "the profile takes the fitted pair."
            : "the rule asked for a fall of 0.05 dB, so the profile keeps its values."}</li>
        <li>The sifted bits, with the paper&apos;s 80 percent gate, the 90 percent of detections not sampled for the QBER and the
          basis match: the model&apos;s {(s.model_sifted_bits / 1e6).toFixed(2)} million against the measured{" "}
          {(d.q2_sifted_bits.measured_sifted_bits / 1e6).toFixed(0)} million, {s.model_over_measured.toFixed(2)}{" "}
          times{s.lag_s != null ? <>; the measured packets close about {Math.abs(s.lag_s).toFixed(0)} s {s.lag_s >= 0 ? "later" : "earlier"} than
            the model&apos;s{s.rms_after_lag_s != null ? <>, {s.rms_after_lag_s.toFixed(0)} s apart on average after that shift</> : null}</> : null}.</li>
        {ratios.length ? <li>The keys by block: each 500-kbit block&apos;s measured key is {Math.min(...ratios).toFixed(2)} to{" "}
          {Math.max(...ratios).toFixed(2)} times the model&apos;s finite bound over the block&apos;s seconds, with the same
          factors{blocks.mean_measured_over_model != null ? <> (mean {blocks.mean_measured_over_model.toFixed(2)})</> : null}.</li> : null}
        <li>The preprint&apos;s three other passes, at their maximum elevations:{" "}
          {d.q4_other_passes.passes.map((p) => `${p.date} (${p.max_elevation_deg} degrees), the model's packets ${p.profile.packets_ratio.toFixed(2)} and its finite key ${p.profile.key_ratio.toFixed(2)} times the published`).join("; ")}.{" "}
          {d.rules["2_key_model_gains_the_factors"] ? "The page's key model gains the gate and sampling factors."
            : "The page's key model keeps its form: the rule asked for every pass within a factor of 1.5."}</li>
      </ul>
      <p className="small muted" style={{ marginTop: 4 }}>
        Pre-registration: <code>{d.preregistration}</code>; record: <code>{d.record}</code>.
      </p>
    </div>
  );
}
