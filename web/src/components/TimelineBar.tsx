// Replay of a run (Round 86, the owner's choice of 2026-11-14): back and forward one stage boundary at a time, or scrub
// to any event of the recorded log. The canvas, the log, the metrics table and "Who holds what" follow the position; the
// cards further down keep the finished run, since a run's artifacts arrive whole at its end. Nothing runs again.
import type { PipelineEvent } from "../lib/api";
import { nextMark, prevMark, stageAt } from "../lib/runTimeline";

export default function TimelineBar({ events, cursor, setCursor, marks, inFlight }: {
  events: PipelineEvent[];
  cursor: number | null;
  setCursor: (c: number | null) => void;
  marks: number[];
  inFlight: boolean;
}) {
  const n = events.length;
  if (n === 0) return null;
  const at = cursor == null ? n : Math.min(cursor, n);
  const following = cursor == null || cursor >= n;
  const go = (k: number) => setCursor(k >= n ? null : Math.max(0, k));
  const last = at > 0 ? events[at - 1] : null;
  const stage = stageAt(events, at);
  const where = `event ${at} of ${n}, ${last ? `${last.t.toFixed(3)} s` : "before the first event"}${stage ? `, ${stage}` : ""}`;
  return (
    <div className="timeline-bar" role="group" aria-label="Replay the run">
      <button className="btn ghost" onClick={() => go(0)} disabled={at === 0} aria-label="To the start of the run">|&lt;</button>
      <button className="btn ghost" onClick={() => go(prevMark(marks, at))} disabled={at === 0}
        aria-label="Back to the previous stage boundary">&lt; Back</button>
      <input type="range" min={0} max={n} step={1} value={at} list="timeline-marks"
        onChange={(e) => go(Number(e.target.value))} aria-label="Position in the run's log" aria-valuetext={where} />
      <datalist id="timeline-marks">{marks.map((m) => <option key={m} value={m} />)}</datalist>
      <button className="btn ghost" onClick={() => go(nextMark(marks, at))} disabled={following}
        aria-label="Forward to the next stage boundary">Forward &gt;</button>
      <button className="btn ghost" onClick={() => setCursor(null)} disabled={following}
        aria-label={inFlight ? "Follow the run again" : "To the finished run"}>{inFlight ? "Live" : "End"} &gt;|</button>
      <span className="muted small timeline-where" aria-live="polite">
        {where}: {following ? (inFlight ? "following the run" : "the finished run") : "replaying the log"}
      </span>
    </div>
  );
}
