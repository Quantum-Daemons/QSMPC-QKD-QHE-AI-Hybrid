// One line at the foot of every page: where the source and the models are, the release, who publishes it, what is
// simulated, and when the pages were last reviewed (the owner's draft of 2026-11-13, added on the owner's choice of
// 2026-11-22). No price, roadmap or contact line, and no commit links: the version is the release's, read from
// web/package.json, which tests/test_r86_the_version_has_one_source.py holds equal to qsh.__version__.
import { version } from "../../package.json";

/** When the pages were last read against the code and the records. Change it when they are next reviewed. */
export const PAGES_LAST_REVIEWED = "22 November 2026";

const ext = { target: "_blank", rel: "noopener noreferrer" } as const;

export default function SiteFooter() {
  return (
    <footer className="site-footer">
      Source: <a href="https://github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid" {...ext}>github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid</a>
      {" "}(Apache-2.0) · Models: <a href="https://huggingface.co/Daemons-Q" {...ext}>huggingface.co/Daemons-Q</a>
      {" "}· v{version} · By Quantum Daemons: <a href="https://quantum.daemons.jp/" {...ext}>quantum.daemons.jp</a>
      {" "}· QKD and QSDC are simulated · Page last reviewed: {PAGES_LAST_REVIEWED}
    </footer>
  );
}
