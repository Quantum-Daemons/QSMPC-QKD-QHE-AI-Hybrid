"""The control plane's rate fields are the ones ETSI GS QKD 015 V2.1.1 defines (Round 84).

WHY THIS EXISTS. The SDN panel subtracted the MCP agent's draw from `eskr`. The spec's `eskr` is the
"Effective secret key rate (in bits per second) generation of the key association link available
after internal consumption", and an application's demand is `expected_consumption`, the "Sum of all
the application's bandwidth (in bits per second) on this particular key association link" (GS QKD
015 V2.1.1, qkdl_performance; the YANG of revision 2022-01-30 says the same). Clause 4.6 names the
internal applications: "keys used for authentication, to create a virtual link". This model spends
no key on either, so `eskr` is `skr`, and the agent's draw is reported where the spec puts it.

On the satellite the panel printed `skr` - the asymptotic rate at the run's range, the top of the
pass - beside the key block's 408.5 s without saying which: over the block the finite key is 139,251
bits, 341 bit/s. And `phys_channel_att`, "Expected attenuation on the quantum channel (in dB)", was
the fibre's 0.2 dB/km on the downlink too: 160 dB at 800 km, where the run computes 32.2.

WHAT IS ASSERTED, on the TypeScript module and the component rendered by React's server renderer:
  - `eskr` equals `skr` on every link, before and after the allocation;
  - `expected_consumption` is the granted bits over the key block's channel time on the link the
    application draws from, 0 on a link no application uses, and absent without a clock;
  - `phys_channel_att` is the run's channel loss, halved per hop on a relay, the fibre law only
    when the run gives none; the page passes the run's `channel_loss_db`;
  - the satellite panel names `skr` the rate at closest approach and prints the block's average.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"

#: The default satellite run, as both languages report it (tests/test_the_satellite_block_is_a_pass.py
#: holds the two to one key): a pass peaking at 800 km, its finite key over its link time.
SAT = dict(usecase="satellite_comms", distanceKm=800, maxSpanKm="Infinity", keyBudgetBits=139_251,
           secretKeyRatePerPulse=1.678e-05, sourceRateHz=1e8, pulses=40_846_100_000, channelSeconds=408.5,
           channelLossDb=32.2, decoyStates="vacuum + one decoy", qkdEnabled=True, blockKind="pass")
#: A fibre run past one span, so the controller relays it.
RELAY = dict(usecase="telecom_ids", distanceKm=120, keyBudgetBits=500_000, secretKeyRatePerPulse=1e-4,
             sourceRateHz=1e9, pulses=5e9, channelSeconds=5.3, channelLossDb=24.0, decoyStates="one decoy",
             qkdEnabled=True)


@pytest.fixture(scope="module")
def node() -> dict:
    if not ESBUILD.exists():
        pytest.skip("web/node_modules/.bin/esbuild is absent; run `npm ci` in web/")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        bundle, runner = d / "bundle.mjs", d / "run.mjs"
        entry = (f"export {{ default as Panel }} from {json.dumps(str(WEB / 'src/components/SdnControlPlanePanel'))};\n"
                 f"export * as sdn from {json.dumps(str(WEB / 'src/lib/clientSdnController'))};\n"
                 'export { createElement } from "react";\n'
                 'export { renderToStaticMarkup } from "react-dom/server.browser";\n')
        subprocess.run([str(ESBUILD), "--bundle", "--loader=tsx", "--format=esm", "--platform=node", "--jsx=automatic",
                        '--define:import.meta.env={"BASE_URL":"/"}', '--define:process.env.NODE_ENV="production"',
                        f"--outfile={bundle}", "--log-level=error"], cwd=WEB, check=True, capture_output=True,
                       input=entry.encode())
        runner.write_text(f"import * as m from {json.dumps(bundle.as_uri())};\n" + f"""
const fix = (o) => ({{ ...o, maxSpanKm: o.maxSpanKm === "Infinity" ? Infinity : o.maxSpanKm }});
const topo = (o, draw) => {{
  const t = m.sdn.buildTopology(fix(o));
  const before = t.links.map((l) => ({{ id: l.qkdl_id, skr: l.skr, eskr: m.sdn.linkEskr(l, t) }}));
  if (draw) m.sdn.allocate(t, t.apps[0].app_id, draw);
  return {{ before, links: t.links.map((l) => ({{ id: l.qkdl_id, type: l.qkdl_type, skr: l.skr, att: l.phys_channel_att,
    eskr: m.sdn.linkEskr(l, t), consumption: m.sdn.linkExpectedConsumption(l, t), consumed: l.consumed_bits,
    apps: l.qkdl_applications }})), secs: t.channel_seconds }};
}};
const text = (props) => m.renderToStaticMarkup(m.createElement(m.Panel, fix(props)))
  .replace(/<[^>]+>/g, " ").replace(/&#x27;|&apos;/g, "'").replace(/&quot;/g, '"').replace(/\\s+/g, " ");
const sat = {json.dumps(SAT)}, relay = {json.dumps(RELAY)};
console.log(JSON.stringify({{
  sat: topo(sat, 34_813), relay: topo(relay, 125_000),
  satNoLoss: topo({{ ...sat, channelLossDb: null }}, 0), noClock: topo({{ ...sat, sourceRateHz: null }}, 34_813),
  satPanel: text(sat), relayPanel: text(relay), heldPanel: text({{ ...sat, blockKind: "seconds" }}),
}}));
""")
        return json.loads(subprocess.run(["node", str(runner)], check=True, capture_output=True, text=True).stdout)


def test_eskr_is_skr_before_and_after_the_draw(node) -> None:
    for case in ("sat", "relay"):
        for when in ("before", "links"):
            for link in node[case][when]:
                assert link["skr"] is not None and link["eskr"] == link["skr"], (case, when, link)
    for link in node["noClock"]["links"]:
        assert link["skr"] is None and link["eskr"] is None


def test_the_agents_draw_is_expected_consumption(node) -> None:
    sat = node["sat"]["links"][0]
    assert sat["consumed"] == 34_813 and sat["consumption"] == pytest.approx(34_813 / 408.5, rel=1e-12)
    relay = {link["id"]: link for link in node["relay"]["links"]}
    assert relay["qkdl-a-b"]["type"] == "VIRT" and relay["qkdl-a-b"]["apps"]
    assert relay["qkdl-a-b"]["consumption"] == pytest.approx(125_000 / 5.3, rel=1e-12)
    # No application draws from the hops; the virtual link's transport over them is internal, not modelled.
    assert relay["qkdl-a-r"]["consumption"] == 0 and relay["qkdl-r-b"]["consumption"] == 0
    # No clock, no rate: absent rather than 0 bit/s.
    assert node["noClock"]["links"][0]["consumption"] is None


def test_the_attenuation_is_the_runs_channel_loss(node) -> None:
    assert node["sat"]["links"][0]["att"] == pytest.approx(32.2)
    relay = {link["id"]: link for link in node["relay"]["links"]}
    assert relay["qkdl-a-r"]["att"] == relay["qkdl-r-b"]["att"] == pytest.approx(12.0)
    assert relay["qkdl-a-b"]["att"] is None                  # VIRT: no quantum channel of its own
    # Only when the run gives no loss does the fibre law stand in.
    assert node["satNoLoss"]["links"][0]["att"] == pytest.approx(800 * 0.2)
    page = (WEB / "src" / "pages" / "UseCasePage.tsx").read_text()
    assert re.search(r"channelLossDb=\{metrics\.channel_loss_db != null \? Number\(metrics\.channel_loss_db\) : null\}", page)


def test_the_satellite_panel_says_which_rate_skr_is(node) -> None:
    t = node["satPanel"]
    assert "skr (bit/s, assumed clock, at closest approach)" in t
    assert "the key block's average rate 341 bit/s" in t
    assert "which on a pass block is the pass's closest approach, where the rate peaks" in t
    assert "expected_consumption (bit/s)" in t and "85.2 bit/s over the key block's channel time" in t
    assert "it equals skr" in t
    row = re.search(r"qkdl-a-b PHYS qkdn-a to qkdn-b ([\d,.]+) ([\d,.]+) ([\d,.]+) ", t)
    assert row and row.group(1) == row.group(2) == "1,678" and row.group(3) == "85.2", row and row.groups()
    # The fibre panel names no closest approach, and says its rate is held for the block; nor does a
    # satellite block held at one range.
    for held in (node["relayPanel"], node["heldPanel"]):
        assert "at closest approach" not in held and "held for the whole block" in held
    page = (WEB / "src" / "pages" / "UseCasePage.tsx").read_text()
    assert "blockKind={metrics.qkd_block as string | undefined}" in page
