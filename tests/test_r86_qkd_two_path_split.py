"""A relayed QKD key can be split over two node-disjoint relay paths, and both controllers count it the same (Round 86).

The control-plane panel (web/src/lib/clientSdnController.ts) built one trusted relay whenever a span was too long, and
said that relay sees the key. Since the owner's choice of 2026-11-13 it can split the key instead: two relays on
node-disjoint paths, two VIRT associations, and an application whose key is their XOR (`qsh_key_combination`, not a GS
QKD 015 field). Held here:

1. the split topology has two relays, four PHYS hops and two VIRT links whose intermediate nodes do not overlap, and
   an application backed by both; one path, or a span short enough to need none, is the topology it was;
2. an XORed key's budget is the SMALLER path's, in `allocate` and in the GS QKD 014 key count, where a sum would grant
   twice what the two paths can make together; each path is charged what is granted;
3. the Python controller allocates by the same rule, grant for grant, and its `xor_split_key` combines path keys so
   that one path's key alone is independent of the result (a chi-square on every byte value);
4. the trust note says the relay sees its own path's key and not the end-to-end key.
"""

from __future__ import annotations

import collections
import json
import os
import secrets
import shutil
import subprocess
from pathlib import Path

import pytest

from qsh.qkd.sdn_controller import QkdApplication, QkdInterface, SdnController, SdQkdNode, xor_split_key

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"

RUNNER = r"""
const m = await import(%(bundle)s);
const base = { usecase: "telecom_ids", keyBudgetBits: 10000, qkdEnabled: true, secretKeyRatePerPulse: 1e-4,
               sourceRateHz: 1e8, pulses: 1e8 };
const out = {};
for (const [name, extra] of Object.entries({ split: { distanceKm: 120, relayPaths: 2 }, one: { distanceKm: 120 },
                                              short: { distanceKm: 50, relayPaths: 2 } })) {
  const t = m.buildTopology({ ...base, ...extra });
  out[name] = { nodes: t.nodes.map((n) => n.qkdn_id), links: t.links.map((l) => [l.qkdl_id, l.qkdl_type, l.virt_next_hop]),
                app: t.apps[0], notes: t.notifications, trust: t.links.map((l) => m.linkTrustNote(l, t)) };
}
// Unequal budgets: path 2 has less left, so the grant is bounded by it.
const t = m.buildTopology({ ...base, distanceKm: 120, relayPaths: 2 });
t.links.find((l) => l.qkdl_id === "qkdl-a-b-2").key_budget_bits = 3000;
out.granted = [m.allocate(t, t.apps[0].app_id, 2500), m.allocate(t, t.apps[0].app_id, 2500)];
out.after = t.links.filter((l) => l.qkdl_type === "VIRT").map((l) => [l.qkdl_id, l.key_budget_bits, l.consumed_bits]);
out.stored = m.kmeStatus(t).stored_key_count;
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def ts(tmp_path_factory) -> dict:
    missing = [n for n, ok in (("web/node_modules/.bin/esbuild", ESBUILD.exists()), ("node", shutil.which("node")))
               if not ok]
    if missing:
        msg = f"{' and '.join(missing)} absent, so the browser module cannot run. This is a SKIP, not a pass."
        if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
            pytest.fail(msg)
        pytest.skip(msg)
    d = tmp_path_factory.mktemp("split")
    entry, bundle, runner = d / "entry.ts", d / "bundle.mjs", d / "run.mjs"
    entry.write_text(f"export * from {json.dumps(str(WEB / 'src/lib/clientSdnController'))};\n")
    subprocess.run([str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node", f"--outfile={bundle}",
                    "--log-level=error"], cwd=WEB, check=True, capture_output=True)
    runner.write_text(RUNNER % {"bundle": json.dumps(bundle.as_uri())})
    proc = subprocess.run(["node", str(runner)], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_the_split_topology_has_two_disjoint_relay_paths(ts) -> None:
    s = ts["split"]
    assert {"qkdn-relay-1", "qkdn-relay-2"} <= set(s["nodes"]) and "qkdn-relay" not in s["nodes"]
    phys = [lid for lid, kind, _ in s["links"] if kind == "PHYS"]
    virt = [(lid, hops) for lid, kind, hops in s["links"] if kind == "VIRT"]
    assert len(phys) == 4 and len(virt) == 2
    assert not set(virt[0][1]) & set(virt[1][1]), "the two paths share a relay"
    assert s["app"]["qsh_key_combination"] == "xor" and sorted(s["app"]["backing_qkdl_id"]) == ["qkdl-a-b-1", "qkdl-a-b-2"]
    # One path is the topology it was; a short span needs no relay whatever the control says.
    assert "qkdn-relay" in ts["one"]["nodes"] and "qsh_key_combination" not in ts["one"]["app"]
    assert [k for _, k, _ in ts["short"]["links"]] == ["PHYS"] and "qsh_key_combination" not in ts["short"]["app"]


def test_an_xored_key_is_budgeted_by_the_smaller_path(ts) -> None:
    assert ts["granted"] == [2500, 500], ts["granted"]   # 3,000 left on path 2: 2,500, then the last 500
    assert ts["after"] == [["qkdl-a-b-1", 7000, 3000], ["qkdl-a-b-2", 0, 3000]], ts["after"]
    assert ts["stored"] == 0   # min(7000, 0) / 352


def test_the_python_controller_allocates_by_the_same_rule() -> None:
    c = SdnController()
    for qkdn_id in ("qkdn-a", "qkdn-relay-1", "qkdn-relay-2", "qkdn-b"):
        c.register_node(SdQkdNode(qkdn_id=qkdn_id, qkdn_location_id=qkdn_id,
                                  interfaces=[QkdInterface(qkdi_id=f"{qkdn_id}-if0")]))
    c.create_virtual_link("qkdl-a-b-1", "qkdn-a", "qkdn-b", path=["qkdn-relay-1"], skr=1e4, key_budget_bits=10000)
    c.create_virtual_link("qkdl-a-b-2", "qkdn-a", "qkdn-b", path=["qkdn-relay-2"], skr=1e4, key_budget_bits=3000)
    c.register_application(QkdApplication(app_id="sae", server_app_id="s", backing_qkdl_id=["qkdl-a-b-1", "qkdl-a-b-2"],
                                          qsh_key_combination="xor"))
    assert [c.allocate("sae", 2500), c.allocate("sae", 2500)] == [2500, 500]
    assert (c.links["qkdl-a-b-1"].key_budget_bits, c.links["qkdl-a-b-2"].key_budget_bits) == (7000, 0)


def test_one_path_key_alone_says_nothing_about_the_split_key() -> None:
    k1 = bytes(64)   # what relay 1 saw: fixed, so any dependence would show in the result's bytes
    counts: collections.Counter[int] = collections.Counter()
    for _ in range(4000):
        key = xor_split_key([k1, secrets.token_bytes(64)])
        counts.update(key)
    n = 4000 * 64
    chi = sum((counts.get(b, 0) - n / 256) ** 2 / (n / 256) for b in range(256))
    assert chi < 350, chi   # 255 degrees of freedom: the 0.9999 quantile is 347.7 (scipy.stats.chi2.ppf)
    with pytest.raises(ValueError):
        xor_split_key([b"ab"])
    assert xor_split_key([b"\x0f\xf0", b"\xff\x00"]) == b"\xf0\xf0"


def test_the_trust_note_says_what_a_relay_sees(ts) -> None:
    virt = [n for n in ts["split"]["trust"] if n.startswith("relayed")]
    assert len(virt) == 2 and all("this path's key in the clear" in n and "learns nothing" in n for n in virt), virt
    assert any("each sees the key in the clear" in n for n in ts["one"]["trust"])
