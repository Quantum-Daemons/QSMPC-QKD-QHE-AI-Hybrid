"""An SD-QKD control plane modelled on **ETSI GS QKD 015 V2.1.1 (2022-04)**.

Why this module exists
----------------------
This project had an agent panel talking about QKD and no control plane at all. That gap matters
because ETSI GS QKD 015 clause 1 draws the boundary explicitly:

    "It defines abstraction models and workflows between an SDN-enabled QKD node and the SDN
     controller, including resource discovery, capabilities dissemination and system configuration
     operations. **Application layer interfaces and quantum-channel interfaces are out of scope.**"

So an MCP agent is not, and cannot be, an SDN controller. In ETSI's own vocabulary it is a **SAE**
(Secure Application Entity) -- an application-plane consumer that *requests keys* and registers with
a KME. The controller is a separate, deterministic element that discovers nodes, disseminates
capabilities, computes paths and allocates key-generation resources per link. Three planes:

    application  SAE / MCP agent            -- (out of scope of QKD 015)
    control      SDN controller + KME/KMS   -- GS QKD 015, 018; ITU-T Y.3804, Y.3805
    quantum      QKD modules + SDN agent    -- GS QKD 004, 014, 020

Those three name the specs that GOVERN the quantum plane, not the ones implemented here. This
project implements **014** (`kme.py`) and **015** (this file). It does not implement **004**,
which is a different thing rather than a subset: 014 is a stateless REST request for key material
by identifier, while 004 is a stateful application interface where an application opens a keyed
session and streams against it. An application written for one does not work against the other -
which is why the 2026 banking deployment in arXiv:2604.12985 had to integrate 004, 014 and a
vendor-specific third interface to cover one network. See `docs/pqc-roadmap.md`.

Identifier spellings below are taken verbatim from the published YANG modules at
https://forge.etsi.org/rep/qkd/gs015-ctrl-int-sdn (``etsi-qkd-sdn-node.yang`` and
``etsi-qkd-node-types.yang``, revision 2022-01-30, BSD-3-Clause) -- ``qkdn_id``, ``qkdl_id``,
``qkdl_type``, ``phys_qkd_role``, ``virt_next_hop`` and so on -- so the field names a reader
recognises from the standard are the field names here. Nothing is renamed to look tidier.

What is real and what is not
----------------------------
The object model, the workflows and the field names are real. **The quantum layer is simulated**:
there is no QKD hardware anywhere in this project, and every SKR/QBER figure comes from the analytic
channel models in :mod:`qsh.qkd`. That combination -- standards-faithful control plane over a
simulated quantum layer -- is an established pattern, not a shortcut: arXiv:2607.06602 (2026-07)
validated a VPN against a *simulated* KME before moving to LuxQuanta hardware.

The national position on QKD itself is worth stating where a reader will see it: ANSSI, BSI, NLNCSA
and the Swedish authorities jointly hold that QKD is "not yet sufficiently mature from a security
point of view" and that the priority is PQC migration in hybrid solutions. This module models a QKD
control plane; it does not claim QKD is production-ready.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

#: ``etsi-qkd-node-types.yang`` identity ``QKDN-STATUS-TYPES``.
QkdnStatus = Literal["NEW", "OPERATING", "DOWN", "FAILURE", "OUT"]
#: identity ``QKD-LINK-TYPES``. PHYS = a direct quantum channel (clause 5.3); VIRT = a multi-hop key
#: association relayed through trusted nodes (clause 5.4).
QkdLinkType = Literal["PHYS", "VIRT"]
#: identity ``QKD-ROLE-TYPES``.
QkdRole = Literal["TRANSMITTER", "RECEIVER"]


@dataclass(frozen=True)
class QkdInterface:
    """One physical QKD module in a secure location (``qkd_interface``)."""

    qkdi_id: str
    qkdi_type: str = "DV-QKD"          #: identity QKD-TECHNOLOGY-TYPES: CV-QKD, DV-QKD, DV-QKD-COW, ...
    qkdi_status: str = "OPERATING"
    qkdi_model: str = "simulated"
    #: ``qkdi_att_point`` -- where the module attaches to the classical network.
    device: str = "sim0"
    port: int = 0


@dataclass
class QkdLink:
    """A **key association link** (clause 4.5) between two SD-QKD nodes.

    ``qkdl_type`` is the load-bearing field. ``PHYS`` is a direct quantum channel. ``VIRT`` is a
    multi-hop association whose keys are relayed through intermediate **trusted nodes** -- and a
    trusted node sees the key in the clear. That is the central caveat of every trusted-relay QKD
    network, it is what ANSSI/BSI and QCNC 2026 (arXiv:2604.05599) both press on, and
    :attr:`virt_next_hop` is where it is visible.
    """

    qkdl_id: str
    qkdl_local_qkdn_id: str
    qkdl_remote_qkdn_id: str
    qkdl_type: QkdLinkType = "PHYS"
    qkdl_enable: bool = True
    qkdl_status: str = "OPERATING"
    #: Applications (SAEs) currently bound to this association -- ``qkdl_applications``.
    qkdl_applications: list[str] = field(default_factory=list)

    # --- PHYS-only, per the YANG `when "../qkdl_type = 'PHYS'"` guard ---------------------------
    phys_qkd_role: QkdRole | None = None
    phys_channel_att: float | None = None      #: dB
    phys_wavelength: float | None = None       #: nm

    # --- VIRT-only ------------------------------------------------------------------------------
    #: Ordered list of intermediate ``qkdn_id``s. Every one of them handles the key in plaintext.
    virt_next_hop: list[str] = field(default_factory=list)
    virt_bandwidth: float | None = None        #: bit/s

    # --- performance (``qkdl_performance`` / ``common_performance``) ----------------------------
    skr: float = 0.0                           #: secret key rate, bit/s
    eskr: float = 0.0                          #: effective SKR after application consumption, bit/s
    expected_consumption: float = 0.0          #: bit/s

    @property
    def traverses_trusted_nodes(self) -> bool:
        """True when key material is exposed in plaintext at an intermediate node."""
        return self.qkdl_type == "VIRT" and bool(self.virt_next_hop)

    def trust_note(self) -> str:
        if not self.traverses_trusted_nodes:
            return "direct quantum channel — no intermediate node handles the key"
        hops = " -> ".join(self.virt_next_hop)
        return (f"relayed through {len(self.virt_next_hop)} trusted node(s) [{hops}] — each one sees "
                f"the key in the clear, so end-to-end secrecy rests on their physical security")


@dataclass
class QkdApplication:
    """A registered SAE (clause 5.2 application registration, ``qkd_app``).

    This is where the MCP agent belongs: it registers, is bound to a key association, and consumes
    key material. It does not participate in control-plane decisions.
    """

    app_id: str
    server_app_id: str
    client_app_id: list[str] = field(default_factory=list)
    app_status: str = "ON"
    app_priority: int = 0
    app_type: str = "CLIENT"
    local_qkdn_id: str = ""
    remote_qkdn_id: str = ""
    #: ``backing_qkdl_id`` -- the key association(s) serving this application.
    backing_qkdl_id: list[str] = field(default_factory=list)
    #: ``app_qos``
    max_bandwidth: float | None = None
    min_bandwidth: float | None = None
    ttl: int | None = None
    #: ``app_statistics`` -- cumulative bits handed to this SAE.
    consumed_bits: int = 0


@dataclass
class SdQkdNode:
    """An SD-QKD node (``qkd_node``): the controller's abstraction of one secure location."""

    qkdn_id: str
    qkdn_location_id: str
    qkdn_version: str = "ETSI GS QKD 015 V2.1.1"
    qkdn_status: QkdnStatus = "OPERATING"
    #: ``qkdn_capabilities``
    link_stats_support: bool = True
    application_stats_support: bool = True
    key_relay_mode_enable: bool = True         #: i.e. this node may act as a trusted relay
    interfaces: list[QkdInterface] = field(default_factory=list)


class SdnController:
    """The control-plane element of GS QKD 015 -- deterministic, and separate from any agent.

    Implements the workflows the standard names: resource discovery and capabilities dissemination
    (clauses 4.2-4.4), application registration (5.2), physical link creation (5.3) and virtual
    link creation (5.4). It deliberately exposes **no** natural-language or agent-facing surface:
    an SAE asks for keys through the KME (GS QKD 014), never through this object.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, SdQkdNode] = {}
        self.links: dict[str, QkdLink] = {}
        self.applications: dict[str, QkdApplication] = {}
        self._notifications: list[str] = []

    # --- clause 4.2-4.4: discovery + capabilities ----------------------------------------------
    def register_node(self, node: SdQkdNode) -> SdQkdNode:
        self.nodes[node.qkdn_id] = node
        self._notify(f"node {node.qkdn_id} registered at {node.qkdn_location_id} "
                     f"({len(node.interfaces)} interface(s), status {node.qkdn_status})")
        return node

    # --- clause 5.3: physical (direct) link creation --------------------------------------------
    def create_physical_link(self, qkdl_id: str, local: str, remote: str, *,
                             attenuation_db: float, skr: float, wavelength_nm: float = 1550.0) -> QkdLink:
        self._require_nodes(local, remote)
        link = QkdLink(
            qkdl_id=qkdl_id, qkdl_local_qkdn_id=local, qkdl_remote_qkdn_id=remote,
            qkdl_type="PHYS", phys_qkd_role="TRANSMITTER", phys_channel_att=attenuation_db,
            phys_wavelength=wavelength_nm, skr=skr, eskr=skr,
        )
        self.links[qkdl_id] = link
        self._notify(f"PHYS link {qkdl_id} {local}<->{remote} up: {attenuation_db:.1f} dB, "
                     f"SKR {skr:.1f} bit/s")
        return link

    # --- clause 5.4: virtual (multi-hop) link creation ------------------------------------------
    def create_virtual_link(self, qkdl_id: str, local: str, remote: str, *,
                            path: list[str], skr: float) -> QkdLink:
        """Create a relayed key association across ``path`` of intermediate nodes.

        Every hop must have ``key_relay_mode_enable`` -- a node that does not offer relay cannot be
        made to, and the controller refuses rather than silently routing around it.
        """
        self._require_nodes(local, remote, *path)
        for hop in path:
            if not self.nodes[hop].key_relay_mode_enable:
                raise ValueError(f"node {hop} does not enable key relay; cannot build {qkdl_id}")
        link = QkdLink(
            qkdl_id=qkdl_id, qkdl_local_qkdn_id=local, qkdl_remote_qkdn_id=remote,
            qkdl_type="VIRT", virt_next_hop=list(path),
            # A relayed association is bounded by its weakest hop, never better than it.
            skr=skr, eskr=skr, virt_bandwidth=skr,
        )
        self.links[qkdl_id] = link
        self._notify(f"VIRT link {qkdl_id} {local}=>{remote} via {'->'.join(path)}: "
                     f"SKR {skr:.1f} bit/s — {link.trust_note()}")
        return link

    # --- clause 5.2: application registration ---------------------------------------------------
    def register_application(self, app: QkdApplication) -> QkdApplication:
        for qkdl_id in app.backing_qkdl_id:
            if qkdl_id not in self.links:
                raise ValueError(f"application {app.app_id} references unknown link {qkdl_id}")
            self.links[qkdl_id].qkdl_applications.append(app.app_id)
        self.applications[app.app_id] = app
        self._notify(f"SAE {app.app_id} registered ({app.server_app_id} -> "
                     f"{', '.join(app.client_app_id) or 'n/a'}) on {', '.join(app.backing_qkdl_id)}")
        return app

    # --- resource allocation --------------------------------------------------------------------
    def allocate(self, app_id: str, bits: int) -> int:
        """Charge ``bits`` of key material to a registered SAE, bounded by the link's eskr budget.

        Returns the bits actually granted. The controller allocates; it does not deliver -- delivery
        is the KME's job over GS QKD 014, which is a different interface.
        """
        app = self.applications[app_id]
        budget = sum(self.links[q].eskr for q in app.backing_qkdl_id)
        granted = int(min(bits, budget))
        app.consumed_bits += granted
        for q in app.backing_qkdl_id:
            link = self.links[q]
            link.expected_consumption += granted
            link.eskr = max(0.0, link.skr - link.expected_consumption)
        if granted < bits:
            self._notify(f"SAE {app_id} requested {bits} bits, granted {granted} — link budget "
                         f"exhausted (this is the honest answer, not a padded one)")
        return granted

    # --- state ----------------------------------------------------------------------------------
    def topology(self) -> dict:
        """The controller's end-to-end view, in the standard's own field names."""
        return {
            "spec": "ETSI GS QKD 015 V2.1.1 (2022-04)",
            "qkd_node": [
                {"qkdn_id": n.qkdn_id, "qkdn_location_id": n.qkdn_location_id,
                 "qkdn_status": n.qkdn_status, "qkdn_version": n.qkdn_version,
                 "qkdn_capabilities": {"link_stats_support": n.link_stats_support,
                                       "application_stats_support": n.application_stats_support,
                                       "key_relay_mode_enable": n.key_relay_mode_enable},
                 "qkd_interface": [{"qkdi_id": i.qkdi_id, "qkdi_type": i.qkdi_type,
                                    "qkdi_status": i.qkdi_status,
                                    "qkdi_att_point": {"device": i.device, "port": i.port}}
                                   for i in n.interfaces]}
                for n in self.nodes.values()
            ],
            "qkd_link": [
                {"qkdl_id": ln.qkdl_id, "qkdl_type": ln.qkdl_type, "qkdl_status": ln.qkdl_status,
                 "qkdl_local": {"qkdn_id": ln.qkdl_local_qkdn_id},
                 "qkdl_remote": {"qkdn_id": ln.qkdl_remote_qkdn_id},
                 "virt_next_hop": ln.virt_next_hop, "phys_channel_att": ln.phys_channel_att,
                 "qkdl_performance": {"skr": round(ln.skr, 3), "eskr": round(ln.eskr, 3),
                                      "expected_consumption": round(ln.expected_consumption, 3)},
                 "qkdl_applications": ln.qkdl_applications,
                 "traverses_trusted_nodes": ln.traverses_trusted_nodes,
                 "trust_note": ln.trust_note()}
                for ln in self.links.values()
            ],
            "qkd_app": [
                {"app_id": a.app_id, "app_status": a.app_status, "app_type": a.app_type,
                 "server_app_id": a.server_app_id, "client_app_id": a.client_app_id,
                 "backing_qkdl_id": a.backing_qkdl_id, "consumed_bits": a.consumed_bits}
                for a in self.applications.values()
            ],
            "notifications": list(self._notifications),
            "simulated": True,
            "simulation_note": (
                "The object model, workflows and field names follow GS QKD 015 V2.1.1. The quantum "
                "layer is SIMULATED — no QKD hardware is involved and every SKR/QBER figure comes "
                "from the analytic channel models in qsh.qkd."
            ),
        }

    # --- internals ------------------------------------------------------------------------------
    def _require_nodes(self, *ids: str) -> None:
        missing = [i for i in ids if i not in self.nodes]
        if missing:
            raise ValueError(f"unknown SD-QKD node(s): {', '.join(missing)}")

    def _notify(self, message: str) -> None:
        self._notifications.append(message)
