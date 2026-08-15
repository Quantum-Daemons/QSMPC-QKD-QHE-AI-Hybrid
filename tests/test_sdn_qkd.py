"""ETSI GS QKD 015 control plane + GS QKD 014 key delivery.

These assert the two properties that make the model worth having rather than decorative: a virtual
(multi-hop) link must declare its trusted-node exposure, and a key ID must not be redeemable by an
SAE it was not issued to.
"""
from __future__ import annotations

import pytest

from qsh.qkd.kme import KeyManagementEntity
from qsh.qkd.sdn_controller import (
    QkdApplication,
    QkdInterface,
    SdnController,
    SdQkdNode,
)


def _three_node_controller() -> SdnController:
    c = SdnController()
    for qkdn_id, loc in (("qkdn-a", "site A"), ("qkdn-r", "relay"), ("qkdn-b", "site B")):
        c.register_node(SdQkdNode(qkdn_id=qkdn_id, qkdn_location_id=loc,
                                  interfaces=[QkdInterface(qkdi_id=f"{qkdn_id}-if0")]))
    return c


def test_physical_link_is_direct_and_declares_no_relay():
    c = _three_node_controller()
    link = c.create_physical_link("qkdl-ab", "qkdn-a", "qkdn-b", attenuation_db=6.0, skr=1000.0)
    assert link.qkdl_type == "PHYS"
    assert link.traverses_trusted_nodes is False
    assert "no intermediate node" in link.trust_note()


def test_virtual_link_declares_trusted_node_exposure():
    """Clause 5.4 relaying means an intermediate node holds the key in plaintext. Say so."""
    c = _three_node_controller()
    link = c.create_virtual_link("qkdl-ab", "qkdn-a", "qkdn-b", path=["qkdn-r"], skr=800.0)
    assert link.qkdl_type == "VIRT"
    assert link.traverses_trusted_nodes is True
    assert "qkdn-r" in link.trust_note()
    assert "in the clear" in link.trust_note()


def test_virtual_link_refuses_a_hop_that_does_not_enable_relay():
    c = _three_node_controller()
    c.nodes["qkdn-r"].key_relay_mode_enable = False
    with pytest.raises(ValueError, match="does not enable key relay"):
        c.create_virtual_link("qkdl-ab", "qkdn-a", "qkdn-b", path=["qkdn-r"], skr=800.0)


def test_unknown_node_is_rejected_rather_than_invented():
    c = _three_node_controller()
    with pytest.raises(ValueError, match="unknown SD-QKD node"):
        c.create_physical_link("qkdl-x", "qkdn-a", "qkdn-ghost", attenuation_db=1.0, skr=10.0)


def test_allocation_is_bounded_by_the_link_budget():
    """The budget is BITS, and it always was - the field it used to read was not.

    This asserted `skr=1000.0` and then that 1000 bits could be allocated, which reads as a test of
    the ledger and was a test of a bit/s field being spent as though it were a bit count. Round 42
    removed that operation from the browser and left it here; the assertions moved with the fix
    because they were part of its surface, not collateral to it.
    """
    c = _three_node_controller()
    c.create_physical_link("qkdl-ab", "qkdn-a", "qkdn-b", attenuation_db=6.0, skr=1000.0,
                           key_budget_bits=1000, channel_time_s=1.0)
    c.register_application(QkdApplication(
        app_id="sae-1", server_app_id="orchestrator", backing_qkdl_id=["qkdl-ab"]))
    assert c.allocate("sae-1", 400) == 400
    # The remaining budget, not the request, decides.
    assert c.allocate("sae-1", 10_000) == 600
    assert c.allocate("sae-1", 1) == 0


def test_the_rate_fields_are_derived_and_cannot_be_written() -> None:
    """`eskr` and `expected_consumption` have no storage, so `allocate()` cannot decrement them.

    A comment asking nobody to spend a rate is weaker than removing the thing to be spent. These are
    read-only properties: the assignment Round 42 deleted from the browser is now an AttributeError
    here rather than a convention.
    """
    c = _three_node_controller()
    c.create_physical_link("qkdl-ab", "qkdn-a", "qkdn-b", attenuation_db=6.0, skr=1000.0,
                           key_budget_bits=1000, channel_time_s=2.0)
    link = c.links["qkdl-ab"]
    assert link.eskr == 1000.0, "with nothing consumed, the effective rate is the rate"

    c.register_application(QkdApplication(
        app_id="sae-1", server_app_id="orchestrator", backing_qkdl_id=["qkdl-ab"]))
    c.allocate("sae-1", 400)
    # 400 bits over 2 s is 200 bit/s of consumption, so the effective rate falls by 200 - not by 400.
    # A rate decremented by a bit count is exactly the dimensional error this file now guards.
    assert link.consumed_bits == 400
    assert link.expected_consumption == 200.0
    assert link.eskr == 800.0

    for attr, value in (("eskr", 1.0), ("expected_consumption", 1.0)):
        with pytest.raises(AttributeError):
            setattr(link, attr, value)


def test_a_link_with_no_clock_reports_no_consumption_rather_than_dividing_by_zero() -> None:
    """`channel_time_s` is optional, and its absence must not produce Infinity.

    A rate of Infinity renders, and a rendered number is believed. Absent clock means the
    consumption rate is unknown, and the honest value for unknown here is zero consumption against
    the stated rate - never a division by zero.
    """
    c = _three_node_controller()
    c.create_physical_link("qkdl-ab", "qkdn-a", "qkdn-b", attenuation_db=6.0, skr=500.0,
                           key_budget_bits=1000)
    c.register_application(QkdApplication(
        app_id="sae-1", server_app_id="orchestrator", backing_qkdl_id=["qkdl-ab"]))
    c.allocate("sae-1", 400)
    link = c.links["qkdl-ab"]
    assert link.consumed_bits == 400, "the bit ledger still moves"
    assert link.expected_consumption == 0.0
    assert link.eskr == 500.0


def test_application_referencing_an_unknown_link_is_rejected():
    c = _three_node_controller()
    with pytest.raises(ValueError, match="unknown link"):
        c.register_application(QkdApplication(
            app_id="sae-1", server_app_id="orchestrator", backing_qkdl_id=["qkdl-nope"]))


def test_topology_uses_the_standard_field_names_and_admits_simulation():
    c = _three_node_controller()
    c.create_virtual_link("qkdl-ab", "qkdn-a", "qkdn-b", path=["qkdn-r"], skr=500.0)
    t = c.topology()
    assert t["spec"].startswith("ETSI GS QKD 015")
    assert {"qkd_node", "qkd_link", "qkd_app"} <= set(t)
    node = t["qkd_node"][0]
    assert {"qkdn_id", "qkdn_status", "qkdn_capabilities", "qkd_interface"} <= set(node)
    link = t["qkd_link"][0]
    assert {"qkdl_id", "qkdl_type", "virt_next_hop", "qkdl_performance"} <= set(link)
    assert link["qkdl_performance"].keys() >= {"skr", "eskr", "expected_consumption"}
    # The quantum layer is simulated and the payload must never imply otherwise.
    assert t["simulated"] is True
    assert "SIMULATED" in t["simulation_note"]


# --------------------------------------------------------------------------------------------- #
#  GS QKD 014 key delivery
# --------------------------------------------------------------------------------------------- #
def test_master_and_slave_resolve_the_same_key_from_an_id_alone():
    kme = KeyManagementEntity(KME_ID="kme-a")
    issued = kme.get_key("sae-slave", master_SAE_ID="sae-master", number=2)
    assert len(issued["keys"]) == 2
    ids = [k["key_ID"] for k in issued["keys"]]
    redeemed = kme.get_key_with_ids("sae-master", ids, slave_SAE_ID="sae-slave")
    assert [k["key"] for k in redeemed["keys"]] == [k["key"] for k in issued["keys"]]


def test_a_key_id_is_not_a_bearer_token_for_any_sae():
    """The binding the spec leans on: an ID seen in transit must not be redeemable by an outsider."""
    kme = KeyManagementEntity(KME_ID="kme-a")
    issued = kme.get_key("sae-slave", master_SAE_ID="sae-master", number=1)
    kid = issued["keys"][0]["key_ID"]
    with pytest.raises(PermissionError, match="was issued for"):
        kme.get_key_with_ids("sae-master", [kid], slave_SAE_ID="sae-eve")


def test_unknown_key_id_is_an_error_not_a_fresh_key():
    kme = KeyManagementEntity(KME_ID="kme-a")
    with pytest.raises(KeyError):
        kme.get_key_with_ids("sae-master", ["nope"], slave_SAE_ID="sae-slave")


def test_kme_refuses_rather_than_silently_returning_fewer_keys():
    kme = KeyManagementEntity(KME_ID="kme-a", max_key_per_request=2)
    with pytest.raises(ValueError, match="exceeds max_key_per_request"):
        kme.get_key("sae-slave", master_SAE_ID="sae-master", number=3)
    kme._available = 1
    with pytest.raises(ValueError, match="cannot satisfy"):
        kme.get_key("sae-slave", master_SAE_ID="sae-master", number=2)


def test_status_declares_simulation_and_names_the_edition():
    kme = KeyManagementEntity(KME_ID="kme-a")
    st = kme.status("sae-slave", master_SAE_ID="sae-master")
    assert st["status_extension"]["simulated"] is True
    # V1.1.1 (2019-02) is still the only published edition; ed2 remains a draft.
    assert "V1.1.1" in st["status_extension"]["spec"]
    assert st["stored_key_count"] > 0


def test_authentication_gap_is_stated_not_implied():
    note = KeyManagementEntity.authentication_note()
    assert "mutual TLS" in note and "client certificate" in note
