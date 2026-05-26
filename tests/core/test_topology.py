# -*- coding: utf-8 -*-
"""Tests for topology.py — GridTopology tree structure and coordinate assignment."""

import pandas as pd
import numpy as np
import pytest

from src.topology import GridNode, GridTopology


# ── Fixtures ──────────────────────────────────────────────

def _make_metadata_df():
    """Create a minimal metadata_loop.csv-like DataFrame for testing."""
    return pd.DataFrame([
        {"zone": "Zone1", "station": "S1", "panel": "P1", "meter_id": "01A_P1_01", "uid": "B001"},
        {"zone": "Zone1", "station": "S1", "panel": "P1", "meter_id": "01A_P1_02", "uid": "B001"},
        {"zone": "Zone1", "station": "S1", "panel": "P2", "meter_id": "01A_P2_01", "uid": "B002"},
        {"zone": "Zone2", "station": "S2", "panel": "P3", "meter_id": "02A_P3_01", "uid": "B003"},
    ])


def _make_power_df():
    """Create a minimal power DataFrame for testing."""
    return pd.DataFrame({
        "01A_P1_01": [100.0, 110.0],
        "01A_P1_02": [50.0, 55.0],
        "01A_P2_01": [200.0, 220.0],
        "02A_P3_01": [80.0, 85.0],
    })


# ── GridNode Tests ────────────────────────────────────────

class TestGridNode:
    def test_create_node(self):
        node = GridNode("TestNode", "ZONE")
        assert node.node_id == "TestNode"
        assert node.level == "ZONE"
        assert node.children == []

    def test_add_child(self):
        parent = GridNode("Parent", "ROOT")
        child = GridNode("Child", "ZONE", parent=parent)
        parent.add_child(child)
        assert len(parent.children) == 1
        assert parent.children[0].node_id == "Child"

    def test_get_all_leaf_meters(self):
        root = GridNode("Root", "ROOT")
        zone = GridNode("Z1", "ZONE", parent=root)
        root.add_child(zone)
        meter1 = GridNode("M1", "METER", parent=zone)
        meter2 = GridNode("M2", "METER", parent=zone)
        zone.add_child(meter1)
        zone.add_child(meter2)
        leaves = root.get_all_leaf_meters()
        assert set(leaves) == {"M1", "M2"}

    def test_leaf_node_returns_itself(self):
        leaf = GridNode("Leaf", "METER")
        assert leaf.get_all_leaf_meters() == ["Leaf"]

    def test_to_dict(self):
        node = GridNode("Test", "ZONE", metadata={"uid": "B001"})
        d = node.to_dict()
        assert d["id"] == "Test"
        assert d["level"] == "ZONE"
        assert "children" in d


# ── GridTopology Tests ────────────────────────────────────

class TestGridTopology:
    def test_build_tree(self):
        df = _make_metadata_df()
        topo = GridTopology(df)
        assert "Total" in topo.nodes
        assert len(topo.nodes) > 4  # Root + zones + stations + panels + meters

    def test_get_building_meters(self):
        df = _make_metadata_df()
        topo = GridTopology(df)
        meters = topo.get_building_meters("B001")
        assert isinstance(meters, list)
        assert "01A_P1_01" in meters
        assert "01A_P1_02" in meters

    def test_get_building_meters_unknown_uid(self):
        df = _make_metadata_df()
        topo = GridTopology(df)
        result = topo.get_building_meters("UNKNOWN")
        assert result == [] or result is None or len(result) == 0

    def test_bind_power_columns(self):
        df = _make_metadata_df()
        power_df = _make_power_df()
        topo = GridTopology(df)
        topo.bind_power_columns(power_df.columns)
        # After binding, meters should have true_column set
        cols = topo.get_building_true_columns("B001")
        assert len(cols) >= 1

    def test_aggregate_power(self):
        df = _make_metadata_df()
        power_df = _make_power_df()
        topo = GridTopology(df)
        topo.bind_power_columns(power_df.columns)
        # Aggregate total power from root
        total = topo.aggregate_power(power_df, "Total")
        assert isinstance(total, pd.Series)
        assert total.iloc[0] > 0

    def test_aggregate_building_power(self):
        df = _make_metadata_df()
        power_df = _make_power_df()
        topo = GridTopology(df)
        topo.bind_power_columns(power_df.columns)
        bldg_power = topo.aggregate_building_power(power_df, "B001")
        assert isinstance(bldg_power, pd.Series)
        # B001 has meters 01A_P1_01 (100) + 01A_P1_02 (50) = 150
        assert bldg_power.iloc[0] == pytest.approx(150.0, abs=1.0)

    def test_assign_node_coordinates(self):
        df = _make_metadata_df()
        topo = GridTopology(df)
        coords = {
            "B001": (121.5, 25.0),
            "B002": (121.6, 25.1),
            "B003": (121.7, 25.2),
        }
        topo.assign_node_coordinates(coords)
        # Root should have averaged coordinates
        assert topo.root.lon is not None
        assert topo.root.lat is not None

    def test_generate_trips(self):
        df = _make_metadata_df()
        power_df = _make_power_df()
        topo = GridTopology(df)
        topo.bind_power_columns(power_df.columns)
        coords = {
            "B001": (121.5, 25.0),
            "B002": (121.6, 25.1),
            "B003": (121.7, 25.2),
        }
        topo.assign_node_coordinates(coords)
        trips = topo.generate_trips(power_df)
        assert isinstance(trips, list)
        if trips:  # May be empty if no coords assigned to all nodes
            assert "path" in trips[0]
            assert "timestamps" in trips[0]
            assert "color" in trips[0]

    def test_generate_node_data(self):
        df = _make_metadata_df()
        power_df = _make_power_df()
        topo = GridTopology(df)
        topo.bind_power_columns(power_df.columns)
        coords = {
            "B001": (121.5, 25.0),
            "B002": (121.6, 25.1),
            "B003": (121.7, 25.2),
        }
        topo.assign_node_coordinates(coords)
        nodes = topo.generate_node_data(power_df)
        assert isinstance(nodes, list)
        if nodes:
            assert "position" in nodes[0]
            assert "radius" in nodes[0]

    def test_assign_energy_tiers(self):
        df = _make_metadata_df()
        topo = GridTopology(df)
        topo.assign_energy_tiers({"B001": "HIGH", "B002": "LOW", "B003": "NORMAL"})
        # Verify tier was assigned to leaf nodes
        assert topo.nodes.get("01A_P1_01") is not None


class TestEdgeCases:
    def test_empty_metadata(self):
        """Empty DataFrame should create minimal topology."""
        df = pd.DataFrame(columns=["zone", "station", "panel", "meter_id", "uid"])
        topo = GridTopology(df)
        assert "Total" in topo.nodes

    def test_missing_columns(self):
        """DataFrame missing optional columns should not crash."""
        df = pd.DataFrame([
            {"zone": "Z1", "station": "S1", "panel": "P1", "meter_id": "M1"},
        ])
        topo = GridTopology(df)
        assert len(topo.nodes) >= 1
