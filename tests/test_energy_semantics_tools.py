from __future__ import annotations

import asyncio
from pathlib import Path

from src.demo_mcp_server import build_server
from src.energy_semantics import list_rtem_sources_impl, map_energy_semantics_impl


def test_rtem_source_registry_marks_placeholders_unavailable(tmp_path: Path) -> None:
    csv_path = tmp_path / "meter.csv"
    csv_path.write_text("timestamp,kW\n2020-01-01,12.5\n", encoding="utf-8")

    result = list_rtem_sources_impl(meter_csv_path=str(csv_path))

    assert result["status"] == "ok"
    sources = {item["source_id"]: item for item in result["sources"]}
    assert sources["electricity_meter_csv"]["available"] is True
    assert sources["DHW"]["available"] is False
    assert sources["GAS"]["available"] is False
    assert sources["STM"]["available"] is False


def test_energy_semantic_mapping_returns_haystack_style_tags(tmp_path: Path) -> None:
    csv_path = tmp_path / "meter.csv"
    csv_path.write_text("timestamp,kW\n2020-01-01,12.5\n", encoding="utf-8")

    result = map_energy_semantics_impl(
        building_uid="B001",
        meter_name="main_meter",
        source_id="electricity_meter_csv",
        campus="NTU",
        meter_csv_path=str(csv_path),
    )

    assert result["status"] == "ok"
    tags = result["semantic_tags"]
    for tag in ("site", "building", "meter", "elec", "kW", "point"):
        assert tag in tags
    assert any(item["predicate"] == "isPartOf" for item in result["relationships"])


def test_energy_semantic_mapping_rejects_unknown_source() -> None:
    result = map_energy_semantics_impl(source_id="unknown")
    assert result["status"] == "unknown_source"
    assert result["semantic_tags"] == {}


def test_energy_semantic_tools_are_registered() -> None:
    server = build_server()
    tools = asyncio.run(server.list_tools())
    names = {tool.name for tool in tools}

    assert "list_rtem_sources" in names
    assert "map_energy_semantics" in names
