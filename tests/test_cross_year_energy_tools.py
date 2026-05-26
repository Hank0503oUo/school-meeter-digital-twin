from __future__ import annotations

import asyncio

from src.demo_mcp_server import (
    build_server,
    compare_building_trends_impl,
    load_cross_year_energy_frame,
    query_energy_records_impl,
    rank_energy_buildings_across_years_impl,
)


def test_load_cross_year_energy_frame_reads_multiple_years():
    frame = load_cross_year_energy_frame(campus="NTU", years=[2014, 2020])

    assert not frame.empty
    assert {2014, 2020}.issubset(set(frame["year"].unique()))
    assert {"name", "mean_kw", "annual_kwh", "eui"}.issubset(frame.columns)


def test_query_energy_records_filters_building_and_metrics():
    result = query_energy_records_impl(
        campus="NTU",
        years=[2014, 2020],
        buildings=["總圖書館"],
        metrics=["mean_kw", "eui"],
    )

    assert result["status"] == "ok"
    assert result["rows"]
    assert {row["year"] for row in result["rows"]} == {2014, 2020}
    assert all("總圖書館" in row["name"] for row in result["rows"])
    assert all("mean_kw" in row and "eui" in row for row in result["rows"])


def test_compare_building_trends_returns_delta_summary():
    result = compare_building_trends_impl(
        campus="NTU",
        years=[2014, 2020],
        buildings=["總圖書館"],
        metric="mean_kw",
    )

    assert result["status"] == "ok"
    assert result["rows"]
    building_summary = result["summary"]["buildings"][0]
    assert building_summary["name"] == "總圖書館"
    assert building_summary["first_year"] == 2014
    assert building_summary["last_year"] == 2020
    assert "delta" in building_summary


def test_rank_energy_buildings_across_years_has_year_building_metric():
    result = rank_energy_buildings_across_years_impl(
        campus="NTU",
        years=[2014, 2020],
        metric="mean_kw",
        top_n=5,
    )

    assert result["status"] == "ok"
    assert len(result["rows"]) == 5
    assert all({"year", "name", "mean_kw"}.issubset(row) for row in result["rows"])


def test_demo_mcp_server_exposes_cross_year_tools():
    server = build_server()
    tools = asyncio.run(server.list_tools())
    names = {tool.name for tool in tools}

    assert {
        "query_energy_records",
        "compare_building_trends",
        "rank_energy_buildings_across_years",
    }.issubset(names)
