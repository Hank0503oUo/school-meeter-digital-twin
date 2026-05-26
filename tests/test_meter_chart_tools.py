from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.demo_mcp_server import build_server, generate_meter_chart_impl


pytest.importorskip("plotly.express")

_ROOT = Path(__file__).resolve().parents[1]


def test_generate_meter_chart_creates_line_html():
    csv_path = _ROOT / "outputs" / "test_meter_csvs" / "meter_line.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        "日期時間,meter_a,meter_b\n"
        "2020-01-01,10,20\n"
        "2020-01-02,12,18\n"
        "2020-01-03,14,24\n",
        encoding="utf-8",
    )

    result = generate_meter_chart_impl(
        csv_path=str(csv_path),
        chart_type="line",
        x="日期時間",
        y=["meter_a"],
        title="test line chart",
    )

    assert result["status"] == "ok"
    assert result["chart_type"] == "line"
    assert result["used_x"] == "日期時間"
    assert result["used_y"] == ["meter_a"]

    html = result["chart_path"]
    assert html.endswith(".html")
    assert "Plotly.newPlot" in open(html, encoding="utf-8").read()


def test_generate_meter_chart_creates_bar_compare_html():
    csv_path = _ROOT / "outputs" / "test_meter_csvs" / "meter_bar.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        "日期時間,meter_a,meter_b\n"
        "2020-01-01,10,20\n"
        "2020-01-02,12,18\n",
        encoding="utf-8",
    )

    result = generate_meter_chart_impl(
        csv_path=str(csv_path),
        chart_type="bar",
        x="日期時間",
        y=["meter_a", "meter_b"],
        aggregation="mean",
        title="test bar chart",
    )

    assert result["status"] == "ok"
    assert result["chart_type"] == "bar"
    assert result["used_y"] == ["meter_a", "meter_b"]
    assert result["plot_mode"] == "meter_aggregate"


def test_generate_meter_chart_uses_default_meter_csv():
    result = generate_meter_chart_impl(chart_type="line", limit=5, title="default meter chart")

    assert result["status"] == "ok"
    assert result["row_count"] == 5
    assert result["source_column_count"] > 1
    assert result["chart_path"].endswith(".html")


def test_demo_mcp_server_exposes_meter_chart_tool():
    server = build_server()
    tools = asyncio.run(server.list_tools())
    names = {tool.name for tool in tools}

    assert "generate_meter_chart" in names
