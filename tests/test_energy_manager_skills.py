from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd

from src.demo_mcp_server import build_server
from src.energy_manager_skills import (
    append_energy_log_impl,
    detect_energy_anomalies_impl,
    generate_energy_saving_report_impl,
)


def test_detect_energy_anomalies_finds_injected_spike(tmp_path: Path) -> None:
    csv_path = tmp_path / "meter.csv"
    values = [100.0] * 24 + [500.0] + [100.0] * 24
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=len(values), freq="h"),
            "kW": values,
            "building_uid": ["B001"] * len(values),
            "meter_name": ["main"] * len(values),
        }
    ).to_csv(csv_path, index=False)

    result = detect_energy_anomalies_impl(
        csv_path=str(csv_path),
        building_uid="B001",
        meter_name="main",
        value_column="kW",
        z_threshold=3.0,
    )

    assert result["status"] == "ok"
    assert result["summary"]["anomaly_count"] >= 1
    assert result["anomalies"][0]["value"] == 500.0


def test_append_energy_log_writes_markdown(tmp_path: Path) -> None:
    log_path = tmp_path / "LOG.md"

    result = append_energy_log_impl(
        event_type="anomaly",
        title="Spike reviewed",
        summary="Operator confirmed one abnormal demand spike.",
        severity="warning",
        evidence={"score": 4.2},
        decisions=["Check schedule."],
        log_path=str(log_path),
    )

    assert result["status"] == "ok"
    assert result["memory_role"] == "audit_log_only"
    text = log_path.read_text(encoding="utf-8")
    assert "Spike reviewed" in text
    assert "[WARNING]" in text
    assert "audit_log_only" in text
    assert "harness_memory_mcp" in text
    assert "Check schedule." in text


def test_generate_energy_saving_report_writes_markdown(tmp_path: Path) -> None:
    output_path = tmp_path / "report.md"

    result = generate_energy_saving_report_impl(
        anomaly_result={
            "status": "ok",
            "summary": {"anomaly_count": 1},
            "anomalies": [{"timestamp": "2020-01-02", "value": 500.0}],
            "suggested_actions": ["Investigate overnight load."],
        },
        building_context={"building_uid": "B001"},
        report_title="B001 Energy Review",
        output_path=str(output_path),
    )

    assert result["status"] == "ok"
    text = output_path.read_text(encoding="utf-8")
    assert "B001 Energy Review" in text
    assert "Investigate overnight load." in text
    assert "harness memory MCP" in text


def test_energy_manager_tools_are_registered() -> None:
    server = build_server()
    tools = asyncio.run(server.list_tools())
    names = {tool.name for tool in tools}

    assert "detect_energy_anomalies" in names
    assert "append_energy_decision_log" in names
    assert "store_energy_memory_pattern" in names
    assert "generate_energy_saving_report" in names
