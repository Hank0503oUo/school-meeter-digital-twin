from __future__ import annotations

from pathlib import Path

import pytest

from src.dashboard_modules.runtime import DashboardPaths
from src.project_paths import campus_data_dir, project_root, resolve_project_path
from src.rtem_codex_bridge import LocalMCPBridgeError, find_rtem_mcp_root
import src.rtem_codex_bridge as rtem_bridge


def test_resolve_project_path_uses_repo_root_for_relative_paths():
    resolved = resolve_project_path("config/demo_config.yaml")
    assert resolved == project_root() / "config" / "demo_config.yaml"
    assert resolved.exists()


def test_dashboard_paths_defaults_are_absolute():
    paths = DashboardPaths()
    assert paths.energy_geojson.is_absolute()
    assert paths.meter_hourly_csv.is_absolute()
    assert paths.build_meta_uid.is_absolute()
    assert paths.build_meta_loop.is_absolute()
    assert paths.weather_dir.is_absolute()
    assert paths.ui_prefs.is_absolute()
    assert paths.v12_summary.is_absolute()
    assert paths.official_patch.is_absolute()


def test_campus_data_dir_targets_uppercase_data_folder():
    assert campus_data_dir("ntu", "ntu_energy.geojson") == project_root() / "data" / "NTU" / "ntu_energy.geojson"


def test_find_rtem_mcp_root_honors_override(tmp_path, monkeypatch):
    override_root = tmp_path / "rtem-mcp-server"
    override_root.mkdir()
    monkeypatch.setenv("RTEM_MCP_SERVER_ROOT", str(override_root))
    assert find_rtem_mcp_root() == override_root.resolve()


def test_find_rtem_mcp_root_raises_clean_error_when_candidates_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("RTEM_MCP_SERVER_ROOT", raising=False)
    missing_a = tmp_path / "missing-a"
    missing_b = tmp_path / "missing-b"
    monkeypatch.setattr(rtem_bridge, "_iter_rtem_mcp_candidates", lambda: [missing_a, missing_b])

    with pytest.raises(LocalMCPBridgeError) as exc_info:
        find_rtem_mcp_root()

    assert "Checked:" in str(exc_info.value)
    assert "missing-a" in str(exc_info.value)
