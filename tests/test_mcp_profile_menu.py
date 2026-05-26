# -*- coding: utf-8 -*-
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.mcp_profile_menu import apply_mcp_profile_menu_if_requested, load_profiles


def test_load_profiles_filters_empty_cmd(tmp_path: Path) -> None:
    p = tmp_path / "p.json"
    p.write_text(
        json.dumps(
            {
                "profiles": [
                    {"label": "bad", "cmd": ""},
                    {"label": "ok", "cmd": "echo hi", "cwd": "D:\\x"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    profs = load_profiles(p)
    assert len(profs) == 1
    assert profs[0]["label"] == "ok"
    assert profs[0]["cmd"] == "echo hi"
    assert profs[0]["cwd"] == "D:\\x"


def test_menu_sets_env_on_choice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "mcp_review_profiles.json"
    cfg.write_text(
        json.dumps(
            {
                "profiles": [
                    {"label": "A", "cmd": "cmd /c echo a"},
                    {"label": "B", "cmd": "cmd /c echo b", "cwd": str(tmp_path)},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ENERGY_MCP_MENU", "1")
    monkeypatch.setenv("ENERGY_MCP_PROFILES_JSON", str(cfg))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdin.readline", lambda: "2\n")

    apply_mcp_profile_menu_if_requested(tmp_path)

    import os

    assert os.environ.get("ENERGY_KNOWLEDGE_REVIEW_ON_START") == "1"
    assert os.environ.get("ENERGY_KNOWLEDGE_REVIEW_CMD") == "cmd /c echo b"
    assert os.environ.get("ENERGY_KNOWLEDGE_REVIEW_CWD") == str(tmp_path)


def test_menu_skip_review_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "mcp_review_profiles.json"
    cfg.write_text(
        json.dumps({"profiles": [{"label": "A", "cmd": "echo"}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ENERGY_MCP_MENU", "1")
    monkeypatch.setenv("ENERGY_KNOWLEDGE_REVIEW_ON_START", "1")
    monkeypatch.setenv("ENERGY_MCP_PROFILES_JSON", str(cfg))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdin.readline", lambda: "0\n")

    apply_mcp_profile_menu_if_requested(tmp_path)

    import os

    assert os.environ.get("ENERGY_KNOWLEDGE_REVIEW_ON_START") == "0"
    assert "ENERGY_KNOWLEDGE_REVIEW_CMD" not in os.environ


def test_menu_timeout_skips_review(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "mcp_review_profiles.json"
    cfg.write_text(
        json.dumps({"profiles": [{"label": "A", "cmd": "echo"}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ENERGY_MCP_MENU", "1")
    monkeypatch.setenv("ENERGY_MCP_PROFILES_JSON", str(cfg))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        "src.mcp_profile_menu._read_choice_line",
        lambda: "__MCP_MENU_TIMEOUT__",
    )

    apply_mcp_profile_menu_if_requested(tmp_path)

    import os

    assert os.environ.get("ENERGY_KNOWLEDGE_REVIEW_ON_START") == "0"
    assert "ENERGY_KNOWLEDGE_REVIEW_CMD" not in os.environ
