from __future__ import annotations

import asyncio
from pathlib import Path

from src import demo_assistant
from src import demo_mcp_server
from src import lm_studio_client


def test_prefetch_search_query_extracts_known_legal_phrase():
    assert lm_studio_client._prefetch_search_query("請查詢 hjplus-kb 裡的排煙窗法規") == "排煙窗法規"


def test_should_prefetch_docs_for_hjplus_and_legal_queries():
    assert lm_studio_client._should_prefetch_docs("請查 HJPLUS 的資料") is True
    assert lm_studio_client._should_prefetch_docs("建築法規怎麼看") is True
    assert lm_studio_client._should_prefetch_docs("今年最高耗電建築是哪一棟") is False


def test_should_prefetch_energy_records_for_cross_year_queries():
    assert lm_studio_client._should_prefetch_energy_records("2014 到 2020 哪些建築平均用電最高") is True
    assert lm_studio_client._infer_energy_tool("2014 到 2020 哪些建築平均用電最高") == "rank_energy_buildings_across_years"
    assert lm_studio_client._infer_energy_metric("請比較 EUI 趨勢") == "eui"
    assert lm_studio_client._extract_years("比較 2014, 2017, 2020") == [2014, 2017, 2020]
    assert lm_studio_client._extract_years("比較 2014 到 2020") == [2014, 2015, 2016, 2017, 2018, 2019, 2020]


def test_harness_memory_tools_hidden_by_default(monkeypatch):
    monkeypatch.delenv("ENERGY_HARNESS_MEMORY_TOOLS_ENABLED", raising=False)

    assert lm_studio_client._local_llm_tool_enabled("search_harness_memory") is False
    assert lm_studio_client._local_llm_tool_enabled("store_harness_memory") is False
    assert lm_studio_client._local_llm_tool_enabled("search_docs") is True


def test_harness_memory_tools_can_be_enabled(monkeypatch):
    monkeypatch.setenv("ENERGY_HARNESS_MEMORY_TOOLS_ENABLED", "1")

    assert lm_studio_client._local_llm_tool_enabled("search_harness_memory") is True


def test_demo_mcp_server_exposes_knowledge_tools():
    server = demo_mcp_server.build_server()
    tools = asyncio.run(server.list_tools())
    names = {tool.name for tool in tools}

    assert {"search_docs", "fetch_chunk", "lookup_building_entity"}.issubset(names)


def test_local_memory_disabled_env_takes_precedence(monkeypatch):
    monkeypatch.setenv("ENERGY_LOCAL_MEMORY_DISABLED", "1")
    monkeypatch.setenv("ENERGY_LM_STUDIO_MEMORY_DISABLED", "0")
    monkeypatch.setenv("ENERGY_WIKI_DISABLED", "0")

    assert lm_studio_client._memory_disabled() is True


def test_lm_studio_memory_disabled_env_remains_compatible(monkeypatch):
    monkeypatch.delenv("ENERGY_LOCAL_MEMORY_DISABLED", raising=False)
    monkeypatch.setenv("ENERGY_LM_STUDIO_MEMORY_DISABLED", "true")
    monkeypatch.setenv("ENERGY_WIKI_DISABLED", "0")

    assert lm_studio_client._memory_disabled() is True


def test_harness_memory_can_be_disabled(monkeypatch):
    monkeypatch.setenv("ENERGY_HARNESS_MEMORY_ENABLED", "0")

    assert demo_assistant.resolve_harness_memory_config() is None


def test_harness_memory_config_uses_env_overrides(monkeypatch, tmp_path):
    root = tmp_path / "memory_mcp"
    python_exe = root / ".venv" / "Scripts" / "python.exe"
    server = root / "server.py"
    config = root / "mcp.config.json"
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("", encoding="utf-8")
    server.write_text("", encoding="utf-8")
    config.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("ENERGY_HARNESS_MEMORY_ENABLED", "1")
    monkeypatch.setenv("ENERGY_HARNESS_MCP_ROOT", str(root))

    resolved = demo_assistant.resolve_harness_memory_config()

    assert resolved == {
        "python": str(python_exe),
        "server": str(server),
        "cwd": str(root),
        "config": str(config),
    }


def test_harness_memory_config_skips_missing_files(monkeypatch, tmp_path):
    monkeypatch.setenv("ENERGY_HARNESS_MEMORY_ENABLED", "1")
    monkeypatch.setenv("ENERGY_HARNESS_MCP_ROOT", str(Path(tmp_path) / "missing"))

    assert demo_assistant.resolve_harness_memory_config() is None
