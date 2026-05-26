# -*- coding: utf-8 -*-
from pathlib import Path
from unittest.mock import patch

import pytest


def test_skip_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKIP_ENERGY_STARTUP_HOOKS", "1")
    from src.knowledge_startup import run_startup_hooks

    r = run_startup_hooks(tmp_path)
    assert r.get("skipped") is True


def test_graphify_runs_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SKIP_ENERGY_STARTUP_HOOKS", raising=False)
    monkeypatch.setenv("ENERGY_GRAPHIFY_ON_START", "1")
    monkeypatch.setenv("ENERGY_KNOWLEDGE_REVIEW_ON_START", "0")
    monkeypatch.setenv("ENERGY_GRAPHIFY_CMD", "echo graphify_ok")
    monkeypatch.setenv("ENERGY_STARTUP_HOOK_ORDER", "graphify")

    from src import knowledge_startup as ks

    log_dir = tmp_path / "logs"
    monkeypatch.setenv("ENERGY_STARTUP_LOG_DIR", str(log_dir))

    with patch.object(ks, "_run_shell_cmd", wraps=ks._run_shell_cmd) as m:
        r = ks.run_startup_hooks(tmp_path)
        m.assert_called_once()
    assert r["steps"]["graphify"]["ok"] is True


def test_strict_raises_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SKIP_ENERGY_STARTUP_HOOKS", raising=False)
    monkeypatch.setenv("ENERGY_GRAPHIFY_ON_START", "1")
    monkeypatch.setenv("ENERGY_GRAPHIFY_CMD", "exit 1")
    monkeypatch.setenv("ENERGY_STARTUP_HOOK_ORDER", "graphify")
    monkeypatch.setenv("ENERGY_STARTUP_STRICT", "1")

    from src.knowledge_startup import run_startup_hooks

    with pytest.raises(RuntimeError, match="啟動前掛鉤失敗"):
        run_startup_hooks(tmp_path)
