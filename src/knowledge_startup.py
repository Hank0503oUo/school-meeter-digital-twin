# -*- coding: utf-8 -*-
"""
啟動前掛鉤：整份 graphify 更新 + 知識 review（Claude MCP / Cursor MCP 等由使用者自訂指令）。

由 open_browser_launcher 在「即將啟動新的 panel serve」時呼叫一次（重用既有服務時不會執行）。

環境變數（皆可選）：
  SKIP_ENERGY_STARTUP_HOOKS=1     略過所有掛鉤
  ENERGY_STARTUP_HOOK_ORDER       預設 graphify,review（逗號分隔：graphify | review）
  ENERGY_STARTUP_LOG_DIR          日誌目錄，預設 dev_artifacts/startup_logs
  ENERGY_STARTUP_STRICT=1         任一步驟 returncode!=0 時拋錯，阻止啟動 panel

整份更新：
  ENERGY_GRAPHIFY_ON_START=1
  ENERGY_GRAPHIFY_CMD             必填；例如呼叫你本機的 graphify 全量腳本或 npx / claude 指令
  ENERGY_GRAPHIFY_CWD             選填；預設 demo 根目錄

知識 review（建議在 graphify 之後，由 ENERGY_STARTUP_HOOK_ORDER 控制）：
  ENERGY_KNOWLEDGE_REVIEW_ON_START=1
  ENERGY_KNOWLEDGE_REVIEW_CMD     必填；連線 MCP 或執行 review 的包裝指令
  ENERGY_KNOWLEDGE_REVIEW_CWD     選填；預設 demo 根目錄
  ENERGY_MCP_MENU=1               啟動前顯示選單（config/mcp_review_profiles.json），覆寫上述 review 相關變數
  ENERGY_MCP_PROFILES_JSON        選單用設定檔路徑，預設 config/mcp_review_profiles.json

說明：graphify 全量管線多半需由 Agent／技能驅動；此處不內建完整 graphify，只執行你提供的 CMD。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _env_truthy(name: str) -> bool:
    val = os.environ.get(name, "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _run_shell_cmd(
    cmd: str,
    *,
    cwd: str,
    log_path: Path,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"cwd={cwd}\n")
        logf.write(f"cmd={cmd}\n\n")
        logf.flush()
        p = subprocess.run(
            cmd,
            cwd=cwd,
            shell=True,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return {
        "ok": p.returncode == 0,
        "returncode": p.returncode,
        "log": str(log_path),
    }


def run_startup_hooks(root: Path) -> dict[str, Any]:
    """依 ENERGY_STARTUP_HOOK_ORDER 執行掛鉤；回傳每步狀態供 launcher 列印。"""
    if _env_truthy("SKIP_ENERGY_STARTUP_HOOKS"):
        return {"skipped": True, "reason": "SKIP_ENERGY_STARTUP_HOOKS"}

    log_dir = root / os.environ.get("ENERGY_STARTUP_LOG_DIR", "dev_artifacts/startup_logs")
    ts = time.strftime("%Y%m%d_%H%M%S")
    order_raw = os.environ.get("ENERGY_STARTUP_HOOK_ORDER", "graphify,review").strip().lower()
    steps = [s.strip() for s in order_raw.split(",") if s.strip()]

    results: dict[str, Any] = {"order": steps, "steps": {}, "ran_any": False}
    failures: list[str] = []

    for step in steps:
        if step == "graphify":
            if not _env_truthy("ENERGY_GRAPHIFY_ON_START"):
                continue
            results["ran_any"] = True
            cmd = os.environ.get("ENERGY_GRAPHIFY_CMD", "").strip()
            cwd = os.environ.get("ENERGY_GRAPHIFY_CWD") or str(root)
            if not cmd:
                results["steps"]["graphify"] = {
                    "ok": False,
                    "error": "ENERGY_GRAPHIFY_ON_START 已開啟但未設定 ENERGY_GRAPHIFY_CMD",
                }
                failures.append("graphify")
                continue
            log_path = log_dir / f"graphify_{ts}.log"
            out = _run_shell_cmd(cmd, cwd=cwd, log_path=log_path)
            results["steps"]["graphify"] = out
            if not out["ok"]:
                failures.append("graphify")
        elif step == "review":
            if not _env_truthy("ENERGY_KNOWLEDGE_REVIEW_ON_START"):
                continue
            results["ran_any"] = True
            cmd = os.environ.get("ENERGY_KNOWLEDGE_REVIEW_CMD", "").strip()
            cwd = os.environ.get("ENERGY_KNOWLEDGE_REVIEW_CWD") or str(root)
            if not cmd:
                results["steps"]["review"] = {
                    "ok": False,
                    "error": "ENERGY_KNOWLEDGE_REVIEW_ON_START 已開啟但未設定 ENERGY_KNOWLEDGE_REVIEW_CMD",
                }
                failures.append("review")
                continue
            log_path = log_dir / f"knowledge_review_{ts}.log"
            out = _run_shell_cmd(cmd, cwd=cwd, log_path=log_path)
            results["steps"]["review"] = out
            if not out["ok"]:
                failures.append("review")
        else:
            results["steps"][step] = {"ok": False, "error": f"unknown step: {step}"}
            failures.append(step)

    if _env_truthy("ENERGY_STARTUP_STRICT") and failures:
        msg = (
            "啟動前掛鉤失敗 (ENERGY_STARTUP_STRICT=1): "
            + ", ".join(failures)
            + "。請查看 "
            + str(log_dir)
        )
        raise RuntimeError(msg)

    return results


def print_hook_summary(results: dict[str, Any]) -> None:
    """將結果寫入 stdout，供 launcher_log.txt 留存。"""
    if results.get("skipped"):
        print(f"[startup hooks] skipped: {results.get('reason', '')}")
        return
    if not results.get("ran_any"):
        print(
            "[startup hooks] no hooks enabled "
            "(set ENERGY_GRAPHIFY_ON_START / ENERGY_KNOWLEDGE_REVIEW_ON_START)"
        )
        return
    print("[startup hooks] order:", ",".join(results.get("order", [])))
    for name, data in results.get("steps", {}).items():
        if "error" in data:
            print(f"[startup hooks] {name}: ERROR {data['error']}")
        else:
            status = "ok" if data.get("ok") else f"exit {data.get('returncode')}"
            print(f"[startup hooks] {name}: {status} log={data.get('log')}")
