# -*- coding: utf-8 -*-
"""
啟動前互動選單：從 config/mcp_review_profiles.json 選擇要執行的知識 review（MCP）指令。

環境變數：
  ENERGY_MCP_MENU=1              啟用選單（與 open_browser_launcher 搭配）
  ENERGY_MCP_PROFILES_JSON       設定檔路徑，預設 demo 根目錄下 config/mcp_review_profiles.json
  ENERGY_MCP_MENU_TIMEOUT_SECONDS  等待輸入秒數，預設 120；逾時自動略過知識 review（避免卡死）。設 0 為無限等待

選擇會寫入目前 process 的：
  ENERGY_KNOWLEDGE_REVIEW_ON_START
  ENERGY_KNOWLEDGE_REVIEW_CMD
  ENERGY_KNOWLEDGE_REVIEW_CWD

輸出優先使用 Windows CONOUT$，以便 open_demo.cmd 將 stdout 導向 launcher_log 仍可顯示在視窗。
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

_TIMEOUT_SENTINEL = "__MCP_MENU_TIMEOUT__"


def _menu_print(msg: str) -> None:
    """在 stdout/stderr 被重新導向檔案時，仍寫入實體主控台（Windows CONOUT$）。"""
    if sys.platform == "win32":
        try:
            with open("CONOUT$", "w", encoding="utf-8", errors="replace") as con:
                con.write(msg + "\n")
            return
        except OSError:
            pass
    print(msg, file=sys.stderr)


def _env_truthy(name: str) -> bool:
    val = os.environ.get(name, "").strip().lower()
    return val in ("1", "true", "yes", "on")


def default_profiles_path(root: Path) -> Path:
    return root / "config" / "mcp_review_profiles.json"


def load_profiles(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    profiles = raw.get("profiles")
    if not isinstance(profiles, list):
        return []
    out: list[dict[str, Any]] = []
    for i, p in enumerate(profiles):
        if not isinstance(p, dict):
            continue
        cmd = str(p.get("cmd", "")).strip()
        if not cmd:
            continue
        label = str(p.get("label", p.get("id", f"profile_{i + 1}"))).strip() or f"Profile {i + 1}"
        cwd = str(p.get("cwd", "")).strip()
        out.append({"label": label, "cmd": cmd, "cwd": cwd})
    return out


def apply_mcp_profile_menu_if_requested(root: Path) -> dict[str, Any]:
    """
    若 ENERGY_MCP_MENU=1 且設定檔存在、stdin 為 TTY，則顯示選單並更新 os.environ。
    回傳摘要供 launcher 列印。
    """
    if not _env_truthy("ENERGY_MCP_MENU"):
        return {"applied": False, "reason": "ENERGY_MCP_MENU is not set"}

    cfg = os.environ.get("ENERGY_MCP_PROFILES_JSON", "").strip()
    path = Path(cfg) if cfg else default_profiles_path(root)
    if not path.is_file():
        _menu_print(f"[mcp menu] skip: profiles file not found: {path}")
        return {"applied": False, "reason": "profiles file missing", "path": str(path)}

    profiles = load_profiles(path)
    if not profiles:
        _menu_print(f"[mcp menu] skip: no valid profiles in {path}")
        return {"applied": False, "reason": "empty profiles", "path": str(path)}

    if not sys.stdin.isatty():
        _menu_print(
            "[mcp menu] skip: stdin is not a TTY (non-interactive); "
            "use ENERGY_KNOWLEDGE_REVIEW_CMD or run from console."
        )
        return {"applied": False, "reason": "not a tty"}

    _interactive_select(profiles, path)
    return {
        "applied": True,
        "path": str(path),
        "on_start": os.environ.get("ENERGY_KNOWLEDGE_REVIEW_ON_START", ""),
        "has_cmd": bool(os.environ.get("ENERGY_KNOWLEDGE_REVIEW_CMD", "").strip()),
    }


def _mcp_menu_timeout_seconds() -> float:
    raw = os.environ.get("ENERGY_MCP_MENU_TIMEOUT_SECONDS", "120").strip()
    try:
        return float(raw)
    except ValueError:
        return 120.0


def _read_choice_line() -> str:
    """讀取一行選項；ENERGY_MCP_MENU_TIMEOUT_SECONDS > 0 時逾時回傳哨兵值，避免無限等待。"""
    timeout_sec = _mcp_menu_timeout_seconds()
    if timeout_sec <= 0:
        try:
            return sys.stdin.readline()
        except Exception:
            return ""

    holder: list[str] = []

    def _target() -> None:
        try:
            holder.append(sys.stdin.readline())
        except Exception:
            holder.append("")

    th = threading.Thread(target=_target, daemon=True)
    th.start()
    th.join(timeout=timeout_sec)
    if th.is_alive():
        return _TIMEOUT_SENTINEL
    return holder[0] if holder else ""


def _interactive_select(profiles: list[dict[str, Any]], path: Path) -> None:
    _menu_print("")
    _menu_print("  === MCP 知識 review：要連線／執行的設定 ===")
    _menu_print(f"  (設定檔: {path})")
    _menu_print("")
    _menu_print("   0  本次略過 review（不執行知識 review）")
    for i, p in enumerate(profiles, start=1):
        _menu_print(f"  {i:2d}  {p['label']}")
    _menu_print("")

    n = len(profiles)
    timeout_disp = _mcp_menu_timeout_seconds()
    while True:
        _menu_print("  請輸入編號並按 Enter: ")
        raw = _read_choice_line().strip()
        if raw == _TIMEOUT_SENTINEL:
            _menu_print(
                f"  （逾時 {timeout_disp:g} 秒未輸入）已自動略過知識 review，將繼續啟動 Panel。"
            )
            os.environ["ENERGY_KNOWLEDGE_REVIEW_ON_START"] = "0"
            os.environ.pop("ENERGY_KNOWLEDGE_REVIEW_CMD", None)
            os.environ.pop("ENERGY_KNOWLEDGE_REVIEW_CWD", None)
            return
        if raw == "":
            continue
        if not raw.isdigit():
            _menu_print("  請輸入數字。")
            continue
        choice = int(raw)
        if choice == 0:
            os.environ["ENERGY_KNOWLEDGE_REVIEW_ON_START"] = "0"
            os.environ.pop("ENERGY_KNOWLEDGE_REVIEW_CMD", None)
            os.environ.pop("ENERGY_KNOWLEDGE_REVIEW_CWD", None)
            _menu_print("  → 已選：略過 knowledge review。")
            return
        if 1 <= choice <= n:
            p = profiles[choice - 1]
            os.environ["ENERGY_KNOWLEDGE_REVIEW_ON_START"] = "1"
            os.environ["ENERGY_KNOWLEDGE_REVIEW_CMD"] = p["cmd"]
            if p.get("cwd"):
                os.environ["ENERGY_KNOWLEDGE_REVIEW_CWD"] = p["cwd"]
            else:
                os.environ.pop("ENERGY_KNOWLEDGE_REVIEW_CWD", None)
            _menu_print(f"  → 已選：{p['label']}")
            return
        _menu_print(f"  請輸入 0～{n}。")
