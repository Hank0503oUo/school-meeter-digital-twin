from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from src.project_paths import iter_ancestor_dirs, project_root, resolve_project_path


class LocalMCPBridgeError(RuntimeError):
    """Raised when the local RTEM MCP project cannot be located or imported."""


def _iter_rtem_mcp_candidates() -> list[Path]:
    demo_root = project_root()
    candidates: list[Path] = []
    seen: set[Path] = set()

    for parent in iter_ancestor_dirs(demo_root):
        for relative in (
            Path("build llm") / "rtem-mcp-server",
            Path("rtem-mcp-server"),
            Path("external") / "rtem-mcp-server",
        ):
            candidate = (parent / relative).resolve()
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
    return candidates


def find_rtem_mcp_root() -> Path:
    override = os.environ.get("RTEM_MCP_SERVER_ROOT", "").strip()
    if override:
        path = resolve_project_path(override)
        if path.exists():
            return path
        raise LocalMCPBridgeError(f"RTEM MCP server override path does not exist: {path}")

    candidates = _iter_rtem_mcp_candidates()
    for candidate in candidates:
        if candidate.exists():
            return candidate

    preview = ", ".join(str(path) for path in candidates[:4])
    raise LocalMCPBridgeError(f"RTEM MCP server project not found. Checked: {preview}")


def ensure_rtem_mcp_import_path() -> Path:
    root = find_rtem_mcp_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


@lru_cache(maxsize=1)
def load_local_mcp_clients() -> dict[str, object]:
    root = ensure_rtem_mcp_import_path()
    try:
        from chronos_backend import ChronosBackend
        from pivd_backend import PIVDBackend
        from rtem_backend import RTEMBackend
    except Exception as exc:  # pragma: no cover - import failure depends on environment
        raise LocalMCPBridgeError(f"Failed to import local MCP backends: {exc}") from exc

    pivd = PIVDBackend()
    return {
        "root": root,
        "rtem": RTEMBackend(),
        "pivd": pivd,
        "chronos": ChronosBackend(pivd_backend=pivd),
    }


def local_mcp_available() -> bool:
    try:
        load_local_mcp_clients()
        return True
    except Exception:
        return False
