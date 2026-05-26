from __future__ import annotations

"""
Thin dashboard entrypoint.

The legacy dashboard implementation has been decomposed into runtime, map,
analysis, assistant, and factory modules under ``src.dashboard_modules``.
"""

from src.dashboard_modules.factory import create_dashboard


__all__ = ["create_dashboard"]
