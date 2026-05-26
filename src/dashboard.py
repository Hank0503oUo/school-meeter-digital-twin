from __future__ import annotations

"""
Dashboard entrypoint.

The original campus twin dashboard is the default experience for this project.
The knowledge workbench remains available as a separate entrypoint.
"""

def create_dashboard():
    """Build and return the original campus twin dashboard."""
    from src.dashboard_impl import create_dashboard as _create_legacy_dashboard_impl

    return _create_legacy_dashboard_impl()


def create_knowledge_workbench():
    """Build and return the Nekaise-style knowledge workbench."""
    from src.nekaise_dashboard import create_dashboard as _create_workbench_dashboard

    return _create_workbench_dashboard()


def create_legacy_knowledge_workbench():
    """Build and return the original form-style knowledge workbench."""
    from src.knowledge_dashboard import create_dashboard as _create_legacy_workbench_dashboard

    return _create_legacy_workbench_dashboard()


def _serve_default() -> None:
    dashboard = create_dashboard()
    dashboard.servable()


if __name__.startswith("bokeh_app") or __name__ == "__main__":
    _serve_default()
