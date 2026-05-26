from __future__ import annotations


def trigger_dashboard_recompute(
    year_sel,
    meter_sel,
    building_sel,
    cold_start_days=None,
) -> None:
    """Centralized trigger helper used after campus/context switches."""
    year_sel.param.trigger("value")
    meter_sel.param.trigger("value")
    building_sel.param.trigger("value")
    if cold_start_days is not None:
        cold_start_days.param.trigger("value")
