# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
import panel as pn

from src.dashboard_modules.models import DashboardWidgets


def iter_panel_objects(root):
    if root is None:
        return
    if isinstance(root, (list, tuple)):
        for item in root:
            yield from iter_panel_objects(item)
        return
    yield root
    for attr in ("objects", "sidebar", "main"):
        value = getattr(root, attr, None)
        if value is not None:
            yield from iter_panel_objects(value)


def find_first(root, component_type):
    for item in iter_panel_objects(root):
        if isinstance(item, component_type):
            return item
    return None


def build_dashboard_test_widgets() -> DashboardWidgets:
    return DashboardWidgets(
        campus_sel=pn.widgets.Select(name="Campus", options={"NTU": "ntu"}, value="ntu"),
        main_spinner=pn.indicators.LoadingSpinner(value=False, size=20, name="Loading"),
        year_sel=pn.widgets.DiscreteSlider(name="Year", options=[2020], value=2020),
        building_sel=pn.widgets.Select(name="Building", options={"[All] Campus (ALL)": "ALL"}, value="ALL"),
        bldg_filter=pn.widgets.MultiSelect(name="Archetype filter", options=["All"], value=["All"]),
        meter_sel=pn.widgets.Select(name="Meter", options=["All"], value="All"),
        time_scale_sel=pn.widgets.Select(name="Time scale", options=["Daily"], value="Daily"),
        cooling_sl=pn.widgets.FloatSlider(name="Cooling", start=-1.0, end=1.0, value=0.0),
        lighting_sl=pn.widgets.FloatSlider(name="Lighting", start=0.0, end=2.0, value=1.0),
        occupancy_sl=pn.widgets.FloatSlider(name="Occupancy", start=0.0, end=2.0, value=1.0),
        equipment_sl=pn.widgets.FloatSlider(name="Equipment", start=0.0, end=2.0, value=1.0),
        assistant_task_sel=pn.widgets.Select(name="Assistant task", options={"Q&A": "qa"}, value="qa"),
        assistant_quick_sel=pn.widgets.Select(name="Quick prompts", options={"None": ""}, value=""),
        assistant_query=pn.widgets.TextAreaInput(name="Ask MCP assistant"),
        assistant_spinner=pn.indicators.LoadingSpinner(value=False, size=16, name="AI"),
        assistant_run_btn=pn.widgets.Button(name="Run"),
        assistant_save_btn=pn.widgets.Button(name="Save"),
        assistant_force_mcp=pn.widgets.Checkbox(name="Force local MCP", value=False),
        assistant_save_memory=pn.widgets.Checkbox(name="Save memory", value=False),
        assistant_status=pn.pane.Markdown(""),
        assistant_chat_log=pn.Column(),
        assistant_structured=pn.pane.JSON({}),
        assistant_citations=pn.widgets.Tabulator(pd.DataFrame(), theme="site"),
        color_mode=pn.widgets.RadioButtonGroup(name="Map color mode", options={"Tier": "tier"}, value="tier"),
        cold_start_days=pn.widgets.IntSlider(name="Cold start days", start=0, end=30, value=30),
        map_display_toggle=pn.widgets.RadioButtonGroup(name="Map content", options={"All": "all"}, value="all"),
        campus_status_indicator=pn.pane.Markdown(""),
        engine_mode_indicator=pn.pane.Markdown(""),
        cloud_local_toggle=pn.widgets.RadioButtonGroup(
            name="LLM mode",
            options={
                "NVIDIA API (線上)": "nvidia",
                "Yunxin API (線上)": "yunxin",
                "LM Studio (本地)": "local",
                "Cloud (Gemini)": "cloud",
            },
            value="cloud",
        ),
    )
