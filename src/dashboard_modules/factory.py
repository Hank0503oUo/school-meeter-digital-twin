from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import pandas as pd
import panel as pn

from src.constants import COLOR_MODE_OPTIONS
from src.dashboard_modules.analysis_views import AnalysisViewController
from src.dashboard_modules.assistant_views import AssistantController
from src.dashboard_modules.map_views import MapViewController
from src.dashboard_modules.models import DashboardWidgets
from src.dashboard_modules.reactive import trigger_dashboard_recompute
from src.dashboard_modules.runtime import DashboardRuntime
from src.dashboard_noncore import (
    build_engine_mode_markdown,
    build_legend_markdown,
    build_paper_ref_markdown,
)


log = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CUSTOM_CSS_PATH = _PROJECT_ROOT / "assets" / "custom.css"


def _ensure_custom_css_loaded(path: Path = _CUSTOM_CSS_PATH) -> None:
    try:
        css_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("Load custom CSS failed: %s", exc)
        return

    if css_text not in pn.config.raw_css:
        pn.config.raw_css.append(css_text)


def _section_heading(kicker: str, title: str, description: str) -> str:
    return (
        '<div class="sidebar-section-heading">'
        f'<div class="sidebar-section-kicker">{kicker}</div>'
        f"<h3>{title}</h3>"
        f"<p>{description}</p>"
        "</div>"
    )


def _dashboard_title(runtime: DashboardRuntime) -> str:
    return f"{runtime.active_campus_name} Energy Digital Twin"


def _set_campus_controls_disabled(widgets: DashboardWidgets, disabled: bool) -> None:
    for control in (
        widgets.year_sel,
        widgets.building_sel,
        widgets.bldg_filter,
        widgets.meter_sel,
    ):
        control.disabled = disabled


def _run_on_next_tick(callback: Callable[[], None]) -> None:
    document = pn.state.curdoc
    if document is not None:
        document.add_next_tick_callback(callback)
        return
    callback()


def create_dashboard() -> pn.template.FastListTemplate:
    _ensure_custom_css_loaded()
    try:
        pn.extension("plotly", "tabulator", "deckgl", sizing_mode="stretch_width")
    except (ImportError, ModuleNotFoundError):
        pass

    runtime = DashboardRuntime()
    default_campus_id, campus_options = runtime.campus_options()

    # DEMO: 暫時屏蔽 NTU,只留中央(NCU)。
    # 改回多校切換時,刪除此區塊即可恢復 runtime.campus_options() 原行為。
    _ncu_only = {label: cid for label, cid in campus_options.items() if cid == "ncu"}
    if _ncu_only:
        campus_options = _ncu_only
        default_campus_id = "ncu"

    runtime.prepare_campus_shell(default_campus_id)

    campus_sel = pn.widgets.Select(name="Campus", options=campus_options, value=default_campus_id)
    if len(campus_options) <= 1:
        campus_sel.disabled = True
    main_spinner = pn.indicators.LoadingSpinner(value=True, size=30, color="primary", name="Loading")
    # NCU demo: only years with cleaned NCU data (民國 109/110/111/114 = 2020/21/22/25)
    # plus 2024 as PIVD-only estimate (no NCU actuals for 民國 113).
    year_sel = pn.widgets.DiscreteSlider(name="Year", options=[2020, 2021, 2022, 2024, 2025], value=2025)
    building_sel = pn.widgets.Select(name="Building", options={"[All] Campus (ALL)": "ALL"}, value="ALL")
    bldg_filter = pn.widgets.MultiSelect(name="Archetype filter", options=["All"], value=["All"], size=5)
    meter_sel = pn.widgets.Select(name="Meter", options=["All"], value="All")
    time_scale_sel = pn.widgets.Select(
        name="Time scale",
        options=[
            "Hourly (3h)",
            "Daily",
            "Weekly",
            "Monthly",
            "Quarterly",
            "Yearly",
        ],
        value="Daily",
    )
    cooling_sl = pn.widgets.FloatSlider(name="Cooling delta (°C)", start=-5.0, end=5.0, step=0.1, value=0.0)
    lighting_sl = pn.widgets.FloatSlider(name="Lighting ratio", start=0.0, end=5.0, step=0.1, value=1.0)
    occupancy_sl = pn.widgets.FloatSlider(name="Occupancy ratio", start=0.0, end=5.0, step=0.1, value=1.0)
    equipment_sl = pn.widgets.FloatSlider(name="Equipment ratio", start=0.0, end=5.0, step=0.1, value=1.0)
    assistant_task_sel = pn.widgets.Select(
        name="Assistant task",
        options={
            "Q&A": "qa",
            "Energy summary": "energy_summary",
            "Report generation": "report_generation",
        },
        value="energy_summary",
    )
    assistant_quick_sel = pn.widgets.Select(
        name="Quick prompts",
        options={
            "None": "",
            "Summarize the selected building and highlight the main load drivers.": "Summarize the selected building and highlight the main load drivers.",
            "Compare this building against campus-level benchmarks and explain the gap.": "Compare this building against campus-level benchmarks and explain the gap.",
            "Draft a short retrofit memo with recommended next steps.": "Draft a short retrofit memo with recommended next steps.",
        },
        value="",
    )
    assistant_query = pn.widgets.TextAreaInput(name="Ask MCP assistant", height=120)
    assistant_image_upload = pn.widgets.FileInput(
        name="Image",
        accept=".png,.jpg,.jpeg,.webp,.bmp,.gif",
        multiple=False,
    )
    assistant_spinner = pn.indicators.LoadingSpinner(value=False, size=18, color="primary", name="AI")
    assistant_run_btn = pn.widgets.Button(name="Run", button_type="primary")
    assistant_save_btn = pn.widgets.Button(name="Save result", button_type="success")
    assistant_force_mcp = pn.widgets.Checkbox(name="Force local MCP", value=False)
    assistant_save_memory = pn.widgets.Checkbox(name="Save reviewed result", value=False)
    assistant_status = pn.pane.Markdown("", sizing_mode="stretch_width")
    assistant_chat_log = pn.Column(
        pn.pane.HTML("<div style='color:#64748b; font-style:italic;'>Assistant messages will appear here.</div>"),
        sizing_mode="stretch_both",
    )
    assistant_structured = pn.pane.JSON({}, depth=3, sizing_mode="stretch_width")
    assistant_citations = pn.widgets.Tabulator(pd.DataFrame(), theme="site", sizing_mode="stretch_both", min_height=220)
    color_mode = pn.widgets.RadioButtonGroup(
        name="Map color mode",
        options=COLOR_MODE_OPTIONS,
        value=runtime.load_default_color_mode(),
        button_type="primary",
        button_style="outline",
    )
    cold_start_days = pn.widgets.IntSlider(name="Cold start days", start=0, end=30, value=30, step=1)
    map_display_toggle = pn.widgets.RadioButtonGroup(
        name="Map content",
        options={"Measured + inferred": "all", "Measured only": "metered_only"},
        value="all",
        button_type="success",
        button_style="outline",
    )
    campus_status_indicator = pn.pane.Markdown(runtime.campus_status_markdown(), sizing_mode="stretch_width")
    engine_mode_indicator = pn.pane.Markdown(
        build_engine_mode_markdown(runtime.engine_mode),
        sizing_mode="stretch_width",
    )
    cloud_local_toggle = pn.widgets.RadioButtonGroup(
        name="LLM mode",
        options={
            "NVIDIA API (線上)": "nvidia",
            "Yunxin API (線上)": "yunxin",
            "Command Code API (線上)": "commandcode",
            "Local Gemma/LLM (本地)": "local",
            "Cloud (Gemini)": "cloud",
        },
        value="local",
        button_type="primary",
        button_style="outline",
    )

    widgets = DashboardWidgets(
        campus_sel=campus_sel,
        main_spinner=main_spinner,
        year_sel=year_sel,
        building_sel=building_sel,
        bldg_filter=bldg_filter,
        meter_sel=meter_sel,
        time_scale_sel=time_scale_sel,
        cooling_sl=cooling_sl,
        lighting_sl=lighting_sl,
        occupancy_sl=occupancy_sl,
        equipment_sl=equipment_sl,
        assistant_task_sel=assistant_task_sel,
        assistant_quick_sel=assistant_quick_sel,
        assistant_query=assistant_query,
        assistant_image_upload=assistant_image_upload,
        assistant_spinner=assistant_spinner,
        assistant_run_btn=assistant_run_btn,
        assistant_save_btn=assistant_save_btn,
        assistant_force_mcp=assistant_force_mcp,
        assistant_save_memory=assistant_save_memory,
        assistant_status=assistant_status,
        assistant_chat_log=assistant_chat_log,
        assistant_structured=assistant_structured,
        assistant_citations=assistant_citations,
        color_mode=color_mode,
        cold_start_days=cold_start_days,
        map_display_toggle=map_display_toggle,
        campus_status_indicator=campus_status_indicator,
        engine_mode_indicator=engine_mode_indicator,
        cloud_local_toggle=cloud_local_toggle,
    )
    _set_campus_controls_disabled(widgets, True)

    analysis_controller = AnalysisViewController(runtime, widgets)
    map_controller = MapViewController(runtime, widgets, analysis_controller.deployment_confidence)
    assistant_controller = AssistantController(runtime, widgets)
    map_controller.refresh_campus_controls()

    color_mode.param.watch(lambda event: runtime.save_color_mode(str(event.new)), "value")
    year_sel.param.watch(map_controller.refresh_building_options, "value")
    bldg_filter.param.watch(map_controller.refresh_building_options, "value")
    assistant_controller.bind_events()
    assistant_controller.sync_nekaise_context()

    def reset_sliders(event=None) -> None:
        cooling_sl.value = 0.0
        lighting_sl.value = 1.0
        occupancy_sl.value = 1.0
        equipment_sl.value = 1.0

    reset_btn = pn.widgets.Button(name="Reset scenario", button_type="light", sizing_mode="stretch_width")
    reset_btn.on_click(reset_sliders)

    kpi_panel = pn.bind(analysis_controller.kpi_panel, year_sel, building_sel, cooling_sl, lighting_sl, occupancy_sl, equipment_sl)
    map_panel = pn.bind(map_controller.map_panel, year_sel, color_mode, meter_sel, map_display_toggle, cold_start_days, building_sel)
    meter_info_panel = pn.bind(map_controller.meter_info_panel, meter_sel)
    building_quick_view = pn.bind(map_controller.building_quick_view, year_sel, building_sel)
    building_source_notice = pn.bind(map_controller.building_source_notice, year_sel, building_sel)
    timeseries_chart = pn.bind(
        analysis_controller.timeseries_chart,
        year_sel,
        building_sel,
        time_scale_sel,
        cooling_sl,
        lighting_sl,
        occupancy_sl,
        equipment_sl,
    )
    deployment_dci_panel = pn.bind(analysis_controller.deployment_dci_panel, cold_start_days, meter_sel)
    cold_start_kpi = pn.bind(analysis_controller.cold_start_kpi, cold_start_days, meter_sel)
    ranking_chart = pn.bind(analysis_controller.ranking_chart, year_sel)
    eui_chart = pn.bind(analysis_controller.eui_chart, year_sel)
    waste_chart = pn.bind(analysis_controller.waste_chart, year_sel)
    r2_scatter = pn.bind(analysis_controller.r2_scatter, year_sel)
    stats_table = pn.bind(analysis_controller.stats_table, year_sel)
    building_cf_table = pn.bind(
        analysis_controller.building_cf_table,
        year_sel,
        cooling_sl,
        lighting_sl,
        occupancy_sl,
        equipment_sl,
    )
    compact_status_bar = pn.bind(analysis_controller.compact_status_bar, year_sel)

    legend_md = pn.pane.Markdown(build_legend_markdown(), sizing_mode="stretch_width")
    paper_ref_md = pn.pane.Markdown(build_paper_ref_markdown(), sizing_mode="stretch_width")

    sidebar_intro = pn.pane.HTML(
        """
<div class="sidebar-intro">
  <div class="sidebar-intro-kicker">Energy Digital Twin</div>
  <h2>Control Room</h2>
  <p>Choose a campus, focus on a building, and compare counterfactual scenarios without leaving the dashboard.</p>
</div>
""",
        sizing_mode="stretch_width",
    )

    session_section = pn.Column(
        pn.Row(main_spinner, align="center"),
        campus_sel,
        campus_status_indicator,
        engine_mode_indicator,
        css_classes=["sidebar-section"],
        sizing_mode="stretch_width",
    )
    focus_section = pn.Column(
        pn.pane.HTML(_section_heading("Focus", "Selection", "Control the current campus scope."), sizing_mode="stretch_width"),
        year_sel,
        building_sel,
        building_quick_view,
        building_source_notice,
        css_classes=["sidebar-section"],
        sizing_mode="stretch_width",
    )
    scenario_card = pn.Card(
        kpi_panel,
        pn.layout.Divider(),
        cooling_sl,
        lighting_sl,
        occupancy_sl,
        equipment_sl,
        reset_btn,
        title="Scenario",
        collapsed=True,
        css_classes=["sidebar-section", "sidebar-section-strong"],
        sizing_mode="stretch_width",
    )
    advanced_card = pn.Card(
        bldg_filter,
        meter_sel,
        meter_info_panel,
        time_scale_sel,
        pn.layout.Divider(),
        deployment_dci_panel,
        cold_start_days,
        cold_start_kpi,
        pn.layout.Divider(),
        legend_md,
        paper_ref_md,
        title="Advanced",
        collapsed=True,
        css_classes=["sidebar-section"],
        sizing_mode="stretch_width",
    )

    tab_map = pn.Column(
        pn.Row(
            pn.pane.Markdown("#### Campus Map", align="center"),
            pn.layout.HSpacer(),
            color_mode,
            map_display_toggle,
            css_classes=["toolbar-bar"],
            sizing_mode="stretch_width",
        ),
        pn.Column(map_panel, sizing_mode="stretch_both"),
        pn.Row(timeseries_chart, sizing_mode="stretch_width", min_height=200, height=220),
        sizing_mode="stretch_both",
    )
    tab_analysis = pn.Column(
        pn.Row(
            pn.Column("### Waste-driven ranking", waste_chart, sizing_mode="stretch_both"),
            pn.Column("### Mean demand ranking", ranking_chart, sizing_mode="stretch_both"),
            sizing_mode="stretch_both",
        ),
        pn.layout.Divider(),
        pn.Row(
            pn.Column("### EUI ranking", eui_chart, sizing_mode="stretch_both"),
            pn.Column("### R² scatter", r2_scatter, sizing_mode="stretch_both"),
            sizing_mode="stretch_both",
        ),
        sizing_mode="stretch_both",
    )
    standalone_chat_tab = assistant_controller.build_chat_tab()
    main_tabs = pn.Tabs(
        ("Map", tab_map),
        ("Analysis", tab_analysis),
        ("AI Console", standalone_chat_tab),
        ("Tables", pn.Column(stats_table, pn.layout.Divider(), building_cf_table, sizing_mode="stretch_both")),
        active=0,
        dynamic=True,
        css_classes=["dashboard-tabs"],
        sizing_mode="stretch_both",
    )

    template = pn.template.FastListTemplate(
        title=_dashboard_title(runtime),
        accent_base_color="#1a73e8",
        header_background="#ffffff",
        header_color="#202124",
        sidebar=[sidebar_intro, session_section, focus_section, scenario_card, advanced_card],
        main=[
            compact_status_bar,
            main_tabs,
        ],
        sidebar_width=320,
    )

    def refresh_dashboard_shell() -> None:
        map_controller.refresh_campus_controls()
        widgets.engine_mode_indicator.object = build_engine_mode_markdown(runtime.engine_mode)
        widgets.campus_status_indicator.object = runtime.campus_status_markdown()
        template.title = _dashboard_title(runtime)
        assistant_controller.sync_nekaise_context()
        trigger_dashboard_recompute(year_sel, meter_sel, building_sel, cold_start_days)

    def load_selected_campus(campus_id: str) -> None:
        widgets.main_spinner.value = True
        _set_campus_controls_disabled(widgets, True)
        try:
            runtime.load_campus(str(campus_id).strip().lower())
            refresh_dashboard_shell()
        except Exception as exc:
            log.exception("Campus load failed for %s", campus_id)
            widgets.campus_status_indicator.object = (
                f"### Campus: {str(campus_id).strip().upper()}\n"
                f"- data readiness: load failed\n"
                f"- error: `{exc}`"
            )
            widgets.engine_mode_indicator.object = build_engine_mode_markdown("Load failed")
        finally:
            _set_campus_controls_disabled(widgets, not runtime.campus_loaded)
            widgets.main_spinner.value = False

    def on_campus_change(event) -> None:
        target_campus_id = str(event.new).strip().lower()
        runtime.campus_loaded = False
        runtime.campus_loading = True
        runtime.loaded_campus_id = None
        runtime.prepare_campus_shell(target_campus_id)
        widgets.main_spinner.value = True
        _set_campus_controls_disabled(widgets, True)
        widgets.engine_mode_indicator.object = build_engine_mode_markdown(runtime.engine_mode)
        widgets.campus_status_indicator.object = runtime.campus_status_markdown()
        template.title = _dashboard_title(runtime)
        trigger_dashboard_recompute(year_sel, meter_sel, building_sel, cold_start_days)
        _run_on_next_tick(lambda: load_selected_campus(target_campus_id))

    def initialize_dashboard() -> None:
        _run_on_next_tick(lambda: load_selected_campus(default_campus_id))

    campus_sel.param.watch(on_campus_change, "value")
    pn.state.onload(initialize_dashboard)

    return template
