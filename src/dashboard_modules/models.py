from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class DashboardWidgets:
    campus_sel: Any
    main_spinner: Any
    year_sel: Any
    building_sel: Any
    bldg_filter: Any
    meter_sel: Any
    time_scale_sel: Any
    cooling_sl: Any
    lighting_sl: Any
    occupancy_sl: Any
    equipment_sl: Any
    assistant_task_sel: Any
    assistant_quick_sel: Any
    assistant_query: Any
    assistant_spinner: Any
    assistant_run_btn: Any
    assistant_save_btn: Any
    assistant_force_mcp: Any
    assistant_save_memory: Any
    assistant_status: Any
    assistant_chat_log: Any
    assistant_structured: Any
    assistant_citations: Any
    color_mode: Any
    cold_start_days: Any
    map_display_toggle: Any
    campus_status_indicator: Any | None = None
    engine_mode_indicator: Any | None = None
    status_light: Any | None = None
    cloud_local_toggle: Any | None = None
    nvidia_online_toggle: Any | None = None
    assistant_image_upload: Any | None = None
