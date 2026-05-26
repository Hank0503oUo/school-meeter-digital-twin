from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd


@dataclass
class CampusState:
    """Runtime state container for the dashboard campus context."""

    campus_id: str = ""
    campus_name: str = ""
    campus_cfg: Optional[object] = None
    campus_ready: bool = False
    missing_paths: list[str] = field(default_factory=list)

    energy_geojson: Path = Path()
    meter_hourly_csv: Path = Path()
    build_power_csv: Path = Path()
    metadata_uid_csv: Path = Path()
    metadata_loop_csv: Path = Path()
    weather_dir: Path = Path()
    v12_summary_csv: Path = Path()

    topo: Optional[Any] = None
    current_power_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    building_stats_base: pd.DataFrame = field(default_factory=pd.DataFrame)
    meter_summary_v12: pd.DataFrame = field(default_factory=pd.DataFrame)
    meter_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    metered_uid_set: set[str] = field(default_factory=set)

    year_meter_scalers: dict[int, dict[str, float]] = field(default_factory=dict)
    campus_year_scalers: dict[int, float] = field(default_factory=dict)

    yearly_inference_cache: dict[int, pd.DataFrame] = field(default_factory=dict)
    yearly_stats_cache: dict[int, pd.DataFrame] = field(default_factory=dict)
    yearly_geojson_cache: dict[int, dict] = field(default_factory=dict)
