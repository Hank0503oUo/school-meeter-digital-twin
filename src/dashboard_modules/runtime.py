from __future__ import annotations

import copy
import json
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.building_inference import aggregate_meter_summary_by_uid
from src.campus_config import (
    CampusConfig,
    CampusConfigError,
    inference_cache_path_candidates,
    normalize_campus_id,
)
from src.constants import BASELINE_DATA_YEAR, COLOR_MODE_OPTIONS
from src.map_builder import get_building_stats_df
from src.project_paths import config_dir, data_dir, models_dir, resolve_project_path
from src.utils import (
    normalize_meter_name as _normalize_meter_name,
    split_meter_names as _split_meter_names,
    to_float as _to_float,
)

from src.dashboard_modules.building_alias import (
    expand_building_aliases,
    geometry_centroid,
    resolve_coord_from_aliases,
)
from src.dashboard_modules.cache import bounded_cache_get

try:
    from src.real_inference_engine import PIVDEngine, load_v12_building_summary
    from src.epw_reader import read_weather

    HAS_ENGINE = True
except (ImportError, ModuleNotFoundError):
    PIVDEngine = Any  # type: ignore[assignment]
    HAS_ENGINE = False

    def load_v12_building_summary(*args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame()

    def read_weather(*args, **kwargs) -> pd.DataFrame | None:
        return None


if TYPE_CHECKING:
    from src.demo_assistant import CampusAssistantService

log = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(slots=True)
class DashboardPaths:
    energy_geojson: Path = field(default_factory=lambda: data_dir("NTU", "ntu_energy.geojson"))
    meter_hourly_csv: Path = field(default_factory=lambda: models_dir("NTU_powerMeter_kW_hourly.csv"))
    build_meta_uid: Path = field(default_factory=lambda: data_dir("NTU", "BUILD DATA", "metadata_uid.csv"))
    build_meta_loop: Path = field(default_factory=lambda: data_dir("NTU", "BUILD DATA", "metadata_loop.csv"))
    weather_dir: Path = field(default_factory=lambda: models_dir("weather"))
    ui_prefs: Path = field(default_factory=lambda: config_dir("ui_prefs.json"))
    v12_summary: Path = field(default_factory=lambda: models_dir("v12_per_building_summary.csv"))
    official_patch: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "NTU" / "ntu_official_buildings_patch.csv"
    )

    def __post_init__(self) -> None:
        self.energy_geojson = resolve_project_path(self.energy_geojson)
        self.meter_hourly_csv = resolve_project_path(self.meter_hourly_csv)
        self.build_meta_uid = resolve_project_path(self.build_meta_uid)
        self.build_meta_loop = resolve_project_path(self.build_meta_loop)
        self.weather_dir = resolve_project_path(self.weather_dir)
        self.ui_prefs = resolve_project_path(self.ui_prefs)
        self.v12_summary = resolve_project_path(self.v12_summary)
        self.official_patch = resolve_project_path(self.official_patch)


def load_building_stats(energy_geojson_path: Path) -> pd.DataFrame:
    if energy_geojson_path.exists():
        return get_building_stats_df(energy_geojson_path)
    return pd.DataFrame(
        columns=[
            "name",
            "meter_name",
            "mean_kw",
            "annual_kwh",
            "annual_mwh",
            "eui",
            "peak_kw",
            "load_factor",
            "best_r2_oof",
            "best_cvrmse_oof",
            "archetype_label",
            "data_source",
            "coverage_ratio",
        ]
    )


def load_meter_year_scalers(
    meter_names: list[str],
    meter_csv_path: Path,
    baseline_year: int = BASELINE_DATA_YEAR,
) -> tuple[dict[int, dict[str, float]], dict[int, float]]:
    names = sorted(set(_normalize_meter_name(name) for name in meter_names if _normalize_meter_name(name)))
    if (not names) or (not meter_csv_path.exists()):
        return {}, {}

    try:
        header = pd.read_csv(meter_csv_path, nrows=0)
        dt_col = header.columns[0]
        available = [column for column in names if column in header.columns]
        if not available:
            return {}, {}
        raw = pd.read_csv(meter_csv_path, usecols=[dt_col] + available, encoding="utf-8")
    except (OSError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
        log.warning("Load meter yearly scalers failed: %s", exc)
        return {}, {}

    dt = pd.to_datetime(raw[dt_col], errors="coerce")
    raw = raw.loc[dt.notna()].copy()
    dt = dt.loc[dt.notna()]
    if raw.empty:
        return {}, {}

    years = dt.dt.year.astype(int)
    numeric = raw[available].apply(pd.to_numeric, errors="coerce")
    valid = numeric.notna() & numeric.ge(0.0)
    masked = numeric.where(valid)

    mean_by_year: dict[int, pd.Series] = {}
    for year, group in masked.groupby(years):
        mean_by_year[int(year)] = group.mean(axis=0)

    base_series = mean_by_year.get(int(baseline_year))
    if base_series is None or base_series.empty:
        return {}, {}

    year_meter_scalers: dict[int, dict[str, float]] = {}
    campus_year_scalers: dict[int, float] = {}
    for year, series in mean_by_year.items():
        scalers: dict[str, float] = {}
        ratio_values: list[float] = []
        for meter_name in available:
            base_value = _to_float(base_series.get(meter_name, np.nan), np.nan)
            current_value = _to_float(series.get(meter_name, np.nan), np.nan)
            if not (np.isfinite(base_value) and base_value > 0 and np.isfinite(current_value) and current_value >= 0):
                continue
            ratio = float(np.clip(current_value / base_value, 0.2, 5.0))
            scalers[meter_name] = ratio
            ratio_values.append(ratio)
        if scalers:
            year_meter_scalers[int(year)] = scalers
            campus_year_scalers[int(year)] = float(np.mean(ratio_values)) if ratio_values else 1.0

    return year_meter_scalers, campus_year_scalers


def year_factor_for_meters(
    meter_name_field: object,
    year: int,
    year_meter_scalers: dict[int, dict[str, float]],
) -> float:
    year_scalers = year_meter_scalers.get(int(year), {})
    meter_names = _split_meter_names(str(meter_name_field or ""))
    ratios = [year_scalers.get(_normalize_meter_name(name)) for name in meter_names]
    valid = [float(ratio) for ratio in ratios if ratio is not None and np.isfinite(ratio)]
    if valid:
        return float(np.mean(valid))
    return 1.0


def adjust_building_stats_by_year(
    building_stats_base: pd.DataFrame,
    year: int,
    year_meter_scalers: dict[int, dict[str, float]],
) -> pd.DataFrame:
    if building_stats_base is None or building_stats_base.empty:
        return building_stats_base.copy() if building_stats_base is not None else pd.DataFrame()

    factor_series = building_stats_base["meter_name"].apply(
        lambda value: year_factor_for_meters(value, year, year_meter_scalers)
    )
    adjusted = building_stats_base.copy()
    adjusted["year_factor"] = factor_series
    for column in ("mean_kw", "annual_kwh", "annual_mwh", "eui", "peak_kw"):
        if column in adjusted.columns:
            adjusted[column] = pd.to_numeric(adjusted[column], errors="coerce") * factor_series
    return adjusted


def normalize_geojson_name(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    for token in (" ", "-", "_", "/", "\\", ".", ",", "(", ")", "（", "）", ":"):
        text = text.replace(token, "")
    return text


def load_geojson(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


_MODELED_DATA_SOURCES = {
    "metered",
    "inferred",
    "measured meter",
    "pi-vd inferred",
    "pi-vd",
    "pivd",
    "csv_annotated",
}


def _load_csv_building_index(metadata_uid_path: Path) -> dict[str, dict[str, Any]]:
    """Load metadata_uid.csv and index by normalized Chinese + English names.

    Returns a dict keyed by normalized name → {uid, name, nameE, area, floors, buildType, ...}.
    """
    index: dict[str, dict[str, Any]] = {}
    if not metadata_uid_path or not metadata_uid_path.exists():
        return index

    try:
        df = pd.read_csv(metadata_uid_path, encoding="utf-8")
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        log.warning("Load metadata_uid.csv failed: %s", exc)
        return index

    def _to_float(val: Any) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    for _, row in df.iterrows():
        uid = str(row.get("uid", "") or "").strip()
        if not uid:
            continue
        record = {
            "uid": uid,
            "name": str(row.get("name", "") or "").strip(),
            "nameE": str(row.get("nameE", "") or "").strip(),
            "area_csv": _to_float(row.get("area")),
            "floors_csv": _to_float(row.get("floors")),
            "build_type_zh": str(row.get("buildType1C", "") or "").strip(),
            "build_type_en": str(row.get("buildType1E", "") or "").strip(),
            "college_zh": str(row.get("buildType2C", "") or "").strip(),
            "college_en": str(row.get("buildType2E", "") or "").strip(),
        }
        for name_field in ("name", "nameE"):
            n = normalize_geojson_name(record.get(name_field))
            if n:
                index.setdefault(n, record)
    return index


def _annotate_with_csv_index(
    base_geojson: dict,
    csv_index: dict[str, dict[str, Any]],
) -> int:
    """Stamp OSM features with CSV uid/name/area when names match.

    Returns the number of features annotated.
    """
    if not csv_index:
        return 0

    annotated = 0
    for feature in base_geojson.get("features", []):
        props = feature.setdefault("properties", {})
        match = None
        for name_field in ("name", "name_en"):
            key = normalize_geojson_name(props.get(name_field))
            if key and key in csv_index:
                match = csv_index[key]
                break
        if match is None:
            continue
        # Only stamp fields that are missing / empty — don't overwrite real energy data
        if not str(props.get("uid", "") or "").strip():
            props["uid"] = match["uid"]
        if not str(props.get("data_source", "") or "").strip():
            props["data_source"] = "csv_annotated"
        if not str(props.get("build_type", "") or "").strip() and match.get("build_type_zh"):
            props["build_type"] = match["build_type_zh"]
        if not str(props.get("college", "") or "").strip() and match.get("college_zh"):
            props["college"] = match["college_zh"]
        if not props.get("b_area") and match.get("area_csv"):
            props["b_area"] = match["area_csv"]
        if not props.get("floors") and match.get("floors_csv"):
            props["floors"] = match["floors_csv"]
        props["csv_matched"] = True
        annotated += 1
    return annotated


def _is_modeled_feature(props: dict[str, Any]) -> bool:
    """True if this feature is a building we can model (has CSV row or energy data)."""
    if props.get("csv_matched"):
        return True
    meter_name = str(props.get("meter_name", "") or "").strip()
    if meter_name:
        return True
    data_source = str(props.get("data_source", "") or "").strip().lower()
    if data_source in _MODELED_DATA_SOURCES:
        return True
    for key in ("mean_kw", "annual_kwh", "eui", "eui_kw_per_m2"):
        value = props.get(key)
        if value in (None, "", 0, 0.0):
            continue
        try:
            if float(value) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def prepare_map_base_geojson(
    buildings_geojson_path: Path,
    energy_geojson_path: Path,
    metadata_uid_path: Path | None = None,
) -> dict:
    if buildings_geojson_path.exists():
        base = load_geojson(buildings_geojson_path)
    else:
        base = load_geojson(energy_geojson_path)

    # Merge energy.geojson into OSM base by osm_id / name
    if energy_geojson_path.exists() and (
        not buildings_geojson_path.exists()
        or buildings_geojson_path.resolve() != energy_geojson_path.resolve()
    ):
        energy = load_geojson(energy_geojson_path)
        by_osm_id: dict[int, dict[str, Any]] = {}
        by_name: dict[str, dict[str, Any]] = {}

        for feature in energy.get("features", []):
            props = feature.get("properties", {}) or {}
            osm_id = props.get("osm_id")
            try:
                if osm_id is not None:
                    by_osm_id[int(osm_id)] = props
            except (TypeError, ValueError):
                pass

            norm_name = normalize_geojson_name(props.get("name"))
            if norm_name:
                by_name[norm_name] = props

        for feature in base.get("features", []):
            props = feature.setdefault("properties", {})
            match = None

            osm_id = props.get("osm_id")
            try:
                if osm_id is not None:
                    match = by_osm_id.get(int(osm_id))
            except (TypeError, ValueError):
                match = None

            if match is None:
                match = by_name.get(normalize_geojson_name(props.get("name")))

            if match is None:
                continue

            for key, value in match.items():
                if value is None:
                    continue
                if isinstance(value, str) and not value.strip():
                    continue
                props[key] = value

    # Annotate OSM features with CSV metadata (expands the modelable set)
    if metadata_uid_path is not None:
        csv_index = _load_csv_building_index(metadata_uid_path)
        n_annotated = _annotate_with_csv_index(base, csv_index)
        if n_annotated:
            log.info("Annotated %d OSM features with CSV metadata (metadata_uid.csv)", n_annotated)

    return _filter_modeled_features(base)


def _filter_modeled_features(geojson: dict) -> dict:
    """Drop features that lack CSV annotation so only modelable buildings show on the map."""
    features = geojson.get("features", []) or []
    kept = [f for f in features if _is_modeled_feature(f.get("properties", {}) or {})]
    dropped = len(features) - len(kept)
    log.info("Map filter: kept %d modelable buildings (%d raw OSM footprints dropped).", len(kept), dropped)
    if not kept and features:
        log.warning("Map filter removed all features; falling back to unfiltered set.")
        return geojson
    geojson["features"] = kept
    return geojson


def generate_campus_baseline(mean_kw: float = 5100.0, hours: int = 8760) -> np.ndarray:
    np.random.seed(42)
    t = np.arange(hours)
    seasonal = 600 * np.sin(2 * np.pi * (t / 8760 - 0.25))
    daily = 400 * np.sin(2 * np.pi * t / 24 - np.pi / 2)
    weekday_mask = ((t // 24) % 7 < 5).astype(float)
    weekday_boost = 200 * weekday_mask
    noise = np.random.normal(0, 150, hours)
    hourly = mean_kw + seasonal + daily + weekday_boost + noise
    return np.clip(hourly, mean_kw * 0.3, mean_kw * 2.0)


def init_pivd_engine(campus_config: CampusConfig | None = None) -> Any | None:
    if not HAS_ENGINE:
        return None
    try:
        if campus_config is not None:
            return PIVDEngine.from_campus(campus_config)
        return PIVDEngine.from_defaults()
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        log.warning("PI-VD engine load failed: %s", exc)
        return None


@dataclass(slots=True)
class DashboardRuntime:
    paths: DashboardPaths = field(default_factory=DashboardPaths)
    active_campus_id: str = "ntu"
    active_campus_cfg: CampusConfig | None = None
    active_campus_ready: bool = False
    active_campus_missing: list[str] = field(default_factory=list)
    active_campus_name: str = "NTU"
    active_energy_geojson: Path = field(default_factory=Path)
    active_buildings_geojson: Path = field(default_factory=Path)
    active_meter_hourly_csv: Path = field(default_factory=Path)
    active_build_meta_uid: Path = field(default_factory=Path)
    active_build_meta_loop: Path = field(default_factory=Path)
    active_weather_dir: Path = field(default_factory=Path)
    active_v12_summary: Path = field(default_factory=Path)
    building_stats_base: pd.DataFrame = field(default_factory=pd.DataFrame)
    campus_baseline: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    pivd_engine: Any | None = None
    engine_mode: str = "Loading campus data"
    meter_summary_v12: pd.DataFrame = field(default_factory=pd.DataFrame)
    meter_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    year_meter_scalers: dict[int, dict[str, float]] = field(default_factory=dict)
    campus_year_scalers: dict[int, float] = field(default_factory=dict)
    metered_uid_set: set[str] = field(default_factory=set)
    building_aliases_by_uid: dict[str, list[str]] = field(default_factory=dict)
    building_focus_coords_by_uid: dict[str, tuple[float, float]] = field(default_factory=dict)
    _assistant_service: CampusAssistantService | None = field(default=None, init=False, repr=False)
    assistant_last_payload: dict[str, object] = field(default_factory=dict)
    yearly_inference_cache: OrderedDict[int, pd.DataFrame] = field(default_factory=OrderedDict)
    yearly_stats_cache: OrderedDict[int, pd.DataFrame] = field(default_factory=OrderedDict)
    yearly_geojson_cache: OrderedDict[int, dict] = field(default_factory=OrderedDict)
    shell_geojson_cache: dict[str, dict] = field(default_factory=dict)
    campus_loaded: bool = False
    campus_loading: bool = False
    loaded_campus_id: str | None = None
    current_llm_model: str = field(
        default_factory=lambda: os.getenv("ENERGY_LLM_MODEL", "LM Studio (local)").strip()
    )

    def __post_init__(self) -> None:
        self.active_energy_geojson = self.paths.energy_geojson
        self.active_buildings_geojson = self.paths.energy_geojson
        self.active_meter_hourly_csv = self.paths.meter_hourly_csv
        self.active_build_meta_uid = self.paths.build_meta_uid
        self.active_build_meta_loop = self.paths.build_meta_loop
        self.active_weather_dir = self.paths.weather_dir
        self.active_v12_summary = self.paths.v12_summary
        self.campus_baseline = generate_campus_baseline()

    @property
    def assistant_service(self) -> CampusAssistantService:
        if self._assistant_service is None:
            from src.demo_assistant import CampusAssistantService

            self._assistant_service = CampusAssistantService()
        return self._assistant_service

    def campus_options(self, default_campus_id: str = "ntu") -> tuple[str, dict[str, str]]:
        campus_ids = CampusConfig.list_available()
        if not campus_ids:
            campus_ids = [default_campus_id]
        if default_campus_id not in campus_ids:
            default_campus_id = campus_ids[0]

        campus_label_to_id: dict[str, str] = {}
        for campus_id in campus_ids:
            try:
                cfg = CampusConfig.load(campus_id)
                campus_label_to_id[f"{cfg.campus_name} ({campus_id.upper()})"] = campus_id
            except (TypeError, ValueError, CampusConfigError):
                campus_label_to_id[campus_id.upper()] = campus_id
        return default_campus_id, campus_label_to_id

    def prepare_campus_shell(self, campus_id: str) -> None:
        self.active_campus_id = normalize_campus_id(campus_id)
        self.active_campus_cfg = None
        self.active_campus_missing = []
        self.active_campus_ready = False
        self.active_campus_name = self.active_campus_id.upper()
        self.engine_mode = "Loading campus data"

        try:
            self.active_campus_cfg = CampusConfig.load(self.active_campus_id)
        except (CampusConfigError, TypeError, ValueError) as exc:
            log.warning("Campus config load failed (%s): %s", self.active_campus_id, exc)

        if self.active_campus_cfg is not None:
            self.active_campus_name = self.active_campus_cfg.campus_name
            self.active_campus_missing = self.active_campus_cfg.missing_required_paths()
            self.active_campus_ready = self.active_campus_cfg.is_data_ready()
            self.active_energy_geojson = (
                self.active_campus_cfg.get_path("energy_geojson", self.paths.energy_geojson) or self.paths.energy_geojson
            )
            self.active_buildings_geojson = (
                self.active_campus_cfg.get_path("buildings_geojson", self.active_energy_geojson)
                or self.active_energy_geojson
            )
            self.active_meter_hourly_csv = (
                self.active_campus_cfg.get_path("meter_csv", self.paths.meter_hourly_csv) or self.paths.meter_hourly_csv
            )
            self.active_build_meta_uid = (
                self.active_campus_cfg.get_path("metadata_uid", self.paths.build_meta_uid) or self.paths.build_meta_uid
            )
            self.active_build_meta_loop = (
                self.active_campus_cfg.get_path("metadata_loop", self.paths.build_meta_loop) or self.paths.build_meta_loop
            )
            self.active_weather_dir = (
                self.active_campus_cfg.get_path("weather_dir", self.paths.weather_dir) or self.paths.weather_dir
            )
            self.active_v12_summary = (
                self.active_campus_cfg.get_path("v12_summary", self.paths.v12_summary) or self.paths.v12_summary
            )
            return

        self.active_energy_geojson = self.paths.energy_geojson
        self.active_buildings_geojson = self.paths.energy_geojson
        self.active_meter_hourly_csv = self.paths.meter_hourly_csv
        self.active_build_meta_uid = self.paths.build_meta_uid
        self.active_build_meta_loop = self.paths.build_meta_loop
        self.active_weather_dir = self.paths.weather_dir
        self.active_v12_summary = self.paths.v12_summary
        self.active_campus_ready = (
            self.active_energy_geojson.exists()
            and self.active_build_meta_uid.exists()
            and self.active_build_meta_loop.exists()
        )

    def build_loading_geojson(self) -> dict:
        cache_key = str(self.active_campus_id or "default")
        cached = self.shell_geojson_cache.get(cache_key)
        if cached is not None:
            return copy.deepcopy(cached)

        if not self.active_energy_geojson.exists():
            return {}

        base_geojson = prepare_map_base_geojson(
            self.active_buildings_geojson,
            self.active_energy_geojson,
            metadata_uid_path=self.active_build_meta_uid,
        )
        for feature in base_geojson.get("features", []):
            props = feature.setdefault("properties", {})
            props["data_source"] = "loading"
            props["has_meter_data"] = False
            props["energy_tier"] = "NO_DATA"
            props["coverage_ratio"] = float(_to_float(props.get("coverage_ratio", 0.6), 0.6))
            props["mean_kw"] = float(_to_float(props.get("mean_kw", 1.0), 1.0))
            props["eui"] = float(_to_float(props.get("eui", 1.0), 1.0))
            props["best_r2_oof"] = float(_to_float(props.get("best_r2_oof", 0.5), 0.5))
            props.setdefault("meter_name", "")

        self.shell_geojson_cache[cache_key] = copy.deepcopy(base_geojson)
        return base_geojson

    def load_ui_prefs(self) -> dict[str, Any]:
        try:
            if self.paths.ui_prefs.exists():
                return json.loads(self.paths.ui_prefs.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Load UI prefs failed: %s", exc)
        return {}

    def save_ui_prefs(self, prefs: dict[str, Any]) -> None:
        try:
            self.paths.ui_prefs.parent.mkdir(parents=True, exist_ok=True)
            self.paths.ui_prefs.write_text(json.dumps(prefs, ensure_ascii=False, indent=2), encoding="utf-8")
        except (OSError, TypeError, ValueError) as exc:
            log.warning("Save UI prefs failed: %s", exc)

    def load_default_color_mode(self) -> str:
        mode = str(self.load_ui_prefs().get("color_mode", "tier")).strip()
        if mode in COLOR_MODE_OPTIONS:
            return mode
        return "tier"

    def save_color_mode(self, mode: str) -> None:
        prefs = self.load_ui_prefs()
        prefs["color_mode"] = str(mode)
        self.save_ui_prefs(prefs)

    def load_campus(self, campus_id: str) -> bool:
        target_campus_id = normalize_campus_id(campus_id)
        if self.campus_loaded and self.loaded_campus_id == target_campus_id and not self.campus_loading:
            return False

        self.campus_loading = True
        self.campus_loaded = False
        self.loaded_campus_id = None
        try:
            self.reload_campus_state(target_campus_id)
            self.campus_loaded = True
            self.loaded_campus_id = self.active_campus_id
            return True
        finally:
            self.campus_loading = False

    def clear_yearly_cache(self) -> None:
        self.yearly_inference_cache.clear()
        self.yearly_stats_cache.clear()
        self.yearly_geojson_cache.clear()

    def build_year_adjusted_geojson(self, base_geojson_path: Path | dict, year: int) -> dict:
        if isinstance(base_geojson_path, dict):
            geojson = copy.deepcopy(base_geojson_path)
        else:
            if not base_geojson_path.exists():
                return {}
            geojson = load_geojson(base_geojson_path)

        name_to_meta: dict[str, dict[str, Any]] = {}
        if self.pivd_engine and self.pivd_engine.metadata_scaler.is_loaded:
            for uid in self.pivd_engine.metadata_scaler.list_uids():
                metadata = self.pivd_engine.metadata_scaler.get_metadata(uid)
                if metadata and metadata.get("name"):
                    name_to_meta[str(metadata["name"])] = metadata

        inference_df = self.get_yearly_inference(year)
        infer_by_uid: dict[str, dict[str, Any]] = {}
        if inference_df is not None and not inference_df.empty and "uid" in inference_df.columns:
            for _, row in inference_df.iterrows():
                uid = str(row.get("uid", "")).strip()
                if uid:
                    infer_by_uid[uid] = row.to_dict()

        tier_colors = {
            "HIGH": [215, 48, 39, 220],
            "NORMAL": [240, 196, 25, 220],
            "LOW": [26, 152, 80, 220],
        }

        for feature in geojson.get("features", []):
            props = feature.get("properties", {})
            factor = year_factor_for_meters(props.get("meter_name", ""), year, self.year_meter_scalers)
            props["year_factor"] = round(factor, 4)
            props["data_year"] = int(year)

            building_name = props.get("name", "")
            if building_name in name_to_meta:
                meta = name_to_meta[building_name]
                props["uid"] = meta.get("uid", "")
                props["b_area"] = meta.get("area", 0)
                props["b_floors"] = meta.get("floors", 0) + meta.get("basement", 0)
                props["b_type"] = meta.get("buildType", "")
                props["b_year"] = meta.get("year", "")
            else:
                props.setdefault("uid", "")
                props.setdefault("b_area", "")
                props.setdefault("b_floors", "")
                props.setdefault("b_type", "")
                props.setdefault("b_year", "")

            for key, digits in (
                ("mean_kw", 1),
                ("mean_kw_raw", 1),
                ("peak_kw", 1),
                ("eui", 1),
                ("eui_raw", 1),
                ("annual_kwh", 0),
                ("annual_kwh_raw", 0),
                ("annual_mwh", 1),
                ("annual_mwh_raw", 1),
            ):
                value = props.get(key)
                if value is None:
                    continue
                numeric = _to_float(value, np.nan)
                if np.isfinite(numeric):
                    props[key] = round(numeric * factor, digits)

            uid = str(props.get("uid", "")).strip()
            infer_row = infer_by_uid.get(uid)
            if infer_row is not None:
                mean_kw = _to_float(infer_row.get("mean_kw", np.nan), np.nan)
                area = _to_float(infer_row.get("area", props.get("b_area", np.nan)), np.nan)
                eui_kw_per_m2 = _to_float(infer_row.get("eui_kw_per_m2", np.nan), np.nan)
                eui_annual = eui_kw_per_m2 * 8760.0 if np.isfinite(eui_kw_per_m2) else np.nan

                if np.isfinite(mean_kw):
                    props["mean_kw"] = round(float(mean_kw), 1)
                    props["annual_kwh"] = round(float(mean_kw) * 8760.0, 0)
                    props["annual_mwh"] = round(float(mean_kw) * 8.76, 1)
                    props["peak_kw"] = round(float(mean_kw) * 1.5, 1)
                if np.isfinite(area):
                    props["b_area"] = round(float(area), 1)
                if np.isfinite(eui_kw_per_m2):
                    props["eui_kw_per_m2"] = round(float(eui_kw_per_m2), 4)
                if np.isfinite(eui_annual):
                    props["eui"] = round(float(eui_annual), 1)

                tier = str(infer_row.get("energy_tier", "NORMAL")).strip().upper()
                source = str(infer_row.get("data_source", "inferred")).strip().lower()
                props["energy_tier"] = tier
                props["tier_color"] = tier_colors.get(tier, tier_colors["NORMAL"])
                props["data_source"] = source
                # Treat any "metered" / "ncu_real_*" / 實測 as real measurements;
                # "pivd_estimate_*" and "inferred" are physics/inferred only.
                is_real = (source == "metered"
                           or "ncu_real" in source
                           or "measured" in source
                           or "實測" in source)
                is_inferred = (source == "inferred"
                               or "inferred" in source
                               or "pivd_estimate" in source
                               or "推估" in source
                               or "virtual" in source)
                props["has_meter_data"] = is_real
                props["outline_dash_array"] = [4, 3] if is_inferred else [1, 0]
                props["confidence_level"] = "medium" if is_real else "low"
                props.setdefault("meter_name", "")

            if "energy_tier" not in props:
                has_data = props.get("has_meter_data", False)
                data_source = str(props.get("data_source", "")).strip().lower()
                if has_data or data_source in {"metered", "inferred"}:
                    props["energy_tier"] = "NORMAL"
                    props["tier_color"] = tier_colors["NORMAL"]
                else:
                    props["energy_tier"] = "NO_DATA"
                    props["tier_color"] = [220, 220, 220, 120]
                props["outline_dash_array"] = [1, 0]

        return geojson

    def reload_campus_state(self, campus_id: str) -> None:
        self.prepare_campus_shell(campus_id)

        self.building_stats_base = load_building_stats(self.active_energy_geojson)
        self.campus_baseline = generate_campus_baseline()
        self.pivd_engine = init_pivd_engine(self.active_campus_cfg) if self.active_campus_ready else None
        self.engine_mode = "PI-VD" if self.pivd_engine else "Fallback"

        try:
            self.meter_summary_v12 = load_v12_building_summary(path=self.active_v12_summary)
        except (OSError, ValueError, pd.errors.EmptyDataError) as exc:
            log.warning("Load v12 summary failed (%s): %s", self.active_campus_id, exc)
            self.meter_summary_v12 = pd.DataFrame()

        meter_rows: list[dict[str, Any]] = []
        if not self.building_stats_base.empty:
            for _, row in self.building_stats_base.iterrows():
                for meter_name in _split_meter_names(str(row.get("meter_name", "") or "")):
                    meter_rows.append(
                        {
                            "meter_name": _normalize_meter_name(meter_name),
                            "building_name": row.get("name", ""),
                            "mean_kw": float(row.get("mean_kw", 0) or 0),
                            "annual_kwh": float(row.get("annual_kwh", 0) or 0),
                            "eui": float(row.get("eui", 0) or 0),
                            "best_r2_oof": float(row.get("best_r2_oof", 0) or 0),
                            "coverage_ratio": float(row.get("coverage_ratio", np.nan))
                            if pd.notna(row.get("coverage_ratio", np.nan))
                            else np.nan,
                        }
                    )
        elif not self.meter_summary_v12.empty:
            for _, row in self.meter_summary_v12.iterrows():
                meter_rows.append(
                    {
                        "meter_name": _normalize_meter_name(row.get("meter_name", "")),
                        "building_name": "",
                        "mean_kw": float(row.get("mean_kw", 0) or 0),
                        "annual_kwh": float(row.get("mean_kw", 0) or 0) * 8760.0,
                        "eui": np.nan,
                        "best_r2_oof": float(row.get("best_r2_oof", 0) or 0),
                        "coverage_ratio": np.nan,
                    }
                )

        if meter_rows:
            self.meter_df = pd.DataFrame(meter_rows).drop_duplicates(subset=["meter_name"])
        else:
            self.meter_df = pd.DataFrame(
                columns=["meter_name", "building_name", "mean_kw", "annual_kwh", "eui", "best_r2_oof", "coverage_ratio"]
            )

        self.year_meter_scalers, self.campus_year_scalers = load_meter_year_scalers(
            self.meter_df["meter_name"].tolist() if not self.meter_df.empty else [],
            meter_csv_path=self.active_meter_hourly_csv,
        )

        metered_uid_summary = aggregate_meter_summary_by_uid(
            self.meter_summary_v12,
            metadata_loop_path=self.active_build_meta_loop,
            metadata_uid_path=self.active_build_meta_uid,
        )
        self.metered_uid_set = set(
            str(value).strip()
            for value in metered_uid_summary.get("uid", pd.Series([], dtype=str)).tolist()
            if str(value).strip()
        )
        self.building_aliases_by_uid = {}
        self.building_focus_coords_by_uid = {}

        source_geojson_path = (
            self.active_campus_cfg.get_path("buildings_geojson", self.active_energy_geojson)
            if self.active_campus_cfg is not None
            else self.active_energy_geojson
        ) or self.active_energy_geojson

        if source_geojson_path.exists():
            base_geojson = load_geojson(source_geojson_path)
            name_to_coord: dict[str, tuple[float, float]] = {}
            for feature in base_geojson.get("features", []):
                props = feature.get("properties", {})
                centroid = geometry_centroid(feature.get("geometry", {}))
                if centroid is None:
                    continue
                for osm_name in (
                    str(props.get("name", "")).strip(),
                    str(props.get("name_en", "")).strip(),
                ):
                    if osm_name:
                        name_to_coord[osm_name] = centroid

            building_coords: dict[str, tuple[float, float]] = {}
            loop_aliases_by_uid: dict[str, list[str]] = {}
            if self.active_build_meta_loop.exists():
                try:
                    uid_loop = pd.read_csv(self.active_build_meta_loop, encoding="utf-8")
                    if len(uid_loop.columns) >= 3:
                        loop_name_col = uid_loop.columns[2]
                        for uid, group in uid_loop.groupby("uid"):
                            uid_key = str(uid).strip()
                            if uid_key:
                                loop_aliases_by_uid[uid_key] = expand_building_aliases(*group[loop_name_col].tolist())
                except (OSError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
                    log.warning("Failed to load metadata_loop for coordinate mapping: %s", exc)

            official_patch_aliases_by_code: dict[str, list[str]] = {}
            if self.active_campus_id == "ntu" and self.paths.official_patch.exists():
                try:
                    official_patch = pd.read_csv(self.paths.official_patch, encoding="utf-8")
                    for _, patch_row in official_patch.iterrows():
                        code = str(patch_row.get("Code", "")).strip()
                        if not code:
                            continue
                        alias_values: list[str] = []
                        for column in ("Name_ZH", "Name_EN", "Alias_ZH"):
                            value = str(patch_row.get(column, "")).strip()
                            if (not value) or value.lower() == "nan":
                                continue
                            if column == "Alias_ZH":
                                alias_values.extend(part.strip() for part in value.split("|") if part.strip())
                            else:
                                alias_values.append(value)
                        if alias_values:
                            official_patch_aliases_by_code[code] = expand_building_aliases(*alias_values)
                except (OSError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
                    log.warning("Failed to load NTU official building patch: %s", exc)

            if self.active_build_meta_uid.exists():
                try:
                    uid_meta = pd.read_csv(self.active_build_meta_uid, encoding="utf-8")
                    for _, row in uid_meta.iterrows():
                        uid = str(row.get("uid", "")).strip()
                        if not uid:
                            continue
                        doorplate = str(row.get("doorplate", "")).strip()
                        aliases = expand_building_aliases(
                            row.get("name", ""),
                            row.get("nameC", ""),
                            row.get("nameE", ""),
                            row.get("nameE.2", ""),
                            row.get("nameE.3", ""),
                            row.get("buildId", ""),
                            row.get("code", ""),
                            *(loop_aliases_by_uid.get(uid, [])),
                            *(official_patch_aliases_by_code.get(doorplate, [])),
                        )
                        if aliases:
                            self.building_aliases_by_uid[uid] = aliases
                            coord = resolve_coord_from_aliases(aliases, name_to_coord)
                            if coord is not None:
                                building_coords[uid] = coord
                except (OSError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
                    log.warning("Failed to load metadata_uid for coordinate mapping: %s", exc)

            self.building_focus_coords_by_uid = dict(building_coords)

        self.clear_yearly_cache()

    def get_yearly_inference(self, year: int) -> pd.DataFrame:
        if not self.campus_loaded:
            return pd.DataFrame()
        cache_key = int(year)

        def _load_inference() -> pd.DataFrame:
            candidates = inference_cache_path_candidates(self.active_campus_id, year=cache_key)
            cache_path = next((path for path in candidates if path.exists()), candidates[0])
            if not cache_path.exists():
                log.warning("No inference cache found for %s in %s", cache_key, ", ".join(str(path) for path in candidates))
                return pd.DataFrame()
            try:
                return pd.read_parquet(cache_path)
            except (ImportError, OSError, ValueError) as exc:
                log.warning("Failed to read parquet cache for %s: %s", cache_key, exc)
                return pd.DataFrame()

        return bounded_cache_get(self.yearly_inference_cache, cache_key, _load_inference)

    @staticmethod
    def stats_from_inference(inference_df: pd.DataFrame) -> pd.DataFrame:
        if inference_df is None or inference_df.empty:
            return pd.DataFrame()

        stats = inference_df.copy()
        stats["name"] = stats.get("name", "")
        stats["meter_name"] = stats.get("meter_name", "")
        stats["eui"] = pd.to_numeric(stats.get("eui_kw_per_m2", np.nan), errors="coerce") * 8760.0
        stats["peak_kw"] = pd.to_numeric(stats.get("mean_kw", np.nan), errors="coerce") * 1.5
        if "annual_kwh" not in stats.columns:
            stats["annual_kwh"] = pd.to_numeric(stats["mean_kw"], errors="coerce") * 8760.0
        stats["annual_mwh"] = pd.to_numeric(stats["annual_kwh"], errors="coerce") / 1000.0
        stats["load_factor"] = 0.55
        stats["archetype_label"] = stats.get("buildType", "")
        if "best_r2_oof" not in stats.columns:
            stats["best_r2_oof"] = np.nan
        if "best_cvrmse_oof" not in stats.columns:
            stats["best_cvrmse_oof"] = np.nan
        if "coverage_ratio" not in stats.columns:
            stats["coverage_ratio"] = np.nan
        stats["data_source"] = stats.get("data_source", "inferred")

        keep_cols = [
            "uid",
            "name",
            "meter_name",
            "mean_kw",
            "annual_kwh",
            "annual_mwh",
            "eui",
            "peak_kw",
            "load_factor",
            "best_r2_oof",
            "best_cvrmse_oof",
            "archetype_label",
            "data_source",
            "coverage_ratio",
            "energy_tier",
            "eui_kw_per_m2",
            "area",
            "floors",
            "buildType",
            "timeseries",
        ]
        return stats[[column for column in keep_cols if column in stats.columns]].copy()

    def get_yearly_stats(self, year: int) -> pd.DataFrame:
        if not self.campus_loaded:
            return pd.DataFrame()
        cache_key = int(year)

        def _load_stats() -> pd.DataFrame:
            inference_df = self.get_yearly_inference(cache_key)
            if inference_df is not None and not inference_df.empty:
                return self.stats_from_inference(inference_df)
            return adjust_building_stats_by_year(self.building_stats_base, cache_key, self.year_meter_scalers)

        return bounded_cache_get(self.yearly_stats_cache, cache_key, _load_stats)

    def get_yearly_geojson(self, year: int) -> dict:
        if not self.campus_loaded:
            return {}
        cache_key = int(year)

        def _load_geojson() -> dict:
            base_geojson = prepare_map_base_geojson(
                self.active_buildings_geojson,
                self.active_energy_geojson,
                metadata_uid_path=self.active_build_meta_uid,
            )
            return self.build_year_adjusted_geojson(base_geojson, cache_key)

        return bounded_cache_get(self.yearly_geojson_cache, cache_key, _load_geojson)

    def campus_status_markdown(self) -> str:
        if self.campus_loading:
            return (
                f"### Campus: {self.active_campus_name}\n"
                f"- id: `{self.active_campus_id}`\n"
                "- data readiness: loading"
            )
        if not self.campus_loaded:
            return (
                f"### Campus: {self.active_campus_name}\n"
                f"- id: `{self.active_campus_id}`\n"
                "- data readiness: pending initial load"
            )
        if self.active_campus_ready:
            return (
                f"### Campus: {self.active_campus_name}\n"
                f"- id: `{self.active_campus_id}`\n"
                "- data readiness: ready"
            )
        missing = ", ".join(self.active_campus_missing) if self.active_campus_missing else "unknown"
        return (
            f"### Campus: {self.active_campus_name}\n"
            f"- id: `{self.active_campus_id}`\n"
            "- data readiness: incomplete\n"
            f"- missing: `{missing}`"
        )

    def total_building_count(self) -> int:
        if not self.campus_loaded:
            return 0
        if self.pivd_engine and self.pivd_engine.metadata_scaler.is_loaded:
            return len(self.pivd_engine.metadata_scaler.list_uids())
        if not self.building_stats_base.empty and "name" in self.building_stats_base.columns:
            return int(self.building_stats_base["name"].nunique())
        return 0

    def selected_building_label(self, selected_uid: str, year: int) -> str:
        if not self.campus_loaded:
            return "Loading campus data"
        uid = str(selected_uid or "").strip()
        if not uid or uid == "ALL":
            return "Campus overview"

        inference_df = self.get_yearly_inference(year)
        if inference_df is not None and not inference_df.empty and "uid" in inference_df.columns:
            matched = inference_df[inference_df["uid"].astype(str).str.strip() == uid]
            if not matched.empty:
                name = str(matched.iloc[0].get("name", "")).strip()
                if name:
                    return f"{name} ({uid})"

        if self.pivd_engine and self.pivd_engine.metadata_scaler.is_loaded:
            metadata = self.pivd_engine.metadata_scaler.get_metadata(uid) or {}
            name = str(metadata.get("name", "")).strip()
            if name:
                return f"{name} ({uid})"

        return uid

    def find_weather_for_year(self, year: int) -> pd.DataFrame | None:
        if not HAS_ENGINE:
            return None
        weather_dir = Path(self.active_weather_dir)
        if not weather_dir.exists():
            return None
        for extension in ("csv", "epw"):
            for file_path in weather_dir.glob(f"*{year}*.{extension}"):
                try:
                    return read_weather(file_path)
                except (OSError, ValueError):
                    continue
        return None
