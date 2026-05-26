# -*- coding: utf-8 -*-
"""
Step 4：anymap-ts 地圖圖層建構

合併 OSM GeoJSON + 電表比對 + V12 指標 → 能耗著色的 2.5D 校園地圖。

三個圖層:
  ① fill-extrusion：2.5D 建物拉伸 + EUI 色彩
  ② line：建物輪廓（實測=亮藍, 無資料=暗灰）
  ③ scatterplot：400+ 電表點位（DeckGL）

用法：
    python -m src.map_builder
    python -m src.map_builder --export test_map.html
"""

from __future__ import annotations

import json
import argparse
import re
import math
import copy
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import pydeck as pdk
import yaml
from pydeck.types import String

from src.project_paths import campus_data_dir, data_dir, project_root, resolve_project_path
from src.utils import to_float as _to_float, weighted_mean as _weighted_mean

_ROOT_DIR = project_root()
_DATA_DIR = data_dir()
_FOCUS_MARKER_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">
  <path d="M32 3C20.4 3 11 12.4 11 24c0 15.3 17.1 35.3 20 38.5.5.5 1.4.5 1.9 0C35.9 59.3 53 39.3 53 24 53 12.4 43.6 3 32 3z"
        fill="#db4437" stroke="#b3261e" stroke-width="2"/>
  <circle cx="32" cy="24" r="9.5" fill="#9d1c11"/>
</svg>
""".strip()
_FOCUS_MARKER_ATLAS = f"data:image/svg+xml;charset=utf-8,{quote(_FOCUS_MARKER_SVG)}"
_FOCUS_MARKER_MAPPING = {
    "pin": {
        "x": 0,
        "y": 0,
        "width": 64,
        "height": 64,
        "anchorX": 32,
        "anchorY": 62,
        "mask": False,
    },
}

DEFAULT_TRUST_POLICY = {
    "hide_absolute_roles": ["shared_total", "unknown", "feeder", "campus_total", "backup"],
    "coverage": {
        "hide_absolute_below": 0.50,
        "low_flag_below": 0.50,
        "mid_flag_below": 0.70,
        "confidence_medium_min": 0.50,
        "confidence_high_min": 0.80,
    },
    "confidence": {
        "high_score_min": 90.0,
        "medium_score_min": 70.0,
        "high_roles": ["building_total", "submeter"],
        "medium_roles": ["building_total", "submeter", "shared_total"],
    },
    "physics": {
        "night_to_day_high_default": 1.20,
        "peak_ratio_high": 8.00,
        "peak_ratio_flat": 1.05,
        "low_eui_kw_threshold": {
            "eui_below": 5.0,
            "kw_above": 50.0,
        },
    },
    "r2_definition_gap_flag": 0.25,
    "eui_caps_by_role": {
        "building_total": 1200.0,
        "submeter": 1400.0,
        "shared_total": 700.0,
        "unknown": 550.0,
        "feeder": 450.0,
        "campus_total": 350.0,
        "virtual": 400.0,
    },
    "relative_eui_ref_by_role": {
        "building_total": 220.0,
        "submeter": 260.0,
        "shared_total": 180.0,
        "unknown": 150.0,
        "feeder": 120.0,
        "campus_total": 100.0,
        "virtual": 180.0,
    },
    "manual_profiles": {
        "hospital": {
            "enabled": True,
            "name_keywords": ["醫院", "癌醫", "Hospital", "NTUH", "Children and Women"],
            "relax_night_to_day_high_to": 1.80,
            "eui_cap_override_by_role": {
                "building_total": 1800.0,
                "shared_total": 1200.0,
                "unknown": 1000.0,
                "feeder": 900.0,
                "virtual": 900.0,
            },
            "relative_eui_ref": 420.0,
        },
        "dorm": {
            "enabled": True,
            "name_keywords": ["宿舍", "學舍", "舍", "Dorm", "Prince House"],
            "relax_night_to_day_high_to": 1.60,
            "eui_cap_override_by_role": {
                "building_total": 350.0,
                "shared_total": 300.0,
                "unknown": 280.0,
                "feeder": 260.0,
                "virtual": 260.0,
            },
            "relative_eui_ref": 180.0,
        },
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_trust_policy(config_path: str | Path = "config/demo_config.yaml") -> dict:
    policy = copy.deepcopy(DEFAULT_TRUST_POLICY)
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = (_ROOT_DIR / cfg_path).resolve()
    if not cfg_path.exists():
        return policy
    try:
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return policy
    if not isinstance(data, dict):
        return policy

    if isinstance(data.get("trust_policy"), dict):
        policy = _deep_merge(policy, data["trust_policy"])
    if isinstance(data.get("manual_profiles"), dict):
        policy["manual_profiles"] = _deep_merge(
            policy.get("manual_profiles", {}),
            data["manual_profiles"],
        )
    return policy


def _detect_usage_profile(building_name: str, meter_name: str, policy: dict) -> str:
    haystack = f"{building_name or ''} {meter_name or ''}".lower()
    profiles = policy.get("manual_profiles", {})
    if not isinstance(profiles, dict):
        return "default"
    for profile_name, profile_cfg in profiles.items():
        if not isinstance(profile_cfg, dict):
            continue
        if not bool(profile_cfg.get("enabled", False)):
            continue
        for keyword in profile_cfg.get("name_keywords", []):
            kw = str(keyword or "").strip().lower()
            if kw and kw in haystack:
                return str(profile_name)
    return "default"

# ── Archetype 分類規則（基於 SHAP 分析）──────────────────
def _classify_archetype(mean_kw: float, r2: float) -> str:
    """依能耗強度 + 模型品質推斷 archetype (精細版)"""
    if mean_kw > 1000:
        return "Heavy-HVAC / Central Plant"
    elif mean_kw > 600:
        if r2 > 0.7:
            return "HVAC-dominant (Predictable)"
        else:
            return "HVAC-dominant (Volatile)"
    elif mean_kw > 300:
        if r2 > 0.6:
            return "Mixed-load (Schedule-driven)"
        else:
            return "Mixed-load (Complex)"
    elif r2 > 0.7:
        return "Lighting-dominant (Highly regular)"
    else:
        return "Baseload-driven (Irregular)"


def _estimate_virtual_kw(levels: int, footprint_area: float = 500.0) -> float:
    """粗略估算無電表建物的平均功率 (kW)"""
    # 假設每平方公尺平均負載為 0.05 kW (50 W/m²)，這是一個粗估值
    # 可以根據台大建築類型再微調
    est_floors = max(levels, 1)
    return max(10.0, est_floors * footprint_area * 0.03)


def _ring_area_m2(ring: list[list[float]]) -> float:
    """經緯度 ring 以局部平面近似計算面積 (m²)。"""
    if not ring or len(ring) < 4:
        return 0.0
    lat0 = float(np.mean([pt[1] for pt in ring]))
    m_per_deg_lat = 111132.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat0))
    xy = [(pt[0] * m_per_deg_lon, pt[1] * m_per_deg_lat) for pt in ring]

    area2 = 0.0
    for i in range(len(xy) - 1):
        x1, y1 = xy[i]
        x2, y2 = xy[i + 1]
        area2 += x1 * y2 - x2 * y1
    return abs(area2) * 0.5


def _geometry_footprint_m2(geometry: dict) -> float:
    """計算 Polygon / MultiPolygon 的 footprint 面積 (m²)。"""
    g_type = geometry.get("type")
    if g_type == "Polygon":
        rings = geometry.get("coordinates", [])
        if not rings:
            return 0.0
        outer = _ring_area_m2(rings[0])
        holes = sum(_ring_area_m2(r) for r in rings[1:])
        return max(0.0, outer - holes)

    if g_type == "MultiPolygon":
        total = 0.0
        for poly in geometry.get("coordinates", []):
            if not poly:
                continue
            outer = _ring_area_m2(poly[0])
            holes = sum(_ring_area_m2(r) for r in poly[1:])
            total += max(0.0, outer - holes)
        return total

    return 0.0


def _estimate_floor_area_m2(
    feature: dict,
    default_footprint_m2: float = 500.0,
    min_floor_area_m2: float = 1000.0,
) -> tuple[float, float, int]:
    """
    估算樓地板面積:
    1) 優先使用 OSM footprint 幾何面積
    2) 若幾何異常過小，回退至預設 footprint
    """
    props = feature.get("properties", {})
    levels_raw = props.get("levels", 3)
    try:
        est_floors = int(float(levels_raw))
    except (TypeError, ValueError):
        est_floors = 3
    est_floors = max(est_floors, 1)

    footprint_m2 = _geometry_footprint_m2(feature.get("geometry", {}))
    if footprint_m2 < 80.0:
        footprint_m2 = default_footprint_m2

    floor_area_m2 = max(footprint_m2 * est_floors, min_floor_area_m2)
    return floor_area_m2, footprint_m2, est_floors


def _estimate_load_factor(meter_role: str, r2: float = 0.5) -> float:
    """
    以角色 + 模型品質估算較合理的 load factor，避免全建物固定值。
    """
    base = {
        "building_total": 0.58,
        "submeter": 0.54,
        "shared_total": 0.56,
        "unknown": 0.50,
        "feeder": 0.45,
        "campus_total": 0.42,
        "virtual": 0.52,
    }.get(str(meter_role or "unknown"), 0.50)
    adj = float(np.clip((float(r2) - 0.5) * 0.15, -0.08, 0.08))
    return float(np.clip(base + adj, 0.35, 0.78))


def _apply_physical_sanity(
    mean_kw: float,
    floor_area_m2: float,
    meter_role: str,
    usage_profile: str,
    trust_policy: dict,
) -> tuple[float, float, float, list[str]]:
    """
    物理合理性修正:
    - 先把負值截斷
    - 對異常高 EUI 依 meter_role 設定上限，避免演示數值誇張
    """
    flags: list[str] = []
    kw_raw = max(float(mean_kw), 0.0)
    if float(mean_kw) < 0:
        flags.append("negative_kw_clipped")

    area = max(float(floor_area_m2), 1.0)
    eui_raw = kw_raw * 8760.0 / area
    kw_new = kw_raw

    role = str(meter_role or "unknown")
    caps_by_role = trust_policy.get("eui_caps_by_role", {})
    cap = _to_float(caps_by_role.get(role, caps_by_role.get("unknown", 550.0)), 550.0)

    profile_cfg = trust_policy.get("manual_profiles", {}).get(usage_profile, {})
    if isinstance(profile_cfg, dict):
        overrides = profile_cfg.get("eui_cap_override_by_role", {})
        if isinstance(overrides, dict) and (role in overrides):
            cap = _to_float(overrides.get(role), cap)
            flags.append(f"manual_profile_{usage_profile}")

    should_cap = (area >= 1000.0) or (str(meter_role or "") in {"shared_total", "unknown", "feeder", "campus_total", "virtual"})
    if should_cap and eui_raw > cap:
        kw_new = cap * area / 8760.0
        flags.append(f"eui_cap_{int(cap)}")

    eui_new = kw_new * 8760.0 / area
    return kw_new, eui_raw, eui_new, flags


# _to_float 已統一移至 src/utils.py


def _normalize_coverage_ratio(n_valid_hours: float, coverage_ratio: float) -> float:
    cov = _to_float(coverage_ratio, np.nan)
    if np.isfinite(cov):
        return float(np.clip(cov, 0.0, 1.0))
    n_valid = _to_float(n_valid_hours, np.nan)
    if np.isfinite(n_valid):
        return float(np.clip(n_valid / 8760.0, 0.0, 1.0))
    return float("nan")


def _should_hide_absolute_values(meter_role: str, coverage_ratio: float, trust_policy: dict) -> bool:
    role = str(meter_role or "unknown")
    hide_roles = set(str(x) for x in trust_policy.get("hide_absolute_roles", []))
    if role in hide_roles:
        return True
    hide_cov = _to_float(trust_policy.get("coverage", {}).get("hide_absolute_below", 0.5), 0.5)
    if np.isfinite(coverage_ratio) and coverage_ratio < hide_cov:
        return True
    return False


def _confidence_level(meter_role: str, match_score: float, coverage_ratio: float, trust_policy: dict) -> str:
    role = str(meter_role or "unknown")
    score = _to_float(match_score, 0.0)
    cov = _to_float(coverage_ratio, np.nan)
    conf_cfg = trust_policy.get("confidence", {})
    cov_cfg = trust_policy.get("coverage", {})

    high_roles = set(str(x) for x in conf_cfg.get("high_roles", []))
    medium_roles = set(str(x) for x in conf_cfg.get("medium_roles", []))
    high_score_min = _to_float(conf_cfg.get("high_score_min", 90.0), 90.0)
    medium_score_min = _to_float(conf_cfg.get("medium_score_min", 70.0), 70.0)
    cov_high_min = _to_float(cov_cfg.get("confidence_high_min", 0.8), 0.8)
    cov_medium_min = _to_float(cov_cfg.get("confidence_medium_min", 0.5), 0.5)

    if role in high_roles and score >= high_score_min and (not np.isfinite(cov) or cov >= cov_high_min):
        return "high"
    if role in medium_roles and score >= medium_score_min and (not np.isfinite(cov) or cov >= cov_medium_min):
        return "medium"
    return "low"


def _extra_physics_flags(
    mean_kw: float,
    eui: float,
    meter_role: str,
    usage_profile: str,
    coverage_ratio: float,
    night_to_day_ratio: float,
    peak_to_mean_ratio_p95: float,
    trust_policy: dict,
) -> list[str]:
    flags: list[str] = []
    role = str(meter_role or "unknown")
    cov_cfg = trust_policy.get("coverage", {})
    phy_cfg = trust_policy.get("physics", {})

    low_cov = _to_float(cov_cfg.get("low_flag_below", 0.5), 0.5)
    mid_cov = _to_float(cov_cfg.get("mid_flag_below", 0.7), 0.7)
    if np.isfinite(coverage_ratio) and coverage_ratio < low_cov:
        flags.append("low_coverage")
    if np.isfinite(coverage_ratio) and coverage_ratio < mid_cov:
        flags.append("mid_coverage")

    ndr = _to_float(night_to_day_ratio, np.nan)
    night_hi = _to_float(phy_cfg.get("night_to_day_high_default", 1.2), 1.2)
    profile_cfg = trust_policy.get("manual_profiles", {}).get(usage_profile, {})
    if isinstance(profile_cfg, dict):
        night_hi = _to_float(profile_cfg.get("relax_night_to_day_high_to", night_hi), night_hi)
    if np.isfinite(ndr) and ndr > night_hi:
        flags.append("night_baseload_high")

    pmr = _to_float(peak_to_mean_ratio_p95, np.nan)
    peak_hi = _to_float(phy_cfg.get("peak_ratio_high", 8.0), 8.0)
    peak_flat = _to_float(phy_cfg.get("peak_ratio_flat", 1.05), 1.05)
    if np.isfinite(pmr) and pmr > peak_hi:
        flags.append("peak_ratio_high")
    if np.isfinite(pmr) and pmr < peak_flat and _to_float(mean_kw, 0.0) > 100:
        flags.append("peak_ratio_too_flat")

    low_eui_cfg = phy_cfg.get("low_eui_kw_threshold", {})
    low_eui_below = _to_float(low_eui_cfg.get("eui_below", 5.0), 5.0)
    low_eui_kw_above = _to_float(low_eui_cfg.get("kw_above", 50.0), 50.0)
    if _to_float(eui, 0.0) < low_eui_below and _to_float(mean_kw, 0.0) > low_eui_kw_above:
        flags.append("eui_too_low_for_kw")
    return flags

# ── 色彩比例尺 (MapLibre interpolate expression) ──────────
# 柔和配色（暗底地圖上舒適閱讀）
EUI_COLOR_EXPR = [
    "interpolate", ["linear"], ["get", "eui"],
    0,   "#5e9e82",   # 低 → 柔綠
    80,  "#8bb86a",   # 中低 → 草綠
    150, "#ccb147",   # 中 → 暖金
    250, "#c17a50",   # 高 → 赤陶
    400, "#9e5578",   # 極高 → 玫粉
]

ENERGY_COLOR_EXPR = [
    "interpolate", ["linear"], ["get", "mean_kw"],
    0,    "#5e9e82",
    200,  "#8bb86a",
    500,  "#ccb147",
    1000, "#c17a50",
    2000, "#9e5578",
]

R2_COLOR_EXPR = [
    "interpolate", ["linear"], ["get", "best_r2_oof"],
    -0.5, "#c17a50",
    0.3,  "#ccb147",
    0.6,  "#8bb86a",
    0.8,  "#5e9e82",
    0.95, "#4eb88a",
]


# ── 合併 GeoJSON ──────────────────────────────────────────

# _weighted_mean 已統一移至 src/utils.py


def _detect_meter_role(meter_name: str, shared_group_size: int = 1) -> str:
    """
    根據電表名稱判斷屬性，避免「總表 + 饋線」重複加總。
    """
    n = str(meter_name or "").upper()

    # 校區層級總表（需同時具備站/區域特徵，避免誤判單棟）
    has_station_phrase = any(
        t in n for t in ["站總用電", "總站", "主站", "總配電", "全校", "總電源"]
    )
    zone_station = bool(re.search(r"(?:^|[^A-Z0-9])([一二三四五六七八九]|\d+)區", n)) and (
        "站" in n or "總用電" in n or "總電源" in n
    )
    if has_station_phrase or zone_station:
        return "campus_total"

    # 明確分表
    if any(t in n for t in ["分表", "子表", "SUBMETER"]):
        return "submeter"

    # 饋線通常是局部支路，不能和總表直接相加
    if any(t in n for t in ["饋線", "FEEDER"]):
        return "feeder"

    # 低可信（備援/測試）
    if any(t in n for t in ["備援", "備用", "TEST", "試驗"]):
        return "backup"

    total_tokens = [
        "總表", "總錶", "總用電", "總", "MAIN",
        "MCB", "GCB", "GCBM", "HTM", "VCB", "ACB",
    ]
    if any(t in n for t in total_tokens):
        if shared_group_size > 1:
            return "shared_total"
        return "building_total"

    if shared_group_size > 1:
        return "shared_total"
    return "unknown"


def _meter_role_priority(role: str) -> int:
    # 數字越大越優先
    priority = {
        "building_total": 60,
        "submeter": 50,
        "feeder": 40,
        "shared_total": 30,
        "unknown": 20,
        "backup": 10,
        "campus_total": 0,
    }
    return priority.get(role, 20)


def _select_rows_for_building(group: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    """
    回傳 (selected_rows, selected_role, aggregation_method)
    """
    g = group.copy()
    g["_mean_kw"] = pd.to_numeric(g["mean_kw"], errors="coerce").fillna(0.0)
    g["_role"] = g["meter_role"].fillna("unknown")
    g["_priority"] = g["_role"].map(_meter_role_priority)

    # 優先排除 campus_total；除非整組只有 campus_total
    non_campus = g[g["_role"] != "campus_total"]
    if not non_campus.empty:
        g = non_campus

    top_priority = int(g["_priority"].max()) if not g.empty else 0
    top = g[g["_priority"] == top_priority]
    if top.empty:
        top = g

    selected_role = str(top["_role"].iloc[0]) if not top.empty else "unknown"

    # 只有明確分表時才允許加總；其餘一律取最大值，避免重複計算
    # Conservative policy: do not auto-sum multiple submeters.
    # Some independent circuit submeters should not be merged.
    best = top.sort_values(["_mean_kw", "match_score"], ascending=False).head(1)
    if selected_role == "submeter" and len(top) > 1:
        return best, selected_role, "max_single_submeter_no_merge"
    return best, selected_role, "max_single_meter"


def _aggregate_meter_group(group: pd.DataFrame) -> dict:
    status_rank = {"auto_matched": 2, "needs_review": 1, "unmatched": 0}
    ranked = group.assign(
        _status_rank=group["match_status"].map(status_rank).fillna(0),
        _score=pd.to_numeric(group["match_score"], errors="coerce").fillna(0),
        _mean_kw=pd.to_numeric(group["mean_kw"], errors="coerce").fillna(0),
    ).sort_values(["_status_rank", "_score", "_mean_kw"], ascending=False)

    selected_rows, selected_role, agg_method = _select_rows_for_building(ranked)
    best = selected_rows.iloc[0] if not selected_rows.empty else ranked.iloc[0]
    mean_kw_values = pd.to_numeric(selected_rows["mean_kw"], errors="coerce").fillna(0.0)

    meter_names = [
        str(name).strip()
        for name in selected_rows["meter_name"].fillna("")
        if str(name).strip()
    ]
    unique_meter_names = list(dict.fromkeys(meter_names))

    selected_mean_kw = float(mean_kw_values.max()) if len(mean_kw_values) else 0.0
    best_n_valid = float(pd.to_numeric(selected_rows.get("n_valid_hours", pd.Series([np.nan])), errors="coerce").max())
    best_cov = float(pd.to_numeric(selected_rows.get("coverage_ratio", pd.Series([np.nan])), errors="coerce").max())
    best_zero_ratio = float(pd.to_numeric(selected_rows.get("zero_ratio_valid", pd.Series([np.nan])), errors="coerce").max())
    best_night_to_day = float(pd.to_numeric(selected_rows.get("night_to_day_ratio", pd.Series([np.nan])), errors="coerce").max())
    best_peak_to_mean = float(pd.to_numeric(selected_rows.get("peak_to_mean_ratio_p95", pd.Series([np.nan])), errors="coerce").max())
    best_r2_corr = float(pd.to_numeric(selected_rows.get("best_r2_from_corr_sq", pd.Series([np.nan])), errors="coerce").max())

    return {
        "osm_name": str(best.get("osm_name", "") or ""),
        "osm_id": best.get("osm_id"),
        "meter_name": " | ".join(unique_meter_names),
        "meter_count": int(len(selected_rows)),
        "candidate_meter_count": int(len(ranked)),
        "meter_role": selected_role,
        "aggregation_method": agg_method,
        "split_policy": str(best.get("split_policy", "")),
        "mass_balance_applied": bool(best.get("mass_balance_applied", False)),
        "match_status": str(best.get("match_status", "")),
        "match_score": float(pd.to_numeric(selected_rows["match_score"], errors="coerce").max()),
        "mean_kw": selected_mean_kw,
        "best_r2_oof": _weighted_mean(selected_rows["best_r2_oof"], mean_kw_values),
        "best_r2_from_corr_sq": best_r2_corr,
        "best_r_oof": _weighted_mean(selected_rows["best_r_oof"], mean_kw_values),
        "best_cvrmse_oof": _weighted_mean(selected_rows["best_cvrmse_oof"], mean_kw_values),
        "n_valid_hours": best_n_valid,
        "coverage_ratio": best_cov,
        "zero_ratio_valid": best_zero_ratio,
        "night_to_day_ratio": best_night_to_day,
        "peak_to_mean_ratio_p95": best_peak_to_mean,
    }


def _build_meter_lookup(match_df: pd.DataFrame) -> tuple[dict[int, dict], dict[str, dict]]:
    matched = match_df[match_df["match_status"] != "unmatched"].copy()
    if matched.empty:
        return {}, {}

    matched["osm_name"] = matched["osm_name"].fillna("").astype(str).str.strip()
    matched["osm_id_num"] = pd.to_numeric(matched["osm_id"], errors="coerce")

    by_id: dict[int, dict] = {}
    by_name: dict[str, dict] = {}

    with_id = matched[matched["osm_id_num"].notna()]
    for osm_id, group in with_id.groupby("osm_id_num"):
        agg = _aggregate_meter_group(group)
        by_id[int(osm_id)] = agg
        if agg["osm_name"] and agg["osm_name"] not in by_name:
            by_name[agg["osm_name"]] = agg

    without_id = matched[(matched["osm_id_num"].isna()) & (matched["osm_name"] != "")]
    for osm_name, group in without_id.groupby("osm_name"):
        by_name[osm_name] = _aggregate_meter_group(group)

    return by_id, by_name


def merge_energy_geojson(
    geojson_path: str | Path,
    match_csv_path: str | Path,
    output_path: str | Path = "data/NTU/ntu_energy.geojson",
    enable_virtual_estimation: bool = False,
    config_path: str | Path = "config/demo_config.yaml",
) -> dict:
    """
    合併 OSM 建物 GeoJSON + 電表匹配結果 → 帶完整能耗屬性的 GeoJSON。

    新屬性欄位（對齊 building_schema）：
      mean_kw, annual_kwh, annual_mwh, peak_kw, load_factor,
      eui, trend_slope, best_r2_oof, best_r_oof, best_cvrmse_oof,
      archetype_label, data_source, meter_name
    """
    trust_policy = _load_trust_policy(config_path=config_path)
    geojson_path = resolve_project_path(geojson_path)
    match_csv_path = resolve_project_path(match_csv_path)
    output_path = resolve_project_path(output_path)

    with open(geojson_path, encoding="utf-8") as f:
        geojson = json.load(f)

    match_df = pd.read_csv(match_csv_path)
    official_name_set: set[str] = set()
    for official_path in [
        _DATA_DIR / "ntu_official_buildings_patch.csv",
        _DATA_DIR / "ntu_official_buildings.csv",
    ]:
        if not official_path.exists():
            continue
        for enc in ("utf-8-sig", "utf-8", "cp950", "big5"):
            try:
                off_df = pd.read_csv(official_path, encoding=enc)
                if "Name_ZH" in off_df.columns:
                    official_name_set.update(
                        str(x).strip() for x in off_df["Name_ZH"].dropna().tolist() if str(x).strip()
                    )
                if "Alias_ZH" in off_df.columns:
                    for v in off_df["Alias_ZH"].dropna().tolist():
                        for alias in str(v).split("|"):
                            s = alias.strip()
                            if s:
                                official_name_set.add(s)
                break
            except (ValueError, TypeError):
                continue

    # 1. 預先計算所有 OSM 建物的樓地板面積（幾何面積優先）
    building_areas = {}
    for feature in geojson["features"]:
        props = feature["properties"]
        name = props.get("name", "")
        if name:
            floor_area_m2, _, _ = _estimate_floor_area_m2(feature)
            building_areas[name] = floor_area_m2

    # 2. 展開共享電表的 osm_name，並計算每個原始電表涵蓋的「總樓地板面積」
    expanded_rows = []
    meter_total_area = {}
    
    for _, row in match_df.iterrows():
        osm_name_str = str(row.get("osm_name", ""))
        m_name = str(row.get("meter_name", ""))
        role_from_matcher = str(row.get("meter_role", "") or "").strip()
        sub_names = [s.strip() for s in osm_name_str.split("|") if s.strip()]

        if not sub_names:
            new_row = row.copy()
            new_row["shared_group_size"] = 1
            new_row["alloc_ratio"] = 1.0
            new_row["split_policy"] = "single"
            new_row["mass_balance_applied"] = False
            new_row["meter_role"] = role_from_matcher or _detect_meter_role(m_name, 1)
            expanded_rows.append(new_row)
            continue

        row_role = role_from_matcher or _detect_meter_role(m_name, len(sub_names))
        can_mass_balance = row_role in {"building_total", "shared_total"}

        # No reliable total meter => avoid hard split.
        if (len(sub_names) > 1) and (not can_mass_balance):
            best_name = max(sub_names, key=lambda n: building_areas.get(n, 0.0))
            new_row = row.copy()
            new_row["osm_name"] = best_name
            new_row["shared_group_size"] = 1
            new_row["alloc_ratio"] = 1.0
            new_row["split_policy"] = "no_split_without_total"
            new_row["mass_balance_applied"] = False
            new_row["meter_role"] = row_role
            expanded_rows.append(new_row)
            continue

        for sub_name in sub_names:
            new_row = row.copy()
            new_row["osm_name"] = sub_name
            new_row["shared_group_size"] = len(sub_names)
            new_row["split_policy"] = "mass_balance_split" if len(sub_names) > 1 else "single"
            new_row["mass_balance_applied"] = bool(len(sub_names) > 1)
            new_row["meter_role"] = row_role
            expanded_rows.append(new_row)
            if (len(sub_names) > 1) and (sub_name in building_areas):
                meter_total_area[m_name] = meter_total_area.get(m_name, 0.0) + building_areas[sub_name]

    # 3. Allocate kW by mass balance only when a reliable total meter exists.
    for r_idx in range(len(expanded_rows)):
        row = expanded_rows[r_idx]
        osm_name = row["osm_name"]
        m_name = row["meter_name"]
        split_policy = str(row.get("split_policy", "single"))

        ratio = 1.0
        if split_policy == "mass_balance_split":
            b_area = building_areas.get(osm_name, 0.0)
            m_area = meter_total_area.get(m_name, 0.0)
            if m_area > 0 and b_area > 0:
                ratio = b_area / m_area

        try:
            old_kw = float(row.get("mean_kw", 0))
        except ValueError:
            old_kw = 0.0

        expanded_rows[r_idx]["mean_kw"] = old_kw * ratio
        expanded_rows[r_idx]["is_shared"] = bool(split_policy == "mass_balance_split")
        expanded_rows[r_idx]["alloc_ratio"] = ratio

        group_size = int(expanded_rows[r_idx].get("shared_group_size", 1) or 1)
        role_from_matcher = str(expanded_rows[r_idx].get("meter_role", "") or "").strip()
        expanded_rows[r_idx]["meter_role"] = role_from_matcher or _detect_meter_role(m_name, group_size)

    match_df = pd.DataFrame(expanded_rows)

    # 4. _build_meter_lookup 會把多個獨立或預先分配好權重的 sub_row 加總進同一棟建物
    meter_lookup_by_id, meter_lookup_by_name = _build_meter_lookup(match_df)

    # 5. 最後掃描一遍 geojson 完成組裝
    n_enriched = 0
    enriched_features = []
    anomaly_rows: list[dict] = []
    
    for feature in geojson["features"]:
        props = feature["properties"]
        name = props.get("name", "")
        osm_id = props.get("osm_id")

        match = None
        try:
            if osm_id is not None:
                match = meter_lookup_by_id.get(int(osm_id))
        except (TypeError, ValueError):
            match = None
        if match is None and name:
            match = meter_lookup_by_name.get(name)

        if match is not None:
            m = match
            mean_kw_raw = _to_float(m.get("mean_kw", 0), 0.0)
            meter_role = str(m.get("meter_role", "unknown") or "unknown")
            usage_profile = _detect_usage_profile(name, str(m.get("meter_name", "")), trust_policy)

            r2_strict_raw = _to_float(m.get("best_r2_oof", np.nan), np.nan)
            r2_from_corr_sq = _to_float(m.get("best_r2_from_corr_sq", np.nan), np.nan)
            if np.isfinite(r2_strict_raw):
                r2 = float(np.clip(r2_strict_raw, -1.0, 1.0))
            elif np.isfinite(r2_from_corr_sq):
                r2 = float(np.clip(r2_from_corr_sq, 0.0, 1.0))
            else:
                r2 = 0.0

            n_valid_hours = _to_float(m.get("n_valid_hours", np.nan), np.nan)
            coverage_ratio = _normalize_coverage_ratio(n_valid_hours, m.get("coverage_ratio", np.nan))
            zero_ratio_valid = _to_float(m.get("zero_ratio_valid", np.nan), np.nan)
            night_to_day_ratio = _to_float(m.get("night_to_day_ratio", np.nan), np.nan)
            peak_to_mean_ratio_p95 = _to_float(m.get("peak_to_mean_ratio_p95", np.nan), np.nan)

            this_area, footprint_area, est_floors = _estimate_floor_area_m2(feature)
            mean_kw, eui_raw, eui, sanity_flags = _apply_physical_sanity(
                mean_kw=mean_kw_raw,
                floor_area_m2=this_area,
                meter_role=meter_role,
                usage_profile=usage_profile,
                trust_policy=trust_policy,
            )
            annual_kwh = mean_kw * 8760.0

            warning_flags = []
            if meter_role in {"feeder", "campus_total"}:
                warning_flags.append("role_low_confidence")
            if float(m.get("match_score", 0) or 0) < 90:
                warning_flags.append("match_score_lt90")
            if int(m.get("candidate_meter_count", 1) or 1) > 1 and str(m.get("aggregation_method", "")).startswith("max_single"):
                warning_flags.append("multi_meter_not_merged")
            if meter_role in {"shared_total", "unknown", "feeder", "campus_total", "backup"}:
                warning_flags.append("inferred_not_direct_submeter")
            r2_gap_thr = _to_float(trust_policy.get("r2_definition_gap_flag", 0.25), 0.25)
            if np.isfinite(r2_from_corr_sq) and np.isfinite(r2) and abs(r2 - r2_from_corr_sq) > r2_gap_thr:
                warning_flags.append("r2_definition_gap")

            extra_flags = _extra_physics_flags(
                mean_kw=mean_kw,
                eui=eui,
                meter_role=meter_role,
                usage_profile=usage_profile,
                coverage_ratio=coverage_ratio,
                night_to_day_ratio=night_to_day_ratio,
                peak_to_mean_ratio_p95=peak_to_mean_ratio_p95,
                trust_policy=trust_policy,
            )
            all_flags = warning_flags + sanity_flags + extra_flags

            load_factor = _estimate_load_factor(meter_role, r2=r2)
            peak_kw = mean_kw / load_factor if load_factor > 0 else mean_kw * 1.6

            hide_absolute = _should_hide_absolute_values(meter_role, coverage_ratio, trust_policy)
            confidence = _confidence_level(meter_role, m.get("match_score", 0), coverage_ratio, trust_policy)

            role_ref_eui = trust_policy.get("relative_eui_ref_by_role", {})
            rel_ref = _to_float(role_ref_eui.get(meter_role, role_ref_eui.get("unknown", 150.0)), 150.0)
            profile_cfg = trust_policy.get("manual_profiles", {}).get(usage_profile, {})
            if isinstance(profile_cfg, dict):
                rel_ref = _to_float(profile_cfg.get("relative_eui_ref", rel_ref), rel_ref)
            rel_ref = max(rel_ref, 1.0)
            rel_eui_idx = float(np.clip(eui / rel_ref, 0.0, 3.0))

            props["mean_kw_raw"] = round(mean_kw_raw, 1)
            props["mean_kw"] = round(mean_kw, 1)
            props["annual_kwh_raw"] = round(annual_kwh, 0)
            props["annual_mwh_raw"] = round(annual_kwh / 1000.0, 1)
            if hide_absolute:
                props["annual_kwh"] = None
                props["annual_mwh"] = None
                props["peak_kw"] = None
            else:
                props["annual_kwh"] = round(annual_kwh, 0)
                props["annual_mwh"] = round(annual_kwh / 1000.0, 1)
                props["peak_kw"] = round(peak_kw, 1)

            props["absolute_values_visible"] = not hide_absolute
            props["load_factor"] = round(load_factor, 3)
            props["eui_raw"] = round(eui_raw, 1)
            props["eui"] = round(eui, 1)
            props["relative_energy_index"] = round(rel_eui_idx, 3)
            props["floor_area_m2"] = round(this_area, 1)
            props["footprint_area_m2"] = round(footprint_area, 1)
            props["trend_slope"] = 0.0
            props["best_r2_oof"] = round(r2, 4)
            props["best_r2_from_corr_sq"] = round(r2_from_corr_sq, 4) if np.isfinite(r2_from_corr_sq) else None
            props["r2_definition_gap"] = round(abs(r2 - r2_from_corr_sq), 4) if (np.isfinite(r2_from_corr_sq) and np.isfinite(r2)) else None
            props["best_r_oof"] = round(_to_float(m.get("best_r_oof", np.nan), 0.0), 4)
            props["best_cvrmse_oof"] = round(_to_float(m.get("best_cvrmse_oof", np.nan), 0.0), 1)
            props["n_valid_hours"] = int(n_valid_hours) if np.isfinite(n_valid_hours) else None
            props["coverage_ratio"] = round(coverage_ratio, 4) if np.isfinite(coverage_ratio) else None
            props["zero_ratio_valid"] = round(zero_ratio_valid, 4) if np.isfinite(zero_ratio_valid) else None
            props["night_to_day_ratio"] = round(night_to_day_ratio, 4) if np.isfinite(night_to_day_ratio) else None
            props["peak_to_mean_ratio_p95"] = round(peak_to_mean_ratio_p95, 4) if np.isfinite(peak_to_mean_ratio_p95) else None
            props["meter_role"] = meter_role
            props["aggregation_method"] = m.get("aggregation_method", "max_single_meter")
            props["split_policy"] = m.get("split_policy", "")
            props["mass_balance_applied"] = bool(m.get("mass_balance_applied", False))
            props["usage_profile"] = usage_profile
            props["physics_flags"] = ";".join(all_flags)
            props["physics_corrected"] = bool(sanity_flags)

            is_shared = m.get("is_shared", False)
            disp_meter = str(m.get("meter_name", ""))
            if bool(is_shared) or "|" in disp_meter:
                disp_meter += " (shared meter)"

            props["meter_name"] = disp_meter
            props["meter_count"] = int(m.get("meter_count", 1))
            props["data_source"] = "Measured Meter"
            if meter_role in {"shared_total", "unknown", "feeder", "campus_total", "backup"} or hide_absolute:
                props["data_source"] = "PI-VD Inferred"
            props["confidence_level"] = confidence
            props["archetype_label"] = _classify_archetype(mean_kw, r2)
            props["has_meter_data"] = True

            if all_flags:
                anomaly_rows.append(
                    {
                        "name": name,
                        "meter_name": disp_meter,
                        "meter_role": meter_role,
                        "match_score": float(m.get("match_score", 0) or 0),
                        "aggregation_method": str(m.get("aggregation_method", "")),
                        "split_policy": str(m.get("split_policy", "")),
                        "usage_profile": usage_profile,
                        "coverage_ratio": round(coverage_ratio, 4) if np.isfinite(coverage_ratio) else None,
                        "confidence_level": confidence,
                        "mean_kw_raw": round(mean_kw_raw, 3),
                        "mean_kw_corrected": round(mean_kw, 3),
                        "eui_raw": round(eui_raw, 3),
                        "eui_corrected": round(eui, 3),
                        "floor_area_m2": round(this_area, 3),
                        "flags": ";".join(all_flags),
                    }
                )

            props["height"] = est_floors * 4.0
            props["min_height"] = 0.0
            n_enriched += 1
            enriched_features.append(feature)
        else:
            name_clean = str(name or "").strip()
            # 保留所有校園範圍內的建物（已由 osm_fetcher 的 BBox + 類型篩選限定）
            # 無名稱的建物給予預設標籤
            if not name_clean:
                name_clean = f"Building_{props.get('osm_id', 'unknown')}"
                props["name"] = name_clean

            est_floor_area, footprint_area, est_floors = _estimate_floor_area_m2(feature)
            usage_profile = _detect_usage_profile(name_clean, "Virtual_Meter", trust_policy)

            if enable_virtual_estimation:
                # 可選模式: 使用虛擬推估
                mean_kw_est_raw = _estimate_virtual_kw(est_floors, footprint_area)
                mean_kw_est, eui_raw, eui_est, sanity_flags = _apply_physical_sanity(
                    mean_kw=mean_kw_est_raw,
                    floor_area_m2=est_floor_area,
                    meter_role="virtual",
                    usage_profile=usage_profile,
                    trust_policy=trust_policy,
                )
                annual_kwh_est = mean_kw_est * 8760.0
                load_factor = _estimate_load_factor("virtual", r2=0.5)
                peak_kw = mean_kw_est / load_factor if load_factor > 0 else mean_kw_est * 1.5

                props["mean_kw_raw"]     = round(mean_kw_est_raw, 1)
                props["mean_kw"]         = round(mean_kw_est, 1)
                props["annual_kwh"]      = round(annual_kwh_est, 0)
                props["annual_mwh"]      = round(annual_kwh_est / 1000, 1)
                props["peak_kw"]         = round(peak_kw, 1)
                props["load_factor"]     = round(load_factor, 3)
                props["eui_raw"]         = round(eui_raw, 1)
                props["eui"]             = round(eui_est, 1)
                props["best_r2_oof"]     = 0.5
                props["best_r_oof"]      = 0.7
                props["best_cvrmse_oof"] = 30.0
                props["meter_name"]      = "Virtual_Meter"
                props["meter_count"]     = 0
                props["meter_role"]      = "virtual"
                props["aggregation_method"] = "estimated_virtual"
                props["split_policy"]    = "estimated_virtual"
                props["mass_balance_applied"] = False
                props["usage_profile"]   = usage_profile
                props["data_source"]     = "PI-VD 虛擬推估"
                props["confidence_level"] = "low"
                props["coverage_ratio"]   = None
                props["absolute_values_visible"] = False
                props["archetype_label"] = _classify_archetype(mean_kw_est, 0.5) + " (Estimated)"
                props["physics_flags"]   = ";".join(sanity_flags)
                props["physics_corrected"] = bool(sanity_flags)
                props["has_meter_data"]  = False
            else:
                # 預設模式: 無資料不估算，避免「看起來像真實量測」的誤導
                props["mean_kw_raw"]     = 0.0
                props["mean_kw"]         = 0.0
                props["annual_kwh"]      = 0.0
                props["annual_mwh"]      = 0.0
                props["peak_kw"]         = 0.0
                props["load_factor"]     = 0.0
                props["eui_raw"]         = 0.0
                props["eui"]             = 0.0
                props["best_r2_oof"]     = 0.0
                props["best_r_oof"]      = 0.0
                props["best_cvrmse_oof"] = 0.0
                props["meter_name"]      = ""
                props["meter_count"]     = 0
                props["meter_role"]      = "unmetered"
                props["aggregation_method"] = "none"
                props["split_policy"]    = "none"
                props["mass_balance_applied"] = False
                props["usage_profile"]   = usage_profile
                props["data_source"]     = "無電表資料 (未估算)"
                props["confidence_level"] = "none"
                props["coverage_ratio"]   = None
                props["absolute_values_visible"] = False
                props["archetype_label"] = "Unmetered"
                props["physics_flags"]   = "no_meter_data"
                props["physics_corrected"] = False
                props["has_meter_data"]  = False

            props["floor_area_m2"]   = round(est_floor_area, 1)
            props["footprint_area_m2"] = round(footprint_area, 1)
            props["height"]          = est_floors * 4.0
            props["min_height"]      = 0.0
            n_enriched += 1
            enriched_features.append(feature)
            
    # 只保留有成功匹配到電表的建築多邊形，或是符合關鍵字的校內無表建築
    geojson["features"] = enriched_features
            
    print("[MapBuilder] 合併完成: 已標記實測分表與無資料建物")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)
    print(f"[MapBuilder] 已儲存: {output_path}")

    report_path = output_path.with_name("physics_anomaly_report.csv")
    if anomaly_rows:
        pd.DataFrame(anomaly_rows).to_csv(report_path, index=False, encoding="utf-8-sig")
        print(f"[MapBuilder] 異常報告: {report_path} ({len(anomaly_rows)} 筆)")
    else:
        print("[MapBuilder] 異常報告: 無需修正")

    return geojson


# ── 輔助：從 GeoJSON 產生電表點位清單 ─────────────────────

def _is_inferred_source(data_source: str) -> bool:
    s = str(data_source or "").strip().lower()
    if not s:
        return False
    return (
        s == "inferred"
        or "inferred" in s
        or "推估" in s
        or "virtual" in s
        or "pivd_estimate" in s     # NCU per-year cache: PIVD physics fallback
    )


def _is_metered_source(data_source: str) -> bool:
    s = str(data_source or "").strip().lower()
    if not s:
        return False
    return (
        s == "metered"
        or "measured" in s
        or "實測" in s
        or "ncu_real" in s          # NCU per-year cache: real meter readings
    )


def _extract_meter_points(
    energy_geojson: dict,
    selected_meter: str | None = None,
    show_virtual: bool = True,
) -> list[dict]:
    """提取點位 (加入 show_virtual 控制)"""
    points = []
    selected_meter = (selected_meter or "").strip()
    for feat in energy_geojson["features"]:
        p = feat["properties"]
        source = p.get("data_source", "")
        is_metered = bool(p.get("has_meter_data", False)) or _is_metered_source(source)
        is_virtual = _is_inferred_source(source)
        
        if not (is_metered or (show_virtual and is_virtual)):
            continue
            
        # 取 centroid
        geom = feat["geometry"]
        if geom["type"] == "Polygon":
            coords = geom["coordinates"][0]
        elif geom["type"] == "MultiPolygon":
            coords = geom["coordinates"][0][0]
        else:
            continue
        lon = np.mean([c[0] for c in coords])
        lat = np.mean([c[1] for c in coords])
        
        mean_kw = p.get("mean_kw", 0) or 0
        meter_name = str(p.get("meter_name", "") or "")
        is_selected = bool(selected_meter) and (selected_meter in meter_name)
        
        # 虛擬電表給稍微不同的透明度
        base_color = _kw_to_rgb(mean_kw)
        if is_virtual:
            base_color[3] = 150 # 虛擬點位稍微透明一點
            
        point_color = [100, 200, 230, 220] if is_selected else base_color
        point_radius = max(15, min(110, mean_kw / 10))
        if is_selected:
            point_radius = min(140, point_radius * 1.6)

        annual_kwh = p.get("annual_kwh", None)
        if annual_kwh is None:
            annual_kwh = np.nan
            
        points.append({
            "position": [float(lon), float(lat)],
            "radius":   point_radius,
            "color":    point_color,
            "name":     p.get("name", ""),
            "meter_name": meter_name,
            "mean_kw":  float(mean_kw),
            "annual_kwh": float(annual_kwh) if not pd.isna(annual_kwh) else np.nan,
            "selected": is_selected,
            "label": f"{meter_name}\n{float(mean_kw):,.1f} kW" if is_selected else "",
        })
    return points


def _kw_to_rgb(kw: float) -> list[int]:
    """mean_kw → RGBA color（柔和色調，暗底圖舒適）"""
    if kw < 200:
        return [94, 158, 130, 190]      # sage green
    elif kw < 500:
        return [139, 184, 106, 200]     # soft lime
    elif kw < 1000:
        return [204, 177, 71, 205]      # warm gold
    elif kw < 2000:
        return [193, 122, 80, 210]      # terracotta
    else:
        return [158, 85, 120, 215]      # dusty mauve


def _eui_to_rgb(eui: float) -> list[int]:
    if eui < 80:
        return [94, 158, 130, 190]
    elif eui < 150:
        return [139, 184, 106, 200]
    elif eui < 250:
        return [204, 177, 71, 205]
    elif eui < 400:
        return [193, 122, 80, 210]
    return [158, 85, 120, 215]


def _r2_to_rgb(r2: float) -> list[int]:
    if r2 < 0.3:
        return [193, 122, 80, 210]
    elif r2 < 0.6:
        return [204, 177, 71, 205]
    elif r2 < 0.8:
        return [139, 184, 106, 200]
    elif r2 < 0.95:
        return [94, 158, 130, 200]
    return [78, 184, 138, 210]


def _dci_to_rgb(dci: float) -> list[int]:
    """Map Deployment Confidence Index (0-100) to RGBA."""
    x = float(np.clip(dci, 0.0, 100.0))
    if x < 40:
        return [193, 122, 80, 210]   # low: terracotta
    elif x < 70:
        return [204, 177, 71, 210]   # medium: warm gold
    return [94, 158, 130, 210]       # high: sage green


def _tier_to_rgb(tier: str) -> list[int]:
    key = str(tier or "").strip().upper()
    if key == "HIGH":
        return [215, 48, 39, 220]
    if key == "LOW":
        return [26, 152, 80, 220]
    return [240, 196, 25, 220]


def _compute_dci_score(coverage_ratio: float, deployment_days: int) -> float:
    """System-level confidence score, separate from model R2."""
    days = int(np.clip(deployment_days, 0, 30))
    cv_rmse = 120.0 * math.exp(-0.15 * float(days))
    day_score = float(np.clip(days / 30.0, 0.0, 1.0))
    cvrmse_score = float(np.clip(1.0 - (cv_rmse / 80.0), 0.0, 1.0))
    coverage_score = float(np.clip(coverage_ratio, 0.0, 1.0))
    dci = 100.0 * (0.50 * cvrmse_score + 0.30 * day_score + 0.20 * coverage_score)
    return float(np.clip(dci, 0.0, 100.0))


def _apply_saturation(rgba: list[int], saturation_scale: float) -> list[int]:
    """Blend color toward grayscale when saturation_scale < 1."""
    if not isinstance(rgba, list) or len(rgba) < 4:
        return rgba
    s = float(np.clip(saturation_scale, 0.0, 1.0))
    r, g, b, a = [int(v) for v in rgba[:4]]
    gray = (r + g + b) / 3.0
    r2 = int(np.clip(round(gray + (r - gray) * s), 0, 255))
    g2 = int(np.clip(round(gray + (g - gray) * s), 0, 255))
    b2 = int(np.clip(round(gray + (b - gray) * s), 0, 255))
    a2 = int(np.clip(round(a * (0.55 + 0.45 * s)), 0, 255))
    return [r2, g2, b2, a2]


def _decorate_visual_properties(
    geojson: dict,
    show_virtual: bool = True,
    saturation_scale: float = 1.0,
    deployment_days: int = 30,
) -> None:
    """把著色/外框樣式先寫入 properties，讓 anymap-ts 直接取用。"""
    for feature in geojson["features"]:
        props = feature["properties"]
        source = props.get("data_source", "")
        is_loading = str(source or "").strip().lower() == "loading"
        is_metered = bool(props.get("has_meter_data", False)) or _is_metered_source(source)
        is_virtual = _is_inferred_source(source)
        
        # 決定是否要上色
        should_color = (not is_loading) and (is_metered or (show_virtual and is_virtual))
        raw_height = _to_float(props.get("height", np.nan), np.nan)
        if not np.isfinite(raw_height):
            raw_height = max(_to_float(props.get("levels", 0), 0.0) * 3.5, 6.0)
        raw_height = float(max(raw_height, 6.0))

        mean_kw = float(props.get("mean_kw") or 0.0)
        eui = float(props.get("eui") or 0.0)
        r2 = float(props.get("best_r2_oof") or 0.0)
        cov = _to_float(props.get("coverage_ratio", np.nan), np.nan)
        if not np.isfinite(cov):
            cov = 0.60
        dci = _compute_dci_score(float(cov), int(deployment_days))
        props["dci_score"] = round(float(dci), 1)
        if dci >= 75:
            props["dci_level"] = "high"
        elif dci >= 50:
            props["dci_level"] = "medium"
        else:
            props["dci_level"] = "low"

        if should_color:
            props["fill_color_energy"] = _kw_to_rgb(mean_kw)
            props["fill_color_eui"] = _eui_to_rgb(eui)
            props["fill_color_r2"] = _r2_to_rgb(r2)
            props["fill_color_dci"] = _dci_to_rgb(dci)
            props["fill_color_tier"] = _tier_to_rgb(props.get("energy_tier", "NORMAL"))
            props["display_height"] = raw_height
            
            # 視覺區分 (因應底圖變白，邊框顏色要稍微加深以凸顯)
            if is_metered:
                props["outline_color"] = [40, 120, 180, 220]   # Deep sky blue
                props["outline_width"] = 2.4
                props["outline_dash_array"] = [1, 0]
            else:
                props["outline_color"] = [180, 120, 20, 200]   # Deep orange/sand
                props["outline_width"] = 2.2
                props["outline_dash_array"] = [4, 3]
        else:
            muted = [210, 215, 220, 110]  # 淡灰色 (適合淺色底圖)
            props["fill_color_energy"] = muted
            props["fill_color_eui"] = muted
            props["fill_color_r2"] = muted
            props["fill_color_dci"] = muted
            props["fill_color_tier"] = muted
            props["outline_color"] = [160, 165, 170, 180]
            props["outline_width"] = 0.5
            props["outline_dash_array"] = [1, 0]
            props["fill_color_energy"] = [223, 227, 232, 62]
            props["fill_color_eui"] = [223, 227, 232, 62]
            props["fill_color_r2"] = [223, 227, 232, 62]
            props["fill_color_dci"] = [223, 227, 232, 62]
            props["fill_color_tier"] = [223, 227, 232, 62]
            props["outline_color"] = [185, 190, 196, 120]
            props["outline_width"] = 0.35
            props["display_height"] = raw_height

        if saturation_scale < 0.999:
            props["fill_color_energy"] = _apply_saturation(props["fill_color_energy"], saturation_scale)
            props["fill_color_eui"] = _apply_saturation(props["fill_color_eui"], saturation_scale)
            props["fill_color_r2"] = _apply_saturation(props["fill_color_r2"], saturation_scale)
            props["fill_color_dci"] = _apply_saturation(props["fill_color_dci"], saturation_scale)
            props["fill_color_tier"] = _apply_saturation(props["fill_color_tier"], saturation_scale)


def _build_focus_marker_layer(focus_marker: dict | None) -> pdk.Layer | None:
    if not isinstance(focus_marker, dict):
        return None

    lon = _to_float(focus_marker.get("lon"), np.nan)
    lat = _to_float(focus_marker.get("lat"), np.nan)
    if not (np.isfinite(lon) and np.isfinite(lat)):
        return None

    marker_data = [{
        "position": [float(lon), float(lat)],
        "icon": "pin",
        "size": 44,
        "pixel_offset": [0, -10],
    }]

    return pdk.Layer(
        "IconLayer",
        marker_data,
        get_icon="icon",
        get_position="position",
        get_pixel_offset="pixel_offset",
        get_size="size",
        size_units=String("pixels"),
        size_scale=1,
        size_min_pixels=32,
        size_max_pixels=72,
        icon_atlas=String(_FOCUS_MARKER_ATLAS),
        icon_mapping=_FOCUS_MARKER_MAPPING,
        billboard=True,
        pickable=False,
        parameters={"depthTest": False},
    )


# ── 建構地圖 ──────────────────────────────────────────────

def build_campus_map(
    energy_geojson_data: dict | str | Path,
    color_by: str = "energy",     # "energy" | "eui" | "r2" | "dci" | "tier"
    selected_meter: str | None = None,
    show_virtual: bool = True,
    saturation_scale: float = 1.0,
    deployment_days: int = 30,
    pitch: int = 60,
    bearing: int = -15,
    map_lon: float | None = None,
    map_lat: float | None = None,
    map_zoom: float | None = None,
    map_min_zoom: float | None = None,
    focus_marker: dict | None = None,
) -> pdk.Deck:
    """
    建構 pydeck 2.5D 校園地圖。支援直接傳入 GeoJSON Dict 或檔案路徑。
    """
    if isinstance(energy_geojson_data, dict):
        energy_geojson = copy.deepcopy(energy_geojson_data)
    else:
        energy_geojson_data = resolve_project_path(energy_geojson_data)
        with open(energy_geojson_data, encoding="utf-8") as f:
            energy_geojson = json.load(f)

    # show_virtual=False 時，只顯示有實測分表的建物
    if not show_virtual:
        energy_geojson = {
            "type": "FeatureCollection",
            "features": [
                feat
                for feat in energy_geojson["features"]
                if (
                    bool(feat.get("properties", {}).get("has_meter_data", False))
                    or _is_metered_source(feat.get("properties", {}).get("data_source", ""))
                )
            ],
        }

    _decorate_visual_properties(
        energy_geojson,
        show_virtual=show_virtual,
        saturation_scale=float(np.clip(saturation_scale, 0.0, 1.0)),
        deployment_days=int(np.clip(deployment_days, 0, 30)),
    )
    fill_accessor = {
        "energy": "fill_color_energy",
        "eui": "fill_color_eui",
        "r2": "fill_color_r2",
        "dci": "fill_color_dci",
        "tier": "fill_color_tier",
    }.get(color_by, "fill_color_energy")

    # ---- 1. Building GeoJSON Layer ----
    buildings_layer = pdk.Layer(
        "GeoJsonLayer",
        energy_geojson,
        opacity=0.9,
        stroked=True,
        filled=True,
        extruded=True,
        wireframe=True,
        get_elevation="properties.display_height",
        get_fill_color=f"properties.{fill_accessor}",
        get_line_color="properties.outline_color",
        get_line_width="properties.outline_width",
        get_line_dash_array="properties.outline_dash_array",
        line_dash_justified=True,
        pickable=True,
        auto_highlight=True,
    )

    center_lon = float(map_lon) if map_lon is not None else 121.5375
    center_lat = float(map_lat) if map_lat is not None else 25.0175
    center_zoom = float(map_zoom) if map_zoom is not None else 15.5
    min_zoom = float(map_min_zoom) if map_min_zoom is not None else max(center_zoom - 1.0, 1.0)

    view_state = pdk.ViewState(
        longitude=center_lon,
        latitude=center_lat,
        zoom=center_zoom,
        min_zoom=min_zoom,  # 鎖定縮放，避免使用者迷失在地圖外
        max_zoom=20,
        pitch=pitch,
        bearing=bearing,
    )

    tooltip = {
        "html": "<b>{properties.name}</b><br/>"
                "EUI: {properties.eui} kWh/m?/yr<br/>"
                "Tier: {properties.energy_tier}<br/>"
                "Annual kWh: {properties.annual_kwh}<br/>"
                "Mean kW: {properties.mean_kw}<br/>"
                "Source: {properties.data_source}<br/>"
                "Confidence: {properties.confidence_level}<br/>"
                "DCI: {properties.dci_score} ({properties.dci_level})<br/>"
                "Coverage: {properties.coverage_ratio}<br/>"
                "Split: {properties.split_policy}<br/>"
                "Meter: {properties.meter_name}",
        "style": {"backgroundColor": "#ffffff", "color": "#333333"}
    }

    layers: list[pdk.Layer] = [buildings_layer]
    focus_marker_layer = _build_focus_marker_layer(focus_marker)
    if focus_marker_layer is not None:
        layers.append(focus_marker_layer)

    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="dark",
        map_provider="carto",
        tooltip=tooltip
    )
    return r

def build_topology_layers(topo, current_power_df, tier_by_uid: dict | None = None) -> list[pdk.Layer]:
    """
    Build Deck.gl layers from GridTopology structure for animated power flow.
    """
    layers = []
    if not hasattr(topo, 'generate_trips'):
        return layers

    if tier_by_uid and hasattr(topo, "assign_energy_tiers"):
        topo.assign_energy_tiers(tier_by_uid)

    trips_data = topo.generate_trips(current_power_df)
    if not trips_data:
        return layers

    # 1. Background static faint line layer for the grid network
    line_layer = pdk.Layer(
        "LineLayer",
        trips_data,
        get_source_position="path[0]",
        get_target_position="path[1]",
        get_color="color",
        get_width="width",
        width_min_pixels=1,
        opacity=0.35,
    )
    layers.append(line_layer)

    # 2. Animated TripsLayer for power flow (glowing pulse)
    trips_layer = pdk.Layer(
        "TripsLayer",
        trips_data,
        get_path="path",
        get_timestamps="timestamps",
        get_color="color",
        opacity=0.85,
        width_min_pixels=3,
        trail_length=30,
        current_time=0,
    )
    trips_layer.id = "topology_trips_layer"
    layers.append(trips_layer)

    # 3. Level-based node dots (ROOT large → METER small)
    if hasattr(topo, 'generate_node_data'):
        nodes_data = topo.generate_node_data(current_power_df)
    else:
        # Fallback: extract unique positions from trips
        nodes_data = []
        seen = set()
        for t in trips_data:
            for pt in t['path']:
                key = tuple(pt)
                if key not in seen:
                    seen.add(key)
                    nodes_data.append({
                        "position": pt,
                        "radius": 8,
                        "color": [255, 230, 0, 180],
                    })

    if nodes_data:
        nodes_layer = pdk.Layer(
            "ScatterplotLayer",
            nodes_data,
            get_position="position",
            get_radius="radius",
            get_fill_color="color",
            stroked=True,
            get_line_color=[255, 255, 255, 160],
            line_width_min_pixels=1,
            radius_min_pixels=3,
            radius_max_pixels=20,
            pickable=True,
        )
        nodes_layer.id = "topology_nodes_layer"
        layers.append(nodes_layer)

    return layers

def get_building_stats_df(
    energy_geojson_path: str | Path = "data/NTU/ntu_energy.geojson",
) -> pd.DataFrame:
    """
    從能耗 GeoJSON 提取有資料建物的統計 DataFrame，
    供 Dashboard 排行榜 / 篩選使用。
    """
    energy_geojson_path = resolve_project_path(energy_geojson_path)
    with open(energy_geojson_path, encoding="utf-8") as f:
        gj = json.load(f)

    rows = []
    for feat in gj["features"]:
        p = feat["properties"]
        if not bool(p.get("has_meter_data", False)):
            continue

        annual_kwh = p.get("annual_kwh", np.nan)
        annual_mwh = p.get("annual_mwh", np.nan)

        rows.append({
            "name":             p.get("name", ""),
            "meter_name":       p.get("meter_name", ""),
            "mean_kw":          p.get("mean_kw", 0),
            "annual_kwh":       annual_kwh,
            "annual_mwh":       annual_mwh,
            "eui":              p.get("eui", 0),
            "peak_kw":          p.get("peak_kw", np.nan),
            "load_factor":      p.get("load_factor", 0),
            "best_r2_oof":      p.get("best_r2_oof", 0),
            "best_cvrmse_oof":  p.get("best_cvrmse_oof", 0),
            "meter_role":       p.get("meter_role", ""),
            "aggregation_method": p.get("aggregation_method", ""),
            "usage_profile":    p.get("usage_profile", "default"),
            "confidence_level": p.get("confidence_level", ""),
            "coverage_ratio":   p.get("coverage_ratio", np.nan),
            "absolute_values_visible": p.get("absolute_values_visible", True),
            "archetype_label":  p.get("archetype_label", ""),
            "data_source":      p.get("data_source", ""),
        })
    return pd.DataFrame(rows)


def export_map_html(
    m: pdk.Deck,
    output_path: str = "campus_map.html",
    title: str = "NTU 校園能源數位分身",
):
    """匯出為獨立 HTML 檔案"""
    # tooltip 已在 Deck 建構時提供，這裡避免重複傳入造成版本衝突
    output_path = str(resolve_project_path(output_path))
    m.to_html(output_path)
    print(f"[MapBuilder] HTML 匯出: {output_path}")


# ── CLI ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Build NTU campus energy map")
    parser.add_argument("--geojson", default=None)
    parser.add_argument("--match", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--export", default=None, help="匯出 HTML 路徑")
    parser.add_argument(
        "--virtual-estimation",
        action="store_true",
        help="啟用無電表建物的虛擬推估（預設關閉）",
    )
    parser.add_argument("--color-by", default="energy",
                        choices=["energy", "eui", "r2", "dci", "tier"])
    args = parser.parse_args()
    geojson_path = resolve_project_path(args.geojson) if args.geojson else campus_data_dir("ntu", "ntu_buildings.geojson")
    match_path = resolve_project_path(args.match) if args.match else campus_data_dir("ntu", "meter_building_map.csv")
    config_path = resolve_project_path(args.config) if args.config else resolve_project_path("config/demo_config.yaml")
    energy_output_path = campus_data_dir("ntu", "ntu_energy.geojson")

    merge_energy_geojson(
        geojson_path,
        match_path,
        energy_output_path,
        enable_virtual_estimation=args.virtual_estimation,
        config_path=config_path,
    )
    m = build_campus_map(energy_output_path, color_by=args.color_by)

    if args.export:
        export_map_html(m, args.export)
    else:
        print("[MapBuilder] 地圖物件已建構，可在 Jupyter 或 Dashboard 中使用")


if __name__ == "__main__":
    main()
