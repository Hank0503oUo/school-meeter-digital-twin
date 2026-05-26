from __future__ import annotations

import copy
from pathlib import Path

import yaml

_ROOT_DIR = Path(__file__).resolve().parent.parent


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
            "name_keywords": ["hospital", "ntuh", "children and women"],
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
            "name_keywords": ["dorm", "prince house"],
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
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
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
        policy["manual_profiles"] = _deep_merge(policy.get("manual_profiles", {}), data["manual_profiles"])
    return policy


def _classify_archetype(mean_kw: float, r2: float) -> str:
    if mean_kw > 1000:
        return "Heavy-HVAC / Central Plant"
    if mean_kw > 600:
        return "HVAC-dominant (Predictable)" if r2 > 0.7 else "HVAC-dominant (Volatile)"
    if mean_kw > 300:
        return "Mixed-load (Schedule-driven)" if r2 > 0.6 else "Mixed-load (Complex)"
    return "Lighting-dominant (Highly regular)" if r2 > 0.7 else "Baseload-driven (Irregular)"
