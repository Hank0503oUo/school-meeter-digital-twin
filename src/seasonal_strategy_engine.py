from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.constants import HOURS_PER_YEAR, ELECTRICITY_PRICE_NTD
from src.counterfactual import run_counterfactual


@dataclass(frozen=True)
class SeasonDef:
    name: str
    name_zh: str
    months: tuple[int, ...]
    cooling_bias: float
    lighting_bias: float
    equipment_bias: float
    description: str


SEASONS: list[SeasonDef] = [
    SeasonDef(
        name="summer",
        name_zh="夏季（6-9月）",
        months=(6, 7, 8, 9),
        cooling_bias=1.0,
        lighting_bias=0.8,
        equipment_bias=1.0,
        description="空調負載高峰，優先調整冷房溫度與 COP",
    ),
    SeasonDef(
        name="winter",
        name_zh="冬季（12-2月）",
        months=(12, 1, 2),
        cooling_bias=0.0,
        lighting_bias=1.2,
        equipment_bias=0.9,
        description="空調需求低，照明負載相對高，優先調整照明與設備待機",
    ),
    SeasonDef(
        name="transition",
        name_zh="過渡季（3-5月、10-11月）",
        months=(3, 4, 5, 10, 11),
        cooling_bias=0.3,
        lighting_bias=1.0,
        equipment_bias=0.95,
        description="可利用外氣冷房，減少機械空調運轉",
    ),
]


SEASONAL_STRATEGY_TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "summer": [
        {
            "label": "冷房溫度上調 +1°C",
            "params": {"cooling_delta_degC": 1.0},
            "focus": "cooling_load",
        },
        {
            "label": "冷房溫度上調 +2°C",
            "params": {"cooling_delta_degC": 2.0},
            "focus": "cooling_load",
        },
        {
            "label": "冰水主機 COP 提升 20%",
            "params": {"cop_ratio": 1.2},
            "focus": "cooling_load",
        },
        {
            "label": "空調 + 照明聯合調適",
            "params": {"cooling_delta_degC": 1.0, "lighting_ratio": 0.9},
            "focus": "cooling_load",
        },
    ],
    "winter": [
        {
            "label": "照明功率密度降至 85%",
            "params": {"lighting_ratio": 0.85},
            "focus": "lighting_load",
        },
        {
            "label": "設備待機功耗降至 90%",
            "params": {"equipment_ratio": 0.90},
            "focus": "equipment_load",
        },
        {
            "label": "照明 + 設備聯合調適",
            "params": {"lighting_ratio": 0.85, "equipment_ratio": 0.95},
            "focus": "lighting_load",
        },
    ],
    "transition": [
        {
            "label": "外氣冷房策略（冷房微調 +1°C）",
            "params": {"cooling_delta_degC": 1.0},
            "focus": "cooling_load",
        },
        {
            "label": "過渡季照明最佳化",
            "params": {"lighting_ratio": 0.9},
            "focus": "lighting_load",
        },
        {
            "label": "人員使用率最佳化（空間排程）",
            "params": {"occupancy_ratio": 0.90},
            "focus": "occupancy_load",
        },
    ],
}


def _hours_for_months(months: tuple[int, ...], year: int = 2017) -> list[int]:
    import calendar
    hours: list[int] = []
    for m in months:
        days_in_month = calendar.monthrange(year, m)[1]
        start_hour = sum(
            calendar.monthrange(year, mm)[1] for mm in range(1, m)
        ) * 24
        hours.extend(range(start_hour, start_hour + days_in_month * 24))
    return hours


def _seasonal_mean_kw(mean_kw: float, season: SeasonDef) -> float:
    if season.name == "summer":
        return mean_kw * 1.25
    if season.name == "winter":
        return mean_kw * 0.70
    return mean_kw * 0.85


def generate_seasonal_strategies(
    mean_kw: float,
    *,
    building_name: str = "",
    area: float = 0.0,
) -> dict[str, Any]:
    if mean_kw <= 0:
        return {"status": "error", "error": "mean_kw must be positive"}

    annual_kwh = mean_kw * HOURS_PER_YEAR
    annual_eui = annual_kwh / area if area > 0 else 0.0

    seasonal_results: list[dict[str, Any]] = []

    for season in SEASONS:
        season_mean = _seasonal_mean_kw(mean_kw, season)
        season_hours = _hours_for_months(season.months)
        n_hours = len(season_hours)
        season_kwh_base = season_mean * n_hours

        templates = SEASONAL_STRATEGY_TEMPLATES.get(season.name, [])
        strategies: list[dict[str, Any]] = []

        for template in templates:
            params = template["params"]
            baseline_arr = np.full(n_hours, season_mean, dtype=float)

            cf_kwargs: dict[str, float] = {
                "cooling_delta_degC": 0.0,
                "lighting_ratio": 1.0,
                "occupancy_ratio": 1.0,
                "equipment_ratio": 1.0,
            }
            for key, val in params.items():
                cf_kwargs[key] = float(val)

            try:
                result = run_counterfactual(baseline_arr, **cf_kwargs)
                summary = result.summary_dict()
            except Exception:
                continue

            saving_kwh = abs(summary["delta_kwh"])
            saving_pct = abs(summary["delta_pct"])
            saving_ntd = abs(summary["delta_ntd"])

            strategies.append({
                "label": template["label"],
                "params": params,
                "focus": template["focus"],
                "saving_kwh": round(saving_kwh, 0),
                "saving_pct": round(saving_pct, 2),
                "saving_ntd": round(saving_ntd, 0),
            })

        strategies.sort(key=lambda s: s["saving_pct"], reverse=True)

        seasonal_results.append({
            "season": season.name,
            "season_zh": season.name_zh,
            "description": season.description,
            "hours": n_hours,
            "baseline_mean_kw": round(season_mean, 2),
            "baseline_kwh": round(season_kwh_base, 0),
            "strategies": strategies,
        })

    total_potential_saving = 0.0
    for sr in seasonal_results:
        if sr["strategies"]:
            total_potential_saving += sr["strategies"][0]["saving_kwh"]

    return {
        "status": "ok",
        "building_name": building_name,
        "annual_kwh": round(annual_kwh, 0),
        "annual_eui": round(annual_eui, 1),
        "seasons": seasonal_results,
        "total_potential_saving_kwh": round(total_potential_saving, 0),
        "total_potential_saving_pct": round(
            total_potential_saving / annual_kwh * 100, 1
        ) if annual_kwh > 0 else 0.0,
        "total_potential_saving_ntd": round(total_potential_saving * ELECTRICITY_PRICE_NTD, 0),
        "recommendation": _build_seasonal_summary(seasonal_results),
    }


def _build_seasonal_summary(seasonal_results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for sr in seasonal_results:
        if not sr["strategies"]:
            continue
        best = sr["strategies"][0]
        lines.append(
            f"{sr['season_zh']}：建議「{best['label']}」，"
            f"可省 {best['saving_kwh']:.0f} kWh ({best['saving_pct']:.1f}%)"
        )
    return "；".join(lines) if lines else "無明顯節能空間"
