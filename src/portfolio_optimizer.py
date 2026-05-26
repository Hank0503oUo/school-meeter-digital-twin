from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.constants import HOURS_PER_YEAR, ELECTRICITY_PRICE_NTD, GRID_EMISSION_FACTOR
from src.counterfactual import run_counterfactual
from src.trust_policy import _classify_archetype
from src.regulation_strategy_map import get_regulation_for_building, classify_bee_level


COST_ESTIMATES: dict[str, dict[str, float]] = {
    "cooling_delta_degC": {"cost_per_kwh_saved": 2.0, "label": "冷房溫度調整", "category": "免費/低成本的排程調整"},
    "cop_ratio": {"cost_per_kwh_saved": 15.0, "label": "冰水主機效率提升", "category": "中成本設備升級"},
    "lighting_ratio": {"cost_per_kwh_saved": 5.0, "label": "照明系統改善", "category": "低成本 LED 更換"},
    "equipment_ratio": {"cost_per_kwh_saved": 3.0, "label": "設備待機管理", "category": "免費/低成本的排程調整"},
    "occupancy_ratio": {"cost_per_kwh_saved": 1.0, "label": "空間排程最佳化", "category": "免費管理措施"},
}

DEFAULT_SCENARIO: dict[str, float] = {
    "cooling_delta_degC": 2.0,
    "lighting_ratio": 0.85,
    "equipment_ratio": 0.90,
}


def _load_all_buildings() -> pd.DataFrame:
    root = Path(__file__).resolve().parent.parent
    for candidate in (
        root / "campuses" / "ntu" / "models" / "v12_per_building_summary.csv",
        root / "models" / "v12_per_building_summary.csv",
    ):
        if candidate.exists():
            df = pd.read_csv(candidate, encoding="utf-8")
            df = df[df["mean_kw"] > 0].copy()
            df["annual_kwh"] = df["mean_kw"] * HOURS_PER_YEAR
            return df
    return pd.DataFrame()


def _score_building(
    row: pd.Series,
    scenario: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    mean_kw = float(row.get("mean_kw", 0))
    if mean_kw <= 0:
        return None

    r2 = float(row.get("best_r2_oof", 0) or 0)
    meter_name = str(row.get("meter_name", "unknown"))
    annual_kwh = float(row.get("annual_kwh", 0))

    archetype = _classify_archetype(mean_kw, r2)

    baseline_arr = np.full(HOURS_PER_YEAR, mean_kw, dtype=float)

    scenario = scenario or DEFAULT_SCENARIO
    try:
        result = run_counterfactual(baseline_arr, **scenario)
        summary = result.summary_dict()
    except Exception:
        return None

    saving_kwh = abs(summary["delta_kwh"])
    saving_pct = abs(summary["delta_pct"])

    best_param = ""
    best_param_saving = 0.0
    for key, val in scenario.items():
        try:
            single = run_counterfactual(
                baseline_arr,
                **{k: v for k, v in scenario.items() if k == key},
            )
            single_saving = abs(single.summary_dict()["delta_kwh"])
            if single_saving > best_param_saving:
                best_param_saving = single_saving
                best_param = key
        except Exception:
            continue

    cost_info = COST_ESTIMATES.get(best_param, COST_ESTIMATES["cooling_delta_degC"])
    estimated_cost = saving_kwh * cost_info["cost_per_kwh_saved"]
    roi_years = estimated_cost / (saving_kwh * ELECTRICITY_PRICE_NTD) if saving_kwh > 0 else 999.0

    short_name = meter_name
    for sep in ("_",):
        parts = meter_name.split(sep)
        if len(parts) >= 3:
            short_name = parts[2] if len(parts[2]) > 2 else sep.join(parts[2:])
            break

    return {
        "meter_name": meter_name,
        "short_name": short_name,
        "mean_kw": round(mean_kw, 1),
        "annual_kwh": round(annual_kwh, 0),
        "r2": round(r2, 3),
        "archetype": archetype,
        "saving_kwh": round(saving_kwh, 0),
        "saving_pct": round(saving_pct * 100, 1),
        "saving_ntd": round(saving_kwh * ELECTRICITY_PRICE_NTD, 0),
        "co2_reduction_tonnes": round(saving_kwh * GRID_EMISSION_FACTOR / 1000, 1),
        "best_param": best_param,
        "best_param_label": cost_info["label"],
        "cost_category": cost_info["category"],
        "estimated_cost_ntd": round(estimated_cost, 0),
        "roi_years": round(roi_years, 1),
        "score": round(saving_kwh * (0.3 + 0.7 * saving_pct), 0),
    }


def optimize_portfolio(
    *,
    budget_ntd: float = 0,
    max_buildings: int = 10,
    min_saving_pct: float = 1.0,
    scenario: dict[str, float] | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()

    df = _load_all_buildings()
    if df.empty:
        return {"status": "error", "error": "No building data available"}

    scored: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        result = _score_building(row, scenario)
        if result is None:
            continue
        if result["saving_pct"] < min_saving_pct:
            continue
        scored.append(result)

    scored.sort(key=lambda b: b["score"], reverse=True)

    if budget_ntd > 0:
        selected: list[dict[str, Any]] = []
        remaining_budget = budget_ntd
        for b in scored:
            if len(selected) >= max_buildings:
                break
            if b["estimated_cost_ntd"] <= remaining_budget:
                selected.append(b)
                remaining_budget -= b["estimated_cost_ntd"]
        for b in scored:
            if len(selected) >= max_buildings:
                break
            if b not in selected:
                selected.append(b)
    else:
        selected = scored[:max_buildings]

    total_saving_kwh = sum(b["saving_kwh"] for b in selected)
    total_saving_ntd = sum(b["saving_ntd"] for b in selected)
    total_cost = sum(b["estimated_cost_ntd"] for b in selected)
    total_co2 = sum(b["co2_reduction_tonnes"] for b in selected)
    avg_roi = np.mean([b["roi_years"] for b in selected]) if selected else 0.0

    all_annual = sum(b["annual_kwh"] for b in scored) if scored else 1
    coverage_pct = total_saving_kwh / all_annual * 100 if all_annual > 0 else 0

    runtime_ms = int((time.perf_counter() - started_at) * 1000)

    return {
        "status": "ok",
        "total_buildings_evaluated": len(scored),
        "total_buildings_selected": len(selected),
        "budget_ntd": budget_ntd if budget_ntd > 0 else None,
        "portfolio": selected,
        "totals": {
            "saving_kwh": round(total_saving_kwh, 0),
            "saving_ntd": round(total_saving_ntd, 0),
            "co2_reduction_tonnes": round(total_co2, 1),
            "estimated_cost_ntd": round(total_cost, 0),
            "coverage_pct": round(coverage_pct, 1),
            "avg_roi_years": round(avg_roi, 1),
        },
        "recommendation": _build_portfolio_summary(selected, total_saving_kwh, total_saving_ntd),
        "provenance": {
            "engine": "portfolio_optimizer_v1",
            "scenario": scenario or DEFAULT_SCENARIO,
            "runtime_ms": runtime_ms,
        },
    }


def _build_portfolio_summary(
    selected: list[dict[str, Any]],
    total_kwh: float,
    total_ntd: float,
) -> str:
    if not selected:
        return "無符合條件的建築"
    top3 = selected[:3]
    names = "、".join(b["short_name"][:12] for b in top3)
    return (
        f"建議優先投資 {len(selected)} 棟建築，"
        f"年省 {total_kwh:,.0f} kWh（約 {total_ntd:,.0f} 元），"
        f"前三名為：{names}"
    )
