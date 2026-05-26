from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.algorithm_mcp_backend import AlgorithmMCPBackend
from src.counterfactual import run_counterfactual, SensitivityCoefficients
from src.constants import HOURS_PER_YEAR
from src.knowledge_mcp_backend import KnowledgeMCPBackend
from src.real_inference_engine import PIVDEngine
from src.regulation_strategy_map import (
    classify_bee_level,
    get_regulation_for_building,
    get_strategies_for_factor,
    FACTOR_STRATEGY_MAP,
)
from src.trust_policy import _classify_archetype


def _load_v12_summary() -> pd.DataFrame:
    root = Path(__file__).resolve().parent.parent
    for candidate in (
        root / "campuses" / "ntu" / "models" / "v12_per_building_summary.csv",
        root / "models" / "v12_per_building_summary.csv",
    ):
        if candidate.exists():
            return pd.read_csv(candidate, encoding="utf-8")
    return pd.DataFrame()


def _find_building_info(
    building_name: str,
    engine: PIVDEngine,
    v12_df: pd.DataFrame,
) -> dict[str, Any] | None:
    name_lower = building_name.strip().lower()
    scaler = engine.metadata_scaler
    meta = scaler._metadata if hasattr(scaler, "_metadata") else {}
    best_uid = ""
    best_meta: dict[str, Any] | None = None
    for uid, info in meta.items():
        bname = str(info.get("name", "")).lower()
        if name_lower in bname or bname in name_lower:
            best_uid = uid
            best_meta = info
            break
    if not best_uid:
        for uid, info in meta.items():
            name_e = str(info.get("nameE", "")).lower()
            if name_lower in name_e or name_e in name_lower:
                best_uid = uid
                best_meta = info
                break

    mean_kw = 0.0
    r2 = 0.0
    meter_name = ""
    if not v12_df.empty:
        for _, row in v12_df.iterrows():
            mn = str(row.get("meter_name", "")).lower()
            if name_lower in mn or mn in name_lower:
                mean_kw = float(row.get("mean_kw", 0) or 0)
                r2 = float(row.get("best_r2_oof", 0) or 0)
                meter_name = str(row.get("meter_name", ""))
                break

    if not best_uid and not mean_kw:
        return None

    area = float(best_meta.get("area", 0)) if best_meta else 0
    build_type = str(best_meta.get("buildType", "Others")) if best_meta else "Others"

    return {
        "uid": best_uid,
        "name": best_meta.get("name", building_name) if best_meta else building_name,
        "nameE": best_meta.get("nameE", "") if best_meta else "",
        "area": area,
        "floors": int(best_meta.get("floors", 0)) if best_meta else 0,
        "build_type": build_type,
        "year": int(best_meta.get("year", 0)) if best_meta else 0,
        "mean_kw": mean_kw,
        "r2": r2,
        "meter_name": meter_name,
    }


def _run_scenario(
    mean_kw: float,
    key: str,
    value: float,
) -> dict[str, Any] | None:
    baseline_kwh = np.full(HOURS_PER_YEAR, mean_kw, dtype=float)
    kwargs: dict[str, float] = {
        "cooling_delta_degC": 0.0,
        "lighting_ratio": 1.0,
        "occupancy_ratio": 1.0,
        "equipment_ratio": 1.0,
    }
    kwargs[key] = value
    try:
        result = run_counterfactual(baseline_kwh, **kwargs)
        return result.summary_dict()
    except Exception:
        return None


def generate_adaptive_strategies(
    building_name: str,
    *,
    focus: str = "",
    max_scenarios: int = 8,
    knowledge_backend: KnowledgeMCPBackend | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()

    try:
        engine = PIVDEngine.from_defaults()
        v12_df = _load_v12_summary()

        info = _find_building_info(building_name, engine, v12_df)
        if info is None:
            return {
                "status": "error",
                "error": f"找不到建築：{building_name}",
                "building_name": building_name,
            }

        mean_kw = info["mean_kw"]
        if mean_kw <= 0:
            return {
                "status": "error",
                "error": f"建築 {info['name']} 缺少 mean_kw 資料",
                "building_name": building_name,
            }

        area = info["area"]
        build_type = info["build_type"]
        r2 = info["r2"]

        archetype = _classify_archetype(mean_kw, r2)

        annual_kwh = mean_kw * HOURS_PER_YEAR
        current_eui = annual_kwh / area if area > 0 else 0.0

        reg_info = get_regulation_for_building(build_type)
        baseline_eui = reg_info["standard_eui"]
        bee = classify_bee_level(current_eui, baseline_eui)

        algo_backend = AlgorithmMCPBackend(engine_factory=lambda: engine)
        pvid_result = algo_backend.run_pvid(
            building_uid=info["uid"],
            hours=24,
            t_out_series=[30.0] * 24,
            humidity_series=[70.0] * 24,
            start_time="2017-07-15T00:00:00",
        )
        cf_result = algo_backend.run_openbse_counterfactual(
            building_uid=info["uid"],
            cooling_delta_degC=1.0,
        )
        correlate_result = algo_backend.correlate_algorithms(
            results=[pvid_result, cf_result],
            question=f"{info['name']} 節能調適策略",
            building_uid=info["uid"],
        )
        dominant_factor = correlate_result.get("dominant_factor", "cooling_load")
        recommended_action = correlate_result.get("recommended_action", "")

        factor_order = [dominant_factor]
        for factor in FACTOR_STRATEGY_MAP:
            if factor not in factor_order:
                factor_order.append(factor)

        if focus:
            focus_lower = focus.lower()
            focus_map = {
                "cooling": "cooling_load",
                "cool": "cooling_load",
                "hvac": "cooling_load",
                "空調": "cooling_load",
                "冷卻": "cooling_load",
                "lighting": "lighting_load",
                "light": "lighting_load",
                "照明": "lighting_load",
                "equipment": "equipment_load",
                "equip": "equipment_load",
                "設備": "equipment_load",
            }
            mapped = focus_map.get(focus_lower)
            if mapped and mapped in FACTOR_STRATEGY_MAP:
                factor_order = [mapped] + [f for f in factor_order if f != mapped]

        strategies: list[dict[str, Any]] = []
        for factor in factor_order:
            if len(strategies) >= max_scenarios:
                break
            strat_info = get_strategies_for_factor(factor)
            for strat in strat_info["strategies"]:
                if len(strategies) >= max_scenarios:
                    break
                for val in strat["values"]:
                    if len(strategies) >= max_scenarios:
                        break
                    summary = _run_scenario(mean_kw, strat["key"], val)
                    if summary is None:
                        continue
                    strategies.append({
                        "factor": factor,
                        "factor_label": strat_info["label"],
                        "param_key": strat["key"],
                        "param_value": val,
                        "param_label": strat["label"],
                        "param_unit": strat["unit"],
                        "difficulty": strat["difficulty"],
                        "cost_level": strat["cost_level"],
                        "saving_kwh": abs(summary["delta_kwh"]),
                        "saving_pct": abs(summary["delta_pct"]),
                        "saving_ntd": abs(summary["delta_ntd"]),
                        "equiv_trees": abs(summary["equiv_trees"]),
                        "regulation_refs": strat_info["regulation_refs"],
                    })

        strategies.sort(key=lambda s: s["saving_pct"], reverse=True)

        combined_saving_pct = 0.0
        top_keys_seen: set[str] = set()
        for s in strategies:
            if s["param_key"] not in top_keys_seen:
                combined_saving_pct += s["saving_pct"]
                top_keys_seen.add(s["param_key"])

        regulation_chunks: list[dict[str, Any]] = []
        if knowledge_backend is not None:
            for query in reg_info["regulation_queries"][:2]:
                try:
                    docs_result = knowledge_backend.search_docs(
                        query=query,
                        building_id="hjplus-kb",
                        top_k=3,
                    )
                    for chunk in docs_result.get("chunks", []):
                        regulation_chunks.append({
                            "query": query,
                            "title": chunk.get("title", ""),
                            "excerpt": chunk.get("text", "")[:300],
                            "score": chunk.get("score", 0),
                        })
                except Exception:
                    pass

        runtime_ms = int((time.perf_counter() - started_at) * 1000)

        return {
            "status": "ok",
            "building": {
                "uid": info["uid"],
                "name": info["name"],
                "nameE": info["nameE"],
                "area": area,
                "floors": info["floors"],
                "build_type": build_type,
                "build_type_label": reg_info["label"],
                "year": info["year"],
                "mean_kw": round(mean_kw, 2),
                "r2": round(r2, 3),
                "annual_kwh": round(annual_kwh, 0),
                "current_eui": round(current_eui, 1),
                "archetype": archetype,
            },
            "regulation_baseline": {
                "standard_eui": baseline_eui,
                "bee_level": bee["level"],
                "bee_label": bee["label"],
                "eui_gap_pct": bee["gap_pct"],
                "hvac_weight": reg_info["bee_weights"]["hvac"],
                "lighting_weight": reg_info["bee_weights"]["lighting"],
                "envelope_weight": reg_info["bee_weights"]["envelope"],
            },
            "diagnosis": {
                "dominant_factor": dominant_factor,
                "recommended_action": recommended_action,
                "correlate_confidence": correlate_result.get("confidence", 0),
            },
            "strategies": strategies,
            "combined_top_saving_pct": round(combined_saving_pct, 1),
            "regulation_chunks": regulation_chunks[:6],
            "provenance": {
                "engine": "adaptive_strategy_engine_v1",
                "scenarios_evaluated": len(strategies),
                "runtime_ms": runtime_ms,
            },
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "building_name": building_name,
        }
