from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.utils import weighted_mean as _weighted_mean


def _detect_meter_role(meter_name: str, shared_group_size: int = 1) -> str:
    """Heuristic meter role detection."""
    name = str(meter_name or "").upper()

    if any(token in name for token in ["CAMPUS", "MAIN STATION", "TOTAL CAMPUS"]):
        return "campus_total"
    if any(token in name for token in ["SUBMETER", "SUB-METER", "SUB METER"]):
        return "submeter"
    if any(token in name for token in ["FEEDER"]):
        return "feeder"
    if any(token in name for token in ["BACKUP", "TEST"]):
        return "backup"

    if re.search(r"\b(MAIN|MCB|GCB|VCB|ACB)\b", name):
        return "shared_total" if shared_group_size > 1 else "building_total"

    return "shared_total" if shared_group_size > 1 else "unknown"


def _meter_role_priority(role: str) -> int:
    priority = {
        "building_total": 60,
        "submeter": 50,
        "feeder": 40,
        "shared_total": 30,
        "unknown": 20,
        "backup": 10,
        "campus_total": 0,
    }
    return priority.get(str(role or ""), 20)


def _select_rows_for_building(group: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    g = group.copy()
    g["_mean_kw"] = pd.to_numeric(g["mean_kw"], errors="coerce").fillna(0.0)
    g["_role"] = g["meter_role"].fillna("unknown")
    g["_priority"] = g["_role"].map(_meter_role_priority)

    non_campus = g[g["_role"] != "campus_total"]
    if not non_campus.empty:
        g = non_campus

    top_priority = int(g["_priority"].max()) if not g.empty else 0
    top = g[g["_priority"] == top_priority]
    if top.empty:
        top = g

    selected_role = str(top["_role"].iloc[0]) if not top.empty else "unknown"
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

    meter_names = [str(name).strip() for name in selected_rows["meter_name"].fillna("") if str(name).strip()]
    unique_meter_names = list(dict.fromkeys(meter_names))

    selected_mean_kw = float(mean_kw_values.max()) if len(mean_kw_values) else 0.0
    best_n_valid = float(pd.to_numeric(selected_rows.get("n_valid_hours", pd.Series([np.nan])), errors="coerce").max())
    best_cov = float(pd.to_numeric(selected_rows.get("coverage_ratio", pd.Series([np.nan])), errors="coerce").max())
    best_zero_ratio = float(
        pd.to_numeric(selected_rows.get("zero_ratio_valid", pd.Series([np.nan])), errors="coerce").max()
    )
    best_night_to_day = float(
        pd.to_numeric(selected_rows.get("night_to_day_ratio", pd.Series([np.nan])), errors="coerce").max()
    )
    best_peak_to_mean = float(
        pd.to_numeric(selected_rows.get("peak_to_mean_ratio_p95", pd.Series([np.nan])), errors="coerce").max()
    )
    best_r2_corr = float(
        pd.to_numeric(selected_rows.get("best_r2_from_corr_sq", pd.Series([np.nan])), errors="coerce").max()
    )

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
