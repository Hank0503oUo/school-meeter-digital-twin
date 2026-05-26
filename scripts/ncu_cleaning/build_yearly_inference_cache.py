"""
Build per-year inference cache parquet files for NCU so dashboard's
get_yearly_inference(year) returns real data when user changes Year slider.

Years processed:
    民國 109  →  2020 (Q2-Q4 only, ~9 months)
    民國 110  →  2021
    民國 111  →  2022
    民國 114  →  2025

For each year, per-building inference row composed by hybrid:
  - mean_kw_actual  = annualized actual kWh from cleaned monthly / hours_in_year
  - mean_kw_pivd    = PIVDEngine.predict(weather_year).physics_pred.mean() × scaler
  - mean_kw         = actual if available, else pivd
  - data_source     = "ncu_real_<year>" or "pivd_estimate_<year>"

Writes to: data/cache/ncu/inference_cache_<ad_year>.parquet
Reads from: outputs/ncu_<roc>/monthly_kwh.csv, models/weather/CWBTP_<ad>.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.real_inference_engine import PIVDEngine
from src.campus_config import CampusConfig, inference_cache_path
from src.epw_reader import read_weather

# Reuse the manual alias dictionary from match_buildings.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from match_buildings import (
    MANUAL_ALIASES,
    load_geojson_names,
    match_one,
)

YEAR_ROC_TO_AD = {109: 2020, 110: 2021, 111: 2022, 114: 2025}
# Bonus year with only PIVD estimate (no actual NCU meter data for 113=2024)
YEARS_PIVD_ONLY = [2024]

CACHE_DIR = ROOT / "data" / "cache" / "ncu"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WEATHER_DIR = ROOT / "models" / "weather"
NCU_GEOJSON = ROOT / "campuses" / "ncu" / "data" / "buildings.geojson"

TIER_HIGH_KWH = 1_500_000
TIER_LOW_KWH = 200_000


def hours_in_year(ad_year: int) -> int:
    is_leap = (ad_year % 4 == 0) and (ad_year % 100 != 0 or ad_year % 400 == 0)
    return 8784 if is_leap else 8760


def make_unit_profile(values: pd.Series | np.ndarray, target_hours: int) -> np.ndarray:
    """Return a positive hourly profile with mean 1.0 and exactly target_hours."""
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.ones(target_hours, dtype=float)

    arr = np.clip(arr, 0.0, None)
    if arr.size != target_hours:
        if arr.size == 1:
            arr = np.full(target_hours, float(arr[0]), dtype=float)
        else:
            old_x = np.linspace(0.0, 1.0, num=arr.size)
            new_x = np.linspace(0.0, 1.0, num=target_hours)
            arr = np.interp(new_x, old_x, arr)

    mean_value = float(np.mean(arr))
    if not np.isfinite(mean_value) or mean_value <= 0:
        return np.ones(target_hours, dtype=float)
    return arr / mean_value


def match_year_monthly_to_uids(roc_year: int, gj_df: pd.DataFrame) -> pd.DataFrame:
    """Read this year's monthly_kwh.csv and attach osm_id via match_buildings."""
    monthly_csv = ROOT / "outputs" / f"ncu_{roc_year}" / "monthly_kwh.csv"
    if not monthly_csv.exists():
        print(f"  [WARN] {monthly_csv} not found")
        return pd.DataFrame()
    monthly = pd.read_csv(monthly_csv, encoding="utf-8-sig")

    name_to_match = {n: match_one(n, gj_df)
                     for n in monthly["building"].dropna().unique()}
    monthly["osm_id"] = monthly["building"].map(
        lambda n: name_to_match.get(n, {}).get("osm_id"))
    monthly["geojson_name"] = monthly["building"].map(
        lambda n: name_to_match.get(n, {}).get("geojson_name"))
    return monthly.dropna(subset=["osm_id"]).copy()


def main():
    # ── Load engine once ──────────────────────────────────────────────────
    print("[init] Loading PIVDEngine.from_campus('ncu')…")
    cfg = CampusConfig.load("ncu")
    engine = PIVDEngine.from_campus(cfg)
    n_uids_meta = len(engine.metadata_scaler.list_uids())
    print(f"        metadata_scaler: {n_uids_meta} buildings")

    print("[init] Loading geojson name table…")
    gj_df = load_geojson_names()
    print(f"        geojson named buildings: {len(gj_df)}")

    # uid → meta lookup
    meta_by_uid = {}
    for uid in engine.metadata_scaler.list_uids():
        m = engine.metadata_scaler.get_metadata(uid) or {}
        meta_by_uid[uid] = m

    # ── Per-year processing ───────────────────────────────────────────────
    summary_rows = []
    all_years = list(YEAR_ROC_TO_AD.items()) + [(None, y) for y in YEARS_PIVD_ONLY]
    for roc_year, ad_year in all_years:
        print()
        label = f"民國 {roc_year} (" if roc_year else "(PIVD-only "
        print(f"=== {label}西元 {ad_year}) ===")

        # 1) actual data (skip if no roc_year for this year)
        if roc_year is not None:
            actual = match_year_monthly_to_uids(roc_year, gj_df)
            if not actual.empty:
                actual["uid"] = "NCU_" + actual["osm_id"].astype("int64").astype(str)
                annual_actual = actual.groupby("uid", as_index=True).agg(
                    annual_kwh=("kwh", "sum"),
                    n_months=("month", "nunique"),
                    geojson_name=("geojson_name", "first"),
                    building_raw=("building", "first"),
                )
            else:
                annual_actual = pd.DataFrame(columns=["annual_kwh", "n_months",
                                                       "geojson_name", "building_raw"])
        else:
            annual_actual = pd.DataFrame(columns=["annual_kwh", "n_months",
                                                   "geojson_name", "building_raw"])
        n_actual_uids = len(annual_actual)
        cov_kwh = float(annual_actual["annual_kwh"].sum()) if n_actual_uids else 0.0
        print(f"  actual: {n_actual_uids} UIDs, total annual {cov_kwh/1e6:.2f} GWh")

        # 2) weather + PIVD physics
        wx_path = WEATHER_DIR / f"CWBTP_{ad_year}.csv"
        wx_proxy_note = ""
        if not wx_path.exists():
            # Fall back to closest available prior year, then 2017 as last resort
            for fallback in [ad_year - 1, ad_year - 2, ad_year - 3, 2017]:
                cand = WEATHER_DIR / f"CWBTP_{fallback}.csv"
                if cand.exists():
                    wx_path = cand
                    wx_proxy_note = f" (proxied from {fallback})"
                    print(f"  weather: {ad_year} not found, using {fallback} as proxy")
                    break
        if not wx_path.exists():
            print(f"  [WARN] no weather file available for {ad_year}")
            mean_kw_pivd_global = None
            physics_profile = np.ones(hours_in_year(ad_year), dtype=float)
        else:
            weather = read_weather(wx_path).sort_index()
            pred = engine.predict(weather)
            phys = pred["physics_pred"].clip(lower=0)
            mean_kw_pivd_global = float(phys.mean())
            physics_profile = make_unit_profile(phys, hours_in_year(ad_year))
            print(f"  PIVD physics mean: {mean_kw_pivd_global:.1f} kW "
                  f"({phys.sum()/1e6:.2f} GWh annual at campus scale)")

        # 3) compute anchor: real "kW per unit scaler" so PIVD estimates land
        #    in the same magnitude as real measurements (this year)
        anchor_kw_per_scaler = None
        if n_actual_uids:
            real_annualized = []
            real_scalers = []
            for uid in annual_actual.index:
                if uid not in meta_by_uid:
                    continue
                row = annual_actual.loc[uid]
                annual_full = float(row["annual_kwh"]) * (12 / max(int(row["n_months"]), 1))
                real_annualized.append(annual_full / hours_in_year(ad_year))
                real_scalers.append(engine.metadata_scaler.get_scaler(uid))
            if real_scalers and sum(real_scalers) > 0:
                anchor_kw_per_scaler = float(
                    np.sum(real_annualized) / np.sum(real_scalers)
                )
                print(f"  anchor: {anchor_kw_per_scaler:.1f} kW per unit scaler "
                      f"(from {len(real_scalers)} real buildings)")
        if anchor_kw_per_scaler is None and mean_kw_pivd_global is not None:
            # No real data: use a conservative fallback (campus PIVD divided by N)
            anchor_kw_per_scaler = mean_kw_pivd_global / max(n_uids_meta, 1)
            print(f"  anchor: {anchor_kw_per_scaler:.2f} kW per unit scaler "
                  f"(fallback: campus_pivd / N_buildings)")

        # 4) compose per-building rows
        rows = []
        hrs = hours_in_year(ad_year)
        for uid, meta in meta_by_uid.items():
            scaler = engine.metadata_scaler.get_scaler(uid)
            area = float(meta.get("area") or 0.0)
            btype = meta.get("buildType", "")
            name = meta.get("name", "")

            # Actual?
            if uid in annual_actual.index:
                row = annual_actual.loc[uid]
                annual_kwh_full = float(row["annual_kwh"]) * (12 / max(int(row["n_months"]), 1))
                mean_kw = annual_kwh_full / hrs
                source = f"ncu_real_{ad_year}"
                gname = row["geojson_name"] or name
                coverage = float(row["n_months"]) / 12.0
            elif anchor_kw_per_scaler is not None:
                # PIVD estimate anchored to real per-scaler kW
                mean_kw = anchor_kw_per_scaler * scaler
                annual_kwh_full = mean_kw * hrs
                source = f"pivd_estimate_{ad_year}{wx_proxy_note}"
                gname = name
                coverage = 0.0
            else:
                continue

            eui_annual_kwh_per_m2 = (annual_kwh_full / area) if area > 0 else np.nan
            eui_kw_per_m2 = (mean_kw / area) if area > 0 else np.nan
            tier = ("HIGH" if annual_kwh_full > TIER_HIGH_KWH
                    else "LOW" if annual_kwh_full < TIER_LOW_KWH
                    else "NORMAL")

            rows.append({
                "uid": uid,
                "name": name,
                "meter_name": gname,
                "buildType": btype,
                "area": area,
                "scaler": scaler,
                "mean_kw": round(mean_kw, 2),
                "annual_kwh": round(annual_kwh_full, 0),
                "eui_kw_per_m2": round(float(eui_kw_per_m2), 4) if not np.isnan(eui_kw_per_m2) else np.nan,
                "eui_annual_kwh_per_m2": round(float(eui_annual_kwh_per_m2), 1) if not np.isnan(eui_annual_kwh_per_m2) else np.nan,
                "energy_tier": tier,
                "data_source": source,
                "data_year": ad_year,
                "coverage_ratio": round(coverage, 2),
                "timeseries": (physics_profile * float(mean_kw)).astype(float),
            })

        df = pd.DataFrame(rows)
        out_path = inference_cache_path("ncu", ad_year)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(out_path, index=False)
        except (ImportError, OSError) as exc:
            print(f"  [ERROR] parquet write failed ({exc}); writing CSV instead")
            df.to_csv(out_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")

        n_real = (df["data_source"].str.startswith("ncu_real")).sum()
        n_pivd = (df["data_source"].str.startswith("pivd_estimate")).sum()
        print(f"  wrote {len(df)} rows ({n_real} real, {n_pivd} PIVD-estimated)")
        print(f"  → {out_path}")

        summary_rows.append({
            "roc_year": roc_year,
            "ad_year": ad_year,
            "n_total": len(df),
            "n_real": n_real,
            "n_pivd": n_pivd,
            "actual_total_GWh": round(cov_kwh / 1e6, 2),
            "cache_path": str(out_path.relative_to(ROOT)),
        })

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("=== Summary ===")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
