"""
Per-building PIVD inference for NCU using the NCU campus config (with our
freshly-populated metadata_uid.csv that now carries Solar-API derived floors).

Pipeline:
  1. PIVDEngine.from_campus("ncu") — loads ncu/models/v9, v10, metadata_uid
     (model files are copies of NTU's; metadata is NCU-specific).
  2. predict() over 2024 weather → campus-total hourly physics_pred.
  3. For every NCU building with a UID, predict_building(weather, uid):
       returns {physics_pred × scaler, building_rank_index, building_eui_index}
  4. Aggregate to monthly, compare against actual NCU 114 monthly kWh.
  5. Output:
       outputs/ncu_114/v9_per_building_hourly.csv      hourly per-building physics
       outputs/ncu_114/v9_per_building_monthly.csv     monthly aggregate
       outputs/ncu_114/v9_actual_vs_predicted.csv      with R2 + scale residual
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.real_inference_engine import PIVDEngine
from src.campus_config import CampusConfig
from src.epw_reader import read_weather

OUT_DIR = ROOT / "outputs" / "ncu_114"
WEATHER_CSV = ROOT / "models" / "weather" / "CWBTP_2024.csv"
MONTHLY_UID_CSV = OUT_DIR / "monthly_kwh_with_uid.csv"


def main():
    print("[1/5] PIVDEngine.from_campus('ncu')…")
    engine = PIVDEngine.from_campus(CampusConfig.load("ncu"))
    n_uids = len(engine.metadata_scaler.list_uids())
    print(f"      metadata_scaler loaded {n_uids} buildings")

    print(f"[2/5] Reading weather: {WEATHER_CSV.name}")
    weather = read_weather(WEATHER_CSV).sort_index()
    print(f"      hours: {len(weather)}")

    print("[3/5] Running campus-level predict…")
    campus_pred = engine.predict(weather)
    physics_total = campus_pred["physics_pred"].clip(lower=0)
    print(f"      physics_pred mean={physics_total.mean():.1f} kW, "
          f"sum={physics_total.sum()/1e6:.2f} GWh")

    print("[4/5] Loading actual NCU 114 monthly kWh…")
    actual = pd.read_csv(MONTHLY_UID_CSV, encoding="utf-8-sig")
    actual = actual.dropna(subset=["osm_id"]).copy()
    actual["osm_id"] = actual["osm_id"].astype("int64")
    actual["uid"] = "NCU_" + actual["osm_id"].astype(str)
    uids_with_data = actual["uid"].unique()
    print(f"      {len(uids_with_data)} buildings have actual kWh data")

    print(f"[5/5] Running predict_building for each of {len(uids_with_data)} UIDs…")
    per_building_hourly = {}
    rank_rows = []
    for i, uid in enumerate(uids_with_data, 1):
        try:
            res = engine.predict_building(weather, uid)
        except Exception as exc:
            print(f"      [{i:3d}] {uid}: predict_building failed — {exc}")
            continue
        per_building_hourly[uid] = res["building_rank_index"]
        meta = engine.metadata_scaler.get_metadata(uid) or {}
        rank_rows.append({
            "uid": uid,
            "name": meta.get("name", ""),
            "area_m2": meta.get("area"),
            "floors": meta.get("floors"),
            "buildType": meta.get("buildType"),
            "scaler": engine.metadata_scaler.get_scaler(uid),
            "annual_rank_index": float(res["building_rank_index"].sum()),
        })

    print()
    print(f"  successful: {len(per_building_hourly)}")
    rank_df = pd.DataFrame(rank_rows)
    rank_df.to_csv(OUT_DIR / "v9_per_building_rank.csv",
                   index=False, encoding="utf-8-sig")

    # Build hourly DataFrame
    hourly_df = pd.DataFrame(per_building_hourly)
    hourly_df.index = weather.index
    hourly_df.to_csv(OUT_DIR / "v9_per_building_hourly.csv", encoding="utf-8")

    # Monthly aggregate
    monthly_df = hourly_df.groupby(hourly_df.index.month).sum()
    monthly_df.index.name = "month"
    monthly_df.to_csv(OUT_DIR / "v9_per_building_monthly.csv", encoding="utf-8-sig")

    # Compare vs actual: per-building R2 + linear scale fit
    print()
    print("  Comparing predicted vs actual (per-building R2 of monthly shape)…")
    eval_rows = []
    for uid in uids_with_data:
        if uid not in hourly_df.columns:
            continue
        pred_monthly = monthly_df[uid]
        # Some uids may have multiple cleaned-name variants mapped to them — sum them
        bld_actual = (actual[actual["uid"] == uid]
                      .groupby("month")["kwh"].sum())
        aligned = pd.concat({"pred": pred_monthly, "act": bld_actual},
                           axis=1).dropna()
        if len(aligned) < 3 or aligned["pred"].sum() == 0:
            continue
        # Closed-form scale to align magnitude
        k = float((aligned["act"] * aligned["pred"]).sum()
                  / (aligned["pred"] ** 2).sum())
        scaled = k * aligned["pred"]
        ss_res = ((aligned["act"] - scaled) ** 2).sum()
        ss_tot = ((aligned["act"] - aligned["act"].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        # Pearson correlation (shape-only, magnitude-free)
        if aligned["pred"].std() > 0 and aligned["act"].std() > 0:
            corr = float(np.corrcoef(aligned["pred"], aligned["act"])[0, 1])
        else:
            corr = float("nan")
        meta = engine.metadata_scaler.get_metadata(uid) or {}
        eval_rows.append({
            "uid": uid,
            "name": meta.get("name", ""),
            "buildType": meta.get("buildType"),
            "scaler": engine.metadata_scaler.get_scaler(uid),
            "annual_actual_kwh": float(aligned["act"].sum()),
            "annual_pred_rank": float(aligned["pred"].sum()),
            "scale_k_to_match": k,
            "shape_R": corr,
            "R2_after_scale": r2,
            "n_months": int(len(aligned)),
        })
    eval_df = pd.DataFrame(eval_rows).sort_values("annual_actual_kwh", ascending=False)
    eval_df.to_csv(OUT_DIR / "v9_actual_vs_predicted.csv",
                   index=False, encoding="utf-8-sig")

    # Top 10 print
    print()
    print("  Top 10 buildings by actual annual kWh:")
    print("  " + "-" * 70)
    print(f"  {'name':24s}  {'actual':>10s}  {'k':>8s}  {'R':>6s}  {'R2':>6s}")
    for _, r in eval_df.head(10).iterrows():
        print(f"  {str(r['name'])[:24]:24s}  "
              f"{r['annual_actual_kwh']:>10,.0f}  "
              f"{r['scale_k_to_match']:>8.4f}  "
              f"{r['shape_R']:>6.2f}  {r['R2_after_scale']:>6.2f}")

    print()
    print(f"Wrote:")
    for f in ["v9_per_building_rank.csv", "v9_per_building_hourly.csv",
              "v9_per_building_monthly.csv", "v9_actual_vs_predicted.csv"]:
        print(f"  outputs/ncu_114/{f}")


if __name__ == "__main__":
    main()
