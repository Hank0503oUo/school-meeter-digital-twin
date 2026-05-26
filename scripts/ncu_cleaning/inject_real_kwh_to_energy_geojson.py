"""
Inject NCU 114 (2025) real annual_kwh into campuses/ncu/data/energy.geojson
so the dashboard map can color buildings by actual measured energy.

For each feature in energy.geojson:
  - Match by osm_id ↔ monthly_kwh_with_uid.csv
  - If matched: write annual_kwh, annual_mwh, mean_kw, peak_kw, eui, eui_kw_per_m2,
    has_meter_data=True, meter_name=<building name>, data_source=ncu_114_real,
    energy_tier (HIGH/NORMAL/LOW), tier_color, levels (from buildings.geojson),
    height (= levels × 3.5)
  - Else: zero out kWh fields, mark has_meter_data=False, data_source=synthetic_zero

Backup written to energy.bak.geojson.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NCU_DIR = ROOT / "campuses" / "ncu" / "data"
ENERGY = NCU_DIR / "energy.geojson"
BUILDINGS = NCU_DIR / "buildings.geojson"
MONTHLY_UID = ROOT / "outputs" / "ncu_114" / "monthly_kwh_with_uid.csv"
BACKUP = ENERGY.with_suffix(".bak.geojson")

HOURS_PER_YEAR_2025 = 8760  # not a leap year — 2025 is not divisible by 4? 2024 is leap. 2025 is not leap.
PEAK_OVER_MEAN_RATIO = 1.6   # rough univ profile heuristic
TIER_HIGH_THRESHOLD_KWH = 1_500_000
TIER_LOW_THRESHOLD_KWH = 200_000

TIER_COLORS = {
    "HIGH":   [215, 48, 39, 220],
    "NORMAL": [240, 196, 25, 220],
    "LOW":    [26, 152, 80, 220],
}


def main():
    print("Loading sources…")
    monthly = pd.read_csv(MONTHLY_UID, encoding="utf-8-sig")
    monthly = monthly.dropna(subset=["osm_id"]).copy()
    monthly["osm_id"] = monthly["osm_id"].astype("int64")
    annual = monthly.groupby("osm_id", as_index=True)["kwh"].sum().to_dict()
    n_months_per_bld = monthly.groupby("osm_id")["month"].nunique().to_dict()

    bld_props = {}
    for ft in json.loads(BUILDINGS.read_text(encoding="utf-8"))["features"]:
        p = ft.get("properties", {})
        oid = p.get("osm_id")
        if oid is not None:
            bld_props[int(oid)] = p

    if not BACKUP.exists():
        BACKUP.write_bytes(ENERGY.read_bytes())
        print(f"  backup → {BACKUP.name}")

    energy_gj = json.loads(ENERGY.read_text(encoding="utf-8"))

    n_real, n_zero = 0, 0
    for feat in energy_gj.get("features", []):
        p = feat.setdefault("properties", {})
        oid = p.get("osm_id")
        try:
            oid = int(oid) if oid is not None else None
        except (TypeError, ValueError):
            oid = None

        # Refresh levels/height from enriched buildings.geojson
        if oid in bld_props:
            bp = bld_props[oid]
            new_levels = bp.get("levels")
            if new_levels:
                p["levels"] = int(new_levels)
                p["b_floors"] = int(new_levels)
                p["height"] = float(bp.get("height_m") or new_levels * 3.5)

        # Refresh footprint area
        area = bld_props.get(oid, {}).get("footprint_area_m2") if oid else None
        if area:
            p["footprint_area_m2"] = float(area)
            p["b_area"] = float(area) * (p.get("levels") or 3)  # total floor area

        if oid in annual:
            kwh = float(annual[oid])
            n_known_months = int(n_months_per_bld.get(oid, 12))
            # Annualize: scale up if fewer than 12 months observed
            kwh_annualized = kwh * (12 / max(n_known_months, 1))
            mean_kw = kwh_annualized / HOURS_PER_YEAR_2025
            peak_kw = mean_kw * PEAK_OVER_MEAN_RATIO
            tfa = p.get("b_area") or (area or 0) * (p.get("levels") or 3)
            eui = (kwh_annualized / tfa) if tfa else 0.0
            eui_kw_per_m2 = (mean_kw / tfa) if tfa else 0.0
            tier = ("HIGH" if kwh_annualized > TIER_HIGH_THRESHOLD_KWH
                    else "LOW" if kwh_annualized < TIER_LOW_THRESHOLD_KWH
                    else "NORMAL")
            p.update({
                "annual_kwh": round(kwh_annualized, 0),
                "annual_kwh_raw": round(kwh, 0),
                "annual_mwh": round(kwh_annualized / 1000, 1),
                "annual_mwh_raw": round(kwh / 1000, 1),
                "mean_kw": round(mean_kw, 1),
                "mean_kw_raw": round(mean_kw, 1),
                "peak_kw": round(peak_kw, 1),
                "eui": round(eui, 1),
                "eui_raw": round(eui, 1),
                "eui_kw_per_m2": round(eui_kw_per_m2, 4),
                "has_meter_data": True,
                "meter_name": p.get("name", "") or f"NCU_{oid}",
                "data_source": "ncu_114_real",
                "data_year": 2025,
                "energy_tier": tier,
                "tier_color": TIER_COLORS[tier],
                "coverage_ratio": round(n_known_months / 12.0, 2),
                "uid": f"NCU_{oid}",
                "load_factor": round(mean_kw / peak_kw if peak_kw else 0, 2),
            })
            n_real += 1
        else:
            p.update({
                "annual_kwh": 0.0,
                "annual_kwh_raw": 0.0,
                "annual_mwh": 0.0,
                "mean_kw": 0.0,
                "peak_kw": 0.0,
                "eui": 0.0,
                "eui_kw_per_m2": 0.0,
                "has_meter_data": False,
                "data_source": "synthetic_zero",
                "data_year": 2025,
                "energy_tier": "LOW",
                "tier_color": [180, 180, 180, 100],
                "uid": f"NCU_{oid}" if oid else "",
            })
            n_zero += 1

    ENERGY.write_text(
        json.dumps(energy_gj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote enriched {ENERGY}")
    print(f"  features with real kWh: {n_real}")
    print(f"  features zeroed out:    {n_zero}")


if __name__ == "__main__":
    main()
