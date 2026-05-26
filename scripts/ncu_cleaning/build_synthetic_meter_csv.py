"""
Build a multi-column powerMeter_kW_hourly.csv that the dashboard's meter selector
will show: NCU campus total + top-10 building sub-meters.

Source columns:
  - V9 per-building hourly (building_rank_index, dimensionless) from
    outputs/ncu_114/v9_per_building_hourly.csv
  - Per-building scale_k_to_match from outputs/ncu_114/v9_actual_vs_predicted.csv
    → multiply rank_index × k → real kW per hour

The campus total column is the sum of all 48 buildings (matches actual annual
kWh from cleaned monthly data, since each k was fit to that).

Output: campuses/ncu/models/powerMeter_kW_hourly.csv (overwritten)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "ncu_114"
NCU_GEOJSON = ROOT / "campuses" / "ncu" / "data" / "buildings.geojson"
TARGET = ROOT / "campuses" / "ncu" / "models" / "powerMeter_kW_hourly.csv"

TOP_N = 10


def main():
    print("Loading per-building hourly + scaling factors…")
    hourly = pd.read_csv(OUT_DIR / "v9_per_building_hourly.csv",
                         index_col=0, parse_dates=True)
    eval_df = pd.read_csv(OUT_DIR / "v9_actual_vs_predicted.csv",
                          encoding="utf-8-sig")

    # Build uid → name lookup from buildings.geojson
    uid_to_name = {}
    gj = json.loads(NCU_GEOJSON.read_text(encoding="utf-8"))
    for ft in gj["features"]:
        p = ft.get("properties", {})
        oid = p.get("osm_id")
        if oid is not None and p.get("name"):
            uid_to_name[f"NCU_{oid}"] = p["name"]

    # Convert rank_index → real kW for each building present in eval_df
    real_kw = {}
    for _, r in eval_df.iterrows():
        uid = r["uid"]
        k = r["scale_k_to_match"]
        if uid in hourly.columns and pd.notna(k) and k > 0:
            real_kw[uid] = hourly[uid] * float(k)
    real_kw_df = pd.DataFrame(real_kw, index=hourly.index)

    # Campus total = sum of all building kW per hour
    campus_total = real_kw_df.sum(axis=1)

    # Pick top N buildings by annual kWh contribution
    annual_by_uid = (real_kw_df.sum(axis=0) * 1.0).sort_values(ascending=False)
    top_uids = annual_by_uid.head(TOP_N).index.tolist()

    out = pd.DataFrame({"日期時間": hourly.index.strftime("%Y-%m-%d %H:%M:%S")})
    out["NCU_校區總表"] = campus_total.round(2).values
    for uid in top_uids:
        col = f"{uid_to_name.get(uid, uid)}"
        out[col] = real_kw_df[uid].round(2).values

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(TARGET, index=False, encoding="utf-8-sig")

    annual_total = campus_total.sum()
    print(f"Wrote {TARGET}")
    print(f"  rows: {len(out)}")
    print(f"  columns: 1 datetime + 1 campus total + {len(top_uids)} top buildings")
    print(f"  campus annual: {annual_total/1e6:.2f} GWh")
    print()
    print(f"  Top {TOP_N} buildings (annual kWh contribution):")
    for uid in top_uids:
        annual_kwh = real_kw_df[uid].sum()
        print(f"    {uid_to_name.get(uid, uid):24s}  {annual_kwh:>10,.0f} kWh "
              f"({annual_kwh/annual_total*100:.1f}%)")


if __name__ == "__main__":
    main()
