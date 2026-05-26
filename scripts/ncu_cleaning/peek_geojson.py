"""Dump NCU buildings.geojson properties to UTF-8 file for inspection."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GJ_PATH = ROOT / "campuses" / "ncu" / "data" / "buildings.geojson"
OUT_PATH = ROOT / "outputs" / "ncu_114" / "_geojson_buildings.csv"

with open(GJ_PATH, encoding="utf-8") as f:
    gj = json.load(f)

rows = []
for ft in gj["features"]:
    p = ft.get("properties", {})
    rows.append({
        "osm_id": p.get("osm_id"),
        "name": p.get("name", ""),
        "building_type": p.get("building_type", ""),
        "building_code": p.get("building_code", ""),
        "meter_name": p.get("meter_name", ""),
        "footprint_area_m2": p.get("footprint_area_m2"),
        "levels": p.get("levels"),
    })

import csv
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"wrote {len(rows)} rows to {OUT_PATH}")
