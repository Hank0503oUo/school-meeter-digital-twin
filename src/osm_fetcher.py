# -*- coding: utf-8 -*-
"""
Step 1：從 OpenStreetMap Overpass API 取得校園建物足跡 GeoJSON
"""

import json
import argparse
from pathlib import Path
import requests
from src.project_paths import campus_data_dir, resolve_project_path

# ── 校園 Bounding Box (精確過濾，排除校外區域如中央宵夜街) ──────────────────
NTU_BBOX = {
    "south": 25.0130,
    "west":  121.5330,
    "north": 25.0220,
    "east":  121.5460,
}

NCU_BBOX = {
    "south": 24.9650,
    "west":  121.1880,
    "north": 24.9730,
    "east":  121.1980,
}

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

DEFAULT_FLOOR_HEIGHT_M = 3.5   # 每層樓高估計
DEFAULT_LEVELS = 3             # 無樓層數資料時預設

def fetch_campus_buildings(bbox: dict) -> dict:
    query = f"""
    [out:json][timeout:60];
    (
      way["building"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
      relation["building"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
    );
    out geom;
    """
    print(f"[OSM Fetcher] 查詢範圍: {bbox}")
    resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=90)
    resp.raise_for_status()
    osm_data = resp.json()
    elements = osm_data.get("elements", [])
    features = []
    for elem in elements:
        geom = _element_to_geometry(elem)
        if geom is None: continue
        tags = elem.get("tags", {})
        
        # ── 篩選邏輯 ──
        # 排除明確非校園建物 (公寓、商業、住宅)
        operator = tags.get("operator", "")
        name = tags.get("name", "")
        btype = tags.get("building", "yes")
        
        # 排除特定名稱
        if "宵夜街" in name or "中央路" in name:
            continue
        # 排除非校園建築類型 (公寓、商業、住宅、零售等)
        _EXCLUDE_TYPES = {"apartments", "residential", "commercial", "retail",
                          "house", "detached", "terrace", "semidetached_house"}
        if btype in _EXCLUDE_TYPES:
            continue
            
        levels = _parse_int(tags.get("building:levels"), DEFAULT_LEVELS)
        height = _parse_float(tags.get("height"), levels * DEFAULT_FLOOR_HEIGHT_M)
        feature = {
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "osm_id": elem["id"],
                "name": name,
                "building_type": tags.get("building", "yes"),
                "operator": operator,
                "levels": levels,
                "height": round(height, 1)
            },
        }
        features.append(feature)
    return {"type": "FeatureCollection", "features": features}

def _element_to_geometry(elem: dict) -> dict | None:
    if elem["type"] == "way" and "geometry" in elem:
        coords = [[n["lon"], n["lat"]] for n in elem["geometry"]]
        if coords and coords[0] != coords[-1]: coords.append(coords[0])
        return {"type": "Polygon", "coordinates": [coords]}
    if elem["type"] == "relation" and "members" in elem:
        outer_coords = []
        for member in elem["members"]:
            if member.get("role") == "outer" and "geometry" in member:
                ring = [[n["lon"], n["lat"]] for n in member["geometry"]]
                if ring and ring[0] != ring[-1]: ring.append(ring[0])
                outer_coords.append(ring)
        if outer_coords:
            if len(outer_coords) == 1: return {"type": "Polygon", "coordinates": outer_coords}
            return {"type": "MultiPolygon", "coordinates": [[ring] for ring in outer_coords]}
    return None

def _parse_int(val, default):
    try: return int(float(val))
    except: return default

def _parse_float(val, default):
    try: return float(val)
    except: return default

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campus", choices=["ntu", "ncu"], default="ntu")
    parser.add_argument("--output", "-o")
    args = parser.parse_args()
    bbox = NTU_BBOX if args.campus == "ntu" else NCU_BBOX
    output = resolve_project_path(args.output) if args.output else campus_data_dir(args.campus, "osm_buildings.geojson")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    geojson = fetch_campus_buildings(bbox)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)
    print(f"[OSM Fetcher] 已儲存: {output}")

if __name__ == "__main__":
    main()
