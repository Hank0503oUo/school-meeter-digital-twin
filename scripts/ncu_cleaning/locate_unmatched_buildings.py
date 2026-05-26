"""
Use Google Places Text Search to locate the 21 unmatched NCU buildings,
then either alias them to a nearby OSM building polygon (if one is close enough)
or inject a synthetic 20m × 20m square footprint at the returned coordinates.

Reads $GOOGLE_MAPS_API_KEY from env.

Pipeline:
  1. Read outputs/ncu_114/name_to_uid.csv → list of unmatched building names
  2. For each, call Google Places (New) Text Search around NCU centroid
     with query like  "<name> 中央大學 中壢" (radius 1.5 km)
  3. Pick the first result whose coords are within 1.5 km of NCU center
  4. Find nearest OSM building polygon in current buildings.geojson:
       - if distance ≤ 30 m AND that polygon is currently nameless → ALIAS:
             write the unmatched name into that polygon's properties.name
       - else → INJECT synthetic feature at the lat/lon
  5. Re-run match_buildings + populate_ncu_campus + build_yearly_inference_cache

Outputs:
  buildings.geojson (in-place; backup buildings.bak2.json)
  outputs/ncu_114/google_locate_audit.csv
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import pandas as pd

# PowerShell on Windows is cp950 by default; force UTF-8 with replacement so
# rare CJK glyphs (e.g. 増 U+5897) don't crash mid-loop.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.google_fetcher import search_place_by_text, latlon_to_meters

NCU_GEOJSON = ROOT / "campuses" / "ncu" / "data" / "buildings.geojson"
BACKUP = NCU_GEOJSON.with_suffix(".bak2.json")
NAME_MAP_CSV = ROOT / "outputs" / "ncu_114" / "name_to_uid.csv"
AUDIT_CSV = ROOT / "outputs" / "ncu_114" / "google_locate_audit.csv"

NCU_CENTER_LAT, NCU_CENTER_LON = 24.9684, 121.1946
SEARCH_RADIUS_M = 1500
NCU_BBOX_RADIUS_M = 1500   # max distance from center to accept a Google hit
ALIAS_RADIUS_M = 30        # max distance to alias an existing nameless polygon
SYNTHETIC_HALF_SIDE_M = 10 # synthetic square half-side


def deg_offset_for_meters(lat: float, dx_m: float, dy_m: float) -> tuple[float, float]:
    """Return (dlon, dlat) for a point dx_m east, dy_m north of (lat, ...)."""
    dlat = dy_m / 111000.0
    dlon = dx_m / (111000.0 * math.cos(math.radians(lat)))
    return dlon, dlat


def make_synthetic_polygon(lat: float, lon: float,
                           half_side: float = SYNTHETIC_HALF_SIDE_M) -> list:
    dlon, dlat = deg_offset_for_meters(lat, half_side, half_side)
    return [[
        [lon - dlon, lat - dlat],
        [lon + dlon, lat - dlat],
        [lon + dlon, lat + dlat],
        [lon - dlon, lat + dlat],
        [lon - dlon, lat - dlat],
    ]]


def polygon_centroid(geometry: dict) -> tuple[float, float] | None:
    g = geometry.get("type")
    coords = geometry.get("coordinates")
    if not coords:
        return None
    if g == "Polygon":
        ring = coords[0]
    elif g == "MultiPolygon":
        ring = coords[0][0]
    else:
        return None
    n = len(ring)
    return sum(p[1] for p in ring) / n, sum(p[0] for p in ring) / n


def polygon_area_m2(geometry: dict) -> float:
    """Rough projected area in m² for a small polygon."""
    g = geometry.get("type")
    coords = geometry.get("coordinates")
    if not coords:
        return 0.0
    ring = coords[0] if g == "Polygon" else coords[0][0] if g == "MultiPolygon" else None
    if not ring or len(ring) < 4:
        return 0.0
    lat_mid = sum(p[1] for p in ring) / len(ring)
    mlat = 111000.0
    mlon = 111000.0 * math.cos(math.radians(lat_mid))
    pts = [(p[0] * mlon, p[1] * mlat) for p in ring]
    area = 0.0
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def find_nearest_nameless(geo_json: dict, lat: float, lon: float):
    best = None
    best_d = float("inf")
    for ft in geo_json["features"]:
        p = ft.get("properties", {})
        name = (p.get("name") or "").strip()
        if name:
            continue   # already named
        cen = polygon_centroid(ft.get("geometry") or {})
        if cen is None:
            continue
        d = latlon_to_meters(lat, lon, cen[0], cen[1])
        if d < best_d:
            best_d = d
            best = ft
    return best, best_d


def main():
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    if not api_key:
        print("[FATAL] GOOGLE_MAPS_API_KEY not set")
        sys.exit(1)

    # Load unmatched names
    name_map = pd.read_csv(NAME_MAP_CSV, encoding="utf-8-sig")
    unmatched = name_map[name_map["match_type"] == "unmatched"]["building"].tolist()
    print(f"unmatched buildings: {len(unmatched)}")

    # Load current geojson
    gj = json.loads(NCU_GEOJSON.read_text(encoding="utf-8"))
    if not BACKUP.exists():
        BACKUP.write_text(json.dumps(gj, ensure_ascii=False), encoding="utf-8")
        print(f"  backup: {BACKUP.name}")

    audit_rows = []
    next_synthetic_id = 9_000_000_001  # synthetic UIDs start here

    for i, name in enumerate(unmatched, 1):
        # Query Google
        query = f"{name} 中央大學 中壢"
        results = search_place_by_text(query, NCU_CENTER_LAT, NCU_CENTER_LON,
                                        SEARCH_RADIUS_M, api_key)
        time.sleep(0.1)
        if not results:
            print(f"  [{i:2d}] {name:30s}  NO_GOOGLE_HIT")
            audit_rows.append({"name": name, "status": "no_google_hit",
                               "lat": None, "lon": None, "action": None,
                               "target_osm_id": None, "distance_m": None})
            continue

        # Filter results within NCU bbox
        place = None
        for r in results:
            loc = r.get("location") or {}
            la = loc.get("latitude")
            lo = loc.get("longitude")
            if la is None or lo is None:
                continue
            d_center = latlon_to_meters(NCU_CENTER_LAT, NCU_CENTER_LON, la, lo)
            if d_center <= NCU_BBOX_RADIUS_M:
                place = r
                break
        if not place:
            print(f"  [{i:2d}] {name:30s}  OUT_OF_CAMPUS (top hit > 1.5km)")
            audit_rows.append({"name": name, "status": "out_of_campus",
                               "lat": None, "lon": None, "action": None,
                               "target_osm_id": None, "distance_m": None})
            continue

        loc = place["location"]
        lat, lon = loc["latitude"], loc["longitude"]
        display = (place.get("displayName") or {}).get("text", "")

        # Try to alias nearest nameless polygon if within ALIAS_RADIUS_M
        nearest, d = find_nearest_nameless(gj, lat, lon)
        if nearest is not None and d <= ALIAS_RADIUS_M:
            nearest_props = nearest["properties"]
            old_id = nearest_props.get("osm_id")
            nearest_props["name"] = name
            nearest_props["name_source"] = "google_places_alias"
            print(f"  [{i:2d}] {name:30s}  ALIAS osm_id={old_id} (d={d:.1f}m)  google={display!r}")
            audit_rows.append({"name": name, "status": "aliased",
                               "lat": round(lat, 6), "lon": round(lon, 6),
                               "action": "alias_existing", "target_osm_id": old_id,
                               "distance_m": round(d, 1)})
        else:
            # Inject synthetic feature
            uid = next_synthetic_id
            next_synthetic_id += 1
            poly = make_synthetic_polygon(lat, lon)
            new_feat = {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": poly},
                "properties": {
                    "osm_id": uid,
                    "name": name,
                    "building_type": "yes",
                    "operator": "",
                    "levels": 3,
                    "height": 10.5,
                    "footprint_area_m2": round(polygon_area_m2({
                        "type": "Polygon", "coordinates": poly}), 1),
                    "data_source": "google_synthesized",
                    "name_source": "google_places_synthesized",
                    "google_place_name": display,
                    "google_lat": round(lat, 6),
                    "google_lon": round(lon, 6),
                },
            }
            gj["features"].append(new_feat)
            print(f"  [{i:2d}] {name:30s}  SYNTH uid={uid} (nearest nameless d={d:.0f}m)  google={display!r}")
            audit_rows.append({"name": name, "status": "synthesized",
                               "lat": round(lat, 6), "lon": round(lon, 6),
                               "action": "inject_synthetic", "target_osm_id": uid,
                               "distance_m": round(d, 1) if d != float("inf") else None})

    # Write geojson back
    NCU_GEOJSON.write_text(json.dumps(gj, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print()
    print(f"Wrote enriched {NCU_GEOJSON}")

    # Audit CSV
    pd.DataFrame(audit_rows).to_csv(AUDIT_CSV, index=False, encoding="utf-8-sig")
    print(f"Audit: {AUDIT_CSV}")

    counts = pd.Series([r["status"] for r in audit_rows]).value_counts()
    print()
    print("=== Summary ===")
    print(counts.to_string())


if __name__ == "__main__":
    main()
