"""
Enrich campuses/ncu/data/buildings.geojson with real building height & floors
estimated from Google Solar API.

For each named building, query Solar API → roofSegmentStats[*].planeHeightAtCenterMeters
(meters above mean sea level). We approximate:
  ground_amsl  ≈ min(plane_heights across this building's roof segments)
  height_m     ≈ max(plane_heights) - ground_amsl
  levels       ≈ max(1, round(height_m / 3.5))   # 3.5 m per floor

Caveats logged in stdout:
  - Solar API ground reference is the LOWEST roof plane, not actual terrain;
    for buildings with rooftop appurtenances only this works fine, for buildings
    with porches/extensions at ground level it slightly under-counts height.
  - We clamp levels to [1, 20] to ignore obviously wrong readings.
  - Buildings without a Solar API response keep the existing levels (default 3).

Reads $GOOGLE_MAPS_API_KEY from environment. Run via:
  $env:GOOGLE_MAPS_API_KEY="..."; python scripts/ncu_cleaning/enhance_geojson_with_solar.py

Output:
  campuses/ncu/data/buildings.geojson  (in-place; original backed up to *.bak.json)
  outputs/ncu_114/solar_height_audit.csv  per-building enrichment record
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.solar_api import (
    SolarAPIError,
    SolarAPIRequestError,
    get_google_maps_api_key,
    probe_building_insights,
)

NCU_GEOJSON = ROOT / "campuses" / "ncu" / "data" / "buildings.geojson"
BACKUP = NCU_GEOJSON.with_suffix(".bak.json")
AUDIT_CSV = ROOT / "outputs" / "ncu_114" / "solar_height_audit.csv"

FLOOR_HEIGHT_M = 3.5
MIN_LEVELS = 1
MAX_LEVELS = 20


def polygon_centroid(geometry: dict) -> tuple[float, float] | None:
    """Return (lat, lon) centroid of the first polygon ring."""
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if not coords:
        return None
    if gtype == "Polygon":
        ring = coords[0]
    elif gtype == "MultiPolygon":
        ring = coords[0][0]
    else:
        return None
    if not ring:
        return None
    n = len(ring)
    lon = sum(p[0] for p in ring) / n
    lat = sum(p[1] for p in ring) / n
    return lat, lon


def estimate_levels(probe_result) -> tuple[float | None, int | None, int, str]:
    """Returns (height_m, levels, n_segments, note)."""
    sp = probe_result.data.get("solarPotential", {}) or {}
    segs = sp.get("roofSegmentStats", []) or []
    heights = [s.get("planeHeightAtCenterMeters") for s in segs
               if s.get("planeHeightAtCenterMeters") is not None]
    if not heights:
        return None, None, 0, "no_height_data"
    if len(heights) == 1:
        return None, None, 1, "single_plane_no_span"
    span = max(heights) - min(heights)
    levels = max(MIN_LEVELS, min(MAX_LEVELS, round(span / FLOOR_HEIGHT_M)))
    return round(span, 2), levels, len(heights), "ok"


def main():
    try:
        api_key = get_google_maps_api_key()
    except SolarAPIError as exc:
        print(f"[FATAL] {exc}")
        sys.exit(1)

    print(f"Loading {NCU_GEOJSON}")
    gj = json.loads(NCU_GEOJSON.read_text(encoding="utf-8"))

    # Backup original
    if not BACKUP.exists():
        BACKUP.write_text(json.dumps(gj, ensure_ascii=False), encoding="utf-8")
        print(f"  backup written: {BACKUP.name}")

    AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)

    audit_rows = []
    n_named = sum(1 for ft in gj["features"]
                  if (ft.get("properties", {}).get("name") or "").strip())
    print(f"Named buildings to query: {n_named}")
    print()

    for i, ft in enumerate(gj["features"]):
        p = ft.setdefault("properties", {})
        name = (p.get("name") or "").strip()
        osm_id = p.get("osm_id")
        if not name:
            continue   # skip unnamed (small structures)
        cen = polygon_centroid(ft.get("geometry") or {})
        if cen is None:
            print(f"  [{i:3d}] {name}: no geometry, skip")
            audit_rows.append({
                "osm_id": osm_id, "name": name, "lat": None, "lon": None,
                "height_m": None, "levels_old": p.get("levels"),
                "levels_new": p.get("levels"), "n_segments": 0,
                "note": "no_geometry",
            })
            continue
        lat, lon = cen
        try:
            res = probe_building_insights(lat, lon, api_key,
                                          initial_quality="HIGH",
                                          allow_base_fallback=True,
                                          timeout=15.0)
        except SolarAPIRequestError as exc:
            note = f"api_error_{exc.status_code}"
            print(f"  [{i:3d}] {name}: {note} ({str(exc)[:60]})")
            audit_rows.append({
                "osm_id": osm_id, "name": name, "lat": lat, "lon": lon,
                "height_m": None, "levels_old": p.get("levels"),
                "levels_new": p.get("levels"), "n_segments": 0,
                "note": note,
            })
            continue
        except Exception as exc:
            note = f"exception_{type(exc).__name__}"
            print(f"  [{i:3d}] {name}: {note}")
            audit_rows.append({
                "osm_id": osm_id, "name": name, "lat": lat, "lon": lon,
                "height_m": None, "levels_old": p.get("levels"),
                "levels_new": p.get("levels"), "n_segments": 0,
                "note": note,
            })
            continue

        height_m, levels, n_seg, note = estimate_levels(res)
        old_levels = p.get("levels")
        if levels is not None:
            p["levels"] = levels
            p["height_m"] = height_m
            p["height_source"] = "google_solar_api"
            print(f"  [{i:3d}] {name:24s} h={height_m:5.1f}m -> levels={levels} (was {old_levels}, n_seg={n_seg})")
        else:
            print(f"  [{i:3d}] {name:24s} {note} (kept old levels={old_levels})")
        audit_rows.append({
            "osm_id": osm_id, "name": name, "lat": lat, "lon": lon,
            "height_m": height_m, "levels_old": old_levels,
            "levels_new": p.get("levels"), "n_segments": n_seg, "note": note,
        })
        # gentle pacing — Solar API has per-second rate limits
        time.sleep(0.05)

    NCU_GEOJSON.write_text(json.dumps(gj, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print()
    print(f"Wrote enriched {NCU_GEOJSON}")

    # Audit CSV
    import csv
    with open(AUDIT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
        w.writeheader()
        w.writerows(audit_rows)
    print(f"Wrote audit {AUDIT_CSV}")

    # Summary
    n_ok = sum(1 for r in audit_rows if r["note"] == "ok")
    n_skip = len(audit_rows) - n_ok
    print()
    print(f"Summary: {n_ok}/{len(audit_rows)} buildings successfully enriched, "
          f"{n_skip} kept original levels")


if __name__ == "__main__":
    main()
