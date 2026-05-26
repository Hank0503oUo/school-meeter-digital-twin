"""
Apply focused NCU demo geometry fixes.

This patch keeps the demo-facing map tidy and traceable:
  - remove known off-campus private buildings
  - replace coarse synthesized blocks with Google/CODiS/OSM-backed footprints
  - add missing demo targets such as swimming pool and research facilities
  - keep user-supplied meter IDs as explicit feature metadata
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.google_fetcher import fetch_building_details
from src.utils import geometry_footprint_m2


NCU_DATA = ROOT / "campuses" / "ncu" / "data"
BUILDINGS = NCU_DATA / "buildings.geojson"
ENERGY = NCU_DATA / "energy.geojson"
ENHANCED = ROOT / "data" / "NCU" / "buildings_enhanced.geojson"
GOOGLE_CACHE = ROOT / "data" / "cache" / "google_maps"
AUDIT = ROOT / "outputs" / "ncu_114" / "demo_geometry_fixes_audit.csv"
METER_LINKS = ROOT / "outputs" / "ncu_114" / "swimming_pool_meter_links.csv"

REMOVE_OSM_IDS = {
    1346093070,  # off-campus Shuiyue Manor
    217833430,  # original OSM Taiyao footprint; retained through demo UID 9000000005
    268561940,  # original OSM Engineering 5 footprint; retained through demo UID 9000000008
    9000000009,  # duplicate Google point for Engineering 5 B extension
    9000000010,  # stale synthetic Engineering 4
    9000000011,  # duplicate Google point for Engineering 5 C
}
REMOVE_NAMES = {"水悅莊園", "水悦莊園"}

GOOGLE_PLACES = {
    "engineering_4": "ChIJX6lfAecjaDQRxZrfwiSMDJ4",
    "covestro": "ChIJ1YwkdeYjaDQRxWCnQET7SHA",
    "wind_qa": "ChIJJ2D5XuojaDQRTWRj5piL3Kg",
    "pool": "ChIJGatd5-sjaDQRb0Ok7x-cBmc",
    "research_center_2": "ChIJbR9PC-IjaDQRmqAOk7gZfa8",
    "electrical_engineering": "ChIJKfJC_-ojaDQRoF9VMT1dESI",
}


def load_geojson(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_geojson(path: Path, gj: dict) -> None:
    path.write_text(json.dumps(gj, ensure_ascii=False, indent=2), encoding="utf-8")


def first_google_polygon(place_id: str) -> dict:
    cache_path = GOOGLE_CACHE / f"details_{place_id}.json"
    if cache_path.exists():
        details = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
        if not api_key:
            raise FileNotFoundError(
                f"Missing cached Google building details for {place_id} and GOOGLE_MAPS_API_KEY is not set."
            )
        details = fetch_building_details(place_id, api_key)
        if not details:
            raise RuntimeError(f"Google building details unavailable for {place_id}")

    for result in details.get("results", []):
        for building in result.get("buildings", []):
            for outline in building.get("building_outlines", []):
                poly = outline.get("display_polygon")
                if poly:
                    return poly
    raise RuntimeError(f"No display_polygon found for Google place {place_id}")


def enhanced_geometry(name: str) -> tuple[dict, dict]:
    gj = load_geojson(ENHANCED)
    for ft in gj.get("features", []):
        props = ft.get("properties", {}) or {}
        if (props.get("name") or "") == name:
            return ft.get("geometry") or {}, props
    raise RuntimeError(f"Enhanced OSM feature not found: {name}")


def square_polygon(lat: float, lon: float, half_side_m: float) -> dict:
    import math

    dlat = half_side_m / 111_132.0
    dlon = half_side_m / (111_320.0 * math.cos(math.radians(lat)))
    ring = [
        [lon - dlon, lat - dlat],
        [lon + dlon, lat - dlat],
        [lon + dlon, lat + dlat],
        [lon - dlon, lat + dlat],
        [lon - dlon, lat - dlat],
    ]
    return {"type": "Polygon", "coordinates": [ring]}


def remove_targets(gj: dict) -> int:
    kept = []
    removed = 0
    for ft in gj.get("features", []):
        props = ft.get("properties", {}) or {}
        name = str(props.get("name") or "").strip()
        try:
            oid = int(props.get("osm_id"))
        except (TypeError, ValueError):
            oid = None
        if oid in REMOVE_OSM_IDS or name in REMOVE_NAMES:
            removed += 1
            continue
        kept.append(ft)
    gj["features"] = kept
    return removed


def canonical_props(
    *,
    osm_id: int,
    name: str,
    geometry: dict,
    building_type: str,
    levels: int,
    source: str,
    aliases: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict:
    area = round(geometry_footprint_m2(geometry), 2)
    props: dict[str, Any] = {
        "osm_id": int(osm_id),
        "name": name,
        "building_type": building_type,
        "operator": "National Central University",
        "levels": int(levels),
        "height": round(float(levels) * 3.5, 1),
        "height_m": round(float(levels) * 3.5, 1),
        "footprint_area_m2": area,
        "mean_kw": 0.0,
        "annual_kwh": 0.0,
        "annual_mwh": 0.0,
        "peak_kw": 0.0,
        "eui": 0.0,
        "eui_kw_per_m2": 0.0,
        "load_factor": 0.0,
        "archetype_label": "Academic",
        "data_source": source,
        "geometry_source": source,
        "has_meter_data": False,
        "building_code": "",
        "meter_name": "",
    }
    if aliases:
        props["name_aliases"] = aliases
    if extra:
        props.update(extra)
    return props


def upsert_feature(gj: dict, feature: dict) -> str:
    target_id = int(feature["properties"]["osm_id"])
    for ft in gj.get("features", []):
        props = ft.get("properties", {}) or {}
        try:
            oid = int(props.get("osm_id"))
        except (TypeError, ValueError):
            oid = None
        if oid == target_id:
            ft["geometry"] = feature["geometry"]
            props.update(feature["properties"])
            ft["properties"] = props
            return "updated"
    gj.setdefault("features", []).append(feature)
    return "added"


def ensure_alias(gj: dict, osm_id: int, aliases: list[str], extra: dict[str, Any] | None = None) -> str:
    for ft in gj.get("features", []):
        props = ft.get("properties", {}) or {}
        try:
            oid = int(props.get("osm_id"))
        except (TypeError, ValueError):
            oid = None
        if oid == osm_id:
            current = props.get("name_aliases") or []
            if isinstance(current, str):
                current = [x.strip() for x in current.split("|") if x.strip()]
            props["name_aliases"] = sorted(set(current).union(aliases))
            if extra:
                props.update(extra)
            return "updated"
    return "missing"


def feature(osm_id: int, name: str, geometry: dict, **kwargs: Any) -> dict:
    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": canonical_props(osm_id=osm_id, name=name, geometry=geometry, **kwargs),
    }


def main() -> None:
    buildings = load_geojson(BUILDINGS)
    energy = load_geojson(ENERGY)
    audit_rows = []

    for label, gj in [("buildings", buildings), ("energy", energy)]:
        removed = remove_targets(gj)
        audit_rows.append({"target": label, "uid": "", "name": "remove_off_campus_and_stale_synthetic", "action": f"removed_{removed}"})

    env_geom, env_props = enhanced_geometry("環工所")
    space_geom, space_props = enhanced_geometry("太空及遙測研究中心")
    eng5_geom, eng5_props = enhanced_geometry("工程五館")
    fixes = [
        feature(
            9000000005,
            "太空遙測中心",
            space_geom,
            building_type="university",
            levels=int(space_props.get("levels") or 3),
            source="osm_enhanced_composite",
            aliases=[
                "太空及遙測研究中心",
                "國立中央大學太空及遙測研究中心",
                "National Central University Research Center for Space and Remote Sensing",
            ],
            extra={
                "building_code": "R1",
                "source_osm_id": 217833430,
                "source_osm_name": "太空及遙測研究中心",
                "geometry_note": "Demo fix: replaces 400 m2 Google point with the full yellow OSM footprint.",
            },
        ),
        feature(
            9000000008,
            "工五館(A、B棟)",
            eng5_geom,
            building_type="university",
            levels=int(eng5_props.get("levels") or 3),
            source="osm_enhanced_composite",
            aliases=[
                "工程五館",
                "工程五館大樓",
                "工五館B棟増建",
                "工程五館C棟",
                "國立中央大學 工程五館",
            ],
            extra={
                "building_code": "E6",
                "source_osm_id": 268561940,
                "source_osm_name": "工程五館",
                "geometry_note": "Demo fix: restores the left-side H-shaped Engineering 5 footprint and folds B/C meter names into this feature.",
            },
        ),
        feature(
            268561422,
            "工四館一期(環化館)",
            env_geom,
            building_type="university",
            levels=int(env_props.get("levels") or 3),
            source="osm_enhanced_alias",
            aliases=["工程四館一期(環化館)", "工程四館", "環工所", "環工化工館", "化工所"],
            extra={"building_code": "E3", "source_osm_name": "環工所"},
        ),
        feature(
            9000000003,
            "垃圾集中處理場",
            square_polygon(24.969867, 121.195939, 10.0),
            building_type="service",
            levels=1,
            source="google_place_point_synthesized",
            aliases=["中央大學 男九舍後垃圾場"],
            extra={"google_lat": 24.969867, "google_lon": 121.195939},
        ),
        feature(
            9000000013,
            "產學營運中心",
            square_polygon(24.966744, 121.190847, 10.0),
            building_type="university",
            levels=3,
            source="google_place_point_synthesized",
            aliases=["Center for Academia and Industry Collaboration"],
            extra={"google_lat": 24.966744, "google_lon": 121.190847},
        ),
        feature(
            9000000019,
            "電機館",
            first_google_polygon(GOOGLE_PLACES["electrical_engineering"]),
            building_type="university",
            levels=4,
            source="google_geocoding_building_outline",
            aliases=["國立中央大學 電機工程學系", "工二館", "資訊電機系館"],
            extra={"google_place_id": GOOGLE_PLACES["electrical_engineering"]},
        ),
        feature(
            9000000021,
            "機電實驗室",
            first_google_polygon(GOOGLE_PLACES["engineering_4"]),
            building_type="university",
            levels=3,
            source="google_geocoding_building_outline",
            aliases=["工程四館二期(機電實驗室)", "工程四館二期", "Mechatronics Lab"],
            extra={"building_code": "E4", "google_place_id": GOOGLE_PLACES["engineering_4"]},
        ),
        feature(
            9000000022,
            "風洞實驗室及品保中心",
            first_google_polygon(GOOGLE_PLACES["wind_qa"]),
            building_type="university",
            levels=2,
            source="google_geocoding_building_outline",
            aliases=["Center of Quality Assurance for Civil Material", "土木品保中心", "風洞實驗室"],
            extra={"google_place_id": GOOGLE_PLACES["wind_qa"]},
        ),
        feature(
            9000000023,
            "科思創研究中心",
            first_google_polygon(GOOGLE_PLACES["covestro"]),
            building_type="university",
            levels=3,
            source="google_geocoding_building_outline",
            aliases=["科思創全球能量固化研發中心@中央大學（桃園）", "Covestro Research Center"],
            extra={"google_place_id": GOOGLE_PLACES["covestro"]},
        ),
        feature(
            9000000024,
            "研究中心大樓二期",
            first_google_polygon(GOOGLE_PLACES["research_center_2"]),
            building_type="university",
            levels=5,
            source="google_geocoding_building_outline",
            aliases=["Research Center Building 2", "生醫理工學院辦公室", "國立中央大學生醫科學與工程學系"],
            extra={"building_code": "R3", "google_place_id": GOOGLE_PLACES["research_center_2"]},
        ),
        feature(
            9000000025,
            "游泳池",
            first_google_polygon(GOOGLE_PLACES["pool"]),
            building_type="sports_centre",
            levels=2,
            source="google_geocoding_building_outline",
            aliases=["國立中央大學游泳館", "室內游泳池", "中大國民運動中心游泳池"],
            extra={
                "meter_ids": ["6660968", "60200330"],
                "meter_note": "6660968 user-supplied electricity submeter; 60200330 is the water tower meter behind the indoor pool.",
                "parent_osm_id": 1041679633,
                "google_place_id": GOOGLE_PLACES["pool"],
            },
        ),
        feature(
            9000000002,
            "地下水4號水塔",
            square_polygon(24.96974, 121.18995, 6.0),
            building_type="service",
            levels=1,
            source="manual_meter_location_pool_rear",
            aliases=["室內游泳池後方水塔", "游泳池後方水塔"],
            extra={
                "meter_ids": ["60200330"],
                "meter_note": "Meter audit location: 室內游泳池後方水塔.",
                "google_lat": 24.96974,
                "google_lon": 121.18995,
            },
        ),
    ]

    for gj_label, gj in [("buildings", buildings), ("energy", energy)]:
        for ft in fixes:
            action = upsert_feature(gj, json.loads(json.dumps(ft, ensure_ascii=False)))
            props = ft["properties"]
            audit_rows.append({
                "target": gj_label,
                "uid": f"NCU_{props['osm_id']}",
                "name": props["name"],
                "action": action,
                "footprint_area_m2": props.get("footprint_area_m2"),
                "geometry_source": props.get("geometry_source"),
            })
        action = ensure_alias(
            gj,
            268561934,
            ["工程四館三期(大型力學實驗室)", "工程四館三期"],
        )
        audit_rows.append({"target": gj_label, "uid": "NCU_268561934", "name": "大型力學實驗室", "action": action})
        action = ensure_alias(
            gj,
            1041679633,
            ["中大國民運動中心", "國立中央大學游泳館"],
            {"facility_note": "Sports center footprint kept; swimming pool also added as a dedicated demo feature."},
        )
        audit_rows.append({"target": gj_label, "uid": "NCU_1041679633", "name": "中大國民運動中心", "action": action})

    write_geojson(BUILDINGS, buildings)
    write_geojson(ENERGY, energy)
    pd.DataFrame(audit_rows).to_csv(AUDIT, index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {
                "facility": "游泳池",
                "uid": "NCU_9000000025",
                "meter_id": "6660968",
                "meter_role": "electricity_submeter",
                "source": "user_supplied",
                "note": "114-year meter_audit has no valid row; older audits list it under 體育館(依仁堂).",
            },
            {
                "facility": "游泳池後方水塔",
                "uid": "NCU_9000000002",
                "meter_id": "60200330",
                "meter_role": "water_tower",
                "source": "meter_audit_114",
                "note": "Location in audit: 室內游泳池後方水塔; 2025 annual kWh is assigned to 地下水4號水塔.",
            },
        ]
    ).to_csv(METER_LINKS, index=False, encoding="utf-8-sig")

    print(f"Wrote {BUILDINGS}")
    print(f"Wrote {ENERGY}")
    print(f"Wrote {AUDIT}")
    print(f"Wrote {METER_LINKS}")


if __name__ == "__main__":
    main()
