"""
Build a canonical NCU building name dictionary by combining:
  1. OSM Overpass API — all `building`-tagged ways within NCU bbox with `name`
  2. Google Places (New) Text Search — for category sweeps NCU 各種 X館 / X舍 / X中心
  3. Existing campuses/ncu/data/buildings.geojson (current OSM snapshot we already have)

Output:
  outputs/_cleaning_diagnosis/ncu_canonical_names.json
       {
         "names": [...],            # deduplicated list of canonical names
         "by_source": {"osm": [...], "google": [...], "geojson": [...]},
         "by_name": {name: {"sources": [...], "lat": ..., "lon": ...}}
       }

Reads $GOOGLE_MAPS_API_KEY from env. OSM doesn't need a key.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "outputs" / "_cleaning_diagnosis" / "ncu_canonical_names.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
GEOJSON_PATH = ROOT / "campuses" / "ncu" / "data" / "buildings.geojson"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

# NCU campus bbox (from campuses/ncu/config.yaml)
NCU_BBOX = {
    "south": 24.96, "west": 121.183,
    "north": 24.978, "east": 121.206,
}
NCU_CENTER = (24.9684, 121.1946)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Google Places (New) query templates — sweep for common NCU building categories
GOOGLE_QUERIES = [
    "中央大學 館", "中央大學 宿舍", "中央大學 中心", "中央大學 實驗室",
    "中央大學 學院", "中央大學 大樓", "中央大學 體育", "中央大學 圖書館",
    "中央大學 餐廳", "中央大學 工程", "中央大學 科學",
    # English
    "National Central University building", "NCU dormitory", "NCU laboratory",
]


def fetch_osm_overpass() -> list[dict]:
    """Return list of {name, name_zh, name_en, lat, lon} for buildings in NCU bbox."""
    query = f"""
    [out:json][timeout:30];
    (
      way["building"]({NCU_BBOX['south']},{NCU_BBOX['west']},{NCU_BBOX['north']},{NCU_BBOX['east']});
      relation["building"]({NCU_BBOX['south']},{NCU_BBOX['west']},{NCU_BBOX['north']},{NCU_BBOX['east']});
    );
    out center tags;
    """
    print(f"[OSM Overpass] querying NCU bbox…")
    headers = {"User-Agent": "ncu-energy-demo/0.1 (cleaning pipeline)"}
    r = requests.post(OVERPASS_URL, data={"data": query},
                      headers=headers, timeout=60)
    r.raise_for_status()
    elements = r.json().get("elements", [])
    out = []
    for e in elements:
        tags = e.get("tags", {}) or {}
        name = tags.get("name") or tags.get("name:zh") or tags.get("name:zh-TW")
        if not name:
            continue
        cen = e.get("center") or {}
        out.append({
            "osm_id": e.get("id"),
            "type": e.get("type"),
            "name": name.strip(),
            "name_zh": (tags.get("name:zh") or tags.get("name:zh-TW") or "").strip(),
            "name_en": (tags.get("name:en") or "").strip(),
            "building_type": tags.get("building", ""),
            "lat": cen.get("lat") or e.get("lat"),
            "lon": cen.get("lon") or e.get("lon"),
        })
    print(f"             got {len(out)} named OSM buildings")
    return out


def fetch_google_places(api_key: str) -> list[dict]:
    """Sweep Google Places New for NCU building names."""
    if not api_key:
        print("[Google Places] no API key, skipping")
        return []

    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.id,places.displayName,places.location,places.types,places.primaryType",
    }
    seen_ids: set[str] = set()
    out: list[dict] = []
    for q in GOOGLE_QUERIES:
        body = {
            "textQuery": q,
            "locationBias": {
                "circle": {
                    "center": {"latitude": NCU_CENTER[0], "longitude": NCU_CENTER[1]},
                    "radius": 1500.0,
                }
            },
            "maxResultCount": 20,
        }
        try:
            r = requests.post(url, json=body, headers=headers, timeout=15)
            r.raise_for_status()
            places = r.json().get("places", [])
        except Exception as exc:
            print(f"[Google Places] query {q!r} failed: {exc}")
            continue
        kept = 0
        for p in places:
            pid = p.get("id", "")
            if pid in seen_ids:
                continue
            loc = p.get("location") or {}
            lat = loc.get("latitude")
            lon = loc.get("longitude")
            if lat is None or lon is None:
                continue
            # Filter to NCU bbox
            if not (NCU_BBOX["south"] <= lat <= NCU_BBOX["north"]
                    and NCU_BBOX["west"] <= lon <= NCU_BBOX["east"]):
                continue
            display = (p.get("displayName") or {}).get("text", "")
            if not display:
                continue
            seen_ids.add(pid)
            out.append({
                "place_id": pid,
                "name": display.strip(),
                "primary_type": p.get("primaryType", ""),
                "types": p.get("types", []),
                "lat": lat,
                "lon": lon,
            })
            kept += 1
        print(f"  query {q!r} → {kept} new place(s)")
        time.sleep(0.15)
    print(f"[Google Places] total unique within NCU bbox: {len(out)}")
    return out


def load_geojson_names() -> list[dict]:
    """Read existing campuses/ncu/data/buildings.geojson named features."""
    if not GEOJSON_PATH.exists():
        return []
    gj = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
    out = []
    for ft in gj.get("features", []):
        p = ft.get("properties", {}) or {}
        name = (p.get("name") or "").strip()
        if not name:
            continue
        out.append({
            "osm_id": p.get("osm_id"),
            "name": name,
            "building_type": p.get("building_type", ""),
        })
    print(f"[geojson] {len(out)} named features in current buildings.geojson")
    return out


# Common Chinese normalization for dedup keys (collapse parens etc)
_PAREN = re.compile(r"[（()）]")
_SP = re.compile(r"\s+")


def name_key(s: str) -> str:
    return _SP.sub("", _PAREN.sub("", (s or "").strip())).lower()


def main():
    osm_rows = fetch_osm_overpass()
    google_rows = fetch_google_places(os.environ.get("GOOGLE_MAPS_API_KEY", "").strip())
    geojson_rows = load_geojson_names()

    # Combine + dedupe by name_key
    by_key: dict[str, dict] = {}
    for src, rows in [("osm", osm_rows), ("google", google_rows), ("geojson", geojson_rows)]:
        for r in rows:
            k = name_key(r["name"])
            if not k or len(k) < 2:
                continue
            entry = by_key.setdefault(k, {
                "name": r["name"],
                "sources": [],
                "lat": r.get("lat"),
                "lon": r.get("lon"),
                "building_type": r.get("building_type", ""),
            })
            if src not in entry["sources"]:
                entry["sources"].append(src)
            # Prefer OSM canonical name (more official) over Google fuzzy text
            if src == "osm" and entry.get("source_priority", "google") != "osm":
                entry["name"] = r["name"]
                entry["lat"] = r.get("lat") or entry["lat"]
                entry["lon"] = r.get("lon") or entry["lon"]
                entry["source_priority"] = "osm"

    canonical_names = sorted({e["name"] for e in by_key.values()})
    by_source = {
        "osm": sorted({r["name"] for r in osm_rows}),
        "google": sorted({r["name"] for r in google_rows}),
        "geojson": sorted({r["name"] for r in geojson_rows}),
    }

    out_doc = {
        "ncu_bbox": NCU_BBOX,
        "n_unique_canonical": len(canonical_names),
        "names": canonical_names,
        "by_source": by_source,
        "by_name": by_key,
    }
    OUT_JSON.write_text(json.dumps(out_doc, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print()
    print(f"Total unique canonical names: {len(canonical_names)}")
    print(f"  OSM only:   {len(by_source['osm'])}")
    print(f"  Google only:{len(by_source['google'])}")
    print(f"  geojson:    {len(by_source['geojson'])}")
    print(f"Wrote: {OUT_JSON}")


if __name__ == "__main__":
    main()
