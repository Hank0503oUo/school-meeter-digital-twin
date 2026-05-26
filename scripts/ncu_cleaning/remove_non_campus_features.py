"""
Remove known non-campus buildings from the NCU demo data products.

The campus OSM extract includes a few nearby private buildings that render
inside the demo viewport.  For the NCU demo we keep only campus facilities, so
this script removes those features from the active GeoJSON, metadata, topology,
and inference caches.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
NCU_DATA = ROOT / "campuses" / "ncu" / "data"
NCU_MODELS = ROOT / "campuses" / "ncu" / "models"
OUT_DIR = ROOT / "outputs" / "ncu_114"
CACHE_DIR = ROOT / "data" / "cache" / "ncu"

REMOVE_OSM_IDS = {1346093070}
REMOVE_NAMES = {"水悅莊園", "水悦莊園"}
REMOVE_UIDS = {f"NCU_{osm_id}" for osm_id in REMOVE_OSM_IDS}


def should_remove_feature(feature: dict) -> bool:
    props = feature.get("properties", {}) or {}
    try:
        osm_id = int(props.get("osm_id"))
    except (TypeError, ValueError):
        osm_id = None
    name = str(props.get("name") or "").strip()
    uid = str(props.get("uid") or "").strip()
    return osm_id in REMOVE_OSM_IDS or name in REMOVE_NAMES or uid in REMOVE_UIDS


def remove_from_geojson(path: Path) -> int:
    if not path.exists():
        return 0
    gj = json.loads(path.read_text(encoding="utf-8"))
    features = gj.get("features", [])
    kept = [ft for ft in features if not should_remove_feature(ft)]
    removed = len(features) - len(kept)
    if removed:
        gj["features"] = kept
        path.write_text(json.dumps(gj, ensure_ascii=False, indent=2), encoding="utf-8")
    return removed


def dataframe_mask_remove(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for col in ("uid", "buildId"):
        if col in df.columns:
            mask |= df[col].astype(str).isin(REMOVE_UIDS)
    for col in ("gid", "osm_id"):
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            mask |= numeric.isin(REMOVE_OSM_IDS)
    for col in ("name", "館舍", "meter_name", "meter_name"):
        if col in df.columns:
            mask |= df[col].astype(str).str.strip().isin(REMOVE_NAMES)
    return mask


def remove_from_csv(path: Path) -> int:
    if not path.exists():
        return 0
    df = pd.read_csv(path, encoding="utf-8-sig")
    mask = dataframe_mask_remove(df)
    removed = int(mask.sum())
    if removed:
        df.loc[~mask].to_csv(path, index=False, encoding="utf-8-sig")
    return removed


def remove_from_parquet(path: Path) -> int:
    if not path.exists():
        return 0
    df = pd.read_parquet(path)
    mask = dataframe_mask_remove(df)
    removed = int(mask.sum())
    if removed:
        df.loc[~mask].to_parquet(path, index=False)
    return removed


def main() -> None:
    targets = [
        NCU_DATA / "buildings.geojson",
        NCU_DATA / "energy.geojson",
        ROOT / "data" / "NCU" / "buildings_enhanced.geojson",
        ROOT / "data" / "NCU" / "osm_buildings.geojson",
    ]
    for path in targets:
        removed = remove_from_geojson(path)
        print(f"{path.relative_to(ROOT)}: removed {removed} feature(s)")

    csv_targets = [
        NCU_DATA / "metadata_uid.csv",
        NCU_DATA / "metadata_loop.csv",
        NCU_DATA / "powerMeter.csv",
        NCU_MODELS / "v12_per_building_summary.csv",
        OUT_DIR / "_geojson_buildings.csv",
        OUT_DIR / "solar_height_audit.csv",
    ]
    for path in csv_targets:
        removed = remove_from_csv(path)
        print(f"{path.relative_to(ROOT)}: removed {removed} row(s)")

    for path in sorted(CACHE_DIR.glob("inference_cache_*.parquet")):
        removed = remove_from_parquet(path)
        print(f"{path.relative_to(ROOT)}: removed {removed} row(s)")


if __name__ == "__main__":
    main()
