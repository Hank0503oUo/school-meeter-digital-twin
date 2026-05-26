# -*- coding: utf-8 -*-
"""
Google Maps Building Discovery & Footprint Fetcher

1. 使用 Places API (New) Nearby Search 在校園範圍內進行 grid scan，
   搜尋 type='university' 或 'school' 或 'establishment' 的 POI，取得 place_id。
2. 使用 Geocoding API 並帶 extra_computations=BUILDING_AND_ENTRANCES 取得 footprint 多邊形。
3. 結果緩存為本地 JSON 文件。

# 用法：
#     # 設定環境變數
#     $env:GOOGLE_MAPS_API_KEY = "YOUR_KEY"
#     # 執行全校掃描並整併 Solar API特徵
#     python -m src.google_fetcher --campus ntu
#     # 測試模糊搜尋 (Text Mapping)
#     python -m src.google_fetcher --campus ntu --text-search "體育館"
#
# 產物用途（詳見 docs/GOOGLE_BUILDINGS_DATA_USAGE.md）：
# - footprint + footprint_area_m2 → 圖台、EUI 面積分母先驗、與 OSM 合併補棟（merge_osm_google）。
# - google_place_id, name, types → 與校內 uid／門牌對照時的外鍵與別名線索（需對照 metadata_uid 等，非自動相等）。
"""

import os
import json
import time
import argparse
from pathlib import Path
import requests
from typing import Dict, List, Any, Optional
from src.project_paths import campus_data_dir, data_dir
from src.utils import geometry_footprint_m2

# 預設格網掃描間距 (公尺)，Google Nearby Search (New) 的半徑建議
GRID_STEP_M = 150
SEARCH_RADIUS_M = 150

# 座標常數 (與 campus config 對照)
CAMPUS_BBOX = {
    "ntu": {
        "south": 25.0130,
        "west":  121.5330,
        "north": 25.0220,
        "east":  121.5460,
    },
    "ncu": {
        "south": 24.9650,
        "west":  121.1880,
        "north": 24.9730,
        "east":  121.1980,
    }
}

CACHE_DIR = data_dir("cache", "google_maps")

def get_api_key() -> str:
    key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not key:
        raise ValueError("請先設定環境變數 GOOGLE_MAPS_API_KEY")
    return key

def discover_buildings_nearby(lat: float, lon: float, radius: int, api_key: str) -> List[Dict[str, Any]]:
    """
    使用 Places API (New) Nearby Search 發現附近的建築物
    """
    url = "https://places.googleapis.com/v1/places:searchNearby"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.id,places.displayName,places.location,places.types,places.primaryType,places.regularOpeningHours"
    }
    
    # 搜尋大學內可能標註為校園建物的類型
    # Places API (New) does not accept generic containers like
    # `point_of_interest` or `establishment` in `includedTypes`.
    data = {
        "includedTypes": ["university", "school"],
        "locationRestriction": {
            "circle": {
                "center": {"latitude": lat, "longitude": lon},
                "radius": float(radius)
            }
        },
        "maxResultCount": 20
    }
    
    try:
        resp = requests.post(url, json=data, headers=headers, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        return result.get("places", [])
    except Exception as e:
        details = ""
        if hasattr(e, "response") and e.response is not None:
            details = f" | body={e.response.text[:300]}"
        print(f"[Google Fetcher] Nearby Search 錯誤 ({lat}, {lon}): {e}{details}")
        return []

def fetch_building_details(place_id: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    使用 Geocoding API 取得建物足跡 (BUILDING_AND_ENTRANCES)
    """
    # 檢查緩存
    cache_file = CACHE_DIR / f"details_{place_id}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
            
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "place_id": place_id,
        "extra_computations": "BUILDING_AND_ENTRANCES",
        "key": api_key
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get("status") == "OK" and data.get("results"):
            # 儲存緩存
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return data
    except Exception as e:
        print(f"[Google Fetcher] Geocoding 錯誤 (place_id={place_id}): {e}")
        
    return None

def fetch_solar_insights(lat: float, lon: float, api_key: str) -> Optional[Dict[str, Any]]:
    """
    使用 Solar API 取得屋頂幾何 (高程/傾角) 與日照潛力 (3D 先驗特徵)
    """
    cache_file = CACHE_DIR / f"solar_{lat:.5f}_{lon:.5f}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
            
    url = "https://solar.googleapis.com/v1/buildingInsights:findClosest"
    params = {
        "location.latitude": lat,
        "location.longitude": lon,
        "requiredSkillLevel": "EXPERIENCE_LEVEL_UNSPECIFIED",
        "key": api_key
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return data
    except Exception as e:
        print(f"[Google Fetcher] Solar API 錯誤 (lat={lat}, lon={lon}): {e}")
        
    return None

def search_place_by_text(query: str, lat: float, lon: float, radius: int, api_key: str) -> List[Dict[str, Any]]:
    """
    使用 Places API (New) Text Search，幫助盲端電表作字串 Mapping
    """
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.id,places.displayName,places.location,places.types,places.formattedAddress,places.primaryType,places.regularOpeningHours"
    }
    
    data = {
        "textQuery": query,
        "locationBias": {
            "circle": {
                "center": {"latitude": lat, "longitude": lon},
                "radius": float(radius)
            }
        },
        "maxResultCount": 5
    }
    
    try:
        resp = requests.post(url, json=data, headers=headers, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        return result.get("places", [])
    except Exception as e:
        print(f"[Google Fetcher] Text Search 錯誤 ({query}): {e}")
        return []

def latlon_to_meters(lat1, lon1, lat2, lon2):
    """粗略估算兩點距離 (公尺)"""
    import math
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlamb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlamb/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def scan_campus(campus_id: str, api_key: str):
    bbox = CAMPUS_BBOX.get(campus_id)
    if not bbox:
        print(f"找不到校園 {campus_id} 的座標範圍")
        return
    
    print(f"[Google Fetcher] 開始掃描校區: {campus_id.upper()}")
    
    # 計算 Grid
    import numpy as np
    
    # 緯度每度約 111km，經度在台灣緯度每度約 100km
    lat_step = GRID_STEP_M / 111000.0
    lon_step = GRID_STEP_M / 100000.0
    
    lats = np.arange(bbox["south"], bbox["north"] + lat_step, lat_step)
    lons = np.arange(bbox["west"], bbox["east"] + lon_step, lon_step)
    
    all_places = {} # place_id -> place_info
    
    print(f"[Google Fetcher] Grid 大小: {len(lats)} x {len(lons)} = {len(lats)*len(lons)} 個節點")
    
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            print(f"\r  掃描進度: {i*len(lons)+j+1}/{len(lats)*len(lons)}", end="")
            places = discover_buildings_nearby(lat, lon, SEARCH_RADIUS_M, api_key)
            for p in places:
                pid = p.get("id")
                if pid:
                    all_places[pid] = p
            time.sleep(0.05) # 稍微緩衝避開 Rate Limit
            
    print(f"\n[Google Fetcher] 共發現 {len(all_places)} 個唯一 Place ID")
    
    # 取得詳細資料 (Polygon)
    features = []
    
    count = 0
    for pid, info in all_places.items():
        count += 1
        print(f"\r  取得足跡過度: {count}/{len(all_places)}", end="")
        
        details = fetch_building_details(pid, api_key)
        if not details:
            continue
            
        # 解析 Geocoding 結果中的建物足跡
        for res in details.get("results", []):
            if "buildings" in res:
                for b in res["buildings"]:
                    if "building_outlines" in b:
                        for outline in b["building_outlines"]:
                            poly = outline.get("display_polygon")
                            if poly:
                                # 轉為 GeoJSON Feature
                                name = info.get("displayName", {}).get("text", "")
                                area_m2 = geometry_footprint_m2(poly)
                                
                                # 取 Solar API 特徵以豐富 3D 幾何與環境物理
                                solar_features = {}
                                slat = info.get("location", {}).get("latitude")
                                slon = info.get("location", {}).get("longitude")
                                if slat and slon:
                                    solar_data = fetch_solar_insights(slat, slon, api_key)
                                    if solar_data and "solarPotential" in solar_data:
                                        pot = solar_data["solarPotential"]
                                        solar_features = {
                                            "max_array_panels_count": pot.get("maxArrayPanelsCount", 0),
                                            "max_sunshine_hours_per_year": pot.get("maxSunshineHoursPerYear", 0),
                                            "carbon_offset_factor": pot.get("carbonOffsetFactorKgPerMwh", 0)
                                        }
                                        # 找出主要屋頂傾角 (幫助推論 Ur 與建築樣式)
                                        if "roofSegmentStats" in pot and pot["roofSegmentStats"]:
                                            main_roof = max(
                                                pot["roofSegmentStats"], 
                                                key=lambda x: x.get("stats", {}).get("areaMeters2", 0)
                                            )
                                            solar_features["roof_pitch_degrees"] = main_roof.get("pitchDegrees", 0)
                                            solar_features["roof_azimuth_degrees"] = main_roof.get("azimuthDegrees", 0)
                                            
                                # 豐富原始的 Place 元數據
                                types = info.get("types", [])
                                primary_type = info.get("primaryType", "")
                                opening_hours = info.get("regularOpeningHours", {})
                                
                                feature = {
                                    "type": "Feature",
                                    "geometry": poly,
                                    "properties": {
                                        "google_place_id": pid,
                                        "name": name,
                                        "types": types,
                                        "primary_type": primary_type,
                                        "has_opening_hours": bool(opening_hours),
                                        "data_source": "google_maps_enhanced",
                                        "footprint_area_m2": round(area_m2, 2),
                                        **solar_features
                                    }
                                }
                                features.append(feature)
        time.sleep(0.05)
        
    print(f"\n[Google Fetcher] 成功取得 {len(features)} 個建物多邊形")
    
    # 輸出結果
    output_path = campus_data_dir(campus_id, "google_buildings.geojson")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)
        
    print(f"[Google Fetcher] 已儲存至: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Fetch building footprints from Google Maps API")
    parser.add_argument("--campus", choices=["ntu", "ncu"], required=True, help="校園 ID")
    parser.add_argument("--test-single", help="測試單一 Place ID 的足跡")
    parser.add_argument("--text-search", help="測試模糊搜尋 (解決電表 Mapping 痛點)")
    args = parser.parse_args()
    
    try:
        api_key = get_api_key()
    except ValueError as e:
        print(e)
        return
        
    if args.test_single:
        details = fetch_building_details(args.test_single, api_key)
        print(json.dumps(details, indent=2, ensure_ascii=False))
    elif args.text_search:
        # 用校園中心點測試 search
        bbox = CAMPUS_BBOX.get(args.campus)
        if bbox:
            center_lat = (bbox["south"] + bbox["north"]) / 2
            center_lon = (bbox["west"] + bbox["east"]) / 2
            results = search_place_by_text(args.text_search, center_lat, center_lon, 1000, api_key)
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            print("找不到校園座標")
    else:
        scan_campus(args.campus, api_key)

if __name__ == "__main__":
    main()
