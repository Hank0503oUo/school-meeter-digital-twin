# -*- coding: utf-8 -*-
"""
Merge OSM Buildings with Google Maps Buildings

合併策略：
1. 讀取 OSM GeoJSON 和 Google GeoJSON。
2. 計算每棟建物的重心 (Centroid)。
3. 如果 Google 建物重心落入 OSM 建物的 20 公尺範圍內，視為同一棟。
4. 優先保留 Google 的多邊形 (Polygon)，因為通常更精確且具有樓地板對齊。
5. 將 Google 發現的全新建物 (OSM 裡沒有的) 加入最終 GeoJSON。
6. 保持屬性相容，以便 demo 地圖能正確著色。

資料用途對照（與 docs/GOOGLE_BUILDINGS_DATA_USAGE.md 一致）：
- footprint + 面積 → 校園圖台、面積／EUI 分母先驗、與 OSM 比對缺棟（本模組產出 buildings_enhanced.geojson）。
- google_place_id / name / types 仍保留在合併後的 properties，供與 uid／門牌對照或外掛腳本使用。
"""

import json
from pathlib import Path
import argparse
import numpy as np
from shapely.geometry import shape, mapping, Point
from shapely.ops import nearest_points
from src.project_paths import campus_data_dir
from src.utils import geometry_footprint_m2

def merge_campus_data(campus_id: str):
    campus_upper = campus_id.upper()
    osm_path = campus_data_dir(campus_id, "osm_buildings.geojson")
    google_path = campus_data_dir(campus_id, "google_buildings.geojson")
    output_path = campus_data_dir(campus_id, "buildings_enhanced.geojson")
    
    if not osm_path.exists():
        print(f"找不到 OSM 資料: {osm_path}")
        return
    if not google_path.exists():
        print(f"找不到 Google 資料: {google_path}")
        return
        
    print(f"[Merge] 開始合併 {campus_upper} 資料...")
    
    with open(osm_path, "r", encoding="utf-8") as f:
        osm_data = json.load(f)
    with open(google_path, "r", encoding="utf-8") as f:
        google_data = json.load(f)
        
    osm_features = osm_data.get("features", [])
    google_features = google_data.get("features", [])
    
    # 準備匹配
    osm_shapes = []
    for f in osm_features:
        try:
            s = shape(f["geometry"])
            osm_shapes.append({"shape": s, "feature": f, "matched": False})
        except:
            continue
            
    final_features = []
    google_new_count = 0
    google_enhanced_count = 0
    
    for gf in google_features:
        try:
            gs = shape(gf["geometry"])
            gc = gs.centroid
            
            # 尋找最近的 OSM 建物
            best_match = None
            min_dist = 999999
            
            for o_item in osm_shapes:
                dist = gc.distance(o_item["shape"].centroid)
                # 簡單用經緯度座標差值估算，0.0002 度約 20 公尺
                if dist < 0.0003 and dist < min_dist:
                    min_dist = dist
                    best_match = o_item
            
            if best_match:
                # 發現重合，使用 Google 的幾何，但保留 OSM 的 properties (尤其是名稱和 ID)
                new_f = {
                    "type": "Feature",
                    "geometry": gf["geometry"],
                    "properties": best_match["feature"]["properties"].copy()
                }
                # 標記來源
                new_f["properties"]["data_source"] = "google_maps_enhanced"
                new_f["properties"]["google_place_id"] = gf["properties"].get("google_place_id")
                new_f["properties"]["footprint_area_m2"] = gf["properties"].get("footprint_area_m2")
                
                # 保留 Places 與 Solar 附加屬性
                added_props = [
                    "primary_type", "opening_hours",
                    "max_sunshine_hours_per_year", "carbon_offset_factor",
                    "max_array_panels_count", "max_array_area_m2",
                    "roof_pitch_degrees", "roof_azimuth_degrees", "roof_segment_count"
                ]
                for prop_name in added_props:
                    if prop_name in gf["properties"]:
                        new_f["properties"][prop_name] = gf["properties"][prop_name]
                new_f["properties"]["footprint_area_m2"] = gf["properties"].get("footprint_area_m2")
                
                # 如果 Google 沒有預算好面積 (例如舊緩存)，則重算
                if not new_f["properties"].get("footprint_area_m2"):
                    new_f["properties"]["footprint_area_m2"] = round(geometry_footprint_m2(gf["geometry"]), 2)
                
                # 如果 Google 有更好的名稱，且 OSM 沒有，則更新
                if not new_f["properties"].get("name") and gf["properties"].get("name"):
                    new_f["properties"]["name"] = gf["properties"]["name"]
                
                final_features.append(new_f)
                best_match["matched"] = True
                google_enhanced_count += 1
            else:
                # 全新建物
                new_f = gf.copy()
                new_f["properties"]["data_source"] = "google_maps_new"
                if not new_f["properties"].get("footprint_area_m2"):
                    new_f["properties"]["footprint_area_m2"] = round(geometry_footprint_m2(gf["geometry"]), 2)
                final_features.append(new_f)
                google_new_count += 1
        except Exception as e:
            print(f"處理 Google 建物錯誤: {e}")
            continue
            
    # 加入原本沒被匹配到的 OSM 建物 (確保資料不遺失)
    osm_remain_count = 0
    for o_item in osm_shapes:
        if not o_item["matched"]:
            o_item["feature"]["properties"]["data_source"] = "osm"
            # 為原始 OSM 建物也補上計算出的面積
            geom = o_item["feature"]["geometry"]
            o_item["feature"]["properties"]["footprint_area_m2"] = round(geometry_footprint_m2(geom), 2)
            final_features.append(o_item["feature"])
            osm_remain_count += 1
            
    print(f"[Merge] 合併完成:")
    print(f"  - 強化建物 (Google 替換 OSM): {google_enhanced_count}")
    print(f"  - 新增建物 (Google 獨有): {google_new_count}")
    print(f"  - 原始建物 (僅 OSM 擁有): {osm_remain_count}")
    print(f"  - 總數: {len(final_features)}")
    
    result = {
        "type": "FeatureCollection",
        "features": final_features
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[Merge] 已儲存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campus", choices=["ntu", "ncu"], required=True)
    args = parser.parse_args()
    merge_campus_data(args.campus)
