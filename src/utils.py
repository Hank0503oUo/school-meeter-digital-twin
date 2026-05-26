# -*- coding: utf-8 -*-
"""
共用工具函式

消除 dashboard.py、map_builder.py、building_inference.py 中的重複定義。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_float(v, default: float = np.nan) -> float:
    """安全地將任意值轉為 float，NaN / 非數值回傳 default。"""
    try:
        x = float(v)
        if np.isnan(x):
            return float(default)
        return x
    except (TypeError, ValueError):
        return float(default)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """加權平均，權重為 0 時回退至簡單平均。"""
    vals = pd.to_numeric(values, errors="coerce").fillna(0.0)
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    if float(w.sum()) > 0:
        return float(np.average(vals, weights=w))
    if len(vals) == 0:
        return 0.0
    return float(vals.mean())


def normalize_meter_name(meter_name: str) -> str:
    """清除電表名稱中的 shared meter 標記。"""
    s = str(meter_name or "").strip()
    s = s.replace("(shared meter)", "").replace("（shared meter）", "")
    return s.strip()


def split_meter_names(meter_str: str) -> list[str]:
    """將 '|' 分隔的複合電表名稱拆為獨立清單。"""
    out: list[str] = []
    for part in str(meter_str or "").split("|"):
        m = normalize_meter_name(part)
        if m:
            out.append(m)
    return out


def geo_ring_area_m2(ring: list[list[float]]) -> float:
    """經緯度 ring 以局部平面近似計算面積 (m²)。"""
    import math
    if not ring or len(ring) < 4:
        return 0.0
    lat0 = float(np.mean([pt[1] for pt in ring]))
    m_per_deg_lat = 111132.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat0))
    xy = [(pt[0] * m_per_deg_lon, pt[1] * m_per_deg_lat) for pt in ring]

    area2 = 0.0
    for i in range(len(xy) - 1):
        x1, y1 = xy[i]
        x2, y2 = xy[i + 1]
        area2 += x1 * y2 - x2 * y1
    return abs(area2) * 0.5


def geometry_footprint_m2(geometry: dict) -> float:
    """計算 Polygon / MultiPolygon 的 footprint 面積 (m²)。"""
    if not geometry:
        return 0.0
    g_type = geometry.get("type")
    if g_type == "Polygon":
        rings = geometry.get("coordinates", [])
        if not rings:
            return 0.0
        outer = geo_ring_area_m2(rings[0])
        holes = sum(geo_ring_area_m2(r) for r in rings[1:])
        return max(0.0, outer - holes)

    if g_type == "MultiPolygon":
        total = 0.0
        for poly in geometry.get("coordinates", []):
            if not poly:
                continue
            outer = geo_ring_area_m2(poly[0])
            holes = sum(geo_ring_area_m2(r) for r in poly[1:])
            total += max(0.0, outer - holes)
        return total

    return 0.0
