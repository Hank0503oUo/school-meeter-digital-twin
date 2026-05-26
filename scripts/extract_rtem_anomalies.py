"""
extract_rtem_anomalies.py
從紐約州 RTEM 資料集抽取真實異常片段，正規化縮放到台大校園等級。

用法：
    python scripts/extract_rtem_anomalies.py \
        --bms-dir "C:\Users\User\Downloads\build llm\drive-download-20260320T153715Z-3-001\RTEM dataset\BMS data" \
        --meta "C:\Users\User\Downloads\build llm\drive-download-20260320T153715Z-3-001\RTEM dataset\meta data\all_points_metadata.csv" \
        --output data/lora/rtem_anomalies_campus_scaled.jsonl \
        --top-n 100
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CAMPUS_SCALE = {
    "power_kw": {"rtem_mean": 500.0, "campus_mean": 100.0},
    "power_kwh": {"rtem_mean": 15000.0, "campus_mean": 3000.0},
    "temp_f": {"scale": False},
    "humidity_pct": {"scale": False},
    "pressure_inh2o": {"rtem_mean": 2.0, "campus_mean": 1.0},
    "flow_cfm": {"rtem_mean": 5000.0, "campus_mean": 1000.0},
    "percent": {"scale": False},
    "volts": {"scale": False},
    "frequency_hz": {"scale": False},
    "rpm": {"rtem_mean": 1200.0, "campus_mean": 800.0},
    "ppm": {"scale": False},
    "unknown": {"scale": False},
}

CAMPUS_BUILDING_IDS = {
    "AT2045": "保健中心",
    "AT2007": "化學工程館",
    "AT5043": "土木研究大樓",
    "library_main": "總圖書館",
    "AT1001": "電機一館",
    "AT3001": "共同教學館",
    "AT6001": "活動中心",
}

UNIT_TO_CATEGORY = {
    "degreesFahrenheit": "temp_f",
    "percentRelativeHumidity": "humidity_pct",
    "kilowatts": "power_kw",
    "kilowattHours": "power_kwh",
    "inchesOfWater": "pressure_inh2o",
    "cubicFeetPerMinute": "flow_cfm",
    "percent": "percent",
    "volts": "volts",
    "hertz": "frequency_hz",
    "revolutionsPerMinute": "rpm",
    "partsPerMillion": "ppm",
    "noUnits": "unknown",
    "degreesPhase": "unknown",
    "poundsForcePerSquareInch": "pressure_inh2o",
    "65535": "unknown",
}


def scale_value(val: float, rtem_mean: float, campus_mean: float) -> float:
    if rtem_mean == 0:
        return val
    s = campus_mean / rtem_mean
    return (val - rtem_mean) * s + campus_mean


def scale_series(series: pd.Series, unit: str) -> pd.Series:
    cat = UNIT_TO_CATEGORY.get(unit, "unknown")
    cfg = CAMPUS_SCALE.get(cat, {"scale": False})
    if not cfg.get("scale", True) and "rtem_mean" not in cfg:
        return series
    rtem_mean = cfg.get("rtem_mean", series.mean())
    campus_mean = cfg.get("campus_mean", rtem_mean)
    return series.apply(lambda x: scale_value(x, rtem_mean, campus_mean))


def detect_anomalies_zscore(
    series: pd.Series, z_threshold: float = 3.5, min_duration: int = 2
) -> list[dict]:
    clean = series.dropna()
    if len(clean) < 20:
        return []
    mu = clean.mean()
    sigma = clean.std()
    if sigma == 0 or np.isnan(sigma):
        return []
    z = (clean - mu).abs() / sigma
    anomaly_mask = z > z_threshold
    if anomaly_mask.sum() < min_duration:
        return []
    groups = (anomaly_mask != anomaly_mask.shift()).cumsum()
    segments = []
    for gid, gdf in clean[anomaly_mask].groupby(groups):
        if len(gdf) < min_duration:
            continue
        peak_idx = gdf.index[np.argmax(np.abs(z[gdf.index]))]
        peak_val = gdf.loc[peak_idx]
        peak_z = z.loc[peak_idx]
        segments.append({
            "start_ts": str(gdf.index[0]),
            "end_ts": str(gdf.index[-1]),
            "duration_points": len(gdf),
            "peak_value": float(peak_val),
            "peak_zscore": float(peak_z),
            "mean_value": float(gdf.mean()),
            "baseline_mean": float(mu),
            "baseline_std": float(sigma),
            "direction": "high" if peak_val > mu else "low",
        })
    return segments


def describe_anomaly(
    segment: dict,
    unit: str,
    point_name: str,
    equip_type: str,
    building_id: str,
    campus_building: str,
    scaled_peak: float,
    scaled_mean: float,
    scaled_baseline: float,
) -> str:
    cat = UNIT_TO_CATEGORY.get(unit, "unknown")
    direction = segment["direction"]
    duration = segment["duration_points"] * 5
    z = segment["peak_zscore"]

    if cat == "temp_f":
        direction_text = "急遽升高" if direction == "high" else "異常降低"
        return (
            f"{campus_building}的{equip_type}感測器「{point_name}」在短時間內"
            f"{direction_text}至 {scaled_peak:.1f}°F（基線 {scaled_baseline:.1f}°F），"
            f"偏離 {z:.1f} 個標準差，持續約 {duration} 分鐘。"
        )
    elif cat in ("power_kw", "power_kwh"):
        direction_text = "飆升" if direction == "high" else "驟降"
        return (
            f"{campus_building}的{equip_type}功率{direction_text}至 {scaled_peak:.1f} kW"
            f"（基線 {scaled_baseline:.1f} kW），偏離 {z:.1f}σ，持續 {duration} 分鐘。"
        )
    elif cat == "pressure_inh2o":
        return (
            f"{campus_building}的{equip_type}「{point_name}」壓力異常，"
            f"讀值 {scaled_peak:.2f} inH2O（基線 {scaled_baseline:.2f}），"
            f"偏離 {z:.1f}σ，持續 {duration} 分鐘。"
        )
    elif cat == "flow_cfm":
        return (
            f"{campus_building}的{equip_type}「{point_name}」流量異常，"
            f"讀值 {scaled_peak:.0f} CFM（基線 {scaled_baseline:.0f}），"
            f"偏離 {z:.1f}σ，持續 {duration} 分鐘。"
        )
    else:
        return (
            f"{campus_building}的{equip_type}「{point_name}」偵測到異常，"
            f"讀值 {scaled_peak:.2f}（基線 {scaled_baseline:.2f}，單位 {unit}），"
            f"偏離 {z:.1f}σ，持續 {duration} 分鐘。"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bms-dir", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--output", default="data/lora/rtem_anomalies_campus_scaled.jsonl")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--z-threshold", type=float, default=3.5)
    parser.add_argument("--buildings", type=int, default=10)
    args = parser.parse_args()

    bms_dir = Path(args.bms_dir)
    meta_path = Path(args.meta)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent.parent / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading metadata...")
    meta = pd.read_csv(meta_path)
    point_meta = {}
    for _, row in meta.iterrows():
        point_meta[str(row["id_point"])] = {
            "building_id": row["building_id"],
            "equip_type": row.get("equip_type_abbr", ""),
            "equip_name": row.get("equip_type_name", ""),
            "point_name": row.get("name_point", ""),
            "units": row.get("units", ""),
            "description": row.get("description", ""),
            "area": row.get("area_served_desc", ""),
        }

    bms_files = sorted(bms_dir.glob("rtem_API_data_*.csv.gzip"))
    building_ids = sorted(set(f.name.split("_")[3] for f in bms_files))
    selected = building_ids[: args.buildings]
    print(f"Scanning {len(selected)} buildings: {selected}")

    all_anomalies = []
    for bldg in selected:
        bldg_files = [f for f in bms_files if f"_data_{bldg}_" in f.name]
        campus_bldg = list(CAMPUS_BUILDING_IDS.values())[
            hash(bldg) % len(CAMPUS_BUILDING_IDS)
        ]
        campus_bldg_id = list(CAMPUS_BUILDING_IDS.keys())[
            hash(bldg) % len(CAMPUS_BUILDING_IDS)
        ]

        for fpath in bldg_files:
            equip_tag = fpath.stem.replace(f"rtem_API_data_{bldg}_", "").replace(".csv", "")
            try:
                with gzip.open(fpath, "rb") as f:
                    df = pd.read_csv(io.BytesIO(f.read()))
            except Exception:
                continue
            if "timestamp" not in df.columns:
                continue
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp").sort_index()

            for col in df.columns[:20]:
                pm = point_meta.get(col, {})
                unit = pm.get("units", "unknown")
                point_name = pm.get("point_name", col)
                equip_type = pm.get("equip_type", equip_tag)
                desc = pm.get("description", "")

                series = df[col].dropna()
                if len(series) < 100:
                    continue

                segments = detect_anomalies_zscore(series, args.z_threshold)
                for seg in segments[:3]:
                    scaled_peak_series = scale_value(seg["peak_value"], seg["baseline_mean"],
                        CAMPUS_SCALE.get(UNIT_TO_CATEGORY.get(unit, "unknown"), {}).get("campus_mean", seg["baseline_mean"]))
                    scaled_mean_series = scale_value(seg["mean_value"], seg["baseline_mean"],
                        CAMPUS_SCALE.get(UNIT_TO_CATEGORY.get(unit, "unknown"), {}).get("campus_mean", seg["baseline_mean"]))
                    scaled_baseline = CAMPUS_SCALE.get(UNIT_TO_CATEGORY.get(unit, "unknown"), {}).get(
                        "campus_mean", seg["baseline_mean"])

                    description = describe_anomaly(
                        seg, unit, point_name, equip_type, bldg, campus_bldg,
                        scaled_peak_series, scaled_mean_series, scaled_baseline,
                    )

                    all_anomalies.append({
                        "rtem_building": bldg,
                        "campus_building_id": campus_bldg_id,
                        "campus_building_name": campus_bldg,
                        "equip_type": equip_type,
                        "point_name": point_name,
                        "unit": unit,
                        "description": desc,
                        "anomaly": description,
                        "zscore": seg["peak_zscore"],
                        "direction": seg["direction"],
                        "duration_minutes": seg["duration_points"] * 5,
                        "scaled_peak": round(scaled_peak_series, 3),
                        "scaled_baseline": round(scaled_baseline, 3),
                        "original_peak": round(seg["peak_value"], 3),
                    })

    all_anomalies.sort(key=lambda x: x["zscore"], reverse=True)
    top = all_anomalies[: args.top_n]

    with open(output_path, "w", encoding="utf-8") as f:
        for a in top:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    print(f"\nExtracted {len(all_anomalies)} anomalies, saved top {len(top)} to {output_path}")
    directions = {"high": 0, "low": 0}
    eq_types = {}
    for a in top:
        directions[a["direction"]] += 1
        eq_types[a["equip_type"]] = eq_types.get(a["equip_type"], 0) + 1
    print(f"Directions: {directions}")
    print(f"Equipment:  {eq_types}")


if __name__ == "__main__":
    main()
