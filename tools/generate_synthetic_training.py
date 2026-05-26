"""
Synthetic anomaly training data generator.

Reads RTEM BMS time-series, detects real anomaly patterns with
classify_anomaly_pattern(), then rewrites the values into NTU campus
scale so the training data matches Gemma's deployment context.

Output: JSONL file with {user, assistant} pairs for LoRA fine-tuning.

Usage:
    python tools/generate_synthetic_training.py
    python tools/generate_synthetic_training.py --limit 50 --output data/synthetic_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.energy_manager_skills import (
    classify_anomaly_pattern,
    cross_sensor_diagnosis,
    _list_building_subsystems,
    _load_rtem_series,
    _RTEM_BMS_DIR,
    _RTEM_META_PATH,
)

_NTU_SUMMARY = _ROOT / "models" / "v12_per_building_summary.csv"
_NTU_BUILDINGS = [
    {"type": "行政辦公大樓", "category": "辦公", "mean_kw": 262},
    {"type": "綜合教學館", "category": "教室", "mean_kw": 200},
    {"type": "理工實驗館A", "category": "實驗室", "mean_kw": 400},
    {"type": "理工實驗館B", "category": "實驗室", "mean_kw": 180},
    {"type": "生農實驗館", "category": "實驗室", "mean_kw": 179},
    {"type": "醫學研究館", "category": "實驗室", "mean_kw": 800},
    {"type": "校級圖書館", "category": "圖書館", "mean_kw": 350},
    {"type": "學生活動中心", "category": "綜合", "mean_kw": 150},
    {"type": "計算機中心", "category": "機房", "mean_kw": 1000},
    {"type": "學生宿舍", "category": "住宿", "mean_kw": 100},
    {"type": "體育館", "category": "體育", "mean_kw": 200},
    {"type": "設計學院大樓", "category": "教室", "mean_kw": 300},
    {"type": "法商教學大樓", "category": "辦公", "mean_kw": 400},
    {"type": "小型研究站", "category": "實驗室", "mean_kw": 80},
    {"type": "跨域大樓", "category": "綜合", "mean_kw": 500},
]

_SUBSYSTEM_ZH = {
    "AHU": "空調箱(AHU)",
    "FCU": "風機盤管(FCU)",
    "CH": "冰水主機",
    "CHWS": "冰水系統",
    "CT": "冷卻水塔",
    "PUMP": "水泵",
    "FAN": "風機",
    "SITE": "場域感測器",
    "METER": "電表",
    "BLR": "鍋爐",
    "HWS": "熱水系統",
    "HX": "熱交換器",
    "LIGHT": "照明",
    "ELEC": "電力",
    "VIRT": "虛擬點",
    "DUCT": "風管",
    "VAV": "變風量箱",
    "COND": "冷凝器",
    "HP": "熱泵",
    "RADIANT": "輻射板",
    "UV": "紫外線",
    "ELEV": "電梯",
}

_PATTERN_ZH = {
    "spike": "突波",
    "drift": "漂移",
    "zero": "歸零",
    "oscillation": "震盪",
    "step": "階梯跳動",
    "stuck": "卡住",
    "noise": "雜訊",
    "normal": "正常",
    "constant": "恆定",
}

_PATTERN_EXPLANATION = {
    "spike": (
        "感測器數值突然出現極端跳動，隨後快速回復。"
        "這通常是感測器故障、電磁干擾、或通訊雜訊造成的，不代表真實的物理變化。"
        "建議：檢查感測器接線與訊號遮蔽，排除干擾源。"
    ),
    "drift": (
        "感測器數值出現持續性偏移趨勢。"
        "可能原因：感測器老化、環境條件緩慢變化、或設備性能退化。"
        "建議：安排感測器校正，檢查設備運行參數是否偏離設定值。"
    ),
    "zero": (
        "感測器讀值突然歸零或接近零。"
        "可能原因：感測器斷線、通訊中斷、設備停機、或保險絲燒斷。"
        "建議：現場確認設備是否運行，檢查訊號迴路。"
    ),
    "oscillation": (
        "數值在高值與低值之間快速交替震盪。"
        "這通常是控制系統不穩定（hunting），PID 參數設定不當，或控制閥作動異常。"
        "建議：檢查 PID 參數、控制閥行程、及感測器回饋是否正常。"
    ),
    "step": (
        "數值突然跳到新基準並持續停留，未回復到原水準。"
        "可能原因：人為更改設定值、設備模式切換、或控制邏輯異常。"
        "建議：確認是否有人為操作記錄，比對設備排程。"
    ),
    "stuck": (
        "數值長時間完全沒有變化。"
        "感測器可能卡死或通訊凍結，數據未更新。"
        "建議：現場確認感測器狀態，重新啟動數據採集。"
    ),
    "noise": (
        "數值波動幅度遠超正常操作範圍。"
        "可能是電磁干擾、感測器解析度不足、或接地不良。"
        "建議：改善接地與遮蔽，或更換高精度感測器。"
    ),
}

_NTU_SCALE = {
    "temperature_F": {"min": 50, "max": 95, "typical": 72, "unit": "°F"},
    "temperature_C": {"min": 10, "max": 35, "typical": 22, "unit": "°C"},
    "humidity_pct": {"min": 30, "max": 80, "typical": 55, "unit": "%RH"},
    "power_kw": {"min": 50, "max": 2500, "typical": 260, "unit": "kW"},
    "pressure_inH2O": {"min": 0.1, "max": 5.0, "typical": 1.2, "unit": "inH2O"},
    "flow_cfm": {"min": 100, "max": 5000, "typical": 800, "unit": "CFM"},
    "valve_pct": {"min": 0, "max": 100, "typical": 40, "unit": "%"},
    "co2_ppm": {"min": 400, "max": 1200, "typical": 600, "unit": "ppm"},
}


def _rescale_value(rtem_val: float, rtem_median: float, rtem_std: float,
                   ntu_typical: float, ntu_range: float) -> float:
    if rtem_std <= 0:
        return ntu_typical
    z = (rtem_val - rtem_median) / rtem_std
    return round(ntu_typical + z * ntu_range * 0.3, 2)


def _pick_ntu_building() -> dict[str, Any]:
    return random.choice(_NTU_BUILDINGS)


def _pick_subsystem_zh(tag: str) -> str:
    return _SUBSYSTEM_ZH.get(tag, tag)


def _load_ntu_summary() -> pd.DataFrame:
    if _NTU_SUMMARY.is_file():
        return pd.read_csv(_NTU_SUMMARY, encoding="utf-8")
    return pd.DataFrame()


def _discover_rtem_buildings(limit: int = 0) -> list[int]:
    if not _RTEM_BMS_DIR.is_dir():
        return []
    bids: set[int] = set()
    for f in _RTEM_BMS_DIR.iterdir():
        name = f.name
        if name.startswith("rtem_API_data_") and name.endswith(".csv.gzip"):
            parts = name.split("_")
            if len(parts) >= 4:
                try:
                    bids.add(int(parts[3]))
                except ValueError:
                    pass
    result = sorted(bids)
    return result[:limit] if limit else result


def _extract_anomalies_from_building(building_id: int) -> list[dict[str, Any]]:
    subsystems = _list_building_subsystems(building_id)
    if not subsystems:
        return []
    anomalies: list[dict[str, Any]] = []
    for entry in subsystems:
        tag = entry["equip_tag"]
        try:
            df = _load_rtem_series(entry["file"])
        except Exception:
            continue
        if df.empty or len(df.columns) < 2:
            continue
        numeric_cols = [c for c in df.columns if c != "timestamp"]
        for col in numeric_cols[:5]:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) < 20:
                continue
            tail = series.tail(min(len(series), 200))
            result = classify_anomaly_pattern(tail.values)
            if not result or result.get("status") == "error":
                continue
            pattern = result.get("primary_pattern", "")
            if not pattern or pattern in ("normal", "constant", "insufficient_data"):
                continue
            rtem_median = float(tail.median())
            rtem_std = float(tail.std())
            sample_vals = tail.tail(10).tolist()
            anomalies.append({
                "building_id": building_id,
                "subsystem_tag": tag,
                "point_id": col,
                "pattern": pattern,
                "confidence": 0.5,
                "severity": "medium",
                "detail": "; ".join(p.get("description", "") for p in result.get("patterns", [])[:2]),
                "rtem_median": rtem_median,
                "rtem_std": rtem_std,
                "rtem_sample": sample_vals,
            })
    return anomalies


def _generate_qa_pair(anomaly: dict[str, Any]) -> dict[str, Any] | None:
    pattern = anomaly["pattern"]
    if pattern not in _PATTERN_EXPLANATION:
        return None
    building = _pick_ntu_building()
    subsystem_zh = _pick_subsystem_zh(anomaly["subsystem_tag"])
    rtem_median = anomaly["rtem_median"]
    rtem_std = anomaly["rtem_std"]
    scale_key = "power_kw"
    tag = anomaly["subsystem_tag"]
    if tag in ("AHU", "FCU", "SITE"):
        if 40 < rtem_median < 120:
            scale_key = "temperature_F"
        elif rtem_median < 30:
            scale_key = "humidity_pct"
    elif tag in ("PUMP", "FAN"):
        scale_key = "flow_cfm"
    elif tag in ("CH", "CHWS", "HWS", "HX"):
        scale_key = "temperature_F"
    ntu_scale = _NTU_SCALE[scale_key]
    rescaled_samples = [
        _rescale_value(v, rtem_median, rtem_std, ntu_scale["typical"],
                       ntu_scale["max"] - ntu_scale["min"])
        for v in anomaly["rtem_sample"]
    ]
    pattern_zh = _PATTERN_ZH.get(pattern, pattern)
    explanation = _PATTERN_EXPLANATION[pattern]
    severity = anomaly["severity"]
    samples_str = ", ".join(f"{v:.1f}" for v in rescaled_samples[:6])
    user_msg = (
        f"一棟{building['type']}的{subsystem_zh}最近 10 筆讀值如下："
        f"[{samples_str}] {ntu_scale['unit']}。"
        f"這看起來正常嗎？請分析異常類型並給出建議。"
    )
    severity_zh = {"critical": "嚴重", "high": "高", "medium": "中", "low": "低"}.get(severity, "中")
    assistant_msg = (
        f"結論：偵測到「{pattern_zh}」異常模式，嚴重度：{severity_zh}。\n\n"
        f"依據：{explanation}\n\n"
        f"感測器讀值經過分析，數據呈現{_PATTERN_ZH.get(pattern, pattern)}的特徵，"
        f"信心水準 {anomaly['confidence']:.0%}。"
        f"分析詳情：{anomaly['detail']}\n\n"
        f"建議工具：可呼叫 diagnose_anomaly() "
        f"進一步交叉比對其他感測器數據。"
    )
    return {
        "user": user_msg,
        "assistant": assistant_msg,
        "metadata": {
            "pattern": pattern,
            "pattern_zh": pattern_zh,
            "severity": severity,
            "confidence": anomaly["confidence"],
            "source_subsystem": anomaly["subsystem_tag"],
            "building_type": building["type"],
            "scale_key": scale_key,
            "layer": "L1_anomaly_reasoning",
        },
    }


def generate_synthetic_dataset(
    limit_buildings: int = 30,
    max_samples: int = 100,
    seed: int = 42,
) -> list[dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    buildings = _discover_rtem_buildings(limit=limit_buildings)
    all_anomalies: list[dict[str, Any]] = []
    for bid in buildings:
        anoms = _extract_anomalies_from_building(bid)
        all_anomalies.extend(anoms)
        if len(all_anomalies) >= max_samples * 3:
            break
    pattern_counts = Counter(a["pattern"] for a in all_anomalies)
    print(f"Extracted {len(all_anomalies)} anomaly segments from {len(buildings)} buildings")
    print(f"Pattern distribution: {dict(pattern_counts)}")
    random.shuffle(all_anomalies)
    qa_pairs: list[dict[str, Any]] = []
    for anomaly in all_anomalies:
        pair = _generate_qa_pair(anomaly)
        if pair:
            qa_pairs.append(pair)
        if len(qa_pairs) >= max_samples:
            break
    return qa_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic anomaly training data")
    parser.add_argument("--limit", type=int, default=30, help="Max RTEM buildings to scan")
    parser.add_argument("--max-samples", type=int, default=100, help="Max QA pairs to generate")
    parser.add_argument("--output", type=str, default="", help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_path = Path(args.output) if args.output else _ROOT / "data" / "synthetic_anomaly_v1.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    qa_pairs = generate_synthetic_dataset(
        limit_buildings=args.limit,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    with output_path.open("w", encoding="utf-8") as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Wrote {len(qa_pairs)} QA pairs to {output_path}")
    pattern_dist = Counter(p["metadata"]["pattern"] for p in qa_pairs)
    print(f"Output pattern distribution: {dict(pattern_dist)}")


if __name__ == "__main__":
    main()
