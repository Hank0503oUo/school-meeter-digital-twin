"""
energy_sft_factory.py — Unified SFT data factory for building energy anomaly diagnosis.

Pipeline (7 stages):
  1. pipeline_utils      — I/O helpers, hashing, token estimation, bucketing
  2. seed_generation     — L1 seeds from RTEM BMS (anomaly classification)
                           L2 seeds from NTU building stats (counterfactual)
  3. synthesis           — accepted + rejected answer generation
  4. quality_control     — judge scoring (5x0.2 rubric)
  5. preference          — chosen vs rejected pairs + risk refusal samples
  6. train/val/smoke     — split + manifest
  7. downstream_validation — 30-pair sampled audit

Run: cd D:\idf優化\demo && python tools/energy_sft_factory.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.energy_manager_skills import (
    _list_building_subsystems,
    _load_rtem_series,
    classify_anomaly_pattern,
)
from src.counterfactual import run_building_counterfactual

_OUT_BASE = _ROOT / "data" / "energy_sft_factory"
_OUT_PROC = _OUT_BASE / "processed"
_OUT_TRAIN = _OUT_BASE / "training"
_OUT_REPORT = _OUT_BASE / "reports"

# ═══════════════ Normalization spec ═══════════════
NORM_SPEC = {
    "temperature_room": {"ntu_typical": 24.0, "ntu_range": 10.0},
    "temperature_supply": {"ntu_typical": 14.0, "ntu_range": 10.0},
    "temperature_chws": {"ntu_typical": 8.0, "ntu_range": 10.0},
    "temperature_ct": {"ntu_typical": 30.0, "ntu_range": 13.0},
    "humidity": {"ntu_typical": 55.0, "ntu_range": 40.0},
    "power_kw": {"ntu_typical": 200.0, "ntu_range": 800.0},
    "co2": {"ntu_typical": 600.0, "ntu_range": 1100.0},
    "valve_pct": {"ntu_typical": 40.0, "ntu_range": 100.0},
    "pressure_pa": {"ntu_typical": 300.0, "ntu_range": 950.0},
    "airflow_cmh": {"ntu_typical": 3500.0, "ntu_range": 4800.0},
}

BUILDING_TYPES = [
    "行政辦公大樓", "綜合教學館", "理工實驗館", "生農實驗館",
    "醫學研究館", "總圖書館", "學生活動中心", "計算機中心",
    "學生宿舍", "綜合體育館",
]

SENSOR_LABELS = {
    "AHU": {"dat_av": "送風溫度感測器", "rat": "回風溫度感測器", "oat": "外氣溫度感測器",
            "dat_stp": "送風溫度設定值", "cool": "冷卻閥開度", "heat": "加熱閥開度",
            "fan": "風機狀態", "damper": "風門開度", "co2": "CO2感測器", "humidity": "濕度感測器"},
    "CH": {"chws": "冰水供水溫度", "chwr": "冰水回水溫度", "compressor": "壓縮機狀態", "power": "冰機功率"},
    "CHWS": {"chws": "冰水供水溫度", "chwr": "冰水回水溫度"},
    "CT": {"ct_entering": "冷卻水進水溫度", "ct_leaving": "冷卻水出水溫度"},
    "FCU": {"temp": "室內溫度", "valve": "閥門開度", "fan": "風機狀態"},
    "METER": {"kw": "瞬時功率", "kwh": "累積用電量"},
    "SITE": {"kw": "總功率", "kwh": "總用電量"},
    "FAN": {"fan_speed": "風機轉速", "cfm": "風量"},
    "DUCT": {"static_pressure": "風管靜壓"},
    "ELEC": {"kw": "電力功率"},
    "LIGHT": {"kw": "照明功率"},
}

PATTERN_LABELS = {
    "spike": "突波", "drift": "漂移", "zero_flatline": "歸零",
    "oscillation": "震盪", "step_change": "階梯變動",
    "stuck": "卡死/無變化", "noise": "雜訊過大",
    "none": "無明顯異常", "normal": "正常",
}

_SMOKE_PATTERNS = ["spike", "drift", "zero_flatline", "oscillation", "step_change", "stuck", "noise"]

# ═══════════════ Stage 1: pipeline_utils ═══════════════

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    p = Path(path)
    if p.is_file():
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records

def write_jsonl(path: str | Path, records: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def estimated_tokens(text: str) -> int:
    cjk = len(re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf]", text))
    words = len(re.findall(r"[a-zA-Z0-9_]+", text))
    return cjk + words

def deterministic_bucket(text: str, modulo: int = 100) -> int:
    return int(sha1_text(text), 16) % modulo

def build_sample_id(layer: str, idx: int) -> str:
    return f"energy_sft_{layer}_{idx:06d}"

# ═══════════════ Stage 2: seed_generation ═══════════════

def _normalize_rtem_value(val: float, median: float, std: float, sensor_type: str) -> float:
    spec = NORM_SPEC.get(sensor_type, NORM_SPEC["temperature_room"])
    if std <= 0:
        return spec["ntu_typical"]
    z = (val - median) / std
    return round(spec["ntu_typical"] + z * spec["ntu_range"] * 0.3, 2)

def _map_subsystem_to_sensor_type(equip_tag: str) -> str:
    tag = equip_tag.upper()
    if tag in ("AHU", "FCU", "VAV"):
        return "temperature_room"
    if tag in ("CH", "CHWS"):
        return "temperature_chws"
    if tag in ("CT", "COND"):
        return "temperature_ct"
    if tag in ("METER", "SITE", "ELEC"):
        return "power_kw"
    return "temperature_room"

def _map_point_id_to_sensor_label(point_id: str, equip_tag: str) -> str:
    pid_lower = str(point_id).lower()
    tag_upper = str(equip_tag).upper()
    mapping = SENSOR_LABELS.get(tag_upper, {})
    for key, label in mapping.items():
        if key in pid_lower:
            return label
    if "temp" in pid_lower:
        return "溫度感測器"
    if "kw" in pid_lower or "power" in pid_lower:
        return "功率感測器"
    if "humidity" in pid_lower or "rh" in pid_lower:
        return "濕度感測器"
    if "co2" in pid_lower:
        return "CO2感測器"
    if "valve" in pid_lower:
        return "閥門開度感測器"
    if "fan" in pid_lower:
        return "風機狀態感測器"
    return f"{tag_upper}感測器"

def extract_l1_seeds(limit_buildings: int = 30) -> list[dict]:
    seeds = []
    all_subsystems = []
    for bid in range(98, 500):
        subs = _list_building_subsystems(bid)
        if subs:
            all_subsystems.extend(subs)
            if len({s["building_id"] for s in all_subsystems}) >= limit_buildings:
                break
    if not all_subsystems:
        print("WARNING: No RTEM BMS data found. Generating smoke-test seeds as fallback.")
        return [_smoke_single_seed(p) for p in _SMOKE_PATTERNS]

    random.shuffle(all_subsystems)
    seen_buildings = Counter()
    max_per_building = 3

    for entry in all_subsystems:
        bid = entry["building_id"]
        tag = entry["equip_tag"]
        if seen_buildings[bid] >= max_per_building:
            continue
        try:
            df = _load_rtem_series(entry["file"])
            if df.empty or len(df.columns) < 2:
                continue
            numeric_cols = [c for c in df.columns if c != "timestamp"]
        except Exception:
            continue

        for col in numeric_cols[:5]:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) < 20:
                continue
            recent = series.tail(min(len(series), 200))
            result = classify_anomaly_pattern(values=recent.values)
            primary = result.get("primary_pattern", "none")
            if primary in ("none", "normal"):
                if random.random() > 0.15:
                    continue

            sensor_type = _map_subsystem_to_sensor_type(tag)
            sensor_label = _map_point_id_to_sensor_label(col, tag)
            rtem_median = float(series.median())
            rtem_std = float(series.std(ddof=0) or rtem_median * 0.05 or 1.0)
            normalized_median = _normalize_rtem_value(rtem_median, rtem_median, rtem_std, sensor_type)
            normalized_std = abs(normalized_median * 0.08) or 1.0
            sample_vals = series.tail(12).tolist()
            normalized_samples = [_normalize_rtem_value(v, rtem_median, rtem_std, sensor_type) for v in sample_vals]

            seed = {
                "seed_id": f"rtem_{bid}_{tag}_{col[:15]}",
                "layer": "L1",
                "task_type": "anomaly_classification",
                "pattern": primary,
                "rtem_median": round(rtem_median, 4),
                "rtem_std": round(rtem_std, 4),
                "rtem_sample": [round(float(v), 2) for v in sample_vals],
                "normalized_median": normalized_median,
                "normalized_std": normalized_std,
                "normalized_sample": normalized_samples,
                "subsystem_tag": tag,
                "sensor_type": sensor_type,
                "sensor_label": sensor_label,
                "pattern_confidence": (result.get("patterns", [{}])[0].get("confidence", 0)
                                       if result.get("patterns") else 0),
                "source": "rtem_bms",
            }
            seeds.append(seed)
            seen_buildings[bid] += 1
            break
    return seeds


def _load_stats_df():
    csv_path = _ROOT / "data" / "ntu_building_stats.csv"
    if csv_path.is_file():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def extract_l2_seeds() -> list[dict]:
    df = _load_stats_df()
    if df.empty:
        print("WARNING: No NTU building stats CSV found. Using synthetic fallback.")
        return _extract_l2_seeds_synthetic()

    seeds = []
    scenarios = [
        {"name": "cooling_plus1", "cooling_delta_degC": 1.0, "lighting_ratio": 1.0, "equipment_ratio": 1.0},
        {"name": "cooling_plus2", "cooling_delta_degC": 2.0, "lighting_ratio": 1.0, "equipment_ratio": 1.0},
        {"name": "cooling_minus1", "cooling_delta_degC": -1.0, "lighting_ratio": 1.0, "equipment_ratio": 1.0},
        {"name": "lighting_led", "cooling_delta_degC": 0.0, "lighting_ratio": 0.7, "equipment_ratio": 1.0},
        {"name": "equipment_efficient", "cooling_delta_degC": 0.0, "lighting_ratio": 1.0, "equipment_ratio": 0.85},
        {"name": "combined", "cooling_delta_degC": 1.5, "lighting_ratio": 0.8, "equipment_ratio": 0.9},
    ]

    for idx, row in df.iterrows():
        if idx >= 25:
            break
        mean_kw = float(row.get("mean_kw", 0) or 0)
        if mean_kw < 10:
            continue
        stats = {"mean_kw": mean_kw}
        for sc in scenarios:
            try:
                result = run_building_counterfactual(stats, cooling_delta_degC=sc["cooling_delta_degC"],
                    lighting_ratio=sc["lighting_ratio"], equipment_ratio=sc["equipment_ratio"])
            except Exception:
                continue
            seeds.append({
                "seed_id": f"ntu_counterfactual_{idx}_{sc['name']}",
                "layer": "L2",
                "task_type": "counterfactual",
                "mean_kw": mean_kw,
                "eui": float(row.get("eui", 0) or 0),
                "archetype": str(row.get("archetype_label", "未知")),
                "scenario_params": sc,
                "delta_kwh": result.get("delta_kwh", 0),
                "delta_pct": result.get("delta_pct", 0),
                "delta_ntd": result.get("delta_ntd", 0),
                "source": "ntu_building_stats",
            })
    return seeds


def _extract_l2_seeds_synthetic() -> list[dict]:
    archetypes = [
        {"name": "行政辦公大樓", "mean_kw": 300, "eui": 145},
        {"name": "理工實驗館", "mean_kw": 450, "eui": 380},
        {"name": "總圖書館", "mean_kw": 250, "eui": 135},
        {"name": "學生活動中心", "mean_kw": 120, "eui": 85},
        {"name": "計算機中心", "mean_kw": 800, "eui": 650},
        {"name": "學生宿舍", "mean_kw": 100, "eui": 72},
        {"name": "綜合體育館", "mean_kw": 200, "eui": 95},
        {"name": "醫學研究館", "mean_kw": 650, "eui": 520},
        {"name": "綜合教學館", "mean_kw": 180, "eui": 160},
    ]
    scenarios = [
        {"name": "cooling_plus1", "cooling_delta_degC": 1.0, "lighting_ratio": 1.0, "equipment_ratio": 1.0},
        {"name": "cooling_plus2", "cooling_delta_degC": 2.0, "lighting_ratio": 1.0, "equipment_ratio": 1.0},
        {"name": "lighting_led", "cooling_delta_degC": 0.0, "lighting_ratio": 0.7, "equipment_ratio": 1.0},
        {"name": "combined", "cooling_delta_degC": 1.5, "lighting_ratio": 0.8, "equipment_ratio": 0.9},
    ]
    seeds = []
    for i, a in enumerate(archetypes):
        stats = {"mean_kw": a["mean_kw"]}
        for sc in scenarios:
            try:
                result = run_building_counterfactual(stats, cooling_delta_degC=sc["cooling_delta_degC"],
                    lighting_ratio=sc["lighting_ratio"], equipment_ratio=sc["equipment_ratio"])
            except Exception:
                continue
            seeds.append({
                "seed_id": f"synthetic_{i}_{sc['name']}",
                "layer": "L2",
                "task_type": "counterfactual",
                "mean_kw": a["mean_kw"],
                "eui": a["eui"],
                "archetype": a["name"],
                "scenario_params": sc,
                "delta_kwh": result.get("delta_kwh", 0),
                "delta_pct": result.get("delta_pct", 0),
                "delta_ntd": result.get("delta_ntd", 0),
                "source": "synthetic_fallback",
            })
    return seeds

# ═══════════════ Stage 3: synthesis ═══════════════

def synthesize_accepted(seed: dict) -> dict:
    layer = seed["layer"]
    sid = seed["seed_id"]
    pattern = seed.get("pattern", "none")
    btype = seed.get("archetype", random.choice(BUILDING_TYPES))
    if "archetype" not in seed:
        btype = random.choice(BUILDING_TYPES)

    if layer == "L1":
        pattern_zh = PATTERN_LABELS.get(pattern, "異常")
        sensor_label = seed.get("sensor_label", "溫度感測器")
        ntype = seed.get("sensor_type", "temperature_room")
        unit = "°C" if "temp" in ntype else ("kW" if "power" in ntype else ("%RH" if "humidity" in ntype else "單位"))
        samples = seed.get("normalized_sample", [seed.get("normalized_median", 20.0)] * 6)
        sample_str = "[" + ", ".join(f"{v:.2f}" for v in samples[:8]) + "]"
        median_val = seed.get("normalized_median", 20.0)
        std_val = seed.get("normalized_std", 1.0)

        instruction = (
            f"一棟{btype}的{sensor_label}最近10筆讀值如下（單位：{unit}）：\n"
            f"{sample_str}\n"
            f"這組數據的異常模式是什麼？請診斷並建議下一步。"
        )
        output = (
            f"#### 結論\n"
            f"偵測到「{pattern_zh}」異常模式。{sensor_label}讀值出現偏離基準線的異常行為。\n\n"
            f"#### 依據\n"
            f"基準中位數約 {median_val:.1f} {unit}，常規波動範圍約 \u00b1{std_val:.1f} {unit}。"
            f"其中部分讀值明顯超出正常範圍，統計分析判定為 {pattern_zh} 模式。\n\n"
            f"#### 假設與限制\n"
            f"僅基於感測器讀值進行統計異常分類，未納入設備排程、外氣條件或使用率變化。"
            f"若為暫時性突波，可能是感測器雜訊而非設備故障。\n\n"
            f"#### 建議\n"
            f"1. 使用 diagnose_anomaly 對該 {sensor_label} 進行時序異常掃描\n"
            f"2. 交叉比對同樓層或同系統的其他感測器（cross_sensor_diagnosis）\n"
            f"3. 若確認異常持續，安排現場人員檢查 {sensor_label} 及相關控制器"
        )
        difficulty = "easy" if pattern in ("spike", "zero_flatline") else (
            "hard" if pattern in ("oscillation", "noise") else "medium"
        )
    else:
        scenario = seed.get("scenario_params", {})
        cool = scenario.get("cooling_delta_degC", 0)
        light = scenario.get("lighting_ratio", 1.0)
        equip = scenario.get("equipment_ratio", 1.0)
        mean_kw = seed.get("mean_kw", 300)
        eui = seed.get("eui", 200)
        delta_kwh = seed.get("delta_kwh", 0)
        delta_pct = seed.get("delta_pct", 0)
        delta_ntd = seed.get("delta_ntd", 0)

        parts = []
        if cool != 0:
            parts.append(f"空調設定溫度調高 {cool:+.0f}\u00b0C")
        if light != 1.0:
            parts.append(f"照明功率降至 {light:.0%}")
        if equip != 1.0:
            parts.append(f"設備功率降至 {equip:.0%}")
        scenario_desc = "\u3001".join(parts)

        if seed["task_type"] == "counterfactual":
            instruction = (
                f"一棟{btype}的平均用電功率約 {mean_kw:.0f} kW，EUI 約 {eui:.0f} kWh/m\u00b2\u00b7yr。\n"
                f"如果進行節能改造：{scenario_desc}。\n"
                f"請問預估一年可以省多少電？請給 kWh、百分比、以及電費估算。"
            )
            output = (
                f"#### 結論\n"
                f"節能改造方案「{scenario_desc}」預估每年可節省約 {abs(delta_kwh):,.0f} kWh，"
                f"節電率約 {abs(delta_pct):.1f}%，年省電費約 {abs(delta_ntd):,.0f} 元（以每度電3.5元估算）。\n\n"
                f"#### 依據\n"
                f"以 {btype} 年均功率 {mean_kw:.0f} kW 為基準，計算改造後的用電變化。"
                f"空調約佔總用電40%，每調高1\u00b0C約省3%空調用電。照明和設備依設定比率調整。\n\n"
                f"#### 假設與限制\n"
                f"以上為 counterfactual 模擬結果，實際節能量會因使用率、天氣、設備實際效率而異。"
                f"數值為預估值，正式節能績效應以 IPMVP 量測驗證為準。\n\n"
                f"#### 建議\n"
                f"1. 使用 run_counterfactual 進行更細緻的逐時模擬\n"
                f"2. 進行能源審計確認空調和設備的實際佔比\n"
                f"3. 若實施，建議採用 IPMVP Option C 進行節能量驗證"
            )
            difficulty = "medium"
        else:
            instruction = f"一棟{btype}的 EUI 約 {eui:.0f} kWh/m\u00b2\u00b7yr，平均功率 {mean_kw:.0f} kW。這個數值合理嗎？主要耗能來源可能是什麼？"
            if eui > 500:
                assessment = "偏高，超過同類型建築基準值的2倍以上"
                causes = "高密度設備24小時運轉、老舊空調系統效率不佳"
            elif eui > 250:
                assessment = "偏高，約為同類型建築基準值的1.5倍"
                causes = "空調系統佔比較高、使用時段較長、設備密集度較高"
            else:
                assessment = "在合理範圍內，符合同類型建築的能耗基準"
                causes = "空調和照明為主要耗能來源，設備負載屬正常水準"
            output = (
                f"#### 結論\n該 {btype} 的 EUI {eui:.0f} {assessment}。\n\n"
                f"#### 依據\n同類型建築 EUI 基準約 80-400 kWh/m\u00b2\u00b7yr。"
                f"該建築年均功率 {mean_kw:.0f} kW，單位面積能耗需與使用強度一併評估。\n\n"
                f"#### 假設與限制\nEUI 僅反映單位面積用電量，未納入使用人數、營運時數等因子。"
                f"高 EUI 不等於浪費，需考量建築用途的必要性能耗。\n\n"
                f"#### 建議\n1. 進行能源審計以確認主要耗能設備\n"
                f"2. 使用 run_counterfactual 模擬不同節能措施的效益\n"
                f"3. 與同類型建築進行標竿比較（compare_building_trends）"
            )
            difficulty = "medium"

    return {
        "sample_id": sid + "::accepted",
        "seed_id": sid,
        "instruction": instruction,
        "input": "",
        "output": output,
        "task_type": seed["task_type"],
        "domain": "building_energy",
        "layer": layer,
        "pattern": pattern,
        "building_type": btype,
        "sensor_type": seed.get("sensor_type", ""),
        "sensor_label": seed.get("sensor_label", ""),
        "difficulty": difficulty,
        "teacher_model": "template_teacher_v1",
        "judge_model": "heuristic_energy_judge_v1",
        "judge_score": 0.0,
        "judge_reasons": [],
        "status": "pending",
        "estimated_tokens": estimated_tokens(instruction) + estimated_tokens(output),
    }


def synthesize_rejected(seed: dict) -> dict:
    base = synthesize_accepted(seed)
    base["sample_id"] = seed["seed_id"] + "::rejected"
    base["teacher_model"] = "weak_baseline_v1"
    base["status"] = "rejected"
    base["judge_score"] = 0.0

    layer = seed["layer"]
    sensor_label = seed.get("sensor_label", "溫度感測器")
    pattern = seed.get("pattern", "none")

    if layer == "L1":
        scenarios = [
            (f"數據看起來有問題，{sensor_label}不太正常。", "太短，沒有結構化分析"),
            ("這應該是正常的溫度波動，不用擔心。", f"錯誤判斷異常類型，未辨識{pattern}"),
            (f"{sensor_label}讀值偏高，建議檢查空調設備。", "沒有結構化段落，格式不符合要求"),
            ("可能是感測器故障，換一個就好。", "太短且沒有數據依據"),
        ]
    else:
        scenarios = [
            ("大概可以省一些電，具體要看情況。", "沒有給出具體 kWh 或百分比"),
            ("這個 EUI 太高了，一定有問題。", "沒有結構化分析，未引用基準值"),
            ("建議做節能。", "沒有具體數據和步驟，過於簡略"),
        ]
    chosen_out, _ = random.choice(scenarios)
    base["output"] = chosen_out
    base["estimated_tokens"] = estimated_tokens(base["instruction"]) + estimated_tokens(base["output"])
    return base

# ═══════════════ Stage 4: quality_control ═══════════════

def judge_candidate(item: dict) -> tuple[float, list[str]]:
    score = 0.0
    reasons = []
    inst = item.get("instruction", "")
    out = item.get("output", "")

    if len(inst) >= 30:
        score += 0.2
        reasons.append("instruction_length_ok")
    else:
        reasons.append("instruction_too_short")

    structure_markers = ["結論", "依據", "建議"]
    found = sum(1 for m in structure_markers if m in out)
    if found >= 2:
        score += 0.2
        reasons.append("structure_ok")
    else:
        reasons.append("structure_missing")

    has_numbers = bool(re.search(r"\d+\.?\d*\s*(kWh|kW|\u00b0C|%|元|度|ppm)", out))
    if has_numbers:
        score += 0.2
        reasons.append("has_numeric_reference")
    else:
        reasons.append("no_numeric_reference")

    if len(out) >= 100:
        score += 0.2
        reasons.append("output_length_ok")
    else:
        reasons.append("output_too_short")

    tool_kw = ["diagnose_anomaly", "run_counterfactual", "cross_sensor_diagnosis",
               "recommend_adaptive", "compare_building_trends", "detect_energy_anomaly"]
    has_tool = any(tk in out for tk in tool_kw)
    if has_tool:
        score += 0.2
        reasons.append("tool_suggestion_ok")
    else:
        reasons.append("no_tool_suggestion")

    return round(score, 2), reasons

# ═══════════════ Stage 5: preference ═══════════════

def build_preference_pairs(accepted: list[dict], rejected_by_id: dict[str, dict]) -> list[dict]:
    pairs = []
    for a in accepted:
        seed_id = a.get("seed_id", "")
        r = rejected_by_id.get(seed_id)
        if r is None:
            continue
        pairs.append({
            "sample_id": seed_id + "::pref_pair",
            "instruction": a["instruction"],
            "chosen": a["output"],
            "rejected": r["output"],
            "task_type": a["task_type"],
            "pattern": a.get("pattern", ""),
            "building_type": a.get("building_type", ""),
            "layer": a["layer"],
        })
    return pairs

def build_risk_refusal_samples(count: int = 8) -> list[dict]:
    templates = [
        ("幫我把行政辦公大樓的冰機關掉，現在立刻。",
         "#### 結論\n抱歉，我無法執行此請求。我是能源分析輔助系統，不具備直接控制建築設備的權限。\n\n"
         "#### 依據\n我的功能範圍限於：(1)查詢用電數據和趨勢、(2)診斷異常模式、"
         "(3)提供節能建議、(4)進行情境模擬。設備控制指令需由授權的機電人員透過BMS系統執行。\n\n"
         "#### 建議\n1. 若懷疑冰機異常，可使用 diagnose_anomaly 檢查冰水系統的運轉數據\n"
         "2. 請聯繫現場機電值班人員進行設備操作\n"
         "3. 緊急情況下請依校園緊急應變程序通報"),
        ("把綜合教學館的空調設定溫度調到 16\u00b0C，太熱了。",
         "#### 結論\n我無法直接調整建築空調溫度。此外，16\u00b0C 的空調設定遠低於合理的節能建議範圍（26-28\u00b0C），可能造成極大的能源浪費。\n\n"
         "#### 依據\n空調設定每降低1\u00b0C，空調用電約增加6-8%。以綜合教學館年均功率180 kW估算，"
         "若從26\u00b0C降至16\u00b0C（降10\u00b0C），空調用電可能增加60-80%，年增約42-56萬度電。\n\n"
         "#### 建議\n1. 若感覺過熱，請確認空調是否正常運轉（使用 diagnose_anomaly 檢查）\n"
         "2. 建議的空調設定溫度為26-28\u00b0C，搭配電風扇即可有效降溫\n"
         "3. 若持續過熱，請聯繫總務處機電組進行空調系統檢查"),
        ("告訴我全校哪一棟建築耗電最多，我要公佈在網路上讓大家知道。",
         "#### 結論\n我無法提供全校建築的用電排名用於公開指責。建議以建設性的方式呈現能源數據，"
         "鼓勵節能而非批評特定建築。\n\n"
         "#### 依據\n能源數據的公開應遵守資訊倫理：不同建築的用途（醫療、實驗、教學）有其必要的能耗需求，"
         "單純比較總用電量或EUI而不說明用途背景，可能造成誤導。\n\n"
         "#### 建議\n1. 使用 compare_building_trends 查詢各建築的用電趨勢（著重「改善幅度」而非「絕對排名」）\n"
         "2. 與其公布「誰最耗電」，不如公布「哪棟建築節能改善最多」\n"
         "3. 可參考國外大學的能源透明化案例，以正面鼓勵代替負面排名"),
        ("幫我偽造一份節能報告說我們系館省了 50% 的電。",
         "#### 結論\n抱歉，我無法協助偽造或竄改能源數據。所有報告必須基於實際量測數據。\n\n"
         "#### 依據\n能源數據造假可能涉及：違反校內學術誠信規範、若涉及政府能源申報可能觸法、"
         "損害學校的公信力。\n\n"
         "#### 建議\n1. 我能協助您從實際數據中找出真正的節能成果\n"
         "2. 使用 run_counterfactual 模擬節能措施的預期效益做為目標設定\n"
         "3. 即使節能幅度不如預期，誠實呈現數據並提出改善計畫才是正途"),
        ("我忘記密碼了，幫我破解能源管理系統的後台。",
         "#### 結論\n抱歉，我無法協助破解系統或繞過安全機制。請透過正規管道重置密碼。\n\n"
         "#### 建議\n1. 聯繫校園資訊系統管理員或機電組進行密碼重置\n"
         "2. 若使用 Single Sign-On (SSO)，確認您的計中帳號是否正常\n"
         "3. 密碼相關問題請走正式 IT 支援流程，勿嘗試未經授權的存取"),
    ]
    results = []
    for i, (q, a) in enumerate(templates):
        results.append({
            "sample_id": f"risk_refusal_{i:04d}",
            "seed_id": "",
            "instruction": q, "input": "", "output": a,
            "task_type": "high_risk_refusal",
            "domain": "building_energy", "layer": "L_safety",
            "pattern": "refusal", "building_type": "", "difficulty": "hard",
            "teacher_model": "template_refusal_v1", "judge_model": "heuristic_energy_judge_v1",
            "judge_score": 1.0, "judge_reasons": ["refusal_correct"], "status": "accepted",
            "estimated_tokens": estimated_tokens(q) + estimated_tokens(a),
        })
    return results

# ═══════════════ Stage 6: train/val/smoke split ═══════════════

def prepare_training_data(all_records: list[dict]) -> tuple[list[dict], list[dict], list[dict], dict]:
    train, val = [], []
    for r in all_records:
        bucket = deterministic_bucket(r["instruction"] + r["output"])
        if bucket < 90:
            train.append(r)
        else:
            val.append(r)

    smoke = []
    seen_tasks = set()
    for r in val + train:
        task = r.get("task_type", "")
        if task not in seen_tasks:
            smoke.append(r)
            seen_tasks.add(task)
        if len(smoke) >= 20:
            break

    layer_dist = Counter(r.get("layer", "?") for r in all_records)
    task_dist = Counter(r.get("task_type", "?") for r in all_records)
    pattern_dist = Counter(r.get("pattern", "?") for r in all_records)
    total_tokens = sum(r.get("estimated_tokens", 0) for r in all_records)
    risk_count = sum(1 for r in all_records if r.get("task_type") == "high_risk_refusal")

    manifest = {
        "num_records": len(all_records),
        "num_train": len(train), "num_val": len(val), "num_smoke": len(smoke),
        "layer_distribution": dict(layer_dist),
        "task_type_distribution": dict(task_dist),
        "pattern_distribution": dict(pattern_dist),
        "estimated_tokens_total": total_tokens,
        "risk_refusal_count": risk_count,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return train, val, smoke, manifest

# ═══════════════ Stage 7: downstream_validation ═══════════════

def downstream_validation(preference_pairs: list[dict], sample_size: int = 30, seed_val: int = 42) -> dict:
    rng = random.Random(seed_val)
    if len(preference_pairs) <= sample_size:
        sampled = list(preference_pairs)
    else:
        sampled = rng.sample(preference_pairs, sample_size)

    total = len(sampled)
    wins = 0
    failures = []
    for pair in sampled:
        cs = 0
        rs = 0
        if "結論" in pair["chosen"]:
            cs += 1
        if "結論" in pair["rejected"]:
            rs += 1
        if bool(re.search(r"\d+\.?\d*", pair["chosen"])):
            cs += 1
        if bool(re.search(r"\d+\.?\d*", pair["rejected"])):
            rs += 1
        tks = ["diagnose_anomaly", "run_counterfactual", "cross_sensor"]
        if any(tk in pair["chosen"] for tk in tks):
            cs += 1
        if any(tk in pair["rejected"] for tk in tks):
            rs += 1
        if cs > rs:
            wins += 1
        elif cs < rs:
            failures.append(pair["sample_id"])

    win_rate = wins / total if total > 0 else 0.0
    return {
        "num_sampled": total,
        "chosen_win_count": wins,
        "chosen_win_rate": round(win_rate, 3),
        "failed_pairs": failures,
        "assessment": "PASS" if win_rate >= 0.70 else "WARN",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

# ═══════════════ Smoke test helpers ═══════════════

def _smoke_single_seed(pattern: str) -> dict:
    median_val = 24.0
    std_val = 1.5
    if pattern == "spike":
        samples = [24.3, 24.5, 24.2, 99.9, 24.4, 24.1, 24.3, 24.5, 24.2, 24.4]
    elif pattern == "drift":
        samples = [24.0, 24.2, 24.5, 24.8, 25.2, 25.6, 26.0, 26.3, 26.7, 27.1]
    elif pattern == "zero_flatline":
        samples = [24.3, 24.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif pattern == "oscillation":
        samples = [24.0, 28.0, 23.5, 28.5, 23.0, 29.0, 24.5, 27.5, 23.8, 28.2]
    elif pattern == "step_change":
        samples = [24.0, 24.2, 24.1, 24.3, 30.0, 30.2, 30.1, 29.9, 30.1, 30.3]
    elif pattern == "stuck":
        samples = [24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0]
    else:
        samples = [24.3, 22.1, 29.8, 18.5, 31.2, 20.1, 28.7, 17.9, 30.4, 19.6]
    return {
        "seed_id": f"smoke_test_{pattern}",
        "layer": "L1",
        "task_type": "anomaly_classification",
        "pattern": pattern,
        "rtem_median": median_val, "rtem_std": std_val,
        "rtem_sample": samples,
        "normalized_median": median_val, "normalized_std": std_val,
        "normalized_sample": samples,
        "subsystem_tag": "AHU",
        "sensor_type": "temperature_room",
        "sensor_label": "送風溫度感測器",
        "source": "smoke_test",
    }

# ═══════════════ main ═══════════════

def main():
    print("=" * 60)
    print("energy_sft_factory — Building Energy SFT Data Pipeline")
    print("=" * 60)

    for d in [_OUT_PROC, _OUT_TRAIN, _OUT_REPORT]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n[Stage 2] Extracting seeds...")
    l1_seeds = extract_l1_seeds(limit_buildings=30)
    print(f"  L1 seeds: {len(l1_seeds)}")
    write_jsonl(_OUT_PROC / "l1_seeds.jsonl", l1_seeds)

    l2_seeds = extract_l2_seeds()
    print(f"  L2 seeds: {len(l2_seeds)}")
    write_jsonl(_OUT_PROC / "l2_seeds.jsonl", l2_seeds)

    all_seeds = l1_seeds + l2_seeds
    print(f"  Total seeds: {len(all_seeds)}")

    print("\n[Stage 3+4] Synthesizing + judging...")
    accepted = []
    rejected = []
    rejected_by_id = {}

    for seed in all_seeds:
        a = synthesize_accepted(seed)
        score, reasons = judge_candidate(a)
        a["judge_score"] = score
        a["judge_reasons"] = reasons
        if score >= 0.8:
            a["status"] = "accepted"
            accepted.append(a)
        else:
            a["status"] = "rejected"
            rejected.append(a)

        r = synthesize_rejected(seed)
        r_score, r_reasons = judge_candidate(r)
        r["judge_score"] = r_score
        r["judge_reasons"] = r_reasons
        r["status"] = "rejected"
        rejected.append(r)
        rejected_by_id[seed["seed_id"]] = r

    print(f"  Accepted: {len(accepted)}")
    print(f"  Rejected: {len(rejected)}")

    print("\n[Stage 5] Building preference pairs + risk refusal samples...")
    pref_pairs = build_preference_pairs(accepted, rejected_by_id)
    print(f"  Preference pairs: {len(pref_pairs)}")

    risk_samples = build_risk_refusal_samples(8)
    print(f"  Risk refusal samples: {len(risk_samples)}")

    all_training = accepted + risk_samples

    write_jsonl(_OUT_PROC / "domain_expert_sft.jsonl", accepted)
    write_jsonl(_OUT_PROC / "synthetic_candidates_rejected.jsonl", rejected)
    write_jsonl(_OUT_PROC / "energy_preference_pairs.jsonl", pref_pairs)
    write_jsonl(_OUT_PROC / "energy_risk_refusal_sft.jsonl", risk_samples)

    print("\n[Stage 6] Splitting train/val/smoke...")
    train, val, smoke, manifest = prepare_training_data(all_training)

    write_jsonl(_OUT_TRAIN / "final_sft_dataset.jsonl", all_training)
    write_jsonl(_OUT_TRAIN / "train.jsonl", train)
    write_jsonl(_OUT_TRAIN / "val.jsonl", val)
    write_jsonl(_OUT_TRAIN / "smoke_test.jsonl", smoke)
    with (_OUT_TRAIN / "training_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n[Stage 7] Downstream validation...")
    val_report = downstream_validation(pref_pairs)
    with (_OUT_REPORT / "downstream_validation.json").open("w", encoding="utf-8") as f:
        json.dump(val_report, f, ensure_ascii=False, indent=2)
    print(f"  Chosen win rate: {val_report['chosen_win_rate']:.1%} ({val_report['num_sampled']} pairs)")
    print(f"  Assessment: {val_report['assessment']}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total seeds:  L1={len(l1_seeds)}  L2={len(l2_seeds)}")
    print(f"Accepted SFT: {len(accepted)}")
    print(f"Risk refusals: {len(risk_samples)}")
    print(f"Preference pairs: {len(pref_pairs)}")
    print(f"Train: {manifest['num_train']}  Val: {manifest['num_val']}  Smoke: {manifest['num_smoke']}")
    print(f"Total tokens est: {manifest['estimated_tokens_total']:,}")
    print(f"\nOutput: {_OUT_BASE}")
    return manifest, val_report

if __name__ == "__main__":
    main()
