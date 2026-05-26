"""
Layer-2 synthetic training data: counterfactual + efficiency diagnosis.

Generates QA pairs from NTU building stats + counterfactual engine.
Questions ask "what if we change X" → answers show delta_kwh, delta_pct, etc.

Usage:
    python tools/generate_synthetic_L2.py --max-samples 80
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.counterfactual import run_building_counterfactual
from src.demo_mcp_server import _load_stats_df

_ARCHETYPE_ZH = {
    "Baseload-driven (Irregular)": "基載為主（不規律）",
    "Lighting-dominant (Highly regular)": "照明為主（高度規律）",
    "Mixed-load (Complex)": "混合負載（複雜）",
    "Mixed-load (Schedule-driven)": "混合負載（排程驅動）",
    "HVAC-dominant (Volatile)": "空調為主（波動大）",
}

_FACTOR_ZH = {
    "cooling": "空調溫度",
    "lighting": "照明功率",
    "occupancy": "人員密度",
    "equipment": "設備負載",
}

_FACTOR_UNIT = {
    "cooling": "°C",
    "lighting": "%",
    "occupancy": "%",
    "equipment": "%",
}

_SCENARIO_TEMPLATES = [
    {
        "name": "cooling_up",
        "params": {"cooling_delta_degC": 1.0},
        "question_tmpl": "如果把{building}的空調設定溫度調高 1°C，一年可以省多少電？",
        "factor": "cooling",
    },
    {
        "name": "cooling_up_2",
        "params": {"cooling_delta_degC": 2.0},
        "question_tmpl": "如果把{building}的空調設定溫度調高 2°C，對年用電量有什麼影響？",
        "factor": "cooling",
    },
    {
        "name": "lighting_80",
        "params": {"lighting_ratio": 0.8},
        "question_tmpl": "如果{building}把照明功率降到 80%（更換 LED），可以節省多少用電？",
        "factor": "lighting",
    },
    {
        "name": "lighting_70",
        "params": {"lighting_ratio": 0.7},
        "question_tmpl": "{building}全面更換高效照明（降至 70%），預估節電效果如何？",
        "factor": "lighting",
    },
    {
        "name": "equipment_90",
        "params": {"equipment_ratio": 0.9},
        "question_tmpl": "如果{building}將設備負載降低 10%（老舊設備更新），節電量多少？",
        "factor": "equipment",
    },
    {
        "name": "combined_conservative",
        "params": {"cooling_delta_degC": 1.0, "lighting_ratio": 0.85, "equipment_ratio": 0.95},
        "question_tmpl": "{building}同時實施：空調+1°C、照明降至85%、設備降至95%，總節電量多少？",
        "factor": "combined",
    },
    {
        "name": "combined_aggressive",
        "params": {"cooling_delta_degC": 2.0, "lighting_ratio": 0.7, "equipment_ratio": 0.85},
        "question_tmpl": "{building}進行大規模節能改造（空調+2°C、照明70%、設備85%），預估年度節電率？",
        "factor": "combined",
    },
    {
        "name": "occupancy_reduce",
        "params": {"occupancy_ratio": 0.8},
        "question_tmpl": "如果{building}實施彈性上班，人員密度降為 80%，對用電有何影響？",
        "factor": "occupancy",
    },
]

_DIAGNOSIS_TEMPLATES = [
    {
        "question_tmpl": "{building}的 EUI 是 {eui}，這個數值合理嗎？應該怎麼改善？",
        "type": "eui_diagnosis",
    },
    {
        "question_tmpl": "{building}年用電 {annual_kwh:,.0f} kWh，平均功率 {mean_kw:.0f} kW，哪個面向最值得優先改善？",
        "type": "priority_diagnosis",
    },
    {
        "question_tmpl": "{building}的負載型態是「{archetype}」，這代表什麼？適合什麼節能策略？",
        "type": "archetype_diagnosis",
    },
]


def _format_number(n: float) -> str:
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:.1f} 百萬"
    if abs(n) >= 1_000:
        return f"{n/1_000:,.0f} 萬" if abs(n) >= 10_000 else f"{n:,.0f}"
    return f"{n:.1f}"


def _anonymize_building(row: dict[str, Any]) -> str:
    mean_kw = float(row.get("mean_kw", 0) or 0)
    archetype = str(row.get("archetype_label", ""))
    if "HVAC" in archetype:
        return random.choice(["體育館", "大型演藝廳"])
    if "Lighting" in archetype:
        return random.choice(["綜合教學館", "校級圖書館", "設計學院大樓"])
    if "Baseload" in archetype:
        return random.choice(["理工實驗館", "計算機中心", "醫學研究館"])
    return random.choice(["行政辦公大樓", "跨域大樓", "文教大樓"])


def _build_counterfactual_qa(
    building_row: dict[str, Any],
    template: dict[str, Any],
) -> dict[str, Any] | None:
    mean_kw = float(building_row.get("mean_kw", 0) or 0)
    if mean_kw <= 0:
        return None
    name = _anonymize_building(building_row)
    rd = {str(k): building_row[k] for k in building_row}
    result = run_building_counterfactual(rd, **template["params"])
    delta_kwh = float(result["delta_kwh"])
    delta_pct = float(result["delta_pct"])
    if abs(delta_kwh) < 1:
        return None
    question = template["question_tmpl"].format(building=name)
    saving_or_cost = "節省" if delta_kwh < 0 else "增加"
    abs_kwh = abs(delta_kwh)
    carbon = abs(delta_kwh) * 0.494 / 1000
    ntd = abs(delta_kwh) * 2.5
    params_desc_parts = []
    for k, v in template["params"].items():
        if k == "cooling_delta_degC":
            params_desc_parts.append("空調設定溫度調高 %.1f°C" % v)
        elif k == "lighting_ratio":
            params_desc_parts.append("照明功率降至 %.0f%%" % (v * 100))
        elif k == "equipment_ratio":
            params_desc_parts.append("設備負載降至 %.0f%%" % (v * 100))
        elif k == "occupancy_ratio":
            params_desc_parts.append("人員密度降至 %.0f%%" % (v * 100))
    params_desc = "、".join(params_desc_parts)
    assistant = (
        f"結論：{params_desc}，預估每年可{saving_or_cost}用電 {_format_number(abs_kwh)} kWh "
        f"（{abs(delta_pct):.1f}%）。\n\n"
        f"明細：\n"
        f"- 電費影響：{saving_or_cost} NT${ntd:,.0f}/年\n"
        f"- 碳排影響：{saving_or_cost} {carbon:,.0f} kgCO2e/年\n"
        f"- 等效減碳：約 {abs(delta_kwh)*0.494/21:.0f} 棵樹的年吸收量\n\n"
        f"依據：此建築平均功率 {mean_kw:.0f} kW，"
        f"年用電量約 {mean_kw*8760:,.0f} kWh。"
        f"counterfactual 模型使用敏感性係數 "
        f"(cooling: -3%/°C, lighting: 15%, equipment: 35%) 估算。\n\n"
        f"建議工具：可呼叫 run_counterfactual_for_building() "
        f"進一步模擬其他參數組合，或呼叫 recommend_adaptive_strategies() 獲取法規對齊的節能建議。"
    )
    return {
        "user": question,
        "assistant": assistant,
        "metadata": {
            "layer": "L2_counterfactual",
            "scenario": template["name"],
            "factor": template["factor"],
            "building_type": name,
            "mean_kw": mean_kw,
            "delta_kwh": delta_kwh,
            "delta_pct": delta_pct,
        },
    }


def _build_diagnosis_qa(
    building_row: dict[str, Any],
    template: dict[str, Any],
) -> dict[str, Any] | None:
    mean_kw = float(building_row.get("mean_kw", 0) or 0)
    eui = float(building_row.get("eui", 0) or 0)
    archetype = str(building_row.get("archetype_label", ""))
    annual_kwh = mean_kw * 8760
    if mean_kw <= 0:
        return None
    name = _anonymize_building(building_row)
    question = template["question_tmpl"].format(
        building=name, eui=eui, annual_kwh=annual_kwh,
        mean_kw=mean_kw, archetype=_ARCHETYPE_ZH.get(archetype, archetype),
    )
    archetype_zh = _ARCHETYPE_ZH.get(archetype, archetype)
    rd = {str(k): building_row[k] for k in building_row}
    best_scenario = None
    best_delta = 0
    for tmpl in _SCENARIO_TEMPLATES:
        r = run_building_counterfactual(rd, **tmpl["params"])
        d = abs(float(r["delta_kwh"]))
        if d > best_delta:
            best_delta = d
            best_scenario = tmpl
    if eui > 300:
        efficiency_level = "偏高"
        suggestion = "建議優先改善照明與空調效率"
    elif eui > 150:
        efficiency_level = "中等"
        suggestion = "仍有節能空間，建議從照明或空調著手"
    else:
        efficiency_level = "良好"
        suggestion = "維持現狀，可考慮微調空調設定"
    dominant_map = {
        "Lighting-dominant": "照明",
        "HVAC-dominant": "空調",
        "Baseload-driven": "基載設備",
        "Mixed-load": "混合負載",
    }
    dominant = "混合"
    for key, val in dominant_map.items():
        if key in archetype:
            dominant = val
            break
    assistant = (
        f"結論：這棟{archetype_zh}建築的 EUI 為 {eui:.0f}，屬於「{efficiency_level}」等級。\n\n"
        f"分析：\n"
        f"- 負載型態：{archetype_zh}，主要耗能來源為「{dominant}」\n"
        f"- 平均功率：{mean_kw:.0f} kW，年用電量約 {annual_kwh:,.0f} kWh\n"
        f"- 效率評估：EUI {eui:.0f} kWh/m²·yr，{suggestion}\n\n"
    )
    if best_scenario:
        r = run_building_counterfactual(rd, **best_scenario["params"])
        assistant += (
            f"最大節電方案：{best_scenario['name']}\n"
            f"- 預估節電 {abs(float(r['delta_kwh'])):,.0f} kWh/年 ({abs(float(r['delta_pct'])):.1f}%)\n"
            f"- 電費節省約 NT${abs(float(r['delta_kwh']))*2.5:,.0f}/年\n\n"
        )
    assistant += (
        f"建議工具：呼叫 recommend_adaptive_strategies() "
        f"可取得法規對齊的詳細節能建議。"
    )
    return {
        "user": question,
        "assistant": assistant,
        "metadata": {
            "layer": "L2_efficiency_diagnosis",
            "diagnosis_type": template["type"],
            "building_type": name,
            "mean_kw": mean_kw,
            "eui": eui,
            "archetype": archetype,
        },
    }


def generate_L2_dataset(max_samples: int = 80, seed: int = 42) -> list[dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    df = _load_stats_df()
    if df.empty:
        print("ERROR: No building stats available.")
        return []
    qa_pairs: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rd = row.to_dict()
        for tmpl in _SCENARIO_TEMPLATES:
            pair = _build_counterfactual_qa(rd, tmpl)
            if pair:
                qa_pairs.append(pair)
        for tmpl in _DIAGNOSIS_TEMPLATES:
            pair = _build_diagnosis_qa(rd, tmpl)
            if pair:
                qa_pairs.append(pair)
    random.shuffle(qa_pairs)
    return qa_pairs[:max_samples]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate L2 counterfactual training data")
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_path = Path(args.output) if args.output else _ROOT / "data" / "synthetic_L2_counterfactual.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    qa_pairs = generate_L2_dataset(max_samples=args.max_samples, seed=args.seed)
    with output_path.open("w", encoding="utf-8") as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    layer_dist = Counter(p["metadata"]["layer"] for p in qa_pairs)
    factor_dist = Counter(p["metadata"].get("factor", p["metadata"].get("diagnosis_type", "?")) for p in qa_pairs)
    print(f"Wrote {len(qa_pairs)} QA pairs to {output_path}")
    print(f"Layer distribution: {dict(layer_dist)}")
    print(f"Factor/type distribution: {dict(factor_dist)}")


if __name__ == "__main__":
    main()
