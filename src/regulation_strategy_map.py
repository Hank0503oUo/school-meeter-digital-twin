from __future__ import annotations

from typing import Any

BUILDING_TYPE_REGULATIONS: dict[str, dict[str, Any]] = {
    "Academic Units": {
        "label": "教學/學術",
        "standard_eui": 85,
        "cooling_load_ref_w_m2": 200,
        "lighting_power_density_ref": 15,
        "regulation_queries": [
            "學校建築能效 教學空間",
            "空調節能 學術建築",
            "照明功率密度 教室",
        ],
        "bee_weights": {"envelope": 0.30, "hvac": 0.40, "lighting": 0.30},
    },
    "Instructional Building": {
        "label": "教學大樓",
        "standard_eui": 85,
        "cooling_load_ref_w_m2": 180,
        "lighting_power_density_ref": 15,
        "regulation_queries": [
            "學校建築能效 教學空間",
            "空調節能 教學大樓",
            "教室照明標準",
        ],
        "bee_weights": {"envelope": 0.30, "hvac": 0.40, "lighting": 0.30},
    },
    "Library": {
        "label": "圖書館",
        "standard_eui": 120,
        "cooling_load_ref_w_m2": 220,
        "lighting_power_density_ref": 20,
        "regulation_queries": [
            "圖書館建築能效",
            "書庫空調 恆溫恆濕",
            "圖書館照明 CNS 照度",
        ],
        "bee_weights": {"envelope": 0.25, "hvac": 0.45, "lighting": 0.30},
    },
    "Administration": {
        "label": "行政辦公",
        "standard_eui": 120,
        "cooling_load_ref_w_m2": 200,
        "lighting_power_density_ref": 12,
        "regulation_queries": [
            "辦公建築能效 EUI",
            "辦公室空調節能 VRF",
            "辦公室照明 LED",
        ],
        "bee_weights": {"envelope": 0.30, "hvac": 0.40, "lighting": 0.30},
    },
    "Dormitories": {
        "label": "宿舍",
        "standard_eui": 65,
        "cooling_load_ref_w_m2": 150,
        "lighting_power_density_ref": 10,
        "regulation_queries": [
            "宿舍建築能效",
            "住宿類 綠建標章",
            "宿舍照明 室內環境",
        ],
        "bee_weights": {"envelope": 0.35, "hvac": 0.35, "lighting": 0.30},
    },
    "Athletics": {
        "label": "運動設施",
        "standard_eui": 100,
        "cooling_load_ref_w_m2": 250,
        "lighting_power_density_ref": 15,
        "regulation_queries": [
            "體育館建築能效",
            "大空間空調 節能",
            "體育場館照明",
        ],
        "bee_weights": {"envelope": 0.25, "hvac": 0.50, "lighting": 0.25},
    },
    "Student AC": {
        "label": "學生活動中心",
        "standard_eui": 110,
        "cooling_load_ref_w_m2": 200,
        "lighting_power_density_ref": 14,
        "regulation_queries": [
            "活動中心建築能效",
            "多功能空間空調",
            "活動中心照明",
        ],
        "bee_weights": {"envelope": 0.30, "hvac": 0.40, "lighting": 0.30},
    },
    "Others": {
        "label": "其他",
        "standard_eui": 100,
        "cooling_load_ref_w_m2": 180,
        "lighting_power_density_ref": 13,
        "regulation_queries": [
            "建築能效 EUI 基準",
            "空調節能",
            "照明節能",
        ],
        "bee_weights": {"envelope": 0.30, "hvac": 0.40, "lighting": 0.30},
    },
}

FACTOR_STRATEGY_MAP: dict[str, dict[str, Any]] = {
    "cooling_load": {
        "label": "冷卻/空調負載",
        "strategies": [
            {
                "key": "cooling_delta_degC",
                "values": [1.0, 2.0, 3.0],
                "label": "冷房溫度上調",
                "unit": "°C",
                "difficulty": "low",
                "cost_level": "free",
            },
            {
                "key": "cop_ratio",
                "values": [1.1, 1.2, 1.3],
                "label": "冰水主機 COP 提升",
                "unit": "倍",
                "difficulty": "medium",
                "cost_level": "medium",
            },
        ],
        "regulation_refs": [
            "建築能效評估：HVAC 系統權重佔 40%",
            "EEWH 日常節能指標（EE）佔 21% 權重",
            "智慧建築標章：節能管理構面佔 10%",
            "建築技術規則：空調設備效率基準（COP 最低要求）",
        ],
    },
    "lighting_load": {
        "label": "照明負載",
        "strategies": [
            {
                "key": "lighting_ratio",
                "values": [0.9, 0.8, 0.7],
                "label": "照明功率密度調降",
                "unit": "倍",
                "difficulty": "low",
                "cost_level": "low",
            },
        ],
        "regulation_refs": [
            "建築能效評估：照明系統權重佔 30%",
            "CNS 照度標準（各空間類型最低照度）",
            "建築技術規則：窗地面積比 ≥ 1/7",
            "綠建築標章：日常節能 - 照明節能",
        ],
    },
    "equipment_load": {
        "label": "設備/插座負載",
        "strategies": [
            {
                "key": "equipment_ratio",
                "values": [0.95, 0.90, 0.85],
                "label": "設備功率調降",
                "unit": "倍",
                "difficulty": "medium",
                "cost_level": "low",
            },
        ],
        "regulation_refs": [
            "建築能效評估：外殼 + 空調 + 照明 = 100%，設備效率影響基準值",
            "智慧建築：設備監控與排程管理",
        ],
    },
    "occupancy_load": {
        "label": "人員/使用率負載",
        "strategies": [
            {
                "key": "occupancy_ratio",
                "values": [0.95, 0.90, 0.85],
                "label": "人員密度/使用率最佳化",
                "unit": "倍",
                "difficulty": "low",
                "cost_level": "free",
            },
        ],
        "regulation_refs": [
            "智慧建築：空間管理與排程最佳化",
            "室內環境品質：CO₂ < 1000 ppm，通風率 ≥ 30 dm³/s·人",
        ],
    },
    "operational_variability": {
        "label": "操作變異/控制品質",
        "strategies": [
            {
                "key": "cooling_delta_degC",
                "values": [1.0, 1.5],
                "label": "控制序列校正（含溫度上調）",
                "unit": "°C",
                "difficulty": "medium",
                "cost_level": "low",
            },
        ],
        "regulation_refs": [
            "智慧建築標章：節能管理構面（自動控制 + BEMS）",
            "EEWH 日常節能：空調效率提升",
        ],
    },
    "weather_driven_load": {
        "label": "氣候驅動負載",
        "strategies": [
            {
                "key": "cooling_delta_degC",
                "values": [1.0, 2.0],
                "label": "外氣溫度連動控制調適",
                "unit": "°C",
                "difficulty": "medium",
                "cost_level": "low",
            },
        ],
        "regulation_refs": [
            "建築外殼性能：U-value 基準（牆面 ≤ 0.70，窗戶 ≤ 3.69 W/m²K）",
            "綠建築標章：日常節能 - 外殼節能",
        ],
    },
}

BEE_RATING_SCALE: list[dict[str, Any]] = [
    {"level": 1, "label": "最優級", "threshold": "≥ 基準 -50%"},
    {"level": 2, "label": "優良級", "threshold": "基準 -35%~-50%"},
    {"level": 3, "label": "佳級", "threshold": "基準 -20%~-35%"},
    {"level": 4, "label": "普級", "threshold": "基準 -5%~-20%"},
    {"level": 5, "label": "基準級", "threshold": "基準 ±5%"},
    {"level": 6, "label": "待改善", "threshold": "基準 +5%~+20%"},
    {"level": 7, "label": "需大幅改善", "threshold": "> 基準 +20%"},
]


def classify_bee_level(current_eui: float, baseline_eui: float) -> dict[str, Any]:
    if baseline_eui <= 0:
        return {"level": 5, "label": "基準級", "gap_pct": 0.0}
    gap_pct = (current_eui - baseline_eui) / baseline_eui * 100.0
    if gap_pct <= -50:
        entry = BEE_RATING_SCALE[0]
    elif gap_pct <= -35:
        entry = BEE_RATING_SCALE[1]
    elif gap_pct <= -20:
        entry = BEE_RATING_SCALE[2]
    elif gap_pct <= -5:
        entry = BEE_RATING_SCALE[3]
    elif gap_pct <= 5:
        entry = BEE_RATING_SCALE[4]
    elif gap_pct <= 20:
        entry = BEE_RATING_SCALE[5]
    else:
        entry = BEE_RATING_SCALE[6]
    return {"level": entry["level"], "label": entry["label"], "gap_pct": round(gap_pct, 1)}


def get_regulation_for_building(build_type: str) -> dict[str, Any]:
    return BUILDING_TYPE_REGULATIONS.get(
        build_type,
        BUILDING_TYPE_REGULATIONS["Others"],
    )


def get_strategies_for_factor(dominant_factor: str) -> dict[str, Any]:
    return FACTOR_STRATEGY_MAP.get(
        dominant_factor,
        FACTOR_STRATEGY_MAP["cooling_load"],
    )
