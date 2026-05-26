from __future__ import annotations

import re
from typing import Any


IntentRule = dict[str, Any]


INTENT_RULES: list[IntentRule] = [
    {
        "intent": "search_docs",
        "tool": "search_docs",
        "domain": "knowledge_base",
        "keywords_zh": [
            "法規", "法律", "排煙", "建築執照", "法規查詢", "法規問題",
            "建築法", "消防", "防火", "避難", "耐震", "綠建築",
            "知識庫", "文件搜尋", "HJPLUS", "hjplus-kb",
        ],
        "keywords_en": [
            "search_docs", "hjplus", "knowledge", "rag",
            "regulation", "building code", "legal",
        ],
        "examples": [
            "排煙窗法規是什麼",
            "HJPLUS 建築法規",
            "搜尋文件",
        ],
    },
    {
        "intent": "compare_energy_usage",
        "tool": "compare_energy_usage",
        "domain": "energy_comparison",
        "keywords_zh": [
            "比較", "對比", "比較用電", "比較耗電", "電力比較",
            "全年電力", "年度用電", "總用電", "總耗電",
            "全校用電", "校園用電", "整體用電", "總體用電",
            "台大用電", "臺大用電",
            "電力消耗", "用電量差異", "耗電量",
            "差多少", "增減", "變化量", "年增率",
            "逐月比較", "月度比較", "月用電",
        ],
        "keywords_en": [
            "compare", "comparison", "vs", "versus",
            "annual electricity", "total consumption",
            "campus-wide", "year-over-year", "yoy",
            "delta", "change in", "difference",
            "NTU total", "campus energy",
        ],
        "examples": [
            "比較2016跟2020的全年電力消耗",
            "台大2018和2019年總用電量差多少",
            "全校2017年用電比2016年多還是少",
            "compare NTU 2016 vs 2017 total electricity",
            "NTU 2018 跟 2020 的全校用電量比較",
        ],
    },
    {
        "intent": "query_energy_records",
        "tool": "query_energy_records",
        "domain": "energy_data",
        "keywords_zh": [
            "跨年", "跨年度", "多年", "歷年",
            "用電量", "耗電", "電力",
            "查詢用電", "用電資料", "能源資料",
            "年度資料", "年度數據",
        ],
        "keywords_en": [
            "energy records", "energy data",
            "query energy", "electricity data",
        ],
        "examples": [
            "查2016到2020的用電資料",
            "給我看歷年耗電數據",
        ],
    },
    {
        "intent": "rank_energy_buildings",
        "tool": "rank_energy_buildings_across_years",
        "domain": "energy_ranking",
        "keywords_zh": [
            "排名", "最高", "最低", "最多", "最少",
            "前幾名", "top", "排行", "用電大戶",
            "最耗電", "最省電",
        ],
        "keywords_en": [
            "top", "rank", "ranking", "highest", "lowest",
            "most energy", "biggest consumer",
        ],
        "examples": [
            "2016年用電最高的建築",
            "全校top 10耗電建築",
        ],
    },
    {
        "intent": "compare_building_trends",
        "tool": "compare_building_trends",
        "domain": "building_trends",
        "keywords_zh": [
            "趨勢", "年增", "趨勢圖",
            "單棟比較", "建築趨勢", "某棟",
        ],
        "keywords_en": [
            "trend", "building trend",
        ],
        "examples": [
            "總圖書館的用電趨勢",
            "體育館2016-2020用電趨勢",
        ],
    },
    {
        "intent": "generate_chart",
        "tool": "generate_meter_chart",
        "domain": "visualization",
        "keywords_zh": [
            "電表", "折線圖", "長條圖", "比較圖",
            "視覺化", "圖表", "畫圖", "產圖", "生成圖",
            "圖", "繪圖", "製圖", "曲線圖",
        ],
        "keywords_en": [
            "csv", "chart", "plot", "visual", "visualize",
            "line chart", "bar chart", "graph",
        ],
        "examples": [
            "畫一張折線圖",
            "電表CSV視覺化",
        ],
    },
    {
        "intent": "analyze_screenshot",
        "tool": "analyze_meter_screenshot",
        "domain": "image_analysis",
        "keywords_zh": [
            "截圖", "圖片", "照片", "看圖",
            "電表截圖", "圖表截圖",
            "辨識", "OCR",
        ],
        "keywords_en": [
            "screenshot", "image", "photo",
            "uploaded_image_path",
        ],
        "examples": [
            "看這張電表截圖",
            "幫我辨識這張圖",
        ],
    },
    {
        "intent": "run_pvid",
        "tool": "run_pvid",
        "domain": "prediction",
        "keywords_zh": [
            "預測", "推論", "負載預測", "用電預測",
            "PI-VD", "pvid", "模擬",
        ],
        "keywords_en": [
            "predict", "pvid", "inference", "forecast",
            "simulate", "simulation",
        ],
        "examples": [
            "預測明天用電量",
            "run PI-VD for 24h",
        ],
    },
    {
        "intent": "counterfactual",
        "tool": "run_counterfactual_for_building",
        "domain": "what_if",
        "keywords_zh": [
            "假設", "如果", "情境分析", "節能方案",
            "降溫", "調整空調", "照明改善",
            "counterfactual", "what-if",
            "省多少", "可以省", "節省",
        ],
        "keywords_en": [
            "counterfactual", "what if", "scenario",
            "saving", "save energy",
        ],
        "examples": [
            "如果空調降1度能省多少",
            "照明改善5%的節能效果",
        ],
    },
    {
        "intent": "anomaly_detection",
        "tool": "detect_energy_anomalies",
        "domain": "anomaly",
        "keywords_zh": [
            "異常", "偵測異常", "異常用電", "突波",
            "不正常", "可疑",
        ],
        "keywords_en": [
            "anomaly", "anomalies", "detect", "outlier",
            "abnormal", "spike",
        ],
        "examples": [
            "偵測異常用電",
            "找出用電突波",
        ],
    },
    {
        "intent": "adaptive_strategy",
        "tool": "recommend_adaptive_strategies",
        "domain": "strategy",
        "keywords_zh": [
            "節能策略", "調適策略", "改善建議", "節能建議",
            "可以怎麼省", "節能對策", "設備調整", "節電方案",
            "節能改善", "調適建議", "節能推薦", "節能調適",
            "法規建議", "節能規劃", "節能方案建議",
        ],
        "keywords_en": [
            "strategy", "strategies", "recommendation",
            "retrofit", "energy saving plan", "adaptive",
            "improvement plan", "action plan",
        ],
        "examples": [
            "共同教學館有什麼節能調適建議",
            "圖書館的節能策略",
            "生科館可以怎麼省電",
            "給我這棟建築的節能改善方案",
        ],
    },
    {
        "intent": "seasonal_strategy",
        "tool": "seasonal_strategies",
        "domain": "seasonal",
        "keywords_zh": [
            "季節策略", "夏季節能", "冬季節能", "過渡季",
            "空調季節", "不同季節", "季節性", "分季節",
            "夏天空調", "冬天照明", "夏季調適", "冬季調適",
        ],
        "keywords_en": [
            "seasonal", "summer", "winter", "transition season",
            "seasonal strategy", "per season",
        ],
        "examples": [
            "這棟建築不同季節的節能策略",
            "夏季空調怎麼調適",
            "冬天照明有什麼改善方案",
        ],
    },
    {
        "intent": "portfolio_optimization",
        "tool": "optimize_energy_portfolio",
        "domain": "portfolio",
        "keywords_zh": [
            "全校", "校園", "最佳化", "組合", "投資",
            "預算", "哪幾棟", "優先順序", "ROI",
            "全校節能", "校園節能", "全校預算",
            "先做哪棟", "最佳組合", "資源配置",
        ],
        "keywords_en": [
            "portfolio", "budget", "optimize", "investment",
            "which buildings", "prioritize", "ROI",
            "campus-wide plan", "resource allocation",
        ],
        "examples": [
            "全校預算500萬應該先做哪幾棟",
            "哪些樓的節能投資報酬率最高",
            "全校節能最佳組合",
        ],
    },
    {
        "intent": "strategy_tracking",
        "tool": "check_strategy_status",
        "domain": "tracking",
        "keywords_zh": [
            "策略追蹤", "採用了沒", "有沒有採納", "上週建議",
            "之前建議", "之前的策略", "策略進度", "策略狀態",
            "確認採用", "已採納", "追蹤策略",
        ],
        "keywords_en": [
            "strategy tracking", "adopted", "previous strategy",
            "strategy status", "follow up",
        ],
        "examples": [
            "上週建議的冷房+2°C採用了沒",
            "這棟建築的策略追蹤狀態",
            "之前的策略有沒有被採納",
        ],
    },
    {
        "intent": "actual_vs_predicted",
        "tool": "compare_actual_predicted",
        "domain": "comparison",
        "keywords_zh": [
            "實際省了多少", "比對", "預測準不準",
            "actual", "實際數據", "效果如何",
            "真的省了", "預測vs實際",
        ],
        "keywords_en": [
            "actual vs predicted", "accuracy", "compare actual",
            "how much actually saved",
        ],
        "examples": [
            "實際省了多少電",
            "之前預測的準不準",
            "比對實際和預測的差異",
        ],
    },
    {
        "intent": "openbse_validation",
        "tool": "validate_strategy_openbse",
        "domain": "validation",
        "keywords_zh": [
            "物理模擬驗證", "OpenBSE驗證", "跑模擬",
            "模擬確認", "驗證策略", "物理驗證",
        ],
        "keywords_en": [
            "physics validation", "openbse", "simulate",
            "verify with simulation",
        ],
        "examples": [
            "用OpenBSE跑一下驗證這個策略",
            "物理模擬確認這個方案",
        ],
    },
    {
        "intent": "calibrate",
        "tool": "calibrate_sensitivity",
        "domain": "calibration",
        "keywords_zh": [
            "校準", "修正係數", "誤差回灌",
            "靈敏度校正", "調整參數", "係數修正",
        ],
        "keywords_en": [
            "calibrate", "calibration", "correct coefficients",
            "error feedback", "adjust sensitivity",
        ],
        "examples": [
            "根據實際數據校準預測係數",
            "修正靈敏度參數",
        ],
    },
]


METRIC_RULES: list[dict[str, Any]] = [
    {"metric": "eui", "keywords_zh": ["EUI", "eui", "能源使用強度"], "keywords_en": ["eui", "energy use intensity"]},
    {"metric": "peak_kw", "keywords_zh": ["尖峰", "峰值", "最大需量"], "keywords_en": ["peak", "peak demand"]},
    {"metric": "annual_kwh", "keywords_zh": ["年用電", "耗電", "用電量", "電力消耗", "全年電力", "年度用電", "總用電", "總耗電", "用電總量"], "keywords_en": ["annual", "kwh", "annual kwh", "total electricity", "consumption"]},
    {"metric": "load_factor", "keywords_zh": ["負載率", "負載因子"], "keywords_en": ["load factor", "load_factor"]},
    {"metric": "mean_kw", "keywords_zh": ["平均用電", "平均功率", "平均負載"], "keywords_en": ["mean kw", "average power", "avg kw"]},
]


CHART_TYPE_RULES: list[dict[str, Any]] = [
    {"chart_type": "bar", "keywords_zh": ["長條", "柱狀", "條狀"], "keywords_en": ["bar", "column"]},
    {"chart_type": "compare", "keywords_zh": ["比較圖", "對比圖"], "keywords_en": ["compare"]},
    {"chart_type": "line", "keywords_zh": ["折線", "曲線", "趨勢線"], "keywords_en": ["line"]},
]


YEAR_PATTERN = re.compile(r"(?<!\d)(20[0-3][0-9])(?!\d)")
YEAR_RANGE_PATTERN = re.compile(r"(到|至|~|-|—|–)")


def match_intent(prompt: str) -> IntentRule | None:
    lowered = prompt.lower()
    best_match: IntentRule | None = None
    best_score = 0

    for rule in INTENT_RULES:
        score = 0
        for kw in rule.get("keywords_zh", []):
            if kw in prompt:
                score += 2
        for kw in rule.get("keywords_en", []):
            if kw in lowered:
                score += 1
        if score > best_score:
            best_score = score
            best_match = rule

    return best_match


def match_tool(prompt: str) -> str:
    rule = match_intent(prompt)
    return rule["tool"] if rule else ""


def match_metric(prompt: str) -> str:
    lowered = prompt.lower()
    for rule in METRIC_RULES:
        for kw in rule.get("keywords_zh", []):
            if kw in prompt:
                return rule["metric"]
        for kw in rule.get("keywords_en", []):
            if kw in lowered:
                return rule["metric"]
    return "annual_kwh"


def match_chart_type(prompt: str) -> str:
    lowered = prompt.lower()
    for rule in CHART_TYPE_RULES:
        for kw in rule.get("keywords_zh", []):
            if kw in prompt:
                return rule["chart_type"]
        for kw in rule.get("keywords_en", []):
            if kw in lowered:
                return rule["chart_type"]
    return "line"


def extract_years(prompt: str) -> list[int]:
    years = sorted({int(m) for m in YEAR_PATTERN.findall(prompt)})
    years = [y for y in years if 2010 <= y <= 2035]
    if len(years) >= 2 and YEAR_RANGE_PATTERN.search(prompt):
        start, end = years[0], years[-1]
        if 0 <= end - start <= 25:
            return list(range(start, end + 1))
    return years


def should_prefetch(prompt: str) -> tuple[str, bool]:
    rule = match_intent(prompt)
    if rule is None:
        return "", False
    return rule["tool"], True


def export_concept_map() -> dict[str, Any]:
    concepts: list[dict[str, Any]] = []
    for rule in INTENT_RULES:
        concepts.append({
            "intent": rule["intent"],
            "tool": rule["tool"],
            "domain": rule["domain"],
            "all_keywords": sorted(set(rule.get("keywords_zh", []) + rule.get("keywords_en", []))),
            "examples": rule.get("examples", []),
        })
    return {
        "version": "1.0",
        "description": "NTU campus energy assistant intent-to-tool mapping",
        "concepts": concepts,
        "metric_rules": METRIC_RULES,
        "chart_type_rules": CHART_TYPE_RULES,
    }


def export_concept_map_markdown() -> str:
    lines = [
        "# NTU Campus Energy Assistant — Intent Concept Map",
        "",
        "## Overview",
        "",
        "This document maps user intent keywords to MCP tools.",
        "The local Gemma model and the programmatic router both use this mapping.",
        "",
    ]
    for rule in INTENT_RULES:
        lines.append(f"## {rule['intent']} → `{rule['tool']}`")
        lines.append(f"**Domain:** {rule['domain']}")
        lines.append("")
        lines.append("**Chinese keywords:** " + ", ".join(f"`{kw}`" for kw in rule.get("keywords_zh", [])))
        lines.append("")
        lines.append("**English keywords:** " + ", ".join(f"`{kw}`" for kw in rule.get("keywords_en", [])))
        lines.append("")
        lines.append("**Example queries:**")
        for ex in rule.get("examples", []):
            lines.append(f"- {ex}")
        lines.append("")

    lines.extend([
        "## Metric Keywords",
        "",
    ])
    for rule in METRIC_RULES:
        all_kw = sorted(set(rule.get("keywords_zh", []) + rule.get("keywords_en", [])))
        lines.append(f"- **`{rule['metric']}`**: {', '.join(f'`{kw}`' for kw in all_kw)}")
    lines.append("")

    return "\n".join(lines)
