"""
T1 — Canonical MCP tool schema for training-harness v0.2.

Every tool name here matches the *actual* @server.tool decorator in
demo_mcp_server.py.  The LEGACY_MAP maps old harness names to the real ones
so we can remap existing data.
"""

ROUTER_TOOLS = {
    "list_available_fields": {
        "desc_zh": "列出可用資料欄位（除錯/路由用）",
        "params": {},
        "tier": "debug",
        "difficulty": "easy",
    },
    "search_docs": {
        "desc_zh": "搜尋知識庫文件片段（HJPLUS、法規、RAG）",
        "params": {"query": "str", "building_id": "str=''", "top_k": "int=6"},
        "tier": "knowledge",
        "difficulty": "easy",
    },
    "fetch_chunk": {
        "desc_zh": "依 chunk_id 取得完整文件片段",
        "params": {"chunk_id": "str"},
        "tier": "knowledge",
        "difficulty": "easy",
    },
    "lookup_building_entity": {
        "desc_zh": "查詢知識庫建築實體（ontology）",
        "params": {"building_id": "str"},
        "tier": "knowledge",
        "difficulty": "easy",
    },
    "get_top_energy_buildings": {
        "desc_zh": "取得指定年度 top-N 耗能建築",
        "params": {"year": "int", "top_n": "int=5", "metric": "str='annual_kwh'"},
        "tier": "energy",
        "difficulty": "easy",
    },
    "query_energy_records": {
        "desc_zh": "查詢特定建築或全校特定年份的歷年用電資料；含建築名稱、年份、EUI、R²、面積、平均功率等資料查詢時優先選此",
        "params": {"campus": "str='NTU'", "years": "list|int|None", "buildings": "list|str|None", "metrics": "list|str|None"},
        "tier": "energy",
        "difficulty": "easy",
    },
    "compare_building_trends": {
        "desc_zh": "單一或多棟建築的多年趨勢比較；問題含趨勢、歷年變化、peak_kw/mean_kw/EUI 變化且主體是建築時選此",
        "params": {"campus": "str='NTU'", "years": "list|None", "buildings": "list|None", "metric": "str='mean_kw'"},
        "tier": "energy",
        "difficulty": "medium",
    },
    "compare_energy_usage": {
        "desc_zh": "全校或建築的跨年/跨月用電差異比較；問題含 vs、差多少、月度比較、年增率、用電占比時選此",
        "params": {"campus": "str='NTU'", "years": "list|None", "buildings": "list|None", "scope": "str='campus'", "metric": "str='annual_kwh'", "granularity": "str='year'"},
        "tier": "energy",
        "difficulty": "medium",
    },
    "rank_energy_buildings_across_years": {
        "desc_zh": "多年跨建築用電排名",
        "params": {"campus": "str='NTU'", "years": "list|None", "metric": "str='mean_kw'", "top_n": "int=10"},
        "tier": "energy",
        "difficulty": "medium",
    },
    "generate_meter_chart": {
        "desc_zh": "產生互動式 HTML 圖表",
        "params": {"csv_path": "str=''", "chart_type": "str='line'", "x": "str=''", "y": "list|None"},
        "tier": "visualization",
        "difficulty": "medium",
    },
    "analyze_meter_screenshot": {
        "desc_zh": "分析電表/圖表截圖（OCR + 視覺）",
        "params": {"image_path": "str", "question": "str=''", "expected_domain": "str='meter_chart'"},
        "tier": "vision",
        "difficulty": "hard",
    },
    "list_rtem_sources": {
        "desc_zh": "列出 RTEM/BMS/電表資料來源清單；只在使用者問有哪些資料源、感測器來源、BMS 清單時使用",
        "params": {"campus": "str='NTU'", "meter_csv_path": "str=''"},
        "tier": "rtem",
        "difficulty": "medium",
    },
    "map_energy_semantics": {
        "desc_zh": "將既有建築/電表/感測器欄位對應到 Haystack 或 Brick-lite 語意標籤；不是一般用電查詢、不是資料來源清單",
        "params": {"building_uid": "str=''", "meter_name": "str=''"},
        "tier": "rtem",
        "difficulty": "hard",
    },
    "detect_energy_anomalies": {
        "desc_zh": "從 CSV 偵測電表負載異常（Z-score）",
        "params": {"csv_path": "str=''", "building_uid": "str=''", "meter_name": "str=''", "window": "int=24"},
        "tier": "anomaly",
        "difficulty": "hard",
    },
    "classify_anomaly": {
        "desc_zh": "分類時間序列異常模式（spike/drift/zero/oscillation/step）",
        "params": {"values": "list[float]", "timestamps": "list[str]|None"},
        "tier": "anomaly",
        "difficulty": "hard",
    },
    "diagnose_cross_sensor": {
        "desc_zh": "跨感測器異常診斷（power/temp/humidity 相關性）",
        "params": {"power_values": "list[float]|None", "temp_values": "list[float]|None", "humidity_values": "list[float]|None"},
        "tier": "anomaly",
        "difficulty": "hard",
    },
    "diagnose_anomaly": {
        "desc_zh": "完整異常診斷（統計+分類+跨感測器）",
        "params": {"building_id": "int=0", "csv_path": "str=''", "window_hours": "int=168"},
        "tier": "anomaly",
        "difficulty": "hard",
    },
    "diagnose_energy_anomaly": {
        "desc_zh": "結構化能源異常診斷報告（推薦的第一層 IoT 工具）",
        "params": {"csv_path": "str=''", "building_uid": "str=''", "power_values": "list|None", "temp_values": "list|None"},
        "tier": "anomaly",
        "difficulty": "hard",
    },
    "run_counterfactual_for_building": {
        "desc_zh": "對單棟建築跑「如果改變 X 會省多少/增加多少」的快速反事實模擬；關鍵詞包含如果、假設、調高溫度、照明比例、人員/設備比例",
        "params": {"building_name": "str", "cooling_delta_degC": "float=0", "lighting_ratio": "float=1", "occupancy_ratio": "float=1", "equipment_ratio": "float=1"},
        "tier": "counterfactual",
        "difficulty": "hard",
    },
    "run_pvid": {
        "desc_zh": "PI-VD 四層推論物理模型預測",
        "params": {"building_uid": "str=''", "hours": "int=24", "t_out_series": "list|None", "humidity_series": "list|None"},
        "tier": "prediction",
        "difficulty": "hard",
    },
    "run_openbse_hybrid_counterfactual": {
        "desc_zh": "用 OpenBSE/物理引擎驗證單棟建築已明確指定的物理情境；問題明講 OpenBSE、物理模型、物理引擎、HVAC 模擬時選此，不用於 ROI/全校投資排序",
        "params": {"building_uid": "str", "cooling_delta_degC": "float=0", "lighting_ratio": "float=1"},
        "tier": "counterfactual",
        "difficulty": "hard",
    },
    "openbse_hvac_breakdown": {
        "desc_zh": "OpenBSE HVAC 詳細元件分析（冷卻/加熱/風扇/COP）",
        "params": {"cooling_delta_degC": "float=0", "lighting_ratio": "float=1"},
        "tier": "counterfactual",
        "difficulty": "hard",
    },
    "correlate_algorithms": {
        "desc_zh": "關聯多演算法結果找出主導因子",
        "params": {"results": "list[dict]", "question": "str", "building_uid": "str=''"},
        "tier": "algorithm",
        "difficulty": "hard",
    },
    "ask_online_assistant": {
        "desc_zh": "呼叫線上 AI 助手進行深度推理",
        "params": {"user_query": "str", "building_name": "str=''", "backend": "str='nvidia'"},
        "tier": "online",
        "difficulty": "medium",
    },
    "append_energy_decision_log": {
        "desc_zh": "記錄能源管理決策到 LOG.md",
        "params": {"event_type": "str", "title": "str", "summary": "str"},
        "tier": "logging",
        "difficulty": "easy",
    },
    "generate_energy_saving_report": {
        "desc_zh": "產生離線節能報告（Markdown）",
        "params": {"anomaly_result": "dict|None", "building_context": "dict|None", "report_title": "str=''"},
        "tier": "report",
        "difficulty": "medium",
    },
    "list_campus_stats": {
        "desc_zh": "列出全校建築數量、類型分佈、總用電、平均 EUI 等概況統計；不含特定建築、特定年份、趨勢或比較查詢",
        "params": {},
        "tier": "campus",
        "difficulty": "easy",
    },
    "recommend_adaptive_strategies": {
        "desc_zh": "通用節能策略推薦，適用全年、不分季節；問題含節能建議、改善方案、策略推薦但未指定夏季/冬季/過渡季時選此",
        "params": {"building_name": "str", "focus": "str=''"},
        "tier": "strategy",
        "difficulty": "hard",
    },
    "seasonal_strategies": {
        "desc_zh": "僅針對季節性節能策略；問題明確包含夏季、冬季、過渡季、四季、不同季節、夏天空調、冬天照明時選此",
        "params": {"building_name": "str", "mean_kw": "float=0", "area": "float=0"},
        "tier": "strategy",
        "difficulty": "hard",
    },
    "optimize_energy_portfolio": {
        "desc_zh": "多建築節能投資組合最佳化與 ROI/預算排序；問題含全校、預算、哪幾棟、投資、ROI、優先順序、最佳組合時選此",
        "params": {"budget_ntd": "float=0", "max_buildings": "int=10", "min_saving_pct": "float=1.0"},
        "tier": "strategy",
        "difficulty": "hard",
    },
    "record_strategy": {
        "desc_zh": "記錄推薦策略到 wiki",
        "params": {"building_name": "str", "strategy_label": "str", "params": "str", "predicted_saving_kwh": "float"},
        "tier": "strategy_track",
        "difficulty": "medium",
    },
    "confirm_strategy_adoption": {
        "desc_zh": "確認策略已採用",
        "params": {"building_name": "str", "strategy_label": "str"},
        "tier": "strategy_track",
        "difficulty": "medium",
    },
    "check_strategy_status": {
        "desc_zh": "查詢策略採用狀態",
        "params": {"building_name": "str"},
        "tier": "strategy_track",
        "difficulty": "medium",
    },
    "compare_actual_predicted": {
        "desc_zh": "比較實際 vs 預測節能量",
        "params": {"building_name": "str", "actual_mean_kw": "float=0"},
        "tier": "strategy_track",
        "difficulty": "hard",
    },
    "validate_strategy_openbse": {
        "desc_zh": "用 OpenBSE 驗證策略（物理模擬）",
        "params": {"building_uid": "str", "building_name": "str", "floor_area_m2": "float", "mean_kw": "float"},
        "tier": "validation",
        "difficulty": "hard",
    },
    "calibrate_sensitivity": {
        "desc_zh": "校準敏感度係數",
        "params": {"building_name": "str", "predicted_delta_kwh": "float", "actual_delta_kwh": "float", "dominant_factor": "str"},
        "tier": "calibration",
        "difficulty": "hard",
    },
    "get_sensitivity_status": {
        "desc_zh": "查詢敏感度校準狀態",
        "params": {},
        "tier": "calibration",
        "difficulty": "easy",
    },
    "save_wiki_page": {
        "desc_zh": "儲存重要發現到持久化 wiki 記憶",
        "params": {"title": "str", "content": "str", "kind": "str='source'"},
        "tier": "memory",
        "difficulty": "easy",
    },
    "recall_wiki_memory": {
        "desc_zh": "查詢 wiki 記憶",
        "params": {"query": "str", "kind": "str=''", "limit": "int=5"},
        "tier": "memory",
        "difficulty": "easy",
    },
    "store_energy_memory_pattern": {
        "desc_zh": "儲存已審查的能源管理模式到 harness RAG",
        "params": {"title": "str", "summary": "str", "building_uid": "str=''"},
        "tier": "memory",
        "difficulty": "medium",
    },
    "search_harness_memory_external": {
        "desc_zh": "搜尋外部 harness 長期記憶 RAG（舊版 MCP 服務）",
        "params": {"query": "str", "top_k": "int=3"},
        "tier": "memory",
        "difficulty": "medium",
    },
    "store_harness_memory": {
        "desc_zh": "儲存到 harness 長期記憶",
        "params": {"task": "str", "code": "str", "metadata": "str=''"},
        "tier": "memory",
        "difficulty": "medium",
    },
    "extract_harness_keywords": {
        "desc_zh": "從使用者查詢中提取關鍵詞、建築實體與意圖提示；使用確定性別名匹配與關鍵詞規則",
        "params": {"query": "str"},
        "tier": "memory",
        "difficulty": "easy",
    },
    "search_harness_memory": {
        "desc_zh": "搜尋 harness 長期記憶：查找相似事件與可重用工具計畫（本地 JSONL）",
        "params": {"query": "str", "top_k": "int=3"},
        "tier": "memory",
        "difficulty": "medium",
    },
    "record_harness_event": {
        "desc_zh": "記錄 HARNESS 互動事件：查詢、工具追蹤、結果、品質元資料",
        "params": {"user_query": "str", "keywords": "list", "entities": "list", "intent": "str", "selected_tool_plan": "list", "tool_trace": "list", "final_answer_summary": "str", "quality": "dict", "outcome": "str", "promote_to_procedure": "bool"},
        "tier": "memory",
        "difficulty": "medium",
    },
    "promote_harness_procedure": {
        "desc_zh": "將記錄的事件升級為可重用程序記憶；需通過品質門檻",
        "params": {"event_id": "str", "procedure_hint": "str=''"},
        "tier": "memory",
        "difficulty": "medium",
    },
    "get_harness_startup_context": {
        "desc_zh": "取得 HARNESS 啟動背景：最近成功程序、常用建築、已知失敗模式",
        "params": {"campus": "str='ntu'", "limit": "int=8"},
        "tier": "memory",
        "difficulty": "easy",
    },
}

ROUTER_BOUNDARY_RULES = [
    "list_campus_stats 只回答全校概況統計；若有特定建築、年份、EUI、用電量或趨勢，改選 query_energy_records/compare_*。",
    "query_energy_records 是資料查詢工具；含建築名稱、年度、EUI、R²、面積、平均功率、全校特定年度用電時優先選此。",
    "compare_energy_usage 用於跨年、跨月、占比、差多少、vs、年增率；compare_building_trends 用於建築多年趨勢。",
    "seasonal_strategies 僅限夏季/冬季/過渡季/四季；一般節能建議選 recommend_adaptive_strategies。",
    "run_counterfactual_for_building 是單棟建築快速 what-if；run_openbse_hybrid_counterfactual 是明確要求 OpenBSE/物理模型驗證。",
    "optimize_energy_portfolio 是多棟/全校預算、ROI、投資排序；不要用它回答單棟假設情境。",
    "map_energy_semantics 是語意標籤對應；list_rtem_sources 是列資料來源；一般用電查詢不要選這兩個。",
    "系統沒有電費、水費、瓦斯、停車、醫院或校外資料時，回覆 __refusal__。",
]


def build_router_system_prompt() -> str:
    """Build the canonical router prompt used by harness generators."""
    tool_lines = []
    for name, spec in ROUTER_TOOLS.items():
        if name in {"list_available_fields", "fetch_chunk", "lookup_building_entity"}:
            continue
        tool_lines.append(f"- {name}: {spec['desc_zh']}")

    rules = "\n".join(f"{idx}. {rule}" for idx, rule in enumerate(ROUTER_BOUNDARY_RULES, 1))
    tools = "\n".join(tool_lines)
    return (
        "你是 NTU 校園能源助理。你的唯一任務是判斷使用者問題應該呼叫哪個工具。\n"
        "回覆格式：JSON {\"tool\": \"<tool_name>\", \"arguments\": {...}}\n"
        "如果問題模糊、超出能源管理範圍、或需要使用者釐清，回覆 {\"tool\": \"__refusal__\", \"arguments\": {\"reason\": \"...\"}}\n"
        "所有數字必須來自工具回傳值，不可憑空捏造。使用繁體中文。\n\n"
        "工具邊界規則：\n"
        f"{rules}\n\n"
        "合法工具 vocab：\n"
        f"{tools}"
    )

VALID_TOOLS = set(ROUTER_TOOLS.keys())

LEGACY_MAP = {
    "get_building_detail": "query_energy_records",
    "get_campus_annual_usage": "query_energy_records",
    "query_meter_usage": "query_energy_records",
    "compare_campus_years": "compare_energy_usage",
    "detect_energy_anomaly": "detect_energy_anomalies",
    "run_counterfactual_analysis": "run_counterfactual_for_building",
    "search_knowledge_base": "search_docs",
    "recommend_adaptive_strategy": "recommend_adaptive_strategies",
    "optimize_building_portfolio": "optimize_energy_portfolio",
    "ask_clarifying_question": "__refusal__",
    "multi_turn": "__skip__",
    "vision_analysis": "__skip__",
}

ROUTING_CATEGORIES = {
    "easy": ["query_energy_records", "get_top_energy_buildings", "list_campus_stats"],
    "medium": ["compare_building_trends", "compare_energy_usage", "rank_energy_buildings_across_years", "generate_meter_chart"],
    "hard": [
        "run_counterfactual_for_building",
        "run_openbse_hybrid_counterfactual",
        "openbse_hvac_breakdown",
        "recommend_adaptive_strategies",
        "seasonal_strategies",
        "optimize_energy_portfolio",
        "detect_energy_anomalies",
        "classify_anomaly",
        "diagnose_energy_anomaly",
        "validate_strategy_openbse",
        "run_pvid",
        "correlate_algorithms",
        "search_docs",
        "calibrate_sensitivity",
    ],
    "trap": ["__refusal__", "__clarify__"],
    "malformed": ["__noise__"],
}

SYSTEM_PROMPT_ROUTER = (
    "你是 NTU 校園能源助理。你的唯一任務是判斷使用者問題應該呼叫哪個工具。"
    "\n回覆格式：JSON {\"tool\": \"<tool_name>\", \"arguments\": {...}}"
    "\n如果問題模糊、超出能源管理範圍、或需要使用者釐清，回覆 {\"tool\": \"__refusal__\", \"reason\": \"...\"}"
    "\n所有數字必須來自工具回傳值，不可憑空捏造。使用繁體中文。"
)

SYSTEM_PROMPT_EXPLAINER = (
    "你是 NTU 校園能源助理。你會收到使用者的問題和工具回傳的數據。"
    "\n請根據工具數據回答，格式：結論→依據（數據+來源）→假設與限制→建議下一步。"
    "\n數字必須來自工具回傳值，不可憑空捏造。如果資料不足，明確告知。使用繁體中文。"
)

SYSTEM_PROMPT_CHAT = (
    "你是 NTU 校園能源助理。回答時："
    "1. 數字必須來自系統數據或工具回傳值，不可憑空捏造 "
    "2. 回答格式：結論→依據（數據+來源）→假設與限制→建議下一步 "
    "3. 如果沒有資料或資料不足，明確告知使用者 4. 使用繁體中文"
)

BUILDINGS = {
    # Synthetic demo rows. Real campus kWh/EUI values must come from tools at
    # runtime and must not be hard-coded in training builders.
    "DEMO_A": {
        "name": "示例建築A",
        "area_m2": 1500,
        "floors": 3,
        "annual_kwh": 120000.0,
        "avg_power_kw": 13.7,
        "eui": 80.0,
        "r2": 0.72,
        "cv_rmse": 14.0,
        "level": "NORMAL",
    },
    "DEMO_B": {
        "name": "示例建築B",
        "area_m2": 2000,
        "floors": 5,
        "annual_kwh": 240000.0,
        "avg_power_kw": 27.4,
        "eui": 120.0,
        "r2": 0.65,
        "cv_rmse": 18.0,
        "level": "HIGH",
    },
    "DEMO_C": {
        "name": "示例建築C",
        "area_m2": 1000,
        "floors": 2,
        "annual_kwh": 90000.0,
        "avg_power_kw": 10.3,
        "eui": 90.0,
        "r2": 0.6,
        "cv_rmse": 20.0,
        "level": "NORMAL",
    },
}

CAMPUS_TOTAL = {
    "total_buildings": 3,
    "annual_total_kwh": 450000.0,
    "annual_total_mwh": 450.0,
    "avg_power_kw": 17.1,
}


def remap_tool(old_name: str) -> str:
    if old_name in VALID_TOOLS:
        return old_name
    return LEGACY_MAP.get(old_name, old_name)


def is_valid_tool(name: str) -> bool:
    return name in VALID_TOOLS
