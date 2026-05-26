"""
T2a — Easy routing samples (60).
All tool names are real MCP tools from tool_schema_v02.py.
Format: Router SFT  user -> {"tool": "...", "arguments": {...}}
"""
import json, pathlib

from tool_schema_v02 import build_router_system_prompt

sp = build_router_system_prompt()

samples: list[dict] = []
sid = 0

def r(user: str, tool: str, args: dict | None = None):
    global sid
    sid += 1
    a = {"tool": tool, "arguments": args or {}}
    return {
        "messages": [
            {"role": "system", "content": sp},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(a, ensure_ascii=False)},
        ],
        "expected_tool": tool,
        "difficulty": "easy",
        "category": "routing",
        "sample_id": sid,
    }

# ── query_energy_records: 單建築查詢 ──
samples += [
    r("保健中心一年用多少電？", "query_energy_records", {"buildings": ["保健中心"]}),
    r("化學工程館的用電量", "query_energy_records", {"buildings": ["化學工程館"]}),
    r("土木研究大樓 年用電", "query_energy_records", {"buildings": ["土木研究大樓"]}),
    r("AT2045 的能耗資料", "query_energy_records", {"buildings": ["AT2045"]}),
    r("AT2007 年度用電統計", "query_energy_records", {"buildings": ["AT2007"]}),
    r("AT5043 的用電數據", "query_energy_records", {"buildings": ["AT5043"]}),
    r("幫我查保健中心的 EUI", "query_energy_records", {"buildings": ["保健中心"], "metrics": ["eui"]}),
    r("土木研究大樓的 R² 是多少", "query_energy_records", {"buildings": ["土木研究大樓"]}),
    r("化學工程館的 CV(RMSE)", "query_energy_records", {"buildings": ["化學工程館"]}),
    r("保健中心有幾層樓？面積多大？", "query_energy_records", {"buildings": ["保健中心"]}),
    r("土木研究大樓平均功率多少 kW", "query_energy_records", {"buildings": ["土木研究大樓"]}),
    r("保健中心是什麼耗能等級", "query_energy_records", {"buildings": ["保健中心"]}),
]

# ── query_energy_records: 全校/多建築 ──
samples += [
    r("台大全校加起來一年用多少電", "query_energy_records", {"campus": "NTU"}),
    r("全校年度用電統計", "query_energy_records", {"campus": "NTU"}),
    r("保健中心、化學工程館、土木研究大樓的用電都列出來", "query_energy_records", {"buildings": ["保健中心", "化學工程館", "土木研究大樓"]}),
    r("幫我比較三棟建築的用電量", "query_energy_records", {"buildings": ["保健中心", "化學工程館", "土木研究大樓"]}),
    r("2016 年 NTU 用電紀錄", "query_energy_records", {"campus": "NTU", "years": [2016]}),
    r("2017 年保健中心的用電", "query_energy_records", {"campus": "NTU", "years": [2017], "buildings": ["保健中心"]}),
]

# ── get_top_energy_buildings ──
samples += [
    r("台大最耗電的建築是哪一棟", "get_top_energy_buildings", {"top_n": 1}),
    r("全校前五大用電建築", "get_top_energy_buildings", {"top_n": 5}),
    r("2017 年用電最高的三棟", "get_top_energy_buildings", {"year": 2017, "top_n": 3}),
    r("EUI 最高的建築", "get_top_energy_buildings", {"metric": "eui", "top_n": 1}),
    r("全校用電 top 10", "get_top_energy_buildings", {"top_n": 10}),
    r("2016 年 EUI 排行前五", "get_top_energy_buildings", {"year": 2016, "metric": "eui", "top_n": 5}),
]

# ── list_campus_stats ──
samples += [
    r("台大有幾棟建築", "list_campus_stats", {}),
    r("全校建築分佈統計", "list_campus_stats", {}),
    r("NTU 建築數量與分類", "list_campus_stats", {}),
]

# ── compare_building_trends ──
samples += [
    r("保健中心 2016 到 2017 用電趨勢", "compare_building_trends", {"buildings": ["保健中心"], "years": [2016, 2017]}),
    r("土木研究大樓歷年用電變化", "compare_building_trends", {"buildings": ["土木研究大樓"]}),
    r("化學工程館用電趨勢圖", "compare_building_trends", {"buildings": ["化學工程館"]}),
    r("保健中心和土木的用電趨勢比較", "compare_building_trends", {"buildings": ["保健中心", "土木研究大樓"]}),
    r("三棟建築 mean_kw 趨勢", "compare_building_trends", {"buildings": ["保健中心", "化學工程館", "土木研究大樓"], "metric": "mean_kw"}),
]

# ── compare_energy_usage ──
samples += [
    r("2016 跟 2017 全校用電差多少", "compare_energy_usage", {"years": [2016, 2017]}),
    r("比較 NTU 2016 vs 2017 總用電", "compare_energy_usage", {"years": [2016, 2017], "scope": "campus"}),
    r("保健中心用電佔全校幾趴", "compare_energy_usage", {"buildings": ["保健中心"], "scope": "campus"}),
    r("全校 2016 年和 2017 年月度用電比較", "compare_energy_usage", {"years": [2016, 2017], "granularity": "month"}),
]

# ── rank_energy_buildings_across_years ──
samples += [
    r("2014 到 2020 用電排名", "rank_energy_buildings_across_years", {"years": [2014, 2015, 2016, 2017, 2018, 2019, 2020]}),
    r("全校 EUI 排名", "rank_energy_buildings_across_years", {"metric": "eui"}),
    r("歷年 top 5 耗能建築", "rank_energy_buildings_across_years", {"top_n": 5}),
]

# ── search_docs ──
samples += [
    r("ISO 50001 能源管理標準是什麼", "search_docs", {"query": "ISO 50001 能源管理"}),
    r("ASHRAE Guideline 14 的 CV(RMSE) 標準", "search_docs", {"query": "ASHRAE Guideline 14 CV RMSE 標準"}),
    r("台灣建築能源效率法規", "search_docs", {"query": "台灣建築能源效率法規"}),
    r("EEWH 綠建築標章", "search_docs", {"query": "EEWH 綠建築標章"}),
    r("HJPLUS 建築知識庫", "search_docs", {"query": "HJPLUS", "building_id": "hjplus-kb"}),
]

# ── list_rtem_sources ──
samples += [
    r("系統有哪些 BMS 資料來源", "list_rtem_sources", {}),
    r("RTEM 感測器清單", "list_rtem_sources", {}),
]

# ── generate_meter_chart ──
samples += [
    r("幫我畫保健中心的用電曲線圖", "generate_meter_chart", {"chart_type": "line"}),
    r("土木研究大樓用電柱狀圖", "generate_meter_chart", {"chart_type": "bar"}),
    r("化學工程館功率趨勢圖", "generate_meter_chart", {"chart_type": "line"}),
]

# ── run_counterfactual_for_building ──
samples += [
    r("保健中心如果調高冷氣溫度 2 度會省多少", "run_counterfactual_for_building", {"building_name": "保健中心", "cooling_delta_degC": 2.0}),
    r("土木研究大樓照明改 LED 節能效果", "run_counterfactual_for_building", {"building_name": "土木研究大樓", "lighting_ratio": 0.6}),
]

# ── recommend_adaptive_strategies ──
samples += [
    r("保健中心有什麼節能建議", "recommend_adaptive_strategies", {"building_name": "保健中心"}),
    r("土木研究大樓節能策略", "recommend_adaptive_strategies", {"building_name": "土木研究大樓"}),
    r("化學工程館平均功率", "query_energy_records", {"buildings": ["化學工程館"]}),
    r("保健中心面積多大", "query_energy_records", {"buildings": ["保健中心"]}),
    r("土木研究大樓有幾層", "query_energy_records", {"buildings": ["土木研究大樓"]}),
    r("全校總用電量", "query_energy_records", {"campus": "NTU"}),
    r("2016 年 NTU 總用電", "query_energy_records", {"campus": "NTU", "years": [2016]}),
    r("全校 top 3 耗能建築 2017", "get_top_energy_buildings", {"year": 2017, "top_n": 3}),
    r("NTU 建築數量統計", "list_campus_stats", {}),
    r("化學工程館節能建議", "recommend_adaptive_strategies", {"building_name": "化學工程館"}),
    r("全校 load factor 排名", "get_top_energy_buildings", {"metric": "load_factor", "top_n": 10}),
    r("土木研究大樓節能策略建議", "recommend_adaptive_strategies", {"building_name": "土木研究大樓"}),
]

out = pathlib.Path("D:/idf優化/demo/tools/harness_v02")
out.mkdir(parents=True, exist_ok=True)
with open(out / "router_easy.jsonl", "w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
print(f"T2a done: {len(samples)} easy routing samples -> router_easy.jsonl")
