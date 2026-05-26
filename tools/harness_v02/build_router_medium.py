"""
T2b — Medium routing samples (60).
Requires disambiguation: campus vs building, electricity vs cost,
multi-year, granularity, metric selection.
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
        "difficulty": "medium",
        "category": "routing",
        "sample_id": sid,
    }

# ── compare_energy_usage: 跨年全校 ──
samples += [
    r("2016 跟 2017 全校總用電差多少", "compare_energy_usage", {"years": [2016, 2017], "scope": "campus"}),
    r("NTU 2015 到 2017 用電量變化", "compare_energy_usage", {"years": [2015, 2016, 2017], "scope": "campus"}),
    r("全校 2016 和 2017 每月用電比較", "compare_energy_usage", {"years": [2016, 2017], "scope": "campus", "granularity": "month"}),
    r("2016 年 1 月和 2017 年 1 月用電差異", "compare_energy_usage", {"years": [2016, 2017], "granularity": "month", "months": [1]}),
    r("保健中心和土木 2017 年用電差多少", "compare_energy_usage", {"buildings": ["保健中心", "土木研究大樓"], "years": [2017]}),
    r("全校 2020 年用電有沒有比 2019 少", "compare_energy_usage", {"years": [2019, 2020]}),
]

# ── compare_building_trends: 單建築多年 ──
samples += [
    r("保健中心 2015 到 2017 用電趨勢", "compare_building_trends", {"buildings": ["保健中心"], "years": [2015, 2016, 2017]}),
    r("土木研究大樓 peak_kw 歷年變化", "compare_building_trends", {"buildings": ["土木研究大樓"], "metric": "peak_kw"}),
    r("化學工程館 EUI 趨勢", "compare_building_trends", {"buildings": ["化學工程館"], "metric": "eui"}),
    r("三棟建築 load_factor 年度比較", "compare_building_trends", {"buildings": ["保健中心", "化學工程館", "土木研究大樓"], "metric": "load_factor"}),
    r("保健中心 2016 2017 年 mean_kw 比較", "compare_building_trends", {"buildings": ["保健中心"], "years": [2016, 2017], "metric": "mean_kw"}),
    r("土木研究大樓 eui 近年變化", "compare_building_trends", {"buildings": ["土木研究大樓"], "metric": "eui"}),
]

# ── rank_energy_buildings_across_years ──
samples += [
    r("2014-2020 年度最高用電建築排名 top 5", "rank_energy_buildings_across_years", {"years": [2014, 2015, 2016, 2017, 2018, 2019, 2020], "top_n": 5}),
    r("全校 EUI 排行榜 top 20", "rank_energy_buildings_across_years", {"metric": "eui", "top_n": 20}),
    r("哪些校區用電最高", "rank_energy_buildings_across_years", {"metric": "annual_kwh", "top_n": 10}),
]

# ── query_energy_records: 需要判斷 scope ──
samples += [
    r("保健中心用電佔全校幾趴", "query_energy_records", {"campus": "NTU", "buildings": ["保健中心"]}),
    r("全校 EUI 最高和最低的建築", "query_energy_records", {"campus": "NTU", "metrics": ["eui"]}),
    r("台大去年的電費總共多少錢", "__refusal__", {"reason": "系統僅有用電量，無電費資料"}),
    r("保健中心二樓電表讀數最近一個月", "query_energy_records", {"buildings": ["保健中心"]}),
    r("化學工程館分層用電狀況", "query_energy_records", {"buildings": ["化學工程館"]}),
    r("保健中心的設備用電占比", "query_energy_records", {"buildings": ["保健中心"]}),
    r("化學工程館主要耗能是空調還是實驗設備", "query_energy_records", {"buildings": ["化學工程館"]}),
    r("我想知道全校 EIU 最高和最低的建築", "query_energy_records", {"campus": "NTU", "metrics": ["eui"]}),
]

# ── search_docs: 法規/標準 ──
samples += [
    r("土木研究大樓的碳排放量", "search_docs", {"query": "建築碳排放計算方法"}),
    r("保健中心要符合 ISO 50001 需要注意什麼", "search_docs", {"query": "ISO 50001 能源審查"}),
    r("台灣建築能源護照是什麼", "search_docs", {"query": "台灣建築能源護照"}),
    r("再生能源躉售制度", "search_docs", {"query": "再生能源躉購合約 FIT"}),
    r("ASHRAE 90.1 對 EUI 的規定", "search_docs", {"query": "ASHRAE 90.1 EUI baseline"}),
    r("綠建築標章等級怎麼分", "search_docs", {"query": "EEWH 綠建築分級"}),
    r("建築技術規則綠建築專章", "search_docs", {"query": "建築技術規則綠建築專章"}),
]

# ── generate_meter_chart ──
samples += [
    r("幫我畫全校用電比較圖", "generate_meter_chart", {"chart_type": "bar"}),
    r("保健中心 2017 用電月趨勢圖", "generate_meter_chart", {"chart_type": "line"}),
    r("三棟建築 EUI 長條圖", "generate_meter_chart", {"chart_type": "bar"}),
]

# ── run_counterfactual_for_building ──
samples += [
    r("化學工程館設備減少 20% 用電會怎樣", "run_counterfactual_for_building", {"building_name": "化學工程館", "equipment_ratio": 0.8}),
    r("土木研究大樓人員減半的用電影響", "run_counterfactual_for_building", {"building_name": "土木研究大樓", "occupancy_ratio": 0.5}),
    r("保健中心空調降 3 度的節能效果", "run_counterfactual_for_building", {"building_name": "保健中心", "cooling_delta_degC": -3.0}),
]

# ── recommend_adaptive_strategies ──
samples += [
    r("保健中心空調節能策略", "recommend_adaptive_strategies", {"building_name": "保健中心", "focus": "cooling"}),
    r("土木研究大樓照明改善建議", "recommend_adaptive_strategies", {"building_name": "土木研究大樓", "focus": "lighting"}),
    r("化學工程館整體節能方案", "recommend_adaptive_strategies", {"building_name": "化學工程館"}),
]

# ── seasonal_strategies ──
samples += [
    r("保健中心夏季空調策略", "seasonal_strategies", {"building_name": "保健中心"}),
    r("土木研究大樓冬天節能方法", "seasonal_strategies", {"building_name": "土木研究大樓"}),
    r("化學工程館四季節能計畫", "seasonal_strategies", {"building_name": "化學工程館"}),
]

# ── optimize_energy_portfolio ──
samples += [
    r("全校哪幾棟最值得投資節能", "optimize_energy_portfolio", {}),
    r("預算 500 萬節能投資最佳組合", "optimize_energy_portfolio", {"budget_ntd": 5000000}),
    r("投資報酬率最高的前三棟建築", "optimize_energy_portfolio", {"max_buildings": 3}),
]

# ── list_rtem_sources / map_energy_semantics ──
samples += [
    r("NTU 有哪些電表資料", "list_rtem_sources", {"campus": "NTU"}),
    r("保健中心的 Haystack 語意標籤", "map_energy_semantics", {"building_uid": "AT2045"}),
    r("RTEM BMS 資料點有哪些", "list_rtem_sources", {}),
]

# ── __refusal__: 模糊/跨域 ──
samples += [
    r("化學工程館跟土木哪個比較熱？不是問用電", "__refusal__", {"reason": "系統無室內溫度感測器資料，僅有用電數據"}),
    r("台大用電怎麼跟台大醫院比", "__refusal__", {"reason": "系統僅涵蓋台大校園，不含台大醫院"}),
    r("電機二館用多少電", "__refusal__", {"reason": "電機二館不在系統資料中，需確認建築編號"}),
    r("全校 2015 到 2017 年用電趨勢", "compare_energy_usage", {"years": [2015, 2016, 2017], "scope": "campus"}),
    r("保健中心 2016 vs 2017 用電比較", "compare_building_trends", {"buildings": ["保健中心"], "years": [2016, 2017]}),
    r("全校 EUI 分佈統計", "rank_energy_buildings_across_years", {"metric": "eui", "top_n": 20}),
    r("土木大樓用電異常偵測", "detect_energy_anomalies", {"building_uid": "土木研究大樓"}),
    r("保健中心歷年用電量變化", "compare_building_trends", {"buildings": ["保健中心"]}),
    r("化學工程館 2016 2017 月度用電", "compare_energy_usage", {"buildings": ["化學工程館"], "years": [2016, 2017], "granularity": "month"}),
]

out = pathlib.Path("D:/idf優化/demo/tools/harness_v02")
out.mkdir(parents=True, exist_ok=True)
with open(out / "router_medium.jsonl", "w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
print(f"T2b done: {len(samples)} medium routing samples -> router_medium.jsonl")
