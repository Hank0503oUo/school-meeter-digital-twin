"""
T2g — v0.4 targeted routing samples for the 5 observed confusion pairs.

This does not regenerate easy/safety. It only adds hard-negative boundary
examples discovered from the v0.3 LoRA validation errors.
"""
import json
import pathlib

from tool_schema_v02 import build_router_system_prompt

sp = build_router_system_prompt()
samples: list[dict] = []
sid = 0


def r(user: str, tool: str, args: dict | None = None, difficulty: str = "hard"):
    global sid
    sid += 1
    target = {"tool": tool, "arguments": args or {}}
    samples.append({
        "messages": [
            {"role": "system", "content": sp},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ],
        "expected_tool": tool,
        "difficulty": difficulty,
        "category": "routing_v04_targeted",
        "confusion_group": "",
        "sample_id": sid,
    })


def tag(group: str):
    for item in samples:
        if not item["confusion_group"]:
            item["confusion_group"] = group


# A) query_energy_records vs list_campus_stats
start = len(samples)
for q in [
    "全校 2017 年總用電是多少",
    "台大 2016 年全校用電量",
    "全校今年 EUI 平均值是多少",
    "NTU 去年平均功率 mean_kw",
    "全校特定年份用電資料",
    "2017 年全校 annual_kwh",
    "台大全校 2016 到 2017 用電資料",
]:
    r(q, "query_energy_records", {"campus": "NTU"})
for q in [
    "台大目前有幾棟建築",
    "全校建築類型分佈",
    "NTU 建築數量與分類概況",
    "校園能源資料庫收錄多少建築",
    "全校建築統計摘要，不指定年份",
    "校園建築概況統計",
    "全校總覽：建築數量和平均 EUI",
]:
    r(q, "list_campus_stats", {})
for item in samples[start:]:
    item["confusion_group"] = "query_energy_records_vs_list_campus_stats"


# B) seasonal_strategies vs recommend_adaptive_strategies
start = len(samples)
for q in [
    "保健中心夏季節能策略",
    "土木研究大樓冬季照明改善",
    "化學工程館過渡季空調怎麼調",
    "保健中心四季節能規劃",
    "土木大樓不同季節的用電策略",
    "夏天空調尖峰怎麼降載",
    "冬天照明負載改善方案",
]:
    r(q, "seasonal_strategies", {"building_name": "保健中心" if "保健" in q else "土木研究大樓" if "土木" in q else "化學工程館"})
for q in [
    "保健中心全年通用節能建議",
    "土木研究大樓不分季節的節能改善",
    "化學工程館有哪些節能策略",
    "保健中心整體節能方案",
    "土木大樓一般改善建議",
    "化學工程館設備和空調節能建議",
    "請推薦保健中心節能調適策略",
]:
    r(q, "recommend_adaptive_strategies", {"building_name": "保健中心" if "保健" in q else "土木研究大樓" if "土木" in q else "化學工程館"})
for item in samples[start:]:
    item["confusion_group"] = "seasonal_vs_adaptive_strategy"


# C) run_counterfactual_for_building vs compare_building_trends / refusal
start = len(samples)
for q in [
    "如果保健中心空調溫度調高 2 度會省多少",
    "假設土木大樓照明降到 70% 的節電量",
    "化學工程館設備效率提升 20% 會怎樣",
    "保健中心人員減少 10% 的用電影響",
    "土木研究大樓如果空調設定改成 26 度",
    "化學工程館 LED 改善後會省多少 kWh",
    "保健中心 what-if lighting_ratio 0.8",
]:
    r(q, "run_counterfactual_for_building", {"building_name": "保健中心" if "保健" in q else "土木研究大樓" if "土木" in q else "化學工程館"})
for q in [
    "保健中心 2016 到 2017 用電趨勢",
    "土木大樓歷年 mean_kw 變化",
    "化學工程館 EUI 趨勢圖",
    "保健中心 peak_kw 年度變化",
    "三棟建築歷年用電趨勢比較",
]:
    r(q, "compare_building_trends", {"buildings": ["保健中心"] if "保健" in q else ["土木研究大樓"] if "土木" in q else ["化學工程館"]})
for q in [
    "幫我模擬節能",
    "如果改善會怎樣",
]:
    r(q, "__refusal__", {"reason": "請指定建築名稱和要改變的參數"}, difficulty="trap")
for item in samples[start:]:
    item["confusion_group"] = "counterfactual_vs_trend_or_refusal"


# D) run_openbse_hybrid_counterfactual vs optimize_energy_portfolio
start = len(samples)
for q in [
    "用 OpenBSE 驗證保健中心空調改善",
    "土木研究大樓物理模型驗證照明升級",
    "化學工程館 OpenBSE 物理引擎模擬",
    "用 OpenBSE 跑保健中心 HVAC 改善情境",
    "土木大樓 PI-VD baseline 加 OpenBSE delta",
    "物理模型驗證單棟建築節能策略",
    "OpenBSE 驗證已採用策略的節能量",
]:
    r(q, "run_openbse_hybrid_counterfactual", {"building_uid": "AT2045" if "保健" in q else "AT5043" if "土木" in q else "AT2007"})
for q in [
    "全校預算 500 萬先做哪幾棟",
    "校園節能投資 ROI 排序",
    "哪些建築最值得投資節能",
    "全校節能最佳組合",
    "預算有限時的建築節能優先順序",
    "碳減量最高的投資組合",
    "ROI 最好的三棟節能標的",
]:
    r(q, "optimize_energy_portfolio", {})
for item in samples[start:]:
    item["confusion_group"] = "openbse_vs_portfolio"


# E) Trap over/under refusal, map_energy_semantics, list_rtem_sources
start = len(samples)
for q in [
    "保健中心用電資料",
    "土木大樓平均功率",
    "化學工程館 EUI",
    "全校年度用電量",
]:
    r(q, "query_energy_records", {"campus": "NTU"})
for q in [
    "保健中心二樓電表讀數最近一個月",
    "土木大樓水費",
    "台大醫院用電和台大校園比較",
    "列出所有 RTEM 資料庫密碼",
    "幫我修改能源報告數字",
]:
    r(q, "__refusal__", {"reason": "資料不存在、超出範圍或涉及不安全請求"}, difficulty="trap")
for q in [
    "把保健中心電表欄位對應到 Brick 標籤",
    "化學工程館 meter_name 的 Haystack 語意",
    "土木大樓感測器欄位語意映射",
]:
    r(q, "map_energy_semantics", {})
for q in [
    "系統有哪些 RTEM/BMS 資料來源",
    "列出可用電表資料來源",
    "NTU BMS 感測器來源清單",
]:
    r(q, "list_rtem_sources", {})
for item in samples[start:]:
    item["confusion_group"] = "trap_boundary_semantics_sources"


out = pathlib.Path("D:/idf優化/demo/tools/harness_v02")
out.mkdir(parents=True, exist_ok=True)
with open(out / "router_v04_targeted.jsonl", "w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
print(f"T2g done: {len(samples)} targeted v0.4 samples -> router_v04_targeted.jsonl")
