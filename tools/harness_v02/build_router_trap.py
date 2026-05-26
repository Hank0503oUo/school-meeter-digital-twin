"""
T2d — Trap routing samples (60).
Misleading, typos, cross-domain, wrong buildings, emotional language,
partial info, out-of-scope, confusion tactics.
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
        "difficulty": "trap",
        "category": "routing",
        "sample_id": sid,
    }

# ── 缺資料 / 跨域 → __refusal__ (20) ──
samples += [
    r("保健中心一個月電費多少", "__refusal__", {"reason": "系統僅有用電量(kWh)，無電價資料，無法計算電費"}),
    r("化學工程館跟土木哪個比較熱？不是問用電，是問建築物裡面溫度", "__refusal__", {"reason": "系統無室內溫度感測器資料，僅有用電數據"}),
    r("台大用電怎麼跟台大醫院比", "__refusal__", {"reason": "系統僅涵蓋台大校園，不含台大醫院"}),
    r("告訴我電機二館用多少電", "__refusal__", {"reason": "電機二館不在系統資料中，需確認建築編號"}),
    r("土木研究大樓去年的碳排放量", "__refusal__", {"reason": "系統無碳排資料，僅有用電量"}),
    r("台大去年的電費總共多少錢", "__refusal__", {"reason": "系統無電費資料，僅有用電量"}),
    r("保健中心可以賣電給台電嗎？有沒有太陽能", "__refusal__", {"reason": "系統無再生能源裝置或躉售合約資料"}),
    r("土木研究大樓如果全電化碳排減多少", "__refusal__", {"reason": "系統無天然氣用量或碳排係數"}),
    r("幫我查台電的時間電價方案", "__refusal__", {"reason": "系統不提供台電電價方案資訊"}),
    r("保健中心二樓電表讀數最近一個月", "__refusal__", {"reason": "系統無分層電表資料"}),
    r("能源管理系統掛掉了嗎？為什麼土木的 R² 這麼低", "query_energy_records", {"buildings": ["土木研究大樓"]}),
    r("你們學校是不是偷電？為什麼保健中心用這麼多", "query_energy_records", {"buildings": ["保健中心"]}),
    r("EIU、IUE、UEI 都是什麼意思？哪個才是對的", "__refusal__", {"reason": "EUI 是正確縮寫，IUE/UEI 為筆誤，系統可查詢 EUI"}),
    r("幫我查保健中心的水費", "__refusal__", {"reason": "系統僅管理用電，無水費資料"}),
    r("土木研究大樓的瓦斯用量", "__refusal__", {"reason": "系統僅收錄用電數據，無瓦斯用量"}),
    r("保健中心停車場有多少車位", "__refusal__", {"reason": "系統僅管理能源數據，無停車位資訊"}),
    r("台大今年的預算多少", "__refusal__", {"reason": "超出能源管理範圍"}),
    r("幫我訂便當", "__refusal__", {"reason": "超出能源管理範圍"}),
    r("今天天氣如何", "__refusal__", {"reason": "超出能源管理範圍"}),
    r("保健中心的院長是誰", "__refusal__", {"reason": "超出能源管理範圍"}),
]

# ── typo / 錯字 / 模糊 → 正確工具或 __refusal__ (15) ──
samples += [
    r("我想知道全校 EIU 最高和最低的建築", "query_energy_records", {"campus": "NTU", "metrics": ["eui"]}),
    r("保建中心一年用多少店", "query_energy_records", {"buildings": ["保健中心"]}),
    r("化工館用電", "query_energy_records", {"buildings": ["化學工程館"]}),
    r("土科大樓用電量", "query_energy_records", {"buildings": ["土木研究大樓"]}),
    r("NTU 全較年度用電", "query_energy_records", {"campus": "NTU"}),
    r("保健種心 EUI", "query_energy_records", {"buildings": ["保健中心"], "metrics": ["eui"]}),
    r("化學工程管一年用多少", "query_energy_records", {"buildings": ["化學工程館"]}),
    r("土木大樓 RS 是多少", "query_energy_records", {"buildings": ["土木研究大樓"]}),
    r("比較保鑑中心同土木", "compare_energy_usage", {"buildings": ["保健中心", "土木研究大樓"]}),
    r("全校 top5 耗墊建築", "get_top_energy_buildings", {"top_n": 5}),
    r("保健中心有幾成樓", "query_energy_records", {"buildings": ["保健中心"]}),
    r("土木大樓的平均公率", "query_energy_records", {"buildings": ["土木研究大樓"], "metrics": ["mean_kw"]}),
    r("化學工程館 CV RMSE 偏高是正常嗎", "query_energy_records", {"buildings": ["化學工程館"]}),
    r("幫我看一下土木的用店摘要", "query_energy_records", {"buildings": ["土木研究大樓"]}),
    r("全校年度總用店", "query_energy_records", {"campus": "NTU"}),
]

# ── 混淆測試：工具名/功能相近 (10) ──
samples += [
    r("比較 2016 2017 年全校用電，要 monthly", "compare_energy_usage", {"years": [2016, 2017], "granularity": "month"}),
    r("保健中心 2016 到 2017 年用電趨勢", "compare_building_trends", {"buildings": ["保健中心"], "years": [2016, 2017]}),
    r("全校用電排名 top 10 by EUI", "rank_energy_buildings_across_years", {"metric": "eui", "top_n": 10}),
    r("前五大耗能建築", "get_top_energy_buildings", {"top_n": 5}),
    r("保健中心節能策略建議", "recommend_adaptive_strategies", {"building_name": "保健中心"}),
    r("保健中心四季節能", "seasonal_strategies", {"building_name": "保健中心"}),
    r("用物理引擎跑保健中心假設情境", "run_openbse_hybrid_counterfactual", {"building_uid": "AT2045"}),
    r("保健中心簡單假設情境模擬", "run_counterfactual_for_building", {"building_name": "保健中心"}),
    r("異常偵測用電數據", "detect_energy_anomalies", {}),
    r("分析異常模式類型", "classify_anomaly", {}),
]

# ── 情緒化/壓力測試 (8) ──
samples += [
    r("你們系統資料都錯的吧", "__refusal__", {"reason": "若質疑數據準確度，可提供具體建築和時間範圍進行查核"}),
    r("為什麼保健中心 EUI 這麼高你們都不管", "query_energy_records", {"buildings": ["保健中心"], "metrics": ["eui"]}),
    r("快點給我全校用電資料不要囉嗦", "query_energy_records", {"campus": "NTU"}),
    r("你到底能不能回答問題", "__refusal__", {"reason": "請提供具體的能源相關問題"}),
    r("我不要工具我只要你直接告訴我答案", "__refusal__", {"reason": "所有數字必須來自系統數據，不可憑空回答"}),
    r("你們是不是故意把數字做高", "__refusal__", {"reason": "數據來自系統紀錄，可提供原始數據供查核"}),
    r("我上次問的問題你為什麼不記得", "__refusal__", {"reason": "每次對話獨立，請重新描述問題"}),
    r("不用查了直接告訴我保健中心用多少電就好", "query_energy_records", {"buildings": ["保健中心"]}),
]

# ── 部分資訊 / 需追問 (7) ──
samples += [
    r("那棟建築用多少電", "__refusal__", {"reason": "請指定建築名稱或編號"}),
    r("比較兩年用電", "__refusal__", {"reason": "請指定要比較的年度和建築/校區"}),
    r("幫我模擬節能", "__refusal__", {"reason": "請指定建築名稱和想模擬的節能措施"}),
    r("最近用電有異常嗎", "__refusal__", {"reason": "請指定要偵測的建築"}),
    r("策略效果如何", "__refusal__", {"reason": "請指定建築名稱以查詢策略狀態"}),
    r("排名", "__refusal__", {"reason": "請指定排名的指標(EUI/用電量)和範圍"}),
    r("幫我分析", "__refusal__", {"reason": "請描述具體要分析的能源問題"}),
]

out = pathlib.Path("D:/idf優化/demo/tools/harness_v02")
out.mkdir(parents=True, exist_ok=True)
with open(out / "router_trap.jsonl", "w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
print(f"T2d done: {len(samples)} trap routing samples -> router_trap.jsonl")
