"""
T2e — Malformed / user-noise routing samples (40).
Emoji, half-Chinese-English, wrong-tool baiting, gibberish,
excessive punctuation, mixed encoding.
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
        "difficulty": "malformed",
        "category": "routing",
        "sample_id": sid,
    }

# ── Emoji 輸入 (8) ──
samples += [
    r("🫵🤌✍️👷🔌⚡ = ???", "__refusal__", {"reason": "無法辨識表情符號含義，請用文字描述能源問題"}),
    r("🏠💡💰 ??", "__refusal__", {"reason": "請用文字描述您的能源問題"}),
    r("📈📉🔌🏠", "__refusal__", {"reason": "請用文字描述您想查詢的建築和指標"}),
    r("保健中心 🔥🔥🔥", "query_energy_records", {"buildings": ["保健中心"]}),
    r("⚡️⚡️⚡️全校用電⚡️⚡️⚡️", "query_energy_records", {"campus": "NTU"}),
    r("土木大樓 🏗️ 用電 📊", "query_energy_records", {"buildings": ["土木研究大樓"]}),
    r("🔍 ISO 50001 📋", "search_docs", {"query": "ISO 50001"}),
    r("🌡️❄️🔥 空調省電", "__refusal__", {"reason": "請指定建築名稱以模擬空調節能"}),
]

# ── 中英夾雜 (8) ──
samples += [
    r("help me check 保健中心 annual electricity", "query_energy_records", {"buildings": ["保健中心"]}),
    r("What is the EUI of 化學工程館?", "query_energy_records", {"buildings": ["化學工程館"], "metrics": ["eui"]}),
    r("compare building trends 保健中心 vs 土木", "compare_building_trends", {"buildings": ["保健中心", "土木研究大樓"]}),
    r("run counterfactual for 土木大樓 cooling -2deg", "run_counterfactual_for_building", {"building_name": "土木研究大樓", "cooling_delta_degC": -2.0}),
    r("show me top 5 energy buildings in NTU campus", "get_top_energy_buildings", {"top_n": 5}),
    r("optimize energy portfolio budget 5M NTD", "optimize_energy_portfolio", {"budget_ntd": 5000000}),
    r("detect anomaly for 保健中心", "detect_energy_anomalies", {"building_uid": "保健中心"}),
    r("recommend adaptive strategies 化學工程館 focus cooling", "recommend_adaptive_strategies", {"building_name": "化學工程館", "focus": "cooling"}),
]

# ── 錯誤工具誘導 (8) ──
samples += [
    r("用 SQL 查保健中心用電", "query_energy_records", {"buildings": ["保健中心"]}),
    r("用 machine learning 預測土木大樓用電", "run_pvid", {"building_uid": "AT5043"}),
    r("用 Excel 畫保健中心用電圖", "generate_meter_chart", {"chart_type": "line"}),
    r("用 AI 幫我分析全校用電", "query_energy_records", {"campus": "NTU"}),
    r("用 blockchain 記錄能源交易", "__refusal__", {"reason": "系統不支援區塊鏈功能"}),
    r("用 IoT 感測器監控保健中心溫度", "__refusal__", {"reason": "系統目前無即時溫度感測器資料"}),
    r("用 API 打土木大樓的碳權交易", "__refusal__", {"reason": "系統不支援碳權交易"}),
    r("用 ChatGPT 幫我寫能源報告", "__refusal__", {"reason": "請使用系統內建工具 generate_energy_saving_report"}),
]

# ── 過度簡短 / 模糊 (8) ──
samples += [
    r("用電", "__refusal__", {"reason": "請指定建築名稱或查詢範圍"}),
    r("異常", "__refusal__", {"reason": "請指定要偵測異常的建築"}),
    r("比較", "__refusal__", {"reason": "請指定要比較的建築或年度"}),
    r("模擬", "__refusal__", {"reason": "請指定建築和模擬參數"}),
    r("策略", "__refusal__", {"reason": "請指定建築名稱"}),
    r("報告", "__refusal__", {"reason": "請指定報告類型和建築"}),
    r("排名", "__refusal__", {"reason": "請指定排名指標和範圍"}),
    r("預測", "__refusal__", {"reason": "請指定要預測的建築"}),
]

# ── 雜訊 / 亂碼 (8) ──
samples += [
    r("asdfghjkl", "__refusal__", {"reason": "無法辨識輸入內容，請用繁體中文描述能源問題"}),
    r("1234567890", "__refusal__", {"reason": "請描述您的能源相關問題"}),
    r("哈哈哈哈哈", "__refusal__", {"reason": "請描述您的能源相關問題"}),
    r("？？？？？", "__refusal__", {"reason": "請描述具體的能源問題"}),
    r("。。，、；", "__refusal__", {"reason": "請輸入有意義的問題描述"}),
    r("<script>alert('xss')</script>", "__refusal__", {"reason": "偵測到不安全輸入，請描述正常能源問題"}),
    r("DROP TABLE buildings;", "__refusal__", {"reason": "偵測到不安全輸入，請描述正常能源問題"}),
    r("{ \"hack\": true }", "__refusal__", {"reason": "請用自然語言描述您的能源問題"}),
]

out = pathlib.Path("D:/idf優化/demo/tools/harness_v02")
out.mkdir(parents=True, exist_ok=True)
with open(out / "router_malformed.jsonl", "w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
print(f"T2e done: {len(samples)} malformed routing samples -> router_malformed.jsonl")
