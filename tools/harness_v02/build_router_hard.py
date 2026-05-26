"""
T2c — Hard routing samples (80).
Counterfactual, OpenBSE, portfolio, anomaly, regulation, strategy tracking,
calibration, PIVD, algorithm correlation.
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
        "difficulty": "hard",
        "category": "routing",
        "sample_id": sid,
    }

# ── run_counterfactual_for_building (20) ──
samples += [
    r("保健中心換變頻空調一年省幾度", "run_counterfactual_for_building", {"building_name": "保健中心", "cooling_delta_degC": 2.0}),
    r("土木研究大樓照明全換 LED", "run_counterfactual_for_building", {"building_name": "土木研究大樓", "lighting_ratio": 0.5}),
    r("化學工程館設備效率提升 30%", "run_counterfactual_for_building", {"building_name": "化學工程館", "equipment_ratio": 0.7}),
    r("保健中心人員密度降 20%", "run_counterfactual_for_building", {"building_name": "保健中心", "occupancy_ratio": 0.8}),
    r("全校換智慧電表節能效果", "run_counterfactual_for_building", {"building_name": "NTU", "equipment_ratio": 0.85}),
    r("保健中心空調溫度調高 1 度的節電量", "run_counterfactual_for_building", {"building_name": "保健中心", "cooling_delta_degC": 1.0}),
    r("土木研究大樓設備減半用電", "run_counterfactual_for_building", {"building_name": "土木研究大樓", "equipment_ratio": 0.5}),
    r("化學工程館夏季空調降 2 度", "run_counterfactual_for_building", {"building_name": "化學工程館", "cooling_delta_degC": -2.0}),
    r("保健中心建築規模擴增 50% 用電影響", "run_counterfactual_for_building", {"building_name": "保健中心", "building_scaler": 1.5}),
    r("土木大樓全面電動化節能", "run_counterfactual_for_building", {"building_name": "土木研究大樓", "equipment_ratio": 1.2}),
    r("保健中心照明降 40% 空調不變", "run_counterfactual_for_building", {"building_name": "保健中心", "lighting_ratio": 0.6}),
    r("化學工程館人員增加 50% 的用電增量", "run_counterfactual_for_building", {"building_name": "化學工程館", "occupancy_ratio": 1.5}),
    r("土木大樓空調溫度降 1 度照明的組合情境", "run_counterfactual_for_building", {"building_name": "土木研究大樓", "cooling_delta_degC": -1.0, "lighting_ratio": 0.8}),
    r("保健中心 COP 提高 20%", "run_counterfactual_for_building", {"building_name": "保健中心"}),
    r("幫我模擬保健中心節能 15% 的情境", "run_counterfactual_for_building", {"building_name": "保健中心"}),
    r("土木研究大樓照明+空調+設備都改善 10%", "run_counterfactual_for_building", {"building_name": "土木研究大樓", "lighting_ratio": 0.9, "cooling_delta_degC": 1.0, "equipment_ratio": 0.9}),
    r("如果保健中心改為 24 小時全時段運轉", "run_counterfactual_for_building", {"building_name": "保健中心", "occupancy_ratio": 1.3}),
    r("化學工程館夏季 vs 冬季空調用電差異模擬", "run_counterfactual_for_building", {"building_name": "化學工程館"}),
    r("保健中心出一份基準線 vs 改善後的報告", "run_counterfactual_for_building", {"building_name": "保健中心"}),
    r("土木大樓 demand response 可以省多少", "run_counterfactual_for_building", {"building_name": "土木研究大樓"}),
]

# ── run_openbse_hybrid_counterfactual (8) ──
samples += [
    r("用 OpenBSE 模擬保健中心 Envelope Improvement", "run_openbse_hybrid_counterfactual", {"building_uid": "AT2045"}),
    r("OpenBSE 跑土木大樓空調降 2 度的物理模擬", "run_openbse_hybrid_counterfactual", {"building_uid": "AT5043", "cooling_delta_degC": -2.0}),
    r("用物理引擎模擬化學工程館照明減半", "run_openbse_hybrid_counterfactual", {"building_uid": "AT2007", "lighting_ratio": 0.5}),
    r("保健中心 OpenBSE 全參數模擬", "run_openbse_hybrid_counterfactual", {"building_uid": "AT2045", "cooling_delta_degC": 1.0, "lighting_ratio": 0.7, "equipment_ratio": 0.8}),
    r("OpenBSE 模擬土木大樓 COP 改善 20%", "run_openbse_hybrid_counterfactual", {"building_uid": "AT5043", "cop_ratio": 1.2}),
    r("物理模型算保健中心空調+照明聯合效果", "run_openbse_hybrid_counterfactual", {"building_uid": "AT2045", "cooling_delta_degC": 2.0, "lighting_ratio": 0.6}),
    r("OpenBSE 跑化學工程館人員密度 80%", "run_openbse_hybrid_counterfactual", {"building_uid": "AT2007", "occupancy_ratio": 0.8}),
    r("用 OpenBSE 幫我做全校最耗能建築的外殼改善模擬", "run_openbse_hybrid_counterfactual", {"building_uid": "AT2045"}),
]

# ── openbse_hvac_breakdown (5) ──
samples += [
    r("保健中心 HVAC 詳細元件分析", "openbse_hvac_breakdown", {}),
    r("土木大樓冷卻負載和風機耗能明細", "openbse_hvac_breakdown", {"cooling_delta_degC": 1.0}),
    r("OpenBSE 分析化學工程館各區域冷房能耗", "openbse_hvac_breakdown", {}),
    r("保健中心 DX coil 和 fan 能耗拆分", "openbse_hvac_breakdown", {}),
    r("土木大樓 HVAC zone 溫度和太陽熱取得分析", "openbse_hvac_breakdown", {}),
]

# ── detect_energy_anomalies (8) ──
samples += [
    r("保健中心有沒有異常用電時段", "detect_energy_anomalies", {"building_uid": "保健中心"}),
    r("土木研究大樓電表異常偵測", "detect_energy_anomalies", {"building_uid": "土木研究大樓"}),
    r("化學工程館最近一週用電異常", "detect_energy_anomalies", {"building_uid": "化學工程館", "window": 168}),
    r("保健中心尖峰用電異常分析", "detect_energy_anomalies", {"building_uid": "保健中心", "z_threshold": 2.5}),
    r("土木大樓用電 Z-score 偵測", "detect_energy_anomalies", {"building_uid": "土木研究大樓"}),
    r("全校電表異常掃描", "detect_energy_anomalies", {}),
    r("保健中心夜間用電異常", "detect_energy_anomalies", {"building_uid": "保健中心", "window": 48}),
    r("化學工程館離峰時段異常偵測", "detect_energy_anomalies", {"building_uid": "化學工程館"}),
]

# ── classify_anomaly (5) ──
samples += [
    r("這段時間序列是突波還是漂移", "classify_anomaly", {}),
    r("分析這組用電數值的異常模式", "classify_anomaly", {}),
    r("用電資料是 step change 還是 oscillation", "classify_anomaly", {}),
    r("這個感測器讀值是歸零flatline嗎", "classify_anomaly", {}),
    r("判斷這組溫度數據的異常類型", "classify_anomaly", {}),
]

# ── diagnose_energy_anomaly (5) ──
samples += [
    r("保健中心完整的異常診斷報告", "diagnose_energy_anomaly", {"building_uid": "保健中心"}),
    r("土木大樓 IoT 異常綜合分析", "diagnose_energy_anomaly", {"building_uid": "土木研究大樓"}),
    r("化學工程館跨感測器異常診斷", "diagnose_energy_anomaly", {"building_uid": "化學工程館"}),
    r("保健中心電表+溫濕度關聯異常分析", "diagnose_energy_anomaly", {"building_uid": "保健中心"}),
    r("土木大樓用電與環境感測器交叉診斷", "diagnose_energy_anomaly", {"building_uid": "土木研究大樓"}),
]

# ── validate_strategy_openbse (5) ──
samples += [
    r("用 OpenBSE 驗證示例建築A節能策略", "validate_strategy_openbse", {"building_uid": "DEMO_A", "building_name": "示例建築A", "floor_area_m2": 1500, "mean_kw": 13.7}),
    r("物理驗證示例建築B空調改善方案", "validate_strategy_openbse", {"building_uid": "DEMO_B", "building_name": "示例建築B", "floor_area_m2": 2000, "mean_kw": 27.4}),
    r("OpenBSE 驗證示例建築C照明節能", "validate_strategy_openbse", {"building_uid": "DEMO_C", "building_name": "示例建築C", "floor_area_m2": 1000, "mean_kw": 10.3}),
    r("示例建築A策略的物理模擬驗證報告", "validate_strategy_openbse", {"building_uid": "DEMO_A", "building_name": "示例建築A", "floor_area_m2": 1500, "mean_kw": 13.7}),
    r("示例建築B多策略組合 OpenBSE 驗證", "validate_strategy_openbse", {"building_uid": "DEMO_B", "building_name": "示例建築B", "floor_area_m2": 2000, "mean_kw": 27.4}),
]

# ── optimize_energy_portfolio (5) ──
samples += [
    r("學校 2030 減碳 30% 哪幾棟最有潛力", "optimize_energy_portfolio", {}),
    r("保健中心化學土木各投 100 萬哪個回收最快", "optimize_energy_portfolio", {"budget_ntd": 3000000}),
    r("全校節能 ROI 排名", "optimize_energy_portfolio", {}),
    r("預算 1000 萬全校節能最佳組合", "optimize_energy_portfolio", {"budget_ntd": 10000000}),
    r("碳減量最高的前三棟投資標的", "optimize_energy_portfolio", {"max_buildings": 3}),
]

# ── calibrate_sensitivity / get_sensitivity_status (4) ──
samples += [
    r("保健中心預測節能量跟實際差很多，校準一下", "calibrate_sensitivity", {"building_name": "保健中心"}),
    r("更新土木大樓的敏感度係數", "calibrate_sensitivity", {"building_name": "土木研究大樓"}),
    r("目前系統敏感度校準狀態", "get_sensitivity_status", {}),
    r("化學工程館模型校準歷史", "get_sensitivity_status", {}),
]

# ── run_pvid (5) ──
samples += [
    r("用 PI-VD 預測保健中心明天用電", "run_pvid", {"building_uid": "AT2045", "hours": 24}),
    r("土木大樓物理模型 24 小時負載預測", "run_pvid", {"building_uid": "AT5043", "hours": 24}),
    r("化學工程館四層推論預測結果", "run_pvid", {"building_uid": "AT2007"}),
    r("全校明天總用電 PIVD 預測", "run_pvid", {"hours": 24}),
    r("保健中心 48 小時負載預測", "run_pvid", {"building_uid": "AT2045", "hours": 48}),
]

# ── correlate_algorithms (3) ──
samples += [
    r("綜合分析保健中心的 counterfactual 和 PIVD 結果", "correlate_algorithms", {"building_uid": "AT2045"}),
    r("土木大樓哪個因子主導用電", "correlate_algorithms", {"building_uid": "AT5043", "question": "主導因子"}),
    r("多演算法交叉比對化學工程館", "correlate_algorithms", {"building_uid": "AT2007"}),
]

# ── strategy tracking: record/confirm/check/compare (7) ──
samples += [
    r("記錄保健中心空調改善策略", "record_strategy", {"building_name": "保健中心", "strategy_label": "空調改善"}),
    r("確認土木大樓照明升級已採用", "confirm_strategy_adoption", {"building_name": "土木研究大樓", "strategy_label": "照明升級"}),
    r("查保健中心的策略採用狀態", "check_strategy_status", {"building_name": "保健中心"}),
    r("比較土木大樓實際 vs 預測節能量", "compare_actual_predicted", {"building_name": "土木研究大樓"}),
    r("化學工程館策略執行進度", "check_strategy_status", {"building_name": "化學工程館"}),
    r("記錄保健中心 LED 照明策略", "record_strategy", {"building_name": "保健中心", "strategy_label": "LED照明"}),
    r("全校策略追蹤總覽", "check_strategy_status", {"building_name": "NTU"}),
]

out = pathlib.Path("D:/idf優化/demo/tools/harness_v02")
out.mkdir(parents=True, exist_ok=True)
with open(out / "router_hard.jsonl", "w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
print(f"T2c done: {len(samples)} hard routing samples -> router_hard.jsonl")
