"""
RTEM → NTU-Scale Normalization Specification
=============================================

目的：將 RTEM 工業級感測器數值正規化為「台大校園等級」的數值範圍，
     讓合成訓練資料的數字貼近 Gemma 部署場景。

關鍵原則：
  1. 只正規化「數值尺度」，不綁定任何真實系館名稱
  2. 建築用通用類型代稱（辦公大樓、實驗室、教學館...），不使用真實名稱
  3. 感測器用通用標籤（溫度感測器、功率感測器），不對應真實 point ID
  4. 模型學的是「看到異常波形 → 判斷異常類型 → 給出建議」，
     不是「某棟建築某個感測器的數值是多少」

== 數值尺度正規化對照表 ==

RTEM 原始範圍 (典型值)          NTU 校園等級範圍             單位      場景
───────────────────────────────────────────────────────────────────────────
溫度感測器 (AHU/FCU)
  40~120°F                     18~35°C (溫度)               °C       室內空間
  median ~70°F                  typical ~24°C
  std ~5°F                      std ~2°C

送風溫度 (AHU discharge)
  45~130°F                     10~20°C (送風)               °C       空調出風口
  median ~55°F                  typical ~14°C

冰水溫度 (CHWS)
  38~55°F                      5~15°C                       °C       冰水系統
  median ~44°F                  typical ~8°C

冷卻水溫度 (CT)
  70~95°F                      25~38°C                      °C       冷卻水塔
  median ~82°F                  typical ~30°C

濕度
  20~90%RH                     40~80%RH                     %RH      室內/室外
  median ~55%                   typical ~55%

功率 (METER/SITE)
  0.001~2.0 (RTEM 縮放值)      50~2500 kW                   kW       單棟建築
  median ~0.1                   typical ~200 kW
  注意：RTEM 的功率值已被縮放，需要乘上倍率

風量 (FAN/AHU)
  100~10000 CFM                200~5000 CMH                 CMH      風機
  typical ~2000 CFM             typical ~3500 CMH

閥門開度
  0~100%                       0~100%                       %        冰水閥/熱水閥
  typical ~40%                  typical ~40%

壓力 (DUCT)
  0~5 inH2O                    0~1250 Pa                    Pa       風管靜壓
  typical ~1.2 inH2O            typical ~300 Pa

CO2
  400~1500 ppm                 400~1500 ppm                 ppm      室內空氣品質
  typical ~600                  typical ~600

電表讀值
  累積 kWh                      累積 kWh                     kWh      建築電表
  日用電                        500~60000 kWh/日             kWh/日
  年用電                        200,000~22,000,000 kWh/年    kWh/年

EUI
  —                            50~1200 kWh/m²·yr            kWh/m²   台大範圍
  typical                       ~200

== 正規化函數 ==

def normalize_rtem_to_ntu(
    rtem_value: float,
    rtem_median: float,
    rtem_std: float,
    ntu_typical: float,
    ntu_range: float,
) -> float:
    if rtem_std <= 0:
        return ntu_typical
    z_score = (rtem_value - rtem_median) / rtem_std
    return round(ntu_typical + z_score * ntu_range * 0.3, 2)

範例：
  RTEM AHU溫度 median=70°F, std=5°F
  NTU 對應 typical=24°C, range=10°C (18~28)
  
  rtem_value=85°F → z=(85-70)/5=3.0 → ntu=24+3*10*0.3=33°C (偏熱)
  rtem_value=55°F → z=(55-70)/5=-3.0 → ntu=24-3*10*0.3=15°C (偏冷)

== 通用建築類型代稱 (禁止使用真實系館名稱) ==

代稱            類型      mean_kw範圍      EUI範圍       特徵
──────────────────────────────────────────────────────────────
行政辦公大樓     辦公      200~400         100~200       規律上下班
綜合教學館       教室      100~300         100~250       日間負載為主
理工實驗館       實驗室    200~800         200~600       24h基載+尖峰
生農實驗館       實驗室    100~400         150~400       通風需求高
醫學/獸醫館      實驗室    400~1200        300~800       高耗能設備
圖書館           圖書館    150~350         100~200       照明+空調
學生活動中心     綜合      80~200          80~150        活動時段集中
計算機中心       機房      500~1200        400~1200      24h高基載
宿舍             住宿      50~150          50~100        晚間尖峰
體育館           體育      100~300         80~200        間歇性高負載

== 通用感測器標籤 (禁止使用 RTEM point ID) ==

RTEM point ID → 通用標籤

AHU 相關:
  *dat_av*    → 送風溫度感測器
  *rat*       → 回風溫度感測器
  *oat*/*oa*  → 外氣溫度感測器
  *dat_stp*   → 送風溫度設定值
  *cool*      → 冷卻閥開度
  *heat*      → 加熱閥開度
  *fan*       → 風機狀態
  *damper*    → 風門開度
  *co2*       → CO2 感測器
  *humidity*  → 濕度感測器

CH/CHWS 相關:
  *chws*      → 冰水供水溫度
  *chwr*      → 冰水回水溫度
  *compressor*→ 壓縮機狀態
  *power*/*kw*→ 冰機功率

FCU 相關:
  *temp*      → 室內溫度
  *valve*     → 閥門開度
  *fan*       → 風機狀態

METER 相關:
  *kw*        → 瞬時功率
  *kwh*       → 累積用電量

== 異常模式 → 通用場景描述 (不綁定真實數據) ==

spike:       「某棟建築的[溫度/功率]感測器突然讀到異常極端值，隨後恢復」
drift:       「[感測器類型]讀值在過去數小時內持續偏離基準」
zero:        「[感測器類型]讀值突然歸零」
oscillation: 「[感測器類型]讀值在高值與低值之間快速交替」
step:        「[感測器類型]讀值突然跳到新的基準線」
stuck:       「[感測器類型]讀值長時間完全沒有變化」
noise:       「[感測器類型]讀值的波動幅度異常大」

== 反事實問題模板 (使用通用建築類型) ==

Q: 「一棟{建築類型}的平均用電功率是 {mean_kw} kW，
    如果把空調設定溫度調高 {delta}°C，預估一年可以省多少電？」

Q: 「某{建築類型}的 EUI 是 {eui} kWh/m²·yr，
    這個數值合理嗎？主要耗能來源可能是什麼？」

Q: 「一棟 {建築類型} 考慮進行節能改造，
    照明降至 80%、設備降至 90%、空調+1°C，預估整體節電率？」

== 禁止事項 ==

1. ❌ 不要在訓練資料中出現真實系館名稱（共同教室、總圖書館、博理館...）
2. ❌ 不要把 RTEM point ID 直接放進訓練資料
3. ❌ 不要建立「某建築 EUI=XXX」的記憶（模型會背起來然後幻覺）
4. ❌ 不要讓模型以為它能查到某棟真實建築的即時數據
5. ✅ 要用通用類型 + 合理數值範圍，教的是推理框架不是死背數字

== 輸出 JSONL 格式 ==

{
  "user": "通用場景的中文問題",
  "assistant": "結構化回答（結論 → 依據 → 建議）",
  "metadata": {
    "layer": "L1_anomaly_reasoning | L2_counterfactual | L2_efficiency_diagnosis | L3_cross_diagnosis",
    "pattern": "spike | drift | zero | oscillation | step | stuck | noise | counterfactual | eui | archetype",
    "building_type": "行政辦公大樓 | 綜合教學館 | 理工實驗館 | ...",
    "sensor_type": "送風溫度 | 室內溫度 | 功率 | 冰水溫度 | ...",
    "difficulty": "easy | medium | hard"
  }
}
"""
