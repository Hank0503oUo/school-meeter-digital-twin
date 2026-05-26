# v5 Demo Edition 規格

## 1. 場景與定位

- **時間**：2026-05-13 系上學術展示
- **聽眾**：系上 + 受邀企業
- **主軸**：PI-VD 演算法 + 校園數位雙生
- **次軸**：基於反事實推理的節能決策支援

### 定位三句話

```
我們是讀取層 + 推論層 + 建議層，不做控制。
PI-VD 與 OpenBSE 做物理推論，反事實量化潛在節能。
LLM 把推論結果轉成自然語言建議，最終決策權在管理者。
```

### 對應企業常見三問

| 企業問 | 我們答 |
|--------|--------|
| 「你們會控制冷氣嗎？」 | 不會。只讀 BMS，建議由管理者決策。 |
| 「失敗風險？」 | DSS 模式，無自動控制 = 無 actuation 風險。 |
| 「準確度？」 | 反事實量化會附信心區間，不會給沒範圍的單一數字。 |

## 2. v5 凍結規格

```
v04 LoRA weights = FROZEN（不再訓練）
工具表面     = 7 個（從 30 個縮減）
LLM 角色     = 路由 + 自然語言報告封裝
推理引擎     = PI-VD + OpenBSE hybrid counterfactual
輸出格式     = 量化節能建議
部署         = 本地筆電（不上雲）
```

不做的事：

- ❌ v04 LoRA 重訓
- ❌ NCU 新資料訓練
- ❌ GCP / Cloud Run 部署
- ❌ MCP stdio → HTTP 重構
- ❌ 校準回灌（calibrate_sensitivity）流程
- ❌ 策略採納寫入（record_strategy / confirm_strategy_adoption）

## 3. 暴露給 LLM 的工具表（7 個）

| 工具 | 用途 | v04 val acc |
|------|------|---|
| `query_energy_records` | 讀取建築現況、年/月用電 | 60% |
| `list_campus_stats` | 校級概況統計 | 96% |
| `get_top_energy_buildings` | 高耗能建築排名 | 92% |
| `detect_energy_anomalies` | 異常用電掃描 | 76% |
| `run_openbse_hybrid_counterfactual` | **核心反事實推理** | 57% |
| `openbse_hvac_breakdown` | HVAC 系統拆解 | 100% |
| `recommend_adaptive_strategies` | 綜合節能建議封裝 | 82% |

被排除的工具不要寫進 system prompt，LLM 看不到就不會選錯。

### 弱點與應對

- `run_openbse_hybrid_counterfactual` 57% 是 demo 主路徑最弱的 routing。
- 應對：演示版加 UI 按鈕直接觸發，不完全靠 LLM routing。
- 備援：手動按鈕 + 預先跑好的 cache 結果。

## 4. 系統 prompt 骨架

```
你是 NTU 校園能源助理。

【你的角色】
- 提供基於 PI-VD 物理推論的節能建議
- 你是讀取層 + 建議層，不做任何控制動作
- 所有建議都應附量化節能與信心區間
- 最終決策權在管理者，你只提供證據

【可用工具】
1. query_energy_records      - 查建築現況用電
2. list_campus_stats         - 校級概況
3. get_top_energy_buildings  - 高耗能排名
4. detect_energy_anomalies   - 異常掃描
5. run_openbse_hybrid_counterfactual - 反事實推理
6. openbse_hvac_breakdown    - HVAC 拆解
7. recommend_adaptive_strategies - 節能策略

【輸出格式】
建議行動：[建築] [子系統]
現況：[數據]
建議：[具體調整]
預估節能：[kWh] [%] [信心區間]
舒適度/影響：[說明]
```

## 5. 五條黃金 Demo Path

### Path 1：現況掌握（暖場）

```
User: 「博理館近一年用電狀況？」

→ query_energy_records(building="博理館", period="last_12m")
→ Tool 回傳: kWh/月 序列、EUI、同期比較

LLM 輸出:
博理館近 12 個月用電 142,000 kWh（EUI 215）
七月最高 18,200 kWh，二月最低 8,400 kWh
比 NTU 教學區同類建築平均高 12%
```

### Path 2：異常診斷

```
User: 「全校最近哪些建築 EUI 異常偏高？」

→ get_top_energy_buildings(metric="eui_anomaly", limit=5)
→ detect_energy_anomalies(scope="campus")

LLM 輸出:
近 30 天 EUI 異常偏高前 5 棟：
1. 綜合大樓  +28%  原因: 週末照明負載未降
2. 工程一館  +21%  原因: 冷卻塔運轉時數異常
3. ...
建議優先檢視前三棟，可進一步 drill-down。
```

### Path 3：反事實推理（**PI-VD 主秀**）

```
User: 「如果博理館冷氣設定從 24°C 調到 26°C 會省多少？」

→ run_openbse_hybrid_counterfactual(
    building="博理館",
    intervention={"hvac_setpoint_offset": "+2"},
    horizon="next_30d"
  )
→ Tool 回傳:
    baseline_kwh: 12,400
    counterfactual_kwh: 10,180
    delta_kwh: -2,220
    delta_pct: -17.9
    confidence_band: ±15%
    pmv_range: [-0.3, +0.4]   # 仍在 ASHRAE 55 內

LLM 輸出:
建議行動：博理館 HVAC setpoint 調整
現況：當前設定 24°C，月用電 12,400 kWh
建議：調至 26°C
預估節能：2,220 kWh/月（約 17.9%，信心區間 ±15%）
舒適度：PMV 仍在 ASHRAE 55 範圍內（-0.3 ~ +0.4）
```

### Path 4：策略建議

```
User: 「夏季博理館有什麼節能策略？」

→ recommend_adaptive_strategies(building="博理館", season="summer")
→ Tool 回傳: 結構化策略清單（HVAC、照明、外殼、排程）

LLM 輸出:
博理館夏季三項可行節能策略：
1. HVAC setpoint 24°C → 26°C: 月省 ~2,200 kWh (17%)
2. 大廳照明週末降為 30%: 月省 ~400 kWh (3%)
3. 冷卻塔週六凌晨停機 4 小時: 月省 ~150 kWh (1%)
合計預估月省 ~2,750 kWh，年化 ~33,000 kWh
```

### Path 5：跨校級決策支援

```
User: 「全校最有節能潛力的前 5 棟是哪些？預算只夠改造 3 棟，怎麼分配？」

→ get_top_energy_buildings(metric="saving_potential")
→ optimize_energy_portfolio (如果保留此工具) 或 LLM 自行 reasoning

LLM 輸出:
依 PI-VD 反事實估算節能潛力前 5 棟：
1. 綜合大樓  潛力 8,500 kWh/月  改造預算 ~80 萬
2. 工程一館  潛力 6,200 kWh/月  改造預算 ~60 萬
3. 博理館    潛力 2,750 kWh/月  改造預算 ~30 萬
4. ...

預算 200 萬下，建議 1+2+3，年化節能 ~206,000 kWh
ROI 預估 3.8 年（電費 4.5/kWh 估算）
最終投資決策由管理者依其他因素（使用率、改造難度等）綜合判斷。
```

## 6. 三棟 demo 主角建築（待確認）

挑選原則：

- EUI 偏高 + 有改善空間
- BMS 資料完整
- 視覺辨識度高（地圖好看）
- 可在五條 path 中作為主角

候選：

| 建築 | 為何適合 |
|------|----------|
| 博理館 | EUI 中高、HVAC 主導、典型教學樓 |
| 綜合大樓 | EUI 高、多種子系統、適合策略多元化展示 |
| 工程一館 | 冷卻塔特徵明顯，異常診斷視覺好看 |

最終由展示者決定。

## 7. 後端準備清單

### 必做（5/12 完成）

- [ ] System prompt 改為 7 工具版（`src/demo_assistant.py` 或對應 config）
- [ ] 確認 `run_openbse_hybrid_counterfactual` 支援的 intervention 種類
  - HVAC setpoint offset：必要
  - Lighting density / schedule：確認是否支援，若不支援列為已知限制
  - 冷卻塔排程：選用
- [ ] 五條 path 各跑一次完整流程，截圖 + 錄影
- [ ] 投影機尺寸 UI 檢查（字體、地圖縮放、Plotly 高度）
- [ ] Pre-warm 流程：開啟 → 暖機 → 跑過所有按鈕一次，確認沒 lazy load 卡頓

### 建議做

- [ ] Counterfactual baseline vs intervention 曲線疊圖視覺化
- [ ] 五條 path 各備一張結果截圖入投影片（demo 失敗時切過去）
- [ ] 備案影片：3-5 分鐘把關鍵流程錄好

### 不做

- v04 LoRA 重訓
- 任何上線/部署作業
- calibrate_sensitivity 相關修法
- NCU LLM 訓練

## 8. Dry-run Checklist（5/12 晚 + 5/13 早各一次）

啟動

- [ ] `open_demo.cmd` 啟動 dashboard（5006）
- [ ] llama.cpp server health check（8088 上 `/v1/models` 200）
- [ ] MCP backend 啟動成功（無 ImportError / port conflict）

UI

- [ ] 地圖正確顯示 NTU 校區，建築可點擊
- [ ] 三棟主角建築 EUI / 用電圖正確
- [ ] LLM 助理欄位回應正常（< 5 秒首字）

五條 Path 各跑一次

- [ ] Path 1 現況查詢 → 圖表 + 文字摘要
- [ ] Path 2 異常診斷 → 排名 + 原因
- [ ] Path 3 反事實 → baseline/cf 曲線 + 量化節能（**最重要**）
- [ ] Path 4 策略建議 → 結構化清單
- [ ] Path 5 校級決策 → 排名 + 預算建議

失敗計時

- [ ] 任一 path > 30 秒未回應 → 切備案截圖
- [ ] 任一 path 工具錯選 → 切備案截圖
- [ ] Counterfactual 結果不合理（負值、爆增） → 切備案截圖

## 9. 演講當天時程（5/13）

| 時段 | 動作 |
|------|------|
| T-90 min | 到場、接投影機、確認電源 |
| T-60 min | 啟動 dashboard + llama.cpp，pre-warm |
| T-30 min | 跑完五條 path 一次，確認所有流程順暢 |
| T-10 min | 關閉除 dashboard 外其他視窗，麥克風測試 |
| T-0      | 開始 |
| T+5 min  | 介紹 PI-VD 演算法（演算法投影片） |
| T+15 min | 雙生地圖 walkthrough |
| T+25 min | Path 3 反事實主秀 |
| T+35 min | Path 4 + 5 決策支援 |
| T+45 min | 架構圖（讀取層 / 推論層 / 建議層） |
| T+55 min | Q&A |

## 10. 應急方案

### 軟體掛掉

1. **LLM 無回應** → 切手動按鈕觸發 counterfactual，自己解釋結果
2. **MCP 工具報錯** → 切備案截圖 / 影片
3. **Panel UI 凍結** → 重啟 dashboard（< 30 秒），中間用備案影片頂著
4. **llama.cpp crash** → 切到 cloud Gemini fallback（toggle）

### 硬體掛掉

5. **筆電當機** → 備援筆電（如果有）or 投影片直接講
6. **投影機問題** → 用筆電螢幕直接展示（圍著看）
7. **斷網** → 影響 cloud fallback，但本地 llama.cpp 不受影響

### 觀眾問題回應

| 問題 | 標準回應 |
|------|----------|
| 「會自動控制嗎？」 | 不會，read-only。 |
| 「準確度如何？」 | 反事實附信心區間，PI-VD 在 NTU 跨年驗證 RMSE 約 X%。 |
| 「能擴展到其他校區嗎？」 | NCU 已部分接入（指地圖切換）。LLM 多校支援列為下階段。 |
| 「為何不用 ChatGPT 直接做？」 | 本地 llama.cpp + LoRA 微調，符合資料不外流要求。 |
| 「OpenBSE 是什麼？」 | EnergyPlus 變體，物理模型，給反事實一個可解釋基礎。 |

## 11. v5 之後（5/13 後再考慮）

不在本次 scope，但保留 backlog：

- v04.1：calibrate_sensitivity + map_energy_semantics relabel + 重訓
- v05：agent trace eval + trajectory SFT（依 V04_ROUTER_TO_AGENT_TRACE_REVIEW.md）
- NCU LLM 多校支援
- Cloud Run 部署（如需 always-on demo URL）
- 校準回灌閉環（policy 與 liability 確認後再做）
