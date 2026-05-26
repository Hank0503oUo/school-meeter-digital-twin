# `src` 模組功能總覽

這份文件整理 `src` 目錄下所有 Python 模組，以及 `src/dashboard_modules` 子資料夾中的模組用途。內容以目前程式碼結構與命名為準；若某模組只是薄封裝、相容層或套件初始化檔，這裡也會直接標明。

## 基礎設定與共用工具

- `__init__.py`：`src` 套件初始化檔，定義這個專案程式碼包的基本身分。
- `constants.py`：集中管理能源相關常數、UI 色彩、基準年與其他全域設定。
- `project_paths.py`：處理資料、模型、設定檔等路徑解析，方便跨環境存取檔案。
- `utils.py`：共用小工具函式，例如型別轉換、加權平均、名稱正規化與字串拆解。
- `campus_config.py`：校區設定載入與管理，通常用來讀取各 campus 的基礎資訊與路徑。
- `building_coord.py`：舊版或相容用的座標相關邏輯，屬於較薄的 legacy 模組。

## 儀表板主流程

- `dashboard.py`：主入口模組，對外提供建立校區儀表板或知識工作台的高階函式。
- `dashboard_state.py`：儀表板執行期狀態容器，保存目前校區、資料快取與 UI 相關狀態。
- `dashboard_data.py`：負責資料重載、延遲載入與年度資料快取管理。
- `dashboard_impl.py`：轉接到模組化 dashboard factory 的薄封裝，保留舊入口相容性。
- `dashboard_charts.py`：建立 Plotly 圖表與視覺樣式，統一時間序列與指標圖的呈現方式。
- `dashboard_map.py`：組裝地圖面板與相關動畫/顯示邏輯。
- `dashboard_noncore.py`：非核心 UI 輔助內容，例如說明文字、KPI 卡片與參考連結區塊。

## `dashboard_modules` 子套件

- `dashboard_modules/__init__.py`：子套件初始化檔，提供模組化 dashboard 的入口與匯出位置。
- `factory.py`：組裝整個 dashboard 的工廠模組，負責把各視圖與 runtime 串起來。
- `runtime.py`：`DashboardRuntime` 的核心執行環境，管理校區資料載入、年度快取與篩選條件。
- `models.py`：儀表板 widgets 與 UI 狀態的資料結構定義。
- `reactive.py`：集中處理 reactive 更新與重新計算的觸發邏輯。
- `map_views.py`：地圖視圖控制器，處理樓棟選擇、年份切換與地圖更新。
- `analysis_views.py`：分析視圖控制器，負責圖表、KPI、counterfactual 與分析結果渲染。
- `assistant_views.py`：助理視圖控制器，處理 LLM 互動、MCP 整合與輔助操作流程。
- `selection.py`：從下拉選單或顯示文字中萃取內部 ID 的小工具。
- `building_alias.py`：樓棟別名、中文名稱與分隔符正規化邏輯。
- `cache.py`：有限容量的快取工具，常用於季節性或重複讀取的資料。

## 建築與電表資料

- `building_inference.py`：把電表資料彙整成樓棟層級摘要，並處理共享或混合電表。
- `meter_classifier.py`：根據名稱與規則判斷電表角色，例如總表、分表、饋線或建築表。
- `meter_matcher.py`：將電表名稱對應到樓棟名稱，通常結合硬編碼規則與模糊比對。

## 推論與情境分析

- `real_inference_engine.py`：主要的 PI-VD 推論引擎，負責多層模型與殘差修正流程。
- `algorithm_mcp_backend.py`：把推論與情境分析能力包成 MCP backend 供外部呼叫。
- `counterfactual.py`：情境分析與敏感度參數，處理冷房、照明、 occupancy、設備等假設變化。

## 法規驅動調適策略

- `regulation_strategy_map.py`：法規→參數對應表，定義各建築類型的 EUI 基準、法規搜尋查詢、BEE 權重，以及 dominant factor → 可調參數範圍與法規依據的對應。
- `adaptive_strategy_engine.py`：核心引擎，串接建築診斷（metadata + archetype + correlate）→ 法規對齊（HJPLUS KB 搜尋）→ 多情境模擬（counterfactual）→ 策略排序輸出。

## 季節動態策略

- `seasonal_strategy_engine.py`：將全年拆分為夏季（6-9月）、冬季（12-2月）、過渡季（3-5月、10-11月），依季節特性推薦不同調適策略（夏季調空調、冬季調照明、過渡季用外氣），各季節獨立模擬省電量。

## 跨建築組合最佳化

- `portfolio_optimizer.py`：全校建築投資組合最佳化，評估所有建築的節電潛力與 ROI，依預算做 knapsack 組合選擇，輸出優先投資清單、總省電量、成本估算與碳減排量。

## 策略追蹤與回饋閉環

- `strategy_tracker.py`：對話式策略追蹤，記錄推薦策略到 wiki、確認採納狀態、查詢策略進度、比對 actual vs predicted 省電量。
- `openbse_strategy_runner.py`：策略確認後自動生成建築專屬 YAML → 跑 OpenBSE baseline + scenario → 回傳 HVAC 逐項拆解 → 結果回寫 wiki。
- `sensitivity_calibration.py`：誤差回灌引擎，根據 actual vs predicted 比對結果，修正 counterfactual 的敏感度係數（cooling/lighting/equipment/occupancy），持久化到 `config/sensitivity_calibration.json`。

## 地圖與地理資料

- `map_builder.py`：對外的地圖建構介面，通常是薄封裝或 re-export。
- `map_builder_impl.py`：實作地圖圖層、樓棟統計與 HTML 輸出的核心邏輯。
- `map_colors.py`：地圖色彩規則與數值對應，例如 EUI、能耗、R²、DCI 等指標。
- `merge_osm_google.py`：合併 OSM 與 Google footprint 資料，優先保留較完整的幾何資訊。
- `osm_fetcher.py`：從 OpenStreetMap/Overpass 取得建築資料與 footprint 幾何。
- `google_fetcher.py`：查詢 Google Places、Geocoding 與相關建物地理資訊。
- `solar_api.py`：Google Solar API 的呼叫與錯誤處理封裝。

## 知識庫與 RAG

- `knowledge_base.py`：知識工作台核心，負責文件切塊、向量索引與相似度搜尋。
- `knowledge_models.py`：知識工作台的資料模型，例如文件、chunk、分析請求與 trace record。
- `knowledge_analysis.py`：雲端優先的分析服務，整合本地與雲端 LLM 的工具呼叫流程。
- `knowledge_tools.py`：對外提供的知識查詢工具，例如搜尋文件、取 chunk、查樓棟與分析資料。
- `knowledge_dashboard.py`：舊式或表單導向的知識工作台 UI。
- `knowledge_startup.py`：啟動前流程與整備邏輯，例如知識審查或額外前置步驟。
- `knowledge_mcp_server.py`：將知識工作台能力包成 MCP server 對外提供工具。
- `knowledge_mcp_backend.py`：knowledge MCP 的後端實作，處理狀態、清單與分析請求。

## LLM 與外部整合

- `demo_assistant.py`：建立助理對話需要的校區快照與上下文資料。
- `lm_studio_client.py`：本地 OpenAI-compatible LLM 客戶端，包含工具呼叫迴圈、schema 轉換、記憶注入與 Gemma/RAG pre-route。
- `local_gemma_runtime.py`：本地 Gemma GGUF runtime 啟動與 health check，透過 `llama-server.exe` 提供 OpenAI-compatible API。
- `rtem_codex_bridge.py`：尋找並串接外部 RTEM / Codex 類服務的橋接層。

### Local Gemma 開發啟動

本地 Gemma 採 CPU-first packaged runtime 設計，學校或公家單位文書機不需要 LM Studio，也不需要 GPU。

第一次整理交付目錄時，先把 `llama-server.exe`、Gemma GGUF 與 optional mmproj 封入 demo：

```powershell
.\scripts\vendor_gemma_runtime.ps1
```

封包後預設路徑如下：

- `runtime\gemma\bin\llama-server.exe`
- `runtime\gemma\models\gemma-4-E2B-it-Q4_K_M.gguf`
- `runtime\gemma\models\mmproj-gemma-4-E2B-it-BF16.gguf`

啟動本地 runtime：

```powershell
.\scripts\start_local_gemma.ps1
```

啟動腳本會優先使用 `runtime\gemma` 內的 packaged files；若尚未封包，才 fallback 到開發機外部路徑。常用環境變數：

- `ENERGY_LLAMA_SERVER_EXE`
- `ENERGY_GEMMA_MODEL_PATH`
- `ENERGY_GEMMA_MMPROJ_PATH`
- `ENERGY_GEMMA_CTX=4096`
- `ENERGY_GEMMA_PORT=8088`
- `ENERGY_LOCAL_LLM_PROVIDER=gemma`
- `ENERGY_LOCAL_LLM_MODEL=gemma-4-E2B-it-Q4_K_M.gguf`

`ENERGY_GEMMA_MMPROJ_PATH` 是 optional vision support；缺少 mmproj 時會以 text-only 模式啟動，不影響文字對話、JSON 解讀、MCP 工具呼叫或記憶/RAG。

啟動後 dashboard 的 `Local Gemma/LLM (本地)` 模式會連到 `http://127.0.0.1:8088/v1`。

## 拓樸與網路視覺化

- `topology.py`：電力拓樸樹結構，描述 ROOT、ZONE、STATION、PANEL、METER 等層級。
- `topology_view.py`：拓樸樹的視覺化元件，通常用於顯示 load 與節點階層。

## 能源助理語意與技能

- `energy_intent_router.py`：能源助理的意圖路由器，根據中英文關鍵字將使用者查詢分派到對應工具（如搜尋文件、比較用電、查詢 EUI、節能調適策略等，共 11 個意圖）。
- `energy_manager_skills.py`：能源管理員技能集，提供用電趨勢分析、異常偵測、峰值統計與報表匯出等功能。
- `energy_semantics.py`：能源資料來源的語意註冊表，定義 CSV、BMS 等資料源的 Haystack 標籤與 Brick class 對應。

## 電表截圖分析

- `meter_screenshot_analysis.py`：電表截圖 OCR 分析，透過 Pillow + Tesseract 辨識讀數、單位與標題，產生結構化量測結果。

## OpenBSE 整合（物理模擬）

- `openbse_building_scaler.py`：將 OpenBSE base YAML 模板依樓地板面積、用電量與樓層數等參數縮放，生成建築專屬模擬輸入檔。
- `openbse_counterfactual.py`：PI-VD + OpenBSE 混合情境分析引擎，以物理模擬提供冷房溫度、照明密度、設備密度等參數的 delta 修正。

## 助理記憶系統

- `wiki_memory.py`：Wiki + 知識圖譜記憶模組，提供 Markdown wiki 頁面管理、wikilink 解析與增量維護的知識圖譜，讓 LLM 透過 MCP 工具持久化對話記憶。

## 其他支援模組

- `trust_policy.py`：電表信任度、覆蓋率與 archetype 判斷政策。
- `epw_reader.py`：輕量級 EPW 氣象檔解析器，用來讀取溫度、濕度等逐時序列。
- `mcp_profile_menu.py`：啟動前的 MCP review profile 選單，常見於 Windows 互動式啟動流程。
- `demo_mcp_server.py`：整合演算法與知識後端的 MCP server，提供單一入口的工具路由；包含 `save_wiki_page`（持久化重要發現到 wiki）與 `recall_wiki_memory`（召回過去記憶）兩個 wiki 記憶工具。
- `nekaise_dashboard.py`：另一套知識工作台 UI，偏向 Plotly 與 Markdown 呈現的工作流。

## 維護提醒

- 新增、刪除或改名 Python 模組時，請同步更新這份文件。
- 如果某個模組只是轉接層、相容層或 package initializer，建議直接在此標註，避免後續維護者誤判其責任。
