# DEMO 版本更新日誌 (Changelog)

## [2026-03-21] — 倉庫精簡（外圍批次／研究文件）

已移除僅供離線訓練與研究用的外圍檔案，保留 `panel serve app.py` / `app_workbench.py` / MCP 與 `models/` 推論資產可運行之最小集合：

- 刪除：`run_tow_adaptive.py`、`run_v10_boot.py`、`run_v11_*.py`、`models/run_tow_adaptive_v9_reference.py`、`tools/`、`scripts/`、`ep_runner` / `idf_modifier` / `evaluator` / `precompute_inference`、根目錄資料管線小工具與多份內部 PLAN／Code review 文件。
- 刪除訓練用設定：`config/config_tow_adaptive_v9.yaml`、`config_v10_boot.yaml`、`config_v11_*.yaml`。
- 刪除測試：`tests/test_v11_validation_entrypoints.py`。

---

## [2026-03-03 v2] — 演算法邏輯修復 & 多核效能優化

### 🔧 Bug 修復
- **`counterfactual.py`**: 新增 `physics_pred` / `residual_pred` 參數，支援接收 PI-VD 引擎真實物理/殘差分離結果，取代先前硬編碼的 75/25 隨機分離。全域 `np.random.seed(42)` 改為本地 `np.random.default_rng(42)`，避免副作用。
- **建物排序工具（原 `evaluate_scaler.py`，已於 2026-03 自 repo 移除）**: 曾修正 campus-level 總量與單棟電表 R²/CVRMSE 比較邏輯，改為 EUI 排名一覽表。
- **`predict_building()`**: 欄位名 `building_pred` → `building_rank_index`、`building_physics` → `building_physics_index`，明確標示為排序指標而非 kW 預測。新增 `building_eui_index`（單位面積排序指標）。

### ⚡ 效能優化
- **Panel 伺服器**: 啟用多進程 (`--num-procs=4`) 與多執行緒 (`--num-threads=4`)，大幅提升 Dashboard 併發連線處理與互動回應速度。
- **V10 BootEnsemble**: 使用 `joblib` 並行化 bootstrap 模型預測，充分利用多核 CPU。

### ✅ 月份循環編碼驗證
- 確認 `build_calendar_features` 使用的 `(month-1)/12` 公式與訓練資料集完全一致，無需修正。

---

這份文件記錄了 `demo` 專案中最新版本的核心功能升級與優化內容。

## [2026-03-03] - 最新版本更新核心項目

### 1. 建築詮釋資料縮放器 (Building Metadata Scaler) 整合
- **提高預測精度**：將 `BuildingMetadataScaler` 整合進入 PI-VD 引擎中，讓模型能夠進一步考慮各別建築物的物理特徵與屬性，從而提升耗電預測的準確度。
- **架構擴充**：為此修改了推論引擎 (Inference Engine)、儀表板 (Dashboard) 以及反事實引擎 (Counterfactual Engine)，使它們皆能支援建築詮釋資料的輸入與處理。
- **測試涵蓋率提升**：新增了相關單元測試程式碼（如 `tests/core/test_building_metadata.py`）以確保改動的穩定性與正確性。

### 2. 動態電網拓撲地圖 (Animated Map Topology)
- **電力流向視覺化**：在地圖上實作了動態的電網拓撲視覺效果，能夠直觀地以動態光流展示電力的流動方向與拓撲結構。
- **空間座標自動計算**：系統會自動根據子建築物群的質心，推算各級電網節點（包含：區域、變電站、電盤）的地理座標，並生成從父節點延伸到子節點的供電關聯路徑。
- **動態圖層無縫整合**：借助 `Deck.gl` 的 `TripsLayer` 與 `ScatterplotLayer` 技術實作動畫迴圈，並將這些動態圖層無縫整合入儀表板現有的 PyDeck 互動地圖中。

### 3. 數位孿生地圖與雙向互動體驗升級 (Digital Twin Map & Interaction)
- **底層渲染引擎轉換**：將地圖渲染與繪製技術從 `anymap_ts` 全面轉移至效能與互動性更佳的 `pydeck`，獲得更良好的原生點擊事件支援，實現了地圖與 Panel 儀表板組件的雙向即時互動。
- **直觀用電狀態呈現**：現在地圖上會直接顯示各建築的即時電表數值，並利用視覺化顏色標示用電狀態（以紅色代表高耗能、黃色代表中等、綠色代表低耗能）。
- **下鑽分析 (Drill-down) 互動功能**：賦予使用者點擊地圖上特定建築物的能力。點擊後，儀表板會即時連動更新，展示該目標建築專屬的電表歷史數據與 PI-VD 模擬預測結果。
- **測繪精準度提升**：成功整合了建築使用執照資料（Building Usage License Data），大幅修正並提升了電表數據與實體建築物在地理圖資上對應的準確度，並同步將地圖視野縮放與檢查點限制在台大校園範圍內。
