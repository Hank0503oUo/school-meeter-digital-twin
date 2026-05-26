# Dashboard 重構計畫與實作紀錄

## 目標
- 依你要求把功能從主幹檔案拆到新資料夾，以模組串聯主幹。
- 保留現有 Dashboard 主要功能（地圖、分析、對話、知識圖譜、總表）。
- 移除使用性不佳的拓樸視覺化路徑，降低維護成本與錯誤風險。

## 新增模組資料夾
已建立：`src/dashboard_modules/`

### 新增檔案
1. `src/dashboard_modules/__init__.py`
- 模組封裝標記。

2. `src/dashboard_modules/cache.py`
- `YEARLY_CACHE_MAX_ENTRIES = 4`
- `bounded_cache_get(...)`
  - 統一年度資料快取取用與 LRU 式淘汰。

3. `src/dashboard_modules/reactive.py`
- `trigger_dashboard_recompute(...)`
  - 集中處理 year/meter/building/cold_start 的重算觸發。

## 主幹檔案調整
調整檔案：`src/dashboard_impl.py`

### A. 改由模組串聯主幹
1. 新增 import
- `from src.dashboard_modules.cache import bounded_cache_get`
- `from src.dashboard_modules.reactive import trigger_dashboard_recompute`

2. 移除主幹內重複 helper（已外移）
- `_bounded_cache_get(...)`
- `_trigger_dashboard_recompute(...)`

3. 全部呼叫點改為新模組函式
- 年度推論/統計/GeoJSON 快取載入改走 `bounded_cache_get(...)`
- 校區初始化與切換後重算改走 `trigger_dashboard_recompute(...)`

### B. 拓樸功能移除（依你的最新要求）
1. UI 分頁層
- 主分頁不再包含拓樸分頁。

2. 互動資訊層
- `meter_info_panel(...)` 的 `[Agg] UID:` 分支不再依賴拓樸物件。
- 改為回傳提示訊息：「拓樸 Drill-down 已停用」。

3. 記錄訊息清理
- 將舊的 topology 字樣 log 改為 `Building coord mapping`，避免誤解目前仍有拓樸視覺化。

## 你可以看到的結果
1. Dashboard 主幹不再自己塞所有共用邏輯，已改由 `src/dashboard_modules/` 串聯。
2. 拓樸相關使用者入口已移除/停用，不再出現看不懂的 Drill-down 視覺化。

## 目前重構完成度（本輪）
- [x] 建立新資料夾並抽出共用模組檔
- [x] 主幹改為模組串聯
- [x] 拓樸功能在 Dashboard 主流程停用
- [ ] 進一步拆出 campus loader / selectors / map interaction（下一輪）

## 下一輪建議拆分
1. `src/dashboard_modules/campus_loader.py`
- `_reload_campus_state(...)`、alias/座標映射邏輯。

2. `src/dashboard_modules/selectors.py`
- 年份/建物/電表選單建構與同步。

3. `src/dashboard_modules/map_interaction.py`
- `on_map_click(...)`、focus 邏輯、DeckGL pane 包裝。

## Phase 2 已完成（拆除重構）

### 新增模組（第二層）
1. `src/dashboard_modules/building_alias.py`
- `normalize_building_name(...)`
- `expand_building_aliases(...)`
- `resolve_coord_from_aliases(...)`
- `geometry_centroid(...)`

2. `src/dashboard_modules/selection.py`
- `coerce_selected_uid(...)`

### 主幹搬移內容
調整檔案：`src/dashboard_impl.py`

1. 主幹改引用新模組
- 新增 import：building alias/selection helpers。

2. 移除主幹內大型 alias/座標工具實作
- 不再在 `create_dashboard()` 內維護 `_normalize_building_name`。
- `_expand_building_aliases` 改為薄包裝，轉呼叫模組函式。
- `_resolve_coord_from_aliases`、`_geometry_centroid` 改由模組函式負責。

3. UID 清洗邏輯外移
- `_coerce_selected_uid` 改為薄包裝，轉呼叫 `coerce_selected_uid(...)`。

4. 拓樸殘留分支再清理
- `_resolve_map_focus(...)` 中 `[Agg] UID` 特例邏輯移除。
- `meter_info_panel(...)` 中 `[Agg] UID` 特例邏輯移除。
- 主幹檔不再保留拓樸聚合分支。

### 驗證
- `tests/dashboard/test_lazy_startup.py`：6 passed
- 第二階段拆除重構後仍可建立 dashboard template。
