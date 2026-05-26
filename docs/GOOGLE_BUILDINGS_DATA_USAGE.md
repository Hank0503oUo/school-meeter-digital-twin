# Google 建物資料（`google_buildings.geojson`）在 DEMO 中的用途

本文件對齊實作位置，避免與「雲端 LLM / Gemini」混淆：此處僅指 **Google Maps / Places / Geocoding / Solar** 透過 `src/google_fetcher.py` 下載的 **矢量建物 GeoJSON**。

---

## 1. 建物 footprint + 面積

| 用途 | 說明 | 專案內如何接上 |
|------|------|----------------|
| **校園圖台底圖建物** | 多邊形決定建物在 pydeck 上的範圍與點選 | 校區 `config.yaml` 的 `paths.buildings_geojson` 指向的 GeoJSON；若要用 Google 強化後的邊界，請改指向合併產物（見下）。 |
| **面積與 EUI 分母** | `footprint_area_m2`（及可由 `geometry` 重算的平面面積）可作 **面積先驗**；儀表板 EUI 多與 **metadata / 推論列** 的 `area`、`eui_kw_per_m2` 連動，更新建物邊界後應回寫或對齊 metadata。 | 幾何面積：`src/utils.py` 的 `geometry_footprint_m2`；fetcher 寫入之 `footprint_area_m2` 見 `google_fetcher.py`。 |
| **與 OSM 比對缺棟** | Google 獨有、OSM 未涵蓋的建物 → 標成 `google_maps_new`；雙方對上但採用 Google 多邊形 → `google_maps_enhanced`。 | `python -m src.merge_osm_google`（輸入 `data/<CAMPUS>/osm_buildings.geojson` + `google_buildings.geojson`，輸出 `buildings_enhanced.geojson`）。合併後將校區 `buildings_geojson` 指到該檔即可讓圖台吃到「補棟／換邊界」結果。 |

---

## 2. `google_place_id` + 名稱 + `types`

| 用途 | 說明 | 專案內如何接上 |
|------|------|----------------|
| **與校內編號（uid／門牌）對照** | Google 名稱與 Places 類型常與校方 `uid`、`doorplate`、中文／英文別名不一致，需 **對照表或模糊比對**，無法僅靠 place_id 自動等於 uid。 | 校區 `metadata_uid.csv`、`metadata_loop.csv`；台大另有 `official_patch` 類 CSV（見 `DashboardRuntime` 載入邏輯）。建議：以 **名稱 + 距離（重心）** 做主對照，`google_place_id` 做穩定外鍵與去重。 |
| **文字搜尋（找建物再對 footprint）** | 用關鍵字在校園附近做 Text Search，輔助「電表／建物名對不起來」的痛點。 | `python -m src.google_fetcher --campus ntu --text-search "關鍵字"`（需設定 `GOOGLE_MAPS_API_KEY`）。 |

---

## 3. 路徑速查

- 下載輸出：`data/NTU/google_buildings.geojson`、`data/NCU/google_buildings.geojson`
- API 快取：`data/cache/google_maps/`
- 合併腳本：`src/merge_osm_google.py`
- 校區設定：`campuses/<id>/config.yaml` → `paths.buildings_geojson`

---

## 4. 與 V12.5 / V12.6 的關係（簡述）

- **Google 矢量**：提供 **幾何與語意標籤**（面積、place、類型），適合餵 **geoPT 類模型** 的輸入特徵與 **OSM 缺棟補洞**。
- **V12.6 response router**：處理的是 **EnergyPlus 蒙特卡羅反應分群 → 幾何路由**，與本 GeoJSON **互補**（一邊是地圖資產，一邊是推論管線先驗），需在 `idf_r2_optimizer` 側對接，非 Panel 內建功能。
