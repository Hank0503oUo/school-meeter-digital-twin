# 專案結構總覽

## 啟動入口（雙擊 .cmd 即可）

| 檔案 | 用途 | Port |
|------|------|------|
| `open_demo.cmd` | 智慧啟動校園能源 Dashboard（推薦） | 5006 |
| `open_workbench.cmd` | 啟動新版 Knowledge Workbench | 5007 |
| `run_demo.cmd` | 命令列版，視窗保持開著（附 autoreload） | 5006 |
| `run_workbench.cmd` | 命令列版 Workbench（附 autoreload） | 5007 |
| `run_mcp_server.cmd` | 啟動 MCP 後端 | — |
| `雙擊開始使用_START_HERE.html` | 入口說明頁（看說明用，無法直接點擊啟動） | — |

## Python 入口

| 檔案 | 用途 |
|------|------|
| `app.py` | Panel serve 入口 → 校園能源 Dashboard |
| `app_workbench.py` | Panel serve 入口 → Knowledge Workbench（新版前端） |
| `open_browser_launcher.py` | 智慧啟動器：偵測 port 衝突、自動選埠、等待就緒後開瀏覽器 |
| `mcp_server.py` | MCP 後端伺服器 |

---

## 目錄結構

```
demo/
├── app.py                      # 校園能源 Dashboard 入口（panel serve）
├── app_workbench.py            # Knowledge Workbench 入口（新版前端）
├── open_browser_launcher.py    # 智慧啟動器（for app.py）
├── mcp_server.py               # MCP 後端
│
├── open_demo.cmd               # 雙擊啟動 → 校園 DEMO
├── open_workbench.cmd          # 雙擊啟動 → 新版 Workbench
├── run_demo.cmd                # 命令列版（附 autoreload）
├── run_workbench.cmd           # 命令列版 Workbench
├── run_mcp_server.cmd          # 啟動 MCP 後端
│
├── src/                        # 核心 Python 原始碼
│   ├── dashboard.py            # 統一 create_dashboard / create_knowledge_workbench
│   ├── dashboard_impl.py       # 舊版 Dashboard 薄包裝（轉發到 dashboard_modules）
│   ├── dashboard_modules/      # Dashboard 模組化拆分（新架構）
│   │   ├── factory.py          # 組裝整個 Dashboard 的工廠
│   │   ├── runtime.py          # DashboardRuntime（資料載入、校區切換）
│   │   ├── models.py           # DashboardWidgets 等共用資料模型
│   │   ├── reactive.py         # 響應式重算觸發 trigger_dashboard_recompute
│   │   ├── cache.py            # bounded_cache_get（LRU 年度資料快取）
│   │   ├── map_views.py        # 地圖分頁 UI（MapViewController）
│   │   ├── analysis_views.py   # 分析分頁 UI（AnalysisViewController）
│   │   ├── assistant_views.py  # AI 對話分頁（AssistantController）
│   │   ├── selection.py        # 建物 / 電表選擇邏輯
│   │   └── building_alias.py   # 建物別名對應
│   │
│   ├── nekaise_dashboard.py    # 新版 Workbench UI（Nekaise 風格）
│   ├── knowledge_dashboard.py  # 舊版 Knowledge Workbench（保留備用）
│   ├── dashboard_charts.py     # Plotly 圖表共用函式
│   ├── dashboard_data.py       # 資料載入 helpers
│   ├── dashboard_map.py        # PyDeck 地圖建構
│   ├── dashboard_noncore.py    # Legend / Paper ref / Engine mode markdown
│   ├── dashboard_state.py      # 全域狀態
│   │
│   ├── real_inference_engine.py# PI-VD 推論引擎主體
│   ├── building_inference.py   # 建物層級推論
│   ├── counterfactual.py       # 反事實分析引擎
│   ├── campus_config.py        # 校區設定載入（ntu / ncu）
│   ├── building_coord.py       # 建物座標解析
│   ├── map_builder.py          # 地圖建構薄包裝
│   ├── map_builder_impl.py     # 地圖建構實作
│   ├── map_colors.py           # 地圖顏色方案
│   ├── topology.py             # 電網拓樸資料結構（UI 層已停用）
│   ├── topology_view.py        # 拓樸視覺化（已從主流程移除）
│   │
│   ├── knowledge_base.py       # KnowledgeWorkbench 核心
│   ├── knowledge_models.py     # 知識庫資料模型
│   ├── knowledge_analysis.py   # CloudFirstAnalysisService
│   ├── knowledge_tools.py      # 知識庫工具函式
│   ├── knowledge_mcp_backend.py# 知識庫 MCP 後端
│   ├── knowledge_mcp_server.py # 知識庫 MCP 伺服器
│   ├── algorithm_mcp_backend.py# 演算法 MCP 後端
│   │
│   ├── demo_assistant.py       # Demo 對話助理
│   ├── demo_mcp_server.py      # Demo MCP 伺服器
│   ├── rtem_codex_bridge.py    # RTEM / Codex 橋接
│   ├── lm_studio_client.py     # LM Studio 本地模型客戶端
│   │
│   ├── meter_classifier.py     # 電表分類器
│   ├── meter_matcher.py        # 電表對應
│   ├── epw_reader.py           # EPW 氣象檔讀取
│   ├── solar_api.py            # 太陽能 API
│   ├── osm_fetcher.py          # OSM 資料抓取
│   ├── google_fetcher.py       # Google Maps API 抓取（用途說明見 docs/GOOGLE_BUILDINGS_DATA_USAGE.md）
│   ├── merge_osm_google.py     # OSM + Google 資料合併 → buildings_enhanced.geojson
│   ├── trust_policy.py         # 信任政策
│   ├── project_paths.py        # 專案路徑常數
│   ├── constants.py            # 全域常數
│   └── utils.py                # 雜項工具
│
├── campuses/                   # 校區設定與資料
│   ├── ntu/
│   │   ├── config.yaml         # NTU 校區設定
│   │   ├── data/               # buildings.geojson、energy.geojson 等
│   │   └── models/             # 推論模型快照
│   └── ncu/
│       ├── config.yaml
│       └── data/
│
├── models/                     # 全域推論模型資產
│   ├── v10_boot_ensemble.pkl   # V10 BootEnsemble 模型
│   ├── v10_boot_dataset_2017.csv
│   ├── v12_per_building_summary.csv
│   ├── best_tow_adaptive_v9.yaml
│   ├── NTU_powerMeter_kW_hourly.csv
│   └── weather/                # 氣象資料
│
├── config/                     # 全域設定
│   ├── demo_config.yaml        # Demo 主設定
│   ├── ui_prefs.json           # UI 偏好
│   └── meter_uid_overrides.csv # 電表 UID 覆寫表
│
├── assets/
│   └── custom.css              # Panel 自訂樣式
│
├── data/                       # 執行期資料與快取
│   ├── NTU/                    # NTU 匯出資料
│   ├── NCU/                    # NCU 匯出資料
│   ├── cache/google_maps/      # Google Maps API 快取
│   └── knowledge_workbench/    # Workbench 知識庫儲存
│
├── tests/                      # 測試套件
│   ├── core/                   # 推論引擎、校區設定等單元測試
│   ├── dashboard/              # Dashboard 啟動器測試
│   ├── integration/            # 端到端整合測試
│   └── knowledge/              # 知識庫 / MCP 測試
│
├── dev_artifacts/              # 開發用產出（log、截圖）
├── results_v11_cross_year/     # V11 跨年模擬快取
├── requirements.txt            # Python 套件依賴
└── PROJECT_STRUCTURE.md        # 本文件
```

---

## 兩個前端的差異

| | 校園能源 Dashboard | Knowledge Workbench（新版） |
|---|---|---|
| 入口檔 | `app.py` | `app_workbench.py` |
| 核心模組 | `src/dashboard_modules/` | `src/nekaise_dashboard.py` |
| Port | 5006 | 5007 |
| 主要功能 | 地圖、推論、反事實分析 | 本地 LLM 知識問答、MCP 互動 |
| 啟動 cmd | `open_demo.cmd` | `open_workbench.cmd` |

## 常用命令

```bash
# 校園能源 Dashboard
python -m panel serve app.py --show --port 5006 --autoreload

# 新版 Knowledge Workbench
python -m panel serve app_workbench.py --show --port 5007 --autoreload

# MCP 後端
python mcp_server.py

# 執行測試
python -m pytest tests/ -v
```
