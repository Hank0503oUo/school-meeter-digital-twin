# -*- coding: utf-8 -*-
"""
共用常數定義

所有能源相關的物理常數與 UI 配色集中管理，避免多處重複定義。
"""

# ── 能源物理常數 ──────────────────────────────────────────
GRID_EMISSION_FACTOR: float = 0.494        # kg CO₂ / kWh (台灣電網排放係數)
ELECTRICITY_PRICE_NTD: float = 2.5         # NT$ / kWh (台電綜合電價)
TREE_ANNUAL_ABSORPTION: float = 21.0       # kg CO₂ / tree·yr (闊葉林年均碳吸收)
BASELINE_DATA_YEAR: int = 2017             # 基準資料年份
HOURS_PER_YEAR: int = 8760                 # 一年小時數

# ── UI 配色（柔和版，在暗色/淺色底圖皆可讀）──────────────
CLR_GREEN:  str = "#5eae82"
CLR_RED:    str = "#d07050"
CLR_BLUE:   str = "#5a9ec0"
CLR_ORANGE: str = "#d4a84b"
CLR_PURPLE: str = "#9e6088"
CLR_CYAN:   str = "#58b8d0"

COLOR_MODE_OPTIONS: list[str] = ["tier", "energy", "eui", "dci"]
