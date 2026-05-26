from __future__ import annotations


def build_kpi_card_markdown(summary: dict, scenario_label: str, color_good: str, color_bad: str) -> str:
    delta_kwh = float(summary.get("delta_kwh", 0.0))
    delta_co2 = float(summary.get("delta_carbon_kg", 0.0))
    delta_ntd = float(summary.get("delta_ntd", 0.0))
    delta_pct = float(summary.get("delta_pct", 0.0))
    trees = float(summary.get("equiv_trees", 0.0))

    color_class = "good" if delta_kwh <= 0 else "bad"
    arrow = "▼" if delta_kwh <= 0 else "▲"

    return f"""
<div class="kpi-container animate-entrance">
<h4 class="kpi-header">📊 反事實 KPI 評估</h4>

<div class="kpi-grid">
<div class="kpi-metric-box">
<div class="metric-label">年耗電變化</div>
<div class="metric-value {color_class}">{arrow} {delta_kwh:+,.0f} <span class="metric-unit">kWh</span></div>
</div>

<div class="kpi-metric-box">
<div class="metric-label">電費變化</div>
<div class="metric-value {color_class}">{arrow} {delta_ntd:+,.0f} <span class="metric-unit">NT$</span></div>
</div>

<div class="kpi-metric-box">
<div class="metric-label">碳排變化</div>
<div class="metric-value {color_class}">{delta_co2:+,.0f} <span class="metric-unit">kg CO₂</span></div>
</div>

<div class="kpi-metric-box">
<div class="metric-label">等效植樹</div>
<div class="metric-value good">🌳 {trees:.0f} <span class="metric-unit">棵</span></div>
</div>
</div>

<div class="kpi-footer">
<div class="kpi-footer-left">變化幅度: <b>{delta_pct:+.2f}%</b></div>
<div class="kpi-footer-right">情境: {scenario_label}</div>
</div>
</div>
"""


def build_building_source_notice(source: str, tier: str, eui_str: str) -> tuple[str, str]:
    src = str(source or "").strip().lower()
    tier_text = str(tier or "NORMAL").strip().upper()
    if src == "inferred":
        return (
            "warning",
            "⚠️ 此建物無實測電表，以下為 PI-VD 模型推估值。"
            f" 目前分級: **{tier_text}** ｜ EUI(kW/m²): **{eui_str}**",
        )
    return (
        "success",
        f"✅ 實測電表覆蓋建物。分級: **{tier_text}** ｜ EUI(kW/m²): **{eui_str}**",
    )


def build_legend_markdown() -> str:
    return """
<div class="info-alert animate-entrance">

**地圖圖例**

| 色階 | 意義 |
|:----:|------|
| <span style="color:#ef4444;font-size:16px">■</span> | HIGH（高耗電） |
| <span style="color:#f59e0b;font-size:16px">■</span> | NORMAL（一般） |
| <span style="color:#10b981;font-size:16px">■</span> | LOW（節能） |

</div>
"""


def build_paper_ref_markdown() -> str:
    return """
<div class="info-alert animate-entrance">

**論文章節對照**

| 功能 | 對應章節 |
|------|------|
| EUI Choropleth | Ch 4.3 Fig 4.6 |
| 反事實模擬 | Ch 3.3 |
| UQ 區間 | Fig 4.3/4.4 |
| Top-20 排行 | Fig 4.7 |
| ROI 分析 | Fig 4.8 |

</div>
"""


def build_deployment_dci_markdown(info: dict) -> str:
    color = str(info.get("level_color", "#0ea5e9"))
    return f"""
<div class="info-alert indicator-dci animate-entrance" style="border-left-color: {color};">

<b>Deployment Confidence Index (DCI)</b><br/>
DCI: <span class="score" style="color:{color};">{float(info.get('dci', 0.0)):.0f}/100</span>
（{info.get('level_label', '中')}）
<br/>
<span style="font-size:11px; color:#64748b;">
Coverage: {float(info.get('coverage', 0.0))*100:.0f}% ｜ 地圖飽和度: {float(info.get('map_saturation', 1.0))*100:.0f}%
</span>

</div>
"""


def build_cold_start_markdown(days: int, cv_rmse: float, is_pass: bool, color_good: str, color_bad: str) -> str:
    color_class = "text-success" if is_pass else "text-danger"
    border_class = "info-card-success" if is_pass else "info-card-danger"
    status = "PASS (ASHRAE Guideline 14)" if is_pass else "FAIL (需更多暖機資料)"
    return f"""
<div class="info-card {border_class} animate-entrance">

**14-Day Cold Start 評估**
- 已用天數: {int(days)} 天
- CV-RMSE: <span class="{color_class} font-bold">{float(cv_rmse):.1f}%</span>
- 判定: <span class="{color_class}">{status}</span>

</div>
"""


def build_engine_mode_markdown(engine_mode: str) -> str:
    return (
        f'<div class="status-badge animate-entrance">'
        f'<b>📡 系統狀態</b><br>{engine_mode}</div>'
    )
