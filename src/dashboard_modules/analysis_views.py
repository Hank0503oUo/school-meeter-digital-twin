from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go

from src.constants import CLR_BLUE, CLR_GREEN, CLR_ORANGE, CLR_PURPLE, CLR_RED
from src.counterfactual import run_building_counterfactual, run_counterfactual
from src.dashboard_charts import apply_custom_theme
from src.dashboard_modules.models import DashboardWidgets
from src.dashboard_modules.runtime import DashboardRuntime
from src.dashboard_modules.selection import coerce_selected_uid
from src.dashboard_noncore import (
    build_cold_start_markdown,
    build_deployment_dci_markdown,
    build_kpi_card_markdown,
)
from src.utils import normalize_meter_name as _normalize_meter_name
from src.utils import to_float as _to_float


class AnalysisViewController:
    def __init__(self, runtime: DashboardRuntime, widgets: DashboardWidgets) -> None:
        self.runtime = runtime
        self.widgets = widgets

    def _campus_loading_placeholder(self, title: str = "Loading campus data"):
        return pn.pane.HTML(
            f"""
<div class="drilldown-card animate-entrance map-quick-view">
  <div class="map-quick-view-kicker">Dashboard loading</div>
  <h3>{escape(title)}</h3>
  <p>The selected campus is still loading. This panel will update automatically when the data is ready.</p>
</div>
""",
            sizing_mode="stretch_width",
        )

    def cold_start_cvrmse(self, days: int) -> float:
        return float(120.0 * np.exp(-0.15 * float(days)))

    def coverage_ratio_for_selection(self, selected_meter: str) -> float:
        if selected_meter != "All" and not self.runtime.meter_df.empty:
            meter_name = _normalize_meter_name(selected_meter)
            row = self.runtime.meter_df[self.runtime.meter_df["meter_name"] == meter_name]
            if not row.empty:
                coverage = pd.to_numeric(row.iloc[0].get("coverage_ratio", np.nan), errors="coerce")
                if pd.notna(coverage):
                    return float(np.clip(float(coverage), 0.0, 1.0))

        if not self.runtime.building_stats_base.empty and "coverage_ratio" in self.runtime.building_stats_base.columns:
            coverage_series = pd.to_numeric(self.runtime.building_stats_base["coverage_ratio"], errors="coerce").dropna()
            if len(coverage_series) > 0:
                return float(np.clip(float(coverage_series.mean()), 0.0, 1.0))
        return 0.60

    def deployment_confidence(self, days: int, selected_meter: str) -> dict[str, float | str]:
        cv_rmse = self.cold_start_cvrmse(int(days))
        coverage = self.coverage_ratio_for_selection(selected_meter)
        day_score = float(np.clip(days / 30.0, 0.0, 1.0))
        cvrmse_score = float(np.clip(1.0 - (cv_rmse / 80.0), 0.0, 1.0))
        coverage_score = float(np.clip(coverage, 0.0, 1.0))
        dci = 100.0 * (0.50 * cvrmse_score + 0.30 * day_score + 0.20 * coverage_score)

        if dci >= 75.0:
            level = "high"
            level_label = "High"
            level_color = CLR_GREEN
        elif dci >= 50.0:
            level = "medium"
            level_label = "Medium"
            level_color = CLR_ORANGE
        else:
            level = "low"
            level_label = "Low"
            level_color = CLR_RED

        return {
            "cvrmse": cv_rmse,
            "coverage": coverage,
            "dci": float(np.clip(dci, 0.0, 100.0)),
            "level": level,
            "level_label": level_label,
            "level_color": level_color,
            "map_saturation": float(np.clip(0.45 + 0.55 * (dci / 100.0), 0.45, 1.0)),
        }

    def deployment_dci_panel(self, days: int, selected_meter: str):
        if not self.runtime.campus_loaded:
            return self._campus_loading_placeholder("Loading deployment confidence")
        return pn.pane.Markdown(
            build_deployment_dci_markdown(self.deployment_confidence(int(days), selected_meter)),
            sizing_mode="stretch_width",
        )

    def cold_start_kpi(self, days: int, selected_meter: str):
        if not self.runtime.campus_loaded:
            return self._campus_loading_placeholder("Loading cold-start KPI")
        info = self.deployment_confidence(int(days), selected_meter)
        cv_rmse = float(info["cvrmse"])
        return pn.pane.Markdown(
            build_cold_start_markdown(
                days=int(days),
                cv_rmse=cv_rmse,
                is_pass=cv_rmse <= 30.0,
                color_good=CLR_GREEN,
                color_bad=CLR_RED,
            ),
            sizing_mode="stretch_width",
        )

    def kpi_panel(
        self,
        year: int,
        selected_uid: str,
        cooling: float,
        lighting: float,
        occupancy: float,
        equipment: float,
    ):
        if not self.runtime.campus_loaded:
            return self._campus_loading_placeholder("Loading KPI summary")
        uid = coerce_selected_uid(selected_uid)
        campus_factor = float(self.runtime.campus_year_scalers.get(int(year), 1.0))
        base_ts = self.runtime.campus_baseline * campus_factor
        scenario_prefix = ""

        if uid and uid != "ALL":
            inference_df = self.runtime.get_yearly_inference(int(year))
            if inference_df is not None and not inference_df.empty:
                row = inference_df[inference_df["uid"].astype(str).str.strip() == uid]
                if not row.empty:
                    record = row.iloc[0]
                    series = np.asarray(record.get("timeseries", []), dtype=float)
                    if series.size > 0:
                        base_ts = series
                        scenario_prefix = f"[{str(record.get('name', '')).strip() or uid}] "

        result = run_counterfactual(
            base_ts,
            cooling_delta_degC=cooling,
            lighting_ratio=lighting,
            occupancy_ratio=occupancy,
            equipment_ratio=equipment,
        )
        return pn.pane.Markdown(
            build_kpi_card_markdown(
                summary=result.summary_dict(),
                scenario_label=scenario_prefix + result.label,
                color_good=CLR_GREEN,
                color_bad=CLR_RED,
            ),
            sizing_mode="stretch_width",
        )

    def timeseries_chart(
        self,
        year: int,
        selected_uid: str,
        time_scale: str,
        cooling: float,
        lighting: float,
        occupancy: float,
        equipment: float,
    ):
        if not self.runtime.campus_loaded:
            return self._campus_loading_placeholder("Loading timeseries")
        self.widgets.main_spinner.value = True
        try:
            inference_df = self.runtime.get_yearly_inference(int(year))
            if inference_df is None or inference_df.empty:
                return pn.pane.Markdown("No yearly inference data available.", sizing_mode="stretch_width")

            uid = coerce_selected_uid(selected_uid)
            if uid == "ALL":
                valid_series = []
                for _, record in inference_df.iterrows():
                    series = np.asarray(record.get("timeseries", []), dtype=float)
                    if series.size > 0:
                        valid_series.append(series)
                if not valid_series:
                    return pn.pane.Markdown("No building timeseries are available.", sizing_mode="stretch_width")
                min_len = min(len(series) for series in valid_series)
                base = np.sum([series[:min_len] for series in valid_series], axis=0)
                building_name = "Campus aggregate"
                data_source = "mixed"
            else:
                row = inference_df[inference_df["uid"].astype(str).str.strip() == uid]
                if row.empty:
                    row = inference_df.iloc[[0]]
                record = row.iloc[0]
                base = np.asarray(record.get("timeseries", []), dtype=float)
                building_name = str(record.get("name", "")).strip() or uid
                data_source = str(record.get("data_source", "inferred")).strip().lower()

            if base.size == 0:
                return pn.pane.Markdown("The selected building has no timeseries.", sizing_mode="stretch_width")

            result = run_counterfactual(
                base,
                cooling_delta_degC=cooling,
                lighting_ratio=lighting,
                occupancy_ratio=occupancy,
                equipment_ratio=equipment,
            )
            aggregated = self._aggregate_timeseries(
                np.asarray(result.timeseries_phy, dtype=float),
                np.asarray(result.timeseries_res, dtype=float),
                np.asarray(result.timeseries_base, dtype=float),
                int(year),
                str(time_scale),
            )
            if aggregated is None:
                return pn.pane.Markdown("Unable to aggregate the selected timeseries.", sizing_mode="stretch_width")

            x_vals = aggregated["x_vals"]
            cf_phy = aggregated["cf_phy"]
            cf_res = aggregated["cf_res"]
            total = aggregated["total"]
            base_min = aggregated["base_min"]
            base_max = aggregated["base_max"]
            cf_total = cf_phy + cf_res
            upper = cf_total * 1.10
            lower = np.maximum(cf_total * 0.90, 0.0)

            is_metered = (
                data_source == "metered"
                or "ncu_real" in data_source
                or "measured" in data_source
                or "實測" in data_source
            )
            line_dash = "solid" if (is_metered or uid == "ALL") else "dash"
            source_label = "Measured" if is_metered else ("Campus aggregate" if uid == "ALL" else "PI-VD inferred")

            x_vals_list = list(x_vals)
            scatter_type = go.Scattergl if len(x_vals_list) > 400 else go.Scatter
            line_mode = "lines+markers" if len(x_vals_list) <= 12 else "lines"

            figure = go.Figure()
            if base_min is not None and base_max is not None:
                figure.add_trace(
                    go.Scatter(
                        x=x_vals_list + x_vals_list[::-1],
                        y=list(base_max) + list(base_min[::-1]),
                        fill="toself",
                        fillcolor="rgba(150,150,150,0.13)",
                        line=dict(color="rgba(150,150,150,0.3)", width=1),
                        name="Baseline range",
                        showlegend=True,
                        hoverinfo="skip",
                    )
                )

            figure.add_trace(
                go.Scatter(
                    x=x_vals_list + x_vals_list[::-1],
                    y=list(upper) + list(lower[::-1]),
                    fill="toself",
                    fillcolor="rgba(231,76,60,0.12)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="Counterfactual band",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )
            figure.add_trace(
                scatter_type(
                    x=x_vals_list,
                    y=cf_phy,
                    mode=line_mode,
                    name="Physics layer",
                    stackgroup="one",
                    fillcolor="rgba(52,152,219,0.7)",
                    line=dict(color="rgba(52,152,219,1.0)", width=2),
                )
            )
            figure.add_trace(
                scatter_type(
                    x=x_vals_list,
                    y=cf_res,
                    mode=line_mode,
                    name="Residual layer",
                    stackgroup="one",
                    fillcolor="rgba(231,76,60,0.7)",
                    line=dict(color="rgba(231,76,60,1.0)", width=2),
                )
            )
            figure.add_trace(
                scatter_type(
                    x=x_vals_list,
                    y=cf_total,
                    mode=line_mode,
                    name="Counterfactual total",
                    line=dict(color="#e74c3c", width=3, dash=line_dash),
                )
            )
            figure.add_trace(
                scatter_type(
                    x=x_vals_list,
                    y=total,
                    mode=line_mode,
                    name="Baseline",
                    line=dict(color="#333333", width=2, dash=line_dash),
                )
            )

            figure.update_layout(
                margin=dict(l=40, r=15, t=30, b=65),
                title=dict(text=f"{year} | {building_name} | {source_label}", font=dict(size=13)),
                xaxis_title=aggregated["x_title"],
                yaxis_title=aggregated["y_title"],
                uirevision=True,
                legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="left", x=0, font=dict(size=10)),
            )
            return pn.pane.Plotly(apply_custom_theme(figure), sizing_mode="stretch_both", min_height=220)
        finally:
            self.widgets.main_spinner.value = False

    def ranking_chart(self, year: int):
        if not self.runtime.campus_loaded:
            return self._campus_loading_placeholder("Loading ranking chart")
        stats = self.runtime.get_yearly_stats(int(year))
        if stats.empty:
            return pn.pane.Markdown("No stats available.")
        top = stats.nlargest(20, "mean_kw").sort_values("mean_kw")
        colors = [CLR_RED if str(row.get("data_source", "")) == "metered" else CLR_PURPLE for _, row in top.iterrows()]
        figure = go.Figure(
            go.Bar(
                x=top["mean_kw"],
                y=top["name"],
                orientation="h",
                marker_color=colors,
                text=[f"{value:,.0f} kW" for value in top["mean_kw"]],
                textposition="auto",
            )
        )
        figure.update_layout(title="Top-20 mean demand", margin=dict(l=180, r=30, t=50, b=40), xaxis_title="kW")
        return pn.pane.Plotly(apply_custom_theme(figure), sizing_mode="stretch_both", min_height=400)

    def eui_chart(self, year: int):
        if not self.runtime.campus_loaded:
            return self._campus_loading_placeholder("Loading EUI chart")
        stats = self.runtime.get_yearly_stats(int(year))
        if stats.empty:
            return pn.pane.Markdown("No stats available.")
        top = stats.nlargest(20, "eui").sort_values("eui")
        figure = go.Figure(
            go.Bar(
                x=top["eui"],
                y=top["name"],
                orientation="h",
                marker=dict(color=top["eui"], colorscale=[[0, CLR_GREEN], [0.5, CLR_ORANGE], [1, CLR_RED]]),
                text=[f"{value:.0f}" for value in top["eui"]],
                textposition="auto",
            )
        )
        figure.update_layout(title="Top-20 EUI", margin=dict(l=180, r=30, t=50, b=40), xaxis_title="kWh/m²/yr")
        return pn.pane.Plotly(apply_custom_theme(figure), sizing_mode="stretch_both", min_height=400)

    def waste_chart(self, year: int):
        if not self.runtime.campus_loaded:
            return self._campus_loading_placeholder("Loading waste chart")
        stats = self.runtime.get_yearly_stats(int(year))
        if stats.empty:
            return pn.pane.Markdown("No stats available.")
        df = stats.copy()
        df["waste_intensity"] = df["mean_kw"] * (1 - df["best_r2_oof"].clip(lower=0))
        top = df.nlargest(20, "waste_intensity").sort_values("waste_intensity")
        figure = go.Figure(
            go.Bar(
                x=top["waste_intensity"],
                y=top["name"],
                orientation="h",
                marker=dict(color=top["waste_intensity"], colorscale=[[0, CLR_BLUE], [0.5, CLR_ORANGE], [1, CLR_RED]]),
                text=[f"{value:,.1f} kW" for value in top["waste_intensity"]],
                textposition="auto",
            )
        )
        figure.update_layout(title="Top-20 waste-driven load", margin=dict(l=180, r=30, t=50, b=40))
        return pn.pane.Plotly(apply_custom_theme(figure), sizing_mode="stretch_both", min_height=400)

    def r2_scatter(self, year: int):
        if not self.runtime.campus_loaded:
            return self._campus_loading_placeholder("Loading model scatter")
        stats = self.runtime.get_yearly_stats(int(year))
        if stats.empty:
            return pn.pane.Markdown("No stats available.")
        figure = go.Figure(
            go.Scatter(
                x=stats["mean_kw"],
                y=stats["best_r2_oof"],
                mode="markers+text",
                text=stats["name"],
                textposition="top center",
                textfont=dict(size=9, color="#555555"),
                marker=dict(
                    size=stats["mean_kw"].clip(upper=1500) / 30 + 5,
                    color=stats["best_r2_oof"],
                    colorscale=[[0, CLR_RED], [0.5, CLR_ORANGE], [1, CLR_GREEN]],
                    colorbar=dict(title="R²"),
                    line=dict(width=0.5, color="#ffffff"),
                ),
                hovertemplate="<b>%{text}</b><br>Mean kW: %{x:.0f}<br>R²: %{y:.3f}<extra></extra>",
            )
        )
        figure.update_layout(title="R² vs mean demand", margin=dict(l=60, r=30, t=50, b=50))
        return pn.pane.Plotly(apply_custom_theme(figure), sizing_mode="stretch_both", min_height=400)

    def stats_table(self, year: int):
        if not self.runtime.campus_loaded:
            return self._campus_loading_placeholder("Loading building table")
        stats = self.runtime.get_yearly_stats(int(year))
        if stats.empty:
            return pn.pane.Markdown("No stats available.")
        display_df = stats[
            ["name", "mean_kw", "annual_mwh", "eui", "peak_kw", "best_r2_oof", "best_cvrmse_oof", "archetype_label"]
        ].copy()
        display_df.columns = [
            "Building",
            "Mean kW",
            "Annual MWh",
            "EUI",
            "Peak kW",
            "R²",
            "CV-RMSE",
            "Archetype",
        ]
        return pn.widgets.Tabulator(
            display_df.sort_values("Mean kW", ascending=False),
            theme="site",
            pagination="remote",
            page_size=15,
            sizing_mode="stretch_both",
            min_height=400,
        )

    def building_cf_table(self, year: int, cooling: float, lighting: float, occupancy: float, equipment: float):
        if not self.runtime.campus_loaded:
            return self._campus_loading_placeholder("Loading counterfactual table")
        stats = self.runtime.get_yearly_stats(int(year))
        if stats.empty:
            return pn.pane.Markdown("No stats available.")

        self.widgets.main_spinner.value = True
        try:
            rows = []
            for _, row in stats.iterrows():
                cf = run_building_counterfactual(
                    {"mean_kw": row["mean_kw"]},
                    cooling_delta_degC=cooling,
                    lighting_ratio=lighting,
                    occupancy_ratio=occupancy,
                    equipment_ratio=equipment,
                )
                rows.append(
                    {
                        "Building": row["name"],
                        "Baseline annual MWh": round(float(row["mean_kw"]) * 8.76, 1),
                        "Delta kWh": cf["delta_kwh"],
                        "Delta CO2 (kg)": cf["delta_carbon_kg"],
                        "Delta NTD": cf["delta_ntd"],
                        "Delta %": cf["delta_pct"],
                    }
                )
            return pn.widgets.Tabulator(
                pd.DataFrame(rows).sort_values("Delta kWh"),
                theme="site",
                pagination="remote",
                page_size=15,
                sizing_mode="stretch_both",
                min_height=400,
            )
        finally:
            self.widgets.main_spinner.value = False

    def compact_status_bar(self, year: int):
        if not self.runtime.campus_loaded:
            return pn.pane.HTML(
                f"""
<div class="compact-status-bar">
  <span class="csb-campus">{escape(self.runtime.active_campus_name)}</span>
  <span class="csb-divider">&middot;</span>
  <span>Preparing dashboard shell</span>
</div>
""",
                sizing_mode="stretch_width",
            )
        mode_label = "PI-VD" if self.runtime.pivd_engine else "Fallback"
        mode_cls = "good" if self.runtime.pivd_engine else "warn"
        return pn.pane.HTML(
            f"""
<div class="compact-status-bar">
  <span class="csb-campus">{escape(self.runtime.active_campus_name)}</span>
  <span class="csb-divider">&middot;</span>
  <span>{int(year)}</span>
  <span class="csb-divider">&middot;</span>
  <span class="csb-pill csb-pill-{mode_cls}">{mode_label}</span>
  <span class="csb-divider">&middot;</span>
  <span>{self.runtime.total_building_count()} buildings / {len(self.runtime.metered_uid_set)} metered</span>
</div>
""",
            sizing_mode="stretch_width",
        )

    def _aggregate_timeseries(
        self,
        ts_phy: np.ndarray,
        ts_res: np.ndarray,
        ts_base: np.ndarray,
        year: int,
        time_scale: str,
    ) -> dict[str, object] | None:
        if len(ts_phy) == 0:
            return None
        n_hours = len(ts_phy)
        scale = time_scale.split(" ")[0].lower()
        if "hour" in scale:
            step = 3
            idx = np.arange(0, n_hours, step)
            return {
                "cf_phy": ts_phy[idx],
                "cf_res": ts_res[idx],
                "total": ts_base[idx],
                "base_min": None,
                "base_max": None,
                "x_vals": [pd.Timestamp(f"{year}-01-01") + pd.Timedelta(hours=int(i)) for i in idx],
                "x_title": "Date/Time (3h)",
                "y_title": "kW",
            }
        if "week" in scale:
            hours_per_week = 24 * 7
            n_full_weeks = n_hours // hours_per_week
            if n_full_weeks <= 0:
                return None
            trim = n_full_weeks * hours_per_week
            daily_base = ts_base[:trim].reshape(-1, 7, 24).mean(axis=2)
            return {
                "cf_phy": ts_phy[:trim].reshape(-1, hours_per_week).mean(axis=1),
                "cf_res": ts_res[:trim].reshape(-1, hours_per_week).mean(axis=1),
                "total": ts_base[:trim].reshape(-1, hours_per_week).mean(axis=1),
                "base_min": daily_base.min(axis=1),
                "base_max": daily_base.max(axis=1),
                "x_vals": list(range(1, n_full_weeks + 1)),
                "x_title": "Week",
                "y_title": "kW",
            }
        start = pd.Timestamp(f"{year}-01-01")
        months = np.array([(start + pd.Timedelta(hours=int(i))).month for i in range(n_hours)], dtype=int)
        if "month" in scale:
            unique_m = sorted(set(months))
            base_daily = [ts_base[months == month].reshape(-1, 24).mean(axis=1) for month in unique_m]
            return {
                "cf_phy": np.array([ts_phy[months == month].mean() for month in unique_m]),
                "cf_res": np.array([ts_res[months == month].mean() for month in unique_m]),
                "total": np.array([ts_base[months == month].mean() for month in unique_m]),
                "base_min": np.array([values.min() for values in base_daily]),
                "base_max": np.array([values.max() for values in base_daily]),
                "x_vals": [pd.Timestamp(year=year, month=month, day=1).strftime("%b") for month in unique_m],
                "x_title": "Month",
                "y_title": "kW",
            }
        if "quarter" in scale:
            quarters = np.array([((month - 1) // 3) + 1 for month in months], dtype=int)
            unique_q = sorted(set(quarters))
            base_daily = [ts_base[quarters == quarter].reshape(-1, 24).mean(axis=1) for quarter in unique_q]
            return {
                "cf_phy": np.array([ts_phy[quarters == quarter].mean() for quarter in unique_q]),
                "cf_res": np.array([ts_res[quarters == quarter].mean() for quarter in unique_q]),
                "total": np.array([ts_base[quarters == quarter].mean() for quarter in unique_q]),
                "base_min": np.array([values.min() for values in base_daily]),
                "base_max": np.array([values.max() for values in base_daily]),
                "x_vals": [f"Q{quarter}" for quarter in unique_q],
                "x_title": "Quarter",
                "y_title": "kW",
            }
        if "year" in scale:
            unique_m = sorted(set(months))
            base_daily = [ts_base[months == month].reshape(-1, 24).mean(axis=1) for month in unique_m]
            return {
                "cf_phy": np.array([ts_phy[months == month].mean() for month in unique_m]),
                "cf_res": np.array([ts_res[months == month].mean() for month in unique_m]),
                "total": np.array([ts_base[months == month].mean() for month in unique_m]),
                "base_min": np.array([values.min() for values in base_daily]),
                "base_max": np.array([values.max() for values in base_daily]),
                "x_vals": [f"{month}月" for month in unique_m],
                "x_title": "Month",
                "y_title": "kW",
            }

        n_full_days = n_hours // 24
        if n_full_days <= 0:
            return None
        trim = n_full_days * 24
        daily_total = ts_base[:trim].reshape(-1, 24).mean(axis=1)
        return {
            "cf_phy": ts_phy[:trim].reshape(-1, 24).mean(axis=1),
            "cf_res": ts_res[:trim].reshape(-1, 24).mean(axis=1),
            "total": daily_total,
            "base_min": daily_total * 0.9,
            "base_max": daily_total * 1.1,
            "x_vals": list(range(1, n_full_days + 1)),
            "x_title": "Day of year",
            "y_title": "kW",
        }
