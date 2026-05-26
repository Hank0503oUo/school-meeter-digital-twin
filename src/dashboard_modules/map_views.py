from __future__ import annotations

from html import escape
from typing import Callable

import numpy as np
import pandas as pd
import panel as pn

from src.dashboard_modules.models import DashboardWidgets
from src.dashboard_modules.runtime import DashboardRuntime
from src.dashboard_modules.selection import coerce_selected_uid
from src.dashboard_map import compose_map_panel
from src.dashboard_noncore import build_building_source_notice
from src.map_builder import build_campus_map
from src.utils import normalize_meter_name as _normalize_meter_name
from src.utils import split_meter_names as _split_meter_names
from src.utils import to_float as _to_float


class MapViewController:
    def __init__(
        self,
        runtime: DashboardRuntime,
        widgets: DashboardWidgets,
        deployment_info_provider: Callable[[int, str], dict[str, float | str]],
    ) -> None:
        self.runtime = runtime
        self.widgets = widgets
        self.deployment_info_provider = deployment_info_provider

    def _campus_loading_placeholder(self, title: str = "Loading campus map"):
        return pn.pane.HTML(
            f"""
<div class="drilldown-card animate-entrance map-quick-view">
  <div class="map-quick-view-kicker">Map loading</div>
  <h3>{escape(title)}</h3>
  <p>The campus map and building metadata will appear after the selected campus finishes loading.</p>
</div>
""",
            sizing_mode="stretch_width",
        )

    def building_options_for_year(self, year: int) -> dict[str, str]:
        if not self.runtime.campus_loaded:
            return {"[All] Campus (ALL)": "ALL"}
        stats = self.runtime.get_yearly_stats(int(year))
        allowed_uids: set[str] | None = None
        selected_archetypes = [str(value).strip() for value in list(self.widgets.bldg_filter.value or [])]
        normalized_filters = {value for value in selected_archetypes if value and value.lower() != "all"}
        if normalized_filters and not stats.empty and "uid" in stats.columns and "archetype_label" in stats.columns:
            filtered = stats[stats["archetype_label"].astype(str).isin(normalized_filters)]
            allowed_uids = set(filtered["uid"].astype(str).str.strip())

        options: dict[str, str] = {"[All] Campus (ALL)": "ALL"}
        inference_df = self.runtime.get_yearly_inference(int(year))
        if inference_df is not None and not inference_df.empty:
            tmp = inference_df.copy()
            tmp["name"] = tmp["name"].fillna("").astype(str)
            for _, row in tmp.sort_values(["name", "uid"]).iterrows():
                uid = str(row.get("uid", "")).strip()
                if not uid:
                    continue
                if allowed_uids is not None and uid not in allowed_uids:
                    continue
                building_name = str(row.get("name", "")).strip() or uid
                source = str(row.get("data_source", "")).strip().lower()
                prefix = "[Inferred] " if source == "inferred" else ""
                options[f"{prefix}{building_name} ({uid})"] = uid
        elif self.runtime.pivd_engine and self.runtime.pivd_engine.metadata_scaler.is_loaded:
            for uid in self.runtime.pivd_engine.metadata_scaler.list_uids():
                uid_text = str(uid).strip()
                if allowed_uids is not None and uid_text not in allowed_uids:
                    continue
                metadata = self.runtime.pivd_engine.metadata_scaler.get_metadata(uid) or {}
                building_name = str(metadata.get("name", "")).strip() or uid_text
                options[f"[Inferred] {building_name} ({uid_text})"] = uid_text
        return options

    def refresh_building_options(self, event=None) -> None:
        options = self.building_options_for_year(int(self.widgets.year_sel.value))
        current = self.widgets.building_sel.value
        self.widgets.building_sel.options = options
        if current not in options.values():
            self.widgets.building_sel.value = next(iter(options.values()), None)

    def refresh_campus_controls(self) -> None:
        total_buildings = self.runtime.total_building_count()
        self.widgets.building_sel.name = (
            f"Building ({total_buildings})" if total_buildings > 0 else "Building"
        )

        archetype_options = ["All"]
        if not self.runtime.building_stats_base.empty and "archetype_label" in self.runtime.building_stats_base.columns:
            unique_types = self.runtime.building_stats_base["archetype_label"].dropna().astype(str).tolist()
            archetype_options += sorted({value for value in unique_types if value and value.lower() != "nan"})
        self.widgets.bldg_filter.options = archetype_options
        if not list(self.widgets.bldg_filter.value or []):
            self.widgets.bldg_filter.value = ["All"]
        elif "All" not in self.widgets.bldg_filter.value and not set(self.widgets.bldg_filter.value).issubset(
            set(archetype_options)
        ):
            self.widgets.bldg_filter.value = ["All"]

        meter_options = (
            ["All"] + sorted(self.runtime.meter_df["meter_name"].astype(str).tolist())
            if not self.runtime.meter_df.empty
            else ["All"]
        )
        self.widgets.meter_sel.options = meter_options
        if self.widgets.meter_sel.value not in meter_options:
            self.widgets.meter_sel.value = "All"

        self.refresh_building_options()

    def resolve_map_focus(
        self,
        year_geojson: dict,
        selected_uid: str | None = None,
        selected_meter: str | None = None,
    ) -> tuple[float, float, float] | None:
        if not self.runtime.campus_loaded:
            return None
        focus_zoom = 18.5
        uid = coerce_selected_uid(selected_uid)
        meter = str(selected_meter or "").strip()

        if uid and uid != "ALL":
            for feature in year_geojson.get("features", []):
                props = feature.get("properties", {})
                if str(props.get("uid", "")).strip() != uid:
                    continue
                centroid = feature.get("geometry", {})
                coord = self.runtime.building_focus_coords_by_uid.get(uid)
                if coord is not None:
                    return coord[0], coord[1], focus_zoom
                geo_centroid = self._geometry_centroid(centroid)
                if geo_centroid is not None:
                    return geo_centroid[0], geo_centroid[1], focus_zoom
                return None

            coord = self.runtime.building_focus_coords_by_uid.get(uid)
            if coord is not None:
                return coord[0], coord[1], focus_zoom

        if not meter:
            return None

        for feature in year_geojson.get("features", []):
            props = feature.get("properties", {})
            if meter not in str(props.get("meter_name", "")) and meter not in str(props.get("name", "")):
                continue
            geo_centroid = self._geometry_centroid(feature.get("geometry", {}))
            if geo_centroid is not None:
                return geo_centroid[0], geo_centroid[1], focus_zoom
            return None
        return None

    def on_map_click(self, click_state: dict | None) -> None:
        if not self.runtime.campus_loaded:
            return
        if not click_state:
            return
        obj = click_state.get("object", {})
        if not isinstance(obj, dict):
            return

        props = obj.get("properties", obj)
        uid = str(props.get("uid", "")).strip()
        if uid:
            current_options = self.widgets.building_sel.options
            option_values = set(current_options.values()) if isinstance(current_options, dict) else set(current_options)
            if uid in option_values:
                self.widgets.building_sel.value = uid
            else:
                merged = dict(current_options) if isinstance(current_options, dict) else {}
                building_name = str(props.get("name", "")).strip() or uid
                source = str(props.get("data_source", "inferred")).strip().lower()
                prefix = "[Inferred] " if source == "inferred" else ""
                merged[f"{prefix}{building_name} ({uid})"] = uid
                self.widgets.building_sel.options = merged
                self.widgets.building_sel.value = uid

        meter_name = props.get("meter_name")
        if meter_name:
            available_meter_options = set(str(value) for value in self.widgets.meter_sel.options)
            for sub_meter in _split_meter_names(str(meter_name)):
                if sub_meter in available_meter_options:
                    self.widgets.meter_sel.value = sub_meter
                    return

        if self.widgets.meter_sel.value != "All":
            self.widgets.meter_sel.value = "All"

    def build_deck_pane(self, deck_object, tooltips):
        pane = pn.pane.DeckGL(
            object=deck_object,
            sizing_mode="stretch_both",
            min_height=320,
            tooltips=tooltips,
        )
        pane.param.watch(lambda event: self.on_map_click(event.new), "click_state")
        return pane

    def map_panel(
        self,
        year: int,
        selected_color_mode: str,
        selected_meter: str,
        display_mode: str,
        cold_days: int,
        selected_uid: str,
    ):
        if not self.runtime.campus_loaded:
            loading_geojson = self.runtime.build_loading_geojson()
            if not loading_geojson:
                return compose_map_panel(self._campus_loading_placeholder(), None)

            target_lon = self.runtime.active_campus_cfg.map_lon if self.runtime.active_campus_cfg else None
            target_lat = self.runtime.active_campus_cfg.map_lat if self.runtime.active_campus_cfg else None
            target_zoom = self.runtime.active_campus_cfg.map_zoom if self.runtime.active_campus_cfg else None
            loading_deck = build_campus_map(
                loading_geojson,
                color_by="tier",
                selected_meter=None,
                show_virtual=True,
                saturation_scale=0.7,
                deployment_days=0,
                map_lon=target_lon,
                map_lat=target_lat,
                map_zoom=target_zoom,
            )
            loading_tooltips = {
                "html": (
                    "<b>{properties.name}</b><br/>"
                    "Status: preparing campus data<br/>"
                    "Rendering building shell before live metrics arrive"
                ),
                "style": {"backgroundColor": "#ffffff", "color": "#333333"},
            }
            return compose_map_panel(self.build_deck_pane(loading_deck, loading_tooltips), None)
        campus_warning = None
        if self.runtime.campus_loaded and not self.runtime.active_campus_ready:
            missing = ", ".join(self.runtime.active_campus_missing) if self.runtime.active_campus_missing else "unknown"
            campus_warning = pn.pane.Alert(
                f"{self.runtime.active_campus_name} data is incomplete: {missing}",
                alert_type="warning",
                sizing_mode="stretch_width",
            )

        if not self.runtime.active_energy_geojson.exists():
            fail = pn.pane.Markdown(
                f"Missing map source: `{self.runtime.active_energy_geojson}`",
                sizing_mode="stretch_width",
            )
            return compose_map_panel(fail, campus_warning)

        selected = None if selected_meter == "All" else selected_meter
        year_geojson = self.runtime.get_yearly_geojson(int(year))
        deploy_info = self.deployment_info_provider(int(cold_days), selected_meter)
        sat_scale = float(deploy_info["map_saturation"]) if selected_color_mode == "dci" else 1.0
        show_virtual = display_mode != "metered_only"

        target_lon = self.runtime.active_campus_cfg.map_lon if self.runtime.active_campus_cfg else None
        target_lat = self.runtime.active_campus_cfg.map_lat if self.runtime.active_campus_cfg else None
        target_zoom = self.runtime.active_campus_cfg.map_zoom if self.runtime.active_campus_cfg else None
        focus_marker = None

        focus_target = self.resolve_map_focus(year_geojson, selected_uid=selected_uid, selected_meter=selected)
        focus_uid = coerce_selected_uid(selected_uid)
        if focus_target is not None:
            target_lon, target_lat, target_zoom = focus_target
            if focus_uid and focus_uid != "ALL":
                focus_marker = {"lon": float(target_lon), "lat": float(target_lat)}

        deck = build_campus_map(
            year_geojson,
            color_by=selected_color_mode,
            selected_meter=selected,
            show_virtual=show_virtual,
            saturation_scale=sat_scale,
            deployment_days=int(cold_days),
            map_lon=target_lon,
            map_lat=target_lat,
            map_zoom=target_zoom,
            focus_marker=focus_marker,
        )

        tooltips = {
            "html": (
                "<b>{properties.name}</b><br/>"
                "EUI: {properties.eui} kWh/m²/yr<br/>"
                "EUI(kW/m²): {properties.eui_kw_per_m2}<br/>"
                "Tier: {properties.energy_tier}<br/>"
                "Annual: {properties.annual_kwh} kWh<br/>"
                "Mean: {properties.mean_kw} kW<br/>"
                "Source: {properties.data_source}<br/>"
                "Meter: {properties.meter_name}<br/>"
                f"Deployment DCI: {float(deploy_info['dci']):.0f}/100"
            ),
            "style": {"backgroundColor": "#ffffff", "color": "#333333"},
        }
        return compose_map_panel(self.build_deck_pane(deck, tooltips), campus_warning)

    def meter_info_panel(self, selected_meter: str):
        if not self.runtime.campus_loaded:
            return pn.pane.Markdown("Loading campus data...", sizing_mode="stretch_width")
        if selected_meter == "All":
            return pn.pane.Markdown("Select a meter to inspect its metadata.", sizing_mode="stretch_width")
        if self.runtime.meter_df.empty:
            return pn.pane.Markdown("Meter metadata is unavailable.", sizing_mode="stretch_width")

        meter_name = _normalize_meter_name(selected_meter)
        row = self.runtime.meter_df[self.runtime.meter_df["meter_name"] == meter_name]
        if row.empty:
            return pn.pane.Markdown("No metadata found for the selected meter.", sizing_mode="stretch_width")

        record = row.iloc[0]
        return pn.pane.Markdown(
            "\n".join(
                [
                    "### Meter detail",
                    f"- meter: `{selected_meter}`",
                    f"- building: `{record['building_name']}`",
                    f"- mean_kw: `{float(record['mean_kw']):.1f}`",
                    f"- annual_kwh: `{float(record['annual_kwh']):,.0f}`",
                    f"- eui: `{float(record['eui']) if pd.notna(record['eui']) else 'N/A'}`",
                    f"- best_r2_oof: `{float(record['best_r2_oof']):.3f}`",
                ]
            ),
            sizing_mode="stretch_width",
        )

    def building_quick_view(self, year: int, selected_uid: str):
        if not self.runtime.campus_loaded:
            return self._campus_loading_placeholder("Loading building snapshot")
        uid = coerce_selected_uid(selected_uid)
        if not uid or uid == "ALL":
            return pn.pane.HTML(
                "<div class='drilldown-card animate-entrance map-quick-view'><h3>Select a building</h3>"
                "<p>Pick a building from the map or dropdown to inspect its current metrics.</p></div>",
                sizing_mode="stretch_width",
            )

        inference_df = self.runtime.get_yearly_inference(int(year))
        row = (
            inference_df[inference_df["uid"].astype(str).str.strip() == uid]
            if inference_df is not None and not inference_df.empty and "uid" in inference_df.columns
            else pd.DataFrame()
        )
        metadata = {}
        if self.runtime.pivd_engine and self.runtime.pivd_engine.metadata_scaler.is_loaded:
            metadata = self.runtime.pivd_engine.metadata_scaler.get_metadata(uid) or {}
        if row.empty and not metadata:
            return pn.pane.Markdown("", sizing_mode="stretch_width")

        record = row.iloc[0].to_dict() if not row.empty else {}
        name = str(record.get("name") or metadata.get("name") or uid).strip()
        source = str(record.get("data_source", "inferred")).strip().lower()
        tier = str(record.get("energy_tier", "NORMAL")).strip().upper()
        mean_kw = _to_float(record.get("mean_kw", np.nan), np.nan)
        annual_kwh = _to_float(record.get("annual_kwh", np.nan), np.nan)
        eui_kw = _to_float(record.get("eui_kw_per_m2", np.nan), np.nan)
        area = _to_float(record.get("area", metadata.get("area", np.nan)), np.nan)
        floors = _to_float(record.get("floors", metadata.get("floors", np.nan)), np.nan)
        meter_name = str(record.get("meter_name", "")).strip() or "Not linked"
        source_label = "Measured" if source == "metered" else "PI-VD inferred"

        def fmt_num(value: float, suffix: str, digits: int = 1) -> str:
            if not np.isfinite(value):
                return "N/A"
            return f"{value:,.{digits}f} {suffix}".strip()

        return pn.pane.HTML(
            f"""
<div class="drilldown-card animate-entrance map-quick-view">
  <div class="map-quick-view-kicker">Building snapshot</div>
  <h3>{escape(name)}</h3>
  <div class="map-quick-view-meta">UID: <strong>{escape(uid)}</strong></div>
  <div class="map-quick-view-tags">
    <span class="detail-pill detail-pill-neutral">{escape(source_label)}</span>
    <span class="detail-pill detail-pill-neutral">Tier: {escape(tier)}</span>
  </div>
  <div class="map-quick-view-grid">
    <div><span>Mean kW</span><strong>{escape(fmt_num(mean_kw, 'kW'))}</strong></div>
    <div><span>Annual kWh</span><strong>{escape(fmt_num(annual_kwh, 'kWh', 0))}</strong></div>
    <div><span>EUI</span><strong>{escape(fmt_num(eui_kw, 'kW/m²', 4))}</strong></div>
    <div><span>Area</span><strong>{escape(fmt_num(area, 'm²'))}</strong></div>
    <div><span>Floors</span><strong>{escape(fmt_num(floors, '', 0)) if np.isfinite(floors) else 'N/A'}</strong></div>
    <div><span>Meter</span><strong>{escape(meter_name)}</strong></div>
  </div>
</div>
""",
            sizing_mode="stretch_width",
        )

    def building_source_notice(self, year: int, selected_uid: str):
        if not self.runtime.campus_loaded:
            return pn.pane.Markdown("Loading campus data...", sizing_mode="stretch_width")
        inference_df = self.runtime.get_yearly_inference(int(year))
        uid = coerce_selected_uid(selected_uid)
        if inference_df is None or inference_df.empty or not uid:
            return pn.pane.Markdown("", sizing_mode="stretch_width")

        row = inference_df[inference_df["uid"].astype(str).str.strip() == uid]
        if row.empty:
            return pn.pane.Markdown("", sizing_mode="stretch_width")

        record = row.iloc[0]
        source = str(record.get("data_source", "inferred")).strip().lower()
        tier = str(record.get("energy_tier", "NORMAL")).strip().upper()
        eui = _to_float(record.get("eui_kw_per_m2", np.nan), np.nan)
        eui_str = f"{eui:.4f}" if np.isfinite(eui) else "N/A"
        alert_type, message = build_building_source_notice(source, tier, eui_str)
        return pn.pane.Alert(message, alert_type=alert_type, sizing_mode="stretch_width")

    @staticmethod
    def _geometry_centroid(geometry: dict) -> tuple[float, float] | None:
        coords: list[list[float]] = []

        def _extract(raw) -> None:
            if not raw:
                return
            if isinstance(raw[0], (int, float)):
                coords.append(raw)
                return
            for item in raw:
                _extract(item)

        _extract(geometry.get("coordinates", []))
        if not coords:
            return None
        lon = float(sum(coord[0] for coord in coords) / len(coords))
        lat = float(sum(coord[1] for coord in coords) / len(coords))
        return lon, lat
