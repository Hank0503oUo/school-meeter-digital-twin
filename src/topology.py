from __future__ import annotations

import numpy as np
import pandas as pd


class GridNode:
    """Represents a node in the electrical grid topology."""

    def __init__(
        self,
        node_id: str,
        level: str,
        parent: "GridNode" | None = None,
        metadata: dict | None = None,
    ):
        self.node_id = node_id
        self.level = level  # 'ROOT', 'ZONE', 'STATION', 'PANEL', 'METER'
        self.parent = parent
        self.children: list[GridNode] = []
        self.metadata = metadata or {}
        # Geolocation for drawing lines
        self.lon: float | None = None
        self.lat: float | None = None

    def add_child(self, child_node: "GridNode") -> None:
        self.children.append(child_node)

    def get_all_leaf_meters(self) -> list[str]:
        """Recursively get all meter IDs (leaf nodes) under this node."""
        if self.level == "METER":
            return [self.node_id]
        meters: list[str] = []
        for child in self.children:
            meters.extend(child.get_all_leaf_meters())
        return meters

    def to_dict(self) -> dict:
        """Convert tree to dictionary for frontend visualization."""
        return {
            "id": self.node_id,
            "name": self.node_id,
            "level": self.level,
            "children": [child.to_dict() for child in self.children],
            **self.metadata,
        }


class GridTopology:
    """Engine for parsing, storing, and querying the grid topology."""

    _METER_ID_COLUMNS = ("迴路編號", "meter_id", "meter", "loop_id")
    _UID_COLUMNS = ("uid", "UID")
    _NAME_COLUMNS = ("館舍", "name", "building_name", "item_name")
    _DISTRICT_COLUMNS = ("分區編號", "district", "zone")

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        campus_config: "CampusConfig" | None = None,
    ):
        self.root = GridNode("Total", "ROOT")
        self.nodes: dict[str, GridNode] = {"Total": self.root}
        self.building_to_meters: dict[str, list[str]] = {}
        # Maps "01A_P1_01" to the actual column name in powerMeter.csv
        self.meter_to_true_col: dict[str, str] = {}
        self._energy_tier_by_uid: dict[str, str] = {}
        self.campus_config = campus_config
        self._build_tree(metadata_df)

    @staticmethod
    def _clean_str(value: object) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if text.lower() == "nan":
            return ""
        return text

    @classmethod
    def _pick_field(cls, row: pd.Series, candidates: tuple[str, ...]) -> str:
        for key in candidates:
            if key in row.index:
                val = cls._clean_str(row.get(key))
                if val:
                    return val
        return ""

    def _get_or_create(self, node_id: str, level: str, parent_id: str) -> GridNode:
        if node_id not in self.nodes:
            parent = self.nodes[parent_id]
            node = GridNode(node_id, level, parent)
            parent.add_child(node)
            self.nodes[node_id] = node
        return self.nodes[node_id]

    def _build_tree(self, df: pd.DataFrame) -> None:
        """Build tree structure from metadata DataFrame."""
        if df is None or df.empty:
            return

        for _, row in df.iterrows():
            meter_id = self._pick_field(row, self._METER_ID_COLUMNS)
            if not meter_id:
                continue

            # Typical format is 01A_P1_01.
            parts = meter_id.split("_")
            parent_id = "Total"
            if len(parts) >= 3:
                zone_stn = parts[0]  # e.g. '01A'
                zone = zone_stn[:2]  # e.g. '01'
                station = zone_stn[2:] if len(zone_stn) > 2 else ""  # e.g. 'A'
                panel = parts[1]  # e.g. 'P1'

                zone_id = f"Zone_{zone}"
                stn_id = f"Stn_{zone}{station}"
                panel_id = f"Pnl_{zone}{station}_{panel}"
                self._get_or_create(zone_id, "ZONE", "Total")
                self._get_or_create(stn_id, "STATION", zone_id)
                self._get_or_create(panel_id, "PANEL", stn_id)
                parent_id = panel_id

            m_node = self._get_or_create(meter_id, "METER", parent_id)

            uid = self._pick_field(row, self._UID_COLUMNS)
            item_name = self._pick_field(row, self._NAME_COLUMNS).replace("\n", "").replace("\r", "")
            district = self._pick_field(row, self._DISTRICT_COLUMNS)

            if uid:
                m_node.metadata["uid"] = uid
            if item_name:
                m_node.metadata["name"] = item_name
            if district:
                m_node.metadata["district"] = district

            if uid:
                self.building_to_meters.setdefault(uid, []).append(meter_id)

    def bind_power_columns(self, power_columns) -> None:
        """
        powerMeter.csv columns can include meter names/noise.
        Bind short meter_id to exact column name for extraction.
        """
        meter_ids = [nid for nid, node in self.nodes.items() if node.level == "METER"]
        meter_ids.sort(key=len, reverse=True)
        skip_cols = {"時間", "ObsTime", "time", "timestamp"}

        for col in power_columns:
            if col in skip_cols:
                continue
            if col in self.nodes and self.nodes[col].level == "METER":
                self.meter_to_true_col[col] = col
                continue

            for meter_id in meter_ids:
                if meter_id in str(col):
                    self.meter_to_true_col[meter_id] = col
                    break

    def get_building_meters(self, uid: str) -> list[str]:
        return self.building_to_meters.get(uid, [])

    def get_building_true_columns(self, uid: str) -> list[str]:
        meters = self.get_building_meters(uid)
        return [self.meter_to_true_col[m] for m in meters if m in self.meter_to_true_col]

    def aggregate_power(self, power_df: pd.DataFrame, node_id: str) -> pd.Series:
        """
        Given a node_id (zone/station/panel/meter),
        find all leaf true columns and sum their power.
        """
        if node_id not in self.nodes:
            return pd.Series(0, index=power_df.index)

        node = self.nodes[node_id]
        leaf_meters = node.get_all_leaf_meters()
        true_cols = [self.meter_to_true_col[m] for m in leaf_meters if m in self.meter_to_true_col]

        valid_cols = [c for c in true_cols if c in power_df.columns]
        if not valid_cols:
            return pd.Series(0, index=power_df.index)

        return power_df[valid_cols].sum(axis=1)

    def aggregate_building_power(self, power_df: pd.DataFrame, uid: str) -> pd.Series:
        """Sum the power of all meters assigned to a specific building UID."""
        true_cols = self.get_building_true_columns(uid)
        valid_cols = [c for c in true_cols if c in power_df.columns]
        if not valid_cols:
            return pd.Series(0, index=power_df.index)
        return power_df[valid_cols].sum(axis=1)

    def assign_node_coordinates(self, building_coords_map: dict[str, tuple[float, float]]) -> None:
        """
        building_coords_map format: {uid: (lon, lat)}.
        Meters use building coords; parents use avg child coords.
        """

        def _compute_coords(node: GridNode) -> tuple[float | None, float | None]:
            if node.level == "METER":
                uid = node.metadata.get("uid")
                if uid in building_coords_map:
                    node.lon, node.lat = building_coords_map[uid]
                return node.lon, node.lat

            lons: list[float] = []
            lats: list[float] = []
            for child in node.children:
                lon, lat = _compute_coords(child)
                if lon is not None and lat is not None:
                    lons.append(lon)
                    lats.append(lat)

            if lons and lats:
                node.lon = sum(lons) / len(lons)
                node.lat = sum(lats) / len(lats)

            # Root node override for aesthetic center.
            if node.level == "ROOT":
                default_lon, default_lat = 121.5375, 25.0175
                if self.campus_config is not None:
                    try:
                        default_lon = float(getattr(self.campus_config, "map_lon", default_lon))
                        default_lat = float(getattr(self.campus_config, "map_lat", default_lat))
                    except (TypeError, ValueError):
                        pass
                if node.lon is None or node.lat is None:
                    node.lon, node.lat = default_lon, default_lat
                # Root is generally further out (fake main station).
                node.lon += 0.001
                node.lat += 0.001

            return node.lon, node.lat

        _compute_coords(self.root)

    @staticmethod
    def _power_to_width(power_val: float) -> float:
        """Map power (kW) to line width in pixels (2..10)."""
        if power_val <= 0:
            return 2.0
        return float(np.clip(2.0 + 8.0 * (np.log10(max(power_val, 1)) / 3.0), 2.0, 10.0))

    @staticmethod
    def _power_to_color(power_val: float) -> list[int]:
        """Map power (kW) to RGBA: blue(low) -> yellow(mid) -> red(high)."""
        if power_val <= 0:
            return [100, 160, 220, 120]
        t = float(np.clip(np.log10(max(power_val, 1)) / 3.0, 0.0, 1.0))
        if t < 0.5:
            s = t / 0.5
            r = int(80 + 175 * s)
            g = int(180 + 20 * s)
            b = int(255 - 205 * s)
        else:
            s = (t - 0.5) / 0.5
            r = 255
            g = int(200 - 150 * s)
            b = int(50 - 50 * s)
        return [r, g, b, 220]

    _LEVEL_RADIUS = {
        "ROOT": 50,
        "ZONE": 30,
        "STATION": 20,
        "PANEL": 12,
        "METER": 6,
    }

    _TIER_COLOR = {
        "HIGH": [215, 48, 39, 230],
        "NORMAL": [240, 196, 25, 220],
        "LOW": [26, 152, 80, 220],
    }

    _TIER_PRIORITY = {
        "LOW": 0,
        "NORMAL": 1,
        "HIGH": 2,
    }

    def assign_energy_tiers(self, tier_by_uid: dict) -> None:
        """Bind building UID -> energy tier mapping to topology nodes."""
        cleaned: dict[str, str] = {}
        for uid, tier in (tier_by_uid or {}).items():
            uid_key = str(uid or "").strip()
            tier_key = str(tier or "").strip().upper()
            if uid_key and tier_key in self._TIER_PRIORITY:
                cleaned[uid_key] = tier_key
        self._energy_tier_by_uid = cleaned

    def _tier_to_color(self, tier: str) -> list[int]:
        return list(self._TIER_COLOR.get(str(tier or "").upper(), [100, 160, 220, 180]))

    def _resolve_node_tier(self, node: GridNode, cache: dict[str, str | None]) -> str | None:
        if node.node_id in cache:
            return cache[node.node_id]

        if node.level == "METER":
            uid = str(node.metadata.get("uid", "")).strip()
            tier = self._energy_tier_by_uid.get(uid)
            cache[node.node_id] = tier
            return tier

        child_tiers = []
        for child in node.children:
            tier = self._resolve_node_tier(child, cache)
            if tier in self._TIER_PRIORITY:
                child_tiers.append(tier)

        if not child_tiers:
            cache[node.node_id] = None
            return None

        tier = max(child_tiers, key=lambda x: self._TIER_PRIORITY.get(x, -1))
        cache[node.node_id] = tier
        return tier

    def generate_trips(self, current_power_df: pd.DataFrame) -> list[dict]:
        """
        Generate Deck.gl TripsLayer compatible line segments from ROOT -> METER.
        """
        trips = []

        def _traverse(node: GridNode) -> None:
            power_val = 0.0
            if len(current_power_df) > 0:
                agg = self.aggregate_power(current_power_df, node.node_id)
                if not agg.empty:
                    power_val = float(agg.iloc[-1])

            if (
                node.parent
                and node.parent.lon is not None
                and node.parent.lat is not None
                and node.lon is not None
                and node.lat is not None
            ):
                trips.append(
                    {
                        "vendor": node.node_id,
                        "path": [
                            [node.parent.lon, node.parent.lat],
                            [node.lon, node.lat],
                        ],
                        "timestamps": [0, 100],
                        "power": power_val,
                        "width": self._power_to_width(power_val),
                        "color": self._power_to_color(power_val),
                        "level": node.level,
                    }
                )
            for c in node.children:
                _traverse(c)

        _traverse(self.root)
        return trips

    def generate_node_data(self, current_power_df: pd.DataFrame) -> list[dict]:
        """Generate node data for ScatterplotLayer."""
        nodes_out: list[dict] = []
        tier_cache: dict[str, str | None] = {}
        for nid, node in self.nodes.items():
            if node.lon is None or node.lat is None:
                continue

            power_val = 0.0
            if len(current_power_df) > 0:
                agg = self.aggregate_power(current_power_df, nid)
                if not agg.empty:
                    power_val = float(agg.iloc[-1])

            radius = self._LEVEL_RADIUS.get(node.level, 8)
            node_tier = self._resolve_node_tier(node, tier_cache)
            color = self._tier_to_color(node_tier) if node_tier else self._power_to_color(power_val)
            if node.level in ("ROOT", "ZONE", "STATION"):
                color[3] = 255

            nodes_out.append(
                {
                    "position": [node.lon, node.lat],
                    "radius": radius,
                    "color": color,
                    "label": nid,
                    "level": node.level,
                    "power": power_val,
                    "energy_tier": node_tier or "",
                }
            )
        return nodes_out
