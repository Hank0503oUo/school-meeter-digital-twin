from __future__ import annotations

"""
Map builder entrypoint.

Core heavy implementation lives in src.map_builder_impl.
This module keeps only the public build APIs and compatibility re-exports.
"""

import pydeck as pdk

from src.map_builder_impl import (
    build_campus_map as _build_campus_map_impl,
    build_topology_layers as _build_topology_layers_impl,
    export_map_html,
    get_building_stats_df,
    main as _impl_main,
    merge_energy_geojson,
)

# Refactored shared modules (split from legacy map_builder implementation).
from src.map_colors import EUI_COLOR_EXPR
from src.meter_classifier import _aggregate_meter_group, _detect_meter_role, _meter_role_priority
from src.trust_policy import DEFAULT_TRUST_POLICY, _classify_archetype, _load_trust_policy


def build_campus_map(*args, **kwargs) -> pdk.Deck:
    return _build_campus_map_impl(*args, **kwargs)


def build_topology_layers(*args, **kwargs) -> list[pdk.Layer]:
    return _build_topology_layers_impl(*args, **kwargs)


__all__ = [
    "build_campus_map",
    "build_topology_layers",
    "merge_energy_geojson",
    "get_building_stats_df",
    "export_map_html",
    "DEFAULT_TRUST_POLICY",
    "_load_trust_policy",
    "_classify_archetype",
    "_detect_meter_role",
    "_meter_role_priority",
    "_aggregate_meter_group",
    "EUI_COLOR_EXPR",
]


if __name__ == "__main__":
    _impl_main()
