from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_BASE_YAML_PATH = Path(__file__).resolve().parent.parent.parent / "idf_r2_optimizer" / "models" / "base_openbse" / "ntu_equivalent.yaml"

_TEMPLATE_TOTAL_FLOOR_AREA = 96.0
_TEMPLATE_TOTAL_LIGHTS_W = 699.0
_TEMPLATE_TOTAL_EQUIP_W = 420.0
_TEMPLATE_TOTAL_PEOPLE = 15.0
_TEMPLATE_HEIGHT = 3.66
_TEMPLATE_MEAN_KW = 12.0
_TEMPLATE_COOLING_FRACTION = 0.40
_TEMPLATE_LIGHTING_FRACTION = 0.15
_TEMPLATE_EQUIP_FRACTION = 0.35

_ZONE_RATIOS = {
    "Sales Area": 0.625,
    "Office": 0.125,
    "Restroom": 0.0625,
    "Storage": 0.1875,
}

_SCHEDULE_MAP = {
    "default": {
        "occupancy_weekday": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.7, 0.9, 1.0, 0.8, 0.9, 1.0, 1.0, 1.0, 0.9, 0.7, 0.4, 0.2, 0.0, 0.0, 0.0],
        "occupancy_weekend": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.5, 0.7, 0.7, 0.7, 0.7, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "lighting_weekday": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.5, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.7, 0.4, 0.2, 0.1, 0.1],
        "lighting_weekend": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.5, 0.8, 1.0, 1.0, 1.0, 1.0, 0.8, 0.5, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    },
    "office": {
        "occupancy_weekday": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "occupancy_weekend": [0.0] * 24,
        "lighting_weekday": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.8, 1.0, 1.0, 1.0, 0.8, 1.0, 1.0, 1.0, 0.8, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        "lighting_weekend": [0.1] * 24,
    },
}


def scale_yaml_for_building(
    *,
    building_uid: str,
    floor_area_m2: float,
    mean_kw: float,
    b_floors: int = 1,
    b_type: str = "",
    archetype_label: str = "",
    eui: float = 0.0,
    night_to_day_ratio: float = 0.2,
    peak_kw: float = 0.0,
    output_path: str | Path | None = None,
    base_yaml_path: str | Path | None = None,
) -> dict[str, Any]:
    base_path = Path(base_yaml_path) if base_yaml_path else _BASE_YAML_PATH
    if not base_path.is_file():
        raise FileNotFoundError(f"Base YAML not found: {base_path}")

    with base_path.open("r", encoding="utf-8") as f:
        model = yaml.safe_load(f)

    height = _TEMPLATE_HEIGHT * max(1, b_floors)
    area_scale = floor_area_m2 / _TEMPLATE_TOTAL_FLOOR_AREA if floor_area_m2 > 0 else 1.0
    power_scale = (mean_kw * 1000.0) / (_TEMPLATE_MEAN_KW * 1000.0) if mean_kw > 0 else area_scale

    for zone in model.get("zones", []):
        zname = zone.get("name", "")
        ratio = _ZONE_RATIOS.get(zname, 0.1)
        zone["floor_area"] = round(floor_area_m2 * ratio, 2)
        zone["volume"] = round(floor_area_m2 * ratio * height, 2)

    if area_scale != 1.0:
        for surface in model.get("surfaces", []):
            verts = surface.get("vertices", [])
            new_verts = []
            for v in verts:
                nv = dict(v)
                nv["x"] = round(v.get("x", 0) * (area_scale ** 0.5), 3)
                nv["y"] = round(v.get("y", 0) * (area_scale ** 0.5), 3)
                nv["z"] = round(v.get("z", 0) * (b_floors / 1) if b_floors > 1 else v.get("z", 3.66), 3)
                new_verts.append(nv)
            surface["vertices"] = new_verts

    for light in model.get("lights", []):
        if "power" in light:
            light["power"] = round(light["power"] * power_scale, 1)

    for equip in model.get("equipment", []):
        if "power" in equip:
            equip["power"] = round(equip["power"] * power_scale, 1)

    for person in model.get("people", []):
        if "count" in person:
            person["count"] = round(person["count"] * power_scale, 1)

    if night_to_day_ratio < 0.1:
        for sched in model.get("schedules", []):
            if "weekday" in sched:
                sched["weekday"] = [max(0.0, v * 0.6) for v in sched["weekday"]]
            if "weekend" in sched:
                sched["weekend"] = [max(0.0, v * 0.3) for v in sched["weekend"]]

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            yaml.dump(model, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    return {
        "building_uid": building_uid,
        "floor_area_m2": floor_area_m2,
        "mean_kw": mean_kw,
        "b_floors": b_floors,
        "b_type": b_type,
        "archetype_label": archetype_label,
        "area_scale": round(area_scale, 4),
        "power_scale": round(power_scale, 4),
        "height_m": round(height, 2),
        "estimated_total_lights_w": round(_TEMPLATE_TOTAL_LIGHTS_W * power_scale, 1),
        "estimated_total_equipment_w": round(_TEMPLATE_TOTAL_EQUIP_W * power_scale, 1),
        "estimated_total_people": round(_TEMPLATE_TOTAL_PEOPLE * power_scale, 1),
    }
