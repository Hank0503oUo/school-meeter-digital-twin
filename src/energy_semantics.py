from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


_SOURCE_REGISTRY: dict[str, dict[str, Any]] = {
    "electricity_meter_csv": {
        "label": "Electricity meter CSV",
        "category": "ELE",
        "protocol": "csv",
        "haystack_tags": ["site", "building", "meter", "elec", "kW", "point"],
        "brick_class": "Electrical_Meter",
        "description": "Whole-building and submeter electricity readings for RTEM-style analysis.",
    },
    "DHW": {
        "label": "Domestic hot water BMS tag",
        "category": "BMS",
        "protocol": "placeholder",
        "haystack_tags": ["site", "building", "dhw", "point"],
        "brick_class": "Domestic_Hot_Water_System",
        "description": "Reserved RTEM/BMS source; no packaged demo data is currently available.",
    },
    "GAS": {
        "label": "Gas BMS tag",
        "category": "BMS",
        "protocol": "placeholder",
        "haystack_tags": ["site", "building", "gas", "point"],
        "brick_class": "Gas_Meter",
        "description": "Reserved RTEM/BMS source; no packaged demo data is currently available.",
    },
    "STM": {
        "label": "Steam BMS tag",
        "category": "BMS",
        "protocol": "placeholder",
        "haystack_tags": ["site", "building", "steam", "point"],
        "brick_class": "Steam_Meter",
        "description": "Reserved RTEM/BMS source; no packaged demo data is currently available.",
    },
}


def _csv_probe(path_text: str) -> tuple[bool, list[str], str]:
    path = Path(str(path_text or "")).expanduser()
    if not path.is_file():
        return False, [], f"CSV file is not available: {path}"
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
        if not header:
            return False, [], f"CSV file has no header row: {path}"
        return True, [str(item).strip() for item in header if str(item).strip()], ""
    except UnicodeDecodeError:
        try:
            with path.open("r", encoding="cp950", errors="replace", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
            return True, [str(item).strip() for item in header if str(item).strip()], ""
        except Exception as exc:
            return False, [], f"CSV file could not be read: {exc}"
    except Exception as exc:
        return False, [], f"CSV file could not be read: {exc}"


def list_rtem_sources_impl(campus: str = "NTU", meter_csv_path: str = "") -> dict[str, Any]:
    warnings: list[str] = []
    sources: list[dict[str, Any]] = []
    csv_available, csv_columns, csv_warning = _csv_probe(meter_csv_path) if meter_csv_path else (False, [], "")
    if csv_warning:
        warnings.append(csv_warning)
    if not meter_csv_path:
        warnings.append("meter_csv_path was not provided; electricity_meter_csv is marked unavailable.")

    for source_id, meta in _SOURCE_REGISTRY.items():
        available = source_id == "electricity_meter_csv" and csv_available
        source = {
            "source_id": source_id,
            "status": "available" if available else "unavailable",
            "available": bool(available),
            "campus": str(campus or "NTU").strip() or "NTU",
            "label": meta["label"],
            "category": meta["category"],
            "protocol": meta["protocol"],
            "description": meta["description"],
            "haystack_tags": list(meta["haystack_tags"]),
            "brick_class": meta["brick_class"],
        }
        if source_id == "electricity_meter_csv":
            source["path"] = str(Path(meter_csv_path).expanduser()) if meter_csv_path else ""
            source["columns"] = csv_columns
        else:
            source["missing_data_reason"] = "No packaged RTEM/BMS stream is configured for this demo yet."
        sources.append(source)

    return {
        "status": "ok",
        "campus": str(campus or "NTU").strip() or "NTU",
        "sources": sources,
        "warnings": warnings,
    }


def map_energy_semantics_impl(
    building_uid: str = "",
    meter_name: str = "",
    source_id: str = "electricity_meter_csv",
    campus: str = "NTU",
    meter_csv_path: str = "",
) -> dict[str, Any]:
    source_key = str(source_id or "electricity_meter_csv").strip() or "electricity_meter_csv"
    campus_key = str(campus or "NTU").strip() or "NTU"
    building_key = str(building_uid or "").strip()
    meter_key = str(meter_name or "").strip()
    warnings: list[str] = []

    meta = _SOURCE_REGISTRY.get(source_key)
    if meta is None:
        return {
            "status": "unknown_source",
            "sources": [],
            "semantic_tags": {},
            "relationships": [],
            "warnings": [f"Unknown RTEM source_id: {source_key}"],
        }

    available = False
    columns: list[str] = []
    if source_key == "electricity_meter_csv":
        available, columns, warning = _csv_probe(meter_csv_path)
        if warning:
            warnings.append(warning)
    else:
        warnings.append(f"{source_key} is a placeholder BMS source and has no packaged demo stream.")

    if not available:
        return {
            "status": "unavailable",
            "sources": [{"source_id": source_key, "available": False, "category": meta["category"]}],
            "semantic_tags": {},
            "relationships": [],
            "warnings": warnings,
        }

    entity_id = meter_key or f"{building_key or campus_key}:electricity_meter"
    semantic_tags: dict[str, Any] = {
        "site": campus_key,
        "building": building_key,
        "meter": meter_key or entity_id,
        "elec": True,
        "kW": True,
        "point": True,
        "kind": "Number",
        "unit": "kW",
        "source_id": source_key,
        "brick_class": meta["brick_class"],
        "haystack_tags": list(meta["haystack_tags"]),
    }
    if not building_key:
        warnings.append("building_uid is empty; building relationship is omitted.")
    if not meter_key:
        warnings.append("meter_name is empty; a synthetic meter identifier was used.")

    relationships: list[dict[str, str]] = [
        {"subject": entity_id, "predicate": "isA", "object": meta["brick_class"]},
        {"subject": entity_id, "predicate": "hasPointUnit", "object": "kW"},
        {"subject": entity_id, "predicate": "isMeasuredBy", "object": source_key},
    ]
    if building_key:
        relationships.append({"subject": entity_id, "predicate": "isPartOf", "object": building_key})
        relationships.append({"subject": building_key, "predicate": "isPartOf", "object": campus_key})

    return {
        "status": "ok",
        "sources": [
            {
                "source_id": source_key,
                "available": True,
                "path": str(Path(meter_csv_path).expanduser()),
                "columns": columns,
            }
        ],
        "semantic_tags": semantic_tags,
        "relationships": relationships,
        "warnings": warnings,
    }
