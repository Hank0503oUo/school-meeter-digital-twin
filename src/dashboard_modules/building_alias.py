from __future__ import annotations

import re
from typing import Any
from typing import Mapping


def normalize_building_name(name: str | None) -> str:
    value = str(name or "").strip().lower()
    if (not value) or value == "nan":
        return ""
    value = value.replace("臺", "台")
    for token in (" ", "　", "-", "_", "/", "\\", ".", "．", ",", "，", "、", ":", "：", ";", "；", "(", ")", "（", "）"):
        value = value.replace(token, "")
    return value


def expand_building_aliases(
    *values: object,
    custom_aliases: Mapping[str, tuple[str, ...]] | None = None,
) -> list[str]:
    queue = [str(v).strip() for v in values if str(v).strip() and str(v).strip().lower() != "nan"]
    aliases: list[str] = []
    seen: set[str] = set()
    alias_map = dict(custom_aliases or {})

    while queue:
        alias = queue.pop(0).strip()
        if (not alias) or alias.lower() == "nan" or alias in seen:
            continue
        seen.add(alias)
        aliases.append(alias)

        normalized_brackets = alias.replace("（", "(").replace("）", ")")
        if normalized_brackets != alias:
            queue.append(normalized_brackets)

        for part in re.split(r"[()/（）]", normalized_brackets):
            part = part.strip()
            if len(part) >= 2:
                queue.append(part)

        for sep in ("暨", "/", "／"):
            if sep in normalized_brackets:
                queue.extend(p.strip() for p in normalized_brackets.split(sep) if len(p.strip()) >= 2)

        if normalized_brackets.endswith("學系"):
            queue.append(normalized_brackets[:-2])
        if normalized_brackets.endswith("系館"):
            queue.append(normalized_brackets[:-1])

        for src, dst in (("化學工程", "化工"), ("臺大", "台大")):
            if src in normalized_brackets:
                queue.append(normalized_brackets.replace(src, dst))

        for extra in alias_map.get(normalized_brackets, ()): 
            queue.append(extra)

    aliases.sort(key=len, reverse=True)
    return aliases


def resolve_coord_from_aliases(
    aliases: list[str],
    name_to_coord: dict[str, tuple[float, float]],
) -> tuple[float, float] | None:
    if not aliases:
        return None

    for alias in aliases:
        if alias in name_to_coord:
            return name_to_coord[alias]

    normalized_name_to_coord: dict[str, tuple[float, float]] = {}
    duplicate_norms: set[str] = set()
    for name, coord in name_to_coord.items():
        norm = normalize_building_name(name)
        if not norm:
            continue
        if norm in normalized_name_to_coord:
            duplicate_norms.add(norm)
            continue
        normalized_name_to_coord[norm] = coord

    for norm in duplicate_norms:
        normalized_name_to_coord.pop(norm, None)

    for alias in aliases:
        norm_alias = normalize_building_name(alias)
        if norm_alias and norm_alias in normalized_name_to_coord:
            return normalized_name_to_coord[norm_alias]

    for alias in aliases:
        norm_alias = normalize_building_name(alias)
        if not norm_alias:
            continue
        hits = [
            coord
            for name, coord in name_to_coord.items()
            if norm_alias in normalize_building_name(name)
            or normalize_building_name(name) in norm_alias
        ]
        unique_hits = list(dict.fromkeys(hits))
        if len(unique_hits) == 1:
            return unique_hits[0]
    return None


def geometry_centroid(geometry: dict[str, Any]) -> tuple[float, float] | None:
    coords: list[list[float]] = []

    def _extract(raw: Any) -> None:
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
    lon = float(sum(c[0] for c in coords) / len(coords))
    lat = float(sum(c[1] for c in coords) / len(coords))
    return lon, lat
