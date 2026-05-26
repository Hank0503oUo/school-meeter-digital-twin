from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests


SOLAR_BUILDING_INSIGHTS_URL = "https://solar.googleapis.com/v1/buildingInsights:findClosest"


class SolarAPIError(RuntimeError):
    """Raised when the Solar API cannot satisfy a request."""


class SolarAPIRequestError(SolarAPIError):
    """Raised for HTTP/API response failures."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}


@dataclass
class SolarProbeResult:
    data: dict[str, Any]
    request_quality: str
    used_expanded_coverage: bool
    attempts: list[dict[str, Any]]


def get_google_maps_api_key(env_var: str = "GOOGLE_MAPS_API_KEY") -> str:
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise SolarAPIError(f"Missing {env_var} in the environment.")
    return key


def fetch_building_insights(
    lat: float,
    lon: float,
    api_key: str,
    *,
    required_quality: str = "HIGH",
    expanded_coverage: bool = False,
    timeout: float = 30.0,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "location.latitude": float(lat),
        "location.longitude": float(lon),
        "requiredQuality": str(required_quality).upper(),
        "key": api_key,
    }
    if expanded_coverage:
        params["experiments"] = "EXPANDED_COVERAGE"

    resp = requests.get(SOLAR_BUILDING_INSIGHTS_URL, params=params, timeout=timeout)
    try:
        payload = resp.json()
    except Exception:
        payload = {"text": resp.text[:1000]}

    if not resp.ok:
        message = f"Solar API request failed with HTTP {resp.status_code}"
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict) and error.get("message"):
            message = str(error["message"])
        raise SolarAPIRequestError(
            message,
            status_code=resp.status_code,
            payload=payload if isinstance(payload, dict) else {"payload": payload},
        )

    if not isinstance(payload, dict):
        raise SolarAPIRequestError(
            "Solar API returned a non-JSON payload.",
            status_code=resp.status_code,
            payload={"payload": payload},
        )

    return payload


def probe_building_insights(
    lat: float,
    lon: float,
    api_key: str,
    *,
    initial_quality: str = "HIGH",
    allow_base_fallback: bool = True,
    timeout: float = 30.0,
) -> SolarProbeResult:
    attempts: list[dict[str, Any]] = []
    first_quality = str(initial_quality).upper()
    fallback_quality = "BASE"

    specs: list[tuple[str, bool]] = [(first_quality, False)]
    if allow_base_fallback and first_quality != fallback_quality:
        specs.append((fallback_quality, True))

    last_error: SolarAPIRequestError | None = None
    for required_quality, expanded_coverage in specs:
        try:
            data = fetch_building_insights(
                lat,
                lon,
                api_key,
                required_quality=required_quality,
                expanded_coverage=expanded_coverage,
                timeout=timeout,
            )
            attempts.append(
                {
                    "required_quality": required_quality,
                    "expanded_coverage": expanded_coverage,
                    "status": "ok",
                }
            )
            return SolarProbeResult(
                data=data,
                request_quality=required_quality,
                used_expanded_coverage=expanded_coverage,
                attempts=attempts,
            )
        except SolarAPIRequestError as exc:
            attempts.append(
                {
                    "required_quality": required_quality,
                    "expanded_coverage": expanded_coverage,
                    "status": "error",
                    "status_code": exc.status_code,
                    "message": str(exc),
                }
            )
            last_error = exc
            should_fallback = (
                allow_base_fallback
                and required_quality == first_quality
                and exc.status_code == 404
            )
            if not should_fallback:
                raise SolarAPIError(_format_probe_failure(exc, attempts)) from exc

    assert last_error is not None
    raise SolarAPIError(_format_probe_failure(last_error, attempts)) from last_error


def summarize_building_insights(
    data: dict[str, Any],
    *,
    label: str,
    query_lat: float,
    query_lon: float,
    request_quality: str,
    used_expanded_coverage: bool,
    attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    solar_potential = data.get("solarPotential") or {}
    building_stats = solar_potential.get("buildingStats") or {}
    whole_roof_stats = solar_potential.get("wholeRoofStats") or {}
    roof_segments = solar_potential.get("roofSegmentStats") or []

    segment_rows: list[dict[str, float]] = []
    for seg in roof_segments:
        stats = seg.get("stats") or {}
        area_m2 = _to_float(stats.get("areaMeters2"))
        segment_rows.append(
            {
                "area_m2": area_m2,
                "ground_area_m2": _to_float(stats.get("groundAreaMeters2")),
                "pitch_degrees": _to_float(seg.get("pitchDegrees")),
                "azimuth_degrees": _to_float(seg.get("azimuthDegrees")),
                "plane_height_m": _to_float(seg.get("planeHeightAtCenterMeters")),
            }
        )

    largest_segment = max(segment_rows, key=lambda row: row["area_m2"], default={})
    plane_heights = [row["plane_height_m"] for row in segment_rows if row["plane_height_m"] is not None]
    weighted_pitch = _weighted_mean(
        [row["pitch_degrees"] for row in segment_rows if row["pitch_degrees"] is not None],
        [row["area_m2"] for row in segment_rows if row["pitch_degrees"] is not None],
    )

    whole_roof_area_m2 = _to_float(whole_roof_stats.get("areaMeters2"))
    whole_roof_ground_area_m2 = _to_float(whole_roof_stats.get("groundAreaMeters2"))
    building_ground_area_m2 = _to_float(building_stats.get("groundAreaMeters2"))

    summary = {
        "label": label,
        "query_latitude": float(query_lat),
        "query_longitude": float(query_lon),
        "building_resource_name": data.get("name"),
        "center_latitude": _to_float((data.get("center") or {}).get("latitude")),
        "center_longitude": _to_float((data.get("center") or {}).get("longitude")),
        "imagery_quality": data.get("imageryQuality"),
        "postal_code": data.get("postalCode"),
        "region_code": data.get("regionCode"),
        "administrative_area": data.get("administrativeArea"),
        "request_quality": request_quality,
        "used_expanded_coverage": bool(used_expanded_coverage),
        "attempts": attempts or [],
        "building_area_m2": _to_float(building_stats.get("areaMeters2")),
        "building_ground_area_m2": building_ground_area_m2,
        "whole_roof_area_m2": whole_roof_area_m2,
        "whole_roof_ground_area_m2": whole_roof_ground_area_m2,
        "roof_segment_count": int(len(segment_rows)),
        "largest_roof_segment_area_m2": _to_float(largest_segment.get("area_m2")),
        "largest_roof_segment_pitch_degrees": _to_float(largest_segment.get("pitch_degrees")),
        "largest_roof_segment_azimuth_degrees": _to_float(largest_segment.get("azimuth_degrees")),
        "mean_roof_pitch_degrees_weighted": weighted_pitch,
        "min_roof_plane_height_m": min(plane_heights) if plane_heights else None,
        "max_roof_plane_height_m": max(plane_heights) if plane_heights else None,
        "roof_plane_height_span_m": (
            max(plane_heights) - min(plane_heights) if len(plane_heights) >= 2 else None
        ),
        "max_array_area_m2": _to_float(solar_potential.get("maxArrayAreaMeters2")),
        "max_array_panels_count": _to_int(solar_potential.get("maxArrayPanelsCount")),
        "solar_panel_count": int(len(solar_potential.get("solarPanels") or [])),
        "solar_panel_config_count": int(len(solar_potential.get("solarPanelConfigs") or [])),
        "panel_height_m": _to_float(solar_potential.get("panelHeightMeters")),
        "max_sunshine_hours_per_year": _to_float(solar_potential.get("maxSunshineHoursPerYear")),
        "carbon_offset_factor_kg_per_mwh": _to_float(data.get("carbonOffsetFactorKgPerMwh")),
        "roof_to_ground_ratio": (
            round(whole_roof_area_m2 / building_ground_area_m2, 4)
            if whole_roof_area_m2 is not None and building_ground_area_m2 not in (None, 0.0)
            else None
        ),
        "whole_roof_to_ground_ratio": (
            round(whole_roof_area_m2 / whole_roof_ground_area_m2, 4)
            if whole_roof_area_m2 is not None and whole_roof_ground_area_m2 not in (None, 0.0)
            else None
        ),
    }
    return summary


def _format_probe_failure(exc: SolarAPIRequestError, attempts: list[dict[str, Any]]) -> str:
    return (
        f"{exc}. Attempts: "
        + "; ".join(
            f"{a.get('required_quality')}"
            f"{'+EXP' if a.get('expanded_coverage') else ''}"
            f"={a.get('status_code', 'ok')}"
            for a in attempts
        )
    )


def _weighted_mean(values: list[float], weights: list[float]) -> float | None:
    pairs = [(float(v), float(w)) for v, w in zip(values, weights) if v is not None and w > 0]
    if not pairs:
        return None
    numerator = sum(v * w for v, w in pairs)
    denominator = sum(w for _, w in pairs)
    if denominator <= 0:
        return None
    return numerator / denominator


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
