from src.solar_api import (
    SolarAPIError,
    SolarAPIRequestError,
    probe_building_insights,
    summarize_building_insights,
)


def _sample_response() -> dict:
    return {
        "name": "buildings/test-building",
        "center": {"latitude": 25.01, "longitude": 121.54},
        "imageryQuality": "HIGH",
        "postalCode": "106",
        "regionCode": "TW",
        "administrativeArea": "Taipei City",
        "solarPotential": {
            "buildingStats": {
                "areaMeters2": 1200.0,
                "groundAreaMeters2": 1000.0,
            },
            "wholeRoofStats": {
                "areaMeters2": 880.0,
                "groundAreaMeters2": 840.0,
            },
            "roofSegmentStats": [
                {
                    "pitchDegrees": 10.0,
                    "azimuthDegrees": 180.0,
                    "planeHeightAtCenterMeters": 22.5,
                    "stats": {"areaMeters2": 600.0, "groundAreaMeters2": 580.0},
                },
                {
                    "pitchDegrees": 30.0,
                    "azimuthDegrees": 90.0,
                    "planeHeightAtCenterMeters": 28.0,
                    "stats": {"areaMeters2": 280.0, "groundAreaMeters2": 260.0},
                },
            ],
            "maxArrayAreaMeters2": 500.0,
            "maxArrayPanelsCount": 240,
            "maxSunshineHoursPerYear": 1400.0,
            "panelHeightMeters": 1.879,
            "solarPanels": [{"id": 1}, {"id": 2}],
            "solarPanelConfigs": [{"id": 1}],
        },
    }


def test_summarize_building_insights_extracts_expected_metrics():
    summary = summarize_building_insights(
        _sample_response(),
        label="ntu_center",
        query_lat=25.0174,
        query_lon=121.5405,
        request_quality="HIGH",
        used_expanded_coverage=False,
        attempts=[{"required_quality": "HIGH", "expanded_coverage": False, "status": "ok"}],
    )

    assert summary["label"] == "ntu_center"
    assert summary["imagery_quality"] == "HIGH"
    assert summary["building_area_m2"] == 1200.0
    assert summary["whole_roof_area_m2"] == 880.0
    assert summary["roof_segment_count"] == 2
    assert summary["largest_roof_segment_area_m2"] == 600.0
    assert summary["largest_roof_segment_pitch_degrees"] == 10.0
    assert summary["largest_roof_segment_azimuth_degrees"] == 180.0
    assert round(summary["mean_roof_pitch_degrees_weighted"], 4) == 16.3636
    assert summary["min_roof_plane_height_m"] == 22.5
    assert summary["max_roof_plane_height_m"] == 28.0
    assert summary["roof_plane_height_span_m"] == 5.5
    assert summary["max_array_panels_count"] == 240
    assert summary["solar_panel_count"] == 2
    assert summary["solar_panel_config_count"] == 1
    assert summary["roof_to_ground_ratio"] == 0.88
    assert round(summary["whole_roof_to_ground_ratio"], 4) == 1.0476


def test_probe_building_insights_falls_back_from_high_to_base(monkeypatch):
    calls: list[tuple[str, bool]] = []

    def _fake_fetch(lat, lon, api_key, *, required_quality="HIGH", expanded_coverage=False, timeout=30.0):
        calls.append((required_quality, expanded_coverage))
        if required_quality == "HIGH":
            raise SolarAPIRequestError(
                "Requested entity was not found.",
                status_code=404,
                payload={"error": {"status": "NOT_FOUND"}},
            )
        return _sample_response()

    monkeypatch.setattr("src.solar_api.fetch_building_insights", _fake_fetch)

    result = probe_building_insights(25.0174, 121.5405, "fake-key")

    assert calls == [("HIGH", False), ("BASE", True)]
    assert result.request_quality == "BASE"
    assert result.used_expanded_coverage is True
    assert len(result.attempts) == 2
    assert result.attempts[0]["status"] == "error"
    assert result.attempts[1]["status"] == "ok"


def test_probe_building_insights_does_not_swallow_non_fallback_error(monkeypatch):
    def _fake_fetch(lat, lon, api_key, *, required_quality="HIGH", expanded_coverage=False, timeout=30.0):
        raise SolarAPIRequestError(
            "Forbidden",
            status_code=403,
            payload={"error": {"status": "PERMISSION_DENIED"}},
        )

    monkeypatch.setattr("src.solar_api.fetch_building_insights", _fake_fetch)

    try:
        probe_building_insights(25.0174, 121.5405, "fake-key")
    except SolarAPIError as exc:
        assert "403" in str(exc)
    else:
        raise AssertionError("Expected SolarAPIError for non-fallbackable failure")
