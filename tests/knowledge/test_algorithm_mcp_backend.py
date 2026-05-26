from __future__ import annotations

import pandas as pd
from unittest.mock import patch

from src.algorithm_mcp_backend import AlgorithmMCPBackend


class _FakeEngine:
    building_meta = {
        "AT1040": {"mean_kw": 135.0},
        "BADZERO": {"mean_kw": 0.0},
        "BADMISSING": {},
    }

    def predict(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "physics_pred": [120.0, 130.0, 140.0],
                "residual_pred": [5.0, 4.0, 6.0],
                "residual_std": [2.0, 3.0, 2.0],
                "total_pred": [125.0, 134.0, 146.0],
            },
            index=weather_df.index,
        )

    def predict_building(self, weather_df: pd.DataFrame, uid: str) -> pd.DataFrame:
        result = self.predict(weather_df).copy()
        result["building_rank_index"] = [88.0, 90.0, 95.0]
        result["building_eui_index"] = [0.42, 0.44, 0.47]
        return result


def test_run_pvid_returns_expected_schema():
    backend = AlgorithmMCPBackend(engine_factory=lambda: _FakeEngine())

    result = backend.run_pvid(
        building_uid="AT1040",
        hours=3,
        t_out_series=[30.0, 31.0, 32.0],
        humidity_series=[70.0, 71.0, 72.0],
        start_time="2024-07-01T00:00:00",
    )

    assert result["algo"] == "pvid"
    assert result["status"] == "ok"
    assert result["building_uid"] == "AT1040"
    assert result["hours"] == 3
    assert result["result"]["timestamps"] == [
        "2024-07-01T00:00:00",
        "2024-07-01T01:00:00",
        "2024-07-01T02:00:00",
    ]
    assert result["result"]["physics_pred"] == [120.0, 130.0, 140.0]
    assert result["result"]["building_rank_index"] == [88.0, 90.0, 95.0]
    assert result["result"]["building_eui_index"] == [0.42, 0.44, 0.47]
    assert result["summary"]["mean_total_pred_kw"] == 135.0
    assert result["summary"]["peak_total_pred_kw"] == 146.0
    assert round(result["summary"]["mean_residual_std"], 4) == round((2.0 + 3.0 + 2.0) / 3.0, 4)
    assert round(result["summary"]["uncertainty_pct"], 4) == round(((7.0 / 3.0) / 135.0) * 100.0, 4)
    assert result["provenance"]["model_version"] == "pivd-v12"
    assert result["provenance"]["engine_layers"] == [
        "PhysicsSurrogate",
        "V9WeightReconstructor",
        "BuildingMetadataScaler",
        "V10BootEnsemble",
    ]
    assert len(result["provenance"]["input_hash"]) == 64


def test_run_pvid_pads_short_temperature_series_when_no_weather_file(monkeypatch):
    """Short t_out is padded when file-based weather is skipped."""
    backend = AlgorithmMCPBackend(engine_factory=lambda: _FakeEngine())
    monkeypatch.setattr(AlgorithmMCPBackend, "_load_hourly_weather_slice", lambda self, _s, _h: None)

    result = backend.run_pvid(
        building_uid="AT1040",
        hours=3,
        t_out_series=[30.0, 31.0],
        humidity_series=[70.0, 71.0, 72.0],
        start_time="2024-07-01T00:00:00",
    )

    assert result["status"] == "ok"
    assert result["hours"] == 3
    assert result["provenance"]["weather"]["t_out_series_source"] == "user_padded"
    assert result["provenance"]["weather"]["humidity_series_source"] == "user_provided"


def test_run_pvid_surfaces_engine_initialization_error():
    def _broken_engine():
        raise RuntimeError("Engine not initialized: missing boot ensemble file")

    backend = AlgorithmMCPBackend(engine_factory=_broken_engine)

    result = backend.run_pvid(
        building_uid="AT1040",
        hours=1,
        t_out_series=[30.0],
        humidity_series=[70.0],
        start_time="2024-07-01T00:00:00",
    )

    assert result == {
        "algo": "pvid",
        "status": "error",
        "error": "Engine not initialized: missing boot ensemble file",
        "building_uid": "AT1040",
    }


def test_correlate_algorithms_links_pvid_and_counterfactual():
    backend = AlgorithmMCPBackend(engine_factory=lambda: _FakeEngine())
    pvid_result = {
        "algo": "pvid",
        "building_uid": "AT1040",
        "result": {
            "timestamps": ["2024-07-01T13:00:00", "2024-07-01T14:00:00", "2024-07-01T15:00:00"],
            "total_pred": [130.0, 145.6, 141.0],
        },
        "summary": {
            "peak_total_pred_kw": 145.6,
            "uncertainty_pct": 1.8,
        },
    }
    counterfactual_result = {
        "algo": "counterfactual",
        "summary": {
            "delta_pct": -8.4,
            "label": "冷卻調降 2°C",
        },
    }

    result = backend.correlate_algorithms(
        results=[pvid_result, counterfactual_result],
        question="PVID peak 跟冷卻情境有什麼關聯？",
        building_uid="AT1040",
    )

    assert result["status"] == "ok"
    assert result["building_uid"] == "AT1040"
    assert result["algos_used"] == ["pvid", "counterfactual"]
    assert result["dominant_factor"] == "cooling_load"
    assert result["reasoning_method"] == "rule_based_v1"
    assert result["relationships"]
    assert "145.6 kW" in result["relationships"][0]["finding"]
    assert "冷卻調降 2°C" in result["relationships"][0]["finding"]
    assert result["recommended_action"].startswith("優先檢查冷卻控制序列")


def test_run_openbse_counterfactual_errors_when_building_uid_missing():
    backend = AlgorithmMCPBackend(engine_factory=lambda: _FakeEngine())

    result = backend.run_openbse_counterfactual(building_uid=" ")

    assert result["status"] == "error"
    assert result["error"] == "building_uid is required"


def test_run_openbse_counterfactual_errors_when_building_unknown():
    backend = AlgorithmMCPBackend(engine_factory=lambda: _FakeEngine())

    result = backend.run_openbse_counterfactual(building_uid="MISSING")

    assert result["status"] == "error"
    assert result["error"] == "building_uid not found in building_meta"


def test_run_openbse_counterfactual_errors_when_mean_kw_invalid():
    backend = AlgorithmMCPBackend(engine_factory=lambda: _FakeEngine())

    zero_result = backend.run_openbse_counterfactual(building_uid="BADZERO")
    missing_result = backend.run_openbse_counterfactual(building_uid="BADMISSING")

    assert zero_result["status"] == "error"
    assert zero_result["error"] == "mean_kw missing or non-positive for building_uid=BADZERO"
    assert missing_result["status"] == "error"
    assert missing_result["error"] == "mean_kw missing or non-positive for building_uid=BADMISSING"


def test_run_openbse_counterfactual_allows_positive_mean_kw_override():
    backend = AlgorithmMCPBackend(engine_factory=lambda: _FakeEngine())

    class _StubOpenBSEEngine:
        baseline_values = {"cop": 3.5}

    class _StubResult:
        def summary_dict(self):
            return {"delta_kwh": 0.0, "label": "ok"}

    with patch("src.algorithm_mcp_backend.run_openbse_hybrid_counterfactual", return_value=_StubResult()) as run_mock:
        with patch("src.openbse_counterfactual.OpenBSEDeltaEngine", return_value=_StubOpenBSEEngine()):
            result = backend.run_openbse_counterfactual(
                building_uid="UNKNOWN",
                mean_kw_override=250.0,
            )

    assert result["status"] == "ok"
    assert result["mean_kw_used"] == 250.0
    run_mock.assert_called_once()
