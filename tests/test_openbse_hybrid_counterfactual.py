# -*- coding: utf-8 -*-
"""Unit tests for PI-VD + OpenBSE hybrid counterfactual."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.counterfactual import (
    CounterfactualResult,
    run_openbse_hybrid_counterfactual,
)


def _make_delta_engine(delta_kw: float = 50.0) -> MagicMock:
    """Return a mock OpenBSEDeltaEngine where scenario uses +delta_kw uniformly."""
    engine = MagicMock()
    baseline_kw = np.full(8760, 100.0)
    scenario_kw = np.full(8760, 100.0 + delta_kw)
    engine.compute_delta.return_value = (scenario_kw - baseline_kw, scenario_kw, baseline_kw)
    engine.baseline_values = {
        "set_point": 23.9,
        "lighting_density": 522.0,
        "equipment_density": 240.0,
        "people_density": 13.0,
        "cop": 3.5,
    }
    return engine


def test_hybrid_result_type():
    engine = _make_delta_engine(delta_kw=0.0)
    baseline = np.full(8760, 500.0)
    result = run_openbse_hybrid_counterfactual(baseline, engine=engine)
    assert isinstance(result, CounterfactualResult)


def test_hybrid_no_change_delta_zero():
    engine = _make_delta_engine(delta_kw=0.0)
    baseline = np.full(8760, 500.0)
    result = run_openbse_hybrid_counterfactual(baseline, engine=engine)
    assert np.allclose(result.delta_kwh, 0.0, atol=1.0)
    assert abs(result.delta_pct) < 1e-6


def test_hybrid_positive_delta_increases_load():
    engine = _make_delta_engine(delta_kw=50.0)
    baseline = np.full(8760, 500.0)
    result = run_openbse_hybrid_counterfactual(baseline, engine=engine)
    # Expected: 50 kW × 8760 h = 438 000 kWh increase
    assert result.delta_kwh > 0
    assert np.allclose(result.delta_kwh, 50.0 * 8760, rtol=0.01)


def test_hybrid_negative_delta_decreases_load():
    engine = _make_delta_engine(delta_kw=-30.0)
    baseline = np.full(8760, 500.0)
    result = run_openbse_hybrid_counterfactual(
        baseline, cooling_delta_degC=2.0, engine=engine
    )
    assert result.delta_kwh < 0


@pytest.mark.parametrize("delta_kw", [50.0, -30.0])
def test_hybrid_decomposition_is_conservative(delta_kw):
    engine = _make_delta_engine(delta_kw=delta_kw)
    baseline = np.full(8760, 500.0)
    result = run_openbse_hybrid_counterfactual(baseline, engine=engine)
    assert np.allclose(
        result.timeseries_new,
        result.timeseries_phy + result.timeseries_res,
    )


def test_hybrid_timeseries_shape():
    engine = _make_delta_engine(delta_kw=10.0)
    baseline = np.full(8760, 200.0)
    result = run_openbse_hybrid_counterfactual(baseline, engine=engine)
    assert result.timeseries_new.shape == (8760,)
    assert result.timeseries_base.shape == (8760,)
    assert (result.timeseries_new >= 0).all()


def test_hybrid_wrong_baseline_shape_raises():
    engine = _make_delta_engine()
    with pytest.raises(ValueError, match="shape .8760"):
        run_openbse_hybrid_counterfactual(np.full(24, 100.0), engine=engine)


def test_hybrid_label_contains_marker():
    engine = _make_delta_engine()
    baseline = np.full(8760, 100.0)
    result = run_openbse_hybrid_counterfactual(
        baseline, cooling_delta_degC=-1.0, engine=engine
    )
    assert "OpenBSE hybrid" in result.label or "冷卻" in result.label


def test_hybrid_summary_dict_keys():
    engine = _make_delta_engine()
    baseline = np.full(8760, 300.0)
    result = run_openbse_hybrid_counterfactual(baseline, engine=engine)
    summary = result.summary_dict()
    required = {"delta_kwh", "delta_carbon_kg", "delta_ntd", "delta_pct", "equiv_trees"}
    assert required.issubset(summary.keys())
