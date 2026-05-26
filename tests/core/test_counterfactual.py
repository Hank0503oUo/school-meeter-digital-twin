# -*- coding: utf-8 -*-
"""Tests for counterfactual.py — pure-function design makes testing straightforward."""

import numpy as np
import pytest

from src.counterfactual import (
    run_counterfactual,
    run_building_counterfactual,
    SensitivityCoefficients,
    CounterfactualResult,
    _auto_label,
)


class TestRunCounterfactual:
    """Core counterfactual engine tests."""

    def test_baseline_no_change(self):
        """All sliders at default → delta should be ~0."""
        baseline = np.full(8760, 100.0)
        result = run_counterfactual(baseline)
        assert isinstance(result, CounterfactualResult)
        assert abs(result.delta_pct) < 0.01  # <1% change

    def test_cooling_increase_raises_load(self):
        """Lowering setpoint (negative delta) should increase cooling load."""
        baseline = np.full(8760, 100.0)
        result = run_counterfactual(baseline, cooling_delta_degC=-2.0)
        assert result.delta_kwh > 0, "Lowering setpoint should increase energy"

    def test_cooling_decrease_lowers_load(self):
        """Raising setpoint (positive delta) should decrease cooling load."""
        baseline = np.full(8760, 100.0)
        result = run_counterfactual(baseline, cooling_delta_degC=2.0)
        assert result.delta_kwh < 0, "Raising setpoint should decrease energy"

    def test_lighting_increase(self):
        """Increasing lighting ratio should increase load."""
        baseline = np.full(8760, 100.0)
        result = run_counterfactual(baseline, lighting_ratio=1.5)
        assert result.delta_kwh > 0

    def test_equipment_increase(self):
        """Increasing equipment ratio should increase load."""
        baseline = np.full(8760, 100.0)
        result = run_counterfactual(baseline, equipment_ratio=1.5)
        assert result.delta_kwh > 0

    def test_carbon_and_cost_consistent(self):
        """CO₂ and NT$ deltas should be proportional to kWh delta."""
        baseline = np.full(8760, 100.0)
        result = run_counterfactual(baseline, cooling_delta_degC=-1.0)
        expected_carbon = result.delta_kwh * 0.494
        expected_ntd = result.delta_kwh * 2.5
        assert abs(result.delta_carbon_kg - expected_carbon) < 1.0
        assert abs(result.delta_ntd - expected_ntd) < 1.0

    def test_tree_equivalence(self):
        """Tree equivalence should be positive when carbon changes."""
        baseline = np.full(8760, 100.0)
        result = run_counterfactual(baseline, cooling_delta_degC=-2.0)
        assert result.equiv_trees > 0

    def test_empty_baseline_raises(self):
        """Empty baseline should raise ValueError."""
        with pytest.raises(ValueError):
            run_counterfactual(np.array([]))

    def test_scalar_baseline(self):
        """Scalar baseline should work."""
        result = run_counterfactual(500.0)
        assert isinstance(result, CounterfactualResult)

    def test_with_real_decomposition(self):
        """When physics_pred and residual_pred are provided, use real decomposition."""
        baseline = np.full(100, 200.0)
        physics = np.full(100, 150.0)
        residual = np.full(100, 50.0)
        result = run_counterfactual(
            baseline,
            cooling_delta_degC=-1.0,
            physics_pred=physics,
            residual_pred=residual,
        )
        assert isinstance(result, CounterfactualResult)
        assert result.timeseries_phy is not None

    def test_shape_mismatch_raises(self):
        """Mismatched shapes should raise ValueError."""
        baseline = np.full(100, 200.0)
        physics = np.full(50, 150.0)
        residual = np.full(100, 50.0)
        with pytest.raises(ValueError, match="shape"):
            run_counterfactual(baseline, physics_pred=physics, residual_pred=residual)

    def test_custom_sensitivity(self):
        """Custom SensitivityCoefficients should override defaults."""
        baseline = np.full(8760, 100.0)
        custom = SensitivityCoefficients(
            cooling_pct_per_degC=-0.10,  # 3x normal
            lighting_fraction=0.30,
        )
        result = run_counterfactual(
            baseline, cooling_delta_degC=-1.0, sensitivity=custom
        )
        # With 3x sensitivity, should have larger delta
        normal = run_counterfactual(baseline, cooling_delta_degC=-1.0)
        assert abs(result.delta_kwh) > abs(normal.delta_kwh)

    def test_summary_dict_keys(self):
        """summary_dict() should return expected keys."""
        result = run_counterfactual(np.full(100, 100.0))
        d = result.summary_dict()
        expected_keys = {"delta_kwh", "delta_carbon_kg", "delta_ntd", "delta_pct", "equiv_trees", "label"}
        assert set(d.keys()) == expected_keys


class TestRunBuildingCounterfactual:
    """Building-level helper tests."""

    def test_basic(self):
        stats = {"mean_kw": 50.0}
        result = run_building_counterfactual(stats, cooling_delta_degC=-1.0)
        assert isinstance(result, dict)
        assert "delta_kwh" in result

    def test_building_scaler(self):
        """Building scaler should proportionally scale the result."""
        stats = {"mean_kw": 100.0}
        r1 = run_building_counterfactual(stats, cooling_delta_degC=-1.0, building_scaler=1.0)
        r2 = run_building_counterfactual(stats, cooling_delta_degC=-1.0, building_scaler=2.0)
        # With 2x scaler, delta should be ~2x larger
        assert abs(r2["delta_kwh"]) > abs(r1["delta_kwh"]) * 1.5


class TestAutoLabel:
    def test_baseline(self):
        label = _auto_label(0.0, 1.0, 1.0, 1.0)
        assert "基準線" in label

    def test_cooling_label(self):
        label = _auto_label(-1.5, 1.0, 1.0, 1.0)
        assert "冷卻" in label

    def test_combined_label(self):
        label = _auto_label(-1.0, 1.1, 1.05, 1.2)
        assert "|" in label
