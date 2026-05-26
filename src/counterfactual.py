from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np

from src.constants import (
    GRID_EMISSION_FACTOR,
    ELECTRICITY_PRICE_NTD,
    TREE_ANNUAL_ABSORPTION,
)

if TYPE_CHECKING:
    from src.openbse_counterfactual import OpenBSEDeltaEngine

# 常數已統一移至 src/constants.py，以下保留舊名稱向後相容
GRID_EMISSION_FACTOR = GRID_EMISSION_FACTOR
ELECTRICITY_PRICE_NTD = ELECTRICITY_PRICE_NTD
TREE_ANNUAL_ABSORPTION = TREE_ANNUAL_ABSORPTION


@dataclass
class SensitivityCoefficients:
    """
    Counterfactual sensitivity assumptions.

    Rules used by current demo:
    - cooling_delta_degC lower -> more cooling load (negative coefficient)
    - lighting/occupancy/equipment ratio higher -> more load
    """

    cooling_pct_per_degC: float = -0.030
    lighting_fraction: float = 0.15
    occupancy_fraction: float = 0.08
    equipment_fraction: float = 0.35


@dataclass
class CounterfactualResult:
    delta_kwh: float
    delta_carbon_kg: float
    delta_ntd: float
    delta_pct: float
    equiv_trees: float
    timeseries_new: np.ndarray
    timeseries_base: np.ndarray
    timeseries_phy: np.ndarray
    timeseries_res: np.ndarray
    timeseries_base_phy: np.ndarray
    label: str = ""

    def summary_dict(self) -> dict:
        return {
            "delta_kwh": float(round(float(self.delta_kwh), 0)),
            "delta_carbon_kg": float(round(float(self.delta_carbon_kg), 0)),
            "delta_ntd": float(round(float(self.delta_ntd), 0)),
            "delta_pct": float(round(float(self.delta_pct) * 100, 2)),
            "equiv_trees": float(round(float(self.equiv_trees), 0)),
            "label": self.label,
        }


def run_counterfactual(
    baseline_kwh: np.ndarray | list[float] | float,
    cooling_delta_degC: float = 0.0,
    lighting_ratio: float = 1.0,
    occupancy_ratio: float = 1.0,
    equipment_ratio: float = 1.0,
    sensitivity: Optional[SensitivityCoefficients] = None,
    label: str = "",
    physics_pred: Optional[np.ndarray] = None,
    residual_pred: Optional[np.ndarray] = None,
) -> CounterfactualResult:
    """
    Compute counterfactual timeseries and KPI summary.

    Parameter semantics in this demo:
    - cooling_delta_degC: lower value means lower setpoint and higher cooling load.
    - lighting_ratio / occupancy_ratio / equipment_ratio: higher means higher load.
    - physics_pred: real PI-VD physics layer prediction (from PIVDEngine).
    - residual_pred: real PI-VD residual layer prediction (from PIVDEngine).
      If both are provided, the counterfactual operates on the real decomposition.
      Otherwise falls back to a heuristic split.
    """

    s = sensitivity or SensitivityCoefficients()

    baseline = np.asarray(baseline_kwh, dtype=float).copy()
    if baseline.ndim == 0:
        baseline = baseline.reshape(1)
    if baseline.size == 0:
        raise ValueError("baseline_kwh must contain at least one value.")

    cooling_multiplier = 1.0 + cooling_delta_degC * s.cooling_pct_per_degC
    lighting_delta_frac = (lighting_ratio - 1.0) * s.lighting_fraction
    occupancy_delta_frac = (occupancy_ratio - 1.0) * s.occupancy_fraction
    equipment_delta_frac = (equipment_ratio - 1.0) * s.equipment_fraction

    cooling_portion = 0.40
    base_portion = 1.0 - cooling_portion

    total_multiplier = (
        cooling_portion * cooling_multiplier
        + base_portion
        + lighting_delta_frac
        + occupancy_delta_frac
        + equipment_delta_frac
    )

    # Guard against pathological inputs producing negative load multiplier.
    total_multiplier = float(max(total_multiplier, 0.01))

    # Decompose baseline into physics + residual components
    if physics_pred is not None and residual_pred is not None:
        # Use real PI-VD engine decomposition
        base_phy = np.asarray(physics_pred, dtype=float).copy()
        base_res = np.asarray(residual_pred, dtype=float).copy()
        # Ensure shapes match
        if base_phy.shape != baseline.shape or base_res.shape != baseline.shape:
            raise ValueError(
                f"physics_pred/residual_pred shape {base_phy.shape}/{base_res.shape} "
                f"must match baseline shape {baseline.shape}"
            )
    else:
        # Fallback: heuristic split (use local RNG to avoid global side effects)
        rng = np.random.default_rng(42)
        base_res = rng.normal(0, np.maximum(baseline * 0.15, 10), baseline.shape)

        if len(base_res) > 24:
            kernel = np.ones(3) / 3
            base_res = np.convolve(base_res, kernel, mode="same")
            base_phy = baseline - base_res
        else:
            base_phy = baseline * 0.75
            base_res = baseline * 0.25

    base_phy = np.maximum(base_phy, 0)

    timeseries_phy = base_phy * total_multiplier
    timeseries_res = base_res
    timeseries_new = np.maximum(timeseries_phy + timeseries_res, 0)

    delta_kwh = float(timeseries_new.sum() - baseline.sum())
    delta_carbon = delta_kwh * GRID_EMISSION_FACTOR
    delta_ntd = delta_kwh * ELECTRICITY_PRICE_NTD
    delta_pct = float(timeseries_new.sum() / baseline.sum() - 1.0) if baseline.sum() > 0 else 0.0
    equiv_trees = abs(delta_carbon) / TREE_ANNUAL_ABSORPTION if delta_carbon != 0 else 0.0

    return CounterfactualResult(
        delta_kwh=delta_kwh,
        delta_carbon_kg=delta_carbon,
        delta_ntd=delta_ntd,
        delta_pct=delta_pct,
        equiv_trees=equiv_trees,
        timeseries_new=timeseries_new,
        timeseries_base=baseline,
        timeseries_phy=timeseries_phy,
        timeseries_res=timeseries_res,
        timeseries_base_phy=base_phy,
        label=label or _auto_label(cooling_delta_degC, lighting_ratio, occupancy_ratio, equipment_ratio),
    )


def run_building_counterfactual(
    building_stats: dict,
    cooling_delta_degC: float = 0.0,
    lighting_ratio: float = 1.0,
    occupancy_ratio: float = 1.0,
    equipment_ratio: float = 1.0,
    building_scaler: float = 1.0,
    physics_pred: Optional[np.ndarray] = None,
    residual_pred: Optional[np.ndarray] = None,
) -> dict:
    """Building-level helper using mean_kw repeated over a year.

    Parameters
    ----------
    building_scaler : float
        Per-building metadata correction factor from BuildingMetadataScaler.
        Applied to the baseline before counterfactual computation.
    physics_pred : optional ndarray
        Real physics layer prediction from PI-VD engine.
    residual_pred : optional ndarray
        Real residual layer prediction from PI-VD engine.
    """

    mean_kw = float(building_stats.get("mean_kw", 0) or 0) * building_scaler
    baseline = np.full(8760, mean_kw, dtype=float)
    result = run_counterfactual(
        baseline,
        cooling_delta_degC=cooling_delta_degC,
        lighting_ratio=lighting_ratio,
        occupancy_ratio=occupancy_ratio,
        equipment_ratio=equipment_ratio,
        physics_pred=physics_pred,
        residual_pred=residual_pred,
    )
    return result.summary_dict()


def _auto_label(cool: float, light: float, occ: float, equip: float) -> str:
    parts: list[str] = []
    if cool != 0:
        parts.append(f"冷卻偏移{cool:+.1f}°C")
    if light != 1.0:
        parts.append(f"照明{light:.2f}")
    if occ != 1.0:
        parts.append(f"人員{occ:.2f}")
    if equip != 1.0:
        parts.append(f"設備{equip:.2f}")
    return " | ".join(parts) if parts else "基準線（無修改）"


def run_openbse_hybrid_counterfactual(
    pivd_baseline: np.ndarray | list[float],
    cooling_delta_degC: float = 0.0,
    lighting_ratio: float = 1.0,
    occupancy_ratio: float = 1.0,
    equipment_ratio: float = 1.0,
    cop_ratio: float = 1.0,
    engine: Optional["OpenBSEDeltaEngine"] = None,
    label: str = "",
) -> CounterfactualResult:
    """
    Hybrid counterfactual combining PI-VD calibrated baseline with OpenBSE physics delta.

    Formula::

        E_new = E_PIVD_baseline + (E_OpenBSE_scenario - E_OpenBSE_baseline)

    Parameters
    ----------
    pivd_baseline : array-like, shape (8760,)
        Hourly kW baseline from PI-VD (or mean_kw repeated 8760 times for single-building).
    cooling_delta_degC : float
        Setpoint change in °C.  Positive = warmer setpoint (less cooling).
    lighting_ratio, occupancy_ratio, equipment_ratio : float
        Multipliers relative to OpenBSE base-model values (1.0 = no change).
    cop_ratio : float
        Chiller COP multiplier (1.0 = no change).
    engine : OpenBSEDeltaEngine or None
        Pre-built engine; if None a default engine is instantiated (lazy).
    label : str
        Human-readable scenario label.

    Returns
    -------
    CounterfactualResult
    """
    baseline = np.asarray(pivd_baseline, dtype=float)
    if baseline.shape != (8760,):
        raise ValueError(
            f"pivd_baseline must have shape (8760,); got {baseline.shape}. "
            "Use np.full(8760, mean_kw) for a constant baseline."
        )

    if engine is None:
        from src.openbse_counterfactual import OpenBSEDeltaEngine  # noqa: PLC0415
        engine = OpenBSEDeltaEngine()

    delta, scenario_kw, base_kw = engine.compute_delta(
        cooling_delta_degC=cooling_delta_degC,
        lighting_ratio=lighting_ratio,
        occupancy_ratio=occupancy_ratio,
        equipment_ratio=equipment_ratio,
        cop_ratio=cop_ratio,
    )

    # Hybrid formula: PIVD baseline + physics delta
    timeseries_new = np.maximum(baseline + delta, 0.0)

    # For decomposition display: attribute delta proportionally to physics/residual
    # We don't have PIVD decomposition here, so split using heuristic
    rng = np.random.default_rng(42)
    base_res = rng.normal(0, np.maximum(baseline * 0.10, 5.0), baseline.shape)
    if len(base_res) > 24:
        kernel = np.ones(3) / 3
        base_res = np.convolve(base_res, kernel, mode="same")
    base_phy = baseline - base_res
    base_phy = np.maximum(base_phy, 0.0)

    # Keep the display decomposition conservative but self-consistent:
    # scale the physics layer, then define residual as the exact remainder.
    phy_scale = timeseries_new / np.maximum(baseline, 1e-6)
    timeseries_phy = base_phy * phy_scale
    timeseries_res = timeseries_new - timeseries_phy

    delta_kwh = float(timeseries_new.sum() - baseline.sum())
    delta_carbon = delta_kwh * GRID_EMISSION_FACTOR
    delta_ntd = delta_kwh * ELECTRICITY_PRICE_NTD
    delta_pct = float(timeseries_new.sum() / baseline.sum() - 1.0) if baseline.sum() > 0 else 0.0
    equiv_trees = abs(delta_carbon) / TREE_ANNUAL_ABSORPTION if delta_carbon != 0 else 0.0

    auto = _auto_label(cooling_delta_degC, lighting_ratio, occupancy_ratio, equipment_ratio)
    return CounterfactualResult(
        delta_kwh=delta_kwh,
        delta_carbon_kg=delta_carbon,
        delta_ntd=delta_ntd,
        delta_pct=delta_pct,
        equiv_trees=equiv_trees,
        timeseries_new=timeseries_new,
        timeseries_base=baseline,
        timeseries_phy=timeseries_phy,
        timeseries_res=timeseries_res,
        timeseries_base_phy=base_phy,
        label=label or f"[OpenBSE hybrid] {auto}",
    )


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    campus_hourly = 5100 + 800 * np.sin(np.linspace(0, 2 * np.pi * 365, 8760)) + rng.normal(0, 200, 8760)
    campus_hourly = np.clip(campus_hourly, 2000, 9000)

    result = run_counterfactual(
        campus_hourly,
        cooling_delta_degC=-1.5,
        lighting_ratio=1.10,
        occupancy_ratio=1.05,
        equipment_ratio=1.10,
    )

    print("=== Counterfactual Summary ===")
    for k, v in result.summary_dict().items():
        print(f"{k}: {v}")
