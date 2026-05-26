# -*- coding: utf-8 -*-
"""
PI-VD + OpenBSE hybrid counterfactual delta engine.

Formula:
    E_new = E_PIVD_baseline + (E_OpenBSE_scenario - E_OpenBSE_baseline)

OpenBSE provides physics-accurate per-parameter deltas; PI-VD provides the
calibrated real-world baseline that captures measurement residuals.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ── Project-relative path resolution ────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent
_DEMO_ROOT = _SRC_DIR.parent
_WORKSPACE_ROOT = _DEMO_ROOT.parent

_OPENBSE_EXE_ENV = "OPENBSE_EXE"
_OPENBSE_OPTIMIZER_ROOT_ENV = "OPENBSE_OPTIMIZER_ROOT"
_OPENBSE_BASE_YAML_ENV = "OPENBSE_BASE_YAML"
_OPENBSE_PARAM_MAPPING_ENV = "OPENBSE_PARAM_MAPPING"
_OPENBSE_WEATHER_FILE_ENV = "OPENBSE_WEATHER_FILE"

# Paths into the base YAML for reading baseline values
_BASELINE_PATHS: dict[str, list] = {
    "set_point": ["thermostats", 0, "cooling_setpoint"],
    "lighting_density": ["lights", 0, "power"],
    "equipment_density": ["equipment", 0, "power"],
    "people_density": ["people", 0, "count"],
    "cop": ["air_loops", 0, "equipment", 2, "cop"],
}

# Known baseline values as fallback (read from ntu_equivalent.yaml)
_KNOWN_BASELINE: dict[str, float] = {
    "set_point": 23.9,
    "lighting_density": 522.0,
    "equipment_density": 240.0,
    "people_density": 13.0,
    "cop": 3.5,
}


def _get_nested(obj: Any, path: list) -> Any:
    cur = obj
    for k in path:
        cur = cur[k]
    return cur


def _existing_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    try:
        return path.resolve()
    except OSError:
        return path


def _first_existing_file(*candidates: str | Path | None) -> Path | None:
    for candidate in candidates:
        path = _existing_path(candidate)
        if path is not None and path.is_file():
            return path
    return None


def _first_existing_dir(*candidates: str | Path | None) -> Path | None:
    for candidate in candidates:
        path = _existing_path(candidate)
        if path is not None and path.is_dir():
            return path
    return None


def _resolve_optimizer_root() -> Path:
    optimizer_root = _first_existing_dir(
        os.environ.get(_OPENBSE_OPTIMIZER_ROOT_ENV),
        _WORKSPACE_ROOT / "idf_r2_optimizer",
    )
    if optimizer_root is None:
        raise FileNotFoundError(
            "OpenBSE optimizer root not found. Set OPENBSE_OPTIMIZER_ROOT or place idf_r2_optimizer beside the demo folder."
        )
    return optimizer_root


def _resolve_required_file(label: str, env_name: str, *candidates: str | Path | None) -> Path:
    path = _first_existing_file(os.environ.get(env_name), *candidates)
    if path is None:
        candidate_text = ", ".join(str(_existing_path(candidate)) for candidate in candidates if candidate is not None)
        raise FileNotFoundError(
            f"{label} not found. Set {env_name} or provide one of: {candidate_text}"
        )
    return path


def _resolve_openbse_exe() -> Path:
    path = _resolve_required_file(
        "OpenBSE executable",
        _OPENBSE_EXE_ENV,
        _DEMO_ROOT / "bin" / "openbse.exe",
        _WORKSPACE_ROOT / "openbse_bin" / "openbse.exe",
        _WORKSPACE_ROOT / "openbse_bin" / "openbse",
    )
    return path


def _ensure_optimizer_on_path(optimizer_src: Path) -> None:
    if str(optimizer_src) not in sys.path:
        sys.path.insert(0, str(optimizer_src))


def _decode_process_output(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


class OpenBSEDeltaEngine:
    """
    Runs OpenBSE twice (baseline + scenario) and returns the physics delta.

    Usage::

        engine = OpenBSEDeltaEngine()
        delta, scen_kw, base_kw = engine.compute_delta(
            cooling_delta_degC=-1.0,
            lighting_ratio=1.1,
        )
        hybrid_series = pivd_baseline_8760 + delta
    """

    def __init__(
        self,
        openbse_exe: str | Path | None = None,
        base_yaml: str | Path | None = None,
        param_mapping: str | Path | None = None,
        weather_file: str | Path | None = None,
        timeout: int = 600,
    ):
        optimizer_root = _resolve_optimizer_root()
        optimizer_src = optimizer_root / "src"

        self.openbse_exe = _resolve_required_file(
            "OpenBSE executable",
            _OPENBSE_EXE_ENV,
            openbse_exe,
            _DEMO_ROOT / "bin" / "openbse.exe",
            _WORKSPACE_ROOT / "openbse_bin" / "openbse.exe",
            _WORKSPACE_ROOT / "openbse_bin" / "openbse",
        )
        self.base_yaml = _resolve_required_file(
            "OpenBSE base YAML",
            _OPENBSE_BASE_YAML_ENV,
            base_yaml,
            optimizer_root / "models" / "base_openbse" / "ntu_equivalent.yaml",
        )
        mapping_path = _resolve_required_file(
            "OpenBSE parameter mapping",
            _OPENBSE_PARAM_MAPPING_ENV,
            param_mapping,
            optimizer_root / "config" / "openbse_param_mapping.yaml",
        )
        self.weather_file = _resolve_required_file(
            "OpenBSE weather file",
            _OPENBSE_WEATHER_FILE_ENV,
            weather_file,
            _DEMO_ROOT / "models" / "weather" / "CWBTP_2017.epw",
            _WORKSPACE_ROOT / "EP_auto_calibration" / "weather" / "CWBTP_2017.epw",
        )
        self.timeout = timeout

        _ensure_optimizer_on_path(optimizer_src)
        from backends.openbse_yaml_modifier import OpenBSEYamlModifier, _PassthroughLoader  # noqa: PLC0415
        from backends.openbse_output_extractor import OpenBSEOutputExtractor  # noqa: PLC0415

        with mapping_path.open("r", encoding="utf-8") as f:
            raw_mapping: dict = yaml.safe_load(f)

        self._modifier = OpenBSEYamlModifier(self.base_yaml, raw_mapping)
        self._extractor = OpenBSEOutputExtractor()

        # Read baseline values from YAML (fall back to known constants)
        try:
            with self.base_yaml.open("r", encoding="utf-8") as f:
                model = yaml.load(f, Loader=_PassthroughLoader)  # noqa: S506
            self._baseline: dict[str, float] = {}
            for name, path in _BASELINE_PATHS.items():
                try:
                    self._baseline[name] = float(_get_nested(model, path))
                except (KeyError, IndexError, TypeError):
                    self._baseline[name] = _KNOWN_BASELINE[name]
        except Exception:
            self._baseline = dict(_KNOWN_BASELINE)

    # ------------------------------------------------------------------
    def _run_openbse(self, yaml_path: Path, output_csv: Path) -> np.ndarray:
        return self._run_openbse_full(yaml_path, output_csv)[0]

    def _run_openbse_full(self, yaml_path: Path, output_csv: Path) -> tuple[np.ndarray, pd.DataFrame]:
        cmd = [
            str(self.openbse_exe),
            str(yaml_path),
            "-w", str(self.weather_file),
            "-o", str(output_csv),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=False,
            timeout=self.timeout,
        )
        if result.returncode != 0:
            stderr_text = _decode_process_output(result.stderr)
            stdout_text = _decode_process_output(result.stdout)
            details = stderr_text or stdout_text or "(no output)"
            raise RuntimeError(
                f"OpenBSE failed (rc={result.returncode}): {details[:500]}"
            )
        raw_df = pd.read_csv(output_csv)
        total_kw = self._extractor.extract(output_csv)
        if total_kw.shape != (8760,):
            raise ValueError(f"Expected shape (8760,); got {total_kw.shape}")
        return total_kw, raw_df

    # ------------------------------------------------------------------
    def compute_delta(
        self,
        *,
        cooling_delta_degC: float = 0.0,
        lighting_ratio: float = 1.0,
        occupancy_ratio: float = 1.0,
        equipment_ratio: float = 1.0,
        cop_ratio: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run baseline + scenario simulations, return (delta, scenario_kw, baseline_kw).

        All arrays have shape (8760,) in kW.
        delta = scenario_kw - baseline_kw
        """
        scenario_params = {
            "set_point": self._baseline["set_point"] + cooling_delta_degC,
            "lighting_density": self._baseline["lighting_density"] * lighting_ratio,
            "people_density": self._baseline["people_density"] * occupancy_ratio,
            "equipment_density": self._baseline["equipment_density"] * equipment_ratio,
            "cop": self._baseline["cop"] * cop_ratio,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            baseline_csv = tmp / "baseline.csv"
            baseline_kw = self._run_openbse(self.base_yaml, baseline_csv)

            scenario_yaml = tmp / "scenario.yaml"
            self._modifier.modify(scenario_params, scenario_yaml)
            scenario_csv = tmp / "scenario.csv"
            scenario_kw = self._run_openbse(scenario_yaml, scenario_csv)

        delta = scenario_kw - baseline_kw
        return delta, scenario_kw, baseline_kw

    @property
    def baseline_values(self) -> dict[str, float]:
        """Read-only view of baseline parameter values extracted from the base YAML."""
        return dict(self._baseline)

    def compute_hvac_breakdown(
        self,
        *,
        cooling_delta_degC: float = 0.0,
        lighting_ratio: float = 1.0,
        occupancy_ratio: float = 1.0,
        equipment_ratio: float = 1.0,
        cop_ratio: float = 1.0,
    ) -> dict[str, Any]:
        scenario_params = {
            "set_point": self._baseline["set_point"] + cooling_delta_degC,
            "lighting_density": self._baseline["lighting_density"] * lighting_ratio,
            "people_density": self._baseline["people_density"] * occupancy_ratio,
            "equipment_density": self._baseline["equipment_density"] * equipment_ratio,
            "cop": self._baseline["cop"] * cop_ratio,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            baseline_csv = tmp / "baseline.csv"
            baseline_kw, base_df = self._run_openbse_full(self.base_yaml, baseline_csv)

            scenario_yaml = tmp / "scenario.yaml"
            self._modifier.modify(scenario_params, scenario_yaml)
            scenario_csv = tmp / "scenario.csv"
            scenario_kw, scen_df = self._run_openbse_full(scenario_yaml, scenario_csv)

        delta = scenario_kw - baseline_kw

        def _extract_hvac(df: pd.DataFrame) -> dict[str, Any]:
            cols = {c: c.replace(" [-]", "").strip() for c in df.columns}
            df = df.rename(columns=cols)

            hvac = {}
            cooling_cols = [c for c in df.columns if ":cooling_load" in c]
            heating_cols = [c for c in df.columns if ":heating_load" in c]
            hvac_cool_cols = [c for c in df.columns if ":hvac_cooling_rate" in c]
            hvac_heat_cols = [c for c in df.columns if ":hvac_heating_rate" in c]
            zone_temp_cols = [c for c in df.columns if ":zone_temp" in c]
            supply_temp_cols = [c for c in df.columns if ":supply_air_temp" in c]
            fan_kw_cols = [c for c in df.columns if "Fan:electric_power" in c]
            dx_kw_cols = [c for c in df.columns if "DX Cooling:electric_power" in c]
            furnace_kw_cols = [c for c in df.columns if "Furnace:electric_power" in c]
            cop_cols = [c for c in df.columns if ":cop_operating" in c]
            internal_cols = [c for c in df.columns if ":q_internal_conv" in c]
            solar_cols = [c for c in df.columns if ":transmitted_solar" in c]
            cond_cols = [c for c in df.columns if ":opaque_conduction" in c]

            def _sum_annual(c_list: list[str]) -> float | None:
                if not c_list:
                    return None
                vals = df[c_list].sum(axis=1).to_numpy(dtype=float) / 1000.0
                return float(vals.sum())

            def _mean(c_list: list[str]) -> float | None:
                if not c_list:
                    return None
                return float(df[c_list].mean(axis=1).mean())

            hvac["cooling_load_annual_kwh"] = _sum_annual(cooling_cols)
            hvac["heating_load_annual_kwh"] = _sum_annual(heating_cols)
            hvac["hvac_cooling_annual_kwh"] = _sum_annual(hvac_cool_cols)
            hvac["hvac_heating_annual_kwh"] = _sum_annual(hvac_heat_cols)
            hvac["fan_annual_kwh"] = _sum_annual(fan_kw_cols)
            hvac["dx_cooling_annual_kwh"] = _sum_annual(dx_kw_cols)
            hvac["furnace_annual_kwh"] = _sum_annual(furnace_kw_cols)
            hvac["mean_cop"] = _mean(cop_cols)
            hvac["internal_heat_mean_w"] = _mean(internal_cols)
            hvac["solar_gain_mean_w"] = _mean(solar_cols)
            hvac["conduction_mean_w"] = _mean(cond_cols)

            zones_info: dict[str, dict[str, Any]] = {}
            zone_names = sorted({c.split(":")[0] for c in cooling_cols})
            for zname in zone_names:
                z = {}
                tc = f"{zname}:zone_temp"
                if tc in df.columns:
                    vals = pd.to_numeric(df[tc], errors="coerce").dropna()
                    z["temp_mean"] = round(float(vals.mean()), 1)
                    z["temp_max"] = round(float(vals.max()), 1)
                    z["temp_min"] = round(float(vals.min()), 1)
                cc = f"{zname}:cooling_load"
                if cc in df.columns:
                    vals = pd.to_numeric(df[cc], errors="coerce").fillna(0)
                    z["cooling_load_annual_kwh"] = round(float(vals.sum() / 1000.0), 0)
                hc = f"{zname}:hvac_cooling_rate"
                if hc in df.columns:
                    vals = pd.to_numeric(df[hc], errors="coerce").fillna(0)
                    z["hvac_cooling_peak_w"] = round(float(vals.max()), 0)
                zones_info[zname] = z
            hvac["zones"] = zones_info

            return hvac

        base_hvac = _extract_hvac(base_df)
        scen_hvac = _extract_hvac(scen_df)

        def _delta(key: str) -> float | None:
            b = base_hvac.get(key)
            s = scen_hvac.get(key)
            if b is None or s is None:
                return None
            return round(s - b, 1)

        deltas = {
            "cooling_load_annual_kwh": _delta("cooling_load_annual_kwh"),
            "heating_load_annual_kwh": _delta("heating_load_annual_kwh"),
            "hvac_cooling_annual_kwh": _delta("hvac_cooling_annual_kwh"),
            "fan_annual_kwh": _delta("fan_annual_kwh"),
            "dx_cooling_annual_kwh": _delta("dx_cooling_annual_kwh"),
            "furnace_annual_kwh": _delta("furnace_annual_kwh"),
        }

        return {
            "status": "ok",
            "scenario": {
                "cooling_delta_degC": cooling_delta_degC,
                "lighting_ratio": lighting_ratio,
                "occupancy_ratio": occupancy_ratio,
                "equipment_ratio": equipment_ratio,
                "cop_ratio": cop_ratio,
            },
            "baseline_hvac": base_hvac,
            "scenario_hvac": scen_hvac,
            "delta": deltas,
            "total_delta_kwh": float(round(delta.sum(), 1)),
            "baseline_total_annual_kwh": float(round(baseline_kw.sum(), 1)),
            "scenario_total_annual_kwh": float(round(scenario_kw.sum(), 1)),
        }
