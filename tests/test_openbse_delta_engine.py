# -*- coding: utf-8 -*-
"""Unit tests for OpenBSEDeltaEngine (mocked subprocess)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from subprocess import CompletedProcess
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(tmp_path):
    """
    Build a minimal OpenBSEDeltaEngine whose YAML paths, modifier, and extractor
    are real but whose subprocess call is mocked to return a flat 8760-row CSV.
    """
    import pandas as pd
    from src.openbse_counterfactual import OpenBSEDeltaEngine

    # Create a tiny stand-in YAML (just needs to be parseable)
    base_yaml = tmp_path / "base.yaml"
    base_yaml.write_text(
        "simulation:\n  start_month: 1\n  end_month: 12\n  timesteps_per_hour: 1\n"
        "weather_files:\n  - dummy.epw\n"
        "thermostats:\n  - cooling_setpoint: 23.9\n"
        "lights:\n  - power: 522.0\n"
        "equipment:\n  - power: 240.0\n"
        "people:\n  - count: 13.0\n"
        "air_loops:\n  - equipment:\n    - type: fan\n    - type: heating_coil\n    - type: cooling_coil\n      cop: 3.5\n",
        encoding="utf-8",
    )

    mapping_yaml = tmp_path / "mapping.yaml"
    mapping_yaml.write_text(
        "set_point: [thermostats, 0, cooling_setpoint]\n"
        "lighting_density: [lights, 0, power]\n"
        "equipment_density: [equipment, 0, power]\n"
        "people_density: [people, 0, count]\n"
        "cop: [air_loops, 0, equipment, 2, cop]\n",
        encoding="utf-8",
    )

    engine = OpenBSEDeltaEngine.__new__(OpenBSEDeltaEngine)
    engine.openbse_exe = Path("D:/openbse_bin/openbse.exe")
    engine.weather_file = Path("dummy.epw")
    engine.timeout = 30
    engine.base_yaml = base_yaml

    # Load real modifier + extractor
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "idf_r2_optimizer" / "src"))
    from backends.openbse_yaml_modifier import OpenBSEYamlModifier
    from backends.openbse_output_extractor import OpenBSEOutputExtractor
    import yaml
    with mapping_yaml.open() as f:
        raw_mapping = yaml.safe_load(f)
    engine._modifier = OpenBSEYamlModifier(base_yaml, raw_mapping)
    engine._extractor = OpenBSEOutputExtractor()
    engine._baseline = {
        "set_point": 23.9,
        "lighting_density": 522.0,
        "equipment_density": 240.0,
        "people_density": 13.0,
        "cop": 3.5,
    }
    return engine


def _write_fake_csv(path: Path, value_kw: float):
    """Write a fake OpenBSE result CSV: 8760 rows, one column, value in W."""
    import pandas as pd
    df = pd.DataFrame({"RTU DX Cooling:electric_power [-]": [value_kw * 1000.0] * 8760})
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_baseline_values_read(tmp_path):
    engine = _make_engine(tmp_path)
    assert abs(engine._baseline["set_point"] - 23.9) < 0.01
    assert abs(engine._baseline["lighting_density"] - 522.0) < 0.01


def test_compute_delta_zero_change(tmp_path):
    engine = _make_engine(tmp_path)

    call_count = 0

    def fake_run(yaml_path, output_csv):
        nonlocal call_count
        _write_fake_csv(output_csv, 100.0)
        call_count += 1
        return engine._extractor.extract(output_csv)

    engine._run_openbse = fake_run
    delta, scen, base = engine.compute_delta()

    assert call_count == 2
    assert delta.shape == (8760,)
    assert np.allclose(delta, 0.0)
    assert np.allclose(base, 100.0)
    assert np.allclose(scen, 100.0)


def test_compute_delta_positive(tmp_path):
    engine = _make_engine(tmp_path)
    call_count = [0]

    def fake_run(yaml_path, output_csv):
        val = 100.0 if call_count[0] == 0 else 130.0
        _write_fake_csv(output_csv, val)
        call_count[0] += 1
        return engine._extractor.extract(output_csv)

    engine._run_openbse = fake_run
    delta, scen, base = engine.compute_delta(lighting_ratio=1.3)

    assert np.allclose(delta, 30.0, atol=1.0)
    assert np.allclose(scen - base, 30.0, atol=1.0)


def test_baseline_values_property(tmp_path):
    engine = _make_engine(tmp_path)
    vals = engine.baseline_values
    assert "set_point" in vals
    assert "cop" in vals
    assert isinstance(vals["cop"], float)


def test_resolve_required_file_prefers_env_override(tmp_path, monkeypatch):
    from src.openbse_counterfactual import _resolve_required_file

    env_file = tmp_path / "env.yaml"
    fallback_file = tmp_path / "fallback.yaml"
    env_file.write_text("env", encoding="utf-8")
    fallback_file.write_text("fallback", encoding="utf-8")
    monkeypatch.setenv("OPENBSE_TEST_FILE", str(env_file))

    resolved = _resolve_required_file("Test file", "OPENBSE_TEST_FILE", fallback_file)

    assert resolved == env_file.resolve()


def test_resolve_required_file_raises_clear_error(tmp_path, monkeypatch):
    from src.openbse_counterfactual import _resolve_required_file

    monkeypatch.delenv("OPENBSE_TEST_FILE", raising=False)
    missing = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Set OPENBSE_TEST_FILE"):
        _resolve_required_file("Test file", "OPENBSE_TEST_FILE", missing)


def test_run_openbse_decodes_non_utf8_stderr(tmp_path):
    from src.openbse_counterfactual import OpenBSEDeltaEngine

    engine = OpenBSEDeltaEngine.__new__(OpenBSEDeltaEngine)
    engine.openbse_exe = Path("openbse.exe")
    engine.weather_file = Path("dummy.epw")
    engine.timeout = 30
    engine._extractor = MagicMock()

    with patch("src.openbse_counterfactual.subprocess.run", return_value=CompletedProcess(
        args=["openbse.exe"],
        returncode=1,
        stdout=b"",
        stderr=b"\xe2\x98\x83 failure",
    )):
        with pytest.raises(RuntimeError, match="OpenBSE failed"):
            engine._run_openbse(tmp_path / "input.yaml", tmp_path / "output.csv")
