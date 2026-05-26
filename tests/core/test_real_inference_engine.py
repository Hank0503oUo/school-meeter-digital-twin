# -*- coding: utf-8 -*-
"""Tests for the PI-VD inference engine."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Skip all if model files are not present
_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_HAS_MODELS = (
    (_MODELS_DIR / "v10_boot_ensemble.pkl").exists()
    and (_MODELS_DIR / "v10_boot_dataset_2017.csv").exists()
    and (_MODELS_DIR / "best_tow_adaptive_v9.yaml").exists()
)


@unittest.skipUnless(_HAS_MODELS, "Model files not found in demo/models/")
class TestPIVDEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.real_inference_engine import PIVDEngine
        cls.engine = PIVDEngine.from_defaults()

    def test_engine_is_ready(self):
        self.assertTrue(self.engine.is_ready)

    def test_surrogate_has_5_models(self):
        self.assertEqual(len(self.engine.surrogate.models), 5)
        for key in ["A", "B", "C", "D", "E"]:
            self.assertIn(key, self.engine.surrogate.models)

    def test_v10_boot_has_models(self):
        self.assertGreater(len(self.engine.v10_boot.models), 0)

    def test_predict_with_synthetic_weather(self):
        """Engine should produce predictions from synthetic weather data."""
        n = 48  # 2 days
        idx = pd.date_range("2017-07-01", periods=n, freq="h")
        weather = pd.DataFrame({
            "t_out": np.random.uniform(25, 35, n),
            "humidity": np.random.uniform(60, 85, n),
        }, index=idx)

        result = self.engine.predict(weather)
        self.assertEqual(len(result), n)
        self.assertIn("physics_pred", result.columns)
        self.assertIn("residual_pred", result.columns)
        self.assertIn("residual_std", result.columns)
        self.assertIn("total_pred", result.columns)
        # All values should be finite
        self.assertTrue(np.isfinite(result["total_pred"]).all())

    def test_physics_surrogate_accuracy(self):
        """Surrogate predictions should correlate >0.99 with cached data."""
        dataset = pd.read_csv(
            _MODELS_DIR / "v10_boot_dataset_2017.csv",
            index_col=0, parse_dates=True,
        )
        from src.real_inference_engine import _SURROGATE_INPUT_COLS
        features = dataset[_SURROGATE_INPUT_COLS]
        sim_pred = self.engine.surrogate.predict(features)

        for key in ["A", "B", "C", "D", "E"]:
            col = f"sim_{key}"
            corr = np.corrcoef(dataset[col].values, sim_pred[col].values)[0, 1]
            self.assertGreater(corr, 0.99, f"Surrogate {col} R={corr:.4f} < 0.99")


@unittest.skipUnless(
    (_MODELS_DIR / "weather").exists() and any((_MODELS_DIR / "weather").glob("*2017*")),
    "Weather files not found",
)
class TestEPWReader(unittest.TestCase):

    def test_read_weather_csv(self):
        from src.epw_reader import read_weather
        csv_files = list((_MODELS_DIR / "weather").glob("*2017*.csv"))
        if csv_files:
            weather = read_weather(csv_files[0])
            self.assertIn("t_out", weather.columns)
            self.assertIn("humidity", weather.columns)
            self.assertGreater(len(weather), 1000)

    def test_read_weather_epw(self):
        from src.epw_reader import read_weather
        epw_files = list((_MODELS_DIR / "weather").glob("*2017*.epw"))
        if epw_files:
            weather = read_weather(epw_files[0])
            self.assertIn("t_out", weather.columns)
            self.assertGreater(len(weather), 1000)


def test_initialize_fails_fast_when_boot_model_is_missing(tmp_path, monkeypatch):
    from src.real_inference_engine import PIVDEngine

    engine = PIVDEngine()

    monkeypatch.setattr(
        engine.surrogate,
        "train",
        lambda *args, **kwargs: engine.surrogate,
    )
    monkeypatch.setattr(
        engine.v9_weights,
        "load",
        lambda *args, **kwargs: engine.v9_weights,
    )
    monkeypatch.setattr(
        engine.v9_weights,
        "train_from_dataset",
        lambda *args, **kwargs: engine.v9_weights,
    )
    monkeypatch.setattr(
        engine.metadata_scaler,
        "load",
        lambda *args, **kwargs: engine.metadata_scaler,
    )

    with pytest.raises(FileNotFoundError):
        engine.initialize(
            dataset_path=tmp_path / "dataset.csv",
            v9_yaml_path=tmp_path / "weights.yaml",
            boot_pkl_path=tmp_path / "missing.pkl",
            metadata_csv=tmp_path / "metadata.csv",
        )

    assert engine.is_ready is False


if __name__ == "__main__":
    unittest.main()
