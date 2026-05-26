# -*- coding: utf-8 -*-
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.counterfactual import run_counterfactual
from src.map_builder import build_campus_map, merge_energy_geojson
from src.map_builder_impl import _decorate_visual_properties
from src.meter_matcher import extract_building_name, match_meters_to_buildings


def _square_polygon(lon: float, lat: float, size: float = 0.0001) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[
            [lon, lat],
            [lon + size, lat],
            [lon + size, lat + size],
            [lon, lat + size],
            [lon, lat],
        ]],
    }


def _feature(osm_id: int, name: str, lon: float, lat: float, levels: int = 3) -> dict:
    return {
        "type": "Feature",
        "geometry": _square_polygon(lon, lat),
        "properties": {
            "osm_id": osm_id,
            "name": name,
            "name_en": "",
            "building_type": "university",
            "levels": levels,
            "height": levels * 3.5,
            "addr": "",
            "operator": "",
        },
    }


class TestCounterfactual(unittest.TestCase):
    def test_counterfactual_returns_expected_shape_and_summary(self):
        baseline = np.full(24, 100.0)
        result = run_counterfactual(
            baseline,
            cooling_delta_degC=1.0,
            lighting_ratio=0.9,
            occupancy_ratio=0.95,
            equipment_ratio=1.0,
        )

        self.assertEqual(result.timeseries_new.shape, (24,))
        self.assertLess(result.delta_kwh, 0.0)

        summary = result.summary_dict()
        for key in [
            "delta_kwh",
            "delta_carbon_kg",
            "delta_ntd",
            "delta_pct",
            "equiv_trees",
            "label",
        ]:
            self.assertIn(key, summary)
        self.assertIsInstance(summary["delta_kwh"], float)

    def test_counterfactual_accepts_scalar_baseline(self):
        result = run_counterfactual(100.0, cooling_delta_degC=1.0)
        self.assertEqual(len(result.timeseries_base), 1)
        self.assertEqual(len(result.timeseries_new), 1)
    def test_counterfactual_with_real_physics_residual(self):
        """When physics_pred and residual_pred are provided, use them directly."""
        baseline = np.full(24, 100.0)
        physics = np.full(24, 75.0)
        residual = np.full(24, 25.0)
        result = run_counterfactual(
            baseline,
            cooling_delta_degC=1.0,
            physics_pred=physics,
            residual_pred=residual,
        )
        self.assertEqual(result.timeseries_new.shape, (24,))
        np.testing.assert_allclose(result.timeseries_res, residual)
        self.assertTrue(np.all(result.timeseries_base_phy >= 0))




class TestMatcherAndMapBuilder(unittest.TestCase):
    def test_extract_building_name(self):
        raw = "01B_P1_01化學館（MVCB）(高壓)"
        self.assertEqual(extract_building_name(raw), "化學館")

    def test_matcher_and_merge_handles_duplicate_meters(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            geojson_path = tmpdir / "ntu_buildings.geojson"
            summary_path = tmpdir / "summary.csv"
            map_path = tmpdir / "meter_map.csv"
            energy_path = tmpdir / "ntu_energy.geojson"

            geojson = {
                "type": "FeatureCollection",
                "features": [
                    _feature(1001, "化學館", 121.53, 25.01),
                    _feature(1002, "總圖書館", 121.531, 25.011),
                ],
            }
            geojson_path.write_text(json.dumps(geojson, ensure_ascii=False), encoding="utf-8")

            pd.DataFrame([
                {"meter_name": "01B_P1_01化學館（MVCB）(高壓)", "mean_kw": 100.0, "best_r2_oof": 0.80, "best_r_oof": 0.90, "best_cvrmse_oof": 12.0},
                {"meter_name": "01B_P1_01化學館（備援）(高壓)", "mean_kw": 50.0, "best_r2_oof": 0.60, "best_r_oof": 0.75, "best_cvrmse_oof": 20.0},
                {"meter_name": "01S_P1_01總圖書館HTM（高壓）", "mean_kw": 200.0, "best_r2_oof": 0.85, "best_r_oof": 0.92, "best_cvrmse_oof": 10.0},
            ]).to_csv(summary_path, index=False, encoding="utf-8")

            match_df = match_meters_to_buildings(
                geojson_path=geojson_path,
                summary_csv_path=summary_path,
                threshold_auto=80,
                threshold_review=55,
            )
            match_df.to_csv(map_path, index=False, encoding="utf-8")

            merged = merge_energy_geojson(
                geojson_path=geojson_path,
                match_csv_path=map_path,
                output_path=energy_path,
            )

            by_name = {f["properties"]["name"]: f["properties"] for f in merged["features"]}
            chem = by_name["化學館"]
            self.assertIn(chem["data_source"], {"Measured Meter", "PI-VD Inferred"})
            # 新策略：避免重複加總，總表/主表優先，未知重複電表採最大值
            self.assertAlmostEqual(chem["mean_kw"], 100.0, places=3)
            self.assertEqual(chem["meter_count"], 1)
            self.assertAlmostEqual(chem["annual_kwh"], 100.0 * 8760, places=3)
            self.assertIn(chem["aggregation_method"], {"max_single_meter", "max_single_submeter_no_merge", "sum_submeters"})
            self.assertIn("usage_profile", chem)
            self.assertIn(chem["usage_profile"], {"default", "hospital", "dorm"})

            campus_map = build_campus_map(energy_path, color_by="energy")
            html_path = tmpdir / "map.html"
            campus_map.to_html(str(html_path))
            self.assertTrue(html_path.exists())
            self.assertGreater(html_path.stat().st_size, 0)

    def test_no_data_buildings_are_muted_while_inferred_buildings_are_highlighted(self):
        geojson = {
            "type": "FeatureCollection",
            "features": [
                _feature(2001, "No Data Building", 121.53, 25.01, levels=5),
                _feature(2002, "Inferred Building", 121.531, 25.011, levels=5),
            ],
        }
        geojson["features"][1]["properties"].update({
            "data_source": "inferred",
            "energy_tier": "NORMAL",
            "mean_kw": 320.0,
            "eui": 180.0,
            "best_r2_oof": 0.62,
        })

        _decorate_visual_properties(geojson, show_virtual=True)
        no_data = geojson["features"][0]["properties"]
        inferred = geojson["features"][1]["properties"]

        self.assertLess(no_data["fill_color_energy"][3], inferred["fill_color_energy"][3])
        self.assertLess(no_data["outline_width"], inferred["outline_width"])
        self.assertEqual(no_data["display_height"], inferred["display_height"])

    def test_loading_buildings_are_rendered_in_gray(self):
        geojson = {
            "type": "FeatureCollection",
            "features": [
                _feature(3001, "Loading Building", 121.53, 25.01, levels=5),
            ],
        }
        geojson["features"][0]["properties"].update({
            "data_source": "loading",
            "has_meter_data": False,
            "energy_tier": "NO_DATA",
            "mean_kw": 500.0,
            "eui": 250.0,
            "best_r2_oof": 0.8,
        })

        _decorate_visual_properties(geojson, show_virtual=True)
        props = geojson["features"][0]["properties"]

        self.assertEqual(props["fill_color_energy"], [223, 227, 232, 62])
        self.assertEqual(props["fill_color_tier"], [223, 227, 232, 62])
        self.assertEqual(props["outline_color"], [185, 190, 196, 120])


if __name__ == "__main__":
    unittest.main()
