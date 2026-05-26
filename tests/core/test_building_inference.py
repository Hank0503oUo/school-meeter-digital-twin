import numpy as np
import pandas as pd

from src.building_inference import infer_all_buildings
from src.real_inference_engine import BuildingMetadataScaler, _DEFAULT_METADATA_UID


class DummyEngine:
    def __init__(self, metadata_scaler, campus_pred):
        self.metadata_scaler = metadata_scaler
        self._campus_pred = np.asarray(campus_pred, dtype=float)

    def predict(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        n = len(weather_df)
        if n != len(self._campus_pred):
            raise ValueError("weather length mismatch")
        return pd.DataFrame({"total_pred": self._campus_pred}, index=weather_df.index)


class TinyScaler:
    def __init__(self):
        self._uids = ["B1", "B2", "B3"]
        self._scalers = {"B1": 1.0, "B2": 1.0, "B3": 1.0}
        self._meta = {
            "B1": {"uid": "B1", "name": "B1", "area": 1.0, "floors": 1, "basement": 0, "buildType": "Academic Units"},
            "B2": {"uid": "B2", "name": "B2", "area": 1.0, "floors": 1, "basement": 0, "buildType": "Academic Units"},
            "B3": {"uid": "B3", "name": "B3", "area": 1.0, "floors": 1, "basement": 0, "buildType": "Academic Units"},
        }

    def list_uids(self):
        return list(self._uids)

    def get_scaler(self, uid: str) -> float:
        return float(self._scalers.get(uid, 1.0))

    def get_metadata(self, uid: str):
        return self._meta.get(uid)


def _weather(hours: int = 48) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=hours, freq="h")
    return pd.DataFrame(
        {
            "t_out": np.linspace(20, 30, hours),
            "humidity": np.linspace(55, 80, hours),
        },
        index=idx,
    )


def test_infer_all_buildings_returns_107_rows():
    scaler = BuildingMetadataScaler().load(_DEFAULT_METADATA_UID)
    weather = _weather(48)
    campus_pred = np.full(len(weather), 5000.0)
    engine = DummyEngine(scaler, campus_pred)

    uids = scaler.list_uids()
    meter_summary = pd.DataFrame(
        {
            "uid": [uids[0], uids[1], uids[2]],
            "mean_kw": [320.0, 150.0, 210.0],
            "best_r2_oof": [0.9, 0.8, 0.7],
        }
    )

    out = infer_all_buildings(engine, weather, meter_summary)

    assert len(out) == 107
    assert out["uid"].nunique() == 107
    assert set(["metered", "inferred"]).issubset(set(out["data_source"].unique()))


def test_inference_sum_matches_campus_total_without_meter_override():
    scaler = BuildingMetadataScaler().load(_DEFAULT_METADATA_UID)
    weather = _weather(72)
    campus_pred = np.linspace(4200.0, 5600.0, len(weather))
    engine = DummyEngine(scaler, campus_pred)

    out = infer_all_buildings(engine, weather, meter_summary=pd.DataFrame())
    ts_matrix = np.vstack(out["timeseries"].to_list())

    np.testing.assert_allclose(ts_matrix.sum(axis=0), campus_pred, rtol=1e-6, atol=1e-6)


def test_energy_tier_classification_high_normal_low():
    scaler = TinyScaler()
    weather = _weather(24)
    campus_pred = np.full(len(weather), 300.0)
    engine = DummyEngine(scaler, campus_pred)

    meter_summary = pd.DataFrame(
        {
            "uid": ["B1", "B2", "B3"],
            "mean_kw": [10.0, 100.0, 190.0],
        }
    )

    out = infer_all_buildings(engine, weather, meter_summary)
    tier_by_uid = dict(zip(out["uid"], out["energy_tier"]))

    assert tier_by_uid["B1"] == "LOW"
    assert tier_by_uid["B2"] == "NORMAL"
    assert tier_by_uid["B3"] == "HIGH"


def test_aggregate_meter_summary_uses_meter_building_map_hints(tmp_path):
    from src.building_inference import aggregate_meter_summary_by_uid

    meter_summary = pd.DataFrame(
        {
            "meter_name": ["05E_P1_01食科站總表（高壓）"],
            "mean_kw": [180.0],
        }
    )

    loop_df = pd.DataFrame(
        {
            "迴路編號": ["99Z_P1_01"],  # intentionally non-matching meter id
            "uid": ["AT9999"],
            "館舍": ["測試館"],
            "分區編號": ["X"],
        }
    )
    loop_path = tmp_path / "metadata_loop.csv"
    loop_df.to_csv(loop_path, index=False, encoding="utf-8")

    uid_df = pd.DataFrame(
        {
            "uid": ["AT5002"],
            "name": ["食品科技館"],
        }
    )
    uid_path = tmp_path / "metadata_uid.csv"
    uid_df.to_csv(uid_path, index=False, encoding="utf-8")

    meter_map_df = pd.DataFrame(
        {
            "meter_name": ["05E_P1_01食科站總表（高壓）"],
            "osm_name": ["食品科技館"],
            "building_name_extracted": ["食科站"],
        }
    )
    meter_map_path = tmp_path / "meter_building_map.csv"
    meter_map_df.to_csv(meter_map_path, index=False, encoding="utf-8")

    out = aggregate_meter_summary_by_uid(
        meter_summary=meter_summary,
        metadata_loop_path=loop_path,
        metadata_uid_path=uid_path,
        meter_building_map_path=meter_map_path,
    )

    assert len(out) == 1
    assert out.iloc[0]["uid"] == "AT5002"
    assert float(out.iloc[0]["mean_kw"]) == 180.0


def test_aggregate_meter_summary_respects_manual_overrides(tmp_path):
    from src.building_inference import aggregate_meter_summary_by_uid

    meter_summary = pd.DataFrame(
        {
            "meter_name": ["07A_P1_03霖澤館饋線"],
            "mean_kw": [55.0],
        }
    )

    loop_df = pd.DataFrame(
        {
            "迴路編號": ["99Z_P1_01"],  # intentionally non-matching
            "uid": ["AT9999"],
            "館舍": ["測試館"],
            "分區編號": ["X"],
        }
    )
    loop_path = tmp_path / "metadata_loop.csv"
    loop_df.to_csv(loop_path, index=False, encoding="utf-8")

    uid_df = pd.DataFrame(
        {
            "uid": ["AT1040", "AT5002"],
            "name": ["化學館", "食品科技館"],
        }
    )
    uid_path = tmp_path / "metadata_uid.csv"
    uid_df.to_csv(uid_path, index=False, encoding="utf-8")

    meter_map_df = pd.DataFrame(
        {
            "meter_name": ["07A_P1_03霖澤館饋線"],
            "osm_name": ["食品科技館"],  # this would resolve to AT5002 if no override
            "building_name_extracted": ["霖澤館"],
        }
    )
    meter_map_path = tmp_path / "meter_building_map.csv"
    meter_map_df.to_csv(meter_map_path, index=False, encoding="utf-8")

    overrides_df = pd.DataFrame(
        {
            "enabled": [1],
            "meter_id": ["07A_P1_03"],
            "meter_name": [""],
            "uid": ["AT1040"],
            "note": ["manual override test"],
        }
    )
    overrides_path = tmp_path / "meter_uid_overrides.csv"
    overrides_df.to_csv(overrides_path, index=False, encoding="utf-8")

    out = aggregate_meter_summary_by_uid(
        meter_summary=meter_summary,
        metadata_loop_path=loop_path,
        metadata_uid_path=uid_path,
        meter_building_map_path=meter_map_path,
        meter_uid_overrides_path=overrides_path,
    )

    assert len(out) == 1
    assert out.iloc[0]["uid"] == "AT1040"
    assert float(out.iloc[0]["mean_kw"]) == 55.0


def test_aggregate_meter_summary_uses_parenthetical_name_hints(tmp_path):
    from src.building_inference import aggregate_meter_summary_by_uid

    meter_summary = pd.DataFrame(
        {
            "meter_name": ["02H_P1_01二區H站總用電（森林館站）(高壓)"],
            "mean_kw": [88.0],
        }
    )

    loop_df = pd.DataFrame(
        {
            "迴路編號": ["99Z_P1_01"],  # intentionally non-matching meter id
            "uid": ["AT9999"],
            "館舍": ["測試館"],
            "分區編號": ["X"],
        }
    )
    loop_path = tmp_path / "metadata_loop.csv"
    loop_df.to_csv(loop_path, index=False, encoding="utf-8")

    uid_df = pd.DataFrame(
        {
            "uid": ["AT2042"],
            "name": ["森林館"],
        }
    )
    uid_path = tmp_path / "metadata_uid.csv"
    uid_df.to_csv(uid_path, index=False, encoding="utf-8")

    # No external meter hints: force resolver to rely on meter_name candidates.
    meter_map_path = tmp_path / "meter_building_map.csv"

    out = aggregate_meter_summary_by_uid(
        meter_summary=meter_summary,
        metadata_loop_path=loop_path,
        metadata_uid_path=uid_path,
        meter_building_map_path=meter_map_path,
    )

    assert len(out) == 1
    assert out.iloc[0]["uid"] == "AT2042"
    assert float(out.iloc[0]["mean_kw"]) == 88.0
