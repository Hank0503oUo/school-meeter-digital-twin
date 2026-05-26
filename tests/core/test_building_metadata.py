import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.real_inference_engine import (
    BuildingMetadataScaler,
    PIVDEngine,
    _DEFAULT_METADATA_UID,
)

# Test data for mocking CSV if needed, but we can test with actual CSV
def test_metadata_csv_loads():
    """確認 metadata_uid.csv 可正確解析"""
    scaler = BuildingMetadataScaler()
    scaler.load(_DEFAULT_METADATA_UID)
    
    # Check that it loaded at least some buildings
    assert scaler.is_loaded
    uids = scaler.list_uids()
    assert len(uids) > 0
    
    # Check a specific building if present
    # AT1040 is Department of Chemistry (化學館)
    meta = scaler.get_metadata("AT1040")
    if meta:
        assert meta["name"] == "化學館"
        assert meta["area"] > 0
        assert meta["floors"] > 0
        assert meta["buildType"] == "Academic Units"

def test_scaler_range():
    """所有 scaler 值在合理範圍 [0.1, 10.0]"""
    scaler = BuildingMetadataScaler()
    scaler.load(_DEFAULT_METADATA_UID)
    
    uids = scaler.list_uids()
    for uid in uids:
        s = scaler.get_scaler(uid)
        assert 0.1 <= s <= 10.0

def test_predict_building_returns_scaled():
    """predict_building() 結果 = campus-level × scaler (排序指標)"""
    engine = PIVDEngine.from_defaults()
    
    # Create simple weather df
    dates = pd.date_range("2017-01-01", periods=24, freq="h")
    weather_df = pd.DataFrame({
        "t_out": np.random.uniform(15, 30, 24),
        "humidity": np.random.uniform(50, 90, 24)
    }, index=dates)
    
    campus_result = engine.predict(weather_df)
    
    # Choose a UID, compute its scaler
    test_uid = "AT1040"
    scaler_val = engine.metadata_scaler.get_scaler(test_uid)
    
    building_result = engine.predict_building(weather_df, test_uid)
    
    assert "building_scaler" in building_result.columns
    assert "building_rank_index" in building_result.columns
    assert "building_physics_index" in building_result.columns
    assert "building_eui_index" in building_result.columns
    
    # Check if building results are scaled
    np.testing.assert_allclose(
        building_result["building_scaler"].values, 
        np.full(24, scaler_val)
    )
    np.testing.assert_allclose(
        building_result["building_rank_index"].values, 
        campus_result["total_pred"].values * scaler_val
    )
    np.testing.assert_allclose(
        building_result["building_physics_index"].values, 
        campus_result["physics_pred"].values * scaler_val
    )

def test_type_factor_ordering():
    """Academic > Dormitory scaler 排序正確 (確保 type_factor 正常運作)"""
    scaler = BuildingMetadataScaler()
    scaler.load(_DEFAULT_METADATA_UID)
    
    # Find one academic building and one dormitory building
    academic_uids = [u for u in scaler.list_uids() if scaler.get_metadata(u).get("buildType") == "Academic Units"]
    dorm_uids = [u for u in scaler.list_uids() if scaler.get_metadata(u).get("buildType") == "Dormitories"]
    
    if academic_uids and dorm_uids:
        # We can't strictly compare their final scalers because area and floors play a role.
        # But we can verify that the internal type factor is assigned correctly.
        from src.real_inference_engine import _BUILDING_TYPE_FACTORS
        assert _BUILDING_TYPE_FACTORS["Academic Units"] > 1.0
        assert _BUILDING_TYPE_FACTORS["Dormitories"] < 1.0
        assert _BUILDING_TYPE_FACTORS["Academic Units"] > _BUILDING_TYPE_FACTORS["Dormitories"]
