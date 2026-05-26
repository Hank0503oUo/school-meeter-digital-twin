# -*- coding: utf-8 -*-

from pathlib import Path

from src.campus_config import (
    CampusConfig,
    inference_cache_path,
    inference_cache_path_candidates,
    normalize_campus_id,
)


def test_list_available_campuses_contains_ntu_and_ncu():
    campuses = CampusConfig.list_available()
    assert "ntu" in campuses
    assert "ncu" in campuses


def test_load_ntu_config_and_paths_exist():
    cfg = CampusConfig.load("ntu")
    assert cfg.campus_id == "ntu"
    assert cfg.campus_name
    assert cfg.get_path("energy_geojson") is not None
    assert cfg.get_path("energy_geojson").exists()
    assert cfg.get_path("metadata_uid").exists()


def test_data_ready_flags_for_ntu_and_ncu():
    ntu = CampusConfig.load("ntu")
    ncu = CampusConfig.load("ncu")

    assert ntu.is_data_ready() is True
    assert ncu.is_data_ready() is False
    assert len(ncu.missing_required_paths()) > 0


def test_paths_are_resolved_to_absolute_paths():
    cfg = CampusConfig.load("ntu")
    energy = cfg.get_path("energy_geojson")
    assert isinstance(energy, Path)
    assert energy.is_absolute()


def test_normalize_campus_id_lowercases_and_strips():
    assert normalize_campus_id(" NTU ") == "ntu"


def test_inference_cache_helpers_support_canonical_and_legacy_paths():
    canonical = inference_cache_path("NTU", 2017)
    candidates = inference_cache_path_candidates("NTU", 2017)

    assert canonical.name == "inference_cache_2017.parquet"
    assert canonical.parent.name == "ntu"
    assert candidates[0] == canonical
    assert candidates[1].parent.name == "NTU"
