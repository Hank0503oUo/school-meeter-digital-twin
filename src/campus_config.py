from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_DEMO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CAMPUSES_DIR = _DEMO_ROOT / "campuses"
_DEFAULT_CACHE_DIR = _DEMO_ROOT / "data" / "cache"

# Keys required to run the full PI-VD + dashboard workflow.
_DEFAULT_REQUIRED_KEYS = (
    "buildings_geojson",
    "energy_geojson",
    "metadata_uid",
    "metadata_loop",
    "meter_csv",
    "v9_yaml",
    "v10_dataset",
    "v10_ensemble",
    "v12_summary",
    "weather_dir",
)


class CampusConfigError(RuntimeError):
    """Raised when a campus config cannot be loaded or validated."""


@dataclass
class CampusConfig:
    campus_id: str
    campus_name: str
    campus_name_en: str
    map_lat: float
    map_lon: float
    map_zoom: int
    map_bounds: dict[str, float]
    paths: dict[str, Path]
    root_dir: Path

    @classmethod
    def load(
        cls,
        campus_id: str,
        campuses_root: Path | str = _DEFAULT_CAMPUSES_DIR,
    ) -> "CampusConfig":
        cid = normalize_campus_id(campus_id)

        campuses_root = Path(campuses_root)
        cfg_path = campuses_root / cid / "config.yaml"
        if not cfg_path.exists():
            raise CampusConfigError(f"Campus config not found: {cfg_path}")

        try:
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            raise CampusConfigError(f"Failed to parse campus config: {cfg_path}") from exc

        root_dir = cfg_path.parent
        map_cfg = raw.get("map", {}) or {}
        bounds_cfg = map_cfg.get("bounds", {}) or {}
        raw_paths = raw.get("paths", {}) or {}
        resolved_paths = {
            str(k): (root_dir / str(v)).resolve()
            for k, v in raw_paths.items()
            if str(k).strip() and str(v).strip()
        }

        return cls(
            campus_id=str(raw.get("campus_id", cid)).strip().lower() or cid,
            campus_name=str(raw.get("campus_name", cid.upper())).strip() or cid.upper(),
            campus_name_en=str(raw.get("campus_name_en", cid.upper())).strip() or cid.upper(),
            map_lat=float(map_cfg.get("latitude", 25.0175)),
            map_lon=float(map_cfg.get("longitude", 121.5375)),
            map_zoom=int(map_cfg.get("zoom", 15)),
            map_bounds={
                "south": float(bounds_cfg.get("south", 0.0)),
                "north": float(bounds_cfg.get("north", 0.0)),
                "west": float(bounds_cfg.get("west", 0.0)),
                "east": float(bounds_cfg.get("east", 0.0)),
            },
            paths=resolved_paths,
            root_dir=root_dir.resolve(),
        )

    @classmethod
    def list_available(
        cls,
        campuses_root: Path | str = _DEFAULT_CAMPUSES_DIR,
    ) -> list[str]:
        campuses_root = Path(campuses_root)
        if not campuses_root.exists():
            return []

        out: list[str] = []
        for d in sorted(campuses_root.iterdir()):
            if not d.is_dir():
                continue
            if (d / "config.yaml").exists():
                out.append(d.name.lower())
        return out

    def get_path(self, key: str, default: Path | None = None) -> Path | None:
        return self.paths.get(str(key), default)

    def missing_required_paths(self, required_keys: tuple[str, ...] | None = None) -> list[str]:
        keys = required_keys or _DEFAULT_REQUIRED_KEYS
        missing: list[str] = []
        for key in keys:
            p = self.paths.get(key)
            if p is None:
                missing.append(key)
            elif key.endswith("_dir"):
                if not p.exists() or not p.is_dir():
                    missing.append(key)
            elif not p.exists():
                missing.append(key)
        return missing

    def is_data_ready(self, required_keys: tuple[str, ...] | None = None) -> bool:
        return len(self.missing_required_paths(required_keys=required_keys)) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "campus_id": self.campus_id,
            "campus_name": self.campus_name,
            "campus_name_en": self.campus_name_en,
            "map": {
                "latitude": self.map_lat,
                "longitude": self.map_lon,
                "zoom": self.map_zoom,
                "bounds": self.map_bounds,
            },
            "paths": {k: str(v) for k, v in self.paths.items()},
        }


def normalize_campus_id(campus_id: str) -> str:
    cid = str(campus_id or "").strip().lower()
    if not cid:
        raise CampusConfigError("campus_id must not be empty")
    return cid


def inference_cache_dir(
    campus_id: str,
    cache_root: Path | str = _DEFAULT_CACHE_DIR,
) -> Path:
    return Path(cache_root) / normalize_campus_id(campus_id)


def inference_cache_path(
    campus_id: str,
    year: int,
    cache_root: Path | str = _DEFAULT_CACHE_DIR,
) -> Path:
    return inference_cache_dir(campus_id, cache_root=cache_root) / f"inference_cache_{int(year)}.parquet"


def inference_cache_path_candidates(
    campus_id: str,
    year: int,
    cache_root: Path | str = _DEFAULT_CACHE_DIR,
) -> list[Path]:
    canonical = inference_cache_path(campus_id, year=year, cache_root=cache_root)
    legacy_upper = canonical.parent.parent / canonical.parent.name.upper() / canonical.name
    if str(legacy_upper) == str(canonical):
        return [canonical]
    return [canonical, legacy_upper]
