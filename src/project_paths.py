from __future__ import annotations

from pathlib import Path
from typing import Iterator


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_root() -> Path:
    return _PROJECT_ROOT


def resolve_project_path(path: str | Path, *, base: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (base or _PROJECT_ROOT) / candidate
    return candidate.resolve()


def data_dir(*parts: str) -> Path:
    return _PROJECT_ROOT.joinpath("data", *parts)


def models_dir(*parts: str) -> Path:
    return _PROJECT_ROOT.joinpath("models", *parts)


def config_dir(*parts: str) -> Path:
    return _PROJECT_ROOT.joinpath("config", *parts)


def campus_data_dir(campus_id: str, *parts: str) -> Path:
    return data_dir(str(campus_id).upper(), *parts)


def iter_ancestor_dirs(start: str | Path) -> Iterator[Path]:
    current = Path(start).resolve()
    yield current
    yield from current.parents
