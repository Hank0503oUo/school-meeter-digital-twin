from __future__ import annotations

from typing import Callable

import pandas as pd

from src.dashboard_state import CampusState


def reload_campus_state(state: CampusState, campus_id: str, campus_cfg: object | None = None) -> CampusState:
    """
    Update basic campus identity fields in a shared state object.
    Heavy loading remains in dashboard implementation code.
    """
    state.campus_id = str(campus_id or "").strip().lower()
    state.campus_cfg = campus_cfg
    state.campus_name = getattr(campus_cfg, "campus_name", state.campus_id.upper())
    return state


def clear_yearly_cache(state: CampusState) -> None:
    """Clear all per-year caches when campus context changes."""
    state.yearly_inference_cache.clear()
    state.yearly_stats_cache.clear()
    state.yearly_geojson_cache.clear()


def get_yearly_inference(
    state: CampusState,
    year: int,
    loader: Callable[[int], pd.DataFrame],
) -> pd.DataFrame:
    """
    Lazy-load yearly inference data with in-memory caching.
    """
    y = int(year)
    if y not in state.yearly_inference_cache:
        state.yearly_inference_cache[y] = loader(y)
    return state.yearly_inference_cache[y]
