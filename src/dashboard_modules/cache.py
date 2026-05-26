from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable

YEARLY_CACHE_MAX_ENTRIES = 4


def bounded_cache_get(
    cache: OrderedDict[int, Any],
    key: int,
    loader: Callable[[], Any],
    max_entries: int = YEARLY_CACHE_MAX_ENTRIES,
):
    """Get or load an item from an OrderedDict cache with simple LRU eviction."""
    if key in cache:
        cache.move_to_end(key)
        return cache[key]

    value = loader()
    cache[key] = value
    cache.move_to_end(key)

    while len(cache) > max_entries:
        cache.popitem(last=False)

    return value
