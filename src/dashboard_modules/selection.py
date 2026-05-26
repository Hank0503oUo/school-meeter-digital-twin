from __future__ import annotations

import re


def coerce_selected_uid(selected_uid: str | None) -> str:
    raw = str(selected_uid or "").strip()
    if not raw:
        return ""
    match = re.search(r"\(([^()]+)\)\s*$", raw)
    if match:
        return match.group(1).strip()
    return raw
