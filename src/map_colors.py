from __future__ import annotations

import numpy as np


EUI_COLOR_EXPR = [
    "interpolate",
    ["linear"],
    ["get", "eui"],
    0,
    "#5e9e82",
    80,
    "#8bb86a",
    150,
    "#ccb147",
    250,
    "#c17a50",
    400,
    "#9e5578",
]

ENERGY_COLOR_EXPR = [
    "interpolate",
    ["linear"],
    ["get", "mean_kw"],
    0,
    "#5e9e82",
    200,
    "#8bb86a",
    500,
    "#ccb147",
    1000,
    "#c17a50",
    2000,
    "#9e5578",
]

R2_COLOR_EXPR = [
    "interpolate",
    ["linear"],
    ["get", "best_r2_oof"],
    -0.5,
    "#c17a50",
    0.3,
    "#ccb147",
    0.6,
    "#8bb86a",
    0.8,
    "#5e9e82",
    0.95,
    "#4eb88a",
]


def _kw_to_rgb(kw: float) -> list[int]:
    if kw < 200:
        return [94, 158, 130, 190]
    if kw < 500:
        return [139, 184, 106, 200]
    if kw < 1000:
        return [204, 177, 71, 205]
    if kw < 2000:
        return [193, 122, 80, 210]
    return [158, 85, 120, 215]


def _eui_to_rgb(eui: float) -> list[int]:
    if eui < 80:
        return [94, 158, 130, 190]
    if eui < 150:
        return [139, 184, 106, 200]
    if eui < 250:
        return [204, 177, 71, 205]
    if eui < 400:
        return [193, 122, 80, 210]
    return [158, 85, 120, 215]


def _r2_to_rgb(r2: float) -> list[int]:
    if r2 < 0.3:
        return [193, 122, 80, 210]
    if r2 < 0.6:
        return [204, 177, 71, 205]
    if r2 < 0.8:
        return [139, 184, 106, 200]
    if r2 < 0.95:
        return [94, 158, 130, 200]
    return [78, 184, 138, 210]


def _dci_to_rgb(dci: float) -> list[int]:
    x = float(np.clip(dci, 0.0, 100.0))
    if x < 40:
        return [193, 122, 80, 210]
    if x < 70:
        return [204, 177, 71, 210]
    return [94, 158, 130, 210]


def _tier_to_rgb(tier: str) -> list[int]:
    key = str(tier or "").strip().upper()
    if key == "HIGH":
        return [215, 48, 39, 220]
    if key == "LOW":
        return [26, 152, 80, 220]
    return [240, 196, 25, 220]
