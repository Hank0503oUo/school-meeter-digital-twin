# -*- coding: utf-8 -*-
"""
輕量 EPW 天氣檔解析器。

只提取 Dashboard 需要的欄位：dry_bulb (°C) 和 rel_humidity (%)。
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path


# EPW 欄位定義 (EnergyPlus 文件)
_EPW_COLS = [
    "year", "month", "day", "hour", "minute", "data_source",
    "dry_bulb", "dew_point", "rel_humidity", "atm_pressure",
    "extraterr_horiz_rad", "extraterr_direct_normal_rad",
    "horiz_infrared_rad", "global_horiz_rad",
    "direct_normal_rad", "diffuse_horiz_rad",
    "global_horiz_illum", "direct_normal_illum", "diffuse_horiz_illum",
    "zenith_luminance", "wind_direction", "wind_speed",
    "total_sky_cover", "opaque_sky_cover", "visibility",
    "ceiling_height", "present_weather_observation",
    "present_weather_codes", "precipitable_water",
    "aerosol_optical_depth", "snow_depth",
    "days_since_last_snowfall", "albedo",
    "liquid_precip_depth", "liquid_precip_quantity",
]


def read_epw(path: str | Path) -> pd.DataFrame:
    """
    讀取 EPW 檔案並回傳 hourly DataFrame。

    Returns
    -------
    pd.DataFrame
        Index = DatetimeIndex (hourly)
        Columns = ['t_out', 'humidity'] (至少)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"EPW file not found: {path}")

    # Skip the first 8 header lines
    df = pd.read_csv(path, skiprows=8, header=None, names=_EPW_COLS[:35])

    # EPW hour is 1-24, convert to 0-23 for proper datetime
    df["hour"] = df["hour"] - 1

    # Build datetime index
    df["datetime"] = pd.to_datetime(
        df[["year", "month", "day", "hour"]].assign(minute=0)
    )
    df = df.set_index("datetime").sort_index()

    # Rename to match our feature convention
    df = df.rename(columns={
        "dry_bulb": "t_out",
        "rel_humidity": "humidity",
    })

    # Keep only the columns we need, ensure numeric
    result = df[["t_out", "humidity"]].apply(pd.to_numeric, errors="coerce")
    result = result.dropna()

    return result


def read_weather_csv(path: str | Path) -> pd.DataFrame:
    """
    讀取 CWBTP 格式的 CSV 氣象檔。

    CSV 欄位: yr, mo, da, hr, mi, ..., db (dry bulb), rh (rel humidity), ...

    Returns
    -------
    pd.DataFrame with DatetimeIndex + ['t_out', 'humidity']
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Weather CSV not found: {path}")

    df = pd.read_csv(path)

    # Build datetime — hr=1..24 in CSV, we need 0..23
    df["hour_adj"] = df["hr"].astype(int) - 1
    df["datetime"] = pd.to_datetime(
        df[["yr", "mo", "da"]].astype(int).assign(hour=df["hour_adj"])
        .rename(columns={"yr": "year", "mo": "month", "da": "day"})
    )
    df = df.set_index("datetime").sort_index()

    result = pd.DataFrame({
        "t_out": pd.to_numeric(df["db"], errors="coerce"),
        "humidity": pd.to_numeric(df["rh"], errors="coerce"),
    }, index=df.index)
    return result.dropna()


def read_weather(path: str | Path) -> pd.DataFrame:
    """
    統一天氣讀取介面：自動偵測 EPW 或 CSV 格式。

    Returns
    -------
    pd.DataFrame with DatetimeIndex + ['t_out', 'humidity']
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".epw":
        return read_epw(path)
    elif suffix == ".csv":
        return read_weather_csv(path)
    else:
        # Try CSV first, then EPW
        try:
            return read_weather_csv(path)
        except (ValueError, TypeError):
            return read_epw(path)


def list_available_weather(search_dir: str | Path) -> list[Path]:
    """列出目錄下可用的天氣檔案 (EPW + CSV)。"""
    search_dir = Path(search_dir)
    if not search_dir.exists():
        return []
    files = list(search_dir.glob("*.epw")) + list(search_dir.glob("*.csv"))
    return sorted(set(files))
