"""
Fetch a specific calendar year of Taipei hourly weather via meteostat.

Usage:
    python scripts/ncu_cleaning/fetch_taipei_year_weather.py --year 2020
    ...
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
from meteostat import Hourly, Stations

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "models" / "weather"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAT, LON = 25.04, 121.51


def is_leap(y: int) -> bool:
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


def fetch(year: int):
    print(f"[1/3] Selecting station near ({LAT}, {LON}) for year {year}…")
    stations = Stations().nearby(LAT, LON).fetch(5)
    station_id = stations.index[0]
    name = stations.iloc[0]["name"]
    print(f"  station: {station_id} ({name})")
    print(f"[2/3] Fetching hourly data…")
    df = Hourly(station_id, dt.datetime(year, 1, 1, 0),
                dt.datetime(year, 12, 31, 23)).fetch()
    expected = (366 if is_leap(year) else 365) * 24
    print(f"  rows: {len(df)} (expected {expected})")
    return df, station_id, name


def to_demo_csv(df, year):
    full_idx = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="h")
    df = df.reindex(full_idx)
    for col in ("temp", "rhum", "pres"):
        if col in df.columns:
            df[col] = df[col].ffill(limit=3).interpolate(method="time", limit_direction="both")
    return pd.DataFrame({
        "yr": full_idx.year,
        "mo": full_idx.month,
        "da": full_idx.day,
        "hr": full_idx.hour + 1,
        "mi": 60,
        "etr": 0.0, "etrn": 0.0, "ghi": 0.0, "dni": 0.0, "dhi": 0.0,
        "tcld": 5,
        "db": df["temp"].round(1).values,
        "dp": (df["temp"].fillna(0) - 5).round(1).values,
        "rh": df["rhum"].round(0).fillna(70).astype(int).values,
        "ps": (df["pres"].fillna(1013.0)).round(1).values,
        "wdir": df.get("wdir", pd.Series(0, index=df.index)).fillna(0).astype(int).values,
        "wspd": df.get("wspd", pd.Series(2.0, index=df.index)).fillna(2.0).round(1).values,
        "hvis": -9900, "ch": -9900, "lpd": 0.0, "lpq": 0.0,
    })


def write_epw(out_csv_df, year, station_id, name):
    epw_path = OUT_DIR / f"CWBTP_{year}.epw"
    header = [
        f"LOCATION,{name},Taiwan,Taiwan,Custom-{station_id},{station_id},{LAT},{LON},8.0,5.0",
        "DESIGN CONDITIONS,0",
        "TYPICAL/EXTREME PERIODS,0",
        "GROUND TEMPERATURES,0",
        "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
        f"COMMENTS 1,Synthesized {year} from Meteostat NOAA ISD station {station_id}",
        "COMMENTS 2,Proxy for Chungli (NCU); only db/rh used by epw_reader",
        f"DATA PERIODS,1,1,Data,Sunday, 1/ 1,12/31",
    ]
    lines = list(header)
    for _, r in out_csv_df.iterrows():
        rec = [
            int(r["yr"]), int(r["mo"]), int(r["da"]), int(r["hr"]), 60,
            "?9?9?9?9E0?9?9?9?9*9?9?9?9?9?9?9*9*9*9*_*9*9*9?9?9",
            float(r["db"]), float(r["dp"]), int(r["rh"]),
            int(r["ps"] * 100),
            0, 0, 999, 0, 0, 0,
            999900, 999900, 999900, 99990,
            int(r["wdir"]), float(r["wspd"]), 5, 5,
            777.7, 77777, 9, 999999999, 0, 0.0000, 0,
            88, 999.000, 0.0, 0.0,
        ]
        lines.append(",".join(str(x) for x in rec))
    epw_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  EPW: {epw_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()
    year = args.year

    df, station_id, name = fetch(year)
    out = to_demo_csv(df, year)
    csv_path = OUT_DIR / f"CWBTP_{year}.csv"
    out.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[3/3] Writing CSV: {csv_path}")
    write_epw(out, year, station_id, name)

    db = out["db"]
    rh = out["rh"]
    print(f"  db [{db.min():.1f}..{db.max():.1f}] mean={db.mean():.1f}°C, "
          f"rh [{rh.min()}..{rh.max()}] mean={rh.mean():.1f}%")


if __name__ == "__main__":
    main()
