"""
Fetch 2024 Taipei hourly weather via the meteostat library
(https://github.com/meteostat/meteostat-python — wraps NOAA ISD bulk data).

The Demo only needs dry-bulb temperature (°C) and relative humidity (%) —
see src/epw_reader.py:read_weather_csv().

Output: models/weather/CWBTP_2024.csv   in the same column layout as
        the existing CWBTP_2017.csv so PIVDEngine accepts it natively.

Caveats surfaced in the report:
  - 2024 Taipei (站號 466920 Songshan / 466880 Banqiao) used as proxy for NCU
    (Chungli, ~30 km SSW). NCU runs ~1-2 °C cooler in winter and similar in summer.
  - Source is NOAA ISD daily-aggregated synoptic obs, hourly density may have
    occasional gaps; we forward-fill ≤ 3 hours and linearly interpolate longer gaps.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
from meteostat import Hourly, Stations

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "models" / "weather"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "CWBTP_2024.csv"
OUT_EPW = OUT_DIR / "CWBTP_2024.epw"

# Taipei (Songshan station — same as CWBTP_2017 used historically)
LAT, LON = 25.04, 121.51

YEAR = 2024


def fetch():
    print(f"[1/4] Locating nearest Meteostat station to ({LAT}, {LON})…")
    stations = Stations().nearby(LAT, LON).fetch(5)
    print(stations[["name", "country", "latitude", "longitude", "elevation",
                    "hourly_start", "hourly_end"]].to_string())
    station_id = stations.index[0]
    name = stations.iloc[0]["name"]
    print(f"  selected station: {station_id}  {name}")

    print(f"[2/4] Fetching hourly data {YEAR}-01-01 → {YEAR}-12-31…")
    start = dt.datetime(YEAR, 1, 1, 0)
    end = dt.datetime(YEAR, 12, 31, 23)
    df = Hourly(station_id, start, end).fetch()
    print(f"  rows fetched: {len(df)}  (expected 8784 for leap year)")
    return df, station_id, name


def to_demo_csv(df: pd.DataFrame, station_id: str, name: str) -> pd.DataFrame:
    """Reshape meteostat dataframe to demo CSV schema.

    meteostat columns: temp (°C), dwpt, rhum (%), prcp, snow, wdir, wspd, wpgt, pres, tsun, coco
    Demo schema (from CWBTP_2017.csv):
        yr,mo,da,hr,mi,etr,etrn,ghi,dni,dhi,tcld,db,dp,rh,ps,wdir,wspd,hvis,ch,lpd,lpq
    """
    # Build a complete hourly index for the year so we can fill gaps explicitly
    full_idx = pd.date_range(f"{YEAR}-01-01 00:00", f"{YEAR}-12-31 23:00", freq="h")
    df = df.reindex(full_idx)
    n_total = len(df)

    n_missing_t = df["temp"].isna().sum()
    n_missing_h = df["rhum"].isna().sum()
    print(f"[3/4] Filling gaps: temp NaN={n_missing_t}, rhum NaN={n_missing_h}")
    # Forward-fill short gaps then linearly interpolate longer gaps
    for col in ("temp", "rhum", "pres"):
        if col in df.columns:
            df[col] = df[col].ffill(limit=3).interpolate(method="time", limit_direction="both")

    out = pd.DataFrame({
        "yr": full_idx.year,
        "mo": full_idx.month,
        "da": full_idx.day,
        "hr": full_idx.hour + 1,   # demo csv uses 1-based hour like EPW
        "mi": 60,
        "etr":  0.0, "etrn": 0.0, "ghi": 0.0, "dni": 0.0, "dhi": 0.0,
        "tcld": 5,                  # no cloud info from this source — neutral
        "db": df["temp"].round(1).values,
        "dp": (df["temp"].fillna(0) - 5).round(1).values,  # crude dew point fallback
        "rh": df["rhum"].round(0).fillna(70).astype(int).values,
        "ps": (df["pres"].fillna(1013.0) / 1.0).round(1).values,
        "wdir": df.get("wdir", pd.Series(0, index=df.index)).fillna(0).astype(int).values,
        "wspd": df.get("wspd", pd.Series(2.0, index=df.index)).fillna(2.0).round(1).values,
        "hvis": -9900, "ch": -9900,
        "lpd": 0.0, "lpq": 0.0,
    })
    return out


def write_minimal_epw(out_csv_df: pd.DataFrame, station_id: str, name: str):
    """Write a minimal but spec-conformant EPW so EnergyPlus / epw_reader.py both work.

    Demo's epw_reader only reads dry_bulb + rel_humidity, so most fields can be
    placeholders (the EPW spec uses 999/9 for missing). We use 8784 (leap) records.
    """
    n = len(out_csv_df)
    header = [
        f"LOCATION,{name},Taiwan,Taiwan,Custom-{station_id},{station_id},{LAT},{LON},8.0,5.0",
        "DESIGN CONDITIONS,0",
        "TYPICAL/EXTREME PERIODS,0",
        "GROUND TEMPERATURES,0",
        "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
        f"COMMENTS 1,Synthesized 2024 from Meteostat NOAA ISD station {station_id} ({name})",
        "COMMENTS 2,For NCU PIVD demo — proxy for Chungli; only db/rh used by epw_reader",
        f"DATA PERIODS,1,1,Data,Sunday, 1/ 1,12/31",
    ]
    lines = list(header)
    for _, r in out_csv_df.iterrows():
        # EPW data record — 35 fields as per spec; we fill required ones, missing flags for rest
        rec = [
            int(r["yr"]), int(r["mo"]), int(r["da"]), int(r["hr"]),
            60,                          # minute
            "?9?9?9?9E0?9?9?9?9*9?9?9?9?9?9?9*9*9*9*_*9*9*9?9?9",  # data source flags
            float(r["db"]),              # 7  dry bulb °C
            float(r["dp"]),              # 8  dew point °C
            int(r["rh"]),                # 9  relative humidity %
            int(r["ps"] * 100),          # 10 atm pressure Pa (csv has hPa)
            0,                            # 11 ext horiz radiation Wh/m²
            0,                            # 12 ext direct normal radiation
            999,                          # 13 horiz infrared
            0, 0, 0,                      # 14-16 GHI/DNI/DHI
            999900, 999900, 999900,        # 17-19 illuminance Lux missing
            99990,                         # 20 zenith luminance missing
            int(r["wdir"]),              # 21 wind dir
            float(r["wspd"]),            # 22 wind speed m/s
            5, 5,                          # 23-24 total/opaque sky cover (1/10ths)
            777.7, 77777,                 # 25-26 visibility / ceiling
            9, 999999999,                 # 27-28 present weather obs / codes
            0, 0.0000, 0,                 # 29-31 precipitable water / aerosol / snow
            88, 999.000,                  # 32-33 days since snow / albedo
            0.0, 0.0,                     # 34-35 liquid precip mm / hr
        ]
        lines.append(",".join(str(x) for x in rec))
    OUT_EPW.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    df, station_id, name = fetch()
    out_csv_df = to_demo_csv(df, station_id, name)

    print(f"[4/4] Writing CSV + EPW…")
    out_csv_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    write_minimal_epw(out_csv_df, str(station_id), str(name))

    # Quality summary
    db = out_csv_df["db"]
    rh = out_csv_df["rh"]
    summary = (
        f"  rows: {len(out_csv_df):,} (expected {366*24} for {YEAR})\n"
        f"  dry-bulb °C  min={db.min():.1f}  mean={db.mean():.1f}  max={db.max():.1f}\n"
        f"  humidity %   min={rh.min()}    mean={rh.mean():.1f}  max={rh.max()}\n"
    )
    print(summary)
    print(f"Wrote:")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_EPW}")


if __name__ == "__main__":
    main()
