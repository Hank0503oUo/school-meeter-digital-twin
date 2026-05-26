"""
Fetch CWA/CODiS hourly station observations and convert them to the demo's
CWBTP weather CSV + minimal EPW format.

Default NCU 2025 source is the active Zhongli automatic station (C0C700).  The
Central University agricultural station (C2C410) starts during 2025 and is not
complete for the full calendar year, so it is useful for checks but not as the
primary full-year EPW source.

Usage:
    python scripts/ncu_cleaning/fetch_cwa_year_weather.py --year 2025 --station zhongli --copy-campus ncu
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "models" / "weather"
AUDIT_DIR = ROOT / "outputs" / "ncu_114"
OUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

CODIS_STATION_URL = "https://codis.cwa.gov.tw/api/station"

STATIONS: dict[str, dict[str, Any]] = {
    "zhongli": {
        "station_id": "C0C700",
        "stn_type": "auto_C0",
        "name": "中壢",
        "name_en": "Zhongli",
        "lat": 24.977661,
        "lon": 121.256375,
        "altitude_m": 151.0,
        "comment": "CWA automatic weather station in Zhongli; active for full 2025.",
    },
    "taipei": {
        "station_id": "466920",
        "stn_type": "cwb",
        "name": "臺北",
        "name_en": "Taipei",
        "lat": 25.037658,
        "lon": 121.514853,
        "altitude_m": 6.3,
        "comment": "CWA manned weather station in Taipei.",
    },
    "ncu": {
        "station_id": "C2C410",
        "stn_type": "agr",
        "name": "中央大學",
        "name_en": "National Central University",
        "lat": 24.967652,
        "lon": 121.185165,
        "altitude_m": 129.0,
        "comment": "CWA agricultural station on the NCU campus; partial 2025 coverage.",
    },
}


def scalar(obj: dict[str, Any], *path: str) -> float | None:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    if cur in (None, "", "--", "/", "x"):
        return None
    if cur == "T":
        return 0.0
    try:
        val = float(cur)
    except (TypeError, ValueError):
        return None
    # CWA uses negative sentinels for some missing fields.
    if val <= -90:
        return None
    return val


def dew_point_c(temp_c: float, rh: float) -> float:
    """Magnus approximation, valid enough for EPW placeholder dew point."""
    if not math.isfinite(temp_c) or not math.isfinite(rh) or rh <= 0:
        return temp_c - 5.0
    rh = min(max(rh, 1.0), 100.0)
    a = 17.625
    b = 243.04
    gamma = math.log(rh / 100.0) + (a * temp_c) / (b + temp_c)
    return (b * gamma) / (a - gamma)


def codis_time_to_hour_start(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.minute == 59 and ts.hour == 23:
        return ts.replace(minute=0)
    return ts - pd.Timedelta(hours=1)


def month_range(year: int, month: int) -> tuple[str, str]:
    last = calendar.monthrange(year, month)[1]
    start = f"{year:04d}-{month:02d}-01T00:00:00"
    end = f"{year:04d}-{month:02d}-{last:02d}T23:59:59"
    return start, end


def fetch_month(station: dict[str, Any], year: int, month: int) -> list[dict[str, Any]]:
    start, end = month_range(year, month)
    payload = {
        "type": "report_date",
        "stn_type": station["stn_type"],
        "stn_ID": station["station_id"],
        "start": start,
        "end": end,
        "more": "",
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://codis.cwa.gov.tw/StationData",
    }
    for attempt in range(1, 4):
        resp = requests.post(CODIS_STATION_URL, data=payload, headers=headers, timeout=60)
        if resp.ok:
            data = resp.json()
            if data.get("code") == 200:
                blocks = data.get("data") or []
                if blocks:
                    return blocks[0].get("dts") or []
                return []
        if attempt < 3:
            time.sleep(0.8 * attempt)
    resp.raise_for_status()
    return []


def parse_observations(rows: list[dict[str, Any]]) -> pd.DataFrame:
    parsed = []
    for row in rows:
        data_time = row.get("DataTime")
        if not data_time:
            continue
        temp = scalar(row, "AirTemperature", "Instantaneous")
        rh = scalar(row, "RelativeHumidity", "Instantaneous")
        dew = scalar(row, "DewPointTemperature", "Instantaneous")
        pressure = scalar(row, "StationPressure", "Instantaneous")
        wind_dir = scalar(row, "WindDirection", "Mean")
        wind_speed = scalar(row, "WindSpeed", "Mean")
        solar_mj = scalar(row, "GlobalSolarRadiation", "Accumulation")

        parsed.append(
            {
                "datetime": codis_time_to_hour_start(data_time),
                "db": temp,
                "rh": rh,
                "dp": dew,
                "ps": pressure,
                "wdir": wind_dir,
                "wspd": wind_speed,
                "ghi": solar_mj * 277.7777778 if solar_mj is not None else 0.0,
            }
        )
    if not parsed:
        return pd.DataFrame(columns=["datetime", "db", "rh", "dp", "ps", "wdir", "wspd", "ghi"])
    df = pd.DataFrame(parsed)
    return df.drop_duplicates(subset=["datetime"], keep="last").set_index("datetime").sort_index()


def complete_year_frame(raw: pd.DataFrame, year: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    idx = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="h")
    df = raw.reindex(idx)
    expected = len(idx)
    audit: dict[str, Any] = {
        "expected_hours": expected,
        "raw_hours": int(raw.shape[0]),
        "raw_coverage_ratio": round(float(raw.shape[0]) / expected, 4),
    }
    for col in ["db", "rh", "dp", "ps", "wdir", "wspd", "ghi"]:
        audit[f"{col}_missing_before_fill"] = int(df[col].isna().sum()) if col in df else expected

    for col in ["db", "rh", "ps", "wspd"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].ffill(limit=3).interpolate(method="time", limit_direction="both")

    df["rh"] = df["rh"].fillna(70.0).clip(lower=0.0, upper=100.0)
    df["ps"] = df["ps"].fillna(df["ps"].median()).fillna(1013.0)
    df["wspd"] = df["wspd"].fillna(2.0).clip(lower=0.0)
    df["wdir"] = pd.to_numeric(df["wdir"], errors="coerce").ffill(limit=3).fillna(0.0)
    df["ghi"] = pd.to_numeric(df["ghi"], errors="coerce").fillna(0.0).clip(lower=0.0)

    df["dp"] = pd.to_numeric(df["dp"], errors="coerce")
    missing_dp = df["dp"].isna()
    df.loc[missing_dp, "dp"] = [
        dew_point_c(t, h) for t, h in zip(df.loc[missing_dp, "db"], df.loc[missing_dp, "rh"])
    ]
    df["dp"] = df["dp"].ffill(limit=3).interpolate(method="time", limit_direction="both")

    for col in ["db", "rh", "dp", "ps", "wdir", "wspd", "ghi"]:
        audit[f"{col}_missing_after_fill"] = int(df[col].isna().sum())

    return df, audit


def to_demo_csv(df: pd.DataFrame, year: int) -> pd.DataFrame:
    idx = df.index
    return pd.DataFrame(
        {
            "yr": idx.year,
            "mo": idx.month,
            "da": idx.day,
            "hr": idx.hour + 1,
            "mi": 60,
            "etr": 0.0,
            "etrn": 0.0,
            "ghi": df["ghi"].round(1).values,
            "dni": 0.0,
            "dhi": 0.0,
            "tcld": 5,
            "db": df["db"].round(1).values,
            "dp": df["dp"].round(1).values,
            "rh": df["rh"].round(0).astype(int).values,
            "ps": df["ps"].round(1).values,
            "wdir": df["wdir"].round(0).astype(int).values,
            "wspd": df["wspd"].round(1).values,
            "hvis": -9900,
            "ch": -9900,
            "lpd": 0.0,
            "lpq": 0.0,
        }
    )


def write_epw(out_csv_df: pd.DataFrame, year: int, station: dict[str, Any], epw_path: Path) -> None:
    start_weekday = dt.date(year, 1, 1).strftime("%A")
    header = [
        (
            f"LOCATION,{station['name_en']},Taiwan,Taiwan,"
            f"CWA-{station['station_id']},{station['station_id']},"
            f"{station['lat']},{station['lon']},8.0,{station['altitude_m']}"
        ),
        "DESIGN CONDITIONS,0",
        "TYPICAL/EXTREME PERIODS,0",
        "GROUND TEMPERATURES,0",
        "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
        f"COMMENTS 1,Source: CWA CODiS report_date hourly station {station['station_id']} ({station['name']}) for {year}",
        "COMMENTS 2,NCU PIVD demo weather; epw_reader uses dry-bulb and relative humidity.",
        f"DATA PERIODS,1,1,Data,{start_weekday}, 1/ 1,12/31",
    ]
    lines = list(header)
    for _, r in out_csv_df.iterrows():
        ghi = max(0, int(round(float(r["ghi"]))))
        rec = [
            int(r["yr"]),
            int(r["mo"]),
            int(r["da"]),
            int(r["hr"]),
            60,
            "?9?9?9?9E0?9?9?9?9*9?9?9?9?9?9?9*9*9*9*_*9*9*9?9?9",
            float(r["db"]),
            float(r["dp"]),
            int(r["rh"]),
            int(float(r["ps"]) * 100),
            0,
            0,
            999,
            ghi,
            0,
            0,
            999900,
            999900,
            999900,
            99990,
            int(r["wdir"]),
            float(r["wspd"]),
            5,
            5,
            777.7,
            77777,
            9,
            999999999,
            0,
            0.0000,
            0,
            88,
            999.000,
            0.0,
            0.0,
        ]
        lines.append(",".join(str(x) for x in rec))
    epw_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(out_csv_df: pd.DataFrame, raw_df: pd.DataFrame, audit: dict[str, Any], year: int, station_key: str, station: dict[str, Any], copy_campus: str | None) -> None:
    default_name = f"CWBTP_{year}" if station_key == "zhongli" else f"CWBTP_{year}_{station['station_id']}"
    csv_path = OUT_DIR / f"{default_name}.csv"
    epw_path = OUT_DIR / f"{default_name}.epw"
    raw_path = AUDIT_DIR / f"weather_{station['station_id']}_{year}_raw.csv"
    audit_path = AUDIT_DIR / f"weather_{station['station_id']}_{year}_audit.json"

    out_csv_df.to_csv(csv_path, index=False, encoding="utf-8")
    write_epw(out_csv_df, year, station, epw_path)
    raw_df.to_csv(raw_path, encoding="utf-8-sig")

    audit = {
        **audit,
        "station_key": station_key,
        "station_id": station["station_id"],
        "station_name": station["name"],
        "year": year,
        "output_csv": str(csv_path.relative_to(ROOT)),
        "output_epw": str(epw_path.relative_to(ROOT)),
        "raw_csv": str(raw_path.relative_to(ROOT)),
        "temperature_c_min": round(float(out_csv_df["db"].min()), 2),
        "temperature_c_mean": round(float(out_csv_df["db"].mean()), 2),
        "temperature_c_max": round(float(out_csv_df["db"].max()), 2),
        "relative_humidity_mean": round(float(out_csv_df["rh"].mean()), 2),
        "note": station["comment"],
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    if copy_campus:
        campus_weather = ROOT / "campuses" / copy_campus / "models" / "weather"
        campus_weather.mkdir(parents=True, exist_ok=True)
        shutil.copy2(csv_path, campus_weather / csv_path.name)
        shutil.copy2(epw_path, campus_weather / epw_path.name)

    print(f"station: {station['name']} ({station['station_id']})")
    print(f"hours: {len(out_csv_df):,}, raw coverage: {audit['raw_coverage_ratio']:.1%}")
    print(
        "db: "
        f"{audit['temperature_c_min']:.1f}..{audit['temperature_c_max']:.1f} C, "
        f"mean {audit['temperature_c_mean']:.1f} C; "
        f"RH mean {audit['relative_humidity_mean']:.1f}%"
    )
    print(f"CSV: {csv_path}")
    print(f"EPW: {epw_path}")
    if copy_campus:
        print(f"Copied to campuses/{copy_campus}/models/weather/")
    print(f"Audit: {audit_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--station", choices=sorted(STATIONS), default="zhongli")
    parser.add_argument("--copy-campus", choices=["ncu", "ntu"], default=None)
    args = parser.parse_args()

    # Keep CWA OpenData authorization available for future extension, but CODiS
    # historical station reports do not require it.
    _ = os.environ.get("CWA_API_AUTHORIZATION", "")

    station = STATIONS[args.station]
    print(f"Fetching {args.year} CWA/CODiS hourly data: {station['name']} ({station['station_id']})")
    all_rows: list[dict[str, Any]] = []
    for month in range(1, 13):
        rows = fetch_month(station, args.year, month)
        print(f"  {args.year}-{month:02d}: {len(rows)} rows")
        all_rows.extend(rows)

    raw_df = parse_observations(all_rows)
    full_df, audit = complete_year_frame(raw_df, args.year)
    out_csv_df = to_demo_csv(full_df, args.year)
    write_outputs(out_csv_df, raw_df, audit, args.year, args.station, station, args.copy_campus)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
