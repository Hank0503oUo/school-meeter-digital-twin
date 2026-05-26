"""
Backfill the 使用單位 (usage_unit) column in per-quarter aligned CSVs.

The meter_audit.csv pipeline does not carry the usage_unit field, so any
meter that is not already in the v4 master comes out with an empty
使用單位 in the per-quarter outputs. This script reads the raw mineru
quarterly CSVs, builds a meter_id → usage_unit lookup, and writes the
backfilled value back into each per-quarter file (only where the cell
was empty — never overwrites a v4-supplied value).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


MINERU_DIR = Path(r"G:\我的雲端硬碟\mineru\mineru_output_mineru_vlm\csv_by_quarter")
PER_QUARTER_DIR = Path(r"C:\Users\User\Downloads\中大電表資料\per_quarter")


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp950"):
        try:
            return pd.read_csv(path, dtype=str, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode {path}")


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def strip_lead_zero(meter_id: str) -> str:
    text = meter_id.strip()
    if not text:
        return ""
    if text.lstrip("0").replace("-", "").isdigit() or text.isdigit():
        return text.lstrip("0") or "0"
    return text


def build_usage_lookup() -> dict[str, str]:
    """Build meter_id → usage_unit dict from every quarterly mineru CSV."""
    lookup: dict[str, str] = {}
    files = sorted(MINERU_DIR.glob("中大電表資料_*Q*_mineru_vlm.csv"))
    print(f"Scanning {len(files)} mineru CSV(s)...")
    for path in files:
        df = read_csv(path)
        df.columns = [c.strip().lstrip("﻿") for c in df.columns]
        df = df.rename(columns={
            "表號": "meter_id",
            "使用單位": "usage_unit",
        })
        if "meter_id" not in df.columns or "usage_unit" not in df.columns:
            print(f"  [skip] {path.name}: missing columns")
            continue
        df["meter_id"] = df["meter_id"].map(clean_text)
        df["usage_unit"] = df["usage_unit"].map(clean_text)
        df = df[(df["meter_id"] != "") & (df["usage_unit"] != "")]
        for _, row in df.iterrows():
            mid_raw = row["meter_id"]
            mid_norm = strip_lead_zero(mid_raw)
            usage = row["usage_unit"]
            # Don't overwrite an existing entry — first non-empty wins.
            for key in (mid_raw, mid_norm):
                if key and key not in lookup:
                    lookup[key] = usage
    print(f"  built lookup with {len(lookup)} unique keys")
    return lookup


def backfill_file(path: Path, lookup: dict[str, str]) -> tuple[int, int]:
    """Backfill 使用單位 in one per-quarter CSV. Returns (rows_filled, rows_total)."""
    df = read_csv(path)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    if "使用單位" not in df.columns or "meter_id" not in df.columns:
        print(f"  [skip] {path.name}: missing required columns")
        return 0, 0

    filled = 0
    for idx in df.index:
        mid = clean_text(df.at[idx, "meter_id"])
        current = clean_text(df.at[idx, "使用單位"])
        if not mid or current:
            continue
        mid_norm = strip_lead_zero(mid)
        usage = lookup.get(mid) or lookup.get(mid_norm) or ""
        if usage:
            df.at[idx, "使用單位"] = usage
            filled += 1

    df.to_csv(path, index=False, encoding="utf-8-sig")
    return filled, len(df)


def main() -> None:
    configure_stdout()
    if not MINERU_DIR.exists():
        raise FileNotFoundError(MINERU_DIR)
    if not PER_QUARTER_DIR.exists():
        raise FileNotFoundError(PER_QUARTER_DIR)

    lookup = build_usage_lookup()
    print()

    files = sorted(PER_QUARTER_DIR.glob("中大電表資料_*Q*_aligned.csv"))
    print(f"Backfilling {len(files)} per-quarter file(s)...")
    total_filled = 0
    for path in files:
        filled, total = backfill_file(path, lookup)
        print(f"  {path.name}: filled {filled}/{total}")
        total_filled += filled

    print()
    print(f"Done. Total 使用單位 cells filled: {total_filled}")


if __name__ == "__main__":
    main()
