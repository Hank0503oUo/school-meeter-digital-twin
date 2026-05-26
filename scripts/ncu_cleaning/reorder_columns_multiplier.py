"""
Move 倍數 to sit immediately before the reading columns in every
per-quarter aligned CSV (and the combined wide CSV) so the visual
flow reads as: ... 倍數 | prev | m1 | m2 | m3.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PER_QUARTER_DIR = Path(r"C:\Users\User\Downloads\中大電表資料\per_quarter")
COMBINED_WIDE = Path(r"C:\Users\User\Downloads\中大電表資料\中大電表資料_combined_109to114_wide.csv")


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp950"):
        try:
            return pd.read_csv(path, dtype=str, encoding=encoding, keep_default_na=False)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode {path}")


def is_reading_col(name: str) -> bool:
    """Match column headers like '2020-04' or '2020-03 (prev)'."""
    s = name.strip()
    if len(s) < 7:
        return False
    head = s[:7]
    return len(head) == 7 and head[4] == "-" and head[:4].isdigit() and head[5:].isdigit()


def reorder(df: pd.DataFrame, mul_col: str = "倍數") -> pd.DataFrame:
    cols = list(df.columns)
    if mul_col not in cols:
        return df
    reading_cols = [c for c in cols if is_reading_col(c)]
    if not reading_cols:
        return df
    first_reading = reading_cols[0]
    other = [c for c in cols if c != mul_col and c not in reading_cols]
    first_reading_idx = other.index(first_reading) if first_reading in other else None
    # Build: everything except 倍數 and readings, then 倍數, then readings.
    new_order = other + [mul_col] + reading_cols
    return df[new_order]


def process_file(path: Path) -> None:
    df = read_csv(path)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    new_df = reorder(df)
    if list(new_df.columns) == list(df.columns):
        print(f"  {path.name}: no change")
        return
    new_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  {path.name}: reordered")


def main() -> None:
    configure_stdout()

    print("Reordering per-quarter files...")
    for path in sorted(PER_QUARTER_DIR.glob("中大電表資料_*Q*_aligned.csv")):
        process_file(path)

    if COMBINED_WIDE.exists():
        print()
        print("Reordering combined wide file...")
        # Combined wide uses 'multiplier_v4' + 'multiplier_audit' rather than a single 倍數.
        # Move 'multiplier_v4' to be just before the reading columns.
        df = read_csv(COMBINED_WIDE)
        df.columns = [c.strip().lstrip("﻿") for c in df.columns]
        new_df = reorder(df, mul_col="multiplier_v4")
        # Also move multiplier_audit right after multiplier_v4 for symmetry.
        cols = list(new_df.columns)
        if "multiplier_audit" in cols and "multiplier_v4" in cols:
            cols.remove("multiplier_audit")
            insert_idx = cols.index("multiplier_v4") + 1
            cols.insert(insert_idx, "multiplier_audit")
            new_df = new_df[cols]
        new_df.to_csv(COMBINED_WIDE, index=False, encoding="utf-8-sig")
        print(f"  {COMBINED_WIDE.name}: reordered")


if __name__ == "__main__":
    main()
