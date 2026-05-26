"""
Merge all quarterly NCU meter readings into one wide-format CSV.

Authoritative master = 中大電表資料_109Q1_v4.csv (user-cleaned).

For every subsequent quarter (109Q2–114Q4), pull readings from the existing
meter_audit.csv pipelines and align them to the v4 master list. Meters that
appear in later quarters but not in v4 are appended with a NEW flag. Meters
whose audit-side building disagrees with the v4 building are flagged for
manual review.

Match strategy
--------------
Primary key: 表號 (meter_id), matched case-sensitively. To absorb the most
common OCR / formatting drift, we also try a "stripped" variant where any
leading zeros are removed; an audit row whose stripped meter_id matches a
v4 meter_id (or vice versa) is treated as the same meter but flagged
ZERO_PAD_ADJUSTED so the user can confirm.

Output
------
1. <out_dir>/中大電表資料_combined_109to114_wide.csv
     Wide format: one row per meter, columns sorted by year-month.
2. <out_dir>/中大電表資料_combined_review_flags.csv
     Only the flagged rows, for fast human review.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


V4_PATH = Path(r"C:\Users\User\Downloads\中大電表資料\中大電表資料_109Q1_v4.csv")
AUDIT_BASE = Path(r"C:\Users\User\demo\outputs")
AUDIT_DIRS = ["ncu_109", "ncu_110", "ncu_111", "ncu_114"]
OUT_DIR = Path(r"C:\Users\User\Downloads\中大電表資料")
WIDE_OUT = OUT_DIR / "中大電表資料_combined_109to114_wide.csv"
REVIEW_OUT = OUT_DIR / "中大電表資料_combined_review_flags.csv"


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
    """Normalize a meter_id by stripping leading zeros from numeric prefixes."""
    text = meter_id.strip()
    if not text:
        return ""
    # If purely numeric (possibly with dashes), strip leading zeros.
    if text.lstrip("0").replace("-", "").isdigit() or text.isdigit():
        stripped = text.lstrip("0") or "0"
        return stripped
    return text


def load_v4_master(path: Path) -> pd.DataFrame:
    df = read_csv(path)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    df = df.rename(columns={
        "PDF來源": "pdf_src",
        "建築物": "building_v4",
        "次": "seq",
        "表號": "meter_id",
        "開關箱": "panel_v4",
        "安裝位置": "location_v4",
        "倍數": "multiplier_v4",
        "使用單位": "usage_unit_v4",
    })
    for col in df.columns:
        df[col] = df[col].map(clean_text)
    # Drop separator / empty rows
    df = df[(df["meter_id"] != "") & (df["building_v4"] != "")].copy()
    df = df.drop_duplicates(subset=["building_v4", "meter_id"])

    # Keep also the four template readings, mapped to actual calendar months
    # (template is 109Q1: 12月→2019-12, 1月→2020-01, etc.)
    rename_reads = {"12月": "2019-12", "1月": "2020-01", "2月": "2020-02", "3月": "2020-03"}
    for src, dst in rename_reads.items():
        if src in df.columns:
            df = df.rename(columns={src: dst})

    df["meter_id_norm"] = df["meter_id"].map(strip_lead_zero)
    return df


def load_audit(audit_path: Path) -> pd.DataFrame:
    df = read_csv(audit_path)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    for col in df.columns:
        df[col] = df[col].map(clean_text)
    df = df[(df["meter_id"] != "") & (df["building"] != "")].copy()
    # Numeric year / month
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df = df.dropna(subset=["year", "month"])
    df["ym"] = (
        df["year"].astype(int).astype(str)
        + "-"
        + df["month"].astype(int).astype(str).str.zfill(2)
    )
    df["meter_id_norm"] = df["meter_id"].map(strip_lead_zero)
    return df[[
        "meter_id", "meter_id_norm", "building", "panel", "location",
        "multiplier", "year", "month", "ym", "reading", "source_q",
    ]]


def main() -> None:
    configure_stdout()

    if not V4_PATH.exists():
        raise FileNotFoundError(V4_PATH)

    print(f"Loading v4 master: {V4_PATH.name}")
    v4 = load_v4_master(V4_PATH)
    print(f"  v4 meters: {len(v4)}")

    audit_frames: list[pd.DataFrame] = []
    for d in AUDIT_DIRS:
        path = AUDIT_BASE / d / "meter_audit.csv"
        if not path.exists():
            print(f"  [skip] missing: {path}")
            continue
        df = load_audit(path)
        df["_source_dir"] = d
        audit_frames.append(df)
        print(f"  audit {d}: {len(df)} rows, "
              f"{df['meter_id'].nunique()} unique meters, "
              f"ym range {df['ym'].min()}–{df['ym'].max()}")

    audit = pd.concat(audit_frames, ignore_index=True)
    print(f"  total audit rows: {len(audit)}")

    # Build lookup from v4 by both raw and normalized meter_id
    v4_by_raw = {row["meter_id"]: row for _, row in v4.iterrows()}
    v4_by_norm = {}
    for _, row in v4.iterrows():
        norm = row["meter_id_norm"]
        # If multiple v4 ids normalize to the same string, prefer the longer one
        if norm not in v4_by_norm or len(row["meter_id"]) > len(v4_by_norm[norm]["meter_id"]):
            v4_by_norm[norm] = row

    # Resolve each audit meter to canonical v4 meter (if any)
    def resolve_to_v4(mid: str, mid_norm: str) -> tuple[str, str]:
        """Return (v4_meter_id_canonical, match_kind)."""
        if mid in v4_by_raw:
            return mid, "EXACT"
        if mid_norm in v4_by_norm:
            return v4_by_norm[mid_norm]["meter_id"], "ZERO_PAD_ADJUSTED"
        return mid, "NEW"

    resolutions = audit.apply(
        lambda r: pd.Series(
            resolve_to_v4(r["meter_id"], r["meter_id_norm"]),
            index=["canonical_meter_id", "match_kind"],
        ),
        axis=1,
    )
    audit = pd.concat([audit, resolutions], axis=1)

    # Pivot wide on canonical_meter_id
    wide_audit = audit.pivot_table(
        index="canonical_meter_id",
        columns="ym",
        values="reading",
        aggfunc="first",
    )

    # Bring in v4 metadata + v4's own readings, joining on canonical id
    v4_meta = v4[[
        "meter_id",
        "building_v4", "panel_v4", "location_v4", "multiplier_v4", "usage_unit_v4",
        "2019-12", "2020-01", "2020-02", "2020-03",
    ]].rename(columns={"meter_id": "canonical_meter_id"})

    wide_audit_reset = wide_audit.reset_index()

    combined = v4_meta.merge(
        wide_audit_reset,
        on="canonical_meter_id",
        how="outer",
        suffixes=("", "_audit"),
    )

    # Prefer v4-side readings for the 2019-12 / 2020-01 / 2020-02 / 2020-03 columns
    for col in ["2019-12", "2020-01", "2020-02", "2020-03"]:
        audit_col = f"{col}_audit"
        if audit_col in combined.columns:
            combined[col] = combined[col].where(
                combined[col].notna() & (combined[col].astype(str) != ""),
                combined[audit_col],
            )
            combined = combined.drop(columns=[audit_col])

    combined = combined.rename(columns={"canonical_meter_id": "meter_id"})

    # Per-meter audit-side building / panel / location / multiplier — first non-empty
    audit_meta = (
        audit.sort_values(["canonical_meter_id", "ym"])
        .groupby("canonical_meter_id")
        .agg(
            building_audit=("building", lambda s: " | ".join(
                sorted({str(x).strip() for x in s if str(x).strip()})
            )),
            panel_audit=("panel", lambda s: " | ".join(
                sorted({str(x).strip() for x in s if str(x).strip()})[:3]
            )),
            location_audit=("location", lambda s: " | ".join(
                sorted({str(x).strip()[:25] for x in s if str(x).strip()})[:2]
            )),
            multiplier_audit=("multiplier", lambda s: " | ".join(
                sorted({str(x).strip() for x in s if str(x).strip()})
            )),
            match_kind=("match_kind", "first"),
            audit_quarters=("source_q", lambda s: ",".join(
                sorted({str(x).strip() for x in s if str(x).strip()})
            )),
        )
        .reset_index()
        .rename(columns={"canonical_meter_id": "meter_id"})
    )

    combined = combined.merge(audit_meta, on="meter_id", how="left")

    # Build flag
    def build_flag(r: pd.Series) -> str:
        flags: list[str] = []
        b_v4 = clean_text(r.get("building_v4"))
        b_au = clean_text(r.get("building_audit"))
        mk = clean_text(r.get("match_kind"))
        if not b_v4 and b_au:
            flags.append("NEW_METER")
        elif b_v4 and b_au:
            if b_v4 not in b_au.split(" | "):
                flags.append("BUILDING_MISMATCH")
        if mk == "ZERO_PAD_ADJUSTED":
            flags.append("ZERO_PAD_ADJUSTED")
        return ";".join(flags)

    combined["flag"] = combined.apply(build_flag, axis=1)

    # Canonical building column: v4 wins, fallback to audit (handle NaN properly)
    b_v4 = combined["building_v4"].fillna("").astype(str).str.strip()
    b_audit = combined["building_audit"].fillna("").astype(str).str.strip()
    combined["建築物"] = b_v4.where(b_v4.ne(""), b_audit)

    # Order year-month columns chronologically
    ym_cols = [c for c in combined.columns if isinstance(c, str)
               and len(c) == 7 and c[4] == "-"]
    ym_cols = sorted(ym_cols)

    meta_cols = [
        "建築物",
        "meter_id",
        "panel_v4",
        "location_v4",
        "multiplier_v4",
        "usage_unit_v4",
        "flag",
        "building_v4",
        "building_audit",
        "panel_audit",
        "location_audit",
        "multiplier_audit",
        "match_kind",
        "audit_quarters",
    ]
    final_cols = meta_cols + ym_cols
    final = combined[final_cols].copy()

    # Sort by canonical building name (NaNs last), then meter_id
    final["_sort_b"] = final["建築物"].fillna("")
    final = final.sort_values(
        ["_sort_b", "meter_id"], kind="mergesort"
    ).drop(columns="_sort_b")

    # Insert a blank separator row between consecutive buildings so the user
    # can scan each building as a visual block.
    blank_row = {col: "" for col in final.columns}
    rows_with_separators: list[dict] = []
    prev_building: str | None = None
    for _, row in final.iterrows():
        current = clean_text(row["建築物"])
        if prev_building is not None and current != prev_building:
            rows_with_separators.append(blank_row.copy())
        rows_with_separators.append(row.to_dict())
        prev_building = current
    final_with_gaps = pd.DataFrame(rows_with_separators, columns=final.columns)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final_with_gaps.to_csv(WIDE_OUT, index=False, encoding="utf-8-sig")
    print()
    print(f"Wrote wide CSV: {WIDE_OUT}")
    print(f"  total meters in output: {len(final)}")
    print(f"  year-month columns:     {len(ym_cols)} "
          f"({ym_cols[0]} – {ym_cols[-1]})")

    # Flags summary
    has_flag = final["flag"].astype(str).str.strip().ne("")
    flagged = final[has_flag].copy()
    flagged.to_csv(REVIEW_OUT, index=False, encoding="utf-8-sig")
    print(f"Wrote review CSV: {REVIEW_OUT}")
    print(f"  flagged rows: {len(flagged)}")

    if not flagged.empty:
        print()
        print("Flag breakdown:")
        flag_counts = (
            flagged["flag"].astype(str).str.split(";").explode().value_counts()
        )
        for k, v in flag_counts.items():
            print(f"  {k}: {v}")

        print()
        print("Sample flagged rows (first 12):")
        cols_for_preview = [
            "建築物", "meter_id", "flag",
            "building_v4", "building_audit", "audit_quarters",
        ]
        with pd.option_context("display.max_colwidth", 30):
            print(flagged[cols_for_preview].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
