"""
Split the merged NCU meter readings into per-quarter CSV files.

For every (ROC year, quarter) we have data for, write one CSV that mirrors
the original mineru paper layout:

  建築物, meter_id, 開關箱, 安裝位置, 倍數, 使用單位, flag,
  <prev_month YYYY-MM>, <m1>, <m2>, <m3>

Where the four reading columns are the previous quarter's last month plus
the three months of the current quarter, labelled by actual calendar
year-month so the user does not have to remember the quarter→month mapping.

Output files:
  C:/Users/User/Downloads/中大電表資料/per_quarter/中大電表資料_<ROC>Q<N>_aligned.csv

Each file is sorted by canonical building name and has a blank separator
row between buildings.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


V4_PATH = Path(r"C:\Users\User\Downloads\中大電表資料\中大電表資料_109Q1_v4.csv")
AUDIT_BASE = Path(r"C:\Users\User\demo\outputs")
AUDIT_DIRS = ["ncu_109", "ncu_110", "ncu_111", "ncu_114"]
OUT_DIR = Path(r"C:\Users\User\Downloads\中大電表資料\per_quarter")


# ROC year (民國) → western calendar year for that ROC year's Q1-Q4
ROC_TO_WEST = {"ncu_109": 2020, "ncu_110": 2021, "ncu_111": 2022, "ncu_114": 2025}
ROC_LABEL   = {"ncu_109": "109", "ncu_110": "110", "ncu_111": "111", "ncu_114": "114"}

# Quarter → (prev-month offset, [m1, m2, m3]) within western calendar
QUARTER_MONTHS = {
    "Q1": (-1,  [1, 2, 3]),   # prev = Dec of prev year, then Jan/Feb/Mar
    "Q2": ( 3,  [4, 5, 6]),   # prev = Mar
    "Q3": ( 6,  [7, 8, 9]),   # prev = Jun
    "Q4": ( 9,  [10, 11, 12]),
}


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
    df = df[(df["meter_id"] != "") & (df["building_v4"] != "")].copy()
    df = df.drop_duplicates(subset=["building_v4", "meter_id"])
    df["meter_id_norm"] = df["meter_id"].map(strip_lead_zero)
    return df


def load_audit(audit_path: Path, source_dir: str) -> pd.DataFrame:
    df = read_csv(audit_path)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    for col in df.columns:
        df[col] = df[col].map(clean_text)
    df = df[(df["meter_id"] != "") & (df["building"] != "")].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df = df.dropna(subset=["year", "month"])
    df["meter_id_norm"] = df["meter_id"].map(strip_lead_zero)
    df["_source_dir"] = source_dir
    return df


def resolve_canonical(audit: pd.DataFrame, v4: pd.DataFrame) -> pd.DataFrame:
    v4_raw = set(v4["meter_id"])
    v4_norm = {}
    for _, row in v4.iterrows():
        norm = row["meter_id_norm"]
        if norm not in v4_norm or len(row["meter_id"]) > len(v4_norm[norm]):
            v4_norm[norm] = row["meter_id"]

    def resolve(mid: str, mid_norm: str) -> tuple[str, str]:
        if mid in v4_raw:
            return mid, "EXACT"
        if mid_norm in v4_norm:
            return v4_norm[mid_norm], "ZERO_PAD_ADJUSTED"
        return mid, "NEW"

    res = audit.apply(
        lambda r: pd.Series(resolve(r["meter_id"], r["meter_id_norm"]),
                            index=["canonical_meter_id", "match_kind"]),
        axis=1,
    )
    return pd.concat([audit, res], axis=1)


def column_specs(source_dir: str, quarter: str) -> list[tuple[int, int, str]]:
    """Return [(year, month, label), ...] for the 4 reading columns of one quarter."""
    west = ROC_TO_WEST[source_dir]
    prev_offset, months = QUARTER_MONTHS[quarter]
    if prev_offset == -1:
        prev = (west - 1, 12)
    else:
        prev = (west, prev_offset)
    specs = [(prev[0], prev[1], f"{prev[0]}-{prev[1]:02d} (prev)")]
    for m in months:
        specs.append((west, m, f"{west}-{m:02d}"))
    return specs


def build_quarter_table(
    audit: pd.DataFrame, v4: pd.DataFrame, source_dir: str, quarter: str
) -> pd.DataFrame:
    grp = audit[(audit["_source_dir"] == source_dir) & (audit["source_q"] == quarter)].copy()
    if grp.empty:
        return pd.DataFrame()

    specs = column_specs(source_dir, quarter)
    ym_set = {(y, m): label for (y, m, label) in specs}
    grp = grp[grp.apply(
        lambda r: (int(r["year"]), int(r["month"])) in ym_set, axis=1
    )].copy()
    grp["col_label"] = grp.apply(
        lambda r: ym_set[(int(r["year"]), int(r["month"]))], axis=1
    )

    # Pivot to wide on (canonical_meter_id, building) × col_label
    wide = grp.pivot_table(
        index=["canonical_meter_id", "building"],
        columns="col_label",
        values="reading",
        aggfunc="first",
    ).reset_index()

    # Bring in v4 metadata
    v4_meta = v4[[
        "meter_id", "building_v4", "panel_v4", "location_v4",
        "multiplier_v4", "usage_unit_v4",
    ]].rename(columns={"meter_id": "canonical_meter_id"})
    merged = wide.merge(v4_meta, on="canonical_meter_id", how="left")

    # Per-meter audit-side metadata (within this quarter only)
    audit_meta = (
        grp.groupby(["canonical_meter_id", "building"])
        .agg(
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
        )
        .reset_index()
    )
    merged = merged.merge(audit_meta, on=["canonical_meter_id", "building"], how="left")

    # Flag
    def build_flag(r: pd.Series) -> str:
        flags: list[str] = []
        b_v4 = clean_text(r.get("building_v4"))
        b_au = clean_text(r.get("building"))
        mk = clean_text(r.get("match_kind"))
        if not b_v4 and b_au:
            flags.append("NEW_METER")
        elif b_v4 and b_au and b_v4 != b_au:
            flags.append("BUILDING_MISMATCH")
        if mk == "ZERO_PAD_ADJUSTED":
            flags.append("ZERO_PAD_ADJUSTED")
        return ";".join(flags)

    merged["flag"] = merged.apply(build_flag, axis=1)
    merged["建築物"] = merged["building_v4"].fillna("").astype(str).str.strip()
    merged["建築物"] = merged["建築物"].where(
        merged["建築物"].ne(""),
        merged["building"].fillna("").astype(str).str.strip(),
    )

    # Choose panel/location/multiplier/usage_unit: v4 wins if present, else audit
    def pick(v4_val, audit_val):
        v = clean_text(v4_val)
        return v if v else clean_text(audit_val)

    merged["開關箱"] = merged.apply(lambda r: pick(r["panel_v4"], r["panel_audit"]), axis=1)
    merged["安裝位置"] = merged.apply(lambda r: pick(r["location_v4"], r["location_audit"]), axis=1)
    merged["倍數"] = merged.apply(lambda r: pick(r["multiplier_v4"], r["multiplier_audit"]), axis=1)
    merged["使用單位"] = merged["usage_unit_v4"].fillna("").astype(str)

    # Column order
    reading_cols = [label for (_, _, label) in specs]
    for col in reading_cols:
        if col not in merged.columns:
            merged[col] = ""
    final_cols = [
        "建築物", "canonical_meter_id", "開關箱", "安裝位置", "倍數", "使用單位",
        "flag", "building_v4", "building", "match_kind",
        *reading_cols,
    ]
    final = merged[final_cols].copy().rename(columns={
        "canonical_meter_id": "meter_id",
        "building": "building_audit",
    })

    # Sort by building, then meter_id
    final = final.sort_values(
        ["建築物", "meter_id"], kind="mergesort"
    ).reset_index(drop=True)

    # Insert blank separator rows between distinct buildings
    blank = {col: "" for col in final.columns}
    rows: list[dict] = []
    prev_b: str | None = None
    for _, row in final.iterrows():
        current = clean_text(row["建築物"])
        if prev_b is not None and current != prev_b:
            rows.append(blank.copy())
        rows.append(row.to_dict())
        prev_b = current
    return pd.DataFrame(rows, columns=final.columns)


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
        df = load_audit(path, d)
        audit_frames.append(df)
        print(f"  audit {d}: {len(df)} rows, "
              f"{df['meter_id'].nunique()} unique meters")

    audit = pd.concat(audit_frames, ignore_index=True)
    audit = resolve_canonical(audit, v4)
    print(f"  total audit rows: {len(audit)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    for source_dir in AUDIT_DIRS:
        for quarter in ["Q1", "Q2", "Q3", "Q4"]:
            table = build_quarter_table(audit, v4, source_dir, quarter)
            if table.empty:
                continue
            roc = ROC_LABEL[source_dir]
            fname = f"中大電表資料_{roc}{quarter}_aligned.csv"
            outpath = OUT_DIR / fname
            table.to_csv(outpath, index=False, encoding="utf-8-sig")
            # Stats (exclude blank-separator rows)
            data_rows = table[table["meter_id"].astype(str).str.strip().ne("")]
            n_flag = (data_rows["flag"].astype(str).str.strip().ne("")).sum()
            summary_rows.append({
                "file": fname,
                "rows_data": len(data_rows),
                "rows_total_incl_gaps": len(table),
                "flagged": int(n_flag),
                "buildings": data_rows["建築物"].nunique(),
            })

    summary = pd.DataFrame(summary_rows)
    print()
    print(f"Wrote {len(summary)} per-quarter CSVs to: {OUT_DIR}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
