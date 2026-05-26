"""
v2 of the topology-ground-truth scaffolder.

Differences vs the original `scaffold_topology_groundtruth_from_meter_audit.py`:

  * `--year` and `--months` are CLI parameters (no longer hard-coded to 2025 / Jan-Mar).
  * `--quarter` selects which `source_q` value to filter (default Q1).
  * `BF` / `B1` / `地下` no longer count as sub-meter markers in the location
    text — those are floor labels, not meter-hierarchy signals, and the
    previous version was demoting genuine PRIMARY meters located in BF
    electrical rooms.
  * A new attention flag `primary_by_multiplier_fallback` highlights rows
    that became PRIMARY only because the multiplier >= 80 fallback fired.
  * A new attention flag `multi_primary_in_building` highlights buildings
    where >= 2 meters were suggested PRIMARY (verify that they really
    are parallel main panels and not double-counting an upstream total).
  * Buildings with `no_suggested_primary` are printed at the end so you
    know which ones require manual fallback selection.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd


ROLE_ORDER = {
    "PRIMARY": 0,
    "PRIMARY_ALT": 1,
    "SUB": 2,
    "SKIP": 3,
    "": 9,
}


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


def read_csv(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp950"):
        try:
            return pd.read_csv(path, dtype=str, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"Could not decode {path}") from last_error


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def unique_join(values: pd.Series, limit: int = 6) -> str:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = clean_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    if len(cleaned) > limit:
        return " | ".join(cleaned[:limit]) + f" | ...(+{len(cleaned) - limit})"
    return " | ".join(cleaned)


def max_numeric(values: pd.Series) -> str:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return ""
    value = float(numeric.max())
    return str(int(value)) if value.is_integer() else f"{value:g}"


def sum_kwh(values: pd.Series) -> float:
    return float(pd.to_numeric(values, errors="coerce").fillna(0).sum())


def suggest_role(
    meter_id: str, panel: str, meter_kind: str, location: str, multiplier: str
) -> tuple[str, str]:
    text = f"{panel} {location}".upper()
    panel_text = panel.upper()
    kind = meter_kind.lower()
    multiplier_value = pd.to_numeric(pd.Series([multiplier]), errors="coerce").iloc[0]

    # Virtual / placeholder meters --------------------------------------------------
    if meter_id.startswith("A1_") or kind == "aggregate":
        return (
            "PRIMARY_ALT",
            "A1_* / aggregate virtual meter; fallback only if physical PRIMARY missing",
        )
    if meter_id.startswith("VL_") or kind == "placeholder":
        return (
            "SKIP",
            "VL_* / placeholder meter; keep for audit, normally not counted",
        )

    # Explicit MAIN markers ---------------------------------------------------------
    if any(token in text for token in ("總表", "主表", "MAIN", "TOTAL", "總盤", "總開關")):
        return "PRIMARY", "main/total marker in panel or location"
    if "電力盤" in panel or "電燈盤" in panel:
        return (
            "PRIMARY",
            "parallel main panels such as 電力盤 / 電燈盤 should usually both count",
        )

    # Sub-meter markers in panel ----------------------------------------------------
    if any(
        token in panel_text
        for token in ("分錶", "分表", "FDR", "饋線", "MP", "ML", "EL", "AC", "PAC")
    ):
        return "SUB", "feeder/sub-panel/sub-meter marker in panel"
    # Sub-meter markers in panel+location text (NOTE: BF / B1 / 地下 removed —
    # those are floor labels, not hierarchy signals, and were demoting genuine
    # PRIMARY meters located in basement electrical rooms.)
    if any(token in text for token in ("分錶", "分表", "空調", "照明", "電梯", "插座")):
        return "SUB", "detail meter marker in panel/location"

    # Last-resort multiplier fallback. Tagged with "FALLBACK:" so the attention
    # filter can flag these for explicit manual review — they are the most
    # likely false positives in the PRIMARY column.
    if pd.notna(multiplier_value) and multiplier_value >= 80:
        return (
            "PRIMARY",
            "FALLBACK: high multiplier (>=80) but no panel marker; "
            "could be a high-CT sub-meter — MUST VERIFY",
        )

    return "SUB", "unclear meter; defaulting to SUB to avoid double counting until reviewed"


def attention_flags(grouped: pd.DataFrame) -> pd.Series:
    flags_by_building: dict[str, list[str]] = {}
    building_counts = grouped.groupby("building")["meter_id"].count()
    primary_counts = grouped.groupby("building")["role_suggested"].apply(
        lambda s: int((s == "PRIMARY").sum())
    )

    duplicate_meter_buildings = (
        grouped.groupby("meter_id")["building"]
        .nunique()
        .loc[lambda s: s > 1]
        .index.astype(str)
        .tolist()
    )
    duplicate_meter_set = set(duplicate_meter_buildings)

    for building, count in building_counts.items():
        flags: list[str] = []
        if count >= 10:
            flags.append("many_meters")
        if primary_counts.get(building, 0) == 0:
            flags.append("no_suggested_primary")
        if primary_counts.get(building, 0) >= 2:
            flags.append("multi_primary_in_building")
        flags_by_building[building] = flags

    result: list[str] = []
    for _, row in grouped.iterrows():
        flags = list(flags_by_building.get(row["building"], []))
        if row["meter_id"] in duplicate_meter_set:
            flags.append("meter_seen_in_multiple_buildings")
        if row["role_suggested"] == "PRIMARY" and row["q_kwh"] == 0:
            flags.append("primary_zero_kwh")
        if (
            row["role_suggested"] == "PRIMARY"
            and "FALLBACK" in str(row.get("suggestion_reason", ""))
        ):
            flags.append("primary_by_multiplier_fallback")
        result.append(";".join(flags))
    return pd.Series(result, index=grouped.index)


def build_scaffold(
    source: Path, quarter: str, year: int, months: list[int]
) -> pd.DataFrame:
    audit = read_csv(source)
    required = {
        "meter_id",
        "building",
        "panel",
        "meter_kind",
        "location",
        "multiplier",
        "year",
        "month",
        "kwh",
        "source_q",
    }
    missing = sorted(required - set(audit.columns))
    if missing:
        raise ValueError(
            f"{source} is missing required columns: {', '.join(missing)}"
        )

    for col in audit.columns:
        audit[col] = audit[col].map(clean_text)

    q_rows = audit[audit["source_q"].eq(quarter)].copy()
    q_rows = q_rows[q_rows["meter_id"].ne("") & q_rows["building"].ne("")]

    # The quarter source includes a prior-month reading for delta calculation.
    # Presence uses all rows in the quarter; usage stats use the configured
    # (year, months) window only.
    q_rows["year_num"] = pd.to_numeric(q_rows["year"], errors="coerce")
    q_rows["month_num"] = pd.to_numeric(q_rows["month"], errors="coerce")
    usage = q_rows[
        q_rows["year_num"].eq(year) & q_rows["month_num"].isin(months)
    ].copy()
    usage["month_label"] = usage["month_num"].astype("Int64").astype(str)

    stats = usage.groupby(["meter_id", "building"], dropna=False).agg(
        q_months=(
            "month_label",
            lambda s: ",".join(sorted(set(s), key=lambda x: int(x))),
        ),
        q_kwh=("kwh", sum_kwh),
    )

    grouped = q_rows.groupby(["meter_id", "building"], dropna=False).agg(
        panel=("panel", unique_join),
        meter_kind=("meter_kind", unique_join),
        location=("location", unique_join),
        multiplier=("multiplier", max_numeric),
        n_records=("meter_id", "count"),
    )
    grouped = grouped.join(stats, how="left").reset_index()
    grouped["q_months"] = grouped["q_months"].fillna("")
    grouped["q_kwh"] = grouped["q_kwh"].fillna(0).round(3)

    suggestions = grouped.apply(
        lambda row: pd.Series(
            suggest_role(
                row["meter_id"],
                row["panel"],
                row["meter_kind"],
                row["location"],
                row["multiplier"],
            ),
            index=["role_suggested", "suggestion_reason"],
        ),
        axis=1,
    )
    grouped = pd.concat([grouped, suggestions], axis=1)

    grouped["role"] = ""
    grouped["parent_meter_id"] = ""
    grouped["note"] = ""
    grouped["topology_attention"] = attention_flags(grouped)
    grouped["source_file"] = str(source)
    grouped["source_quarter"] = quarter
    grouped["source_year_window"] = f"{year}:{','.join(str(m) for m in months)}"

    grouped["_sort_role"] = grouped["role_suggested"].map(ROLE_ORDER).fillna(9)
    grouped = grouped.sort_values(
        ["building", "_sort_role", "panel", "meter_id"]
    ).drop(columns="_sort_role")

    return grouped[
        [
            "meter_id",
            "building",
            "role",
            "parent_meter_id",
            "note",
            "panel",
            "meter_kind",
            "location",
            "multiplier",
            "q_months",
            "q_kwh",
            "n_records",
            "role_suggested",
            "suggestion_reason",
            "topology_attention",
            "source_file",
            "source_quarter",
            "source_year_window",
        ]
    ]


def write_role_guide(path: Path) -> None:
    rows = [
        {
            "role": "PRIMARY",
            "meaning": "建物加總時要算這顆；同棟可多顆並列加總",
            "when_to_use": "主表、總表、電力盤/電燈盤平行主盤，或確認上游沒有其他主表包含它",
        },
        {
            "role": "SUB",
            "meaning": "子表或下游分表；加總時不要算，避免 double count",
            "when_to_use": "分錶、FDR/饋線、MP/ML/EL/AC 等下游盤",
        },
        {
            "role": "PRIMARY_ALT",
            "meaning": "替代主表；只有 PRIMARY 全缺時才 fallback",
            "when_to_use": "A1_* 虛擬總表、合成總量、或人工判定只可當備援的總表",
        },
        {
            "role": "SKIP",
            "meaning": "保留審計但不納入該棟拓樸",
            "when_to_use": "誤抓、臨時表、共用設施、placeholder、或不屬於該建物",
        },
        {
            "role": "",
            "meaning": "留空表示後續程式可以沿用 role_suggested 或舊邏輯",
            "when_to_use": "還沒人工確認時先留白",
        },
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["role", "meaning", "when_to_use"]
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an editable NCU topology ground-truth template from "
            "meter_audit.csv. Parametric in (year, months, quarter)."
        )
    )
    parser.add_argument("--source", type=Path, required=True, help="path to meter_audit.csv")
    parser.add_argument("--output", type=Path, required=True, help="output scaffold CSV path")
    parser.add_argument("--guide", type=Path, default=None, help="optional role-guide CSV (defaults next to --output)")
    parser.add_argument("--quarter", type=str, default="Q1", help="value of source_q to filter (default Q1)")
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Western calendar year for the usage window (e.g. 2020 for ROC 109)",
    )
    parser.add_argument(
        "--months",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[1, 2, 3],
        help="comma-separated months belonging to the quarter (default 1,2,3)",
    )
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(f"source not found: {args.source}")

    scaffold = build_scaffold(args.source, args.quarter, args.year, args.months)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scaffold.to_csv(args.output, index=False, encoding="utf-8-sig")

    guide_path = args.guide or (args.output.parent / "topology_ground_truth_role_guide.csv")
    write_role_guide(guide_path)

    print(f"Wrote scaffold: {args.output.resolve()}")
    print(f"Wrote role guide: {guide_path.resolve()}")
    print(f"Usage window: year={args.year}  months={args.months}  quarter={args.quarter}")
    print(f"Rows: {len(scaffold)}")
    print(f"Buildings: {scaffold['building'].nunique()}")
    print(f"Meters: {scaffold['meter_id'].nunique()}")
    print()
    print("Suggested roles:")
    print(scaffold["role_suggested"].value_counts().to_string())
    print()

    no_primary = (
        scaffold.groupby("building")["role_suggested"]
        .apply(lambda s: int((s == "PRIMARY").sum()))
        .loc[lambda s: s == 0]
        .index.tolist()
    )
    if no_primary:
        print(
            f"⚠️ {len(no_primary)} buildings have NO suggested PRIMARY — "
            "you MUST manually promote one SUB to PRIMARY or PRIMARY_ALT:"
        )
        for b in no_primary:
            print(f"  - {b}")
        print()

    multi_primary = (
        scaffold.groupby("building")["role_suggested"]
        .apply(lambda s: int((s == "PRIMARY").sum()))
        .loc[lambda s: s >= 2]
        .sort_values(ascending=False)
    )
    if not multi_primary.empty:
        print(
            f"ℹ️  {len(multi_primary)} buildings have ≥2 suggested PRIMARY "
            "(verify they are parallel mains, not double-count of upstream total):"
        )
        print(multi_primary.head(20).to_string())
        print()

    print("Top topology hotspots (by meter count):")
    top = (
        scaffold.groupby("building")
        .agg(
            rows=("meter_id", "count"),
            suggested_primary=("role_suggested", lambda s: int((s == "PRIMARY").sum())),
            suggested_sub=("role_suggested", lambda s: int((s == "SUB").sum())),
            q_kwh=("q_kwh", "sum"),
        )
        .sort_values(["rows", "q_kwh"], ascending=[False, False])
        .head(15)
    )
    print(top.to_string())


if __name__ == "__main__":
    main()
