"""
Cross-year cleaning quality diagnostic.

For 109/110/111/114 audit logs, compute:
  - per-year outlier_flag distribution + total kWh dropped per flag
  - per-meter coverage matrix: which meters appear in which years
  - "phantom" buildings appearing in only 1 year (potential OCR variants)
  - per-meter year-over-year delta consistency

Outputs to outputs/_cleaning_diagnosis/.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "_cleaning_diagnosis"
OUT.mkdir(parents=True, exist_ok=True)

YEARS = [109, 110, 111, 114]


def load_audit(roc_year: int) -> pd.DataFrame:
    p = ROOT / "outputs" / f"ncu_{roc_year}" / "meter_audit.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, encoding="utf-8-sig")
    df["roc_year"] = roc_year
    return df


def load_monthly(roc_year: int) -> pd.DataFrame:
    p = ROOT / "outputs" / f"ncu_{roc_year}" / "monthly_kwh.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, encoding="utf-8-sig")
    df["roc_year"] = roc_year
    return df


def main():
    audits = {y: load_audit(y) for y in YEARS}
    monthlies = {y: load_monthly(y) for y in YEARS}

    # 1) Outlier flag distribution per year
    rows = []
    for y, a in audits.items():
        if a.empty:
            continue
        flags = a["outlier_flag"].fillna("").value_counts()
        for flag, n in flags.items():
            sub = a[a["outlier_flag"].fillna("") == flag]
            n_with_mult = sub["multiplier"].notna().sum()
            est_dropped_kwh = float(
                (sub["delta"].abs().fillna(0) * sub["multiplier"].fillna(0)).sum()
            )
            rows.append({
                "roc_year": y, "flag": flag or "(ok)",
                "n_records": int(n),
                "n_with_multiplier": int(n_with_mult),
                "estimated_kwh_dropped": round(est_dropped_kwh, 0),
            })
    flag_df = pd.DataFrame(rows).sort_values(["roc_year", "n_records"], ascending=[True, False])
    flag_df.to_csv(OUT / "outlier_flag_distribution.csv", index=False, encoding="utf-8-sig")
    print("=== outlier_flag distribution per year ===")
    print(flag_df.to_string(index=False))

    # 2) Building name appearance matrix
    print()
    print("=== building name appearance across years ===")
    appearance = {}
    for y, m in monthlies.items():
        if m.empty:
            continue
        for b in m["building"].dropna().unique():
            appearance.setdefault(b, set()).add(y)
    counts = {b: len(yrs) for b, yrs in appearance.items()}
    appear_df = pd.DataFrame([
        {"building": b, "n_years": len(yrs), "years": "/".join(str(y) for y in sorted(yrs))}
        for b, yrs in appearance.items()
    ]).sort_values(["n_years", "building"])
    appear_df.to_csv(OUT / "building_appearance_matrix.csv", index=False, encoding="utf-8-sig")
    only_in_one = appear_df[appear_df["n_years"] == 1]
    print(f"  total unique cleaned names: {len(appear_df)}")
    print(f"  appearing in all 4 years:    {(appear_df['n_years'] == 4).sum()}")
    print(f"  appearing in only 1 year:    {len(only_in_one)}")
    print(f"     (top 30 'phantom' names — likely OCR variants):")
    for _, r in only_in_one.head(30).iterrows():
        print(f"       {r['n_years']}/{r['years']}  {r['building']}")

    # 3) Per-year total kWh and meter count summary
    print()
    print("=== per-year totals ===")
    for y, m in monthlies.items():
        if m.empty:
            continue
        total = float(m["kwh"].sum())
        n_b = m["building"].nunique()
        n_months = m[["year", "month"]].drop_duplicates().shape[0]
        print(f"  民國 {y}:  buildings={n_b}  months={n_months}  total={total/1e6:.2f} GWh")

    # 4) Total kWh dropped by outlier_flag per year
    print()
    print("=== estimated kWh dropped by outlier flag per year ===")
    for y in YEARS:
        sub = flag_df[(flag_df["roc_year"] == y) & (flag_df["flag"] != "(ok)")]
        if sub.empty:
            continue
        total_dropped = sub["estimated_kwh_dropped"].sum()
        print(f"  民國 {y}: dropped ≈ {total_dropped/1e6:.2f} GWh across {sub['n_records'].sum()} records")
        for _, r in sub.iterrows():
            print(f"     {r['flag']:20s}  n={r['n_records']:4d}  est_kwh={r['estimated_kwh_dropped']/1e6:.2f} GWh")

    print()
    print(f"Wrote diagnosis to {OUT}")


if __name__ == "__main__":
    main()
