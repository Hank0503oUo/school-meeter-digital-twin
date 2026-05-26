"""
Generic NCU electricity meter cleaner — parameterized over 民國 year.

Usage:
    python scripts/ncu_cleaning/clean_ncu_year.py --year 109
    python scripts/ncu_cleaning/clean_ncu_year.py --year 110
    python scripts/ncu_cleaning/clean_ncu_year.py --year 111
    python scripts/ncu_cleaning/clean_ncu_year.py --year 114

Reuses every algorithm from clean_ncu_114.py (rollover handling, OCR digit-shift
detection, virtual/aggregate/physical meter topology, name normalization).

Per-year quirks handled:
  - 109 starts at Q2 (no Q1 file). First valid delta is 2020-05 (May), since
    Q2's first reading (2020-04) has no prior reading to subtract.

Outputs to outputs/ncu_<year>/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(r"G:/我的雲端硬碟/mineru/mineru_output_mineru_vlm/csv_by_quarter")
# Hand-curated NCU-supplied CSVs (e.g. the 109Q1 template predating the MinerU
# pipeline); same schema, much cleaner than OCR output.
EXTRA_SRC_DIR = Path(r"G:/我的雲端硬碟/NUC meter")

# Add scripts dir to import the existing cleaning functions
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import the shared cleaning logic from the original 114 script
import clean_ncu_114 as base  # noqa: E402


def quarter_months_for_year(roc_year: int) -> dict[str, list[tuple[int, int]]]:
    """Return mapping of "Q1".."Q4" → list of (year, month) for the 4 reading
    columns in the source CSV.

    Convention: each quarter's first reading is the LAST month of the previous
    quarter (overlap), and the next 3 readings are months in sequence.
        Q1 → [last yr Dec,    Jan, Feb, Mar]
        Q2 → [Mar,             Apr, May, Jun]
        Q3 → [Jun,             Jul, Aug, Sep]
        Q4 → [Sep,             Oct, Nov, Dec]
    """
    ad_year = 1911 + int(roc_year)
    return {
        "Q1": [(ad_year - 1, 12), (ad_year, 1),  (ad_year, 2),  (ad_year, 3)],
        "Q2": [(ad_year, 3),       (ad_year, 4),  (ad_year, 5),  (ad_year, 6)],
        "Q3": [(ad_year, 6),       (ad_year, 7),  (ad_year, 8),  (ad_year, 9)],
        "Q4": [(ad_year, 9),       (ad_year, 10), (ad_year, 11), (ad_year, 12)],
    }


def find_quarter_files(roc_year: int) -> dict[str, Path]:
    """Locate quarter CSVs that exist on disk for this 民國 year.

    Searches MinerU output first, then falls back to the hand-curated
    NCU-supplied CSV directory (`EXTRA_SRC_DIR`, e.g. `中大電表資料_109Q1_v4.csv`).
    """
    out = {}
    for q in ("Q1", "Q2", "Q3", "Q4"):
        # Primary: MinerU OCR output
        primary = SRC_DIR / f"中大電表資料_{roc_year}{q}_mineru_vlm.csv"
        if primary.exists():
            out[q] = primary
            continue
        # Fallback: hand-curated v* CSV (any version suffix)
        for cand in sorted(EXTRA_SRC_DIR.glob(f"中大電表資料_{roc_year}{q}_*.csv")):
            out[q] = cand
            break
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True,
                        help="民國 year, e.g. 109 110 111 114")
    args = parser.parse_args()

    roc_year = args.year
    ad_year = 1911 + roc_year
    quarters_present = find_quarter_files(roc_year)
    if not quarters_present:
        print(f"[ERROR] No source CSVs found for 民國 {roc_year} in {SRC_DIR}")
        sys.exit(1)

    out_dir = ROOT / "outputs" / f"ncu_{roc_year}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Patch base module's globals so its functions look at this year's files
    qm = quarter_months_for_year(roc_year)
    base.QUARTER_MONTHS = {q: qm[q] for q in quarters_present}
    base.QUARTER_FILES = quarters_present
    base.OUT_DIR = out_dir
    base.SRC_DIR = SRC_DIR

    print(f"=== Cleaning NCU 民國 {roc_year} (西元 {ad_year}) ===")
    print(f"  quarters present: {sorted(quarters_present.keys())}")
    print(f"  output dir: {out_dir}")
    print()

    print("[1/4] Loading quarters…")
    long = base.load_all_quarters()
    print(f"      melt rows: {len(long):,}")

    print("[2/4] Computing per-meter deltas…")
    meter_kwh = base.compute_meter_deltas(long)
    n_rollover = int(meter_kwh["rollover_fix"].sum())
    print(f"      rollover fixes: {n_rollover}, audit rows: {len(meter_kwh):,}")

    print("[3/4] Aggregating per building x month…")
    monthly = base.aggregate_building_monthly(meter_kwh)
    print(f"      buildings: {monthly['building'].nunique()}")
    print(f"      total annual kWh: {monthly['kwh'].sum():,.0f}")

    monthly.to_csv(out_dir / "monthly_kwh.csv", index=False, encoding="utf-8-sig")
    meter_kwh.to_csv(out_dir / "meter_audit.csv", index=False, encoding="utf-8-sig")

    print("[4/4] Writing report…")
    report = base.write_report(long, meter_kwh, monthly)
    report = report.replace(
        "# NCU 114 年電表資料清洗報告",
        f"# NCU 民國 {roc_year} ({ad_year}) 電表資料清洗報告",
    )
    (out_dir / "cleaning_report.md").write_text(report, encoding="utf-8")

    print()
    print("Wrote:")
    print(f"  {out_dir / 'monthly_kwh.csv'}")
    print(f"  {out_dir / 'meter_audit.csv'}")
    print(f"  {out_dir / 'cleaning_report.md'}")


if __name__ == "__main__":
    main()
