"""Print phantom building names + their meter audit context."""
import sys
import pandas as pd
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

ROOT = Path(__file__).resolve().parents[2]
appear = pd.read_csv(ROOT / "outputs/_cleaning_diagnosis/building_appearance_matrix.csv",
                      encoding="utf-8-sig")
phantoms = appear[appear["n_years"] == 1].copy()
print(f"phantom names ({len(phantoms)}):")
for _, r in phantoms.iterrows():
    yr = int(r["years"])
    audit = pd.read_csv(ROOT / f"outputs/ncu_{yr}/meter_audit.csv",
                        encoding="utf-8-sig")
    sub = audit[audit["building"] == r["building"]]
    n_meters = sub["meter_id"].nunique()
    n_records = len(sub)
    n_kwh = float(sub["kwh"].fillna(0).sum())
    sample_meters = list(sub["meter_id"].unique())[:5]
    print(f"  [year {yr}] {r['building']!r}")
    print(f"      meters={n_meters} records={n_records} kWh={n_kwh:.0f}")
    print(f"      sample meter_ids: {sample_meters}")
