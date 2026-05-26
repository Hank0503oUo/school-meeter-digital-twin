"""Print uncanonical names per year."""
import sys, pandas as pd
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass
ROOT = Path(__file__).resolve().parents[2]
for y in [109, 110, 111, 114]:
    df = pd.read_csv(ROOT / f"outputs/_cleaning_diagnosis/canonicalization_audit_{y}.csv",
                     encoding="utf-8-sig")
    uncan = df[df["method"] == "uncanonical_kept"]
    print(f"=== ROC {y}: {len(uncan)} uncanonical ===")
    for _, r in uncan.iterrows():
        print(f"  '{r['input_name']}'  ({r['n_records']} rows)")
    print()
