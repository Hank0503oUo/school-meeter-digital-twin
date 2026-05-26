"""Show canonicalization audit method distribution per year."""
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
    print(f"=== ROC {y} ===")
    print(df["method"].value_counts().to_string())
    print()
    fp = df[df["method"] == "fuzzy_partial"].sort_values("score")
    if len(fp):
        print(f"  fuzzy_partial matches (low → high score):")
        for _, r in fp.iterrows():
            print(f"    score={r['score']:5.0f}  '{r['input_name']}'  ->  '{r['canonical_name']}'")
    print()
