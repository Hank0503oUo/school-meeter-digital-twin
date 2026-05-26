"""Inspect outliers in cleaned audit to understand rollover/multiplier failures."""
import pandas as pd
from pathlib import Path

OUT = Path(r"D:/idf優化/demo/outputs/ncu_114")
a = pd.read_csv(OUT / "meter_audit.csv", encoding="utf-8-sig")

# Top single-month kwh records
print("=== top 20 single-month kWh meter records ===")
top = a.nlargest(20, "kwh")[["meter_id", "building", "panel", "multiplier",
                              "year", "month", "reading", "prev_reading",
                              "delta", "kwh", "rollover_fix", "outlier_flag"]]
top.to_csv(OUT / "_top_kwh_outliers.csv", index=False, encoding="utf-8-sig")
print(top.to_string(index=False))

print()
print("=== multiplier distribution ===")
print(a["multiplier"].describe())

print()
print("=== top 20 deltas (raw) ===")
top_d = a.nlargest(20, "delta")[["meter_id", "building", "multiplier",
                                  "reading", "prev_reading", "delta", "kwh",
                                  "rollover_fix", "year", "month"]]
top_d.to_csv(OUT / "_top_delta_outliers.csv", index=False, encoding="utf-8-sig")
print(top_d.to_string(index=False))

print()
print("=== 國鼎光電大樓 full timeline ===")
gd = a[a["building"].astype(str).str.contains("國鼎光電", na=False)]
gd.to_csv(OUT / "_guoding_audit.csv", index=False, encoding="utf-8-sig")
print(f"  {len(gd)} rows, {gd['meter_id'].nunique()} meters")
print(gd[["meter_id", "multiplier", "year", "month", "reading", "prev_reading",
          "delta", "kwh", "rollover_fix"]].to_string(index=False))
