"""
T2f — Assemble all router sub-files into router_sft.jsonl.
Run after T2a~T2e are all generated.
"""
import json, pathlib, sys

base = pathlib.Path("D:/idf優化/demo/tools/harness_v02")
sub_files = [
    "router_easy.jsonl",
    "router_medium.jsonl",
    "router_hard.jsonl",
    "router_trap.jsonl",
    "router_malformed.jsonl",
    "router_v04_targeted.jsonl",
]

all_samples: list[dict] = []
stats: dict[str, int] = {}

for fname in sub_files:
    path = base / fname
    if not path.exists():
        print(f"  MISSING: {fname} — run the corresponding build script first")
        continue
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            all_samples.append(obj)
            count += 1
    diff = fname.replace("router_", "").replace(".jsonl", "")
    stats[diff] = count
    print(f"  {fname}: {count} samples")

sid = 0
for s in all_samples:
    sid += 1
    s["sample_id"] = sid

out_path = base / "router_sft.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for s in all_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"\nTotal: {len(all_samples)} samples -> router_sft.jsonl")
print(f"  Breakdown: {stats}")
print(f"  Target: 300 | Actual: {len(all_samples)} | Delta: {len(all_samples) - 300}")
