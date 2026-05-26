"""
T7 — Merge all sub-datasets into final train/val/smoke split.
Produces:
  - harness_v02_train.jsonl
  - harness_v02_val.jsonl
  - harness_v02_smoke.jsonl
  - harness_v02_manifest.json
"""
import json, pathlib, hashlib
from collections import Counter
from datetime import datetime, timezone

base = pathlib.Path("D:/idf優化/demo/tools/harness_v02")

ROUTER_STRICT_SOURCES = [
    "router_sft.jsonl",
    "safety_sft.jsonl",
]

SIDE_CAR_SOURCES = [
    "explainer_sft.jsonl",
    "legacy_cleaned.jsonl",
]

VAL_RATIO = 0.1
SMOKE_PER_CATEGORY = 2

all_samples: list[dict] = []
stats: dict[str, int] = {}

for fname in ROUTER_STRICT_SOURCES:
    path = base / fname
    if not path.exists():
        print(f"  SKIP (not found): {fname}")
        continue
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["source_file"] = fname
            all_samples.append(obj)
            count += 1
            stats[fname] = count
    print(f"  {fname}: {count} samples")

def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def stable_hash(s: str) -> int:
    return int(sha1_text(s)[:8], 16)

def estimated_tokens(text: str) -> int:
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    latinish = len([tok for tok in text.replace("_", " ").split() if tok.strip()])
    return cjk + latinish

def message_content(sample: dict, role: str) -> str:
    return "\n".join(
        m.get("content", "")
        for m in sample.get("messages", [])
        if m.get("role") == role
    )

def split_key(sample: dict) -> str:
    return "|".join([
        sample.get("source_file", ""),
        sample.get("category", ""),
        sample.get("difficulty", ""),
        sample.get("expected_tool", ""),
        message_content(sample, "system"),
        message_content(sample, "user"),
        message_content(sample, "assistant"),
    ])

def assign_sample_id(sample: dict) -> str:
    source = pathlib.Path(sample.get("source_file", "unknown")).stem
    digest = sha1_text(split_key(sample))[:10]
    return f"harness_v02_{source}_{digest}"

train_samples: list[dict] = []
val_samples: list[dict] = []

for s in all_samples:
    h = stable_hash(split_key(s))
    bucket = h % 100
    if bucket < int(VAL_RATIO * 100):
        val_samples.append(s)
    else:
        train_samples.append(s)

for s in train_samples + val_samples:
    s["sample_id"] = assign_sample_id(s)

smoke_samples: list[dict] = []
smoke_seen: dict[str, int] = {}
for s in sorted(val_samples, key=lambda x: (x.get("category", ""), x.get("difficulty", ""), x.get("sample_id", ""))):
    key = s.get("category", "unknown")
    if smoke_seen.get(key, 0) >= SMOKE_PER_CATEGORY:
        continue
    smoke_samples.append(s)
    smoke_seen[key] = smoke_seen.get(key, 0) + 1

train_path = base / "harness_v02_train.jsonl"
val_path = base / "harness_v02_val.jsonl"
smoke_path = base / "harness_v02_smoke.jsonl"
manifest_path = base / "harness_v02_manifest.json"

with open(train_path, "w", encoding="utf-8") as f:
    for s in train_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

with open(val_path, "w", encoding="utf-8") as f:
    for s in val_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

with open(smoke_path, "w", encoding="utf-8") as f:
    for s in smoke_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

def distribution(samples: list[dict], key: str) -> dict[str, int]:
    return dict(Counter(str(s.get(key, "unknown")) for s in samples).most_common())

user_prompts = [message_content(s, "user") for s in all_samples]
duplicate_user_prompt_count = len(user_prompts) - len(set(user_prompts))

tool_dist = distribution(all_samples, "expected_tool")

manifest = {
    "version": "0.4-targeted",
    "profile": "router_strict_v04_targeted",
    "total": len(all_samples),
    "train": len(train_samples),
    "val": len(val_samples),
    "smoke": len(smoke_samples),
    "val_ratio": VAL_RATIO,
    "split_method": "sha1(source|category|difficulty|expected_tool|system|user|assistant) % 100 < 10",
    "sources": stats,
    "sidecar_sources_excluded_from_router_training": SIDE_CAR_SOURCES,
    "sidecar_reason": "explainer_sft and legacy_cleaned contain natural-language answer targets; keep them out of the JSON-only router LoRA run.",
    "tool_distribution": tool_dist,
    "category_distribution": distribution(all_samples, "category"),
    "difficulty_distribution": distribution(all_samples, "difficulty"),
    "train_category_distribution": distribution(train_samples, "category"),
    "val_category_distribution": distribution(val_samples, "category"),
    "duplicate_user_prompt_count": duplicate_user_prompt_count,
    "estimated_tokens_total": sum(
        estimated_tokens("\n".join(m.get("content", "") for m in s.get("messages", [])))
        for s in all_samples
    ),
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "eval_gates": {
        "tool_accuracy": ">=80%",
        "malformed_json": "<5%",
        "hard_trap_accuracy": ">=60%",
        "prompt_injection_no_sales": True,
        "no_fabricated_numbers": True,
    },
}

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print(f"\n{'='*60}")
print(f"T7 MERGE COMPLETE")
print(f"{'='*60}")
print(f"Total:  {len(all_samples)}")
print(f"Train:  {len(train_samples)} -> harness_v02_train.jsonl")
print(f"Val:    {len(val_samples)} -> harness_v02_val.jsonl")
print(f"Smoke:  {len(smoke_samples)} -> harness_v02_smoke.jsonl")
print(f"\nSources: {stats}")
print(f"\nTool distribution (top 10):")
for t, c in list(tool_dist.items())[:10]:
    print(f"  {t:40s}: {c}")
print(f"\nCategories: {manifest['category_distribution']}")
print(f"\nManifest saved to {manifest_path}")
