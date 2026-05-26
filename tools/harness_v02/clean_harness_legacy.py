"""
T5 — Clean legacy harness data.
Reads harness_output.jsonl, remaps tool names, unifies expected_tool to
top-level, removes duplicates, flags malformed JSON and bad units.
"""
import json, pathlib, sys

LEGACY_MAP = {
    "get_building_detail": "query_energy_records",
    "get_campus_annual_usage": "query_energy_records",
    "query_meter_usage": "query_energy_records",
    "compare_campus_years": "compare_energy_usage",
    "detect_energy_anomaly": "detect_energy_anomalies",
    "run_counterfactual_analysis": "run_counterfactual_for_building",
    "search_knowledge_base": "search_docs",
    "recommend_adaptive_strategy": "recommend_adaptive_strategies",
    "optimize_building_portfolio": "optimize_energy_portfolio",
    "ask_clarifying_question": "__refusal__",
}

SKIP_TOOLS = {"multi_turn", "vision_analysis"}

src = pathlib.Path("D:/idf優化/demo/data/lora/harness_output.jsonl")
out = pathlib.Path("D:/idf優化/demo/tools/harness_v02/legacy_cleaned.jsonl")

if not src.exists():
    print(f"Source not found: {src}")
    sys.exit(1)

cleaned: list[dict] = []
skipped = 0
remapped = 0
seen_queries: set[str] = set()
duplicates = 0

with open(src, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            print(f"  SKIP line {line_no}: malformed JSON")
            skipped += 1
            continue

        meta = obj.get("metadata", {})
        old_tool = meta.get("expected_tool", "")

        if old_tool in SKIP_TOOLS:
            skipped += 1
            continue

        user_msg = ""
        for m in obj.get("messages", []):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break
        q_key = user_msg.strip()[:80]
        if q_key in seen_queries:
            duplicates += 1
            continue
        seen_queries.add(q_key)

        new_tool = LEGACY_MAP.get(old_tool, old_tool)

        if new_tool != old_tool:
            remapped += 1

        obj["expected_tool"] = new_tool
        if "metadata" in obj and "expected_tool" in obj["metadata"]:
            obj["metadata"]["expected_tool_original"] = old_tool
            obj["metadata"]["expected_tool"] = new_tool

        obj["category"] = "legacy_cleaned"
        obj["source"] = "harness_output.jsonl"
        cleaned.append(obj)

out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    for s in cleaned:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"T5 done: {len(cleaned)} cleaned samples -> legacy_cleaned.jsonl")
print(f"  Skipped: {skipped} (skip-tools: multi_turn/vision_analysis + malformed)")
print(f"  Duplicates removed: {duplicates}")
print(f"  Tool names remapped: {remapped}")
