"""
convert_deepseek_to_lora.py
將 DeepSeek 產生的測試集 JSONL 轉換為 LoRA 微調用的 chat format JSONL。

用法：
    python scripts/convert_deepseek_to_lora.py \
        --input data/lora/deepseek_batch_01.jsonl \
        --output data/lora/lora_train.jsonl \
        --mode routing          # routing | qa | both

三種 mode：
    routing  — 只有 tool routing 題（expected_tool → 訓練模型選工具）
    qa       — 只有 Q&A 題（standard_answer → 訓練模型回答品質）
    both     — 合併兩種（預設）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SYSTEM_PROMPT_ROUTING = (
    "你是 NTU 校園能源助理的意圖路由器。根據使用者問題，判斷應該呼叫哪個 MCP 工具，並產生標準化參數。\n\n"
    "可用工具：\n"
    "compare_campus_years(years, campus, metric)\n"
    "get_campus_annual_usage(campus, year, metric)\n"
    "compare_building_trends(building_id, years, metric)\n"
    "get_building_detail(building_id, year)\n"
    "query_meter_usage(meter_id, date_range)\n"
    "detect_energy_anomaly(building_id, year, sensitivity)\n"
    "get_kpi_summary(campus, year, kpi_type)\n"
    "recommend_adaptive_strategy(building_id, scenario)\n"
    "search_knowledge_base(query, collection)\n"
    "run_counterfactual_analysis(building_id, adjustments)\n"
    "run_openbse_simulation(building_id, params)\n"
    "optimize_building_portfolio(campus, budget, strategy)\n"
    "render_energy_map(campus, year, layer)\n"
    "query_strategy_memory(building_id)\n"
    "ask_clarifying_question(reason)\n\n"
    "輸出格式：JSON {\"tool\": \"...\", \"arguments\": {...}}\n"
    "只輸出 JSON，不要其他文字。"
)

SYSTEM_PROMPT_QA = (
    "你是 NTU 校園能源助理。回答時：\n"
    "1. 數字必須來自工具或儀表 snapshot，不可憑空捏造\n"
    "2. 回答格式：結論 → 依據（數據） → 假設與限制 → 建議\n"
    "3. 如果沒有資料，明確告知使用者\n"
    "4. 使用繁體中文回答"
)


def convert_routing(sample: dict) -> dict | None:
    if "expected_tool" not in sample:
        return None
    tool = sample["expected_tool"]
    args = sample.get("expected_arguments", {})
    assistant_msg = json.dumps({"tool": tool, "arguments": args}, ensure_ascii=False)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_ROUTING},
            {"role": "user", "content": sample["user_query"]},
            {"role": "assistant", "content": assistant_msg},
        ],
        "type": "routing",
        "difficulty": sample.get("difficulty", "medium"),
    }


def convert_qa(sample: dict) -> dict | None:
    answer = sample.get("standard_answer")
    if not answer:
        return None
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_QA},
            {"role": "user", "content": sample["user_query"]},
            {"role": "assistant", "content": answer},
        ],
        "type": "qa",
        "difficulty": sample.get("difficulty", "medium"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DeepSeek JSONL to LoRA training format")
    parser.add_argument("--input", required=True, help="Input JSONL from DeepSeek")
    parser.add_argument("--output", required=True, help="Output JSONL for LoRA training")
    parser.add_argument("--mode", choices=["routing", "qa", "both"], default="both", help="Conversion mode")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Skip samples below this difficulty-weighted confidence")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_samples: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                raw_samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: skipping line {line_num}: {e}", file=sys.stderr)

    converted: list[dict] = []
    stats = {"routing": 0, "qa": 0, "skipped": 0}

    for sample in raw_samples:
        added = False
        if args.mode in ("routing", "both"):
            item = convert_routing(sample)
            if item is not None:
                converted.append(item)
                stats["routing"] += 1
                added = True
        if args.mode in ("qa", "both"):
            item = convert_qa(sample)
            if item is not None:
                converted.append(item)
                stats["qa"] += 1
                added = True
        if not added:
            stats["skipped"] += 1

    with open(output_path, "w", encoding="utf-8") as f:
        for item in converted:
            out = {"messages": item["messages"]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Input:  {len(raw_samples)} samples from {input_path}")
    print(f"Output: {len(converted)} samples → {output_path}")
    print(f"  routing: {stats['routing']}")
    print(f"  qa:      {stats['qa']}")
    print(f"  skipped: {stats['skipped']}")


if __name__ == "__main__":
    main()
