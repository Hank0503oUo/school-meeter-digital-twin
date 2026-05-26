"""Build LoRA training candidates from HARNESS event memory.

Reads ``data/knowledge_workbench/state/harness_events.jsonl`` and emits:
- ``memory_router_sft.jsonl`` — router SFT candidates
- ``answer_sft_from_events.jsonl`` — answer SFT candidates
- ``preference_pairs_from_events.jsonl`` — preference pair candidates

Usage::

    python tools/harness_v02/build_from_harness_events.py
    python tools/harness_v02/build_from_harness_events.py --input PATH --output-dir DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.harness_memory import HarnessMemory


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def build_router_sft(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        if "router_sft_candidate" not in event.get("training_tags", []):
            continue
        quality = event.get("quality", {})
        if not quality.get("tool_correct"):
            continue
        tool_plan = event.get("selected_tool_plan", [])
        if not tool_plan:
            continue
        first_tool = tool_plan[0]
        target = {
            "tool": first_tool.get("tool", ""),
            "arguments": first_tool.get("arguments", {}),
        }
        if not target["tool"]:
            continue
        rows.append({
            "messages": [
                {"role": "user", "content": event.get("user_query", "")},
                {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
            ],
            "source": "harness_event",
            "event_id": event.get("event_id", ""),
            "intent": event.get("intent", ""),
        })
    return rows


def build_answer_sft(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        if "answer_sft_candidate" not in event.get("training_tags", []):
            continue
        quality = event.get("quality", {})
        if not quality.get("answer_grounded"):
            continue
        judge = float(quality.get("judge_score", 0))
        if judge < 0.8:
            continue
        summary = event.get("final_answer_summary", "")
        if not summary:
            continue
        tool_data = event.get("tool_trace", [])
        context_parts = []
        for trace in tool_data:
            context_parts.append(
                f"[{trace.get('tool', '')}] {trace.get('summary', '')}"
            )
        context = "\n".join(context_parts) if context_parts else ""
        rows.append({
            "messages": [
                {"role": "user", "content": event.get("user_query", "")},
                {"role": "context", "content": context},
                {"role": "assistant", "content": summary},
            ],
            "source": "harness_event",
            "event_id": event.get("event_id", ""),
            "judge_score": judge,
        })
    return rows


def build_preference_pairs(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    grouped: dict[str, list[dict]] = {}
    for event in events:
        nq = event.get("normalized_query", "") or event.get("user_query", "")
        key = nq.lower().strip()[:80]
        if not key:
            continue
        grouped.setdefault(key, []).append(event)

    for key, group in grouped.items():
        if len(group) < 2:
            continue
        successes = [e for e in group if e.get("outcome") == "success"]
        failures = [e for e in group if e.get("outcome") == "failure"]
        if not successes or not failures:
            continue
        best = max(successes, key=lambda e: float(e.get("quality", {}).get("judge_score", 0)))
        worst = failures[0]
        rows.append({
            "prompt": best.get("user_query", ""),
            "chosen": best.get("final_answer_summary", ""),
            "rejected": worst.get("final_answer_summary", ""),
            "source": "harness_event",
            "chosen_event_id": best.get("event_id", ""),
            "rejected_event_id": worst.get("event_id", ""),
        })
    return rows


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build LoRA training candidates from HARNESS events")
    parser.add_argument("--input", default="", help="Path to harness_events.jsonl")
    parser.add_argument("--output-dir", default="", help="Output directory")
    args = parser.parse_args()

    harness = HarnessMemory()
    events_path = Path(args.input) if args.input else harness.events_path
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent

    events = _load_events(events_path)
    harness.build_training_candidates()
    events = _load_events(events_path)

    print(f"Loaded {len(events)} events from {events_path}")

    router_rows = build_router_sft(events)
    answer_rows = build_answer_sft(events)
    pref_rows = build_preference_pairs(events)

    output_dir.mkdir(parents=True, exist_ok=True)

    router_path = output_dir / "memory_router_sft.jsonl"
    with router_path.open("w", encoding="utf-8") as f:
        for row in router_rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    answer_path = output_dir / "answer_sft_from_events.jsonl"
    with answer_path.open("w", encoding="utf-8") as f:
        for row in answer_rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    pref_path = output_dir / "preference_pairs_from_events.jsonl"
    with pref_path.open("w", encoding="utf-8") as f:
        for row in pref_rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    print(f"Router SFT candidates:  {len(router_rows)} -> {router_path}")
    print(f"Answer SFT candidates:  {len(answer_rows)} -> {answer_path}")
    print(f"Preference pairs:       {len(pref_rows)} -> {pref_path}")


if __name__ == "__main__":
    main()
