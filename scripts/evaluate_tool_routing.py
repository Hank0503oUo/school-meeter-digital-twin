"""
evaluate_tool_routing.py
評測微調後的 Gemma 模型在工具路由（tool routing）上的準確率。

用法：
    # 對本機 llama-server 評測
    python scripts/evaluate_tool_routing.py \
        --eval-set data/lora/deepseek_batch_01.jsonl \
        --base-url http://127.0.0.1:8088/v1 \
        --model gemma-4-e2b-it-energy-lora-Q4_K_M.gguf

    # 只評測 routing 類型
    python scripts/evaluate_tool_routing.py \
        --eval-set data/lora/deepseek_batch_01.jsonl \
        --mode routing
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("pip install requests", file=sys.stderr)
    sys.exit(1)


ROUTING_SYSTEM_PROMPT = (
    "你是 NTU 校園能源助理的意圖路由器。根據使用者問題，判斷應該呼叫哪個 MCP 工具，並產生標準化參數。\n\n"
    "輸出格式：JSON {\"tool\": \"...\", \"arguments\": {...}}\n"
    "只輸出 JSON，不要其他文字。"
)


def query_local_model(
    user_query: str,
    base_url: str,
    model: str,
    system_prompt: str = ROUTING_SYSTEM_PROMPT,
    temperature: float = 0.1,
    max_retries: int = 2,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        "temperature": temperature,
        "max_tokens": 512,
    }
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
            if attempt == max_retries:
                return f"ERROR: {e}"
            time.sleep(1)


def parse_tool_from_response(response: str) -> dict:
    response = response.strip()
    if response.startswith("```"):
        lines = response.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        response = "\n".join(lines).strip()
    try:
        parsed = json.loads(response)
        return parsed
    except json.JSONDecodeError:
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
    return {"tool": "__parse_error__", "arguments": {}, "raw": response}


def _messages_by_role(sample: dict, role: str) -> list[dict]:
    return [m for m in sample.get("messages", []) if m.get("role") == role]


def extract_eval_fields(sample: dict) -> tuple[str, str, dict, list[str], str, str]:
    """Support both legacy DeepSeek JSONL and harness_v02 messages JSONL."""
    user_messages = _messages_by_role(sample, "user")
    system_messages = _messages_by_role(sample, "system")
    assistant_messages = _messages_by_role(sample, "assistant")

    query = sample.get("user_query") or (user_messages[0].get("content", "") if user_messages else "")
    system_prompt = system_messages[0].get("content", ROUTING_SYSTEM_PROMPT) if system_messages else ROUTING_SYSTEM_PROMPT
    expected_tool = sample.get("expected_tool", "")
    expected_args = sample.get("expected_arguments", {})

    if (not expected_tool or not expected_args) and assistant_messages:
        parsed_target = parse_tool_from_response(assistant_messages[-1].get("content", ""))
        expected_tool = expected_tool or parsed_target.get("tool", "")
        expected_args = expected_args or parsed_target.get("arguments", {})

    return (
        query,
        expected_tool,
        expected_args,
        sample.get("should_not_use", []),
        sample.get("difficulty", "?"),
        system_prompt,
    )


def evaluate(
    eval_set_path: Path,
    base_url: str,
    model: str,
    mode: str,
    output_path: Path | None,
    verbose: bool,
) -> None:
    samples = []
    with open(eval_set_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    results = []
    tool_correct = 0
    tool_total = 0
    args_correct = 0
    args_total = 0
    avoid_violations = 0
    avoid_total = 0

    for i, sample in enumerate(samples):
        if mode == "routing" and "expected_tool" not in sample:
            continue
        if mode == "qa" and "standard_answer" not in sample:
            continue

        query, expected_tool, expected_args, should_not_use, difficulty, system_prompt = extract_eval_fields(sample)
        if not query:
            continue

        response = query_local_model(query, base_url, model, system_prompt=system_prompt)
        parsed = parse_tool_from_response(response)
        predicted_tool = parsed.get("tool", "UNKNOWN")

        is_tool_correct = predicted_tool == expected_tool
        is_avoid_ok = predicted_tool not in should_not_use

        if expected_tool:
            tool_total += 1
            if is_tool_correct:
                tool_correct += 1
            if expected_args:
                args_total += 1
                if parsed.get("arguments") == expected_args:
                    args_correct += 1
        if should_not_use:
            avoid_total += 1
            if is_avoid_ok:
                avoid_violations += 1

        result = {
            "idx": i,
            "query": query,
            "expected_tool": expected_tool,
            "predicted_tool": predicted_tool,
            "is_correct": is_tool_correct,
            "is_avoid_ok": is_avoid_ok,
            "difficulty": difficulty,
            "response": response,
        }
        results.append(result)

        mark = "v" if is_tool_correct else "X"
        if verbose or not is_tool_correct:
            print(f"[{mark}] #{i} ({difficulty}) '{query[:40]}...' → {predicted_tool} (expected: {expected_tool})")

    print("\n" + "=" * 60)
    print(f"RESULTS: {len(results)} samples evaluated")
    if tool_total:
        print(f"  Tool accuracy:     {tool_correct}/{tool_total} = {tool_correct/tool_total:.1%}")
    if args_total:
        print(f"  Arguments accuracy: {args_correct}/{args_total} = {args_correct/args_total:.1%}")
    if avoid_total:
        print(f"  Avoid violations:  {avoid_violations}/{avoid_total} OK")

    by_diff: dict[str, list] = {}
    for r in results:
        by_diff.setdefault(r["difficulty"], []).append(r)
    for diff, items in sorted(by_diff.items()):
        correct = sum(1 for r in items if r["is_correct"])
        total = len(items)
        print(f"  {diff:8s}: {correct}/{total} = {correct/total:.1%}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nDetailed results saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tool routing accuracy")
    parser.add_argument("--eval-set", required=True, help="Path to eval JSONL")
    parser.add_argument("--base-url", default="http://127.0.0.1:8088/v1")
    parser.add_argument("--model", default="gemma4-e2b-it-router-v04-Q4_K_M.gguf")
    parser.add_argument("--mode", choices=["routing", "qa", "all"], default="routing")
    parser.add_argument("--output", help="Save detailed results to JSONL")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    evaluate(
        eval_set_path=Path(args.eval_set),
        base_url=args.base_url,
        model=args.model,
        mode=args.mode,
        output_path=Path(args.output) if args.output else None,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
