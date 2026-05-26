"""
T6 — Unified eval harness for training-harness v0.2.
Reads router_sft.jsonl, sends user query to local Gemma, parses JSON response,
checks tool accuracy, malformed rate, hard/trap rates.
"""
import json, os, pathlib, sys, time
from typing import Any

try:
    import requests
except ImportError:
    print("pip install requests first")
    sys.exit(1)

GEMMA_BASE_URL = os.getenv("ENERGY_EVAL_BASE_URL", "http://127.0.0.1:8088/v1").rstrip("/")
GEMMA_URL = os.getenv("ENERGY_EVAL_CHAT_URL", f"{GEMMA_BASE_URL}/chat/completions")
GEMMA_MODEL = os.getenv("ENERGY_EVAL_MODEL", "gemma4-e2b-it-router-v04-Q4_K_M.gguf")
MAX_TOKENS = 1024

def call_gemma(messages: list[dict]) -> str:
    payload = {
        "model": GEMMA_MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.1,
    }
    try:
        resp = requests.post(GEMMA_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        content = ""
        for c in data.get("choices", []):
            msg = c.get("message", {})
            content += msg.get("content", "")
        return content.strip()
    except Exception as exc:
        return f"ERROR: {exc}"

def parse_tool_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for i in range(len(text)):
            if text[i] == "{":
                for j in range(len(text), 0, -1):
                    try:
                        return json.loads(text[i:j])
                    except Exception:
                        continue
        return {"tool": "__parse_error__", "raw": text[:200]}

def run_eval(jsonl_path: str, max_samples: int = 0):
    path = pathlib.Path(jsonl_path)
    if not path.exists():
        print(f"File not found: {path}")
        return

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))

    if max_samples > 0:
        samples = samples[:max_samples]

    results = {
        "total": 0,
        "correct": 0,
        "malformed": 0,
        "per_difficulty": {},
        "errors": [],
    }

    for i, s in enumerate(samples):
        expected = s.get("expected_tool", "")
        diff = s.get("difficulty", "unknown")
        user_msgs = [m for m in s.get("messages", []) if m.get("role") == "user"]
        if not user_msgs:
            continue

        sys_msgs = [m for m in s.get("messages", []) if m.get("role") == "system"]
        api_messages = []
        if sys_msgs:
            api_messages.append({"role": "system", "content": sys_msgs[0]["content"]})
        api_messages.append({"role": "user", "content": user_msgs[0]["content"]})

        raw = call_gemma(api_messages)
        parsed = parse_tool_response(raw)
        predicted = parsed.get("tool", "__parse_error__")

        results["total"] += 1
        if predicted == "__parse_error__":
            results["malformed"] += 1
        elif predicted == expected:
            results["correct"] += 1

        d = results["per_difficulty"].setdefault(diff, {"total": 0, "correct": 0, "malformed": 0})
        d["total"] += 1
        if predicted == "__parse_error__":
            d["malformed"] += 1
        elif predicted == expected:
            d["correct"] += 1

        if predicted != expected:
            results["errors"].append({
                "sample_id": s.get("sample_id", i),
                "difficulty": diff,
                "expected": expected,
                "predicted": predicted,
                "user_query": user_msgs[0]["content"][:60],
            })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(samples)}] acc={results['correct']}/{results['total']} = {results['correct']/max(1,results['total'])*100:.1f}%")

        time.sleep(0.3)

    total = max(1, results["total"])
    print(f"\n{'='*60}")
    print(f"EVAL RESULTS: {jsonl_path}")
    print(f"{'='*60}")
    print(f"Total:      {results['total']}")
    print(f"Correct:    {results['correct']} ({results['correct']/total*100:.1f}%)")
    print(f"Malformed:  {results['malformed']} ({results['malformed']/total*100:.1f}%)")
    print()
    for diff, d in sorted(results["per_difficulty"].items()):
        dt = max(1, d["total"])
        print(f"  {diff:12s}: {d['correct']}/{d['total']} = {d['correct']/dt*100:.1f}% acc, {d['malformed']} malformed")

    if results["errors"][:10]:
        print(f"\nFirst 10 errors:")
        for e in results["errors"][:10]:
            print(f"  #{e['sample_id']} [{e['difficulty']}] expected={e['expected']} got={e['predicted']} | {e['user_query']}")

    report_path = pathlib.Path(jsonl_path).parent / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "D:/idf優化/demo/tools/harness_v02/router_sft.jsonl"
    max_n = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    run_eval(target, max_n)
