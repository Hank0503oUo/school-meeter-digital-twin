"""
harness_deepseek_gemma.py
DeepSeek ↔ Gemma 對話 harness，自動產生 LoRA 訓練資料。

流程：
1. DeepSeek 出題（多角色、多難度）
2. 題目丟給本地 Gemma 回答
3. DeepSeek 評分 + 修正回答
4. 修正後的 Q&A 存成 JSONL → 丟 Colab Unsloth 訓練

用法：
    python scripts/harness_deepseek_gemma.py \
        --deepseek-key sk-xxx \
        --gemma-url http://127.0.0.1:8088/v1 \
        --topics routing,qa,vision \
        --rounds 3 \
        --output data/lora/harness_output.jsonl

前置：
1. 本機 Gemma 已啟動（start_local_gemma.ps1，Vulkan GPU）
2. DeepSeek API key（https://platform.deepseek.com/api_keys）
3. pip install openai requests
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import requests
    import openai
except ImportError:
    print("pip install openai requests", file=sys.stderr)
    sys.exit(1)

GEMMA_SYSTEM_PROMPT = (
    "你是 NTU 校園能源助理。回答時：\n"
    "1. 數字必須來自工具或儀表 snapshot，不可憑空捏造\n"
    "2. 回答格式：結論 → 依據（數據） → 假設與限制 → 建議\n"
    "3. 如果沒有資料，明確告知使用者\n"
    "4. 使用繁體中文回答"
)

DEEPSEEK_SYSTEM_PROMPT = (
    "你是一個測試工程師，負責幫 NTU 校園能源管理系統產生測試資料。\n"
    "你會：\n"
    "1. 模擬真實使用者問能源相關問題\n"
    "2. 檢查系統回答的品質\n"
    "3. 如果回答有誤，產生修正版本\n"
    "4. 所有輸出嚴格 JSON Lines 格式"
)

CAMPUS_DATA = (
    "示例校區建築列表（合成資料，僅供 harness 測試）：\n"
    "- 示例建築A (DEMO_A)，面積 1500 m²，3F，Administration，年用電 120,000.0 kWh，"
    "平均 13.7 kW，EUI 80.0，R²=0.72，CV(RMSE)=14.0%，energy_tier NORMAL\n"
    "- 示例建築B (DEMO_B)，面積 2000 m²，5F，Academic Units，年用電 240,000.0 kWh，"
    "平均 27.4 kW，EUI 120.0，energy_tier HIGH\n"
    "- 示例建築C (DEMO_C)，面積 1000 m²，2F，Library，年用電 90,000.0 kWh，"
    "平均 10.3 kW，EUI 90.0，R²=0.65，CV(RMSE)=18.0%，energy_tier NORMAL\n"
    "校園統計：3 棟，年總用電 450,000.0 kWh，平均 17.1 kW，最高 27.4 kW\n"
)

MCP_TOOLS = (
    "系統 MCP 工具：\n"
    "compare_campus_years, get_campus_annual_usage, compare_building_trends, "
    "get_building_detail, query_meter_usage, detect_energy_anomaly, get_kpi_summary, "
    "recommend_adaptive_strategy, search_knowledge_base, run_counterfactual_analysis, "
    "run_openbse_simulation, optimize_building_portfolio, render_energy_map, "
    "query_strategy_memory, ask_clarifying_question"
)

USER_ROLES = [
    {"role": "總務處能源管理員", "style": "專業術語，關心全校數據與異常"},
    {"role": "系館承辦人", "style": "只關心自己那棟樓，問 EUI 和節能改善"},
    {"role": "校方主管", "style": "看 KPI 趨勢，問投資優先順序"},
    {"role": "顧問公司工程師", "style": "做 counterfactual 和物理模擬，會問假設情境"},
    {"role": "稽核人員", "style": "查法規依據、改善計畫合理性"},
    {"role": "一般學生", "style": "口語、模糊、不完整，可能問奇怪問題"},
]


def query_gemma(
    question: str,
    base_url: str,
    model: str = "",
    temperature: float = 0.3,
    timeout: float = 60.0,
) -> str:
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": GEMMA_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "temperature": temperature,
        "max_tokens": 512,
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[GEMMA_ERROR] {e}"


def deepseek_generate_questions(
    client: openai.OpenAI,
    topic: str,
    count: int = 10,
    role_override: str | None = None,
) -> list[dict]:
    role_info = role_override or "輪流使用各種角色"
    prompt = (
        f"請產生 {count} 筆測試問題，主題：{topic}。\n\n"
        f"系統資料：\n{CAMPUS_DATA}\n\n"
        f"{MCP_TOOLS}\n\n"
        f"使用者角色：{role_info}\n"
        f"語氣要口語、自然、像真人問的。\n\n"
        f"每筆嚴格一行 JSON：\n"
        f'{{"user_role":"...","user_query":"...","expected_tool":"...","difficulty":"easy|medium|hard|trap"}}\n'
        f"不要 markdown，不要解釋，直接輸出 {count} 行 JSONL。"
    )
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=8000,
    )
    raw = resp.choices[0].message.content or ""
    questions = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            questions.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return questions


def deepseek_evaluate_and_fix(
    client: openai.OpenAI,
    question: str,
    gemma_answer: str,
    expected_tool: str = "",
) -> dict:
    prompt = (
        f"你是一個能源助理回答品質評審。\n\n"
        f"使用者問題：{question}\n\n"
        f"預期使用的工具：{expected_tool or '未指定'}\n\n"
        f"系統回答：\n{gemma_answer}\n\n"
        f"系統資料：\n{CAMPUS_DATA}\n\n"
        f"請評估並輸出嚴格 JSON（一行）：\n"
        f"{{"
        f'"score": 1-5, '
        f'"issues": ["問題1", ...], '
        f'"corrected_answer": "修正後的標準回答", '
        f'"tool_correct": true/false, '
        f'"numbers_correct": true/false'
        f"}}\n"
        f"修正回答必須：數字精確、格式為結論→依據→假設→建議、繁體中文。"
    )
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1500,
    )
    raw = resp.choices[0].message.content or ""
    for line in raw.strip().split("\n"):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {
        "score": 0,
        "issues": ["parse_error"],
        "corrected_answer": raw,
        "tool_correct": False,
        "numbers_correct": False,
    }


def run_harness(
    deepseek_key: str,
    gemma_url: str,
    gemma_model: str,
    topics: list[str],
    rounds: int,
    questions_per_round: int,
    output_path: Path,
    score_threshold: int,
) -> None:
    client = openai.OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")

    all_samples: list[dict] = []
    stats = {"total": 0, "good": 0, "fixed": 0, "failed": 0}

    for round_num in range(1, rounds + 1):
        print(f"\n{'='*60}")
        print(f"Round {round_num}/{rounds}")
        print(f"{'='*60}")

        for topic in topics:
            print(f"\n--- Topic: {topic} ---")
            questions = deepseek_generate_questions(client, topic, questions_per_round)
            print(f"Generated {len(questions)} questions")

            for i, q in enumerate(questions):
                query = q.get("user_query", "")
                if not query:
                    continue
                role = q.get("user_role", "unknown")
                expected = q.get("expected_tool", "")
                difficulty = q.get("difficulty", "medium")

                print(f"\n  [{i+1}/{len(questions)}] ({role}, {difficulty}) {query[:50]}...")

                gemma_answer = query_gemma(query, gemma_url, gemma_model)
                if gemma_answer.startswith("[GEMMA_ERROR]"):
                    print(f"    ⚠ Gemma error: {gemma_answer}")
                    continue

                evaluation = deepseek_evaluate_and_fix(
                    client, query, gemma_answer, expected
                )
                score = evaluation.get("score", 0)
                corrected = evaluation.get("corrected_answer", "")
                issues = evaluation.get("issues", [])

                stats["total"] += 1

                if score >= score_threshold and corrected:
                    sample = {
                        "messages": [
                            {"role": "system", "content": GEMMA_SYSTEM_PROMPT},
                            {"role": "user", "content": query},
                            {"role": "assistant", "content": corrected},
                        ],
                        "metadata": {
                            "round": round_num,
                            "topic": topic,
                            "user_role": role,
                            "difficulty": difficulty,
                            "score": score,
                            "gemma_raw_score": score,
                            "issues": issues,
                            "expected_tool": expected,
                        },
                    }
                    all_samples.append(sample)
                    if score >= 4:
                        stats["good"] += 1
                        print(f"    ✓ Score {score}/5 (direct)")
                    else:
                        stats["fixed"] += 1
                        print(f"    ~ Score {score}/5 (fixed by DeepSeek)")
                else:
                    stats["failed"] += 1
                    print(f"    ✗ Score {score}/5 (skipped, issues: {issues})")

                time.sleep(0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            out = {"messages": s["messages"]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    meta_path = output_path.with_suffix(".meta.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"HARNESS COMPLETE")
    print(f"{'='*60}")
    print(f"Total questions:  {stats['total']}")
    print(f"Good (≥4):        {stats['good']}")
    print(f"Fixed (<4):       {stats['fixed']}")
    print(f"Failed:           {stats['failed']}")
    print(f"Training samples: {len(all_samples)}")
    print(f"Output:           {output_path}")
    print(f"Metadata:         {meta_path}")
    print(f"Cost estimate:    ~${stats['total'] * 0.005:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek ↔ Gemma conversation harness")
    parser.add_argument("--deepseek-key", required=True, help="DeepSeek API key")
    parser.add_argument("--gemma-url", default="http://127.0.0.1:8088/v1", help="Local Gemma URL")
    parser.add_argument("--gemma-model", default="", help="Model name (auto-detect if empty)")
    parser.add_argument("--topics", default="routing,qa,multiturn,trap", help="Comma-separated topics")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds")
    parser.add_argument("--questions-per-round", type=int, default=10, help="Questions per topic per round")
    parser.add_argument("--output", default="data/lora/harness_output.jsonl", help="Output file")
    parser.add_argument("--score-threshold", type=int, default=2, help="Min score to include (1-5)")
    args = parser.parse_args()

    topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent.parent / output_path

    run_harness(
        deepseek_key=args.deepseek_key,
        gemma_url=args.gemma_url,
        gemma_model=args.gemma_model,
        topics=topics,
        rounds=args.rounds,
        questions_per_round=args.questions_per_round,
        output_path=output_path,
        score_threshold=args.score_threshold,
    )


if __name__ == "__main__":
    main()
