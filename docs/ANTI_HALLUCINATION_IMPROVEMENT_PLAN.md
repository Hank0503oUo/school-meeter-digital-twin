# Demo Agent Anti-Hallucination Improvement Plan

## Current Diagnosis

The current v04 local agent is hallucinating because it is being asked to do too many jobs at once:

1. Route the user intent.
2. Resolve the target building/year.
3. Decide whether the question is answerable.
4. Call MCP tools.
5. Read tool JSON.
6. Produce a final narrative.
7. Preserve conversation context.

This is too much trust to place in a small local LoRA model. The most dangerous failure pattern observed is:

> User explicitly asks about Building A, but the agent answers using Building B from recent context or dashboard focus.

Examples:

- User asks `禮賢樓2017的情況`
- Agent answers with `化學館` strategy output
- Cause: context leakage + strategy fallback too eager + missing entity lock

The correct solution is not just more prompt text. The runtime must enforce a stricter contract:

> LLM may choose or explain, but cannot change ground-truth entity, year, tool result, or answerability.

## Core Principle

For demo-critical factual questions, the agent should become:

```text
User query
-> deterministic entity/year/answerability parser
-> deterministic workflow/tool dispatcher
-> MCP tool execution
-> deterministic evidence validator
-> template answer composer
-> optional LLM polish only if evidence is valid
```

The LLM should not be the only component deciding whether a tool result is valid.

## Immediate Rule Changes

### 1. Entity Lock

If the user explicitly names a building, that building becomes locked for the entire turn.

Required behavior:

- `禮賢樓2017的情況` must only answer about `禮賢樓`.
- Recent conversation cannot override explicit building names.
- Dashboard current focus cannot override explicit building names.
- Tool result must be rejected if returned building does not match the locked building.

Implementation target:

- Add `locked_entities` to the local agent turn state.
- Store:

```json
{
  "locked_building_names": ["禮賢樓"],
  "locked_years": [2017],
  "locked_metric": null
}
```

Validation rule:

```text
if locked_building_names exists:
    every factual answer must cite a tool result whose building name matches one locked name
```

### 2. Answerability Gate

Before final response, classify the tool result:

```text
answerable_with_values
answerable_but_zero_or_missing
building_not_found
unsupported_scope
needs_clarification
tool_error
```

For example:

`禮賢樓2017的情況`

Tool result:

```json
{
  "name": "禮賢樓",
  "annual_kwh": 0,
  "mean_kw": 0,
  "eui": 0,
  "meter_name": ""
}
```

Correct classification:

```text
answerable_but_zero_or_missing
```

Correct answer:

```text
有建築紀錄，但沒有有效能源值；不能把 0 解讀成真實耗電為 0。
```

Incorrect answer:

```text
提供其他建築的節能策略。
```

### 3. Deterministic Answer Composer

For the main demo tool families, do not let the model freely narrate. Use deterministic templates first.

Priority tools:

- `query_energy_records`
- `compare_energy_usage`
- `rank_energy_buildings_across_years`
- `recommend_adaptive_strategies`
- `run_pvid`
- `detect_energy_anomalies`

The LLM can polish only after the answer composer has generated a grounded draft.

Recommended mode:

```text
ENERGY_LOCAL_LLM_STRICT_TOOL_SUMMARY=1
```

Default should be on for demo.

### 4. Strategy Guard

`recommend_adaptive_strategies` must not run if the user asks for a specific building/year and that building/year has no valid energy values.

Example:

```text
禮賢樓2017有什麼熱點跟改善嗎
```

Correct flow:

1. Query `禮賢樓` 2017.
2. See annual_kwh / mean_kw / eui are missing or zero.
3. Refuse to infer hotspots.
4. Say what data is missing.

Incorrect flow:

1. Use current focus or previous building.
2. Generate chemical building strategy.

### 5. Conversation Memory Separation

Separate conversation context into two layers:

```text
soft_context: previous user/assistant text
hard_context: locked entity/year/tool evidence from current turn
```

Rules:

- Soft context may help resolve pronouns like `那棟`.
- Soft context cannot override explicit entity names.
- Tool evidence from previous turns cannot be reused unless the current turn says `剛剛那棟` / `上一題`.

## Recommended Runtime Architecture

### Turn State Object

Create a per-turn object before invoking any model:

```json
{
  "query": "禮賢樓2017的情況",
  "locked_entities": {
    "building_names": ["禮賢樓"],
    "years": [2017],
    "metrics": ["annual_kwh", "mean_kw", "eui"]
  },
  "answerability": null,
  "selected_workflow": "single_building_year_status",
  "tool_calls": [],
  "evidence": [],
  "allowed_final_buildings": ["禮賢樓"]
}
```

### Workflow Layer

Add deterministic workflow patterns:

```text
single_building_year_status
building_hotspot_and_improvement
campus_top_energy_buildings
campus_year_compare
building_strategy_plan
document_search
unsupported_or_clarify
```

For `single_building_year_status`:

```text
input: building + year
tool: query_energy_records
required args: buildings, years, metrics
composer: single_building_year_status_answer
```

For `building_hotspot_and_improvement`:

```text
input: building + year + hotspot/improvement
step 1: query_energy_records
step 2: if energy values missing -> stop with missing-data answer
step 3: recommend_adaptive_strategies
step 4: validate returned building matches locked building
composer: strategy_answer
```

## Training Data Implication

v05 should not train the model to directly answer these questions.

Train v05 only for:

```json
{
  "dispatch_type": "workflow",
  "workflow_id": "single_building_year_status",
  "locked_entities": {
    "building_names": ["禮賢樓"],
    "years": [2017]
  }
}
```

Not:

```json
{
  "answer": "禮賢樓 2017 年..."
}
```

The runtime should execute the workflow and compose the grounded answer.

## Evaluation Plan

Add adversarial eval cases that specifically test context leakage:

```text
Turn 1: 化學館怎麼改善？
Turn 2: 禮賢樓2017的情況
Expected: answer only about 禮賢樓, not 化學館.
```

```text
Turn 1: 總圖書館最高耗電
Turn 2: 禮賢樓有什麼熱點？
Expected: if 禮賢樓 data missing, refuse hotspot inference.
```

Metrics:

```text
entity_lock_accuracy
wrong_building_answer_rate
missing_data_refusal_accuracy
tool_result_grounding_accuracy
latency_p95
```

Hard gates:

```text
wrong_building_answer_rate = 0%
missing_data_refusal_accuracy >= 95%
latency_p95 <= 10s for deterministic workflows
```

## Implementation Priority

### Today

1. Keep deterministic fallback for `query_energy_records`.
2. Add entity lock validation for all tool results.
3. Add workflow short-circuit for `building + year`.
4. Prevent strategy tools from running when required energy data is missing.
5. Add context-leak regression tests.

### Next

1. Build `workflow_dispatcher.py`.
2. Move keyword workflow rules out of `lm_studio_client.py`.
3. Add `answer_composer.py`.
4. Add `answerability.py`.
5. Generate v05 data for workflow dispatch only.

### Later

1. Use LoRA only for dispatch classification.
2. Keep final answer composition tool-grounded.
3. Use DeepSeek only as judge / teacher for trace review, not runtime truth source.

## Summary

The effective fix is:

> Do not ask the model to be truthful by instruction. Make the runtime make hallucination structurally hard.

For this demo, the local model should decide less, not more. The model can help choose the workflow, but the runtime must lock entities, validate evidence, classify answerability, and compose factual answers from MCP JSON.
