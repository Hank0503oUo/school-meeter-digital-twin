# V7 to V8 Improvement Plan

## Goal

This document summarizes the v7 evaluation outcome and defines the next round of fixes for v8.

The short version:

> v7 is better than v6 at producing structured dispatch output, but it now overuses `clarify_needed`, still confuses `no_evidence`, and has not learned a stable boundary between `single_tool` and `workflow_chain`.

The next round should focus on:

1. tightening the dispatch schema contract,
2. cleaning the answerability boundary,
3. separating `clarify_needed` from `workflow_chain`,
4. improving `no_evidence` handling,
5. keeping format stability while raising workflow accuracy.

---

## v7 Summary

### Main evaluation result

Relevant files:

- `G:\我的雲端硬碟\energy_lora_router_v07\outputs\gemma_dispatch_v07\eval\v07_final_selection_report.json`
- `G:\我的雲端硬碟\energy_lora_router_v07\outputs\gemma_dispatch_v07\eval\v07_final_val_eval_summary.json`
- `G:\我的雲端硬碟\energy_lora_router_v07\outputs\gemma_dispatch_v07\eval\v07_final_val_confusion_matrix.csv`
- `G:\我的雲端硬碟\energy_lora_router_v07\outputs\gemma_dispatch_v07\eval\v07_final_format_smoke_eval_summary.json`

### Key numbers

From `v07_final_selection_report.json`:

- selection score: `0.2637`
- answerability accuracy: `29.9%` (`44/147`)
- tool accuracy on answerable: `18.4%`
- refusal correctness: `30.0%`
- over-refusal on answerable: `0`

From `v07_final_val_eval_summary.json`:

- overall accuracy: `24.5%` (`36/147`)
- malformed rate: `20.4%` (`30/147`)
- workflow accuracy: `31.3%`
- parse error rate is still above target

From `v07_final_format_smoke_eval_summary.json`:

- format smoke accuracy: `37.5%`
- malformed rate: `25.0%`
- over-refusal: `0`

### What improved

Compared with v6:

- format collapse improved a lot
- the model is more willing to emit structured JSON
- `single_tool` and `clarify_needed` are both recognized more consistently
- `unsupported_scope` remains stable

### What is still broken

The biggest failure mode is now:

> the model defaults to `clarify_needed` too often, including cases that should be `workflow_chain`.

This shows up clearly in the confusion matrix:

- `workflow_chain -> clarify_needed : 30`
- `workflow_chain -> __parse_error__ : 16`
- `single_tool -> clarify_needed : 12`
- `no_evidence -> __parse_error__ : 10`
- `no_evidence -> single_tool : 7`

So v7 is no longer mostly a format failure.
It is now a schema boundary failure.

---

## Root Cause Analysis

## 1. `clarify_needed` is over-triggered

The model has learned to be cautious, but the caution is too strong.

Examples of failure pattern:

- questions that should become `workflow_chain` get collapsed into `clarify_needed`
- single-tool questions with enough context still get demoted into clarification
- document and strategy tasks are treated as if they are too ambiguous to act on

This means v7 is biased toward:

```text
if uncertain -> clarify
```

instead of:

```text
if enough context -> dispatch workflow
if insufficient context -> clarify
```

The model is not learning the threshold well.

---

## 2. `no_evidence` is still not learned

The validation split shows:

- `no_evidence_expected : 0/21`

This is one of the most important gaps in the whole pipeline.

The model still cannot reliably distinguish:

- "I can answer this with tools"
- "I can try, but there is no evidence in the corpus"
- "I need clarification first"

That boundary must be explicit in v8.

---

## 3. `workflow_chain` vs `single_tool` is still unstable

This is the second major issue.

Observed pattern:

- strategy tasks often collapse into `single_tool`
- some counterfactual tasks get treated as a single tool call
- some document tasks are not recognized as multi-step workflows

This suggests the model does not yet have a reliable mental model of:

```text
single_tool = one tool is enough
workflow_chain = need planning, evidence gathering, or multi-step validation
```

---

## 4. Format is better, but still not safe enough

The parse error rate improved relative to v6, but `20.4%` malformed output is still too high for a production dispatch model.

That means the next round must still keep:

- strict JSON output
- small format curriculum
- format smoke gating

Do not drop format work yet.

---

## 5. Answerability labels still need cleanup

Some boundary cases are likely causing unnecessary confusion:

- `ambiguous_reference`
- `missing_required_arguments`
- `no_evidence_expected`
- `unsupported_scope`

If those are not labeled consistently, the model will keep learning a blurry decision boundary.

---

## v8 Goals

v8 should not be "more data only".
It should be a boundary-repair round.

### Minimum targets

- parse error rate below `10%`
- format smoke accuracy above `90%`
- answerability accuracy above `70%`
- `clarify_needed` precision/recall clearly separated from `workflow_chain`
- `no_evidence` accuracy above `80%`
- workflow accuracy above `70%` on the core workflows

### Practical targets

- `single_tool` should stop bleeding into `clarify_needed`
- `workflow_chain` should stop collapsing into `clarify_needed`
- `document_search_dci` should behave as a real workflow, not a fallback clarification
- `building_strategy_plan` should stay multi-step, not become one-tool advice

---

## v8 Required Fixes

## A. Rewrite the system prompt

The system prompt must become a strict dispatch contract, not a general energy assistant prompt.

It should clearly say:

1. output only one JSON object,
2. do not emit prose, markdown, or explanations,
3. choose exactly one dispatch type,
4. use the schema consistently,
5. distinguish `clarify_needed`, `no_evidence`, `refusal`, `single_tool`, and `workflow_chain`.

The prompt should include a short decision rubric like:

- `clarify_needed`: missing required building/year/metric/context
- `no_evidence`: theoretically answerable, but corpus evidence is missing
- `refusal`: unsafe or unsupported
- `single_tool`: one tool is enough and the answer is likely direct
- `workflow_chain`: requires multiple steps, evidence gathering, or validation

---

## B. Make `clarify_needed` stricter

In v8, `clarify_needed` should only be used when the current input truly lacks required arguments.

It should not be used as a safe default for hard questions.

Useful negative examples to add:

- `PI-VD 模型說明文件` should not become `clarify_needed`
- `<BUILDING_B> 的節能改造優先順序` should not become `clarify_needed`
- `<BUILDING_B> 有哪些可行的節能方案` should not become `clarify_needed`

Those should go to workflow dispatch, even if the workflow later determines it needs evidence.

---

## C. Clean `no_evidence` vs `clarify_needed`

This is one of the most important data fixes.

Rule:

- `clarify_needed` = missing user input
- `no_evidence` = enough input exists, but evidence is missing

Examples:

- `然後呢` -> `clarify_needed`
- `<BUILDING_F> 2015 的用電` -> likely `no_evidence`
- `CV-RMSE 的定義在哪份文件` -> `workflow_chain(document_search_dci)`
- `幫我找節能方法` -> `clarify_needed` only if no scope is given, otherwise workflow

If these are mixed, the model will keep over-clarifying.

---

## D. Rebalance workflow training

v8 should add more contrast pairs between:

- `clarify_needed` vs `workflow_chain`
- `single_tool` vs `workflow_chain`
- `no_evidence` vs `clarify_needed`
- `refusal` vs `no_evidence`

The model needs to see the same surface phrasing in different labels.

Suggested pairs:

- `夏季 <BUILDING_B> 的節能策略`
- `<BUILDING_B> 的節能改造優先順序`
- `<BUILDING_B> 照明降 30% 一年省多少`
- `CV-RMSE 的定義在哪份文件`
- `PI-VD 模型說明文件`

These should be labeled so the model learns when to plan, when to clarify, and when to say no evidence.

---

## E. Stabilize workflow IDs

The workflow taxonomy must remain abstract and consistent.

The model should not have to guess whether a workflow ID is:

- a tool name,
- a subtask,
- or a final answer.

Keep workflow IDs stable and task-like:

- `single_building_year_status`
- `building_strategy_plan`
- `counterfactual_saving_estimate`
- `campus_top_energy_buildings`
- `campus_year_compare`
- `document_search_dci`

Do not let workflow IDs drift toward tool names.

---

## F. Keep format smoke in front

The format smoke test should remain a hard gate.

Before any full retrain, run a tiny smoke set that checks:

- valid JSON
- required keys present
- enums in range
- no prose fallback

If format smoke is bad, stop there and fix prompt or examples first.

---

## v8 Training Order

### Phase 1: Prompt and schema alignment

1. rewrite system prompt to strict dispatch mode
2. add explicit decision rubric
3. include all valid tools and workflows

### Phase 2: Data cleanup

1. relabel ambiguous boundary cases
2. split `clarify_needed` from `no_evidence`
3. review `unsupported_scope` and `unsafe_operation`

### Phase 3: Contrastive augmentation

1. add more `workflow_chain` vs `clarify_needed` pairs
2. add more `single_tool` vs `workflow_chain` pairs
3. add more `no_evidence` examples where evidence is missing but scope is valid

### Phase 4: Smoke validation

1. run format smoke
2. check parse rate
3. only then run full validation

### Phase 5: Full v8 training

1. train on the cleaned v8 split
2. evaluate on smoke
3. evaluate on full val

---

## What Not To Do

Do not start with:

- larger GPU
- longer epochs
- more generic data
- more natural-language explanation style
- adding extra output prose

Those will not fix the current problem.

The problem is not capacity.
The problem is the boundary logic.

---

## Suggested Acceptance Criteria

v8 should be considered a meaningful improvement only if:

- `parse_error_rate < 10%`
- `format_smoke_accuracy > 90%`
- `clarify_needed` no longer dominates workflow questions
- `no_evidence_expected` becomes reliably correct
- `workflow_chain` beats `clarify_needed` on strategy/document/counterfactual questions
- overall selection score is clearly above v7

---

## One-Line Summary

v7 shows the model can speak the dispatch format, but it still confuses when to clarify, when to plan, and when to report no evidence. v8 should therefore focus on boundary repair, not just more training.
