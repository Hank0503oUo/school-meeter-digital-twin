# v5 DeepSeek Dispatch Data Generation Prompt

這份文件是給 DeepSeek / teacher model 用的資料生成任務書。目標不是再產生 v04 那種單純 `expected_tool` 選擇題，而是產生 v5 用的 **Agent Dispatch Training** 資料。

核心改變：

```text
v04: 使用者問題 -> 選一個 expected_tool
v5 : 使用者問題 -> 判斷 dispatch_type / workflow_id / entity lock / answerability / required_tools
```

v5 LoRA 不負責直接回答能源數字，也不負責自由生成節能建議。它只學會把問題分流到正確的 runtime workflow。真正的工具執行、證據驗證與回答組裝由 demo runtime 處理。

---

## DeepSeek 任務

請根據下方 schema 生成 JSONL 訓練資料。每一行必須是一個完整 JSON object，不要輸出 Markdown，不要輸出程式碼區塊，不要補充解釋。

你要生成的不是問答，而是「調度標註」。

每筆資料都要回答：

1. 這題是單工具、工具鍊、需要追問、查無證據、拒答，還是超出能力？
2. 使用者明確鎖定了哪些建築、年份、指標？
3. 如果可以處理，應該走哪個 workflow？
4. 需要哪些工具，順序是什麼？
5. 什麼情況下 runtime 應該停止，不讓 LLM 硬掰？

---

## 嚴格輸出 Schema

每一行 JSONL 必須符合這個結構：

```json
{
  "sample_id": "v5_dispatch_000001",
  "user_role": "operator|energy_manager|researcher|facility_staff|student",
  "conversation_context": [],
  "user_query": "使用者原句",
  "expected": {
    "dispatch_type": "single_tool|workflow_chain|clarify_needed|no_evidence|refusal",
    "workflow_id": "single_building_year_status|building_hotspot_improvement|campus_top_energy_buildings|campus_year_compare|building_strategy_plan|counterfactual_saving_estimate|document_search_dci|harness_wiki_event|none",
    "answerability": "answerable_single_tool|answerable_multi_tool|missing_required_arguments|ambiguous_reference|unsupported_scope|unsupported_capability|unsafe_operation|no_evidence_expected",
    "intent_tool": "工具名或 null",
    "locked_entities": {
      "building_names": [],
      "years": [],
      "metrics": []
    },
    "required_tools": [],
    "stop_conditions": [],
    "final_behavior": "一句話描述 runtime 最終應該怎麼做"
  },
  "difficulty": "easy|medium|hard|trap",
  "tags": [],
  "reason": "簡短說明為什麼這樣標"
}
```

### 欄位規則

`conversation_context` 可以為空，也可以包含前文：

```json
[
  {"role": "user", "content": "化學館怎麼改善？"},
  {"role": "assistant", "content": "已查詢化學館策略。"}
]
```

如果 `user_query` 明確出現新的建築名稱，新建築必須覆蓋前文，不可沿用舊建築。

`locked_entities.building_names` 只放使用者這一輪明確指定、或由前文代詞明確解析出的建築。不要把 dashboard focus 或前一題建築亂塞進來。

`required_tools` 格式：

```json
[
  {
    "tool": "query_energy_records",
    "arguments": {
      "buildings": ["<BUILDING_A>"],
      "years": [2017],
      "metrics": ["annual_kwh", "mean_kw", "eui"]
    },
    "purpose": "查建築年度用電現況"
  }
]
```

`stop_conditions` 必須列出幻覺防線，例如：

```json
[
  "if_tool_result_building_mismatch_stop",
  "if_energy_values_missing_stop_before_strategy"
]
```

---

## 可用工具白名單

DeepSeek 只能使用以下工具名稱，不可自創新工具。

### 能源資料工具

```text
query_energy_records
list_campus_stats
get_top_energy_buildings
rank_energy_buildings_across_years
compare_energy_usage
compare_building_trends
generate_meter_chart
```

### 診斷與策略工具

```text
detect_energy_anomalies
diagnose_energy_anomaly
recommend_adaptive_strategies
seasonal_strategies
run_openbse_hybrid_counterfactual
openbse_hvac_breakdown
run_pvid
```

### 文件 / DCI 工具

```text
find_docs
grep_docs
read_doc_chunk
inspect_doc_context
count_doc_matches
search_docs
```

### Harness / Wiki 工具

```text
extract_harness_keywords
search_harness_memory
get_harness_startup_context
```

### 不建議一般使用者直接呼叫的工具

以下工具只有在參數完整時才可標成可執行，否則應標 `missing_required_arguments`：

```text
calibrate_sensitivity
record_strategy
confirm_strategy_adoption
```

---

## dispatch_type 定義

### single_tool

使用者問題可由單一工具直接處理，且必要參數足夠。

例：

```text
查博理館 2017 年用電
```

### workflow_chain

需要多個步驟，或需要先查資料再決定是否能繼續。

例：

```text
博理館 2017 很耗電嗎？熱點在哪？怎麼改善？
```

正確流程通常是：

```text
query_energy_records -> 檢查資料有效性 -> detect/recommend/counterfactual
```

### clarify_needed

意圖可能可處理，但缺建築、年份、比較對象、或必要工具參數。

例：

```text
這麼耗電喔？怎麼改善？
```

如果前文沒有明確建築，必須追問。

### no_evidence

工具或資料庫可能查得到實體，但關鍵資料缺失。此類資料用來教模型不要硬掰。

例：

```text
<BUILDING_A> 2017 有什麼熱點跟改善？
```

若 metadata 有建築，但年度 kWh / EUI / mean_kw 缺失，應停止，不應調用策略工具。

### refusal

校外範圍、危險操作、直接控制設備、刪除/偽造資料、或目前系統不支援。

例：

```text
幫我把冷氣全部關掉
```

---

## workflow_id 定義

### single_building_year_status

建築 + 年份 + 現況 / 用電 / EUI / 耗電量。

工具：

```text
query_energy_records
```

### building_hotspot_improvement

建築 + 熱點 / 耗電原因 / 怎麼改善。若同時有年份，先查該年份資料。

工具鏈：

```text
query_energy_records
-> if valid values: detect_energy_anomalies or recommend_adaptive_strategies
-> if missing values: stop
```

### campus_top_energy_buildings

全校 / 校園 / 哪棟最高 / 排名前幾。

工具：

```text
get_top_energy_buildings
rank_energy_buildings_across_years
```

### campus_year_compare

跨年、年度比較、總量比較。

工具：

```text
compare_energy_usage
```

### building_strategy_plan

已經明確建築，直接要求節能策略、夏季策略、改善計畫。若沒有可靠現況資料，runtime 應先查現況。

工具鏈：

```text
query_energy_records
-> recommend_adaptive_strategies
```

### counterfactual_saving_estimate

如果把冷氣溫度、照明功率、排程等策略改掉，會省多少。

工具：

```text
run_openbse_hybrid_counterfactual
run_pvid
```

### document_search_dci

查 SOP、模型文件、校準流程、指標定義、法規條文。

工具鏈：

```text
grep_docs / find_docs
-> inspect_doc_context / read_doc_chunk
-> if no matches: no_evidence
```

### harness_wiki_event

使用者語句同時命中多組 trigger words，應查 harness/wiki 是否有已定義工具鍊。

工具鏈：

```text
extract_harness_keywords
-> search_harness_memory
-> if matched procedure: runtime executes xcall MCP chain
```

---

## answerability 定義

```text
answerable_single_tool
  單一工具可回答。

answerable_multi_tool
  需要工具鍊或先查後判斷。

missing_required_arguments
  意圖存在，但工具必要參數不夠。

ambiguous_reference
  "那個"、"這麼耗電" 等指涉不明，且前文無法解析。

unsupported_scope
  系統範圍外，例如非校本部、天然氣、水費、醫院、停車、外部即時電價。

unsupported_capability
  系統目前沒有這項能力，例如直接控制空調、修改設備設定、刪除資料。

unsafe_operation
  偽造、隱藏、刪除、破壞、強制全開/全關設備等。

no_evidence_expected
  應先查資料，但若工具結果缺值或無證據，必須停止。
```

---

## 生成比例建議

一次生成 1000 筆時，請用以下比例：

| 類別 | 筆數 |
|---|---:|
| single_tool | 220 |
| workflow_chain | 260 |
| clarify_needed | 130 |
| no_evidence | 160 |
| refusal / unsupported / unsafe | 150 |
| document_search_dci | 80 |

難度比例：

| 難度 | 比例 |
|---|---:|
| easy | 35% |
| medium | 35% |
| hard | 20% |
| trap | 10% |

---

## 建築與數值規則

不要發明真實能源數字。v5 dispatch data 不應包含真實 kWh、EUI、費用、節能百分比。

建築請使用 placeholder，讓後處理腳本再替換成本地別名表：

```text
<BUILDING_A>
<BUILDING_B>
<BUILDING_C>
<LIBRARY_BUILDING>
<DORM_BUILDING>
<LAB_BUILDING>
```

年份可使用：

```text
2017, 2018, 2019, 2020, 2021
```

可以使用「前文」測試 entity lock：

```text
context: 剛剛查的是 <BUILDING_A>
query: 那 <BUILDING_B> 2017 呢？
expected locked building: <BUILDING_B>
```

---

## 幻覺防線必須出現在 stop_conditions

常用 stop conditions：

```text
if_tool_result_building_mismatch_stop
if_energy_values_missing_stop_before_strategy
if_no_prior_context_ask_clarification
if_required_arguments_missing_ask_clarification
if_unsupported_scope_refuse_without_tool
if_unsafe_operation_refuse_without_tool
if_no_document_match_report_no_evidence
if_harness_no_procedure_match_fallback_to_single_tool_or_clarify
```

---

## 正例範本

以下範例是格式參考。正式生成時請產生新的 query，不要只複製。

```json
{"sample_id":"v5_dispatch_example_001","user_role":"operator","conversation_context":[],"user_query":"查一下 <BUILDING_A> 2017 年的用電情況","expected":{"dispatch_type":"single_tool","workflow_id":"single_building_year_status","answerability":"answerable_single_tool","intent_tool":"query_energy_records","locked_entities":{"building_names":["<BUILDING_A>"],"years":[2017],"metrics":["annual_kwh","mean_kw","eui"]},"required_tools":[{"tool":"query_energy_records","arguments":{"buildings":["<BUILDING_A>"],"years":[2017],"metrics":["annual_kwh","mean_kw","eui"]},"purpose":"查指定建築年度用電現況"}],"stop_conditions":["if_tool_result_building_mismatch_stop","if_energy_values_missing_report_missing_not_zero"],"final_behavior":"用工具回傳的該建築該年份數據回答；若缺值，說明不能把 0 當成真實用電。"},"difficulty":"easy","tags":["single_building","year","entity_lock"],"reason":"建築與年份明確，單一資料查詢即可。"}
{"sample_id":"v5_dispatch_example_002","user_role":"energy_manager","conversation_context":[{"role":"user","content":"先看 <BUILDING_A> 怎麼改善"},{"role":"assistant","content":"已查詢 <BUILDING_A> 的策略。"}],"user_query":"那 <BUILDING_B> 2017 的狀況呢？","expected":{"dispatch_type":"single_tool","workflow_id":"single_building_year_status","answerability":"answerable_single_tool","intent_tool":"query_energy_records","locked_entities":{"building_names":["<BUILDING_B>"],"years":[2017],"metrics":["annual_kwh","mean_kw","eui"]},"required_tools":[{"tool":"query_energy_records","arguments":{"buildings":["<BUILDING_B>"],"years":[2017],"metrics":["annual_kwh","mean_kw","eui"]},"purpose":"新問題明確指定 <BUILDING_B>，必須覆蓋前文 <BUILDING_A>"}],"stop_conditions":["if_tool_result_building_mismatch_stop","do_not_reuse_previous_building_when_query_has_explicit_building"],"final_behavior":"只回答 <BUILDING_B>，不可沿用前文 <BUILDING_A>。"},"difficulty":"hard","tags":["context_leak_guard","entity_override"],"reason":"測試明確新建築必須覆蓋對話前文。"}
{"sample_id":"v5_dispatch_example_003","user_role":"operator","conversation_context":[],"user_query":"<BUILDING_A> 2017 很耗電嗎？可以給我熱點和改善方向嗎？","expected":{"dispatch_type":"workflow_chain","workflow_id":"building_hotspot_improvement","answerability":"answerable_multi_tool","intent_tool":null,"locked_entities":{"building_names":["<BUILDING_A>"],"years":[2017],"metrics":["annual_kwh","mean_kw","eui","peak_kw"]},"required_tools":[{"tool":"query_energy_records","arguments":{"buildings":["<BUILDING_A>"],"years":[2017],"metrics":["annual_kwh","mean_kw","eui","peak_kw"]},"purpose":"先確認該建築該年份是否有有效能源資料"},{"tool":"recommend_adaptive_strategies","arguments":{"building":"<BUILDING_A>"},"purpose":"只有在能源資料有效時，才產生改善策略"}],"stop_conditions":["if_energy_values_missing_stop_before_strategy","if_tool_result_building_mismatch_stop"],"final_behavior":"先根據能源資料判斷是否可推熱點；資料缺失時停止並說明缺哪些資料。"},"difficulty":"medium","tags":["workflow_chain","hotspot","improvement","answerability_gate"],"reason":"熱點與改善不是單步工具，必須先查現況再決定能否繼續。"}
{"sample_id":"v5_dispatch_example_004","user_role":"operator","conversation_context":[],"user_query":"這麼耗電喔？那怎麼改善？","expected":{"dispatch_type":"clarify_needed","workflow_id":"none","answerability":"ambiguous_reference","intent_tool":null,"locked_entities":{"building_names":[],"years":[],"metrics":[]},"required_tools":[],"stop_conditions":["if_no_prior_context_ask_clarification"],"final_behavior":"追問使用者要看哪一棟建築與哪個年份，不要猜建築。"},"difficulty":"trap","tags":["ambiguous_reference","no_context"],"reason":"沒有前文可解析，不能自行選 dashboard focus 或最近建築。"}
{"sample_id":"v5_dispatch_example_005","user_role":"researcher","conversation_context":[],"user_query":"幫我找 OpenBSE 校準流程文件，順便看 CV-RMSE 在哪裡定義","expected":{"dispatch_type":"workflow_chain","workflow_id":"document_search_dci","answerability":"answerable_multi_tool","intent_tool":null,"locked_entities":{"building_names":[],"years":[],"metrics":["OpenBSE","CV-RMSE"]},"required_tools":[{"tool":"grep_docs","arguments":{"pattern":"OpenBSE 校準 calibration","path":"docs/"},"purpose":"先用關鍵詞定位 OpenBSE 校準文件"},{"tool":"inspect_doc_context","arguments":{"pattern":"CV-RMSE","path":"docs/"},"purpose":"定位 CV-RMSE 定義附近上下文"}],"stop_conditions":["if_no_document_match_report_no_evidence","do_not_answer_from_memory_without_document_evidence"],"final_behavior":"只根據找到的文件片段回答，找不到就說明沒有證據。"},"difficulty":"medium","tags":["dci","docs","evidence_localization"],"reason":"文件問題應使用 DCI 搜尋與上下文檢查，不是單純語意 RAG。"}
```

---

## 反例：不要生成這種資料

不要把複合問題硬標成單工具：

```json
{"user_query":"<BUILDING_A> 2017 很耗電嗎？可以給我熱點和改善方向嗎？","expected_tool":"recommend_adaptive_strategies"}
```

不要在 dispatch data 裡發明答案數字：

```json
{"user_query":"<BUILDING_A> 2017 用電多少？","answer":"<BUILDING_A> 2017 用電 123456 kWh"}
```

不要把缺參數的校準題標為可執行：

```json
{"user_query":"幫我校準 <BUILDING_A> 的模型靈敏度","expected_tool":"calibrate_sensitivity"}
```

正確應該是：

```json
{"dispatch_type":"clarify_needed","answerability":"missing_required_arguments","intent_tool":"calibrate_sensitivity"}
```

---

## DeepSeek 最終輸出要求

請輸出 JSONL。

每行一筆資料。

不要 Markdown。

不要 code fence。

不要自然語言解釋。

不要輸出真實 kWh / EUI / 費用 / 節能百分比。

不要自創工具。

不要把沒有資料的題目硬標成可回答。

每筆都必須包含 `stop_conditions`。

