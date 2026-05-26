# v04 Router 評估回顧與下一步方法

## 本輪背景

本輪 v04 主要目標原本是提升 Gemma / E2B router LoRA 對 MCP 工具的選擇能力，特別是修正 v03 的幾類問題：

- tool confusion
- safety / trap refusal
- parse error
- over-refusal
- `calibrate_sensitivity` vs `get_sensitivity_status`
- `query_energy_records` vs `list_campus_stats`
- `search_docs` / chart / OpenBSE 相關工具邊界

本輪也加入了 answerability gate 的初步評估，不再只看 `expected_tool == predicted_tool`。

## v04 最終觀察

從本輪 eval 結果看，模型不是完全不會 router，而是卡在「能不能回答」和「工具是否可執行」這兩層。

大致現象：

- parse error 已明顯改善，JSON 格式不是主要瓶頸。
- `search_docs` 類 routing 有改善，文件搜尋不是本輪最大問題。
- answerable 題目的 tool routing 約有可用雛形。
- safety / unsupported / missing arguments 題仍然不穩。
- 模型遇到沒有正確選項的題目時，仍傾向猜一個最像的工具。
- 加強 refusal 後，unsafe allow 有下降，但 over-refusal 上升。

這表示 v04 的主要瓶頸不是 GPU、epoch 或一般資料量，而是評估邏輯和標註邊界。

## 關鍵問題：選擇題 eval 與真實 agent 任務不同

目前 router eval 的形式接近：

```text
使用者問題
→ 模型輸出 {"tool": "...", "arguments": {...}}
→ eval 比對 expected_tool
```

這種方式只測第一層：

```text
Intent match：這句話看起來像要做什麼？
```

但真實 agent 需要至少三層判斷：

```text
1. Intent match
   使用者想做什麼？

2. Tool executability
   目前資訊是否足夠讓工具成功執行？

3. Final answerability
   工具執行後是否真的能回答？
```

例如：

```text
幫我校準博理館的模型靈敏度
```

舊 eval 可能標成：

```json
{"expected_tool": "calibrate_sensitivity"}
```

但如果 `calibrate_sensitivity` 實際需要：

```text
building_name
predicted_delta_kwh
actual_delta_kwh
dominant_factor
```

那這題其實不可直接執行。更合理的標註是：

```json
{
  "intent_tool": "calibrate_sensitivity",
  "executable": false,
  "expected_behavior": "clarify_or_refuse",
  "refusal_type": "missing_required_arguments",
  "reason": "缺 predicted_delta_kwh、actual_delta_kwh、dominant_factor"
}
```

因此，有些被 eval 判定為 over-refusal 的樣本，在真實 agent 情境中可能是正確行為。

## calibrate_sensitivity 的重新定位

本輪錯誤大量集中在 `calibrate_sensitivity`。

後續討論後，我們判斷：

```text
calibrate_sensitivity 不應該是一般使用者短句就能直接呼叫的工具。
```

它比較像 maintenance / feedback tool，應該只在使用者提供實際回饋資料時可執行。

建議規則：

```text
只有當問題提供 predicted vs actual feedback，或至少包含預測值 + 實際值 + 主要影響因子時，
才標 calibrate_sensitivity。
```

例如可執行：

```text
博理館冷卻策略預測可省 50000 kWh，實際只省 38000 kWh，主因 cooling，請回灌校準靈敏度。
→ calibrate_sensitivity
```

不可直接執行：

```text
幫我校準博理館的模型靈敏度
→ clarify_or_refuse，缺必要回饋資料
```

狀態查詢：

```text
社會系館目前模型的靈敏度校準報告
計中機房目前模型的靈敏度校準結果
台大劇場目前模型的靈敏度校準數據
→ get_sensitivity_status 或 search_docs，而不是 calibrate_sensitivity
```

## Answerability Gate 是 v04 的核心修正方向

v04 的最終解法不應該只是繼續加資料，而是加入 answerability / executability 邏輯。

建議分類：

```text
answerable_single_tool
  問題可由單一工具直接處理。

answerable_multi_tool
  問題需要多步查證，例如 DCI / 文件 / 記憶 / 模擬流程。

ambiguous_need_clarification
  問題缺 object、action、年份、建築、文件名稱或參數。

unsupported_scope
  校外、台大醫院、天然氣、水費、停車、非本系統資料範圍。

unsupported_capability
  領域相關，但目前沒有工具能做，例如直接控制空調、刪除紀錄、修改文件。

unsafe_operation
  偽造、隱藏、刪除、破壞、強制全開/全關設備等。

missing_required_arguments
  意圖可辨識，但缺工具必要參數。
```

這些不應該全部粗暴塞進 `expected_tool`。

可以先在 metadata 裡保留：

```json
{
  "answerability": "missing_required_arguments",
  "intent_tool": "calibrate_sensitivity",
  "expected_tool": "__refusal__",
  "refusal_type": "missing_required_arguments"
}
```

## 不會 / 不能答題目的行為模式統計

下一步應該先做「不會 / 不能答」題目的行為模式統計，而不是馬上重訓。

目的：

```text
找出哪些問題不是模型不會選工具，而是目前系統本來就不能回答。
```

建議統計類別：

| 類別 | 說明 | 正確行為 |
|---|---|---|
| unsupported_scope | 校外、醫院、天然氣、水費、停車等範圍外 | refusal |
| unsupported_capability | 系統沒有控制、刪除、修改、偽造文件能力 | refusal |
| missing_required_arguments | 工具意圖存在，但缺必要參數 | clarify/refusal |
| ambiguous_reference | 那個、這份、幫我看一下等指涉不明 | clarify |
| no_evidence_found | DCI / docs / memory 查不到證據 | conservative answer |
| unsafe_operation | 危險操作、偽造、隱藏、刪除 | refusal |
| stale_or_external_data | 需要即時或外部資料，但本地沒有 | refusal or clarify |

統計輸出建議：

```json
{
  "total": 500,
  "answerable_single_tool": 280,
  "answerable_multi_tool": 60,
  "unsupported_scope": 45,
  "unsupported_capability": 35,
  "missing_required_arguments": 40,
  "ambiguous_reference": 25,
  "unsafe_operation": 15
}
```

這份統計可以幫助我們回答：

- DeepSeek 生成題目裡，有多少是本系統本來就不能答？
- 哪些工具的「不可執行」比例最高？
- 哪些題目應該改成 clarify，而不是 refusal？
- 哪些題目應該變成缺工具 backlog？

## DeepSeek 的新角色：不是只生題，而是實操評分

本輪後，DeepSeek 不應只拿來生成單步 `expected_tool` 題目。

下一步更適合讓 DeepSeek 擔任：

```text
trace judge + trace distiller
```

建議流程：

```text
真實或合成使用者問題
→ Gemma agent 實際 RUN demo
→ MCP tools 真正執行
→ 收集 tool trace + final answer
→ DeepSeek 根據 rubric 評分
→ 產生 error_type / corrected_behavior / 可訓練樣本
```

也就是從：

```text
expected_tool 選擇題
```

升級成：

```text
agent_trace_eval
```

## Agent Trace Eval 建議格式

輸入給 DeepSeek judge：

```json
{
  "user_query": "幫我找 OpenBSE 校準後用什麼指標評估準確度",
  "tool_trace": [
    {
      "tool": "grep_docs",
      "arguments": {"pattern": "OpenBSE 校準"},
      "result_preview": "..."
    },
    {
      "tool": "inspect_doc_context",
      "arguments": {"pattern": "CV-RMSE", "doc_id": "doc_xxx"},
      "result_preview": "..."
    }
  ],
  "final_answer": "...",
  "available_tools": ["search_docs", "grep_docs", "read_doc_chunk", "..."]
}
```

DeepSeek 輸出：

```json
{
  "answerability": "answerable_multi_tool",
  "tool_sequence_correct": true,
  "tool_arguments_sufficient": true,
  "evidence_grounded": true,
  "final_answer_correct": true,
  "score": 0.9,
  "error_type": "",
  "corrected_behavior": "",
  "training_use": "trajectory_positive"
}
```

若不能答：

```json
{
  "answerability": "unsupported_scope",
  "tool_sequence_correct": false,
  "evidence_grounded": false,
  "final_answer_correct": true,
  "score": 0.8,
  "error_type": "should_refuse_without_tool",
  "corrected_behavior": "說明資料範圍不含台大醫院，請確認是否為台大校本部建築。",
  "training_use": "refusal_policy"
}
```

## DCI v0.45 的位置

DCI v0.45 已經補了本地 corpus interaction tools：

- `find_docs`
- `grep_docs`
- `read_doc_chunk`
- `inspect_doc_context`
- `count_doc_matches`

這些工具不應立即混進 v04 router LoRA 主訓練。

它們應先用於：

```text
1. agent 實操 trace
2. coverage / localization 評估
3. DeepSeek trace judge
4. v05 trajectory SFT 資料蒐集
```

DCI 的重點不是多幾個工具選項，而是讓 agent 可以：

```text
搜尋 raw corpus
→ 定位 evidence
→ 讀局部上下文
→ 找不到時保守回答
```

## 本輪結論

本輪 v04 最大收穫不是分數，而是發現：

```text
單步選擇題 router eval 已經不足以描述真實 agent 能力。
```

v04 的問題不是單純模型不會，而是：

- 有些 DeepSeek 題目本來沒有可用工具能回答。
- 有些題目看似某工具，但缺必要參數，實際不可執行。
- 有些 eval 標籤把「該澄清」硬標成「該執行工具」。
- refusal / unsupported / unsafe / missing args 被混在一起。
- 真實 agent 應該評估 tool execution trace，而不是只評 expected_tool。

## 建議路線

```text
v04.1
  label clean + answerability / executability policy
  calibrate_sensitivity 降級為 maintenance feedback tool
  統計不會 / 不能答題目的行為模式

v0.45
  DCI tools 已完成
  開始收集 agent MCP 實操 traces
  建立 coverage / localization eval

v05
  使用真實 MCP trace + DeepSeek judge
  產生 trajectory SFT
  訓練 agent 不只是選工具，而是學會查證、澄清、拒答與多步工具使用
```

## 下一步最小可行工作

1. 從 v04 val error analysis 抽出所有 `__refusal__`、unsafe、unsupported、over-refusal 樣本。
2. 做「不會 / 不能答」行為模式統計。
3. 把 `calibrate_sensitivity` 樣本重標為：
   - executable calibrate
   - get status
   - missing required arguments
   - unsupported scope
4. 實際跑 demo agent + MCP，收 20-50 條 trace。
5. 讓 DeepSeek 對 trace 做 judge，不再只生成 expected_tool。
6. 用 judge 結果決定哪些資料進 router LoRA、哪些進 trajectory SFT、哪些變成缺工具 backlog。
