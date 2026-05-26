# V6 Dispatch Recovery Plan

## 目的

這份文件用來交接目前 v05 dispatch 訓練的真實狀況，並定義下一輪 v6 的修正方向。

重點結論先講：

> v05 並不是 dispatch 方法本身失敗，而是 system prompt、資料標註、輸出契約三者沒有對齊，導致模型大量用自然語言回答，沒有輸出合法 dispatch JSON。

因此 v6 不應該先做「更多資料」或「更久訓練」，而應該先做：

1. strict dispatch prompt 對齊  
2. label 邊界清洗  
3. format-first smoke eval  
4. 小輪驗證後再完整重訓

---

## v05 本輪總結

### 真正有效的本輪評估

本輪使用的是新的 dispatch-aware pipeline，不是舊 router pipeline。

關鍵檔案：

- `G:\我的雲端硬碟\energy_lora_router_v05\notebooks\router_strict_lora_colab_v05.ipynb`
- `G:\我的雲端硬碟\energy_lora_router_v05\scripts\train_lora.py`
- `G:\我的雲端硬碟\energy_lora_router_v05\scripts\evaluate_router.py`
- `G:\我的雲端硬碟\energy_lora_router_v05\outputs\gemma_dispatch_v05\eval\v05_final_val_eval_summary.json`

### v05 最終分數

From `v05_final_val_eval_summary.json`:

- overall accuracy: `9.5%` (`14/147`)
- answerability accuracy: `12.2%` (`18/147`)
- tool accuracy on answerable: `0.0%`
- refusal correctness: `30.0%`
- malformed / parse error rate: `87.1%` (`128/147`)

### 重要觀察

這輪不是主要死在 semantic routing，而是死在 output format collapse。

confusion matrix 顯示：

- `workflow_chain -> __parse_error__ : 56`
- `single_tool -> __parse_error__ : 31`
- `no_evidence -> __parse_error__ : 21`
- `clarify_needed -> __parse_error__ : 14`

也就是說，大多數題目不是選錯，而是根本沒有輸出合法 JSON。

---

## 根因分析

## 1. System prompt 與 target schema 打架

目前 `G:\我的雲端硬碟\energy_lora_router_v05\scripts\00_config_v05.py` 的 `render_system_prompt()` 仍然把模型定義成：

- NTU 校園能源助理
- 提供 PI-VD 節能建議
- 用自然語言輸出：
  - 建議行動
  - 現況
  - 建議
  - 預估節能
  - 舒適度/影響

但 v05 的訓練 target 實際上是：

```json
{
  "dispatch_type": "...",
  "workflow_id": "...",
  "locked_entities": {...},
  "required_tools": [...],
  "stop_conditions": [...],
  "answerability": "..."
}
```

這會造成模型被同時教成兩種角色：

1. 助理型自然語言回答者
2. 嚴格 JSON dispatch classifier

最後它多半回到自然語言。

### 直接證據

`v05_final_val_error_analysis.jsonl` 中的錯例大量出現：

- 一整段中文澄清
- 一整段能力說明
- 單純輸出 `query_energy_records`
- 文件題回答「抱歉我沒有直接存取文件庫」

這些都不是邏輯錯，而是格式契約錯。

---

## 2. DCI 文件工具沒有被完整教進 system prompt

v05 資料中已經包含：

- `find_docs`
- `grep_docs`
- `read_doc_chunk`
- `inspect_doc_context`
- `count_doc_matches`

以及 workflow:

- `document_search_dci`

但目前 core system prompt 只列出 7 個能源工具，沒有把 DCI tools 正式列入。

後果：

- 訓練資料要求模型輸出 DCI tools
- prompt 卻暗示這些工具不存在

模型會退回：

- 自然語言說明
- 自然語言拒答
- 模糊 fallback

---

## 3. 部分資料標註邊界不乾淨

從 train split 統計可看到幾種搭配：

- `workflow_chain + answerable_multi_tool`: 279
- `single_tool + answerable_single_tool`: 188
- `no_evidence + no_evidence_expected`: 103
- `refusal + unsafe_operation`: 75
- `refusal + unsupported_scope`: 54
- `clarify_needed + ambiguous_reference`: 68
- `clarify_needed + missing_required_arguments`: 43
- `no_evidence + unsupported_scope`: 27

最後這一類是有風險的：

> `unsupported_scope` 通常應屬於 refusal，而不是 no_evidence。

若工具本來就不支援該領域，不能標成「查不到證據」。

這種衝突會讓模型對 answerability 邊界學歪。

---

## 4. 本輪不應優先怪罪 GPU、epoch 或 bf16

本輪最主要的失敗不是算力不足，而是：

- prompt contract mismatch
- tool visibility mismatch
- label boundary mismatch

在 parse error 還有 `87%` 的情況下：

- 改顯卡
- 加 epoch
- 加資料量
- 微調 learning rate

都不是第一優先。

---

## v6 核心策略

v6 的定位不是「v05 加大版」，而是：

> **dispatch contract recovery round**

任務是先讓模型穩定輸出合法 JSON，並在此基礎上再看 dispatch 語義對不對。

### v6 的四個優先目標

1. parse error rate 降到 `<10%`  
2. refusal / clarify / no_evidence 三者邊界明確  
3. DCI workflow 能合法輸出工具鏈  
4. 再開始追求 workflow accuracy

---

## v6 必做修改

## A. 改 system prompt 成 strict dispatch-only

### 需要修改的檔案

- `G:\我的雲端硬碟\energy_lora_router_v05\scripts\00_config_v05.py`

### 新 prompt 原則

模型不再是「回答型助理」，而是「dispatch 決策器」。

system prompt 必須清楚寫：

1. 你只負責輸出 dispatch 決策
2. 只能輸出單一 JSON object
3. 不可輸出自然語言、解釋、markdown、前後綴
4. 必須使用固定 schema
5. DCI tools 與 7 core tools 都是合法工具

### 建議 prompt 骨架

```text
你是一個 dispatch classifier，不是一般對話助理。

你的唯一任務是根據使用者輸入，輸出一個 JSON object，決定：
- dispatch_type
- workflow_id
- locked_entities
- required_tools
- stop_conditions
- answerability

你只能輸出 JSON。
不得輸出解釋、中文句子、markdown、程式碼區塊、前言、結語。

若問題超出支援範圍，仍然輸出合法 JSON，dispatch_type 應為 refusal 或 no_evidence。
若資訊不足，仍然輸出合法 JSON，dispatch_type 應為 clarify_needed。
```

### 必須列入的工具白名單

Core tools:

- `query_energy_records`
- `list_campus_stats`
- `get_top_energy_buildings`
- `detect_energy_anomalies`
- `run_openbse_hybrid_counterfactual`
- `openbse_hvac_breakdown`
- `recommend_adaptive_strategies`

DCI tools:

- `search_docs`
- `find_docs`
- `grep_docs`
- `read_doc_chunk`
- `inspect_doc_context`
- `count_doc_matches`

---

## B. 重套新的 system prompt 到全部資料

### 需要修改或執行

- 重建 `train_v05_dispatch.jsonl`
- 重建 `val_v05_dispatch.jsonl`
- 重建 `smoke_v05_dispatch.jsonl`

注意：

這不是重生資料內容，而是把每一筆 `messages[0].content` 的 system prompt 換成新的 strict dispatch prompt。

如果不重套 prompt，模型仍然會被舊助理 prompt 汙染。

---

## C. 清洗 answerability / dispatch 邊界

### 目標規則

#### `refusal`

使用於：

- `unsafe_operation`
- `unsupported_scope`
- `unsupported_capability`

#### `no_evidence`

使用於：

- 工具理論上可答
- 但目前資料查不到
- 或文件搜尋無 match

#### `clarify_needed`

使用於：

- 缺建築
- 缺年份
- 指代模糊
- 缺必要參數

### 必做清洗

重新檢查所有：

- `expected_answerability == unsupported_scope`
- `expected_dispatch_type == no_evidence`

把不合理的樣本改成 `refusal`。

---

## D. 新增 format-first 小型資料組

本輪最大的問題是格式，所以 v6 應加入一批專門教格式的樣本。

### 建議新增 80 到 150 筆 format curriculum

類型：

1. 極短問句  
   - `然後呢`
   - `幫我看一下`
   - `節能方法`

2. 文件題  
   - `CV-RMSE 的定義在哪份文件`
   - `PI-VD 模型說明文件`

3. 明確單工具  
   - `<BUILDING_X> 2019 用電`
   - `<BUILDING_Y> 有異常嗎`

4. workflow 題  
   - `<BUILDING_Z> 很耗電，熱點跟改善是什麼`

5. refusal 題  
   - `幫我隱藏異常事件`
   - `查天然氣`

### 這批資料的目標

不是增加語義覆蓋，而是強化：

- 永遠吐 JSON
- 永遠帶齊欄位
- 永遠不用自然語言

---

## E. 加一個 format smoke test，先卡住 parse error

在 full training 前，先做一個超快的小驗證。

### 建議新增 eval

`format_smoke_v6.jsonl`

數量：

- 16 到 32 筆即可

只測：

- 能不能輸出合法 JSON
- key 有沒有齊
- enum 是否合法

### Gate

若 `format_smoke parse error > 10%`：

> 不進 full validation，不做大輪訓練。

先修 prompt / sample，再重跑。

---

## v6 非必要修改

以下不是第一優先，可先不要花時間：

- 換更大的 GPU
- 改成 full fine-tune
- 大量增加資料到數千筆以上
- 先追求多輪 chain-of-thought
- 先把 runtime 複雜化

這些都應該排在 prompt contract 修好之後。

---

## 建議施工順序

## Phase 1: Prompt 對齊

1. 改 `00_config_v05.py` 的 `render_system_prompt()`
2. 把角色改成 strict dispatch classifier
3. 加入 DCI tools 白名單
4. 明寫只能輸出 JSON

## Phase 2: 資料清洗

1. 重套新 prompt 到 train / val / smoke
2. 清理 `no_evidence + unsupported_scope`
3. 檢查 refusal / no_evidence / clarify 三類邊界

## Phase 3: 格式課程

1. 新增 format curriculum 小資料
2. 加入 train split
3. 保持 val split 邏輯乾淨

## Phase 4: 小輪訓練與快驗證

1. 先跑 0.3 到 0.5 epoch 或小樣本 smoke train
2. 先看 `format_smoke_v6`
3. parse error 若仍高，先停下修 prompt

## Phase 5: 正式 v6 train

1. 再跑完整 1 epoch
2. 看 smoke
3. 再看 full val

---

## v6 驗收門檻

至少先達到以下條件，才值得再談 demo 接入：

### 基礎格式門檻

- parse error rate `< 10%`
- format smoke accuracy `> 90%`

### 類型邊界門檻

- refusal correctness `> 85%`
- clarify accuracy `> 80%`
- no_evidence accuracy `> 80%`

### 派工門檻

- single_tool accuracy `> 75%`
- workflow_chain accuracy `> 70%`
- document_search_dci workflow 有可用雛形

注意：

在 v6，先把格式與邊界救回來，比追求 90% overall 更重要。

---

## 實作提醒

## 1. 不要再把模型當終端回答器

v6 的 LoRA 目標是：

> decide dispatch, not answer the user

最終自然語言回答應該交給：

- runtime template
- tool result validator
- 或後置 LLM polish

而不是讓這顆 dispatch LoRA 一次做完全部事情。

## 2. `document_search_dci` 不要被當成 unsupported

既然本地已經完成：

- `find_docs`
- `grep_docs`
- `read_doc_chunk`
- `inspect_doc_context`
- `count_doc_matches`

那資料、prompt、eval 就應一致承認它們存在。

## 3. 看到 parse error 爆高時，不要先怪 learning rate

本輪已經證明：

> prompt contract 錯，比 optimizer 錯更致命。

---

## 建議交給其他 AI 的明確任務

### 任務 1

重寫 `G:\我的雲端硬碟\energy_lora_router_v05\scripts\00_config_v05.py` 的 `render_system_prompt()`  
目標：strict dispatch-only prompt，列出完整 schema 與工具白名單。

### 任務 2

重建：

- `train_v05_dispatch.jsonl`
- `val_v05_dispatch.jsonl`
- `smoke_v05_dispatch.jsonl`

目標：全部換成新的 system prompt。

### 任務 3

掃描並修正：

- `expected_dispatch_type == no_evidence`
- `expected_answerability == unsupported_scope`

的交叉樣本，避免 unsupported 被標成 no_evidence。

### 任務 4

新增一個 `format_smoke_v6.jsonl`，只驗證 JSON schema 輸出。

### 任務 5

新增 80 到 150 筆 format curriculum 樣本，專門修正：

- 自然語言回答
- 單字工具輸出
- 文件題亂拒答
- clarify 題輸出長段文字

---

## 最後結論

v05 的主要問題不是「dispatch 架構不行」，而是：

1. prompt 仍在把模型往自然語言助理方向拉  
2. DCI tools 沒被完整宣告  
3. 標註邊界有部分混淆  

所以 v6 的正確做法不是立刻擴資料或換硬體，而是：

> 先把輸出契約修正成單一明確的 dispatch JSON 任務，再重訓。

只要 parse error 能先從 `87%` 大幅降下來，v6 就有機會回到可迭代的軌道。
