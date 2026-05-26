# Pre-LoRA 改善清單 + Post-LoRA 路線圖

> 作用範圍：`D:\idf優化\demo` 的 AI Agent（`lm_studio_client.py` / `demo_assistant.py` / `energy_intent_router.py`）
> 訓練資料：`demo/data/lora/`、`demo/data/energy_sft_factory/`
> 知識庫：`demo/data/knowledge_workbench/`
> 撰寫日期：2026-05-08（v2 — 對齊 AcademicForge AI-Research-SKILLs）

本文件分三段：
1. **Part A — LoRA 微調前可立即動工的改善**（不需要重訓任何模型）
2. **Part B — LoRA 微調完成後才解鎖、值得做的工作**
3. **Part C — 訓練流程與框架選型**（依據 `Downloads/skill/AGENT/AcademicForge-master/skills/AI-research-SKILLs/` 的 Unsloth、TRL、knowledge-distillation、lm-evaluation-harness 等 SKILL.md）

---

## Part A. LoRA 之前先做這些事

### A0. （新增）轉換成 Unsloth / TRL 看得懂的對話格式

對齊 `03-fine-tuning/unsloth/` 與 `06-post-training/trl-fine-tuning/` SKILL.md：

- 我們的資料已經是 OpenAI `messages` 陣列格式（system/user/assistant），可直接餵 `SFTTrainer`，但 **Gemma 系列必須套對 chat_template**（Gemma 3 = `gemma-3`，不是 ChatML 也不是 ShareGPT）。Unsloth 提供 `get_chat_template(tokenizer, "gemma-3")`，要在 dataset map 階段套用。
- `harness_full_phase1.jsonl` 的 assistant 內容是「結論→依據→假設→建議」純文字，**沒有 tool_call 結構**。如果想讓 LoRA 學「自己決定何時呼叫工具」，必須把 prefetch 命中的那些樣本改成標準 OpenAI tool-calling 格式（`assistant.tool_calls` + `tool` role 訊息）。否則 LoRA 只會學「文字格式」、不會學「routing」。
- 訓練時開 `train_on_responses_only=True`（unsloth 內建），mask 掉 user/system tokens、只計算 assistant 段的 loss。這對我們的長 system prompt 很重要（不然 loss 被 60 行系統提示稀釋）。

### A1. 修掉訓練資料的標籤錯誤（最高優先）

`demo/data/energy_sft_factory/training/train.jsonl` 的 L1 異常分類樣本有大量「標籤跟讀值對不上」的情形：

| sample_id | 讀值 | 標的標籤 | 實際應該是 |
|---|---|---|---|
| rtem_99_FAN_191116 | `[23.93, 23.93, 23.92, ..., 23.92]`（std<0.01） | 震盪 | zero_flatline |
| rtem_121_PUMP_359896 | `[24.00, 24.00, ..., 24.00]` | 突波 | zero_flatline |
| rtem_120_CT_196807 | `[30.00] × 8` | 震盪 | zero_flatline |

而 `output` 內的「依據」段落還會寫 `「明顯超出正常範圍」`，自相矛盾。
這是 `template_teacher_v1` 在生資料時的 bug。**LoRA 沒做之前必須修掉**，否則模型會學到「平直 = 震盪/突波」。

**動作**：
- 在 `data factory` 加一條 post-filter：`std < 0.01 → 強制覆寫 pattern = zero_flatline`，並重生 `output` 文字。
- 重跑 `judge`：把現在 `judge_score = 1.0` 但讀值/標籤矛盾的樣本標 `rejected`。
- 移除 `synthetic_candidates_rejected.jsonl` 重複的句型，避免後續 DPO 誤判。

### A2. 補上真正的 held-out 評估集（並接 lm-evaluation-harness）

現況：
- `data/lora/test.jsonl` → 0 行
- `data/lora/test_batch.jsonl` → 1 行
- `data/lora/test_zh.jsonl` → 1 行
- `harness_routing_qa_90.jsonl` 與 `harness_routing_40.jsonl` 前幾筆完全相同（90 是 40 的超集）

沒有 held-out test 等於「無法量化 LoRA 是否有效」。

**動作**：
- 從 `harness_full_phase1.jsonl`（429 筆）切 50–100 筆做 **保留集**，**不要進** `train.jsonl`。
- 加上 3 種噪音樣態：模糊建物名（「那棟新蓋的圖書館」）、多 hop（「比較 A、B 哪個 EUI 高且 R² 較差」）、OOD 拒答（「幫我算員工薪水」）。
- 把 routing_40 從訓練清單刪掉，只保留 routing_qa_90。
- **接到 `lm-evaluation-harness`**（`11-evaluation/lm-evaluation-harness/SKILL.md`）：把 50–100 題寫成 custom task YAML，定義 `metric: tool_correct ∧ numbers_correct`（兩者已存在 metadata），LoRA 前後各跑一次比較。Unsloth 的 `references/llms-txt.md` 第 1524 段 `(3) Adding an evaluation loop / OOMs` 給的是訓練中 eval；訓練後 benchmark 用 lm-evaluation-harness 才能跨 epoch 比較。

### A3. 清理 wiki memory 雜訊

`data/knowledge_workbench/memory/wiki/sessions/` 有 20 條記錄，其中 17 條都是
「**Prompt:** You are an NTU campus energy assistant with MCP tool access.」
—— 那是 system prompt 不是 user query，被當成 prompt 存起來了。

**動作**：在 `lm_studio_client.py:740 _record_session_note()` 加 guard：
```python
if prompt.lower().startswith(("you are", "system", "instruction:")):
    return
```
並把現有的 17 條 session 從 `sessions/` 與 `index.md`、`log.md` 一起刪掉。

### A4. 路由與 prefetch 的去重複化

`energy_intent_router.py` 跟 `lm_studio_client.py` 兩邊各維護一份意圖關鍵字（`INTENT_RULES` vs `_should_prefetch_*`），規則重複且會漂移。

**動作**：
- 讓 `_should_prefetch_*` 都改呼叫 `match_intent()` 並判斷 `rule["tool"]`，唯一規則來源放 `INTENT_RULES`。
- `比較 + 趨勢` 同時出現會誤抓 `compare_building_trends`；在 `match_intent` 加「全校 / 校園 / NTU / 整體」優先 boost → `compare_energy_usage`，把 system prompt 裡 hardcode 的提示移掉。

### A5. 建物名擴充

`lm_studio_client.py:_extract_strategy_building` 只 hardcode 約 25 棟建物，無法覆蓋完整校區建築清單；全校彙總數字應由授權資料工具回傳，不能寫死在訓練資料中。

**動作**：改用 `campuses/ntu/data/buildings.geojson` + `building_alias.py` 動態載入，並在 demo 啟動時建一份 trie / regex，比 list comprehension 快也不會漏。

### A6. 大 CSV 的 chunk 策略

`knowledge_workbench/state/chunks.json` 裡 `general` 群組存了 `NTU_powerMeter_kW_daily_2014-2020.csv`（約 800 個欄位）。目前的 chunking 把整列欄位 dump 進 chunk，對 retrieval 完全沒幫助（chunk #1 全是欄位名）。

**動作**：CSV 走 metadata-only ingest（columns + per-column 統計），把 row-level 查詢交給 `query_meter_or_kpi` MCP 工具，不要進 vector index。

### A7. 知識圖譜其實是空的

`memory/graph/GRAPH_REPORT.md` 顯示 `Pages indexed: 2, Edges: 2`。CLAUDE.md 規定要先讀這份 graph，但現在裡面沒東西。

**動作**：用 graphify 對整個 `memory/wiki/` 跟 `groups/*/MEMORY.md` 重跑，加上 `tools_used`、`building_id`、`source_type` 三種邊，god nodes 才有意義。

### A8. Curated trace 的 quality gate

`state/curated_traces.jsonl` 第 1 條 `confidence: 0.15` 且 `answer = "No matching evidence was found"`，竟然 `approved: true` 並進入 `MEMORY.md`。這種 trace 進 LoRA 訓練資料只會教壞模型。

**動作**：
- workbench `approve` 流程加 `confidence >= 0.6` 與 `len(cited_chunks) > 0` 的閘門。
- 寫個一次性 cleanup script 把現有 5 條 trace 過濾、重新生 `memory_index.json`。

### A9. Agent loop 的可觀察性

`lm_studio_client.py` 的 `chat_with_mcp` 預設走 prefetch-only 路徑（`prefetch_only_answer=True`），實際上 6 個 turn 的迭代從來沒被用到。`tool_trace` 也沒回灌進 wiki memory。

**動作**：
- prefetch 命中時也印出 `prefetch_tool` 與 `arguments` 到 panel UI，讓使用者可以判斷工具決策對不對。
- `_record_session_note` 把 `tool_trace` 摘要寫進 wiki note，這樣下次 `recall_wiki_memory` 可以查到「上次 X 用了什麼工具」。

### A10. Prompt 結構優化（不需要重訓也能省 token）

`chat_with_mcp` 的 system prompt 約 60 行，每個 user message 都送一次。Gemma 4 E2B 沒 prompt cache 的話，這成本很大。

**動作**：
- 拆成 4 段（`role` / `tool_routing` / `wiki_memory_rules` / `format_rules`），依 prompt 命中的意圖動態選擇。
- 例如純 RAG 問題不需要送「ranking / counterfactual / openbse」相關的工具規則。

---

## Part B. LoRA 微調完成後才解鎖的事

> 前提：以 `final_sft_dataset.jsonl`（116 筆）+ 修正過的 `harness_full_phase1.jsonl` 訓出 LoRA adapter，並掛到 LM Studio / llama.cpp。

### B1. 大幅瘦身 `lm_studio_client.py`

目前那 1300 行裡，超過一半都是 prefetch heuristic、loose tool-call regex、fallback-answer 模板。
LoRA 訓練後模型會內化「結論→依據→假設→建議」格式與工具路由，這些 hack 都可以拿掉：

| 可刪除 | 行數 | 為什麼 |
|---|---|---|
| `_should_prefetch_*` 系列 | ~120 行 | 模型自己會選工具 |
| `_extract_text_tool_calls` 系列 | ~50 行 | 模型直接回標準 OpenAI tool_calls |
| `_fallback_answer_from_tool_result` | ~150 行 | 模型自己會把工具結果格式化 |
| 多重 prefetch 路徑 | ~400 行 | 改成統一 agent loop |

預估縮到 < 500 行。

### B2. 啟用真正的 multi-hop agent loop

現在 `max_iterations=6` 但 prefetch-only 路徑直接 1 turn 結束。LoRA 後可以放心讓模型跑 multi-tool plan：

- 「比 A、B 哪個節能改造 ROI 高」→ `get_building_detail(A)` → `get_building_detail(B)` → `run_counterfactual(A)` → `run_counterfactual(B)` → 綜合回答
- 「上週建議的冷房 +2°C 採用了沒」→ `check_strategy_status` → `compare_actual_predicted` → `calibrate_sensitivity`

### B3. 信任本地、減少 cloud 依賴

`demo_assistant.py:analyze` 目前的 fallback 順序是 local MCP → local LLM → cloud → heuristic。
現況 cloud（Gemini）效果明顯比 local 好，所以 cloud token 還是燒得多。
LoRA 後 local 至少在「結論→依據→假設→建議」結構與工具路由上會接近 cloud，可以：

- 把 `CloudModelAdapter` 改成 **僅 quality-check 抽樣 5–10%** 的請求，用作 LLM-as-judge 訓練 next-round。
- `ENERGY_LOCAL_LLM_PROVIDER=gemma` 設成預設且去掉 fallback warning。

### B4. 啟用 thinking / CoT 模式

現在 `payload["chat_template_kwargs"] = {"enable_thinking": False}`，因為 base Gemma 開 thinking 會吐很多噪音。
LoRA 訓練資料如果加上「assistant_thinking → assistant_final」的 two-step 範例，就可以打開 thinking，多 hop 推理品質提升而且可以追蹤推理路徑（用於 audit）。

### B5. 從 SFT 升級到 DPO（→ 視情況再上 GRPO）

`processed/energy_preference_pairs.jsonl` 已經有 111 條 preference pair，`reports/downstream_validation.json` 顯示 `chosen_win_rate=1.0`—— 但這是因為 chosen 跟 rejected 都用同一支 `template_teacher_v1` 生，沒有真正鑑別力。

依 `06-post-training/trl-fine-tuning/SKILL.md` 與 `grpo-rl-training/SKILL.md`：

| 階段 | 框架 | 何時用 | 注意 |
|---|---|---|---|
| DPO | `TRL DPOTrainer`（β=0.1） | preference pairs 有 chosen/rejected 但沒 reward | Windows 可直接跑，Unsloth 也支援 |
| GRPO | `TRL GRPOTrainer` + 可程式化 reward function | 想用「答案正確性 / 格式合規 / 工具呼叫成功率」當 reward 訊號 | 需要 vLLM，Windows **必須走 WSL**；每 prompt ≥2 generations |
| SimPO | `simpo` | 不想維護 reference model，且資源吃緊 | 比 DPO 簡化、無需 reference policy |

LoRA 後具體做法：
- 用 LoRA 模型本身重生 `rejected`（**自身的弱點**才有 DPO 價值），cloud 模型生 `chosen`。
- 先跑 DPO，預期能在 anomaly 邊界 case（std 微小、僅幾筆 outlier）大幅修正。
- 後期若想優化「工具叫對了沒」，可寫 reward function：`+1 if expected_tool == called_tool else 0`，跑 GRPO（metadata 已經有 `expected_tool`、`tool_correct`、`numbers_correct` 三個訊號可用）。

### B6. Per-Building Multi-LoRA Adapters

`knowledge_workbench/groups/{at2007, at2045, at5043, ...}/MEMORY.md` 每棟有自己的歷史 finding。可以：

- 為大耗能戶（`energy_tier = HIGH`，以經授權資料中標示為高耗能的建築為例）各訓練一個 small adapter（128 rank）。
- runtime 切換：使用者選 AT2045 → 載入 `adapters/at2045.safetensors`，回答更精準。
- 只需要在 LM Studio 端做 adapter swap，不用重啟整個 server。

### B7. 在 Edge / 工地裝置做 on-device 異常分類

L1 異常分類（4 個 pattern：zero_flatline / spike / oscillation / step_change）佔 75 筆 SFT。LoRA 後這個任務 latency 應該夠低，可以：

- 編成 GGUF 4-bit，部署到 NTU 機房邊緣 GPU。
- 取代 cloud LLM 做即時 BMS 警示，每分鐘一輪。
- 不需要 demo 主機線上才能跑，純 BMS 端 on-prem。

### B7.5. 正式做 cloud → local 知識蒸餾（不只是 SFT 模仿）

對齊 `19-emerging-techniques/knowledge-distillation/SKILL.md`：

目前 `template_teacher_v1` 只用文字模板生 `output`，這是 **response distillation** 的弱化版（連 logits 都沒蒸到）。LoRA 跑完後可以升級成：

- **Soft target distillation**：cloud（Gemini / GPT）生 chosen + **保留 logits / top-k 機率**，student（local LoRA Gemma）做 KL divergence loss（temperature=2.0、α=0.5）。
- **Reverse KLD（MiniLLM）**：對 student 已經會錯的 case 反向蒸餾，避免 student 記住 teacher 的低機率噪聲。
- 對應到我們的架構：`demo_assistant.py` 的 cloud_adapter 已經會跑一遍 cloud，只要把 `logprobs=true`（或抓 top-5 probs）存進 trace，後續離線蒸餾就有材料。

### B8. Data Flywheel：trace 回灌

LoRA 訓完之後，本地回答的 trace 質量會提升。配合 A9 把 `tool_trace` 寫進 wiki：

1. 收 7 天 production trace
2. cloud judge 抽樣 100 條 → 標 chosen/rejected
3. 加進 `energy_preference_pairs.jsonl` 跑下一輪 DPO
4. 每 2 週迭代一次

`docs/LOCAL_ENERGY_STEWARD_ROADMAP.md` 已有提到資料飛輪概念，這裡是具體落地步驟。

### B9. 拿掉「強迫繁中」的硬規則

system prompt 寫了 `「使用繁體中文」`、`「Respond in the same language as the user query」`，是因為 base Gemma 在中英混雜 prompt 會掉繁中。
LoRA 訓練資料 100% 繁中後，可以拿掉這條規則，給使用者 explicit `lang` 參數，並且讓英文 prompt 不再被強制翻成中文。

### B10. 把 `energy_intent_router.py` 退役

LoRA 後模型自己會 routing，目前的 keyword router 在 production 是個 **後備保險（safety net）** 而不是主要決策者。可以：

- 改成 `intent_router_audit_only` 模式，只負責記錄「規則認為應該用 X 工具，但模型選了 Y」 → 收集分歧 case 做下一輪訓練。
- 或者反過來：把 router 當成 ground-truth label generator，繼續批次生成新的 SFT 資料。

---

## Part C. 訓練流程與框架選型（依據 AcademicForge 的 SKILL 文件）

來源：`Downloads/skill/AGENT/AcademicForge-master/skills/AI-research-SKILLs/`

### C1. 框架選擇矩陣

| 階段 | 推薦框架 | SKILL 路徑 | 為什麼 |
|---|---|---|---|
| SFT（主要） | **Unsloth** | `03-fine-tuning/unsloth/` | Gemma 3 / 3n 原生支援，2–5x 快、50–80% 省 VRAM，含 `train_on_responses_only`、`get_chat_template("gemma-3")`、QAT+LoRA 一條龍 |
| SFT（備選 / 多 GPU） | LLaMA-Factory | `03-fine-tuning/llama-factory/` | 想要 YAML 設定檔、WebUI、原生多卡 |
| LoRA / QLoRA 細節 | PEFT | `03-fine-tuning/peft/` | r、alpha、target_modules 規劃 |
| DPO | TRL `DPOTrainer` | `06-post-training/trl-fine-tuning/` | Windows 可跑、上手快 |
| GRPO（可選） | TRL `GRPOTrainer` | `06-post-training/grpo-rl-training/` | 需 vLLM（Windows 走 WSL），可寫自訂 reward |
| 蒸餾 | MiniLLM / 自寫 KL loss | `19-emerging-techniques/knowledge-distillation/` | cloud → local 升級版 |
| 評估 | lm-evaluation-harness | `11-evaluation/lm-evaluation-harness/` | LoRA 前後跨 epoch 比較 |
| 實驗追蹤 | W&B | `13-mlops/weights-and-biases/` | run name 帶 `metadata.topic`，方便分群看 |

### C2. 推薦的 LoRA 超參數（套到 demo 訓練 manifest）

依 Unsloth `references/llms-txt.md:1144` 的範本，加上我們資料量小（116 筆 SFT + 429 筆 routing）的調整：

```python
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-3-4b-it",   # 若 Gemma 4 E2B 已上 unsloth 改用之；否則先用 3-4b-it 練手
    max_seq_length = 2048,                  # 我們最大樣本 ~283 token，2048 綽綽有餘
    load_in_4bit = True,                    # QLoRA，省 VRAM
)
tokenizer = get_chat_template(tokenizer, "gemma-3")

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                                  # 資料量小，r=8~16 即可
    lora_alpha = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_dropout = 0.05,
    use_gradient_checkpointing = "unsloth",  # 省更多 VRAM
    random_state = 3407,
)

# SFTConfig: 545 筆 × 3 epoch ≈ 1635 steps；val 9 筆每 50 step eval 一次
```

**重點**：
- `target_modules` 要**全打**（含 MLP 的 gate/up/down）—— 只打 attention 不夠，工具路由會學不起來。
- `train_on_responses_only(trainer, instruction_part="<start_of_turn>user\n", response_part="<start_of_turn>model\n")` —— Gemma 3 chat template 的 turn 標記，務必對齊，否則 mask 會錯位。
- LR：`2e-4`（QLoRA 標準），warmup_ratio=0.03，cosine schedule。
- 若 OOM：先降 `per_device_train_batch_size=1`、開 `gradient_accumulation_steps=8`，比降 `max_seq_length` 安全。

### C3. Windows / WSL / Colab 的決策點

| 情境 | 環境 | 原因 |
|---|---|---|
| SFT + DPO | Windows + Unsloth（pip 版） | Unsloth 官方支援 Windows，`pip install "unsloth[windows] @ git+..."` |
| GRPO | **WSL2** 或 Colab | vLLM 在 Windows 原生不支援（Unsloth llms-txt:845 明確寫） |
| 量化導出 GGUF | Windows / WSL 都可 | 已有 `Documents/llama-cpp-turboquant` |
| 第一次練手 | Colab T4（Gemma 3 1B notebook） | 不用設環境，1B 模型 15 分鐘練完 |
| 正式跑 4B+ | 本機 RTX 30/40 系（≥12GB） 或 Colab A100 | 4B QLoRA 約需 8–10GB VRAM |

`docs/COLAB_LORA_TUTORIAL.md` 已有 Colab 流程，但建議改成 **直接從 Unsloth 官方 notebook fork**（每月更新、bug 修得快）：
- Gemma 3 1B GRPO：`https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(1B)-GRPO.ipynb`
- Gemma 3 4B SFT：unslothai/notebooks 的 `Gemma3_(4B)-Conversational.ipynb`

### C4. 資料工程：對齊 `llm-data-factory` Skill 的流程

`.claude/skills/llm-data-factory/SKILL.md` 與我們現有的 `data/energy_sft_factory/` 結構幾乎一樣（`processed/` + `training/` + `reports/`），但**少做了 4 步**：

1. **Synthetic data + Judge 雙模型**：現在 teacher 跟 judge 都是 `template_teacher_v1` / `heuristic_energy_judge_v1`。改成 cloud 模型當 judge（GPT/Gemini/Claude），對 sample 抽 20% 做盲評，重算 `judge_score`。
2. **Train/Val 分布平衡**：現在 val 只有 9 筆，且都是 anomaly_classification。應該 stratify by `task_type` × `pattern` × `building_type`。
3. **Tool-trajectory 樣本**：把 `harness_full_phase1.jsonl` 裡 `expected_tool` 對應的 trace 從 demo runtime 撈出來（現在沒撈），補成 multi-turn `assistant.tool_calls → tool → assistant` 三段格式。
4. **Refusal 擴增**：從 5 筆擴到 30+。可參考 `07-safety-alignment/llamaguard/` 與 `constitutional-ai/` 的拒答模板。

### C5. 訓練後驗收清單（取代目前 `downstream_validation.json` 的形式驗證）

對齊 `11-evaluation/lm-evaluation-harness/SKILL.md`：

```yaml
# tasks/ntu_energy_eval.yaml
task: ntu_energy_routing
dataset_path: D:/idf優化/demo/data/lora/test_holdout.jsonl
output_type: generate_until
metric_list:
  - metric: tool_correct        # metadata.expected_tool 對齊
  - metric: numbers_correct     # 答案中數字 ∈ tool result
  - metric: format_compliance   # 結論/依據/假設/建議 四段都在
  - metric: refusal_recall      # OOD/造假題正確拒答
generation_kwargs:
  temperature: 0.1
  max_new_tokens: 1024
```

跑 `lm_eval --model hf --model_args pretrained=...,peft=adapter_path --tasks ntu_energy_routing` 對比 base vs LoRA。**Pass 條件**：4 項 metric 都不退化、tool_correct 提升 ≥ 10pp、format_compliance ≥ 95%。

---

## 建議的執行順序

```
Week 1 — Part A 阻擋訓練品質的紅字項目
  A0（chat_template + tool-call 格式轉換）
  A1（標籤錯）→ A2（test set + lm-eval-harness task yaml）→ A8（trace gate）

Week 2 — Part A 工程清理 + Part C 環境
  A3, A4, A5, A6, A7, A9, A10
  C3（決定 Windows / WSL / Colab）→ 安裝 Unsloth + TRL + lm-eval-harness

Week 3 — 啟動 LoRA 訓練
  C2 超參：Unsloth + Gemma 3 4B-it（或 Colab Gemma 3 1B 練手）
  資料：清理後的 final_sft_dataset.jsonl + harness_full_phase1.jsonl + 補的 tool-call 樣本
  驗收：C5 lm-eval-harness 跑 base vs LoRA

Week 4 — Part B 上線項目
  B1（瘦身）+ B2（multi-hop）+ B3（local-first）+ B9（拿掉強迫繁中）

Week 5+ — Part B / C 進階
  B5 DPO（TRL）→ B8 飛輪 → B7.5 蒸餾（cloud logits）→ B6 multi-LoRA
  視需要 WSL 上 GRPO（reward = tool_correct + numbers_correct）
```

---

## 相關檔案速查

| 主題 | 路徑 |
|---|---|
| Agent 主迴圈 | `demo/src/lm_studio_client.py` |
| 意圖路由 | `demo/src/energy_intent_router.py` |
| Assistant 服務 | `demo/src/demo_assistant.py` |
| 訓練資料（總） | `demo/data/lora/harness_full_phase1.jsonl`（429 筆） |
| 訓練資料（routing） | `demo/data/lora/harness_routing_qa_90.jsonl` |
| SFT 工廠輸出 | `demo/data/energy_sft_factory/training/{train,val,smoke_test}.jsonl` |
| 訓練 manifest | `demo/data/energy_sft_factory/training/training_manifest.json` |
| 知識庫狀態 | `demo/data/knowledge_workbench/state/{chunks,documents,memory_index,curated_traces}.json` |
| Wiki memory | `demo/data/knowledge_workbench/memory/wiki/` |
| 既有 roadmap | `demo/docs/LOCAL_ENERGY_STEWARD_ROADMAP.md` |
| Colab 教學 | `demo/docs/COLAB_LORA_TUTORIAL.md` |
| AcademicForge 技能庫根 | `C:/Users/User/Downloads/skill/AGENT/AcademicForge-master/skills/AI-research-SKILLs/` |
| Unsloth SKILL | `…/03-fine-tuning/unsloth/SKILL.md` + `references/llms-txt.md` |
| LLaMA-Factory SKILL | `…/03-fine-tuning/llama-factory/SKILL.md` |
| PEFT SKILL | `…/03-fine-tuning/peft/SKILL.md` |
| TRL（SFT/DPO/GRPO） | `…/06-post-training/trl-fine-tuning/SKILL.md` |
| GRPO RL | `…/06-post-training/grpo-rl-training/SKILL.md` |
| 知識蒸餾 | `…/19-emerging-techniques/knowledge-distillation/SKILL.md` |
| 評估 | `…/11-evaluation/lm-evaluation-harness/SKILL.md` |
| 實驗追蹤 | `…/13-mlops/weights-and-biases/SKILL.md` |
