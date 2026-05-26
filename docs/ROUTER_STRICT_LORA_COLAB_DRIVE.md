# 第一輪 Router-Strict LoRA：Colab 與雲端硬碟清單

這份文件只處理第一輪 LoRA：**工具路由 + 安全拒答 + JSON-only tool call**。
不要把 `explainer_sft.jsonl` 或 `legacy_cleaned.jsonl` 混進這輪，因為它們是自然語言回答資料，會污染「只輸出 JSON」的訓練目標。

## 1. 本機先確認資料庫

在本機跑：

```powershell
cd D:\idf優化\demo
python tools\harness_v02\merge_harness_v02.py
python tools\harness_v02\audit_harness_v02.py
```

確認：

- `tools\harness_v02\harness_v02_manifest.json` 裡 `profile` 以 `router_strict` 開頭
- `tools\harness_v02\harness_v02_audit.md` 是 `PASS_WITH_WARNINGS` 或更好
- 不可以有 `assistant_not_json`、`expected_tool_mismatch`、`train_val_prompt_overlap`

目前 v0.4 targeted 資料量：

| 檔案 | 筆數 | 用途 |
|------|------|------|
| `harness_v02_train.jsonl` | 364 | LoRA 訓練 |
| `harness_v02_val.jsonl` | 42 | 固定驗證 |
| `harness_v02_smoke.jsonl` | 6 | 快速檢查 |
| `harness_v02_manifest.json` | 1 | 資料庫版本與 eval gates |

## 2. Google Drive 目錄

在 Google Drive 建這個資料夾：

```text
MyDrive/energy_lora_router_v02/
```

裡面放成這樣：

```text
MyDrive/energy_lora_router_v02/
├── colab_train_router_strict_lora.py
├── router_strict_lora_colab_cells.ipynb
├── data/
│   ├── harness_v02_train.jsonl
│   ├── harness_v02_val.jsonl
│   ├── harness_v02_smoke.jsonl
│   └── harness_v02_manifest.json
└── reports/
    ├── harness_v02_audit.md        # 建議放，方便查
    └── harness_v02_audit.json      # 建議放，方便查
```

## 3. 要上傳哪些本機檔案

必要檔案：

```text
D:\idf優化\demo\scripts\colab_train_router_strict_lora.py
D:\idf優化\demo\notebooks\router_strict_lora_colab_cells.ipynb
D:\idf優化\demo\tools\harness_v02\harness_v02_train.jsonl
D:\idf優化\demo\tools\harness_v02\harness_v02_val.jsonl
D:\idf優化\demo\tools\harness_v02\harness_v02_smoke.jsonl
D:\idf優化\demo\tools\harness_v02\harness_v02_manifest.json
```

建議一起上傳：

```text
D:\idf優化\demo\tools\harness_v02\harness_v02_audit.md
D:\idf優化\demo\tools\harness_v02\harness_v02_audit.json
```

不要上傳到第一輪資料夾的訓練資料：

```text
D:\idf優化\demo\tools\harness_v02\explainer_sft.jsonl
D:\idf優化\demo\tools\harness_v02\legacy_cleaned.jsonl
```

它們之後另開 answer SFT 或清理後再用。

## 4. Colab 前置設定

1. 開 Google Colab，選 GPU runtime。
2. 建議 A100；T4 也可，但要把 `LOAD_IN_4BIT=true`。
3. Hugging Face 需要能讀 Gemma 模型。若模型是 gated，先到 Hugging Face 同意授權。
4. 建議設定 Colab 環境變數：

```python
import os
os.environ["HF_TOKEN"] = "hf_xxx"
os.environ["WANDB_API_KEY"] = "xxx"       # 可選
```

如果 Gemma 4 E2B 的 Hugging Face repo id 不同，改：

```python
os.environ["MODEL_ID"] = "正確的-huggingface-model-id"
```

T4 runtime 用：

```python
os.environ["LOAD_IN_4BIT"] = "true"
os.environ["TRAIN_BATCH_SIZE"] = "2"
os.environ["GRAD_ACCUM_STEPS"] = "8"
```

## 5. 在 Colab 執行

### 推薦：用 Notebook cell-by-cell 跑

在 Google Drive 直接打開：

```text
MyDrive/energy_lora_router_v02/router_strict_lora_colab_cells.ipynb
```

依序跑 Cell 1 到 Cell 12。這樣你可以分段檢查：

| Cell | 做什麼 |
|------|--------|
| 1 | 掛載 Drive |
| 2 | 設定模型、batch、LoRA 參數 |
| 3 | 安裝依賴 |
| 4 | 載入訓練工具模組 |
| 5 | 登入與資料檢查 |
| 6 | 載入模型 |
| 7 | render dataset |
| 8 | 掛 LoRA |
| 9 | 訓練 |
| 10 | smoke 評測 |
| 11 | validation 評測 |
| 12 | 匯出 GGUF |

如果 Cell 3 安裝後 Colab 要你重啟 runtime，重啟後從 Cell 1 開始，Cell 3 可以略過。

### 備用：一格跑完整腳本

最簡單方式：在 Colab 新增一個 cell，跑：

```python
from google.colab import drive
drive.mount("/content/drive")

%run /content/drive/MyDrive/energy_lora_router_v02/colab_train_router_strict_lora.py
```

如果第一次安裝套件後 Colab 要求重啟 runtime：

1. 重啟 runtime
2. 再跑：

```python
import os
os.environ["INSTALL_DEPS"] = "false"

from google.colab import drive
drive.mount("/content/drive")

%run /content/drive/MyDrive/energy_lora_router_v02/colab_train_router_strict_lora.py
```

## 6. 訓練完成後會產生什麼

輸出會在：

```text
MyDrive/energy_lora_router_v02/outputs/gemma_router_strict_v02/
```

重要檔案：

```text
adapter/                         # LoRA adapter
checkpoints/                     # 訓練 checkpoints
eval/smoke_after_train_summary.json
eval/val_after_train_summary.json
merged_16bit/                    # 合併後模型
gguf_q4_k_m/                     # Unsloth 原始 GGUF 匯出
final_gguf/gemma-4-e2b-it-energy-router-v02-Q4_K_M.gguf
```

看 `eval/val_after_train_summary.json`：

- `accuracy` 目標先看是否接近或超過 `0.80`
- `malformed_rate` 目標低於 `0.05`
- hard/trap 類若仍低，下一輪補 hard negatives

## 7. 拿回本機 demo

把這個檔案下載回本機：

```text
MyDrive/energy_lora_router_v02/outputs/gemma_router_strict_v02/final_gguf/gemma-4-e2b-it-energy-router-v02-Q4_K_M.gguf
```

放到：

```text
D:\idf優化\demo\runtime\gemma\models\gemma-4-e2b-it-energy-router-v02-Q4_K_M.gguf
```

然後設定本機環境變數或啟動前指定：

```cmd
set ENERGY_GEMMA_MODEL_PATH=D:\idf優化\demo\runtime\gemma\models\gemma-4-e2b-it-energy-router-v02-Q4_K_M.gguf
set ENERGY_LOCAL_LLM_MODEL=gemma-4-e2b-it-energy-router-v02-Q4_K_M.gguf
```

重啟：

```powershell
cd D:\idf優化\demo
.\scripts\start_local_gemma.ps1
```

再跑本機評測：

```powershell
python scripts\evaluate_tool_routing.py `
  --eval-set tools\harness_v02\harness_v02_val.jsonl `
  --base-url http://127.0.0.1:8088/v1 `
  --model gemma-4-e2b-it-energy-router-v02-Q4_K_M.gguf `
  --output dev_artifacts\eval\router_strict_v02_local_val.jsonl `
  --verbose
```

## 8. 第一輪通過標準

這輪只看 router，不看回答文采。v0.4 targeted 額外補了 5 組混淆：`query_energy_records/list_campus_stats`、`seasonal/recommend`、`counterfactual/trend/refusal`、`OpenBSE/portfolio`、`trap/semantics/sources`。

| 指標 | 目標 |
|------|------|
| tool accuracy | >= 80% |
| malformed JSON | < 5% |
| hard/trap accuracy | >= 60% |
| safety refusal | 不亂答、不亂呼叫工具 |

如果沒過，先不要加回答資料。先回收錯題，補到 `router_sft.jsonl` 或 `safety_sft.jsonl`，重跑 merge/audit，再訓第二輪 router。
