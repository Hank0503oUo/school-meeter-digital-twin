# Gemma 4 E2B IT — Colab LoRA 微調完整教學

> 目標：在 Google Colab Pro（**A100 40 GB**，7 compute units/hr，300 單位 ≒ 42 小時）
> 上對 **Gemma 4 E2B IT** 做 LoRA 微調。
> 核心引擎用 **Unsloth**（2x 速度、-60% VRAM、一鍵 GGUF 匯出），
> 訓練完成直接拿到本機 `demo` 的 `llama-server` 跑。

---

## 目錄

1. [前置需求與授權](#1-前置需求與授權)
2. [Colab 筆記本：GPU 選擇與基本環境](#2-colab-筆記本gpu-選擇與基本環境)
3. [VSCode Colab 插件（選用但推薦）](#3-vscode-colab-插件選用但推薦)
4. [安裝依賴（Unsloth + W&B）](#4-安裝依賴unsloth--wb)
5. [下載模型 — HuggingFace](#5-下載模型--huggingface)
6. [準備訓練資料 — 使用 harness v0.2 資料庫](#6-準備訓練資料--使用-harness-v02-資料庫)
7. [載入模型（Unsloth 加速）](#7-載入模型unsloth-加速)
8. [LoRA 設定與訓練](#8-lora-設定與訓練)
9. [合併 + 匯出 GGUF（Unsloth 一鍵）](#9-合併--匯出-ggufunsloth-一鍵)
10. [部署回本機 demo](#10-部署回本機-demo)
11. [評測與迭代（W&B 實驗追蹤）](#11-評測與迭代wb-實驗追蹤)
12. [常見問題](#12-常見問題)

---

## 1. 前置需求與授權

| 項目 | 說明 |
|------|------|
| Google 帳號 | Colab Pro（300 compute units）。A100 40 GB = 7 units/hr，可用 **~42 小時** |
| HuggingFace 帳號 | 需到 [gemma 模型頁面](https://huggingface.co/google) 同意授權條款，再產生 **Read** token |
| 本機 Gemma GGUF | 你目前使用 `gemma-4-E2B-it-Q4_K_M.gguf`，微調後會取代或並存 |
| harness v0.2 資料庫 | 位於 `demo/tools/harness_v02/`，已產出 train/val/manifest，可直接上 Colab 微調 |

### HuggingFace Token 設定

1. 登入 https://huggingface.co → Settings → Access Tokens → New token（Read 權限）
2. 到 Gemma 模型頁面（如 `google/gemma-4-e2b-it`）點 **Agree and access repository**
3. 在 Colab 裡用 `huggingface-cli login` 輸入 token

---

## 2. Colab 筆記本：GPU 選擇與基本環境

### 2.1 建立 Colab 筆記本

1. 開啟 https://colab.research.google.com → 新增筆記本
2. **Runtime → Change runtime type** → 選 **`A100`**（7 compute units/hr）
3. Colab Pro 可用 GPU 對照：

| GPU | VRAM | Units/hr | 你的情境 |
|-----|------|----------|---------|
| T4  | 16 GB | 0（免費）| 備用，需 QLoRA 4-bit |
| V100 | 16 GB | ~3 | 不推薦 |
| L4  | 24 GB | ~3–5 | 可用 |
| A100 | 40 GB | **7** | **推薦首選** |
| A100 | 80 GB | ~13 | 有錢就選這個 |

### 2.2 確認 GPU

```python
!nvidia-smi
```

預期看到 `Tesla T4` 或 `A100`，VRAM ≥ 15 GB。

---

## 3. VSCode Colab 插件（選用但推薦）

VSCode 的 **Google Colab** 擴充功能讓你直接從 VSCode 連到 Colab runtime，
享有完整 IDE 補全、終端機、檔案管理。

### 3.1 安裝步驟

1. VSCode 安裝 **Google Colab** 擴充功能（搜尋 `Google Colab` by Google）
2. 在 VSCode 命令面板（`Ctrl+Shift+P`）輸入 `Colab: Open Notebook`
3. 選擇 Google 帳號授權 → 開啟既有筆記本或新建
4. VSCode 會以 `.ipynb` 編輯器模式開啟，cell 直接執行，等同網頁版但更順手

### 3.2 替代方案：SSH tunnel（Colab → VSCode）

如果 Colab 插件不順，可改用 `colab-ssh` 把 Colab 變遠端機器：

```python
# Colab cell
!pip install colab_ssh --quiet
from colab_ssh import launch_ssh_cloudflared
launch_ssh_cloudflared(password="your_password_here")
```

然後在 VSCode 用 **Remote-SSH** 連線。兩種方式效果一樣——都是把 Colab GPU 當遠端機。

### 3.3 為什麼推薦用 VSCode

| 好處 | 說明 |
|------|------|
| 終端機存取 | 可直接 `ls`、`cat`、`pip install`，不用每步都寫 cell |
| 檔案總管 | 上傳 `harness_v02_train.jsonl` / `harness_v02_val.jsonl`、下載 GGUF 拖拉就好 |
| Git 整合 | 訓練完直接 commit 權重到分支 |
| 補全 + Lint | 寫資料處理腳本更有效率 |

---

## 4. 安裝依賴（Unsloth + W&B）

### 4.1 技術棧總覽

| 套件 | 角色 | 為什麼需要 |
|------|------|-----------|
| **Unsloth** | 底層加速引擎 | Triton kernel 優化，**2x 速度、-60% VRAM**，內建 GGUF 匯出 |
| **transformers** | 模型載入 | HF 標準 API，Unsloth 包裝後呼叫 |
| **peft** | LoRA 設定 | 參數高效微調配置（rank、alpha、target modules） |
| **trl (SFTTrainer)** | 訓練迴圈 | 監督式微調，處理 padding、batching、logging |
| **datasets** | 資料載入 | `load_dataset("json", data_files=...)` 一秒讀入 JSONL |
| **wandb** | 實驗追蹤 | 自動畫 loss 曲線，比較消融實驗（Baseline vs Tool-routing vs 格式） |

### 4.2 安裝

```python
# Cell 1: Install Unsloth + dependencies
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes
!pip install wandb datasets huggingface_hub sentencepiece
```

> **Unsloth** 會自動安裝正確版本的 `torch`、`transformers`、`xformers`。
> 如果有衝突，重啟 runtime 再跑一次。

### 4.3 登入 HuggingFace + W&B

```python
# Cell 2: Login
from huggingface_hub import login
login(token="hf_xxxxxxxxxxxxxxxxxxxxxxxxx")

import wandb
wandb.login(key="xxxxxxxxxxxxxxxxxxxxxxxxxx")  # https://wandb.ai/authorize
```

或互動式：

```python
!huggingface-cli login
!wandb login
```

---

## 5. 下載模型 — HuggingFace

### 5.1 確認模型 ID

Gemma 4 E2B IT 的 HuggingFace ID 請到 https://huggingface.co/models 搜尋 `gemma-4` 確認。
目前最可能的 ID 為：

```
google/gemma-4-e2b-it
```

如果搜不到，可能是 `google/gemma-4-2b-it` 或其他變體——以 HF 頁面為準。

> 不需要手動下載。§7 的 Unsloth `AutoModelForCausalLM` 會自動處理。

---

## 6. 準備訓練資料 — 使用 harness v0.2 資料庫

### 6.1 目前資料庫狀態

現在已經不是 5 筆 curated traces 的階段。`demo/tools/harness_v02/` 已經構好第一輪
LoRA/SFT 資料庫。依照 SFT data-factory 審視結果，這一輪先採 **router-strict**：
只訓「工具路由 + 安全拒答」的 JSON 介面紀律，不把自然語言回答風格混進同一輪。

| 檔案 | 用途 | 筆數 |
|------|------|------|
| `harness_v02_train.jsonl` | Colab 訓練集 | 297 |
| `harness_v02_val.jsonl` | 固定驗證與本機評測集 | 38 |
| `harness_v02_smoke.jsonl` | 快速 smoke 評測集 | 4 |
| `harness_v02_manifest.json` | 版本、來源、分佈、eval gates | 1 |

manifest 目前 router-strict 總計 335 筆：

| 來源 | 筆數 | 目的 |
|------|------|------|
| `router_sft.jsonl` | 300 | 工具路由、混淆題、malformed query |
| `safety_sft.jsonl` | 35 | 拒答、高風險請求、不可捏造數字 |

以下兩個檔案先當 sidecar，不進第一輪 router LoRA：

| sidecar | 筆數 | 暫不混入原因 |
|------|------|------|
| `explainer_sft.jsonl` | 10 | assistant target 是自然語言回答，不是 JSON tool call |
| `legacy_cleaned.jsonl` | 90 | 多數是完整回答或舊工具名，適合另做 answer SFT/清理後再進 |

目前最低資料量門檻已達標。這一版先拿來訓練「工具路由穩定輸出 JSON」；
回答風格、完整能源分析與多輪 tool trace 另開下一輪資料庫，避免把兩種行為信號混在一起。

### 6.2 上傳到 Colab

從本機上傳這四個檔案：

```text
D:\idf優化\demo\tools\harness_v02\harness_v02_train.jsonl
D:\idf優化\demo\tools\harness_v02\harness_v02_val.jsonl
D:\idf優化\demo\tools\harness_v02\harness_v02_smoke.jsonl
D:\idf優化\demo\tools\harness_v02\harness_v02_manifest.json
```

Colab 目錄建議放成：

```text
/content/data/harness_v02_train.jsonl
/content/data/harness_v02_val.jsonl
/content/data/harness_v02_smoke.jsonl
/content/data/harness_v02_manifest.json
```

### 6.3 快速檢查資料庫

```python
# Cell 4: Validate harness v0.2 dataset
import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path("/content/data")
TRAIN_FILE = DATA_DIR / "harness_v02_train.jsonl"
VAL_FILE = DATA_DIR / "harness_v02_val.jsonl"
SMOKE_FILE = DATA_DIR / "harness_v02_smoke.jsonl"
MANIFEST_FILE = DATA_DIR / "harness_v02_manifest.json"

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            assert "messages" in item, f"{path}:{line_no} missing messages"
            assert item["messages"][-1]["role"] == "assistant", f"{path}:{line_no} missing assistant target"
            rows.append(item)
    return rows

train_rows = read_jsonl(TRAIN_FILE)
val_rows = read_jsonl(VAL_FILE)
smoke_rows = read_jsonl(SMOKE_FILE)
manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))

print("train:", len(train_rows))
print("val:", len(val_rows))
print("smoke:", len(smoke_rows))
print("manifest:", manifest["version"], manifest["profile"], manifest["total"], manifest["eval_gates"])
print("train categories:", Counter(x.get("category", "unknown") for x in train_rows))
print("val difficulty:", Counter(x.get("difficulty", "unknown") for x in val_rows))
```

預期：

```text
train: 297
val: 38
smoke: 4
manifest: 0.2 router_strict 335 ...
```

### 6.4 如果要重新合併資料庫

本機已經有合併腳本。只有在新增 DeepSeek 補題、手寫 trap 題或清理 legacy 資料後才需要重跑：

```powershell
cd D:\idf優化\demo
python tools\harness_v02\merge_harness_v02.py
python tools\harness_v02\audit_harness_v02.py
```

重跑後確認：

- `harness_v02_manifest.json` 的 `profile` 是 `router_strict`
- `harness_v02_audit.md` 是 `PASS_WITH_WARNINGS` 或更好
- 沒有 `assistant_not_json`、`expected_tool_mismatch`、`train_val_prompt_overlap`
- 剩餘 warning 若是「真實建物名稱」可接受於封閉 demo；若要公開模型，需匿名化

### 6.5 下一輪補題方向

| 缺口 | 建議補題 |
|------|----------|
| base Gemma 過度拒答 | 增加合法能源查詢、簡短查詢、樓棟別名題 |
| JSON 格式不穩 | 增加 malformed/repair 題，assistant 只輸出 JSON |
| hard 類工具混淆 | 補 `run_pvid`、`OpenBSE`、`detect_energy_anomalies` 對照題 |
| 回答品質 | 另建 answer SFT，不要混進純 router 第一輪 |

---

## 7. 載入模型（Unsloth 加速）

Unsloth 用 Triton kernel 替換 HuggingFace 的原生 attention/MLP forward，
在 A100 上直接 **2x 速度、-60% VRAM**。對 E2B 小模型來說，A100 40 GB 根本用不完。

> **視覺能力保護**：Gemma 4 E2B IT 是多模態模型，搭配 `mmproj-gemma-4-E2B-it-BF16.gguf`
> 可處理電費單、用電圖表等影像輸入。LoRA 只訓語言層，**不碰 vision encoder**，
> 訓練完仍可搭配原版 mmproj 做視覺推理。

```python
# Cell 6: Load model with Unsloth
from unsloth import FastLanguageModel
import torch

MODEL_ID = "google/gemma-4-e2b-it"  # 確認後替換

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=2048,
    dtype=None,                  # 自動偵測（A100 → bfloat16，T4 → float16）
    load_in_4bit=False,          # A100 不需要量化！T4 改 True
)

print(f"Model loaded: {MODEL_ID}")
print(f"Vocab size: {tokenizer.vocab_size}")

# 確認 vision encoder 存在但凍結（不訓練）
vision_params = sum(
    p.numel() for n, p in model.named_parameters()
    if "vision" in n.lower() or "image" in n.lower() or "mmproj" in n.lower()
)
total_params = sum(p.numel() for p in model.parameters())
if vision_params > 0:
    print(f"Vision params: {vision_params:,} / {total_params:,} ({vision_params/total_params:.1%}) — will be FROZEN")
else:
    print("No separate vision encoder found (language-only model or shared architecture)")
```

### 7.1 Unsloth vs 原生 HuggingFace 比較

| | 原生 HF | Unsloth |
|---|---------|---------|
| 載入速度 | 基準 | **快 2x** |
| VRAM 佔用 | 基準 | **少 60%** |
| 訓練速度 | 基準 | **快 2x** |
| GGUF 匯出 | 需手動 llama.cpp | **一行指令** |
| ShareGPT 格式 | 需手動 tokenize | **內建支援** |

> **T4 備用**：改 `load_in_4bit=True` 即可，Unsloth 會自動處理 QLoRA。

---

## 8. LoRA 設定與訓練

### 8.1 LoRA 配置（Unsloth 一行搞定）

**只訓語言層（attention + MLP），不碰 vision encoder。**
這樣微調後的 GGUF 仍可搭配原版 mmproj 讀電費單、圖表。

```python
# Cell 7: Apply LoRA with Unsloth — language-only targets
model = FastLanguageModel.get_peft_model(
    model,
    r=32,                        # A100 夠大，拉到 32（T4 用 16）
    lora_alpha=64,               # 縮放因子 = 2 × r
    lora_dropout=0.05,
    target_modules=[
        # ── 語言模型層（LoRA 訓練）──
        "q_proj", "k_proj", "v_proj", "o_proj",     # attention
        "gate_proj", "up_proj", "down_proj",          # MLP
    ],
    # ❌ 不包含：vision_tower, multi_modal_projector, image 相关層
    # 這些層會自動被凍結，LoRA 不會動到
    bias="none",
    use_gradient_checkpointing="unsloth",   # Unsloth 優化版的 GC
    random_state=42,
)

model.print_trainable_parameters()

# 二次確認：沒有任何 vision 參數是 trainable
trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
vision_leaked = [n for n in trainable_names if any(k in n.lower() for k in ("vision", "image", "mmproj", "visual"))]
if vision_leaked:
    print(f"⚠️ WARNING: vision params leaked into LoRA: {vision_leaked[:5]}")
else:
    print("✅ Confirmed: all LoRA targets are language-only, vision encoder is frozen")
```

預期輸出類似：
```
trainable params: 26,xxx,xxx || all params: 2,xxx,xxx,xxx || trainable%: ~1%
```

> Unsloth 的 `use_gradient_checkpointing="unsloth"` 比原生 `gradient_checkpointing_enable()`
> **再省 30% VRAM**，因為它用自製 Triton kernel 重寫了 checkpoint 邏輯。

### 8.2 載入訓練資料（datasets 一秒讀入）

```python
# Cell 8: Load dataset
from datasets import load_dataset

data_files = {
    "train": "/content/data/harness_v02_train.jsonl",
    "validation": "/content/data/harness_v02_val.jsonl",
}
dataset = load_dataset("json", data_files=data_files)
print(dataset)

def render_chat(example):
    # harness_v02 是 messages 格式；先轉成純 text，讓 SFTTrainer 穩定吃。
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }

dataset = dataset.map(render_chat, remove_columns=dataset["train"].column_names)
train_ds = dataset["train"]
eval_ds = dataset["validation"]
print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")
```

### 8.3 開始訓練（A100 + Unsloth + W&B）

```python
# Cell 9: Train with Unsloth + W&B tracking
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel

EXPERIMENT_NAME = "gemma-e2b-energy-lora-v1"  # 每次實驗改名字

training_args = SFTConfig(
    output_dir="./gemma-lora-output",
    num_train_epochs=3,
    per_device_train_batch_size=8,       # A100: 大方開 8
    gradient_accumulation_steps=2,       # 有效 batch = 8 × 2 = 16
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    bf16=True,
    report_to="wandb",                   # ← W&B 自動追蹤
    run_name=EXPERIMENT_NAME,
    max_seq_length=2048,
    packing=False,
    dataset_text_field="text",           # §8.2 已把 messages render 成 text
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
)

# W&B 會自動記錄 train/eval loss、learning rate、GPU 使用率
trainer.train()
print("Training complete!")
```

### 8.4 W&B 實驗追蹤

訓練過程中，W&B 會自動在 https://wandb.ai 顯示：

- **Train Loss** — 應該穩定下降
- **Eval Loss** — 如果開始上升 = overfitting（該停了）
- **Learning Rate** — cosine schedule 的曲線
- **GPU Memory** — 確認沒爆 VRAM

你的消融實驗命名建議：

```python
# 四組消融實驗
EXPERIMENTS = {
    "v02-router-only":       "只用 router_sft.jsonl，確認 JSON tool 格式先穩",
    "v02-router-strict":     "使用 harness_v02_train/val，僅 routing + safety JSON target",
    "v03-hard-negative":     "v02 + 新增 PIVD/OpenBSE/anomaly hard negatives",
    "v03-answer-style":      "另加回答品質資料，訓練結論→依據→限制→建議",
}
```

每次跑實驗只要改 `EXPERIMENT_NAME` 和訓練資料，W&B 會自動把四條 loss 曲線疊在一起比較。

### 8.5 訓練時間（Unsloth on A100）

| GPU | 資料量 | Batch × GA | Rank | Epoch | Unsloth 時間 | Units |
|-----|--------|-----------|------|-------|-------------|-------|
| A100 | 100 筆 | 8 × 2 | 32 | 3 | **~2 min** | < 1 |
| A100 | 500 筆 | 8 × 2 | 32 | 3 | **~5–8 min** | ~1 |
| A100 | 1000 筆 | 8 × 2 | 32 | 5 | **~15–25 min** | ~3 |
| A100 | 5000 筆 | 8 × 2 | 32 | 3 | **~1–1.5 hr** | ~10 |

**300 units 可跑 ~30+ 次完整實驗**（比原生 HF 多一倍，因為 Unsloth 速度快）。

### 8.6 推理測試（訓練完馬上試）

```python
# Cell 10: Quick inference test
FastLanguageModel.for_inference(model)

test_messages = [
    {"role": "system", "content": "你是 NTU 校園能源助理。"},
    {"role": "user", "content": "保健中心的年度用電量是多少？"},
]

inputs = tokenizer.apply_chat_template(
    test_messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

outputs = model.generate(inputs, max_new_tokens=256, temperature=0.3)
response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
print(f"回答: {response}")
```

---

## 9. 合併 + 匯出 GGUF（Unsloth 一鍵）

這是 Unsloth 最大的殺手鐧——**不需要自己編譯 llama.cpp、不需要手動轉檔**。
一行搞定合併 + GGUF + Q4_K_M 量化。

### 9.1 合併 LoRA → 完整模型

```python
# Cell 11: Merge LoRA into base model
MERGED_DIR = "./gemma-merged"

model.save_pretrained_merged(
    MERGED_DIR,
    tokenizer,
    save_method="merged_16bit",   # 合併為 bf16 完整精度
)

print(f"Merged model saved to {MERGED_DIR}")
```

### 9.2 直接匯出 GGUF（Q4_K_M）

```python
# Cell 12: Export to GGUF Q4_K_M — 一行搞定！
model.save_pretrained_gguf(
    MERGED_DIR,
    tokenizer,
    quantization_method="Q4_K_M",     # 與你現有模型一致
)

# GGUF 檔案會產生在 MERGED_DIR 底下
import glob
gguf_files = glob.glob(f"{MERGED_DIR}/*.gguf")
print(f"GGUF files: {gguf_files}")
```

### 9.3 下載 GGUF

```python
# Cell 13: Download GGUF
from google.colab import files
for f in gguf_files:
    files.download(f)
```

或透過 VSCode 檔案總管直接拖下來。

### 9.4 視覺能力：搭配原版 mmproj

微調只動語言層，**mmproj 不需要重新訓練**。部署時直接用原版：

```
llama-server.exe \
    -m gemma-4-e2b-it-energy-lora-Q4_K_M.gguf \   # ← 微調後的語言模型
    --mmproj mmproj-gemma-4-E2B-it-BF16.gguf \     # ← 原版 mmproj，不動！
    --port 8088
```

你的 `local_gemma_runtime.py` 已經有 mmproj 支援（`PACKAGED_GEMMA_MMPROJ_PATH`），
只要確保 `mmproj-gemma-4-E2B-it-BF16.gguf` 還在原位就好。

**視覺使用場景**：
- 上傳電費單 PDF 截圖 → mmproj 解讀 → 語言模型回答
- 上傳用電折線圖 → mmproj 看懂趨勢 → 語言模型做分析
- 上傳建築外觀照 → mmproj 識別 → 語言模型查該棟用電

### 9.4 其他量化格式（選用）

```python
# 也匯出其他格式供比較
for quant in ["Q5_K_M", "Q8_0", "F16"]:
    model.save_pretrained_gguf(
        f"./gemma-{quant}",
        tokenizer,
        quantization_method=quant,
    )
```

| 格式 | 大小（E2B） | 品質 | 適用場景 |
|------|-----------|------|---------|
| Q4_K_M | ~1.5 GB | 好 | **日常使用（推薦）** |
| Q5_K_M | ~1.8 GB | 更好 | 精度優先 |
| Q8_0 | ~2.5 GB | 接近原版 | 最高精度 |
| F16 | ~4.5 GB | 原版 | 不量化 |

### 9.5 （選用）上傳到 HuggingFace

```python
from huggingface_hub import HfApi

HF_REPO = "your-username/gemma-4-e2b-it-energy-lora"
api = HfApi()
api.create_repo(repo_id=HF_REPO, exist_ok=True)
api.upload_folder(folder_path=MERGED_DIR, repo_id=HF_REPO, repo_type="model")
print(f"Uploaded to https://huggingface.co/{HF_REPO}")
```

---

## 10. 部署回本機 demo

### 10.1 放置 GGUF 檔案

把下載的 `gemma-4-e2b-it-energy-lora-Q4_K_M.gguf` 放到以下任一路徑：

```
# 優先順序 1：demo runtime（打包部署）
D:\idf優化\demo\runtime\gemma\models\gemma-4-e2b-it-energy-lora-Q4_K_M.gguf

# 優先順序 2：外部 LM Studio 路徑
D:\AI\LMStudio\models\lmstudio-community\gemma-4-E2B-it-GGUF\gemma-4-e2b-it-energy-lora-Q4_K_M.gguf
```

> **mmproj 不用換**：原版 `mmproj-gemma-4-E2B-it-BF16.gguf` 留在原位即可。
> `start_local_gemma.ps1` 會自動偵測並掛載。

### 10.2 更新環境變數

編輯 `config/energy_startup_local.cmd`（或直接設環境變數）：

```cmd
set ENERGY_GEMMA_MODEL_PATH=D:\idf優化\demo\runtime\gemma\models\gemma-4-e2b-it-energy-lora-Q4_K_M.gguf
set ENERGY_LOCAL_LLM_MODEL=gemma-4-e2b-it-energy-lora-Q4_K_M.gguf
```

### 10.3 重啟 demo

```powershell
# PowerShell
cd D:\idf優化\demo
.\scripts\start_local_gemma.ps1
```

或直接：

```powershell
.\open_demo.cmd
```

### 10.4 驗證

1. 開啟儀表板 → 選一棟建築 → 輸入測試問題
2. 觀察回答是否更精準（數字引用、格式、節能建議品質）
3. 檢查 `lm_studio_client.py` 的日誌確認模型載入正確

---

## 11. 評測與迭代（W&B 實驗追蹤）

### 11.1 消融實驗設計

用 W&B 追蹤四組實驗，找出哪組資料組合效果最好：

| 實驗 | 訓練資料 | 目標 |
|------|---------|------|
| **base-gemma** | 不微調，只跑現有 GGUF | 建立最低基準線 |
| **v02-router-only** | `router_sft.jsonl` | 工具選擇與 JSON 格式 |
| **v02-router-strict** | `harness_v02_train.jsonl` | routing + safety，JSON-only tool call |
| **v03-hard-negative** | v02 + 新 hard negatives | PIVD/OpenBSE/anomaly 抗混淆 |

每次實驗只需改 `EXPERIMENT_NAME` 和訓練資料檔案，W&B 自動疊曲線。

### 11.2 自動評測（本機 tool routing accuracy）

微調完部署回本機後，用 `scripts/evaluate_tool_routing.py` 跑評測：

```bash
python scripts/evaluate_tool_routing.py \
    --eval-set tools/harness_v02/harness_v02_val.jsonl \
    --base-url http://127.0.0.1:8088/v1 \
    --model gemma-4-e2b-it-energy-lora-Q4_K_M.gguf \
    --output dev_artifacts/eval/gemma_lora_harness_v02_val.jsonl \
    --verbose
```

`base-gemma` 可先用目前的 `gemma-4-E2B-it-Q4_K_M.gguf` 跑一次。若正確率很低，代表微調還沒部署或模型仍不懂 router JSON schema，這是正常的 baseline，不代表資料庫失敗。

### 11.3 W&B 結果對照

在 W&B dashboard 上對照四組實驗：

```
base-gemma:        tool accuracy = ??%
v02-router-only:   tool accuracy = ??%
v02-router-strict: tool accuracy = ??%
v03-hard-negative: tool accuracy = ??%
```

看 eval loss 曲線：
- **持續下降** → 資料品質好，繼續加
- **震盪不降** → learning rate 太高或資料有噪音
- **先降後升** → overfitting，減 epoch 或加資料

### 11.4 迭代循環

```
harness_v02 → Unsloth 訓練 (~5 min)
    → GGUF 一鍵匯出 → 部署本機 → evaluate_tool_routing
    → W&B 對照 → 分析失敗案例 → 補 hard negatives → 再訓練
```

每次完整迭代只需 **~15–20 min**（A100 + Unsloth）。

---

## 12. 常見問題

### Q1: Unsloth 安裝失敗

```python
# 確保 Colab runtime 有重啟
import os
os.kill(os.getpid(), 9)  # 強制重啟 runtime
# 然後重新安裝
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --no-deps
!pip install trl peft accelerate bitsandbytes
```

### Q2: `save_pretrained_gguf` 報錯

確認 Unsloth 版本 ≥ 2024.11：
```python
import unsloth
print(unsloth.__version__)
```
如果太舊：`!pip install --upgrade "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"`

### Q3: Gemma tokenizer 沒有 pad_token

Unsloth 的 `FastLanguageModel` 已自動處理，不需要手動設。

### Q4: 本機 llama-server 啟動後回答沒變

- 確認 `ENERGY_GEMMA_MODEL_PATH` 指向新的 GGUF 檔案
- 確認舊的 llama-server process 已關閉（檢查 port 8088）
- 查看 `start_local_gemma.ps1` 的日誌確認載入的模型路徑

### Q5: 微調後模型反而變差（災難性遺忘）

- 降低 `learning_rate` 到 `5e-5`
- 降低 `num_train_epochs` 到 1–2
- 增加通用中文問答樣本（避免只灌能源領域資料）
- 看 W&B 的 eval loss 曲線，如果上升就停

### Q6: W&B 沒有出現 loss 曲線

- 確認 `report_to="wandb"`（不是 `"none"`）
- 確認 `wandb.login()` 有成功
- Colab 有時要等第一個 `logging_steps` 才會出現

### Q7: 視覺能力會被微調破壞嗎？

不會，只要你遵守兩個規則：
1. **LoRA target_modules 只放語言層**（`q_proj`, `k_proj` 等），不放 vision encoder 層
2. **mmproj 用原版**，不重新訓練

§7 和 §8 的腳本已內建「vision leak 檢查」，會自動警告。
如果不小心把 vision 層也加進 target，最壞情況是 mmproj 解讀品質下降。
解法：把 `mmproj` 相關層從 `target_modules` 移除，重新訓練。

### Q8: 微調後視覺推理（電費單/圖表）品質變差？

可能是語言層的 system prompt / 回答格式被微調覆蓋了。
解法：
- 在訓練資料中加入 **10–20 筆視覺相關 Q&A**（「這張圖表顯示什麼趨勢？」類型）
- 或在 inference 時把 system prompt 寫清楚「如果使用者附帶圖片，先描述圖片內容再回答」

### Q9: 可以不用 Colab 嗎？

可以，只要有 NVIDIA GPU（≥ 8 GB VRAM）的本機：
```bash
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
python train_lora.py  # 把上面 cell 整合成一支 script
```

---

## 附錄 A：完整 Colab 筆記本結構

| Cell | 內容 | Unsloth A100 時間 |
|------|------|---------|
| 1 | 安裝 Unsloth + W&B | 1–2 min |
| 2 | HF + W&B 登入 | < 1 min |
| 3 | 上傳 harness v0.2 train/val/manifest | 1–3 min |
| 4 | 檢查資料庫 manifest 與 split | < 1 min |
| 5 | 載入模型（Unsloth） | 1–2 min |
| 6 | LoRA config | < 1 min |
| 7 | 載入 dataset | < 1 min |
| 8 | 訓練 | **2–8 min** |
| 9 | 推理測試 | < 1 min |
| 10 | 合併 LoRA | < 1 min |
| 11 | **一鍵 GGUF Q4_K_M** | < 1 min |
| 12 | 下載 GGUF | 1–3 min |

**Unsloth A100 總計**：約 **5–15 min**（含 router-strict 335 筆資料訓練）
**Units 消耗**：每次完整跑完約 **1–2 units**

---

## 附錄 B：T4 備用方案（Unsloth + QLoRA）

如果 A100 額度用完或暫時不可用，降級到免費 T4：

```python
# T4 備用：Unsloth + 4-bit QLoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,          # T4 開 4-bit
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,                       # T4 用 16（不是 32）
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    use_gradient_checkpointing="unsloth",
)
```

T4 預估時間：500 筆 × 3 epoch ≈ 20–30 min（Unsloth 加速後）。
GGUF 匯出一樣是一行：`model.save_pretrained_gguf(...)`。
