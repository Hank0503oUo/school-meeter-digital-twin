# Colab script: first-round router-strict LoRA for the NTU energy assistant.
#
# Recommended runtime:
#   - Google Colab Pro
#   - A100 if available
#   - GPU runtime enabled
#
# Required Google Drive files:
#   MyDrive/energy_lora_router_v02/data/harness_v02_train.jsonl
#   MyDrive/energy_lora_router_v02/data/harness_v02_val.jsonl
#   MyDrive/energy_lora_router_v02/data/harness_v02_smoke.jsonl
#   MyDrive/energy_lora_router_v02/data/harness_v02_manifest.json
#
# Run:
#   python /content/drive/MyDrive/energy_lora_router_v02/colab_train_router_strict_lora.py
#
# Notes:
#   - This first round is JSON-only tool routing + safety refusal.
#   - Do not mix explainer_sft.jsonl or legacy_cleaned.jsonl into this run.

from __future__ import annotations

import inspect
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


# =============================================================================
# 0. User config
# =============================================================================

DRIVE_PROJECT_DIR = Path(os.getenv("DRIVE_PROJECT_DIR", "/content/drive/MyDrive/energy_lora_router_v02"))
DATA_DIR = DRIVE_PROJECT_DIR / "data"
OUTPUT_DIR = DRIVE_PROJECT_DIR / "outputs" / "gemma_router_strict_v02"

TRAIN_FILE = DATA_DIR / "harness_v02_train.jsonl"
VAL_FILE = DATA_DIR / "harness_v02_val.jsonl"
SMOKE_FILE = DATA_DIR / "harness_v02_smoke.jsonl"
MANIFEST_FILE = DATA_DIR / "harness_v02_manifest.json"

# Change MODEL_ID here if Hugging Face uses a different final Gemma 4 E2B repo id.
MODEL_ID = os.getenv("MODEL_ID", "google/gemma-4-e2b-it")

EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "gemma-e2b-energy-router-strict-v02")
GGUF_BASENAME = os.getenv("GGUF_BASENAME", "gemma-4-e2b-it-energy-router-v02-Q4_K_M.gguf")

MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "false").lower() in {"1", "true", "yes"}
LORA_R = int(os.getenv("LORA_R", "32"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", str(LORA_R * 2)))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))

NUM_TRAIN_EPOCHS = float(os.getenv("NUM_TRAIN_EPOCHS", "3"))
TRAIN_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "8"))
GRAD_ACCUM_STEPS = int(os.getenv("GRAD_ACCUM_STEPS", "2"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-4"))

INSTALL_DEPS = os.getenv("INSTALL_DEPS", "true").lower() in {"1", "true", "yes"}
USE_WANDB = os.getenv("USE_WANDB", "true").lower() in {"1", "true", "yes"}
RUN_FULL_VAL_EVAL = os.getenv("RUN_FULL_VAL_EVAL", "true").lower() in {"1", "true", "yes"}
EXPORT_GGUF = os.getenv("EXPORT_GGUF", "true").lower() in {"1", "true", "yes"}


# =============================================================================
# 1. Environment setup
# =============================================================================

def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)


def maybe_mount_drive() -> None:
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        print("google.colab not detected; assuming Drive is already mounted or using local paths.")
        return
    drive.mount("/content/drive")


def maybe_install_deps() -> None:
    if not INSTALL_DEPS:
        return
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-U",
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
    ])
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-U",
        "--no-deps",
        "trl",
        "peft",
        "accelerate",
        "bitsandbytes",
    ])
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-U",
        "datasets",
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
        "wandb",
    ])
    print("Dependencies installed. If Colab asks for a runtime restart, restart and rerun with INSTALL_DEPS=false.")


def login_services() -> None:
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        from huggingface_hub import login

        login(token=hf_token)
    else:
        print("No HF_TOKEN env var found. If the model is gated, run huggingface_hub.notebook_login() manually.")

    if USE_WANDB:
        wandb_key = os.getenv("WANDB_API_KEY")
        if wandb_key:
            import wandb

            wandb.login(key=wandb_key)
        else:
            print("USE_WANDB=true but WANDB_API_KEY is not set. The trainer may prompt or fall back.")


# =============================================================================
# 2. Data validation
# =============================================================================

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            assert "messages" in item, f"{path}:{line_no} missing messages"
            roles = [m.get("role") for m in item["messages"]]
            assert roles[:2] == ["system", "user"], f"{path}:{line_no} expected system,user first; got {roles}"
            assert roles[-1] == "assistant", f"{path}:{line_no} expected assistant target last; got {roles}"
            rows.append(item)
    return rows


def parse_assistant_json(item: dict[str, Any]) -> dict[str, Any]:
    target = item["messages"][-1]["content"].strip()
    if target.startswith("```"):
        target = target.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(target)


def validate_drive_files() -> None:
    required = [TRAIN_FILE, VAL_FILE, SMOKE_FILE, MANIFEST_FILE]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required Drive files:\n" + "\n".join(missing))

    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    if not str(manifest.get("profile", "")).startswith("router_strict"):
        raise ValueError(f"Expected manifest profile to start with router_strict, got {manifest.get('profile')!r}")

    train_rows = read_jsonl(TRAIN_FILE)
    val_rows = read_jsonl(VAL_FILE)
    smoke_rows = read_jsonl(SMOKE_FILE)

    for split_name, rows in [("train", train_rows), ("val", val_rows), ("smoke", smoke_rows)]:
        for idx, item in enumerate(rows, 1):
            parsed = parse_assistant_json(item)
            if parsed.get("tool") != item.get("expected_tool"):
                raise ValueError(
                    f"{split_name}:{idx} expected_tool mismatch: "
                    f"{item.get('expected_tool')} != assistant tool {parsed.get('tool')}"
                )

    print("Data OK")
    print("manifest:", manifest["version"], manifest["profile"], manifest["total"])
    print("train:", len(train_rows), Counter(x.get("category", "unknown") for x in train_rows))
    print("val:", len(val_rows), Counter(x.get("difficulty", "unknown") for x in val_rows))
    print("smoke:", len(smoke_rows))


# =============================================================================
# 3. Model + tokenizer
# =============================================================================

def load_model_and_tokenizer():
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )

    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Official Unsloth guidance is to render messages through a chat template
    # before feeding SFTTrainer. If the tokenizer has no template, use Gemma.
    try:
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "ping"}, {"role": "assistant", "content": "{}"}],
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        from unsloth.chat_templates import get_chat_template

        tokenizer = get_chat_template(tokenizer, chat_template="gemma")

    return model, tokenizer


def attach_lora(model):
    from unsloth import FastLanguageModel

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    leaked = [n for n in trainable_names if any(k in n.lower() for k in ("vision", "image", "mmproj", "visual"))]
    if leaked:
        raise RuntimeError(f"Vision/mmproj params leaked into LoRA targets: {leaked[:10]}")
    return model


# =============================================================================
# 4. Dataset rendering
# =============================================================================

def build_datasets(tokenizer):
    from datasets import load_dataset

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_FILE),
            "validation": str(VAL_FILE),
            "smoke": str(SMOKE_FILE),
        },
    )

    def render_batch(batch: dict[str, Any]) -> dict[str, list[str]]:
        texts = []
        for messages in batch["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    remove_cols = dataset["train"].column_names
    rendered = dataset.map(render_batch, batched=True, remove_columns=remove_cols)

    print("Rendered dataset:", rendered)
    print("First rendered training sample:")
    print(rendered["train"][0]["text"][:1200])
    return rendered


# =============================================================================
# 5. Training
# =============================================================================

def build_sft_config():
    from trl import SFTConfig
    from unsloth import is_bfloat16_supported

    params = inspect.signature(SFTConfig.__init__).parameters
    kwargs: dict[str, Any] = {
        "output_dir": str(OUTPUT_DIR / "checkpoints"),
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "logging_steps": 5,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "report_to": "wandb" if USE_WANDB else "none",
        "run_name": EXPERIMENT_NAME,
        "bf16": bool(is_bfloat16_supported()),
        "fp16": not bool(is_bfloat16_supported()),
    }
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "epoch"
    if "max_seq_length" in params:
        kwargs["max_seq_length"] = MAX_SEQ_LENGTH
    if "packing" in params:
        kwargs["packing"] = False
    if "dataset_text_field" in params:
        kwargs["dataset_text_field"] = "text"

    return SFTConfig(**kwargs)


def train(model, tokenizer, rendered_dataset):
    from trl import SFTTrainer

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = build_sft_config()

    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": rendered_dataset["train"],
        "eval_dataset": rendered_dataset["validation"],
    }
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    if "dataset_text_field" in trainer_params:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in trainer_params:
        trainer_kwargs["max_seq_length"] = MAX_SEQ_LENGTH
    if "packing" in trainer_params:
        trainer_kwargs["packing"] = False

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()

    adapter_dir = OUTPUT_DIR / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print("Adapter saved to:", adapter_dir)
    return model


# =============================================================================
# 6. In-memory smoke/val evaluation
# =============================================================================

def parse_tool_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {"tool": "__parse_error__", "arguments": {}, "raw": text[:300]}


def generate_tool_call(model, tokenizer, messages: list[dict[str, str]]) -> str:
    import torch

    prompt_messages = [m for m in messages if m["role"] in {"system", "user"}]
    inputs = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            inputs,
            max_new_tokens=160,
            temperature=0.0,
            do_sample=False,
            use_cache=True,
        )
    return tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True).strip()


def evaluate_split(model, tokenizer, path: Path, output_name: str, max_samples: int | None = None) -> dict[str, Any]:
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    rows = read_jsonl(path)
    if max_samples:
        rows = rows[:max_samples]

    results = []
    correct = 0
    malformed = 0
    by_diff: dict[str, dict[str, int]] = {}

    for idx, item in enumerate(rows, 1):
        raw = generate_tool_call(model, tokenizer, item["messages"])
        parsed = parse_tool_response(raw)
        predicted = parsed.get("tool", "__parse_error__")
        expected = item.get("expected_tool")
        difficulty = item.get("difficulty", "unknown")

        if predicted == "__parse_error__":
            malformed += 1
        if predicted == expected:
            correct += 1

        bucket = by_diff.setdefault(difficulty, {"total": 0, "correct": 0, "malformed": 0})
        bucket["total"] += 1
        bucket["correct"] += int(predicted == expected)
        bucket["malformed"] += int(predicted == "__parse_error__")

        results.append({
            "idx": idx,
            "sample_id": item.get("sample_id"),
            "difficulty": difficulty,
            "query": next((m["content"] for m in item["messages"] if m["role"] == "user"), ""),
            "expected_tool": expected,
            "predicted_tool": predicted,
            "is_correct": predicted == expected,
            "raw": raw,
        })

        mark = "OK" if predicted == expected else "MISS"
        print(f"[{mark}] {idx}/{len(rows)} expected={expected} predicted={predicted} diff={difficulty}")

    report = {
        "total": len(rows),
        "correct": correct,
        "accuracy": correct / max(1, len(rows)),
        "malformed": malformed,
        "malformed_rate": malformed / max(1, len(rows)),
        "by_difficulty": by_diff,
    }

    eval_dir = OUTPUT_DIR / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = eval_dir / f"{output_name}.jsonl"
    out_json = eval_dir / f"{output_name}_summary.json"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Eval report:", json.dumps(report, ensure_ascii=False, indent=2))
    print("Saved:", out_jsonl)
    return report


# =============================================================================
# 7. Merge + GGUF export
# =============================================================================

def export_outputs(model, tokenizer) -> None:
    merged_dir = OUTPUT_DIR / "merged_16bit"
    gguf_dir = OUTPUT_DIR / "gguf_q4_k_m"

    print("Saving merged 16-bit model to:", merged_dir)
    model.save_pretrained_merged(
        str(merged_dir),
        tokenizer,
        save_method="merged_16bit",
    )

    if not EXPORT_GGUF:
        return

    print("Exporting GGUF Q4_K_M to:", gguf_dir)
    model.save_pretrained_gguf(
        str(gguf_dir),
        tokenizer,
        quantization_method="Q4_K_M",
    )

    ggufs = sorted(gguf_dir.rglob("*.gguf"))
    if not ggufs:
        print("No GGUF file found after export. Check Unsloth logs.")
        return

    named_dir = OUTPUT_DIR / "final_gguf"
    named_dir.mkdir(parents=True, exist_ok=True)
    named_path = named_dir / GGUF_BASENAME
    shutil.copy2(ggufs[0], named_path)
    print("Final named GGUF:", named_path)


# =============================================================================
# 8. Main
# =============================================================================

def main() -> None:
    maybe_mount_drive()
    maybe_install_deps()
    login_services()
    validate_drive_files()

    model, tokenizer = load_model_and_tokenizer()
    rendered_dataset = build_datasets(tokenizer)
    model = attach_lora(model)
    model = train(model, tokenizer, rendered_dataset)

    print("Running smoke evaluation...")
    evaluate_split(model, tokenizer, SMOKE_FILE, "smoke_after_train")

    if RUN_FULL_VAL_EVAL:
        print("Running full validation evaluation...")
        evaluate_split(model, tokenizer, VAL_FILE, "val_after_train")

    export_outputs(model, tokenizer)
    print("Done. Outputs are under:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
