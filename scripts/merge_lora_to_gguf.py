"""Merge PEFT LoRA adapter into base model and export to GGUF for llama-server.

This is the recommended path for Gemma 4, since llama.cpp's convert_lora_to_gguf.py
does not yet support the gemma4 architecture.

Steps:
  1. Load base model (full precision or 4-bit)
  2. Merge LoRA adapter weights into the base model
  3. Save merged model to temp directory
  4. Export merged model to GGUF using llama.cpp's convert_hf_to_gguf.py
  5. Copy GGUF to runtime/gemma/models/

After conversion, set:
    set ENERGY_GEMMA_MODEL_PATH=D:\idf優化\demo\runtime\gemma\models\gemma4-e2b-it-router-v04-Q4_K_M.gguf

Usage:
    python scripts/merge_lora_to_gguf.py
    python scripts/merge_lora_to_gguf.py --adapter-dir PATH --output-name NAME
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ADAPTER_DIR = Path(
    r"G:\我的雲端硬碟\energy_lora_router_v04\outputs\gemma_router_strict_v04\adapter"
)
DEFAULT_BASE_MODEL = "google/gemma-4-e2b-it"
DEFAULT_LLAMA_CPP_DIR = Path(
    r"C:\Users\User\Documents\llama-cpp-turboquant"
)
DEFAULT_OUTPUT_DIR = ROOT / "runtime" / "gemma" / "models"
DEFAULT_OUTPUT_NAME = "gemma4-e2b-it-router-v04-Q4_K_M.gguf"
DEFAULT_MERGE_DIR = ROOT / "temp_merged_model"


def step1_merge_lora(
    *,
    adapter_dir: Path,
    base_model: str,
    merge_dir: Path,
    load_in_4bit: bool = True,
) -> Path:
    """Merge LoRA adapter into base model and save to merge_dir."""
    print(f"Step 1: Merging LoRA adapter into base model")
    print(f"  base_model:   {base_model}")
    print(f"  adapter_dir:  {adapter_dir}")
    print(f"  merge_dir:    {merge_dir}")
    print()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    use_bnb = load_in_4bit
    if use_bnb:
        try:
            import bitsandbytes
            from transformers import BitsAndBytesConfig
        except ImportError:
            print("  bitsandbytes not available, falling back to FP16 loading")
            use_bnb = False

    bnb_config = None
    if use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

    print(f"Loading base model (4bit={use_bnb})...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto" if use_bnb else "cpu",
        torch_dtype=torch.float16,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(adapter_dir))

    print("Merging and unloading LoRA weights...")
    model = model.merge_and_unload()

    print("Dequantizing 4-bit weights to FP16...")
    import gc

    saved_config = model.config

    state_dict = {}
    for name, param in model.named_parameters():
        if hasattr(param, "quant_state"):
            try:
                dequant = param.data.dequantize()
                state_dict[name] = dequant.to(torch.float16)
            except Exception:
                state_dict[name] = param.data.to(torch.float16)
        elif param.dtype != torch.float16:
            state_dict[name] = param.data.to(torch.float16)
        else:
            state_dict[name] = param.data

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    merge_dir.mkdir(parents=True, exist_ok=True)

    import safetensors.torch
    import json as _json
    print(f"Saving merged FP16 state_dict to {merge_dir}...")
    safetensors.torch.save_file(state_dict, str(merge_dir / "model.safetensors"))

    config_dict = saved_config.to_dict() if hasattr(saved_config, 'to_dict') else saved_config
    config_dict.pop("quantization_config", None)
    (merge_dir / "config.json").write_text(
        _json.dumps(config_dict, indent=2), encoding="utf-8"
    )

    print("Saving tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(str(merge_dir))

    print(f"Step 1 complete: merged model saved to {merge_dir}")
    return merge_dir


def step2_export_gguf(
    *,
    merge_dir: Path,
    output_path: Path,
    llama_cpp_dir: Path,
    outtype: str = "q4_k_m",
) -> Path:
    """Export merged HF model to GGUF using llama.cpp's convert_hf_to_gguf.py."""
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.is_file():
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found at {convert_script}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(convert_script),
        str(merge_dir),
        "--outfile", str(output_path),
        "--outtype", outtype,
    ]

    print(f"\nStep 2: Exporting to GGUF")
    print(f"  merge_dir:  {merge_dir}")
    print(f"  output:     {output_path}")
    print(f"  outtype:    {outtype}")
    print(f"  command:    {' '.join(cmd)}")
    print()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(llama_cpp_dir) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        cmd,
        cwd=str(llama_cpp_dir),
        env=env,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"\nGGUF export FAILED with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    if output_path.is_file():
        size_gb = output_path.stat().st_size / (1024 * 1024 * 1024)
        print(f"\nStep 2 complete: GGUF exported successfully!")
        print(f"  Output: {output_path} ({size_gb:.2f} GB)")
    else:
        print(f"\nWarning: output file not found at {output_path}", file=sys.stderr)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge PEFT LoRA into base model and export GGUF"
    )
    parser.add_argument(
        "--adapter-dir",
        default=str(DEFAULT_ADAPTER_DIR),
        help="Path to adapter directory with adapter_model.safetensors",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="HuggingFace model name for base model",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="Output GGUF filename",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for GGUF",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        default=str(DEFAULT_LLAMA_CPP_DIR),
        help="Path to llama.cpp repository",
    )
    parser.add_argument(
        "--merge-dir",
        default=str(DEFAULT_MERGE_DIR),
        help="Temporary directory for merged HF model",
    )
    parser.add_argument(
        "--outtype",
        default="auto",
        choices=["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"],
        help="GGUF tensor type (default: auto=f16)",
    )
    parser.add_argument(
        "--keep-merged",
        action="store_true",
        help="Keep the merged HF model directory after GGUF export",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=False,
        help="Load base model in 4-bit (requires bitsandbytes)",
    )
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.output_dir)
    output_path = output_dir / args.output_name
    merge_dir = Path(args.merge_dir)
    llama_cpp_dir = Path(args.llama_cpp_dir)

    step1_merge_lora(
        adapter_dir=adapter_dir,
        base_model=args.base_model,
        merge_dir=merge_dir,
        load_in_4bit=args.load_in_4bit,
    )

    step2_export_gguf(
        merge_dir=merge_dir,
        output_path=output_path,
        llama_cpp_dir=llama_cpp_dir,
        outtype=args.outtype,
    )

    if not args.keep_merged and merge_dir.is_dir():
        print(f"\nCleaning up merged model: {merge_dir}")
        shutil.rmtree(merge_dir, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"Done! To use the merged model, set:")
    print(f'  set ENERGY_GEMMA_MODEL_PATH={output_path}')
    print(f'  set ENERGY_GEMMA_LORA_PATH=')
    print(f"")
    print(f"Or use vendor_gemma_runtime.ps1 to copy into the runtime dir.")


if __name__ == "__main__":
    main()
