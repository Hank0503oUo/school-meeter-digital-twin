"""Convert PEFT/PEFT LoRA adapter (safetensors) to GGUF LoRA for llama-server --lora.

Usage:
    python scripts/convert_lora_to_gguf.py
    python scripts/convert_lora_to_gguf.py --adapter-dir "G:\path\to\adapter"
    python scripts/convert_lora_to_gguf.py --adapter-dir PATH --output PATH --llama-cpp-dir PATH

This script wraps llama-cpp's convert_lora_to_gguf.py to convert a HuggingFace PEFT
LoRA adapter (adapter_model.safetensors + adapter_config.json) into a GGUF LoRA file
that llama-server can load with --lora.

After conversion, set the environment variable:
    set ENERGY_GEMMA_LORA_PATH=D:\idf優化\demo\runtime\gemma\lora\<name>.gguf

Or simply place the .gguf in runtime/gemma/lora/ and it will be auto-detected.
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
DEFAULT_LLAMA_CPP_DIR = Path(
    r"C:\Users\User\Documents\llama-cpp-turboquant"
)
DEFAULT_OUTPUT_DIR = ROOT / "runtime" / "gemma" / "lora"


def find_convert_script(llama_cpp_dir: Path) -> Path:
    candidate = llama_cpp_dir / "convert_lora_to_gguf.py"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"convert_lora_to_gguf.py not found in {llama_cpp_dir}. "
        f"Ensure llama.cpp is cloned at that path."
    )


def validate_adapter(adapter_dir: Path) -> tuple[Path, Path]:
    safetensors = adapter_dir / "adapter_model.safetensors"
    config = adapter_dir / "adapter_config.json"
    if not safetensors.is_file():
        raise FileNotFoundError(f"adapter_model.safetensors not found in {adapter_dir}")
    if not config.is_file():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")
    return safetensors, config


def convert(
    *,
    adapter_dir: Path,
    output_path: Path,
    llama_cpp_dir: Path,
    base_model: str = "",
    outtype: str = "f16",
) -> Path:
    convert_script = find_convert_script(llama_cpp_dir)
    validate_adapter(adapter_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(convert_script),
        str(adapter_dir),
        "--outfile", str(output_path),
        "--outtype", outtype,
    ]
    if base_model:
        cmd.extend(["--base-model", base_model])

    print(f"Running: {' '.join(cmd)}")
    print(f"  adapter:  {adapter_dir}")
    print(f"  output:   {output_path}")
    print(f"  outtype:  {outtype}")
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
        print(f"\nConversion FAILED with exit code {result.returncode}", file=sys.stderr)
        print("This usually means the base model architecture is not yet supported", file=sys.stderr)
        print("for LoRA-to-GGUF conversion in this version of llama.cpp.", file=sys.stderr)
        print("", file=sys.stderr)
        print("Alternative: merge the LoRA into the base model and re-export GGUF.", file=sys.stderr)
        print("See scripts/merge_lora_to_gguf.py for that path.", file=sys.stderr)
        sys.exit(result.returncode)

    if output_path.is_file():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\nConversion successful!")
        print(f"  Output: {output_path} ({size_mb:.1f} MB)")
        print(f"\nTo use, set:")
        print(f'  set ENERGY_GEMMA_LORA_PATH={output_path}')
        print(f"\nOr copy to runtime auto-detect dir:")
        print(f"  copy \"{output_path}\" \"{DEFAULT_OUTPUT_DIR}\\\"")
    else:
        print(f"\nWarning: output file not found at {output_path}", file=sys.stderr)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PEFT LoRA adapter to GGUF for llama-server --lora"
    )
    parser.add_argument(
        "--adapter-dir",
        default=str(DEFAULT_ADAPTER_DIR),
        help="Path to adapter directory containing adapter_model.safetensors",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output GGUF path (default: runtime/gemma/lora/<adapter_dir_name>.gguf)",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        default=str(DEFAULT_LLAMA_CPP_DIR),
        help="Path to llama.cpp repository",
    )
    parser.add_argument(
        "--outtype",
        default="f16",
        choices=["f32", "f16", "bf16", "q8_0", "auto"],
        help="Output tensor type (default: f16)",
    )
    parser.add_argument(
        "--base-model",
        default="",
        help="HuggingFace model name or local path for the base model (optional)",
    )
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir)
    llama_cpp_dir = Path(args.llama_cpp_dir)

    if args.output:
        output_path = Path(args.output)
    else:
        adapter_name = adapter_dir.parent.name if adapter_dir.name == "adapter" else adapter_dir.name
        output_path = DEFAULT_OUTPUT_DIR / f"{adapter_name}.gguf"

    convert(
        adapter_dir=adapter_dir,
        output_path=output_path,
        llama_cpp_dir=llama_cpp_dir,
        base_model=args.base_model,
        outtype=args.outtype,
    )


if __name__ == "__main__":
    main()
