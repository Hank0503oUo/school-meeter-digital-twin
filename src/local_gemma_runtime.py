from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import subprocess
import time
import json
import urllib.error
import urllib.request


_ROOT = Path(__file__).resolve().parent.parent

PACKAGED_LLAMA_SERVER_EXE = _ROOT / "runtime" / "gemma" / "bin" / "llama-server.exe"
PACKAGED_GEMMA_MODEL_PATH = _ROOT / "runtime" / "gemma" / "models" / "gemma4-e2b-it-router-v04-Q4_K_M.gguf"
PACKAGED_GEMMA_MMPROJ_PATH = _ROOT / "runtime" / "gemma" / "models" / "mmproj-gemma-4-E2B-it-BF16.gguf"
PACKAGED_LORA_DIR = _ROOT / "runtime" / "gemma" / "lora"

EXTERNAL_LLAMA_SERVER_EXE = Path(r"C:\Users\User\Documents\llama-cpp-turboquant\build-cpu\bin\Release\llama-server.exe")
EXTERNAL_GEMMA_MODEL_PATH = Path(
    r"D:\AI\LMStudio\models\lmstudio-community\gemma-4-E2B-it-GGUF\gemma4-e2b-it-router-v04-Q4_K_M.gguf"
)
EXTERNAL_GEMMA_MMPROJ_PATH = Path(
    r"D:\AI\LMStudio\models\lmstudio-community\gemma-4-E2B-it-GGUF\mmproj-gemma-4-E2B-it-BF16.gguf"
)

# Backward-compatible export for older tests/imports.
DEFAULT_GEMMA_MODEL_PATH = str(PACKAGED_GEMMA_MODEL_PATH)


@dataclass(frozen=True, slots=True)
class LocalGemmaConfig:
    model_path: Path
    server_exe: Path | None
    mmproj_path: Path | None = None
    lora_path: Path | None = None
    host: str = "127.0.0.1"
    port: int = 8088
    ctx_size: int = 131072
    gpu_layers: int | None = None
    startup_timeout_seconds: float = 20.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


def _optional_int(raw: str | None) -> int | None:
    value = str(raw or "").strip()
    return int(value) if value else None


def _resolve_lora_path(env: os._Environ[str] | dict[str, str] | None = None) -> Path | None:
    source = env if env is not None else os.environ
    raw = str(source.get("ENERGY_GEMMA_LORA_PATH", "")).strip()
    if raw:
        p = Path(raw)
        if p.is_file():
            return p
    for candidate in sorted(PACKAGED_LORA_DIR.glob("*.gguf")) if PACKAGED_LORA_DIR.is_dir() else []:
        if candidate.is_file():
            return candidate
    return None


def _resolve_packaged_or_external(env_value: str | None, packaged: Path, fallback: Path) -> Path:
    raw = str(env_value or "").strip()
    if raw:
        return Path(raw)
    if packaged.is_file():
        return packaged
    if fallback.is_file():
        return fallback
    return packaged


def resolve_local_gemma_config(env: os._Environ[str] | dict[str, str] | None = None) -> LocalGemmaConfig:
    source = env if env is not None else os.environ
    return LocalGemmaConfig(
        model_path=_resolve_packaged_or_external(
            source.get("ENERGY_GEMMA_MODEL_PATH"),
            PACKAGED_GEMMA_MODEL_PATH,
            EXTERNAL_GEMMA_MODEL_PATH,
        ),
        server_exe=_resolve_packaged_or_external(
            source.get("ENERGY_LLAMA_SERVER_EXE"),
            PACKAGED_LLAMA_SERVER_EXE,
            EXTERNAL_LLAMA_SERVER_EXE,
        ),
        mmproj_path=_resolve_packaged_or_external(
            source.get("ENERGY_GEMMA_MMPROJ_PATH"),
            PACKAGED_GEMMA_MMPROJ_PATH,
            EXTERNAL_GEMMA_MMPROJ_PATH,
        ),
        lora_path=_resolve_lora_path(source),
        host=str(source.get("ENERGY_GEMMA_HOST", "127.0.0.1")).strip() or "127.0.0.1",
        port=int(str(source.get("ENERGY_GEMMA_PORT", "8088")).strip() or "8088"),
        ctx_size=int(str(source.get("ENERGY_GEMMA_CTX", "4096")).strip() or "4096"),
        gpu_layers=_optional_int(source.get("ENERGY_GEMMA_GPU_LAYERS")),
        startup_timeout_seconds=float(str(source.get("ENERGY_GEMMA_STARTUP_TIMEOUT_SECONDS", "20")).strip() or "20"),
    )


def gemma_autostart_enabled(env: os._Environ[str] | dict[str, str] | None = None) -> bool:
    source = env if env is not None else os.environ
    provider = str(source.get("ENERGY_LOCAL_LLM_PROVIDER", "gemma")).strip().lower()
    raw = str(source.get("ENERGY_GEMMA_AUTOSTART", "")).strip().lower()
    if provider in {"lmstudio", "lm_studio", "openai_compatible", "custom"}:
        return raw in {"1", "true", "yes", "on"}
    return provider in {"", "gemma", "local_gemma"} or raw in {"1", "true", "yes", "on"}


def is_server_healthy(config: LocalGemmaConfig) -> bool:
    try:
        with urllib.request.urlopen(f"{config.base_url}/models", timeout=2.0) as response:
            return 200 <= int(response.status) < 300
    except (OSError, urllib.error.URLError, ValueError):
        return False


def served_model_ids(config: LocalGemmaConfig) -> list[str]:
    try:
        with urllib.request.urlopen(f"{config.base_url}/models", timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError):
        return []
    models = payload.get("data", []) if isinstance(payload, dict) else []
    if not isinstance(models, list):
        return []
    return [str(item.get("id", "")).strip() for item in models if isinstance(item, dict) and item.get("id")]


def is_expected_model_served(config: LocalGemmaConfig) -> bool:
    expected = config.model_path.name
    return expected in served_model_ids(config)


def build_llama_server_command(config: LocalGemmaConfig) -> list[str]:
    if config.server_exe is None:
        raise RuntimeError("ENERGY_LLAMA_SERVER_EXE is required when the Gemma server is not already running.")

    command = [
        str(config.server_exe),
        "-m",
        str(config.model_path),
        "--host",
        config.host,
        "--port",
        str(config.port),
        "-c",
        str(config.ctx_size),
    ]
    if config.gpu_layers is not None:
        command.extend(["--n-gpu-layers", str(config.gpu_layers)])
    if config.mmproj_path is not None and config.mmproj_path.is_file():
        command.extend(["--mmproj", str(config.mmproj_path)])
    if config.lora_path is not None and config.lora_path.is_file():
        command.extend(["--lora", str(config.lora_path)])
    return command


def start_local_gemma_server(config: LocalGemmaConfig) -> subprocess.Popen[bytes] | None:
    if is_server_healthy(config):
        if is_expected_model_served(config):
            return None
        raise RuntimeError(
            f"Gemma server at {config.base_url} is already running but is not serving {config.model_path.name}. "
            f"Current models: {', '.join(served_model_ids(config)) or '(unknown)'}. "
            "Stop the stale llama-server.exe process and retry."
        )

    if config.server_exe is None:
        raise RuntimeError("ENERGY_LLAMA_SERVER_EXE is required when the Gemma server is not already running.")
    if not config.server_exe.is_file():
        raise FileNotFoundError(f"llama.cpp server executable not found: {config.server_exe}")
    if not config.model_path.is_file():
        raise FileNotFoundError(f"Gemma GGUF model not found: {config.model_path}")

    startupinfo = None
    creationflags = 0
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    process = subprocess.Popen(
        build_llama_server_command(config),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        startupinfo=startupinfo,
        creationflags=creationflags,
    )

    deadline = time.monotonic() + max(0.0, config.startup_timeout_seconds)
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Gemma server exited early with code {process.returncode}.")
        if is_server_healthy(config):
            return process
        time.sleep(0.5)

    raise TimeoutError(f"Gemma server did not become healthy within {config.startup_timeout_seconds:g} seconds.")
