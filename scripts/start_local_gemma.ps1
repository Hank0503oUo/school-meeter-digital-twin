$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

if (-not $env:ENERGY_LLAMA_SERVER_EXE) {
    $packagedExe = Join-Path $repoRoot "runtime\gemma\bin\llama-server.exe"
    $cpuExe = "C:\Users\User\Documents\llama-cpp-turboquant\build-cpu\bin\Release\llama-server.exe"
    $vulkanExe = "C:\Users\User\.docker\bin\inference\llama-server.exe"
    if (Test-Path -LiteralPath $packagedExe -PathType Leaf) {
        $env:ENERGY_LLAMA_SERVER_EXE = $packagedExe
    } elseif (Test-Path -LiteralPath $cpuExe -PathType Leaf) {
        $env:ENERGY_LLAMA_SERVER_EXE = $cpuExe
    } else {
        $env:ENERGY_LLAMA_SERVER_EXE = $vulkanExe
    }
}
if (-not $env:ENERGY_GEMMA_GPU_LAYERS) {
    $cpuExe = "C:\Users\User\Documents\llama-cpp-turboquant\build-cpu\bin\Release\llama-server.exe"
    if ($env:ENERGY_LLAMA_SERVER_EXE -ne $cpuExe) {
        $env:ENERGY_GEMMA_GPU_LAYERS = "99"
    }
}
if (-not $env:ENERGY_GEMMA_MODEL_PATH) {
    $packagedModel = Join-Path $repoRoot "runtime\gemma\models\gemma4-e2b-it-router-v04-Q4_K_M.gguf"
    if (Test-Path -LiteralPath $packagedModel -PathType Leaf) {
        $env:ENERGY_GEMMA_MODEL_PATH = $packagedModel
    } else {
        $env:ENERGY_GEMMA_MODEL_PATH = "D:\AI\energy_models\gemma4-e2b-it-router-v04-Q4_K_M.gguf"
    }
}
if (-not $env:ENERGY_GEMMA_MMPROJ_PATH) {
    $packagedMmproj = Join-Path $repoRoot "runtime\gemma\models\mmproj-gemma-4-E2B-it-BF16.gguf"
    $fallbackMmproj = "D:\AI\LMStudio\models\lmstudio-community\gemma-4-E2B-it-GGUF\mmproj-gemma-4-E2B-it-BF16.gguf"
    if (Test-Path -LiteralPath $packagedMmproj -PathType Leaf) {
        $env:ENERGY_GEMMA_MMPROJ_PATH = $packagedMmproj
    } elseif (Test-Path -LiteralPath $fallbackMmproj -PathType Leaf) {
        $env:ENERGY_GEMMA_MMPROJ_PATH = $fallbackMmproj
    } else {
        Write-Warning "Gemma mmproj not found; starting in text-only mode."
    }
}
if (-not $env:ENERGY_GEMMA_CTX) {
    $env:ENERGY_GEMMA_CTX = "131072"
}
if (-not $env:ENERGY_GEMMA_PORT) {
    $env:ENERGY_GEMMA_PORT = "8088"
}
if (-not $env:ENERGY_GEMMA_STARTUP_TIMEOUT_SECONDS) {
    $env:ENERGY_GEMMA_STARTUP_TIMEOUT_SECONDS = "240"
}

$env:ENERGY_LOCAL_LLM_PROVIDER = "gemma"
$env:ENERGY_GEMMA_AUTOSTART = "1"
if (-not $env:ENERGY_LOCAL_LLM_MODEL) {
    $env:ENERGY_LOCAL_LLM_MODEL = "gemma4-e2b-it-router-v04-Q4_K_M.gguf"
}

if (-not (Test-Path -LiteralPath $env:ENERGY_LLAMA_SERVER_EXE -PathType Leaf)) {
    throw "llama-server.exe not found: $env:ENERGY_LLAMA_SERVER_EXE"
}
if (-not (Test-Path -LiteralPath $env:ENERGY_GEMMA_MODEL_PATH -PathType Leaf)) {
    throw "Gemma GGUF model not found: $env:ENERGY_GEMMA_MODEL_PATH"
}

@'
from src.local_gemma_runtime import resolve_local_gemma_config, start_local_gemma_server, is_server_healthy, served_model_ids

config = resolve_local_gemma_config()
process = start_local_gemma_server(config)
print(f"[gemma] base_url={config.base_url}")
print(f"[gemma] model={config.model_path}")
print(f"[gemma] server={config.server_exe}")
if config.mmproj_path is not None and config.mmproj_path.is_file():
    print(f"[gemma] mmproj={config.mmproj_path}")
else:
    print("[gemma] text_only=true")
print(f"[gemma] ctx={config.ctx_size}")
print(f"[gemma] healthy={is_server_healthy(config)}")
print(f"[gemma] served_models={served_model_ids(config)}")
if process is not None:
    print(f"[gemma] pid={process.pid}")
else:
    print("[gemma] already_running=true")
'@ | python -
