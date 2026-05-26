param(
    [string] $LlamaServerExe = "C:\Users\User\Documents\llama-cpp-turboquant\build-cpu\bin\Release\llama-server.exe",
    [string] $GemmaModelPath = "D:\idf優化\demo\runtime\gemma\models\gemma4-e2b-it-router-v04-Q4_K_M.gguf",
    [string] $GemmaMmprojPath = "D:\AI\LMStudio\models\lmstudio-community\gemma-4-E2B-it-GGUF\mmproj-gemma-4-E2B-it-BF16.gguf"
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$binDir = Join-Path $repoRoot "runtime\gemma\bin"
$modelsDir = Join-Path $repoRoot "runtime\gemma\models"

if (-not (Test-Path -LiteralPath $LlamaServerExe -PathType Leaf)) {
    throw "llama-server.exe not found: $LlamaServerExe"
}
if (-not (Test-Path -LiteralPath $GemmaModelPath -PathType Leaf)) {
    throw "Gemma model not found: $GemmaModelPath"
}

$mmprojExists = Test-Path -LiteralPath $GemmaMmprojPath -PathType Leaf
if (-not $mmprojExists) {
    Write-Warning "mmproj not found (vision optional): $GemmaMmprojPath"
}

New-Item -ItemType Directory -Path $binDir -Force | Out-Null
New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null

$serverDest = Join-Path $binDir "llama-server.exe"
$modelDest = Join-Path $modelsDir "gemma4-e2b-it-router-v04-Q4_K_M.gguf"
$mmprojDest = Join-Path $modelsDir "mmproj-gemma-4-E2B-it-BF16.gguf"

Copy-Item -LiteralPath $LlamaServerExe -Destination $serverDest -Force
Copy-Item -LiteralPath $GemmaModelPath -Destination $modelDest -Force
if ($mmprojExists) {
    Copy-Item -LiteralPath $GemmaMmprojPath -Destination $mmprojDest -Force
}

Write-Host "=== Gemma Runtime Package Summary ==="
Write-Host "Server : $serverDest"
Write-Host "Model  : $modelDest"
if ($mmprojExists) {
    Write-Host "Mmproj : $mmprojDest"
} else {
    Write-Host "Mmproj : (text-only, vision projector not available)"
}
Write-Host "=== Done ==="
