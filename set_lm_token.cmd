@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === LM Studio Token Setup ===
echo.
echo Paste your LM Studio token below.
echo It will be saved to USER environment variable: ENERGY_LOCAL_LLM_API_KEY
echo.
set /p INPUT_TOKEN=Token: 

if "%INPUT_TOKEN%"=="" (
  echo.
  echo [ERROR] Empty token. Nothing saved.
  pause
  exit /b 1
)

setx ENERGY_LOCAL_LLM_API_KEY "%INPUT_TOKEN%" >nul
if errorlevel 1 (
  echo.
  echo [ERROR] Failed to save token via setx.
  pause
  exit /b 1
)

echo.
echo [OK] Saved ENERGY_LOCAL_LLM_API_KEY to user environment.
echo Please close this window and open a NEW terminal before launching demo.
echo.
pause

