@echo off
setlocal
cd /d "%~dp0"
echo === NTU Energy Demo Launcher ===
echo.

REM Optional local startup settings. Copy config\energy_startup_local.example.cmd
REM to config\energy_startup_local.cmd and edit it for local-only settings.
if exist "%~dp0config\energy_startup_local.cmd" (
    call "%~dp0config\energy_startup_local.cmd"
    echo Startup hooks config: config\energy_startup_local.cmd ^(loaded^)
) else (
    echo Startup hooks config: none ^(optional: config\energy_startup_local.cmd^)
)
echo.

if "%PYTHONIOENCODING%"=="" set "PYTHONIOENCODING=utf-8"
if "%PYTHONUTF8%"=="" set "PYTHONUTF8=1"

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not found in PATH.
    echo Please add Python to PATH and try again.
    pause
    exit /b 1
)

if "%ENERGY_LOCAL_LLM_BASE_URL%"=="" set "ENERGY_LOCAL_LLM_BASE_URL=http://127.0.0.1:1234/v1"
if "%ENERGY_LOCAL_LLM_API_KEY%"=="" (
    if not "%LM_API_TOKEN%"=="" set "ENERGY_LOCAL_LLM_API_KEY=%LM_API_TOKEN%"
)
if "%ENERGY_LOCAL_LLM_API_KEY%"=="" (
    if not "%LM_STUDIO_API_KEY%"=="" set "ENERGY_LOCAL_LLM_API_KEY=%LM_STUDIO_API_KEY%"
)
echo LM Studio endpoint: %ENERGY_LOCAL_LLM_BASE_URL%
if "%ENERGY_LOCAL_LLM_API_KEY%"=="" (
    echo LM Studio token: MISSING ^(set ENERGY_LOCAL_LLM_API_KEY or LM_API_TOKEN^) - local assistant will 401.
    echo Tip: run set_lm_token.cmd once to save token permanently.
) else (
    echo LM Studio token: present
)

REM Optional startup hooks.
REM set ENERGY_GRAPHIFY_ON_START=1
REM set ENERGY_GRAPHIFY_CMD=path\to\your_graphify_full.cmd
REM set ENERGY_KNOWLEDGE_REVIEW_ON_START=1
REM set ENERGY_KNOWLEDGE_REVIEW_CMD=path\to\your_mcp_knowledge_review.cmd
REM set ENERGY_STARTUP_HOOK_ORDER=graphify,review
REM set ENERGY_STARTUP_STRICT=1

set "DEMO_PORT_HINT=%ENERGY_DEMO_PORT%"
if "%DEMO_PORT_HINT%"=="" set "DEMO_PORT_HINT=%PORT%"
if "%DEMO_PORT_HINT%"=="" set "DEMO_PORT_HINT=5006"
echo Demo preferred port: %DEMO_PORT_HINT% ^(set ENERGY_DEMO_PORT or PORT to override^)
if "%ENERGY_DEMO_FALLBACK_PORTS%"=="" (
    echo Demo fallback ports: 5008,5009,5010,5011 ^(set ENERGY_DEMO_FALLBACK_PORTS to override^)
) else (
    echo Demo fallback ports: %ENERGY_DEMO_FALLBACK_PORTS%
)
echo Checking preferred port %DEMO_PORT_HINT%...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$raw='%DEMO_PORT_HINT%'; $port=0; if ([int]::TryParse($raw, [ref]$port)) { if (Get-NetTCPConnection -LocalPort $port -State Listen -EA SilentlyContinue) { Write-Host 'port in use' } else { Write-Host 'port free' } } else { Write-Host 'port value is not numeric; launcher will validate it' }"

echo.
echo Starting launcher...
python open_browser_launcher.py > launcher_log.txt 2>&1
set "EXITCODE=%errorlevel%"

echo.
if not "%EXITCODE%"=="0" (
    echo FAILED. Error details:
    echo.
    type launcher_log.txt
) else (
    echo OK - browser opened.
    echo.
    type launcher_log.txt
    echo.
    echo -----------------------------------------------
    echo  If the dashboard is BLANK inside the browser:
    echo    1. Open Task Manager
    echo    2. End all "python.exe" processes
    echo    3. Double-click open_demo.cmd again
    echo  Root cause: a stale server without WebSocket
    echo  permission was reused ^(now auto-detected^).
    echo -----------------------------------------------
)

echo.
pause
