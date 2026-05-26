@echo off
chcp 65001 >nul
title MCP Server
cd /d "%~dp0"
echo [啟動] MCP Server...
python mcp_server.py
pause
