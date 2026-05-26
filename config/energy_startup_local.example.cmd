@echo off
REM 複製此檔為同目錄下的 energy_startup_local.cmd 後編輯路徑。
REM open_demo.cmd 會在啟動前自動 call 該檔（若存在）。
REM 不要將含本機路徑／密鑰的 energy_startup_local.cmd 提交到 git。

REM ---------- Command Code / OpenCode online adapter（選用）----------
REM 預設使用 OpenCode Go OpenAI-compatible endpoint:
REM   https://opencode.ai/zen/go/v1/chat/completions
REM 實際密鑰建議用 Windows 使用者環境變數設定，不要寫進檔案：
REM   setx COMMAND_CODE_API_KEY "your_key"
REM   setx OPENCODE_API_KEY "your_key"
REM set "COMMAND_CODE_BASE_URL=https://opencode.ai/zen/go/v1"
REM set "COMMAND_CODE_MODEL=deepseek-v4-pro"
REM set "COMMAND_CODE_API_FORMAT=openai_chat"
REM 少數 Anthropic-style /messages 模型可改成：
REM set "COMMAND_CODE_API_FORMAT=anthropic_messages"
REM set "COMMAND_CODE_ENDPOINT_PATH=/messages"

REM ---------- Graphify 整份更新（選用）----------
REM set "ENERGY_GRAPHIFY_ON_START=1"
REM set "ENERGY_GRAPHIFY_CWD=D:\path\to\your\corpus_or_repo"
REM 範例：呼叫你自己的全量腳本；需能在 cmd 下一行跑完（必要時用 cmd /c "..."）
REM set "ENERGY_GRAPHIFY_CMD=cmd /c D:\tools\my_graphify_full.cmd"

REM ---------- 知識 review / MCP 橋接（選用）----------
REM 方式 A：啟動前選單（複製 mcp_review_profiles.example.json → mcp_review_profiles.json 並編輯）
REM set "ENERGY_MCP_MENU=1"
REM 方式 B：固定一條指令（不開選單時）
REM set "ENERGY_KNOWLEDGE_REVIEW_ON_START=1"
REM set "ENERGY_KNOWLEDGE_REVIEW_CWD=%~dp0.."
REM set "ENERGY_KNOWLEDGE_REVIEW_CMD=cmd /c D:\tools\mcp_knowledge_review.cmd"

REM 執行順序：先 graphify 再 review（預設）
REM set "ENERGY_STARTUP_HOOK_ORDER=graphify,review"
REM 任一步失敗就中止、不啟動 Panel：
REM set "ENERGY_STARTUP_STRICT=1"
