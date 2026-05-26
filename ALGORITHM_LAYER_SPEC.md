# Algorithm Orchestration Layer — Implementation Spec

> **目標讀者**：負責施工的工程師（GPT）  
> **目的**：在現有 demo 的 MCP server 上加入演算法推理層，並讓 OpenNekaise agent 能透過 Slack 呼叫這些工具。

---

## 1. 背景與現況

### 現有架構

```
demo/
├── mcp_server.py                  ← 入口，只有一行：from src.knowledge_mcp_server import main
├── src/
│   ├── knowledge_mcp_server.py    ← FastMCP server 定義（目前有 5 個 knowledge 工具）
│   ├── knowledge_mcp_backend.py   ← 工具的實作邏輯
│   ├── real_inference_engine.py   ← PIVDEngine 主體（四層架構）
│   ├── counterfactual.py          ← CounterfactualResult 計算
│   └── ...
```

### 現有 MCP 工具（已實作）

| Tool | 功能 |
|------|------|
| `search_docs` | 搜尋知識庫文件 chunk |
| `fetch_chunk` | 取單一 chunk |
| `lookup_building_entity` | 讀取建物本體資料 |
| `query_meter_or_kpi` | 讀取 CSV 電表/KPI 摘要 |
| `run_analysis` | 呼叫 cloud-first LLM 分析 |
| `save_curated_trace` | 儲存審核後的結果 |
| `ask_gemini_inference` | 直接問 Gemini |

### MCP server 目前是 stdio 模式

FastMCP 預設 stdio，需要改成 SSE HTTP 模式才能讓 OpenNekaise（Docker container）連進來。

---

## 2. 要新增的功能

### 2.1 兩個新 MCP 工具

#### Tool A：`run_pvid`

呼叫 `PIVDEngine`，回傳結構化預測結果。

**Input schema：**

```python
building_uid: str          # 建物 UID，例如 "AT1040"。空字串代表 campus-level
hours: int = 24            # 預測時數（1–168）
t_out_series: list[float]  # 逐時室外溫度 °C，長度必須等於 hours
humidity_series: list[float]  # 逐時相對濕度 %，長度必須等於 hours
start_time: str = ""       # ISO 8601，例如 "2024-07-01T00:00:00"。空字串用 now
```

**Output schema：**

```json
{
  "algo": "pvid",
  "status": "ok",
  "building_uid": "AT1040",
  "hours": 24,
  "result": {
    "timestamps": ["2024-07-01T00:00:00", "..."],
    "physics_pred": [120.3, 118.7, "..."],
    "residual_pred": [5.2, 4.8, "..."],
    "residual_std": [2.1, 2.3, "..."],
    "total_pred": [125.5, 123.5, "..."],
    "building_rank_index": [88.4, 87.1, "..."],   // 只在 building_uid 非空時存在
    "building_eui_index": [0.42, 0.41, "..."]     // 只在有面積資料時存在
  },
  "summary": {
    "mean_total_pred_kw": 124.3,
    "peak_total_pred_kw": 145.6,
    "mean_residual_std": 2.2,
    "uncertainty_pct": 1.8
  },
  "provenance": {
    "model_version": "pivd-v12",
    "engine_layers": ["PhysicsSurrogate", "V9WeightReconstructor", "BuildingMetadataScaler", "V10BootEnsemble"],
    "input_hash": "<sha256 of input params>",
    "runtime_ms": 0
  }
}
```

**Error 回傳：**

```json
{
  "algo": "pvid",
  "status": "error",
  "error": "Engine not initialized: missing boot ensemble file",
  "building_uid": "AT1040"
}
```

---

#### Tool B：`correlate_algorithms`

接收多個演算法的結果，推理跨演算法的關聯性。目前版本用規則型推理，未來可替換成 LLM 推理。

**Input schema：**

```python
results: list[dict]   # 每個 element 是任意演算法的 output dict（含 "algo" 欄位）
question: str = ""    # 選填：使用者的具體問題，幫助聚焦推理
building_uid: str = ""
```

**Output schema：**

```json
{
  "status": "ok",
  "building_uid": "AT1040",
  "algos_used": ["pvid", "counterfactual"],
  "relationships": [
    {
      "finding": "PVID 預測峰值 145.6 kW 出現在 14:00，counterfactual 顯示冷卻調降 2°C 可減少 8.4%",
      "confidence": 0.82,
      "type": "cooling_peak_correlation"
    }
  ],
  "dominant_factor": "cooling_load",
  "recommended_action": "優先檢查 14:00–16:00 的冷卻控制序列",
  "confidence": 0.78,
  "reasoning_method": "rule_based_v1"
}
```

---

### 2.2 MCP server 改成 SSE HTTP 模式

讓 OpenNekaise Docker container 可以透過 `http://host.docker.internal:8765` 連進來。

FastMCP 支援兩種啟動方式，需要讓 `mcp_server.py` 支援環境變數切換：

```python
# mcp_server.py 改法
import os
from src.knowledge_mcp_server import build_server

if __name__ == "__main__":
    server = build_server()
    transport = os.getenv("MCP_TRANSPORT", "stdio")   # stdio（預設）或 sse
    port = int(os.getenv("MCP_PORT", "8765"))
    
    if transport == "sse":
        server.run(transport="sse", host="0.0.0.0", port=port)
    else:
        server.run()   # stdio，維持現有行為
```

啟動 SSE 模式：

```bash
set MCP_TRANSPORT=sse
set MCP_PORT=8765
python mcp_server.py
```

---

## 3. 實作位置與步驟

### Step 1 — 新增 `AlgorithmMCPBackend`

建立 `src/algorithm_mcp_backend.py`，包含：

```python
class AlgorithmMCPBackend:
    def __init__(self):
        self._engine: PIVDEngine | None = None   # lazy init

    def _get_engine(self) -> PIVDEngine:
        if self._engine is None:
            self._engine = PIVDEngine.from_defaults()
        return self._engine

    def run_pvid(self, building_uid, hours, t_out_series, humidity_series, start_time) -> dict:
        # 1. 組裝 weather_df（DatetimeIndex + t_out + humidity）
        # 2. 呼叫 engine.predict() 或 engine.predict_building()（依 building_uid）
        # 3. 計算 summary stats
        # 4. 回傳標準 schema dict

    def correlate_algorithms(self, results, question, building_uid) -> dict:
        # 規則型推理
        # 未來可改成呼叫 LocalLLMAdapter 做 LLM 推理
```

### Step 2 — 在 `knowledge_mcp_server.py` 加入新工具

在 `build_server()` 裡加：

```python
from src.algorithm_mcp_backend import AlgorithmMCPBackend
algo_backend = AlgorithmMCPBackend()

@server.tool(description="Run PI-VD four-layer inference engine for building energy prediction.")
def run_pvid(
    building_uid: str = "",
    hours: int = 24,
    t_out_series: list[float] = [],
    humidity_series: list[float] = [],
    start_time: str = "",
) -> dict:
    return algo_backend.run_pvid(building_uid, hours, t_out_series, humidity_series, start_time)

@server.tool(description="Correlate results from multiple algorithms and reason about cross-algorithm relationships.")
def correlate_algorithms(
    results: list[dict],
    question: str = "",
    building_uid: str = "",
) -> dict:
    return algo_backend.correlate_algorithms(results, question, building_uid)
```

### Step 3 — 修改 `mcp_server.py` 支援 SSE 模式

見 2.2 節。

---

## 4. OpenNekaise 端設定

### 修改檔案：`container/agent-runner/src/index.ts`

找到 `mcpServers:` 的區塊，加入第二個 server：

```typescript
mcpServers: {
  opennekaise: {
    command: 'node',
    args: [mcpServerPath],
    env: {
      OPENNEKAISE_CHAT_JID: containerInput.chatJid,
      OPENNEKAISE_GROUP_FOLDER: containerInput.groupFolder,
      OPENNEKAISE_IS_MAIN: containerInput.isMain ? '1' : '0',
    },
  },
  "building-energy-knowledge": {
    url: process.env.BUILDING_ENERGY_MCP_URL ?? "http://host.docker.internal:8765/sse",
  },
},
```

在 `allowedTools` 加入：

```typescript
allowedTools: [
  // ... 現有 tools ...
  'mcp__opennekaise__*',
  'mcp__building-energy-knowledge__*',   // ← 加這行
],
```

### 修改 OpenNekaise `.env`

```
BUILDING_ENERGY_MCP_URL=http://host.docker.internal:8765/sse
```

### 修改 `groups/global/CLAUDE.md`

在適當位置加入一節：

```markdown
## Building Energy Tools

You have access to `mcp__building-energy-knowledge__*` tools for building energy analysis:

- `run_pvid` — Run physics-informed energy prediction (PI-VD engine). Requires `t_out_series` and `humidity_series` arrays.
- `correlate_algorithms` — Reason about relationships between multiple algorithm outputs.
- `run_analysis` — General Q&A against the knowledge base with LLM reasoning.
- `lookup_building_entity` — Get building metadata, meters, KPIs.
- `query_meter_or_kpi` — Get CSV energy meter summaries.
**Agent role**: You call these tools, receive structured JSON, and translate results into natural language for the user. Do not perform numerical calculations yourself — always delegate to the tools.

## When to escalate to external reasoning

You have access to `mcp__claude-reasoning__*` (Claude) and `mcp__codex-reasoning__*` (Codex) for complex reasoning escalation.

Call these when:
- The question requires correlating 3+ data sources simultaneously
- You need to reason about causality between algorithm results
- You are uncertain about your answer after using building energy tools
- The question requires deep multi-step inference beyond your confidence

Do NOT escalate for:
- Simple data lookups → use `search_docs` or `query_meter_or_kpi`
- Algorithm execution → use `run_pvid`, `correlate_algorithms`
- Straightforward NL translation of tool results

Prefer `mcp__claude-reasoning__*` over Codex for open-ended reasoning. Use Codex for code or structured output generation.
```

---

## 5. 新增 Tool C：`ask_cloud_model`

讓 9B 主 agent 在推理遇到瓶頸時，主動呼叫雲端大模型取得推理結果，再由 9B 組裝最終回答。

### 設計原則

- 9B 永遠是主 agent，雲端只是它的一個工具
- 9B 自己判斷何時需要升級（不是系統強制 fallback）
- 工具回傳雲端的推理文字，9B 負責整合進最終回答

### Tool 定義（加入 `knowledge_mcp_server.py`）

```python
@server.tool(
    description=(
        "Escalate a complex reasoning question to a cloud LLM. "
        "Call this when the question requires multi-step inference, "
        "cross-domain correlation, or you are uncertain about the answer. "
        "Returns the cloud model's reasoning as a string."
    )
)
def ask_cloud_model(
    question: str,
    context_json: str = "",   # 把相關演算法結果或文件摘要序列化成 JSON 字串
    reason: str = "",         # 升級原因：complex_reasoning / low_confidence / multi_step
) -> dict:
    return backend.ask_cloud_model(question, context_json, reason)
```

### Backend 實作（加入 `knowledge_mcp_backend.py`）

```python
def ask_cloud_model(self, question: str, context_json: str, reason: str) -> dict:
    from src.knowledge_analysis import CloudModelAdapter, CloudAdapterError
    adapter = CloudModelAdapter()
    if not adapter.configured():
        return {
            "status": "unavailable",
            "answer": "",
            "reason": "Cloud model not configured. Set GEMINI_API_KEY or ENERGY_LLM_API_KEY."
        }
    try:
        answer = adapter.ask_direct(question=question, context=context_json)
        return {
            "status": "ok",
            "answer": answer,
            "escalation_reason": reason,
            "model": adapter.model,
        }
    except CloudAdapterError as e:
        return {"status": "error", "answer": "", "error": str(e)}
```

### 設計變更：升級推理走外部 MCP server，不走 Python SDK

**不在 demo Python 程式碼裡直接呼叫 Claude / Codex API。**

改由 OpenNekaise agent 直接持有 Claude MCP server 和 Codex MCP server 的工具，9B agent 在推理不足時主動呼叫這些外部 MCP 工具。

```
9B Agent
  └── 推不動 → mcp__claude__ask(question, context)
             或 mcp__codex__complete(prompt)
                  ↓
             外部 MCP server（Claude / Codex）
                  ↓
             推理結果回傳給 9B
                  ↓
             9B 整合成最終回答
```

### OpenNekaise `index.ts` 加入外部推理 MCP server

```typescript
mcpServers: {
  opennekaise: { /* 現有，不動 */ },

  "building-energy-knowledge": {
    url: process.env.BUILDING_ENERGY_MCP_URL ?? "http://host.docker.internal:8765/sse",
  },

  // Claude MCP（升級推理用）
  "claude-reasoning": {
    command: "npx",
    args: ["-y", "@anthropic-ai/claude-code", "mcp"],
    env: {
      ANTHROPIC_API_KEY: process.env.ANTHROPIC_API_KEY ?? "",
    },
  },

  // Codex MCP（升級推理用，二選一或同時掛）
  "codex-reasoning": {
    command: "npx",
    args: ["-y", "@openai/codex", "--mcp"],
    env: {
      OPENAI_API_KEY: process.env.OPENAI_API_KEY ?? "",
    },
  },
},
```

在 `allowedTools` 加入：

```typescript
allowedTools: [
  // ... 現有 ...
  'mcp__building-energy-knowledge__*',
  'mcp__claude-reasoning__*',   // Claude 推理工具
  'mcp__codex-reasoning__*',    // Codex 推理工具
],
```

### OpenNekaise `.env` 加入

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...            # 只在用 Codex 時需要
```

### `ask_cloud_model` 工具從 demo MCP server 移除

原本設計在 demo MCP server 裡加 `ask_cloud_model` 工具，**現在不需要了**。

9B agent 直接持有 `mcp__claude-reasoning__*` 和 `mcp__codex-reasoning__*`，不需要透過 demo MCP server 中轉。demo MCP server 保持純粹：**只負責建物能源資料與演算法，不處理 LLM 推理路由**。

---

## 6. 資料流總覽

### 情境 A：簡單預測（9B 自己處理）

```
Slack user：「預測 AT1040 明天的用電，室外溫 32°C」
    ↓
9B Agent
    ↓ run_pvid(building_uid="AT1040", hours=24, t_out_series=[32,...], humidity_series=[75,...])
Demo MCP server (SSE @ :8765)
    ↓ PIVDEngine.predict_building()
    ↑ { result: {...}, summary_for_agent: "峰值 145kW @14:00，不確定性 ±2%" }
9B Agent
    ↑ 「AT1040 明天預測平均用電 124 kW，峰值在下午 2 點達 145 kW。」
Slack user
```

### 情境 B：複雜推理（9B 升級給雲端）

```
Slack user：「PVID residual 突然升高，結合過去三年歷史，可能是什麼系統問題？」
    ↓
9B Agent
    ↓ run_pvid()  →  拿到結果，residual_std 異常
    ↓ search_docs("residual anomaly cause")  →  找到相關文件
    ↓ 判斷：需要跨域推理，升級
    ↓ ask_cloud_model(
          question="PVID residual_std 升至 12kW，氣象正常，可能系統原因？",
          context_json=json.dumps({pvid_result, doc_chunks}),
          reason="complex_reasoning"
      )
Demo MCP server → CloudModelAdapter.ask_direct() → Gemini
    ↑ { status: "ok", answer: "可能原因：1. 冷卻塔結垢... 2. 傳感器漂移..." }
9B Agent 整合結果
    ↑ 「根據 PVID 分析和歷史文件，residual 異常升高最可能的原因是...」
Slack user
```

---

## 7. 注意事項

1. `PIVDEngine` 初始化很慢（需載入 `.pkl` 模型），用 lazy singleton，不要每次請求都重新 init。
2. `t_out_series` 和 `humidity_series` 長度必須等於 `hours`，在 `run_pvid` 入口做 validation 並回傳清楚的 error message。
3. `correlate_algorithms` 目前用規則型推理即可，留好介面讓未來替換成 LLM 推理。
4. FastMCP SSE mode 在 Windows 上確認用 `uvicorn`，需要 `pip install uvicorn`。
5. OpenNekaise `index.ts` 改完需要重新 build container：`docker build -t opennekaise:latest ./container`。
6. `ask_cloud_model` 的後端模型透過環境變數切換，見第 5 節。

---

## 8. 完成驗收標準

- [ ] `python mcp_server.py`（stdio 模式）行為不變，現有工具全部正常
- [ ] `MCP_TRANSPORT=sse python mcp_server.py` 啟動後 `http://localhost:8765/sse` 回應正常
- [ ] `run_pvid` 工具可在 Claude Code MCP inspector 呼叫，回傳符合 schema 的 JSON
- [ ] `correlate_algorithms` 接收 `[pvid_result, counterfactual_result]`，回傳有意義的 `relationships`
- [ ] OpenNekaise container 內 `mcp__claude-reasoning__*` 工具可被 agent 呼叫
- [ ] 9B agent 在 CLAUDE.md 指引下正確判斷何時升級推理
- [ ] OpenNekaise container build 成功，`mcp__building-energy-knowledge__run_pvid` 出現在 agent 可用工具清單
