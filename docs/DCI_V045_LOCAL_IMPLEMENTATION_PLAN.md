# DCI v0.45 本地施工規格

## 目標

本階段不是重訓 LoRA，也不是重寫 agent。目標是在現有 `Agent + MCP + harness + memory` 本體上，補一層 Direct Corpus Interaction（DCI）文件工具，讓本地 E2B / Gemma agent 可以用更高解析度的文件操作能力處理法規、SOP、OpenBSE、模型文件與知識庫問題。

現有系統已經具備：

- Agent loop: `src/lm_studio_client.py`
- MCP tools: `src/demo_mcp_server.py`, `src/knowledge_mcp_server.py`
- Knowledge backend: `src/knowledge_mcp_backend.py`, `src/knowledge_base.py`
- Wiki memory: `src/wiki_memory.py`
- Router harness: `tools/harness_v02/`

v0.45 只補本地可施工的 DCI layer，不要破壞 v0.4 router 訓練線。

## 核心原則

DCI 不等於把 unrestricted shell 開給模型。正式做法是提供安全白名單 MCP tools，底層可以用 Python / `rg` / 檔案讀取，但模型只能呼叫受控 API。

新增工具要限制在 knowledge corpus 範圍內：

- `data/knowledge_workbench/groups/**/docs`
- `data/knowledge_workbench/groups/**/parsed`
- 必要時可讀 `MEMORY.md` / `ONTOLOGY.ttl`

不得允許任意讀取整個磁碟、`.git`、環境變數、金鑰、模型檔、cache raw secrets。

## 建議新增 MCP 工具

### 1. find_docs

用途：依文件標題、doc_id、路徑、source_type、building_id 找文件。

建議 schema：

```python
def find_docs(
    query: str,
    building_id: str = "",
    source_type: str = "",
    limit: int = 20,
) -> dict:
    ...
```

回傳：

```json
{
  "status": "ok",
  "query": "...",
  "count": 3,
  "docs": [
    {
      "doc_id": "doc_xxx",
      "title": "...",
      "building_id": "general",
      "source_type": "markdown",
      "path": "...",
      "parsed_md_path": "..."
    }
  ]
}
```

### 2. grep_docs

用途：對 parsed markdown / docs 做精確字串或 regex 搜尋。適合找法規條號、OpenBSE、EUI、CV-RMSE、HVAC、meter id、building_id。

建議 schema：

```python
def grep_docs(
    pattern: str,
    building_id: str = "",
    selected_docs: list[str] | None = None,
    regex: bool = False,
    case_sensitive: bool = False,
    limit: int = 50,
) -> dict:
    ...
```

回傳 match 時要包含文件、行號、短摘錄：

```json
{
  "status": "ok",
  "pattern": "CV-RMSE",
  "count": 5,
  "matches": [
    {
      "doc_id": "doc_xxx",
      "title": "OpenBSE calibration",
      "path": ".../parsed/doc_xxx.md",
      "line": 42,
      "text": "CV-RMSE is used to evaluate..."
    }
  ],
  "truncated": false
}
```

### 3. read_doc_chunk

用途：讀指定文件的局部內容。可以支援 `doc_id`、`path`、`chunk_id`、`start_line`。

建議 schema：

```python
def read_doc_chunk(
    doc_id: str = "",
    chunk_id: str = "",
    path: str = "",
    start_line: int = 1,
    max_lines: int = 80,
) -> dict:
    ...
```

限制：

- `max_lines` 預設 80，最多 200。
- path 必須落在 knowledge corpus 允許範圍。
- 回傳需包含 `start_line` / `end_line`。

### 4. inspect_doc_context

用途：針對 match 前後文做局部檢查。這是 DCI 的關鍵工具之一，比一次塞整篇文件更適合小模型。

建議 schema：

```python
def inspect_doc_context(
    pattern: str,
    doc_id: str = "",
    path: str = "",
    before: int = 5,
    after: int = 8,
    regex: bool = False,
    case_sensitive: bool = False,
    limit: int = 10,
) -> dict:
    ...
```

回傳每個 match 的上下文行：

```json
{
  "status": "ok",
  "pattern": "EUI",
  "contexts": [
    {
      "doc_id": "doc_xxx",
      "title": "metrics.md",
      "line": 18,
      "before": ["..."],
      "match": "EUI = annual energy / area",
      "after": ["..."]
    }
  ]
}
```

### 5. count_doc_matches

用途：統計某關鍵字在文件庫中出現在哪些文件，幫 agent 先縮小範圍。

建議 schema：

```python
def count_doc_matches(
    pattern: str,
    building_id: str = "",
    regex: bool = False,
    case_sensitive: bool = False,
    limit: int = 50,
) -> dict:
    ...
```

回傳：

```json
{
  "status": "ok",
  "pattern": "OpenBSE",
  "docs": [
    {"doc_id": "doc_a", "title": "...", "count": 12},
    {"doc_id": "doc_b", "title": "...", "count": 3}
  ]
}
```

## 建議修改檔案

### `src/knowledge_base.py`

新增後端方法，供 MCP backend 呼叫：

- `find_documents(...)`
- `grep_documents(...)`
- `read_document_lines(...)`
- `inspect_document_context(...)`
- `count_document_matches(...)`

實作建議：

- 優先讀 `DocumentRecord.parsed_md_path`，因為 PDF / CSV / MD 都已轉成可搜尋 markdown。
- 對 path 做 allowlist 檢查。
- 用 Python 逐行搜尋即可，先不必引入外部 `rg`。
- regex 模式要 try/except，regex 無效時回傳 `status: error`。
- 每個工具都要有 `limit` 和 `truncated`。

### `src/knowledge_mcp_backend.py`

加薄 wrapper：

- `find_docs(...)`
- `grep_docs(...)`
- `read_doc_chunk(...)`
- `inspect_doc_context(...)`
- `count_doc_matches(...)`

不要在 backend wrapper 裡寫大量邏輯，核心留在 `KnowledgeWorkbench`。

### `src/knowledge_mcp_server.py`

把新工具註冊成 MCP tools，描述要明確區分：

- `search_docs`: 仍是高階 chunk retrieval / lexical RAG。
- `find_docs`: 找文件本身。
- `grep_docs`: 精確關鍵字搜尋。
- `read_doc_chunk`: 讀指定文件片段。
- `inspect_doc_context`: 看 match 前後文。
- `count_doc_matches`: 統計關鍵詞分布。

### `src/demo_mcp_server.py`

如果 demo agent 實際使用的是 `demo_mcp_server.py`，也要註冊同樣的 DCI tools，或確認它會轉接 `KnowledgeMCPBackend`。

目前 `lm_studio_client.py` 預設 server script 是 `src/demo_mcp_server.py`，所以這裡很重要。

### `tools/harness_v02/tool_schema_v02.py`

先不要把 DCI tools 混進 v0.4 主訓練。建議新增一個 v0.45 區塊或註解，等 harness 完成後再正式加入。

可先準備工具描述：

```python
"find_docs": {
    "desc_zh": "依文件標題、路徑、doc_id 或 building_id 找知識庫文件；不是讀內容",
    "tier": "dci",
}
"grep_docs": {
    "desc_zh": "在知識庫原始/解析文件中做精確關鍵字或 regex 搜尋；適合法規條號、OpenBSE、EUI、CV-RMSE、HVAC 等精確線索",
    "tier": "dci",
}
"read_doc_chunk": {
    "desc_zh": "讀取指定文件、chunk 或行號附近的局部內容",
    "tier": "dci",
}
"inspect_doc_context": {
    "desc_zh": "檢查某個關鍵字 match 的前後文，用於定位證據",
    "tier": "dci",
}
"count_doc_matches": {
    "desc_zh": "統計關鍵字在各文件中的出現次數，用於縮小搜尋範圍",
    "tier": "dci",
}
```

## 不建議本階段做的事

- 不要把 unrestricted bash 暴露給模型。
- 不要直接把 DCI tools 加進 v0.4 LoRA 訓練包。
- 不要刪掉 `search_docs`。
- 不要把 `fetch_chunk` 改壞；它可以繼續存在。
- 不要一次做 trajectory LoRA，v0.45 先做工具和 eval。

## Agent 路由建議

`src/lm_studio_client.py` 目前已有 `_should_prefetch_docs()`，可以先小改成：

- 如果 query 含 `法規`, `條文`, `HJPLUS`, `OpenBSE`, `CV-RMSE`, `EUI`, `HVAC`, `定義`, `在哪`, `文件`, `SOP`，優先 prefetch `grep_docs` 或 `find_docs`。
- 如果找到 1-3 個高信心 match，再 prefetch `inspect_doc_context`。
- 如果完全找不到，才用 `search_docs`。

但這個 prefetch 可放第二階段。第一階段先確保 MCP tools 可被 agent 自己 call。

## v0.45 Harness / Eval 建議

新增目錄：

```text
tools/harness_v045_dci/
```

建議資料：

- `dci_router_seed.jsonl`
- `dci_eval_set.jsonl`
- `eval_dci_tools.py`

先做 100-200 筆，不要追求量。

類別：

| 類別 | 目標工具 | 例子 |
|---|---|---|
| find docs | `find_docs` | 找 OpenBSE 校準文件在哪 |
| exact grep | `grep_docs` | 搜尋 CV-RMSE 定義 |
| local context | `inspect_doc_context` | 看 EUI 附近段落 |
| read chunk | `read_doc_chunk` | 讀 doc_xxx 第 40 行附近 |
| count matches | `count_doc_matches` | 哪些文件提到 OpenBSE 最多 |
| no evidence | `grep_docs` then conservative answer | 找不到某館校準報告 |

Eval 指標：

- `tool_accuracy`: 是否選對 DCI tool。
- `coverage`: 是否找到正確 doc_id / path。
- `localization`: 是否定位到正確 line / section / context。
- `no_evidence_safety`: 找不到時是否保守，不亂答。
- `malformed_rate`: JSON tool call 是否可解析。

## 測試要求

至少新增或更新測試：

```text
tests/knowledge/test_dci_doc_tools.py
tests/knowledge/test_knowledge_mcp_server.py
```

測試情境：

1. `find_docs("OpenBSE")` 能回傳相關文件或空結果，不能 crash。
2. `grep_docs("EUI")` 能回傳 line number。
3. `read_doc_chunk(... max_lines=20)` 不超出限制。
4. `inspect_doc_context("CV-RMSE")` 回傳 before/match/after。
5. path traversal 被拒絕，例如 `../../.env`。
6. regex 無效時回傳乾淨錯誤。
7. limit 生效，且 `truncated` 正確。

## 驗收標準

本地施工完成後，至少要能跑：

```powershell
pytest tests/knowledge/test_dci_doc_tools.py
pytest tests/knowledge/test_knowledge_mcp_server.py
python mcp_server.py
```

MCP server 啟動後，工具清單應包含：

- `find_docs`
- `grep_docs`
- `read_doc_chunk`
- `inspect_doc_context`
- `count_doc_matches`

不要求本階段 LoRA 分數提升。v0.45 的成功標準是：

1. 工具安全可用。
2. 可以對 knowledge corpus 做精確搜尋與局部讀取。
3. 回傳格式穩定，適合 agent 消化。
4. 已具備 coverage / localization eval 的資料基礎。

## 版本路線

建議版本切法：

```text
v0.4  = 修 router 準確率、安全拒答、parse error
v0.45 = 本地 DCI tools + DCI eval harness
v0.5  = DCI trajectory LoRA / hybrid RAG + DCI ablation
```

v0.45 完成後，再用真實 demo trace 與 DCI eval 結果，決定哪些 DCI tools 要加入 v0.5 LoRA 訓練。
