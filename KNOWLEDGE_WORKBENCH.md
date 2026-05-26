# Knowledge Workbench

This workbench is the deployable first version of the building energy knowledge platform.

## What It Does

- Upload PDF, Markdown, text, and CSV files into a building-scoped knowledge base
- Rebuild a local chunk index and generate OpenNekaise-style `ONTOLOGY.ttl` and `MEMORY.md`
- Run four task types:
  - `qa`
  - `structured_extraction`
  - `energy_summary`
  - `report_generation`
- Save approved outputs into `curated_traces.jsonl` for later Colab distillation
- Promote approved outputs into `MEMORY.md` when they should become long-term building knowledge

## Default Demo

The original campus twin demo remains the default experience.

```bash
cd demo
python -m panel serve app.py --show --port 5006
```

On Windows, you can also run:

```bat
open_demo.cmd
```

## Knowledge Workbench

```bash
cd demo
pip install -r requirements.txt
panel serve app_workbench.py --show --port 5007 --autoreload
```

On Windows, you can also run:

```bat
run_workbench.cmd
```

Or double-click:

```bat
open_workbench.cmd
```

## Run The MCP Server

This repo now includes a minimal MCP server backed by the same knowledge workbench.

```bash
cd demo
python mcp_server.py
```

On Windows, you can also run:

```bat
run_mcp_server.cmd
```

Available MCP tools:

- `search_docs`
- `fetch_chunk`
- `lookup_building_entity`
- `query_meter_or_kpi`
- `run_analysis`
- `save_curated_trace`

Available MCP resources:

- `knowledge://status`
- `knowledge://buildings`
- `knowledge://building/{building_id}/ontology`
- `knowledge://building/{building_id}/memory`
- `knowledge://curated-traces`

For local MCP clients, see [demo/.mcp.json](/Users/HANK/Downloads/台大教案/國科會/idf優化/demo/.mcp.json).

## Optional Cloud Model

The app works without a cloud model. In that case it uses a heuristic fallback so you can still upload data and curate traces.

To enable cloud-first inference, set:

```bash
set ENERGY_LLM_API_URL=https://your-openai-compatible-endpoint/v1/chat/completions
set ENERGY_LLM_API_KEY=your_api_key
set ENERGY_LLM_MODEL=your_model_name
```

## Storage Layout

All workbench data is stored under:

```text
demo/data/knowledge_workbench/
```

Important files:

- `groups/<building>/ONTOLOGY.ttl`
- `groups/<building>/MEMORY.md`
- `state/documents.json`
- `state/chunks.json`
- `state/curated_traces.jsonl`
- `reports/*.md`

## Architecture Notes

- The workbench is the default `src.dashboard.create_dashboard()` entrypoint.
- The original campus twin dashboard is still available as `src.dashboard.create_legacy_dashboard()`.
- Existing in-house algorithms remain part of the tool layer. The first integrated algorithm preview is the counterfactual savings estimate used during energy summary generation.
- The MCP server reuses the same backend so the web app and MCP clients read/write the same local knowledge base.
