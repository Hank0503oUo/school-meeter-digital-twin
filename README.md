# School Meeter Digital Twin

Campus energy digital twin demo for school metering, building-level inference, counterfactual analysis, and knowledge workbench workflows.

## What is included

- Panel dashboard entry point: `app.py`
- Knowledge Workbench entry point: `app_workbench.py`
- Core application code: `src/`
- Campus configuration and reusable datasets: `campuses/`, `config/`, `models/`
- Documentation and project notes: `docs/`, `PROJECT_STRUCTURE.md`
- Tests: `tests/`

## MCP and HARNESS layer map

The MCP framework and HARNESS memory layer are included as source code and docs, not as private runtime memory dumps.

- MCP entry point: `mcp_server.py`
- Windows MCP launcher: `run_mcp_server.cmd`
- MCP tool server implementation: `src/knowledge_mcp_server.py`
- Knowledge backend used by MCP tools: `src/knowledge_mcp_backend.py`
- Demo/local MCP server helpers: `src/demo_mcp_server.py`
- HARNESS long-term memory implementation: `src/harness_memory.py`
- HARNESS training/evaluation builders and schema references: `tools/harness_v02/`
- Design spec: `docs/HARNESS_FIRST_MEMORY_IMPLEMENTATION_SPEC.md`
- MCP/knowledge tests: `tests/knowledge/`, `tests/test_local_memory_config.py`, `tests/test_knowledge_startup.py`

Runtime memory and derived workbench indexes under `data/knowledge_workbench/` are intentionally excluded from Git. That folder can contain parsed CSV summaries, ontology indexes, session logs, curated traces, and building memory notes generated from local data. Keep it local or share it only through an explicitly permissioned artifact folder after sanitization.

NTU meter readings and NTU-derived energy artifacts are also excluded from Git. This includes raw meter CSVs, metadata maps, per-building energy summaries, anomaly reports, trained local model outputs, and generated energy GeoJSON/cache files. Those artifacts should stay local or be shared through a permissioned artifact folder after review.

Generated HARNESS/LoRA datasets are treated the same way: the repository keeps the builders, schemas, and tests, while generated JSONL examples, audit reports, routing traces, and response-authoring datasets stay local. These generated files can contain tool-returned kWh/EUI values, so they are not part of the GitHub package.

## Local artifacts

Large generated and local-only artifacts are intentionally excluded from Git:

- `runtime/`, including local GGUF model files
- `outputs/`
- `data/cache/`
- `data/knowledge_workbench/`
- `data/lora/`, generated HARNESS/LoRA JSONL files, and generated SFT datasets
- `campuses/ntu/data/`, `campuses/ntu/models/`, and `data/NTU/`
- NTU-derived model summaries under `models/`
- `results_v11_cross_year/`
- `temp_v11_cross_year/`
- `dev_artifacts/`

The excluded runtime model currently includes a multi-GB `.gguf` file, which is not suitable for a normal GitHub repository.

## Quick start

```bash
pip install -r requirements.txt
python -m panel serve app.py --show --port 5006 --autoreload
```

Knowledge Workbench:

```bash
python -m panel serve app_workbench.py --show --port 5007 --autoreload
```

See `PROJECT_STRUCTURE.md` for the full project map and launch commands.
