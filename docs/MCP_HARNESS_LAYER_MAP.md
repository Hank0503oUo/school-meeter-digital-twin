# MCP and HARNESS Layer Map

This document identifies which files make up the reusable MCP framework and HARNESS memory layer, and which files are local runtime data that should not be uploaded to GitHub.

## Keep In Git

MCP framework:

- `mcp_server.py`: top-level MCP entry point.
- `run_mcp_server.cmd`: Windows launcher for the MCP server.
- `src/knowledge_mcp_server.py`: FastMCP tool registration and server construction.
- `src/knowledge_mcp_backend.py`: knowledge-search backend used by MCP tools.
- `src/demo_mcp_server.py`: demo assistant tool server helpers.
- `src/algorithm_mcp_backend.py`: algorithm backend connected to MCP tools.
- `config/mcp_review_profiles.example.json`: safe example profile config.

HARNESS memory layer:

- `src/harness_memory.py`: event memory, procedure memory, keyword extraction, retrieval, and template rebinding.
- `docs/HARNESS_FIRST_MEMORY_IMPLEMENTATION_SPEC.md`: design spec for HARNESS-first memory.
- `tools/harness_v02/*.py`: router/evaluation dataset builders and schema references.
- `tests/knowledge/`, `tests/test_local_memory_config.py`, `tests/test_knowledge_startup.py`: regression tests for the MCP and memory path.

## Keep Local

Do not commit `data/knowledge_workbench/`.

Do not commit generated HARNESS/LoRA datasets or reports, including `data/lora/`, generated SFT JSONL files, and generated `tools/harness_v02/*.jsonl` / audit / manifest files.

That directory is runtime state, not framework source. It may contain:

- parsed CSV summaries,
- generated ontology and memory indexes,
- session logs,
- curated traces,
- graph/wiki exports,
- building-specific memory notes,
- HARNESS event and procedure JSONL logs.

Because those artifacts can be derived from private meter or reviewer data, they should stay local or be shared only after a separate sanitization pass.

## Identification Rule

Use this split when preparing GitHub:

- If a file defines tools, schemas, launchers, tests, or memory logic, keep it.
- If a file records a specific run, session, parsed source document, building facts, numeric summaries, generated training rows, audit outputs, or generated index state, keep it out of Git.

The GitHub repository should explain how the system works. It should not be the storage location for private meter readings or derived memory state.

## NTU Data Boundary

Do not commit NTU meter readings or NTU-derived outputs, including:

- raw meter CSVs,
- metadata UID/loop maps,
- meter-building maps,
- per-building kWh/mean-kW/EUI summaries,
- anomaly reports,
- trained local model artifacts derived from NTU meter data,
- generated energy GeoJSON/year-cache files,
- parsed workbench documents or chunks generated from those sources.

The code may still reference local paths so the demo can run on an authorized machine. The referenced data files themselves must remain outside Git.
