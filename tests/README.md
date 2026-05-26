# Test Layout

The test suite is organized by responsibility so related checks stay together.

- `tests/core/`: shared domain logic, inference engines, topology, and config
- `tests/dashboard/`: Panel dashboard startup, launcher, and dashboard-only helpers
- `tests/knowledge/`: knowledge workbench, MCP backend, and local assistant behavior
- `tests/integration/`: cross-module pipeline and external-facing integration tests

## Common Commands

Run everything:

```bash
python -m pytest tests -q
```

Run only dashboard checks:

```bash
python -m pytest tests/dashboard -q
```

Run a single file:

```bash
python -m pytest tests/integration/test_demo_pipeline.py -q
```
