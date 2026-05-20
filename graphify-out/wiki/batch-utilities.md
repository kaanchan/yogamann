# Batch Utilities & Model-Major Design

**15 nodes · Community 11 · Cohesion 0.19**

`src/_batch_utils.py` · `tests/test_batch_utils.py` · ADR-004

---

## What it does

Pure Python helpers for VLM batch orchestration — no torch, no DB, no model loading. These utilities are intentionally dependency-free so they can be tested without a GPU.

## Functions

| Function | Role |
|----------|------|
| `chunked(items, size)` | Splits a list into sublists of at most `size` items. Handles empty input and non-divisible lengths. |
| `format_summary(model_id, stats)` | Formats a per-model batch-run summary as an aligned table string for log output. |

## Tests

`test_batch_utils.py` covers `chunked()` exhaustively:
- `test_chunked_empty()` — empty input returns empty list
- `test_chunked_exact_divisor()` — 10 items / size 5 = two equal chunks
- Additional cases for remainder, size=1, size > len

## ADR-004

This community also holds ADR-004 (Model-Major Batch Orchestration with DB-Driven Resumability), which explains *why* `compare_vlm.py` uses the model-major loop. The `chunked()` function is the micro-batching primitive that `_run_model_phase()` uses to split a model's workload into VRAM-safe units.

## Connects to

- [Batch Orchestration](batch-orchestration.md) — `compare_vlm.py` imports `chunked()` and `format_summary()`
- [Architecture Decisions & ADRs](architecture-decisions.md) — ADR-004
