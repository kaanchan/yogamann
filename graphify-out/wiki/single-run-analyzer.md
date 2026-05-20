# Single-Run Analyzer

**18 nodes · Community 5 · Cohesion 0.21**

`src/analyze.py`

---

## What it does

A lightweight daemon-style script that polls the database for unannotated runs and annotates them one at a time using the currently active VLM model. Unlike `compare_vlm.py`, it runs a single model (whichever is `active_model` in `vlm.yml`) and processes runs as they appear — useful during active rendering sessions when you want annotations to accumulate in the background.

## Key functions

| Function | Role |
|----------|------|
| `main()` | Entry point. Opens DB, loops over unannotated runs, calls `_process_run()` for each. |
| `_process_run(run_id, conn)` | Loads the source and render images for a run, calls `annotate()`, writes result to DB. |
| `_handle_sigint()` | Catches Ctrl-C gracefully — finishes the current annotation before exiting. |

## Difference from batch orchestration

`analyze.py` is single-model, run-by-run. `compare_vlm.py` is multi-model, batch-optimised. Use `analyze.py` during a render session; use `compare_vlm.py` for systematic comparison passes.

## Connects to

- [VLM Inference Core](vlm-inference-core.md) — calls `annotate()`
- [Database — Run & Annotation Queries](database-queries.md) — `get_unanalyzed_runs()`, `save_vlm_annotation()`, `save_rating()`
