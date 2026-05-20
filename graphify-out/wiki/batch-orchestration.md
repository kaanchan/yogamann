# Batch Orchestration

**27 nodes · Community 1 · Cohesion 0.12**

`src/compare_vlm.py`

---

## What it does

Runs all 5 VLM models against a candidate set of runs in a **model-major loop**: load model A → annotate all images → evict → load model B → annotate all → evict → … This is the primary tool for large comparison runs (hundreds to thousands of images). It is resumable — if interrupted, it picks up where it left off using the DB as state.

## Why model-major?

Consumer GPU VRAM (12–16 GB) cannot hold two models simultaneously. Loading and evicting once per model rather than once per image dramatically reduces load time overhead. This is documented in ADR-004 and the design was informed by the WDDM VRAM constraints discovered in Issue #34 research.

## Key functions

| Function | Role |
|----------|------|
| `main()` | CLI entry point. Parses args, iterates models, calls `_run_model_phase()` for each. |
| `_run_model_phase(model_id, candidates)` | Annotates up to `max_images` runs for one model in micro-batches. Handles cooldown between batches. |
| `_fetch_run_candidates()` | Queries the DB for runs that need annotation. Respects `--slice N` for partial runs. |
| `_count_completed(model_id)` | How many runs this model has already annotated in this pass. |
| `_compute_slice_size()` | Chooses how many images to send per micro-batch based on available VRAM headroom. |
| `_historical_avg_latency(model_id)` | Reads prior annotation timings from DB to estimate ETA. |
| `_fmt_eta(seconds)` | Formats ETA as `HH:MM:SS` or `Xm Ys`. |
| `_ts()` | Compact timestamp for log lines. |

## CLI flags

- `--model MODEL_ID` — run a single model only
- `--slice N` — limit to N images per model (useful for smoke tests)
- `--prompt-key KEY` — use a named prompt variant instead of `active_prompt`
- `--dry-run` — plan without executing

## Resumability

Uses `get_unanalyzed_runs(model_id)` from the DB to find only runs without an existing `vlm_annotations` row for that model. Re-running the script is safe — it skips already-annotated runs.

## Connects to

- [VLM Inference Core](vlm-inference-core.md) — calls `annotate_batch()` per model
- [Batch Utilities & Model-Major Design](batch-utilities.md) — uses `chunked()`, `format_summary()`
- [Database — Run & Annotation Queries](database-queries.md) — queries run candidates, writes annotations
- [Batch Lock (Process Safety)](batch-lock.md) — acquires GPU lock before starting, releases on exit
- [GPU Stability & Memory Management](gpu-stability.md) — uses `gpu_monitor.cooldown_if_hot()`
- [Architecture Decisions & ADRs](architecture-decisions.md) — ADR-004 (model-major design)
