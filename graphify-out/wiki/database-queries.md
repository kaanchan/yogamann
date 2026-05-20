# Database — Run & Annotation Queries

**23 nodes · Community 4 · Cohesion 0.10**

`src/db.py` (high-level query layer)

---

## What it does

The upper half of the database module — functions that answer business questions: which runs need annotating, how do VLM results compare across models, what has a user rated? These are the functions called by `analyze.py`, `compare_vlm.py`, and `gallery.py`.

The lower-level plumbing (connections, thumbnails, test fixtures) lives in [DB Thumbnail & Test Fixtures](db-thumbnail-fixtures.md).

## Schema summary

The SQLite database (`yogamann.db`) tracks:
- **runs** — each render output: source image path, render image path, profile, timestamps
- **vlm_annotations** — `(run_id, model_id, prompt_tag)` unique. Stores the JSON rating dict from each model
- **human_annotations** — manual ratings from gallery users
- **users** — gallery login credentials

## Key functions

| Function | Role |
|----------|------|
| `get_unanalyzed_runs(model_id)` | Returns runs with no `vlm_annotations` row for this model. The queue for batch and single-run modes. |
| `save_vlm_annotation(run_id, model_id, result, prompt_tag)` | Writes a VLM annotation dict to the DB. |
| `get_vlm_comparison_page(page, per_page)` | Paginated query returning runs with annotations pivoted by model — powers the gallery table. |
| `count_vlm_annotations(model_id)` | How many annotations exist for a model. Used for progress reporting. |
| `get_all_annotations(run_id)` | All VLM and human annotations for a single run. |
| `save_annotation(run_id, user_id, rating)` | Writes a human annotation. |
| `ingest_json(path)` | Reads a `.metrics.json` sidecar file and upserts a run record. |
| `ingest_all(root)` | Walks `output/` for `*.metrics.json` files and ingests them all. |
| `get_or_create_user(name, password)` | Returns an existing user row or creates one. |
| `make_thumbnail(img_path, size)` | Generates a JPEG thumbnail (bytes) for gallery display. |

## Connects to

- [DB Thumbnail & Test Fixtures](db-thumbnail-fixtures.md) — shares the same file; lower-level plumbing
- [Batch Orchestration](batch-orchestration.md) — calls `get_unanalyzed_runs()`, `save_vlm_annotation()`
- [Single-Run Analyzer](single-run-analyzer.md) — same DB calls
- [Review Gallery UI](review-gallery-ui.md) — calls `get_vlm_comparison_page()`, `get_all_annotations()`
