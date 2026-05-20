# Review Gallery UI

**18 nodes · Community 6 · Cohesion 0.13**

`src/gallery.py`

---

## What it does

A Streamlit application that lets you review VLM annotations, compare model agreement, and trigger new inference runs — all from a browser. This is the human-facing side of the annotation pipeline.

## Key components

| Function | Role |
|----------|------|
| `_render_vlm_table()` | Paginated table of runs with VLM ratings per model. Colour-coded by rating (`good` / `acceptable` / `poor`). |
| `_render_agreement_matrix()` | Shows which model pairs agree or disagree on rating. Useful for spotting systematic model biases. |
| `_human_badge_html()` | Renders a small HTML badge showing the human annotation for a run, if one exists. |
| `_vlm_badge_html()` | Same for VLM annotations — rating + model name. |
| `_launch_vlm_inference()` | Spawns a `compare_vlm.py` subprocess for a specific (model, run) pair. Streams its stdout into the modal. |
| `_inference_terminal_dialog()` | Modal dialog showing live stdout from the inference subprocess. |
| `get_thumbnail()` | Loads a thumbnail for a gallery card from the DB (calls `get_or_create_thumbnail()`). |

## Usage

```
streamlit run src/gallery.py
```

Opens on `localhost:8501`. Requires `yogamann.db` to exist with ingested runs.

## Connects to

- [Database — Run & Annotation Queries](database-queries.md) — `get_vlm_comparison_page()`, `get_all_annotations()`
- [DB Thumbnail & Test Fixtures](db-thumbnail-fixtures.md) — `get_or_create_thumbnail()`
- [Batch Orchestration](batch-orchestration.md) — spawns `compare_vlm.py` as a subprocess
