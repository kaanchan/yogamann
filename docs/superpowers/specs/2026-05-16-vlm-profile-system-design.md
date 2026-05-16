# Design: VLM Profile System — Prompt Variants, Named Profiles, Gallery Filters

**Date**: 2026-05-16
**Issues**: [#37](https://github.com/kaanchan/yogamann/issues/37) (prompt tightening), [#38](https://github.com/kaanchan/yogamann/issues/38) (MiniCPM-V structure fix)
**Branch**: feature/vlm-profile-system (new from main)

---

## Goal

Enable running the same models against the same images with different prompts, storing all variants in the DB, and comparing them side-by-side in the gallery. Maximum flexibility: ad-hoc `--prompt-tag` for one-off experiments, named profile files for reproducible experiment configs.

---

## Architecture Overview

Four layers, each independently testable:

```
profiles/vlm/<name>.yml   ←  named experiment profiles (models + prompt_tag + optional overrides)
       ↓
profiles/vlm/vlm.yml      ←  model registry + named prompts (v1, v2, ...)
       ↓
compare_vlm.py            ←  --profile / --prompt-tag CLI flags; writes prompt_tag to DB
       ↓
vlm_annotations (SQLite)  ←  (run_id, model_id, prompt_tag) unique key
       ↓
gallery.py                ←  collapsible filter panel; columns = active (model × prompt_tag) pairs
```

---

## Layer 1 — DB Schema

### Change to `vlm_annotations`

```sql
-- New column (added via ALTER TABLE in open_db DDL)
ALTER TABLE vlm_annotations ADD COLUMN prompt_tag TEXT NOT NULL DEFAULT 'v1';

-- Replace old unique index
DROP INDEX IF EXISTS idx_vlm_run_model;
CREATE UNIQUE INDEX idx_vlm_run_model_prompt
    ON vlm_annotations(run_id, model_id, prompt_tag);
```

**Migration**: `ADD COLUMN ... DEFAULT 'v1'` is idempotent when wrapped in a `try/except` inside `open_db()` (same pattern as existing column migrations). All existing rows silently receive `prompt_tag = 'v1'`. No data loss, no separate script.

### Updated `db.py` functions

**`save_vlm_annotation()`** — add `prompt_tag: str = 'v1'` parameter. Default preserves backward compatibility. `ON CONFLICT` target becomes `(run_id, model_id, prompt_tag)`.

**`get_vlm_comparison_page()`** — add `prompt_tags: list[str] | None = None` parameter. When set, filters `WHERE va.prompt_tag IN (...)`. When `None`, returns all prompt_tags (backward-compatible).

**`get_unanalyzed_runs()`** — add `prompt_tag: str = 'v1'` parameter so the runner only queues runs not yet annotated for the requested tag.

---

## Layer 2 — Config: `vlm.yml` + Profile Files

### `profiles/vlm/vlm.yml` — model registry + prompt library

The existing `prompt:` top-level key is replaced by `prompts:` (a named dict). The `models:` block is unchanged.

```yaml
prompts:
  v1:                                    # original prompt — preserved exactly
    system: |
      You are a pose-matching accuracy analyst. You receive two images:
      Image 1: a source photo of a person in a yoga pose.
      Image 2: a 3D wooden mannequin render meant to replicate that exact pose.
      Your sole task is to judge how accurately the mannequin replicates the human body position.
      Do NOT evaluate pose quality, difficulty, naturalness, or relaxation of the human.
      Do NOT comment on clothing, skin, background, or artistic style.
      Output ONLY valid JSON — no prose, no markdown, no explanation outside the JSON.
    user: |
      Does the mannequin in Image 2 match the body position shown in Image 1?
      Focus exclusively on joint angles, limb positions, and weight distribution.

      Output JSON with exactly these keys:
        rating            — "good", "acceptable", or "poor"
        misaligned        — list of specific body-part discrepancies. Empty list if none.
        unwanted_features — list of render artifacts unrelated to pose. Empty list if none.
        fail_patterns     — list of systematic render issues. Empty list if none.
        notes             — one sentence: overall match quality and single most critical discrepancy.

  v2:                                    # tighter prompt — body-part checklist, explicit exclusions
    system: |
      You are a pose-matching accuracy analyst. You receive two images:
      Image 1: a source photo of a person in a yoga pose.
      Image 2: a 3D wooden mannequin render meant to replicate that exact pose.

      Your sole task: judge how accurately the mannequin replicates the human body position.

      EXPLICITLY EXCLUDE — do not mention or consider:
        facial features, facial expression, face shape, hair
        clothing, skin tone, tattoos
        background, lighting, environment, shadows
        artistic style, realism, rendering quality
        safety, injury risk, pose difficulty or naturalness

      Output ONLY valid JSON — no prose, no markdown, no explanation outside the JSON.
    user: |
      Compare the mannequin (Image 2) to the human (Image 1).

      Check EACH body part in order and note any discrepancy:
        1. Head tilt (left/right/forward/back)
        2. Neck rotation
        3. Shoulder elevation (raised/lowered) and rotation (forward/back), left and right separately
        4. Elbow angle (bent/straight/angle degrees), left and right separately
        5. Wrist angle and hand position, left and right separately
        6. Finger spread and curl, left and right separately
        7. Torso lean (forward/back/side) and twist (rotation around spine)
        8. Hip angle and rotation, left and right separately
        9. Knee angle (bent/straight), left and right separately
        10. Ankle angle and foot position (pointed/flexed/rotated), left and right separately

      A rating of "good" requires ALL major body parts to match closely.
      A rating of "acceptable" means 1–3 minor joint divergences.
      A rating of "poor" means significant mismatch in one or more major segments.

      Output JSON with exactly these keys:
        rating            — "good", "acceptable", or "poor"
        misaligned        — list of specific body-part discrepancies (e.g. ["left elbow 30° too straight",
                            "right hip not rotated forward enough"]). Empty list [] if none.
        unwanted_features — list of render geometry artifacts (e.g. ["floating left hand",
                            "torso clipping into leg"]). Empty list [] if none.
        fail_patterns     — list of systematic render issues across this run (e.g.
                            ["torso consistently tilted right vs source"]). Empty list [] if none.
        notes             — one sentence: overall match quality and the single most critical
                            discrepancy, if any. Do not mention facial features, background, or realism.
```

### Profile files — `profiles/vlm/<name>.yml`

Each profile is a standalone YAML file defining a reproducible experiment. All fields except `prompt_tag` and `models` are optional.

**`profiles/vlm/default.yml`**:
```yaml
# Default experiment profile — all models, original prompt
prompt_tag: v1
models:
  - qwen2_5_vl_7b
  - internvl2_5_8b
  - minicpm_o_2_6
  - minicpm_v_2_6
  - molmo_7b_d
```

**`profiles/vlm/strict-v2.yml`**:
```yaml
# Stricter body-part prompt — selective model comparison
prompt_tag: v2
models:
  - qwen2_5_vl_7b
  - internvl2_5_8b
# Optional per-model parameter overrides:
model_overrides:
  qwen2_5_vl_7b:
    max_new_tokens: 512
```

### JSON Schema files

**`profiles/vlm/vlm.schema.json`** — validates `vlm.yml`:
- `prompts` required: dict of named prompt objects, each with `system` (string) and `user` (string)
- `models` required: dict of model config objects; each requires `repo` (string); `backend`, `inference_style`, `load_in_4bit`, `max_new_tokens`, `max_pixels`, `vlm_batch_size` are optional
- No additional top-level keys

**`profiles/vlm/profile.schema.json`** — validates any `profiles/vlm/*.yml` profile:
- `prompt_tag` required: string — must match a key in `vlm.yml`'s `prompts` dict
- `models` required: list of strings — each must match a key in `vlm.yml`'s `models` dict
- `model_overrides` optional: dict keyed by model name; values are partial model config objects (`max_new_tokens`, `vlm_batch_size`, `max_pixels` only — repo and backend cannot be overridden)
- No additional keys

---

## Layer 3 — CLI (`compare_vlm.py`)

### New flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--profile NAME` | str | None | Load `profiles/vlm/<NAME>.yml` for models + prompt_tag + overrides |
| `--prompt-tag TAG` | str | `'v1'` | Tag annotations with this prompt variant |

### Resolution order (highest to lowest)

1. Explicit CLI flags (`--prompt-tag`, `--models`, `--vlm-batch-size`)
2. Profile file (`--profile`)
3. Built-in defaults (`prompt_tag='v1'`, models from `vlm.yml`)

### Prompt loading

`_get_model_cfg()` in `vlm_inference.py` currently reads `cfg["prompt"]`. After this change it reads `cfg["prompts"][prompt_tag]` where `prompt_tag` is passed in from the runner. The function signature gains `prompt_tag: str = 'v1'`.

### Example invocations

```bash
# Use named profile (reproducible experiment)
uv run python src/compare_vlm.py --profile strict-v2 --limit 50

# Ad-hoc: all models with new prompt, tag as v2
uv run python src/compare_vlm.py --prompt-tag v2 --models qwen2_5_vl_7b internvl2_5_8b

# No flags — identical to current behaviour (prompt_tag='v1', all models)
uv run python src/compare_vlm.py --limit 0

# Override profile's model list
uv run python src/compare_vlm.py --profile strict-v2 --models qwen2_5_vl_7b
```

---

## Layer 4 — Gallery (`gallery.py`)

### Filter panel

Collapsible panel above the VLM comparison table. Collapsed state shows active-filter chips + "Edit filters ▼". Expanded state shows four pill groups:

| Group | Controls | Default |
|-------|----------|---------|
| **Prompt** | One pill per distinct `prompt_tag` found in DB | all selected |
| **Models** | One pill per model with annotations | all selected |
| **Human rating** | Gold / Silver / Bronze / Poor / Unrated | all selected |
| **Options** | "Disagreements only" toggle | off |

Active filter chips appear in the collapsed bar (e.g. `prompt: v2 ×`, `Qwen ×`, `Gold ×`). "Reset all" clears to defaults.

### Table columns

Columns are `(model_id, prompt_tag)` pairs, filtered to active selections. Column header format: `Model v2` (model label + prompt tag suffix when >1 prompt_tag is active). When only one prompt_tag is active, no suffix shown — table stays clean.

### `get_vlm_comparison_page()` changes

- `prompt_tags: list[str] | None` — filter rows; `None` = all
- `ref_user_id` remains unchanged
- Pivot query generates one column alias per `(model, prompt_tag)` pair

### Streamlit session state keys added

- `st.session_state.vlm_prompt_tags` — list of active prompt tags (default: all)
- `st.session_state.vlm_filter_expanded` — bool, filter panel open/closed

---

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| `--profile` file not found | `FileNotFoundError` with clear message: `"Profile 'X' not found at profiles/vlm/X.yml"` |
| Profile references unknown `prompt_tag` | `KeyError` with message listing valid prompt tags from `vlm.yml` |
| Profile references unknown model key | `KeyError` with message listing valid model keys |
| DB migration on existing column | `try/except OperationalError` — silently skip (idempotent) |
| Gallery: no annotations for selected (model, prompt_tag) | Column absent from table; no error |

---

## Testing

| Test | Location | Confirms |
|------|----------|---------|
| `test_prompt_tag_default` | `tests/test_db_vlm.py` | existing save/get still works with default tag |
| `test_prompt_tag_separate_rows` | `tests/test_db_vlm.py` | same run+model with two tags → two rows, no conflict |
| `test_prompt_tag_upsert` | `tests/test_db_vlm.py` | same run+model+tag → upserts in place |
| `test_get_vlm_comparison_page_filter` | `tests/test_db_vlm.py` | `prompt_tags=['v2']` returns only v2 rows |
| `test_profile_load_valid` | `tests/test_profile_loader.py` | valid profile YAML parsed correctly |
| `test_profile_load_missing` | `tests/test_profile_loader.py` | missing file raises `FileNotFoundError` |
| `test_profile_load_unknown_prompt` | `tests/test_profile_loader.py` | unknown prompt_tag raises `KeyError` |
| `test_cli_resolution_order` | `tests/test_compare_vlm.py` | CLI flag overrides profile value |
| Manual: gallery filter | human | prompt pill toggles change visible columns |
| Manual: run with `--profile strict-v2` | human | annotations appear with `prompt_tag='v2'` |

---

## Build Sequence

Dependencies flow top-to-bottom — implement in this order:

1. **DB** (`db.py`) — schema + migration + updated save/get functions + tests
2. **Config** (`vlm.yml` prompts block + `profiles/vlm/*.yml` files + JSON schemas)
3. **Runner** (`compare_vlm.py` + `vlm_inference.py`) — profile loader, CLI flags, prompt resolution
4. **Gallery** (`gallery.py`) — filter panel UI + updated `get_vlm_comparison_page()` call

Each step is mergeable independently if needed.
