# VLM Pose Analysis — Design Spec
Date: 2026-05-12
Status: Awaiting approval
Issue: #29

---

## Problem

Pipeline renders are currently reviewed manually. We need an autonomous agent that
can load a (source photo, mannequin render) pair, compare the poses, and write
structured feedback into the database — with no human in the loop.

---

## Models

Three Vision-Language Models evaluated and confirmed (RTX 5080, 16 GB VRAM):

| Priority | Model | HF Repo | VRAM (Q4) | Multi-image |
|---|---|---|---|---|
| 1 | Qwen2.5-VL 7B | `Qwen/Qwen2.5-VL-7B-Instruct` | ~6–7 GB | Native |
| 2 | InternVL2.5-8B | `OpenGVLab/InternVL2_5-8B-MPO` | ~7–8 GB | Native |
| 3 | MiniCPM-V 4.6 | `openbmb/MiniCPM-V-2_6` | ~3–4 GB | Native |

Fallback if any fail integration: `allenai/Molmo-7B-D-0924` (partial multi-image).

---

## Architecture

```
src/
  vlm_inference.py      ← shared inference core
  compare_vlm.py        ← comparison harness (run N models, side-by-side report)
  analyze.py            ← async runner (poll DB, annotate unanalyzed runs)

profiles/
  vlm.yml               ← active model, all model configs, prompt template

docs/
  ideal-targets/        ← reference mannequin images (evaluation grounding)
```

---

## Components

### 1. `profiles/vlm.yml` — configuration

```yaml
active_model: qwen2_5_vl_7b

models:
  qwen2_5_vl_7b:
    repo: Qwen/Qwen2.5-VL-7B-Instruct
    backend: transformers          # transformers | lmdeploy | llama_cpp
    load_in_4bit: true
    max_new_tokens: 512

  internvl2_5_8b:
    repo: OpenGVLab/InternVL2_5-8B-MPO
    backend: lmdeploy
    load_in_4bit: true
    max_new_tokens: 512

  minicpm_v_4_6:
    repo: openbmb/MiniCPM-V-2_6
    backend: transformers
    load_in_4bit: true
    max_new_tokens: 512

prompt:
  system: |
    You are a yoga pose alignment analyst. You receive two images:
    Image 1 is a source photo of a person in a yoga pose.
    Image 2 is a rendered wooden mannequin that should match that pose.
    Compare the poses and output ONLY valid JSON — no prose, no markdown.
  user: |
    Compare the pose in Image 1 (source) with Image 2 (mannequin render).
    Output JSON with exactly these keys:
      rating         — string: "good" | "acceptable" | "poor"
      misaligned     — list of strings: body parts where pose diverges
      unwanted_features — list of strings: artifacts in the render
      fail_patterns  — list of strings: systematic issues (e.g. "head always forward")
      notes          — string: one sentence of free-form observation

schema:
  rating: [good, acceptable, poor]
  required: [rating, misaligned, unwanted_features, fail_patterns, notes]
```

---

### 2. `src/vlm_inference.py` — inference core

Responsibilities:
- Load model + processor from HF hub (respects `HF_HUB_OFFLINE=1`)
- Accept `(photo_path, render_path, model_key)` → return parsed `dict`
- Build the multi-image prompt per backend conventions:
  - Transformers (Qwen): `[{"type": "image"}, {"type": "image"}, {"type": "text", "text": ...}]`
  - LMDeploy (InternVL): `Image-1: <IMG_TOKEN>\nImage-2: <IMG_TOKEN>\n{text}`
  - MiniCPM: same as Transformers pattern
- Strip any prefix tokens from raw output (`assistant:`, `<|im_start|>`, etc.)
- Parse output as JSON; on parse failure, retry once with explicit correction prompt
- Validate against schema; raise `VLMSchemaError` if required keys still missing after retry
- Return `{"model_id": str, "rating": str, "misaligned": [...], ...}`

Model loading is **lazy and cached** — load once per process, reuse across calls.
Quantization (`load_in_4bit`) handled via `BitsAndBytesConfig` for Transformers backend,
`--quantization awq` for LMDeploy.

---

### 3. `src/compare_vlm.py` — comparison harness

Responsibilities:
- Accept `--run-ids` (specific DB run IDs) or `--limit N` (N most recent unanalyzed runs)
- For each run: load `(source photo path, output render path)` from DB
- Call `vlm_inference.py` for each configured model
- Write one `vlm_annotations` row per model per run
- Print a side-by-side comparison table to stdout:

```
Run 3421  yoga-pose-sample-4.jpg
─────────────────────────────────────────────────────────
                    Qwen2.5-VL    InternVL2.5   MiniCPM-V
  rating            acceptable    good          acceptable
  misaligned        [left arm]    []            [left arm, head]
  unwanted          []            []            [dark shadow]
  fail_patterns     []            []            []
  notes             "Torso ok..." "Clean match" "Minor drift..."
─────────────────────────────────────────────────────────
```

- Optionally write a markdown report to `.claude/tmp/vlm-comparison-{timestamp}.md`
- Does NOT close issues, merge branches, or modify `runs` table directly

---

### 4. `src/analyze.py` — async single-model runner

Responsibilities:
- Poll DB every N seconds for `runs` rows with no `vlm_annotations` entry for `active_model`
- Call `vlm_inference.py` with the active model
- Write result to `vlm_annotations`
- Promote result to `runs` columns (`rating`, `notes`, `misaligned`, `unwanted_features`)
  only if this is the designated primary model
- Log each annotation with run ID, model, rating, timing
- Graceful shutdown on SIGINT / Ctrl-C
- `--once` flag: process all pending runs and exit (for batch use in `batch.ps1`)

```powershell
# Run once after a batch completes:
.\.venv\Scripts\python.exe src/analyze.py --once

# Run as persistent daemon:
.\.venv\Scripts\python.exe src/analyze.py --poll-interval 30
```

---

## Database Schema Addition

New table added to `src/db.py`:

```sql
CREATE TABLE IF NOT EXISTS vlm_annotations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL REFERENCES runs(id),
    model_id      TEXT    NOT NULL,   -- e.g. "qwen2_5_vl_7b"
    timestamp     TEXT    NOT NULL,
    rating        TEXT,
    misaligned    TEXT,               -- JSON array stored as text
    unwanted_features TEXT,           -- JSON array stored as text
    fail_patterns TEXT,               -- JSON array stored as text
    notes         TEXT,
    raw_output    TEXT,               -- full model response before parsing
    latency_s     REAL,
    UNIQUE(run_id, model_id)
);

CREATE INDEX IF NOT EXISTS idx_vlm_run   ON vlm_annotations(run_id);
CREATE INDEX IF NOT EXISTS idx_vlm_model ON vlm_annotations(model_id);
```

The `UNIQUE(run_id, model_id)` constraint prevents duplicate annotations and
makes upserts safe. Existing `runs` columns (`rating`, `notes`, `misaligned`,
`unwanted_features`) are promoted from the active model's annotation by `analyze.py`.

---

## Error Handling

| Failure | Handling |
|---|---|
| No pose detected / blank image | Log warning, write `rating: "poor"`, `notes: "no pose visible"` |
| JSON parse failure | Retry once with correction prompt; on second failure write `raw_output` and skip |
| Model OOM | Catch `torch.cuda.OutOfMemoryError`, log, skip run, continue to next |
| Missing photo/render file | Log error with run ID, skip, continue |
| DB locked | Retry with exponential backoff (3 attempts, 2/4/8 s) |

---

## Download Extension

`src/download_models.py` extended with three new entries:

```python
("Qwen/Qwen2.5-VL-7B-Instruct",     None, None, "VLM pose analyzer — primary (~15 GB)"),
("OpenGVLab/InternVL2_5-8B-MPO",    None, None, "VLM pose analyzer — secondary (~16 GB)"),
("openbmb/MiniCPM-V-2_6",           None, None, "VLM pose analyzer — lightweight (~8 GB)"),
```

Added under a new `make.ps1` target: `vlm-download`.

---

## Testing Plan

1. **Unit** — `vlm_inference.py`: mock the model, test prompt construction, JSON parse,
   retry logic, schema validation
2. **Integration** — `compare_vlm.py`: run against 2–3 known (photo, render) pairs from
   `docs/reference-outputs/v1-baseline/`, verify output table renders and DB rows written
3. **Swap test** — change `active_model` in `vlm.yml`, re-run `analyze.py --once`,
   confirm new model's annotations appear without disturbing previous model's rows
4. **Failure injection** — pass a blank image, a corrupt path, a model that returns
   malformed JSON; verify graceful handling in all cases

---

## Open Questions

1. Should `compare_vlm.py` load all three models simultaneously (3× VRAM) or sequentially
   (slower but fits in 16 GB)? Recommendation: sequential by default, `--parallel` flag for
   future use when multi-GPU is available.
2. Fine-tuning path: if VLM accuracy on yoga-specific poses is insufficient after testing,
   PEFT on a small annotated dataset is viable for Qwen and InternVL (both Apache/MIT).
   Out of scope for this implementation.
