# VLM Inference Core

**53 nodes · Community 0 · Cohesion 0.07**

`src/vlm_inference.py`

---

## What it does

This is the execution engine for all VLM pose comparison. It loads a model into GPU memory, runs inference on one or more image pairs (source yoga photo + mannequin render), and returns a structured JSON annotation. Everything else in the system — the batch orchestrator, the single-run analyzer, the gallery — ultimately calls into this module.

## Key entry points

| Function | Role |
|----------|------|
| `annotate(run_id, src_img, render_img)` | Single-pair inference. Returns a validated dict with `rating`, `misaligned`, `unwanted_features`, `fail_patterns`, `notes`. |
| `annotate_batch(run_ids, ...)` | Batched wrapper. Calls `annotate()` in a loop with VRAM-aware micro-batching. |
| `_load_model(model_key)` | Loads a model+processor from HuggingFace (4-bit quantized by default). Caches the loaded model in a module-level dict. |
| `evict_model(model_key)` | Unloads a specific model from GPU memory. |
| `evict_all()` | Unloads all cached models. Called between model phases in batch mode. |

## The 5 inference backends

Each model has its own private `_infer_*()` function because their APIs are not compatible:

| Backend function | Model | API style |
|-----------------|-------|-----------|
| `_infer_qwen2_5_vl()` | Qwen2.5-VL-7B | `processor + model.generate()` |
| `_infer_internvl()` | InternVL2.5-8B | `model.chat()` with pixel_values |
| `_infer_minicpm_v()` | MiniCPM-V-2.6 / MiniCPM-o-2.6 | `model.chat(system_prompt=...)` |
| `_infer_molmo()` | Molmo-7B-D | `processor.process() + model.generate_from_batch()` |

The `inference_style` key in `profiles/vlm.yml` selects which backend is used for each model.

## Prompt and config loading

`_load_config(prompt_key=None)` reads `profiles/vlm.yml` and returns:
- The active model's repo, backend, and quantization settings
- The active prompt's `system` and `user` strings (from `prompts:` dict)
- Falls back to `active_prompt` in the YAML if no `prompt_key` is passed

`_build_messages(prompt, images)` assembles the message list, prepending a `system` role when the prompt has one. The system message is what tells the model *not* to comment on clothing, naturalness, or artistic style.

## JSON parsing and validation

`_parse_output(raw_text)` handles the messy reality that models don't always return clean JSON:
- Strips markdown fences (` ```json ... ``` `)
- Validates required keys: `rating`, `misaligned`, `unwanted_features`, `fail_patterns`, `notes`
- Raises `VLMSchemaError` on missing keys or invalid `rating` value

## Known issue

Molmo populates `unwanted_features` and `fail_patterns` with naturalness critiques ("unnaturally rigid limbs") despite the system prompt forbidding this. The other 4 models comply. Needs prompt or field-description patch.

## Connects to

- [Batch Orchestration](batch-orchestration.md) — calls `annotate_batch()` per model phase
- [Single-Run Analyzer](single-run-analyzer.md) — calls `annotate()` for individual runs
- [VLM Model Registry](vlm-model-registry.md) — config lives in `profiles/vlm.yml`
- [Architecture Decisions & ADRs](architecture-decisions.md) — ADR-002 (per-model dispatchers), ADR-003 (transformers pin)
