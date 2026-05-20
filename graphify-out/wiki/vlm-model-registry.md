# VLM Model Registry

**17 nodes · Community 9 · Cohesion 0.17**

`profiles/vlm.yml` · `docs/superpowers/specs/2026-05-12-vlm-pose-analysis-design.md` · `docs/superpowers/specs/2026-05-16-vlm-profile-system-design.md`

---

## What it is

The configuration layer for the VLM subsystem: which models are available, their HuggingFace repos and load settings, and which prompt variants exist. Changing `active_model` or `active_prompt` in `vlm.yml` is the only thing needed to switch the entire system to a different model or prompt.

## The 5 registered models

| Key | Model | Notes |
|-----|-------|-------|
| `qwen2_5_vl_7b` | Qwen/Qwen2.5-VL-7B-Instruct | **Active default.** `max_new_tokens: 128` (VRAM cap), `max_pixels: 401408` (~2× faster than default) |
| `internvl2_5_8b` | OpenGVLab/InternVL2_5-8B-MPO | `inference_style: internvl` — uses `model.chat()` |
| `minicpm_o_2_6` | openbmb/MiniCPM-o-2_6 | Omni-modal (audio + TTS). Required `_patch_qwen2_init_weights_for_bnb()` to work (see ADR-003) |
| `minicpm_v_2_6` | openbmb/MiniCPM-V-2_6 | Lighter sibling of MiniCPM-o |
| `molmo_7b_d` | allenai/Molmo-7B-D-0924 | `inference_style: molmo` — uses `processor.process + generate_from_batch` |

All models load in 4-bit quantization (`load_in_4bit: true`).

## Prompt variants

| Key | Description |
|-----|-------------|
| `v1` | Original baseline prompt |
| `v2_checklist` | Numbered body-part checklist — tends to over-report misalignment |
| `v2a` | Checklist in system prompt with guard against echoing the list |
| `v2b` | **Active default.** Joint-by-joint comparison, strict rating criteria, no numbered list |

`v2b` was chosen as the winner after a comparison run showing it produces the most selective, accurate misalignment lists across all 5 models.

## The `prompt_tag` column

The `vlm_annotations` table has a `prompt_tag` column alongside `model_id`. This means the same run can be annotated by the same model with different prompts and both results are stored. Useful for prompt evaluation without re-running from scratch.

## BitsAndBytes 4-bit quantization

All models use `load_in_4bit: true` to fit within 12–16 GB VRAM. BitsAndBytes compresses model weights to 4-bit at load time, reducing memory usage by ~4×. The tradeoff is a slight quality reduction — empirically acceptable for pose comparison tasks.

## Connects to

- [VLM Inference Core](vlm-inference-core.md) — reads this YAML to select model and prompt
- [Architecture Decisions & ADRs](architecture-decisions.md) — ADR-001 (version pins), ADR-002 (per-model dispatchers), ADR-003 (transformers pin)
- [Pose Pipeline Research](pose-pipeline-research.md) — Issue #33 research informed the model selection
