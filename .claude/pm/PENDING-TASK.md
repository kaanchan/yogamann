# PENDING TASK

_Pre-MiniCPM-o orchestration refactor landed. Branch ready for user-led MiniCPM-o research._

## Active context for MiniCPM-o investigation (issue #32 follow-up)

**Working models on transformers==4.49.0:** Qwen2.5-VL-7B, InternVL2.5-MPO, Molmo-7B-D, MiniCPM-V-2.6. Orchestrator now runs all 4 in one batch (model-major, micro-batched, resumable).

**Blocked model:** MiniCPM-o-2.6.

**Failure on transformers==4.49.0 + bitsandbytes 0.49.2:**
```
NotImplementedError: "normal_kernel_cpu" not implemented for 'Byte'
  File "transformers/models/qwen2/modeling_qwen2.py", line 385, in _init_weights
    module.weight.data.normal_(mean=0.0, std=std)
```

**Root cause:** MiniCPM-o uses a Qwen2 audio-language backbone. transformers' native Qwen2 `_init_weights` calls `.normal_()` on bnb-quantized uint8 weights, which has no kernel implementation. This is the only model in our suite where bnb 4-bit fights *native* transformers code; the other three only had bnb fighting `trust_remote_code` code.

## Search queries to run (user-led)

1. **Most direct:** `"normal_kernel_cpu" "Byte" MiniCPM-o bitsandbytes 4-bit`
2. **HF discussions:** https://huggingface.co/openbmb/MiniCPM-o-2_6/discussions — filter for "bitsandbytes" / "4bit" / "quantization"
3. **Official inference recipe:** https://github.com/OpenBMB/MiniCPM-o — check `requirements.txt` and inference example
4. **Fallback:** `MiniCPM-o-2_6 bf16 inference` — bf16 path would sidestep bnb entirely (but on 15.92 GB GPU, 8B bf16 = 16 GB weights with zero KV cache room → not viable on this hardware; reserved for future workstation upgrade or CPU inference)

## Constraints on any proposed solution

- ~15.9 GB peak VRAM budget on RTX 5080 Laptop (bf16 8B weights ≥ 16 GB → not viable on current hardware)
- Cannot change `transformers==4.49.0` pin without breaking the other 4 models (see ADR-003)
- `bnb_4bit_skip_modules` config option would need exact module names from MiniCPM-o's audio-language Qwen2 backbone to skip quantizing it
- Vendor-forking the modeling file is also valid (patch the `_init_weights` to early-return for already-quantized modules)

## What's also landed this session (for context)

- `feature/v4-downgrade-testing` branch, 6 commits ahead since orchestration session start
- `requirements.txt` + `pyproject.toml` pin `transformers==4.49.0` exactly
- New batch orchestrator: `python src/compare_vlm.py --limit N --output-root D:/Temp/yogamann-output` runs all 4 working models model-major, resumable, with `--force` / `--models` / `--batch-size` flags
- Plan at `docs/superpowers/plans/2026-05-15-vlm-batch-orchestration.md` (for reference / future similar work)
- All architectural decisions captured in ADR-001 through ADR-004 in `docs/architecture-decisions.md`

## Last commit on the active branch
- `39036a3` — docs: fix VRAM refs (16 GB) + add ADR-004 for orchestration (#32) (+ Wave 6 commit pending after PM updates)
