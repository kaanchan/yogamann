# PENDING TASK

_No active task. **5/5 VLMs working** on `transformers==4.49.0` with the model-major orchestrator._

## What's working as of this session (feature/v4-downgrade-testing)

| Model | bnb 4-bit | Inference style | Sample rating |
|---|---|---|---|
| Qwen2.5-VL-7B | ✅ | qwen_style (native) | good |
| InternVL2.5-MPO | ✅ | internvl | poor |
| Molmo-7B-D | ✅ | molmo | acceptable |
| MiniCPM-V-2.6 | ✅ | minicpm_v | (numeric) |
| MiniCPM-o-2.6 | ✅ | minicpm_v (sibling API) | good |

Orchestrator: `python src/compare_vlm.py --limit N --output-root D:/Temp/yogamann-output`
- Model-major loop, micro-batched at K=25
- Resumable via DB filter (skips already-done `(run_id, model_id)` pairs)
- `--force` overrides skip filter
- `--models` filters which subset to run
- Per-model elapsed time + final summary table

## What's next — user decision required

1. **Run the full 5-model × 981-image batch?** Estimated ~10-12 hours wall clock. Resumable, so safe to start and interrupt.
2. **Merge `feature/v4-downgrade-testing` back to `feature/vlm-analysis` (or to main)?** All work since the orchestration session is on this branch.
3. **Tackle issue #33** (pose pipeline alternatives evaluation)? Synthesis is already in `docs/research/issue-33-pose-pipeline-evaluation/SYNTHESIS.md`.
4. **Schema-compliance pass on MiniCPM-V?** Its outputs are valid JSON but use non-schema rating values (e.g. `"3"` instead of `"good"`/`"acceptable"`/`"poor"`). Prompt tightening would fix.

## Last commits on the active branch
- `b68397d` — docs: append PROGRESS + update PENDING-TASK; track orchestration plan (#32)
- `39036a3` — docs: fix VRAM refs (16 GB) + add ADR-004 for orchestration (#32)
- `aec7524` — feat: model-major batch orchestration with resumability (#32)
- (next commit will land MiniCPM-o resolution)
