# PENDING TASK

## Issue #29 — VLM pose analysis implementation

**Branch:** `feature/vlm-analysis`
**Last commit:** `6388d7b` — design spec committed; master pushed to origin/main
**Design spec:** `docs/superpowers/specs/2026-05-12-vlm-pose-analysis-design.md`

### Status
- [x] Brainstorming complete (single model runner + swap mechanism; Approach A first, C as target)
- [x] Design spec written and committed (`6388d7b`)
- [x] Issue #29 updated with VLM design summary comment
- [ ] Spec review by user
- [ ] `writing-plans` skill → implementation plan
- [ ] Implementation (vlm_inference.py, compare_vlm.py, vlm_annotations DB table, gallery integration)

### Key decisions from spec
- `vlm_annotations` table with `UNIQUE(run_id, model_id)` — safe upserts, multi-model
- Sequential model loading by default (fits 16 GB VRAM), `--parallel` flag optional
- Approach A: standalone VLM analysis tool first; Approach C (pipeline integration) as later phase
- Models to target: Qwen2.5-VL, InternVL2.5, MiniCPM-V (all runnable on RTX 5080 16 GB)

### Constraints
- Must not touch `db.py` schema until implementation plan is reviewed and agreed
- `feature/vlm-analysis` branch only — no merges to master until user validates

## Up next (backlog)
- #14 — torch.compile cudagraphs
- #16 — OKS self-eval
- #18 — standard test set
- #19 — mark.ps1 / gold index
