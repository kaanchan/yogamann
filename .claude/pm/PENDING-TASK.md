# PENDING TASK

_No active task._

## What just landed (2026-05-15 eve)

- `feature/v4-downgrade-testing` pushed at commit 4d5d4cc
- Per-model dispatcher in `src/vlm_inference.py` (Qwen-style ✅, InternVL + Molmo scaffolded as learning artifacts)
- Architecture lessons doc: `docs/architecture-decisions.md` (ADR-001, ADR-002)
- Issues opened: #32 (transformers pinning), #33 (pose pipeline alternatives)
- Deep-research results dropped by user into both research folders; ready for synthesis review

## Up next — user decision required

1. **#32:** Review the two deep-research deliverables in `docs/research/issue-32-transformers-version-pinning/` and consolidate into a FINAL-synthesis. Decide on exact `transformers==4.X.Y` pin + add to `requirements.txt` / `pyproject.toml`.
2. **#33:** Review the two deep-research deliverables in `docs/research/issue-33-pose-pipeline-evaluation/` (already includes a `SYNTHESIS.md` from the user). Pick top candidates to prototype after #32 + #29 stabilize.
3. **#29 follow-up:** Merge strategy — either land `feature/v4-downgrade-testing` -> `feature/vlm-analysis` (carrying the dispatcher + the v4 venv state) or revert venv to v5 and keep dispatcher branch as docs-only reference.
4. **Backlog (deferred):** #14, #16, #18, #19 — unchanged from prior session.

## Last commit on the active branch
- `4d5d4cc` — feat: per-model VLM inference dispatcher + transformers v4 pin investigation (#32)
