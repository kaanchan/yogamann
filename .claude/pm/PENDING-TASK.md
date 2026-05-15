# PENDING TASK

## Active task — issue #32: transformers version pinning

**Branch:** `feature/v4-downgrade-testing` (to be created from `feature/vlm-analysis` at 440b99c)
**GH issue:** [#32](https://github.com/kaanchan/yogamann/issues/32) — tech-debt: pin transformers; v5 breaks trust_remote_code VLMs
**Parent context:** #29 — VLM pose analysis (currently working on `feature/vlm-analysis`)

### Why this exists

`(yogamann)` venv has `transformers==5.8.0` (unpinned install). v5 silently broke 3 of 4 VLMs:
- InternVL2.5-MPO: bnb quantizer crash (`all_tied_weights_keys` rename)
- MiniCPM-o-2.6: `WHISPER_ATTENTION_CLASSES` removed in v5
- Molmo-7B-D: needs explicit `AutoModelForCausalLM` (not v5-specific, but compounds)
- Qwen2.5-VL: native v5 support, works

### Plan

1. [x] Issues created: #32 (transformers pinning), #33 (pose pipeline eval research)
2. [x] Research folders created: `docs/research/issue-32-transformers-version-pinning/`, `docs/research/issue-33-pose-pipeline-evaluation/`
3. [in-progress] Agent A: writing `PROMPT.md` for #32 deep research (background)
4. [in-progress] Agent B: drafting `docs/architecture-decisions.md` (background)
5. [ ] Create branch `feature/v4-downgrade-testing`
6. [ ] `pip install "transformers>=4.49,<5"` in venv
7. [ ] Test Qwen -> InternVL -> MiniCPM-o -> Molmo (one at a time, stop on first failure)
8. [ ] Commit + push v4-downgrade branch
9. [ ] Append PROGRESS.md, clear PENDING-TASK.md

### Decisions / constraints

- **One model at a time** when testing (user directive — avoid blocked rabbit hole)
- **Keep current venv** (no separate v4 venv) — accepting that switching back to `feature/vlm-analysis` will run that branch's code against v4 transformers (mostly OK; bf16 + local_files_only hacks still work on v4)
- **Molmo** likely still needs the `AutoModelForCausalLM` fallback code change regardless of transformers version — that's a separate v4-independent fix
- **InternVL** on v4 should work with bnb 4-bit (the `all_tied_weights_keys` check doesn't exist) — saves 11 GB VRAM vs bf16 hack

### Side task — issue #33 (deferred)

User will drop deep-research results from yesterday's "Pose analyzer and pose render evaluation" prompt into `docs/research/issue-33-pose-pipeline-evaluation/`. Synthesis work waits until #32 + #29 stabilize.

### Last commit before this task: 440b99c (fix: local_files_only + revision support in vlm_inference)
