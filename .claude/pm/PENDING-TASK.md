# PENDING TASK

## Active Branch
`feature/batched-inference` — off `main` (created 2026-05-15)
Last commit: `cf8c071` — docs: issue-34 GPU management research reports

## GH Issue
No issue created yet — open one before first code commit.
Suggested title: "perf: batched GPU inference for Qwen2.5-VL (vlm_batch_size)"
Reference: closes relationship to #29 (VLM integration) or standalone.

## Goal
Process N (photo, render) image pairs in a single `model.generate()` call
instead of one at a time. Expected: ~3-5x throughput at batch=2-4.

Current steady state (Qwen 4-bit, single item): ~7.5s/image
Target (batch=2): ~4s/image effective; (batch=4): ~2.5s/image effective

## Agreed Approach (spec fully designed — see context below)

### Files to touch
| File | Change |
|---|---|
| `src/vlm_inference.py` | Add `_infer_batch()` (private) + `annotate_batch()` (public) |
| `src/compare_vlm.py` | Inner GPU-batch loop in `_run_model_phase()`, `--vlm-batch-size` flag |
| `profiles/vlm.yml` | Add `vlm_batch_size: 2` under `qwen2_5_vl_7b` only |

### Sub-tasks (build sequence) — SESSION 2026-05-15 AGENT DISPATCH
- [ ] Open GH issue; note issue number in this file
- [ ] **WAVE 1 (parallel agents, worktree isolation)**
  - [ ] Agent A: `_infer_batch()` + `annotate_batch()` in vlm_inference.py
  - [ ] Agent B: `vlm_batch_size: 2` in profiles/vlm.yml
  - [ ] Merge both Wave 1 branches → feature/batched-inference
- [ ] **WAVE 2 (sequential, after Wave 1 merged)**
  - [ ] Agent C: inner-loop + CLI changes in compare_vlm.py
  - [ ] Merge Wave 2 branch → feature/batched-inference
- [ ] Phase 1 — `_infer_batch()` in vlm_inference.py
  - After existing `_infer()` (~line 399)
  - Signature: `(model, processor, messages_list, images_list, config) -> list[str]`
  - Set `processor.image_processor.max_pixels = max_px` before call (kwarg silently ignored)
  - Call `processor(text=[t0,t1,...], images=[p0,r0,p1,r1,...], padding=True, max_pixels=max_px)`
  - `model.generate(**inputs, max_new_tokens=max_new_tokens)` — ONE GPU call
  - Slice per-item output: `generated[i] = out_ids[i][len(in_ids[i]):]`
  - OOM catch → raise MemoryError so caller can halve and retry
- [ ] Phase 2 — `annotate_batch()` in vlm_inference.py
  - After existing `annotate()` (~line 480)
  - Signature: `(pairs: list[tuple[Path,Path]], model_key, ...) -> list[dict | Exception]`
  - Gate on `inference_style` absent (qwen_style only); others: sequential fallback
  - Per-item JSON retry uses single-item `_infer()`, not re-batching
  - OOM: split pairs in half, recurse with half-size; if len==1 re-raise
  - latency_s = batch_wall_time / len(valid_pairs) so ETA stays accurate
- [ ] Phase 3 — orchestrator changes in compare_vlm.py
  - Add `annotate_batch` to import
  - Add `vlm_batch_size: int = 1` param to `_run_model_phase()`
  - Compatibility gate: force vlm_batch_size=1 for non-qwen_style models
  - Inner loop: `for gpu_batch in chunked(batch, vlm_batch_size):`
  - Pre-inference log: show range `run={ids[0]}..{ids[-1]}  N={len(gpu_batch)}`
  - Per-item result dispatch in inner loop
  - Add `--vlm-batch-size N` CLI arg; resolution: CLI > vlm.yml > 1
- [ ] Phase 4 — config
  - `profiles/vlm.yml`: add `vlm_batch_size: 2` to `qwen2_5_vl_7b`
- [ ] Phase 5 — testing (user validates each before proceeding)
  - Correctness: batch=1 output matches single `annotate()` (structurally)
  - Throughput: time 4 images as batch=1 vs 2 vs 3 vs 4; record s/img
  - OOM safety: force batch=8 → confirm halving retry, no crash
  - Non-Qwen fallback: call with InternVL model key → confirm sequential

## Constraints
- Do NOT change `annotate()` signature or behaviour — backward compat required
- `_infer()` also unchanged — single-item path must remain for retry fallback
- All models except Qwen2.5-VL (qwen_style) auto-fall back to batch=1
- DB write is still per-item (crash-resumability must be preserved)
- `first_img` / `load_plus_first` ETA logic: treat first GPU batch as "first image"

## Related Research (issue #34 SYNTHESIS.md)
Key findings applicable here:
- **pynvml** (nvidia-ml-py): replace nvidia-smi subprocess in gpu_monitor.py for
  zero-overhead polling. Current subprocess approach causes process buildup in long runs.
  Priority: medium — do after batching is working.
- **nvitop**: VRAM-by-PID visibility, throttle_reasons() bitmask detection.
  Could replace/augment gpu_monitor.py cooldown_if_hot().
- **torch fragmentation guard**: check `allocated/reserved < 0.85` before empty_cache()
  rather than unconditional flush. Prevents unnecessary re-allocation penalty.
- **80% TGP power cap**: pynvml.nvmlDeviceSetPowerManagementLimit() — needs admin.
  Saves 12-15°C at 3-5% throughput cost. Relevant for overnight runs.
- **PM2 supervisor**: auto-restart with exp_backoff on crash. Relevant for
  unattended 12-14h runs after batching is validated.

## Hanging Threads (from prior session, not yet addressed)
- Qwen first-84-images used old prompt (naturalness comments). Consider
  `--force --models qwen2_5_vl_7b` re-annotation after new prompt is validated.
- `[qwen _infer] seq_len=NNN` diagnostic print in vlm_inference.py — remove
  once batching is confirmed working (currently useful for batch size verification).
- MiniCPM-V schema compliance: outputs numeric ratings ("3") not "good"/"acceptable"/"poor".
  Needs prompt tightening or output normalisation in _parse_output().
- Folder-based scheduling: proposed but not implemented in compare_vlm.py.
- gallery.py: stale lock check not wired into the terminal polling loop
  (_inference_terminal_dialog auto-refresh). Minor — auto-clears on next spawn attempt.

## Open Questions
- What batch size is actually safe on this GPU (RTX 5080, 16GB)?
  → empirically test 2, 3, 4 before committing a default in vlm.yml
- Does padding=True in the processor call produce correct results when
  images have different pixel counts (different aspect ratios)?
  → verify by checking rated outputs match expected for each item in batch
