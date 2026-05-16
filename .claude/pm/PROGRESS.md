# PROGRESS

## 2026-05-15 — feature/batched-inference closed → main (a066eee)

- Added `_infer_batch()` + `annotate_batch()` to `vlm_inference.py` (#35)
- Added `--vlm-batch-size` CLI flag + two-level GPU-batch loop to `compare_vlm.py` (#35)
- Added fragmentation guard (allocated/reserved < 0.85) replacing unconditional `empty_cache()`
- Removed `[qwen _infer] seq_len` diagnostic print
- Tested batch=2: 11.5s/img vs 7.5s/img single — VRAM at 96%, batching counterproductive on 16GB
- vlm_batch_size set to 1; infrastructure ready for 24GB revisit
- Opened #36: gallery display bug for runs 957-981 (files present, display broken)
- Qwen full annotation run queued for overnight (~897 images remaining)

## 2026-05-15 — Session close: v4 branch merged, batched-inference branch created

### Accomplished
- **8x Qwen inference speedup**: root cause was `max_pixels` kwarg silently ignored by
  transformers 4.49 Qwen2VLProcessor. Fix: set `processor.image_processor.max_pixels`
  attribute directly. 60s/image → 7.5s/image steady state (seq_len 2200→1118 tokens).
- **Round-robin orchestration** with lag-first sort and time-budget rotation
  (`--time-budget 300s` default, fallback `--slice-pct 10%`). CLI: `--strategy`,
  `--time-budget`, `--slice-pct`, `--slice`. Old model-major preserved via `--strategy model-major`.
- **Granular logging**: pre-inference "→ inferring ..." print + `flush=True` eliminates
  silent-hang appearance during 60s inference gaps.
- **GPU safety layer**: `batch_lock.py` (lock file, PID-based stale detection),
  `gpu_monitor.py` (nvidia-smi polling, thermal cooldown), `win_job.py`
  (Windows Job Object KILL_ON_JOB_CLOSE auto-kills orphaned subprocesses).
- **VLM Compare gallery tab**: side-by-side model vs human ratings, misaligned +
  artifacts display, render thumbnail alongside source, sqlite3.Row→dict fix,
  lock guard on Generate buttons, blocked-by-lock banner, beforeunload JS.
- **db.py**: pivot query now includes `{model}_misaligned` + `{model}_unwanted`.
- **VLM prompt rewrite**: explicit "Do NOT evaluate naturalness/relaxation/quality"
  constraints; focus narrowed to joint angles/limb positions/weight distribution.
- **Issue #34 research**: SYNTHESIS.md consolidated from ChatGPT + Gemini deep-research.
  Key: pynvml > nvidia-smi subprocess; nvitop for throttle bitmasks; PM2 for crash recovery.
- **Merged** `feature/v4-downgrade-testing` → `main`, pushed to GitHub.
- **Created** `feature/batched-inference` branch off main.

### Issues opened/closed
- #34 research complete (SYNTHESIS.md written); issue remains open for implementation
- #32 remains open (full 5-model annotation run still in progress)

### Known limitations still open
- Qwen first 84 annotations used old prompt (naturalness bias) — needs --force re-run
- MiniCPM-V outputs numeric ratings ("3") not schema values — needs normalisation
- `[qwen _infer] seq_len=NNN` diagnostic print still in vlm_inference.py — remove post-validation
- pynvml not yet adopted (gpu_monitor.py still uses nvidia-smi subprocess)

## 2026-05-15 (very late eve) — MiniCPM-o-2.6 resolved; 5/5 VLMs working (#32)

- Diagnosed: native Qwen2 `_init_weights` (in `transformers.models.qwen2.modeling_qwen2:382`) calls `.normal_()` on bnb-quantized uint8 weights when MiniCPM-o's omni stack triggers post-load weight init. PyTorch has no `normal_kernel` for Byte dtype → crash.
- Fix: `_patch_qwen2_init_weights_for_bnb()` in `src/vlm_inference.py` — wraps `Qwen2PreTrainedModel._init_weights` with an early-return for uint8 weights. Applied at module load (one-time, idempotent via `_bnb_safe` sentinel). Benign for the other Qwen2-based models (Qwen2.5-VL, MiniCPM-V-2.6) since they don't trigger this init path on quantized weights — verified by regression: Qwen2.5-VL post-patch still produces `rating="good"` in 27.2s, identical to pre-patch behavior.
- MiniCPM-o-2.6 re-enabled in `profiles/vlm.yml` with `inference_style: minicpm_v` (sibling models share the `model.chat(image=None, msgs=[...], tokenizer=)` API).
- Empirical validation through orchestrator: 1 ok / 0 err / 29.9s on run 981. Output: `rating="good"`, `misaligned=["torso"]`, 10.3s pure-inference latency. Load: 19.5s, 6.25 GB VRAM.
- ADR-003 updated to reflect resolution. PENDING-TASK.md cleared.
- All 5 VLMs in our comparison suite now operational on transformers==4.49.0.

## 2026-05-15 (late eve) — VLM batch orchestration refactor (#32)

- Inverted `compare_vlm.py` loop from image-major to model-major. Each model phase: load → annotate K-image micro-batches → evict. Total cold-load cost is now N_models, not N_models × N_images.
- Added `evict_model(key)` + `evict_all()` to `vlm_inference.py` — public eviction so callers manage GPU memory between phases. Verified empirically: 0.00 → 5.82 GB (load) → 0.00 GB (evict) — full VRAM freed.
- DB-driven resumability via existing `get_unanalyzed_runs`. Each `save_vlm_annotation` auto-commits, so kill-and-resume picks up precisely from the last completed image. Pair-atomic state model — no session/batch ID, idempotent at the (run_id, model_id) level.
- New CLI flags on `compare_vlm.py`: `--models`, `--force`, `--batch-size` (default 25).
- New module `src/_batch_utils.py` with `chunked()` + `format_summary()`, pytest-covered (7/7 in 0.05s). New DB helper `count_vlm_annotations(conn, model_id)`.
- Architecture-decisions ADR-004 added. ADR-003 + PROMPT.md corrected to 15.92 GB VRAM (RTX 5080 Laptop), not 24 GB. The bf16 MiniCPM-o fallback path in ADR-003 marked not-viable on this hardware.
- Acceptance smoke (commit aec7524): 4 models × 10 images, 28 new annotations, 12 skipped, 0 errors, 382.9s total wall clock. No OOM. Per-model phase elapsed: Qwen 95s, InternVL 51s, MiniCPM-V 113s, Molmo 124s.
- Plan written via `superpowers:writing-plans` skill at `docs/superpowers/plans/2026-05-15-vlm-batch-orchestration.md`. Executed in 6 waves with parallel sub-agents on independent tasks (some Bash-permission denied; consolidated inline in main session).
- Commits: `48e531c` (evict helpers), `1157814` (db count), `da72d3e` (_batch_utils+tests), `21a679f` (flags), `aec7524` (loop inversion), `39036a3` (docs + ADR-004).

## 2026-05-15 (eve) — transformers v5 breakage + per-model dispatcher investigation (#32), feature/v4-downgrade-testing

- Diagnosed: `(yogamann)` venv had `transformers==5.8.0` (unpinned install). v5 silently broke 3 of 4 VLMs via `trust_remote_code` rot.
- Issues opened: #32 (transformers version pinning), #33 (pose pipeline evaluation alternatives).
- Research folders created with deep-research prompts + agent-written deliverables: `docs/research/issue-32-transformers-version-pinning/`, `docs/research/issue-33-pose-pipeline-evaluation/`. User dropped their independent deep-research results into both folders during this session.
- Branched `feature/v4-downgrade-testing` from `feature/vlm-analysis@440b99c`. Downgraded transformers 5.8.0 → 4.57.6 in venv. Qwen2.5-VL confirmed working on v4 (no regression).
- Implemented per-model inference dispatcher in `src/vlm_inference.py`: `_load_model` skips processor when `inference_style` set, `annotate()` dispatches via lambda. New helpers: `_get_tokenizer`, `_internvl_preprocess`, `_infer_internvl`, `_infer_molmo`, `_patch_internvl_generation_mixin`. Added `AutoModelForCausalLM` to model-load fallback chain.
- Findings (documented in ADR-002 at `docs/architecture-decisions.md`): the dispatcher pattern works architecturally, but each `trust_remote_code` model has *independent* rot beyond just transformers version — InternVL needs 4+ cascading v4.50+ patches (GenerationMixin, generation_config, DynamicCache), Molmo's hosted preprocessing file has unconditional `import tensorflow` that HF's import-check can't deduce is dead code. Dispatchers retained as learning artifacts + reference implementations.
- Architecture lessons doc created: `docs/architecture-decisions.md` ADR-001 (pin ML libs like prod DB drivers) + ADR-002 (per-model dispatcher pattern + trust_remote_code multi-axis rot). Both drafted by parallel background agents while main session worked on the dispatcher implementation.
- Side-quest dependencies installed during the v5 chase (still in venv on this branch): `soundfile`, `torchaudio` (cu130 nightly matched), `librosa` + transitive (numba/llvmlite/sklearn), `sentencepiece`. Carried over to v4 venv state.
- Commit: 4d5d4cc (1926 insertions, 13 files). Pushed to origin. PR not yet opened — branch is exploratory; merge decision pending #32 deep-research review.

## 2026-05-15 — VLM Pose Analysis implementation (#29), feature/vlm-analysis

- Implemented all 7 tasks via Subagent-Driven Development skill
- Task 1: vlm_annotations DDL + save/get/unanalyzed helpers in db.py (5d2d161, 00d4c9c)
- Task 2: profiles/vlm.yml + bitsandbytes dep in pyproject.toml (323d369)
- Task 3: tests/conftest.py + 6 DB tests in tests/test_db_vlm.py (e12ecfe)
- Task 4: src/vlm_inference.py lazy-cached inference core + 13 unit tests (886d5ee)
- Task 5: src/compare_vlm.py multi-model comparison harness (03b11f2, 94e423c)
- Task 6: src/analyze.py polling daemon + --once batch mode (594c9bc, abfbbf6)
- Task 7: download_models.py refactor (PIPELINE_MODELS+VLM_MODELS, --vlm-only) + make.ps1 targets (7c8721f, aef3e61)
- All 19 tests passing; branch pushed to origin/feature/vlm-analysis
- Issue #29 open; branch kept for user to merge/PR when ready

## 2026-05-13 — PM reconciliation, push to remote, ctx-upgrade

- Confirmed #25 (README, 3c6b800) and #26 (multi-batch, d428c63) already committed from prior sessions
- Pushed master → origin/main: 9 commits now live (e221e6c..976145d)
- ctx-upgrade: v1.0.134 (already latest); all hooks PASS, doctor PASS
- PM docs (PENDING-TASK, PROGRESS, TODO) reconciled with May 12 session work

## 2026-05-12 — Gallery major UI sprint + VLM design spec (#27, #28, #29)

- SQLite threading crash fixed (ProgrammingError) in gallery.py
- Gallery: stopped pulsing/throbbing; pagination added; ratings → radio buttons
- Added `misaligned` pane (body part checkboxes), Notes field, metadata at top of cards
- Multi-user auth: login/signup, `users` table, single-step account creation confirmation blocker
- Fixed IndexError on misaligned field, `no such table: users` error
- Notes + patterns pane added side-by-side layout
- OBS-NNN pattern system discussed (for tagging recurring generation artifacts)
- Deep-research VLM prompt written: `.claude/tmp/deep-search-vision-models.md`
- VLM design spec committed: `docs/superpowers/specs/2026-05-12-vlm-pose-analysis-design.md` (6388d7b, `feature/vlm-analysis`)
- Gallery commits: ccc8570 (#27 multi-user annotations, #28 patterns catalogue, gallery UX)
- #29 VLM open — spec done, implementation plan not yet written

## 2026-05-12 — Repo housekeeping + VLM research organization (pre-#29 groundwork)

### Completed
- Audited BACKUPS/: v1 used SD 1.5 + MediaPipe (not SDXL + DWPose); profile drifted on all params (steps 30→25, cond_scale 1.0→1.5, guidance 7.5→7.0, prompts rewritten)
- profiles/yoga_asana_v1_archival.yml — fully annotated infrastructure + param diff
- Back-dated commit 2025-07-20: docs/archive/v1-mediapipe-2025-07-20/ preserves full v1 source in git history
- docs/reference-outputs/v1-baseline/ (19 PNGs) + batch-feature mannequin committed; v1 sample added to README
- docs/observations/ — pipeline-observations.md + observations.json committed (were untracked)
- Scripts reorganized: batch.ps1 + multi-batch.ps1 + monitor.ps1 → scripts/
- docs/research/issue-29-vlm-pose-analysis/ — all 4 VLM reports added
- BACKUPS/ deleted (fully preserved via archive + reference-outputs)
- Makefile synced with make.ps1 (ingest, review, download + cross-platform OS detection)
- out/ → output/ consolidated; all scripts, README, .gitignore updated
- docs/ideal-targets/ — 3 reference mannequin images + README
- output/README.md tracked via .gitignore negation
- All commits pushed (94065fc..976145d)

### Issues
- #29 open — VLM analysis: brainstorming complete, design doc pending next session

## 2026-05-11 — README, multi-batch, cond_scale sweep, pipeline fixes

- #25 README.md: full pipeline docs — make.ps1 targets, batch.ps1 flags, multi-batch.ps1, profiles, DB, advanced CLI
- #26 multi-batch.ps1: overnight queue runner; fixed batch.ps1 SourceRoot Resolve-Path; dry-run tree view; hashtable splatting fix
- #17 cond_scale sweep: 5-variant sweep (1.0→2.0, seed 42) on sample-4; v3=1.5 preferred, v1=1.0 clean alternative; yoga_asana.yml updated to cond_scale 1.5
- prompt overhaul: removed SD1.5 hacks (8k, "advanced"), added face/gender/age negatives, human/skin negatives, "head turned to match pose"
- versioning bug fixed: next_free_stem was stripping version tag, causing silent overwrites on every run
- sd_make.py: MIN_DIM=768 floor prevents sub-768px generation on small inputs
- sd_make.py: full absolute paths in Saved/Metrics/Contact-sheets log lines
- sd_make.py: output path logged at render() start
- sd_make.py: per-task try/except — bad files report+continue, don't abort worklist
- make_mannequin.py: profile name now recorded in metrics.json and DB
- New profiles: yoga_asana_strong.yml (cond_scale 2.0), yoga_asana_cs_sweep.yml (5-step sweep)

## 2026-05-11 — SQLite revision history, DB-backed gallery, skip-existing

- **#24 closed** — `src/db.py`: schema (source_images, runs, thumbnails), SHA256 identity, 150×150 thumbnail blobs, ingest-from-JSON, rating write, query helpers.
- **#21 closed** — `src/gallery.py` rewritten: DB-backed, run history expander per source image, rated/gold stats sidebar, auto-refresh, pagination.
- **#23 closed** — `batch.ps1`: `-Overwrite` switch; skips images with existing output by default; summary reports skip count.
- **make.ps1** — added `ingest` and `review` targets, `-OutputRoot` param.
- **sd_make.py** — `source_sha256` added to metrics JSON; best-effort DB insert after each run; EXIF fix also applied to comparison panel (contact sheets now use corrected src_img).
- **pyproject.toml** — added `streamlit>=1.37` dependency.

## 2026-05-11 — BSOD fix, EXIF orientation, batch/monitor improvements

- **#20 closed** — BSOD (kernel panic 0x00020001) from phone photos (5184×3888) generating at full native resolution. Fixed in `sd_make.py`: scale down so long side ≤ 1024, snap to nearest 64.
- **#22 closed** — EXIF orientation: phone photos processed rotated. Fixed with `ImageOps.exif_transpose()` on first open; single PIL object reused for pose detection and dimension reads.
- **batch.ps1** — Added `-Monitor` switch (spawns `monitor.ps1` in new terminal window).
- **monitor.ps1** — Added GPU active-time tracker with dip-tolerance, per-process yogamann CPU/RAM rows, wall-clock timers.
- **Issues opened** — #21 (Streamlit gallery + feedback), #23 (skip-existing flag).

## 2026-05-11 — Batch pipeline + warning suppressions

- Closed #20: batch.ps1 (tree walker, DryRun, output dir exclusion, D:\Temp\yogamann-output default)
- make_mannequin.py: --output-dir, single-subprocess worklist (one CUDA warm-up per folder)
- sd_make.py: 6 warning suppressions; onnxruntime severity=3 in dwpose_onnx.py
- Commit: d4a828d

<!-- newest session at top — append-only, never rewrite -->n## 2026-05-11 — HF offline + metrics JSONn- Closed #13: HF_HUB_OFFLINE=1 in make.ps1 test/test-all; .metrics.json sidecar per run (seed, timing phases, rating/notes stub)n- Commit: 1cd431f

## 2026-05-11 — First working end-to-end pipeline run (#7)
- Installed hf_transfer 0.1.9, rtmlib 0.0.15, matplotlib via uv
- Replaced deprecated huggingface-cli with Python download_models.py (fp16-only SDXL, allow_patterns per model)
- Replaced broken controlnet_aux DWposeDetector (requires mmcv) with rtmlib ONNX wrapper (dwpose_onnx.py)
- Fixed sd_make.py: variant="fp16", graceful torch.compile fallback, corrected DWposeDetector instantiation
- Pipeline runs end-to-end: 130s/image, 0.8s/step (no compile), 2.0s pose (CPU ONNX)
- Baseline defects: head faces wrong direction, no per-run timing telemetry
- GH issues: #13 (HF offline+telemetry), #14 (torch.compile/cudagraphs), #15 (onnxruntime-gpu), #16 (OKS self-eval), #17 (head direction), #18 (baseline test set)
- Committed 24e3b93, e8d73b7 → pushed to main

## 2026-05-11 — Session 3 cont: model download + make.ps1 fixes

- HF_HUB_CACHE set to D:\models\hub (system-wide, all projects share cache)
- make.ps1 download target fixed: huggingface-cli path corrected, $Profile renamed
- Blocked: safetensors download extremely slow even with HF_TOKEN set — need fast download tool
- Models still needed: xinsir/controlnet-openpose-sdxl-1.0 (~2.5 GB), stabilityai/stable-diffusion-xl-base-1.0 (~7 GB), depth-anything/Depth-Anything-V2-Small-hf (~100 MB)
- Commit: bdb8294

## 2026-05-11 — Session 3 cont: logging + run.ps1 runner

- Issue #11 closed: run.ps1 created — targets: test/test-all/diag/gallery/open; -LogLevel param; colored headers; browser auto-open on success
- Issue #12 closed: structured logging wired through make_mannequin, sd_make, extract_pose, create_contact_sheet; YOGAMANN_LOG_LEVEL env var propagates to subprocesses
- Commit: 13fa538
- Open: #7 (first end-to-end image test, user runs .\run.ps1), #10 (torchao/NVFP4)

## 2026-05-11 — Session 3: FP8 fix + torch.compile + venv rebuild

- Rebuilt venv: Python 3.13.5 / torch 2.13.0.dev20260510+cu130 (uv cache hit, fast)
- Fixed uv triton resolver bug: added `environments = ["sys_platform == 'win32'"]` to pyproject.toml
- Issue #8 closed: FP8 benchmark corrected (torch._scaled_mm + b.T column-major); E4M3FN confirmed live on sm_120 (~8× faster than BF16 on equivalent workload)
- Issue #9 closed: torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=False) added to build_pipe()
- Same fixes applied to System-tools/pytorch/ (local only, no remote)
- Open: #7 (end-to-end image test), #10 (torchao/NVFP4)
- Commit: 2d24a76

## 2026-05-11 — First clean run on new laptop

- Resolved mediapipe cp312 incompatibility: pinned mediapipe==0.10.13 — #6 (614b3d5)
- Fixed Windows cp1252 UnicodeEncodeError on ✓/⚠ print chars across all src files — #6
- Added PYTHONUTF8=1 to subprocess env in make_mannequin.py
- Full pipeline confirmed working: pose extraction → SD diffusion (30 steps, ~2.5 it/s on RTX 5080) → contact sheets → gallery
- smgrep CLI works (needs `smgrep index` first); MCP integration hits wrong store (stale empty store)

## 2026-05-11 — Env migration to pip / RTX 5080

- Deleted `environment.yml`; added `requirements.txt` with PyTorch cu128 + all pip deps — #5 closed (9679e40)
- `sd_make.py` `build_pipe()`: removed xformers, SDPA auto-used by PyTorch 2.6+
- README: venv setup instructions replacing conda section

## 2026-05-11 — Bug fix, gallery generator, README

- Fixed `OUTPUT_DIR_png` → `output_png` key bug in `build_tasks()` — #2 closed (7fe5cdc)
- Added `src/make_gallery.py` — post-batch HTML gallery, dark grid, lightbox, reads PNG metadata — #3 closed (fe22951)
- Updated `README.md` with single-image, batch, gallery, profile table, and worklist instructions — #4 closed (fe22951)
- Set up `.claude/pm/` PM structure and `.claude/tmp/` gitignored staging
- Pushed master to `origin/master`

