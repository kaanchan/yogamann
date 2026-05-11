# PROGRESS

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

