# PROGRESS

<!-- newest session at top — append-only, never rewrite -->

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

