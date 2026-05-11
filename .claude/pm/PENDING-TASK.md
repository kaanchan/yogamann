# PENDING TASK

**Branch:** master  
**Last commit:** f2a3891  
**Session date:** 2026-05-11

## GH Issues being addressed

| # | Title |
|---|-------|
| [#7](https://github.com/kaanchan/yogamann/issues/7) | Modernise stack: SDXL + DWPose + Depth Anything V2 + Python 3.13 / cu130 nightly |

## Sub-tasks

- [x] Create GH issue #7
- [x] Update PENDING-TASK.md
- [ ] Create `pyproject.toml` (uv, Python 3.13, cu130 nightly)
- [ ] Create `src/diagnostics/rtx5080-test.py`
- [ ] Update `src/sd_make.py` (SDXL, DWPose, bfloat16, 1024px)
- [ ] Update `src/extract_pose.py` (remove mediapipe, DWposeDetector)
- [ ] Update `src/create_contact_sheet.py` (Depth Anything V2)
- [ ] Update profiles (SDXL parameters)
- [ ] Rebuild venv with `uv sync`, run diagnostics — user confirms
- [ ] Single image test run — user confirms
- [ ] Commit, push, close #7, write PROGRESS.md

## Agreed approach

- Delete `environment.yml`; replace with `requirements.txt` using PyTorch cu128 wheels
- Remove `pipe.enable_xformers_memory_efficient_attention()` from `build_pipe()` in `sd_make.py` — PyTorch 2.6+ SDPA is faster on Blackwell and requires no separate package
- Update README setup section with venv + pip instructions

## Files / functions affected

| File | Change |
|------|--------|
| `environment.yml` | Deleted |
| `requirements.txt` | New file |
| `src/sd_make.py` → `build_pipe()` | Remove xformers line |
| `README.md` → setup section | venv instructions |

## Sub-tasks

- [x] Create GH issue #5
- [x] Update PENDING-TASK.md
- [ ] Delete `environment.yml`, create `requirements.txt`
- [ ] Update `src/sd_make.py` `build_pipe()`
- [ ] Update `README.md`
- [ ] Commit, push, close #5, write PROGRESS.md

## Constraints / decisions

- Pin `torch>=2.6.0` (first PyTorch release with full Blackwell/cu128 support)
- Keep `diffusers==0.34.0` pinned — code was written against this version
- Use `--extra-index-url` not `--index-url` so PyPI remains primary source for non-torch packages
- No xformers in requirements — SDPA is built into torch 2.6+ and faster on Blackwell

## Open questions

- None currently
