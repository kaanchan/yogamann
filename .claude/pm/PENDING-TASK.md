# PENDING TASK

**Branch:** master
**Last commit:** 2754d75
**Session date:** 2026-05-11

## GH Issues being addressed

| # | Title |
|---|-------|
| [#8](https://github.com/kaanchan/yogamann/issues/8) | Fix FP8 benchmark: use torch._scaled_mm not torch.matmul |
| [#9](https://github.com/kaanchan/yogamann/issues/9) | Add torch.compile to SDXL pipeline for Blackwell kernel fusion |
| [#10](https://github.com/kaanchan/yogamann/issues/10) | Install torchao + add NVFP4 benchmark and optional UNet weight quantization |

## Sub-tasks

- [x] Fix FP8 benchmark (both rtx5080-test.py) — #8 closed, commit 2d24a76
- [x] torch.compile in sd_make.py — #9 closed, commit 2d24a76
- [x] make.ps1 test runner — #11 closed, commit 13fa538
- [x] Structured logging throughout — #12 closed, commit 13fa538
- [ ] Fix slow model download — investigate fast download tool (aria2c / hf_transfer / huggingface-cli with resume)
- [ ] Download all models to D:\models\hub — user confirms downloads complete
- [ ] First end-to-end image test — `.\make.ps1` — user confirms (#7)
- [ ] Commit, push, close #7
- [ ] Check torchao nightly for cu130 Windows wheel (#10)
- [ ] Install torchao, add real NVFP4 benchmark (#10)
- [ ] Commit #10, push, close issue

## Agreed approach

- FP8 fix: use `torch._scaled_mm(a, b, scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16)` — single scalar scale tensor per operand
- torch.compile: `torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=False)` — tolerates diffusers graph breaks
- Apply changes to BOTH rtx5080-test.py copies (yogamann + System-tools/pytorch) in sync
- Issue order: #8 → #9 → #10

## Constraints / decisions

- `uv pip install` for any new deps (not `uv add`) to avoid re-triggering resolver
- torch.compile warmup latency ~2-5 min on first call — expected, not a bug
- #10 (torchao) blocked on confirming cu130 Windows wheel availability; source build as fallback

---
<!-- previous sessions below -->


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
