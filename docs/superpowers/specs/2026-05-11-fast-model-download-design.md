# Fast Model Download — Design Spec
Date: 2026-05-11
Status: Approved

## Problem
`huggingface-cli download` is deprecated and downloads are <1 MB/s because `hf_transfer` (the Rust-based fast-path) is not installed in the venv. Three models totalling ~10 GB need to land in `D:\models\hub` before the first end-to-end pipeline test (issue #7) can run.

## Models
| Repo ID | Size |
|---|---|
| `xinsir/controlnet-openpose-sdxl-1.0` | ~2.5 GB |
| `stabilityai/stable-diffusion-xl-base-1.0` | ~7 GB |
| `depth-anything/Depth-Anything-V2-Small-hf` | ~100 MB |
| `lllyasviel/Annotators` | small |

## Changes

### 1. venv — add `hf_transfer`
```
uv add hf_transfer
```
Lightweight Rust wheel. HF hub auto-activates it when `HF_HUB_ENABLE_HF_TRANSFER=1`.

### 2. make.ps1 — `download` target
- Replace `huggingface-cli.exe` → `hf.exe` (fixes deprecation)
- Set `$env:HF_HUB_ENABLE_HF_TRANSFER = "1"` scoped to download block
- Pass `--token $env:HF_TOKEN` when token is present
- Add `--max-workers 16`
- Leave `--cache-dir` unset; `HF_HUB_CACHE=D:\models\hub` (user env var) stays authoritative

## Data Flow
```
.\make.ps1 -Target download
  └─ sets HF_HUB_ENABLE_HF_TRANSFER=1
  └─ foreach model: hf download --max-workers 16 [--token ...] <repo>
       └─ blobs → D:\models\hub\models--<org>--<name>\blobs\
  └─ pipeline: from_pretrained() resolves from same cache via HF_HUB_CACHE
```

## Verification
After download completes:
1. Confirm snapshot dirs exist under `D:\models\hub` for all four models
2. Run `.\make.ps1 -Target test` — pipeline should load from local cache without any network calls to HF

## Out of Scope
- aria2c integration (not installed, unnecessary given hf_transfer)
- Changing where models are stored (D:\models\hub stays)
- Any pipeline code changes (this is download-only)
