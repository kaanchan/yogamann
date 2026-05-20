# Batch Runner Script

**7 nodes · Community 17 · Cohesion 0.43**

`scripts/batch.ps1`

---

## What it does

The outer PowerShell driver for per-directory render batches. Given a directory of source images, it runs `make_mannequin.py` for each image and manages output directories, cleanup, and error handling. This is the entry point for a typical render session before you move on to VLM annotation.

## Key functions

| Function | Role |
|----------|------|
| `Invoke-DirBatch(dir, profile)` | Main loop — iterates images in a directory, calls `Invoke-Python` for each |
| `Get-ImageFiles(dir)` | Returns all supported image files in a directory |
| `Get-OutputDir(src)` | Computes the output directory path for a source image |
| `Invoke-Python(cmd)` | Wrapper for Python subprocess calls with error propagation |
| `Invoke-Cleanup()` | Removes incomplete output directories after errors |
| `Test-OutputExists(src)` | Checks whether a render already exists for a source (skip logic) |

## Usage

```powershell
.\scripts\batch.ps1 -Dir input/reference -Profile yoga_asana
```

## Connects to

- [Mannequin Generator](mannequin-generator.md) — calls `make_mannequin.py`
- [Pose Pipeline Research](pose-pipeline-research.md) — documented in README as part of pipeline
