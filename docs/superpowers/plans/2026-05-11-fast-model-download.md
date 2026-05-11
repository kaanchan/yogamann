# Fast Model Download Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace slow deprecated `huggingface-cli` download with `hf` + `hf_transfer` for ~100× faster model caching.

**Architecture:** Install the `hf_transfer` Rust wheel into the venv via uv, then update the `download` target in make.ps1 to use the new `hf` CLI with the transfer env var set and `--max-workers 16`. No pipeline code changes — models land in the same `D:\models\hub` cache the pipeline already reads from.

**Tech Stack:** uv, hf_transfer (Rust wheel), huggingface_hub `hf` CLI, PowerShell make.ps1

---

### Task 0: Write PENDING-TASK.md

**Files:**
- Modify: `.claude/pm/PENDING-TASK.md`

- [ ] **Step 1: Write PENDING-TASK.md before any code changes**

Replace contents with:

```markdown
# PENDING TASK

Branch: master
Last commit: c643bd0
GH issue: #7 (first end-to-end test — blocked on models)

## Task
Wire in hf_transfer for fast model downloads (unblocks #7).

## Approach
1. uv add hf_transfer
2. Update make.ps1 download target: hf.exe + HF_HUB_ENABLE_HF_TRANSFER=1 + --max-workers 16
3. Run .\make.ps1 -Target download, verify blobs land in D:\models\hub
4. Run .\make.ps1 -Target test to confirm pipeline loads locally

## Sub-tasks
- [ ] Task 0: Write PENDING-TASK.md
- [ ] Task 1: Install hf_transfer via uv
- [ ] Task 2: Update make.ps1 download target
- [ ] Task 3: Run download, verify cache
- [ ] Task 4: Verify pipeline loads from local cache
- [ ] Task 5: Commit and push

## Constraints
- HF_TOKEN must be set in PS session before running download
- HF_HUB_CACHE=D:\models\hub must remain authoritative (no --cache-dir flag)
- Download target only — no pipeline code changes
```

---

### Task 1: Install hf_transfer

**Files:**
- Modify: `pyproject.toml` (via uv)
- Modify: `uv.lock` (auto-updated)

- [ ] **Step 1: Add hf_transfer to the venv**

```bash
uv add hf_transfer
```

Expected output includes a line like:
```
+ hf_transfer==0.1.x
```

- [ ] **Step 2: Verify it imports in the venv**

```bash
.venv/Scripts/python.exe -c "import hf_transfer; print('hf_transfer OK:', hf_transfer.__version__)"
```

Expected:
```
hf_transfer OK: 0.1.x
```

- [ ] **Update PENDING-TASK.md sub-task checkbox**

Mark Task 1 complete in PENDING-TASK.md.

---

### Task 2: Update make.ps1 download target

**Files:**
- Modify: `make.ps1:56-68`

- [ ] **Step 1: Replace the download block**

Current block (lines 56–68):
```powershell
    "download" {
        if (-not $env:HF_TOKEN) {
            Write-Host "`n[WARN] HF_TOKEN not set — downloads will be slow (unauthenticated)." -ForegroundColor Yellow
            Write-Host "       Get a free read token at https://huggingface.co/settings/tokens" -ForegroundColor Yellow
            Write-Host "       Then: `$env:HF_TOKEN = 'hf_...'; .\make.ps1 -Target download`n" -ForegroundColor Yellow
        }
        foreach ($model in $Models) {
            Invoke-Step "Download $model" {
                & .\.venv\Scripts\huggingface-cli.exe download $model
            }
        }
        Write-Host "`nAll models cached. Run .\make.ps1 to generate." -ForegroundColor Green
    }
```

Replace with:
```powershell
    "download" {
        $env:HF_HUB_ENABLE_HF_TRANSFER = "1"
        if (-not $env:HF_TOKEN) {
            Write-Host "`n[WARN] HF_TOKEN not set — downloads will be slower (unauthenticated)." -ForegroundColor Yellow
            Write-Host "       Get a free read token at https://huggingface.co/settings/tokens" -ForegroundColor Yellow
            Write-Host "       Then: `$env:HF_TOKEN = 'hf_...'; .\make.ps1 -Target download`n" -ForegroundColor Yellow
        }
        foreach ($model in $Models) {
            Invoke-Step "Download $model" {
                $tokenArgs = if ($env:HF_TOKEN) { @("--token", $env:HF_TOKEN) } else { @() }
                & .\.venv\Scripts\hf.exe download $model --max-workers 16 @tokenArgs
            }
        }
        Write-Host "`nAll models cached. Run .\make.ps1 to generate." -ForegroundColor Green
    }
```

- [ ] **Step 2: Smoke-test the command syntax (dry run)**

In a PS7 terminal, from the repo root:
```powershell
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"
.\.venv\Scripts\hf.exe download depth-anything/Depth-Anything-V2-Small-hf --max-workers 16 --dry-run
```

Expected: lists files that would be downloaded, no errors.

- [ ] **Update PENDING-TASK.md sub-task checkbox**

Mark Task 2 complete in PENDING-TASK.md.

---

### Task 3: Run the download and verify cache

**Files:** none — runtime verification only

- [ ] **Step 1: Set HF_TOKEN in your PS7 terminal**

```powershell
$env:HF_TOKEN = "hf_..."   # your read token
```

- [ ] **Step 2: Run the download target**

```powershell
.\make.ps1 -Target download
```

Expected: each model prints download progress at high speed (MB/s visible in terminal). Completes with:
```
All models cached. Run .\make.ps1 to generate.
```

- [ ] **Step 3: Verify snapshot dirs exist for all four models**

```powershell
@(
  "models--xinsir--controlnet-openpose-sdxl-1.0",
  "models--stabilityai--stable-diffusion-xl-base-1.0",
  "models--depth--anything--Depth-Anything-V2-Small-hf",
  "models--lllyasviel--Annotators"
) | ForEach-Object {
    $path = "D:\models\hub\$_\snapshots"
    if (Test-Path $path) { Write-Host "OK  $_" -ForegroundColor Green }
    else { Write-Host "MISSING  $_" -ForegroundColor Red }
}
```

Expected: all four lines print `OK`.

- [ ] **Update PENDING-TASK.md sub-task checkbox**

Mark Task 3 complete in PENDING-TASK.md.

---

### Task 4: Verify pipeline loads from local cache

**Files:** none — smoke test only

- [ ] **Step 1: Run the pipeline test**

```powershell
.\make.ps1 -Target test
```

Expected: pipeline loads models from `D:\models\hub` (no HF network calls), generates output images, opens gallery. No `ConnectionError` or download messages in the log.

- [ ] **Step 2: Confirm user is satisfied with the result**

Show the user the gallery output and confirm before proceeding to commit.

---

### Task 5: Commit and push

**Files:**
- `pyproject.toml`
- `uv.lock`
- `make.ps1`
- `docs/superpowers/specs/2026-05-11-fast-model-download-design.md`
- `docs/superpowers/plans/2026-05-11-fast-model-download.md`
- `.claude/pm/PENDING-TASK.md`

- [ ] **Step 1: Stage and commit**

```bash
git add pyproject.toml uv.lock make.ps1 docs/superpowers/ .claude/pm/PENDING-TASK.md
git commit -m "feat: fast model download via hf_transfer + hf CLI (#7)"
```

- [ ] **Step 2: Push**

```bash
git push
```

- [ ] **Step 3: Append to PROGRESS.md**

Use Edit to prepend (short anchor = first line of file):

```
## 2026-05-11 — Fast model download (hf_transfer)
- Installed hf_transfer Rust wheel via uv
- Updated make.ps1 download target: hf.exe, HF_HUB_ENABLE_HF_TRANSFER=1, --max-workers 16
- All four models downloaded to D:\models\hub
- Pipeline confirmed loading from local cache
- Committed: feat: fast model download via hf_transfer + hf CLI (#7)
```

- [ ] **Step 4: Clear PENDING-TASK.md**

Replace contents with `# PENDING TASK\n\n(none)`.
