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
- [x] Task 0: Write PENDING-TASK.md
- [ ] Task 1: Install hf_transfer via uv
- [ ] Task 2: Update make.ps1 download target
- [ ] Task 3: Run download, verify cache
- [ ] Task 4: Verify pipeline loads from local cache
- [ ] Task 5: Commit and push

## Constraints
- HF_TOKEN must be set in PS session before running download
- HF_HUB_CACHE=D:\models\hub must remain authoritative (no --cache-dir flag)
- Download target only — no pipeline code changes
