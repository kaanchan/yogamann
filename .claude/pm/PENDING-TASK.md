# PENDING TASK

## Issue #25 — README usage guide

**Branch:** master  
**Approach:** Explore all scripts/flags, then write README.md at repo root covering full pipeline.

### Sub-tasks
- [x] Explore: make.ps1, batch.ps1, src/sd_make.py, src/make_mannequin.py, src/gallery.py, profiles/yoga_asana.yml
- [x] Write README.md with all sections from issue #25
- [ ] User tests / confirms content is accurate
- [ ] Commit referencing #25, push

## Issue #26 — multi-batch.ps1 multi-batch overnight runner

**Branch:** master  
**Approach:** New `multi-batch.ps1` with editable `$batches` array; fix SourceRoot resolution in `batch.ps1`.

### Sub-tasks
- [x] Fix `$SourceRoot` Resolve-Path in `batch.ps1`
- [x] Write `multi-batch.ps1`
- [x] Add multi-batch section to README.md
- [x] Dry-run tree view in `batch.ps1`
- [x] Per-task error resilience in `sd_make.py` (report + continue on bad file)
- [ ] User tests
- [ ] Commit referencing #26, push

## Up next (backlog)
- #17 — cond_scale sweep to fix head direction
- #19 — mark.ps1 / gold index  
- #14 — torch.compile cudagraphs
- #16 — OKS self-eval
- #18 — standard test set
