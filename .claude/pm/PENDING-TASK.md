# PENDING TASK

**Branch:** master  
**Last commit:** 4d6a580  
**Session date:** 2026-05-11

## GH Issues being addressed

| # | Title |
|---|-------|
| [#2](https://github.com/kaanchan/yogamann/issues/2) | Bug: make_mannequin.py uses wrong key `OUTPUT_DIR_png` → should be `output_png` |
| [#3](https://github.com/kaanchan/yogamann/issues/3) | Feature: HTML gallery generator `src/make_gallery.py` |
| [#4](https://github.com/kaanchan/yogamann/issues/4) | Docs: Add run instructions to README.md |

## Agreed approach

- Fix the key bug first (one-liner, lowest risk)
- Add `src/make_gallery.py` — standalone script, zero new dependencies, vanilla HTML/CSS/JS output
- Update README.md with usage instructions covering single-image, batch, and gallery commands
- Each change gets its own commit referencing its issue number
- User tests between fix and gallery addition

## Sub-tasks

- [x] Set up `.claude/pm/` and `.claude/tmp/` directories
- [x] Add `.claude/tmp/*` to `.gitignore`
- [x] Create GH issues #2, #3, #4
- [x] Write PENDING-TASK.md
- [ ] Fix `OUTPUT_DIR_png` → `output_png` in `src/make_mannequin.py` line 126 — closes #2
- [ ] Add `src/make_gallery.py` — closes #3
- [ ] Update `README.md` — closes #4
- [ ] Commit each change with issue reference
- [ ] Push to remote (main/master)
- [ ] Write PROGRESS.md entry

## Constraints / decisions

- Gallery uses only stdlib + Pillow (already a project dep) — no new pip installs
- Gallery reads PNG metadata (`tEXt` chunks written by sd_make.py / create_contact_sheet.py) to populate card labels
- Images referenced by relative path so the `out/` folder is self-contained and portable
- No server required — plain `file://` works

## Open questions

- None currently
