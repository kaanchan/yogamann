# `.claude/tmp/` - agent scratch space

Rules: [tmp-folder-protocol](https://github.com/kaanchan/mooladhara/blob/main/standards/tmp-folder-protocol.md) (hub standard).
This README is the one tracked file here; everything else in this folder is git-ignored.

## Purpose
Working space for Claude sessions and agents in this repo: logs, staging, intermediate files, trial outputs. Nothing here is a
source of truth. Anything worth keeping is committed elsewhere (for example `r&d/<issue#>-<slug>/` or `reports/`) before the folder is deleted.

## Who creates content
Claude sessions and background agents. Each folder belongs to the session that made it; nobody maintains this by hand.

## Layout - no stray files
- Only this README (and `.gitkeep`) sit at the top level. Every other file lives in a folder named `<issue#>-<slug>/`, so the GitHub
  issue shows in the path: `933-gate-logs/`, `118-epub-trial/`.
- Work with no issue: `noissue-<yyyymmdd>-<slug>/`. File the issue (and rename the folder) or delete it within 14 days.
- No clones, virtual environments or artefacts over ~500 MB here: put them on `D:\` or delete them at unit close.

## Expiry
- Idle 14 days (no file inside changed) = stale. The disk-audit tools list it.
- Idle 30 days and not named in `.claude/pm/PENDING-TASK.md` = deletable.
- To keep a folder longer, put a `.keep` file inside it: one line with the reason and a review date (`yyyy-mm-dd`). The exemption
  ends on the review date.

## Cleanup request
- The session that made a folder deletes it at unit close, after anything worth keeping is committed.
- Found something stale that is not yours? Do not delete it silently (nothing here has a git copy): list it in your report so KA can
  confirm, or add a `.keep` with a reason.
