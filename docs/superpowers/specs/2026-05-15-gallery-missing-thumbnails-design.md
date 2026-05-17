# Design: Gallery Missing Thumbnails — Issue #36 + #33 PM Close

**Date**: 2026-05-15  
**Issues**: [#36](https://github.com/kaanchan/yogamann/issues/36) (gallery bug), [#33](https://github.com/kaanchan/yogamann/issues/33) (research close)  
**Branch**: feature/gallery-missing-thumbnails (new, from main)

---

## Problem

Runs 957-981 show annotation text in the Streamlit VLM comparison gallery but no source photo thumbnail. The render thumbnail column is also blank for these runs.

### Root Cause

`get_thumbnail(conn, sha256)` in `db.py` returns `None` when the sha256 is absent from the `thumbnails` table. The `thumbnails` table was not populated for these 25 runs (likely a gap during an ingest session).

The render map has a silent `except Exception: pass` at `gallery.py:1068` that swallows any render-thumbnail failure without logging.

Note: `_render_vlm_table` already has correct None-guards (lines 162-165) — it shows a grey placeholder when `src_b64` is None, so there is no `base64,None` rendering bug in the current code.

---

## Design

### Scope

| File | Change |
|------|--------|
| `src/db.py` | Add `get_or_create_thumbnail()` after existing `get_thumbnail()` |
| `src/gallery.py` | Update import + call site + fix silent except |

No schema changes. No new tables. No startup-time repair pass.

---

### 1. `src/db.py` — `get_or_create_thumbnail(conn, sha256)`

Placed immediately after `get_thumbnail()` (~line 446).

**Signature**: `get_or_create_thumbnail(conn: sqlite3.Connection, sha256: str) -> bytes | None`

**Logic**:

1. `SELECT data FROM thumbnails WHERE sha256=?` — return `bytes` on hit (fast path, same as `get_thumbnail`)
2. On miss: `SELECT path FROM source_images WHERE sha256=?`
3. If no source row: log a warning, return `None` (orphaned sha256, unrecoverable)
4. `Image.open(path).convert("RGB")` → `.thumbnail((150, 150))` → save to `io.BytesIO` as JPEG `quality=80`
5. `INSERT OR IGNORE INTO thumbnails (sha256, data) VALUES (?, ?)` + `conn.commit()`
   - `INSERT OR IGNORE` is safe under concurrent gallery sessions: one insert wins, the other no-ops silently
6. Return the bytes
7. On any exception in steps 4-5: `logging.warning("get_or_create_thumbnail failed %s: %s", sha256, e)`, return `None`

**Imports needed in db.py**: `io` (already imported), `logging` (add if absent), `Image` from PIL (already imported).

---

### 2. `src/gallery.py` — Three-line change

**Import** (line 24-26): add `get_or_create_thumbnail` to the import from `db`.

**Call site** (~line 1058): 
```python
# Before:
data = get_thumbnail(conn, vrow["source_sha256"])
# After:
data = get_or_create_thumbnail(conn, vrow["source_sha256"])
```

**Render exception** (~line 1068):
```python
# Before:
except Exception:
    pass
# After:
except Exception as e:
    logging.warning("render thumbnail failed run %d: %s", rid, e)
```

---

### 3. Pre-fix investigation (agent runs this first)

Before writing any code, the agent runs this query against the live DB at
`D:/Temp/yogamann-output/yogamann.db`:

```sql
SELECT r.id,
       r.source_sha256,
       CASE WHEN t.sha256 IS NULL THEN 'MISSING' ELSE 'OK' END AS thumb_status
FROM runs r
LEFT JOIN thumbnails t ON t.sha256 = r.source_sha256
WHERE r.id BETWEEN 957 AND 981;
```

Expected result: all 25 rows show `MISSING`. The agent posts this output as a comment
on GH issue #36 to confirm the root cause before patching.

---

## Issue #33 — PM Close

SYNTHESIS.md already exists at
`docs/research/issue-33-pose-pipeline-evaluation/SYNTHESIS.md`.

The only remaining action: post the SYNTHESIS.md content to GH issue #33 as a
comment, then close the issue. No code changes.

---

## Agent Orchestration

Two agents run in parallel after PENDING-TASK.md is written:

| Agent | Task | Isolation |
|-------|------|-----------|
| **Agent A** | Investigate DB → implement `get_or_create_thumbnail` + gallery.py changes | `worktree` |
| **Agent B** | Post SYNTHESIS.md to GH issue #33, close issue | none (PM only) |

Agent A does not close GH issue #36 or merge its branch — user validates first.

---

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| sha256 absent from both thumbnails and source_images | `logging.warning`, return `None`, gallery shows grey placeholder |
| Source file path exists in DB but file missing from disk | PIL raises, caught, `logging.warning`, return `None` |
| Concurrent gallery sessions backfilling same sha256 | `INSERT OR IGNORE` — one wins, other no-ops, both return bytes |
| render thumbnail PIL failure | `logging.warning` with run_id and exception; gallery shows no render image |

---

## Testing

1. After merge: reload the gallery VLM tab and confirm runs 957-981 now show source thumbnails.
2. Check Python logs for any `get_or_create_thumbnail failed` or `render thumbnail failed` warnings.
3. Re-load the page — second load should hit the thumbnails table (fast path), not disk.
4. User confirms visually before issue #36 is closed.
