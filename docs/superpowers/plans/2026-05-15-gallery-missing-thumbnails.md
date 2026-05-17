# Gallery Missing Thumbnails Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix runs 957-981 showing no source thumbnail in the VLM comparison gallery by adding a self-healing disk-fallback to `db.py`, and close the completed #33 research issue.

**Architecture:** Add `get_or_create_thumbnail(conn, sha256)` to `db.py` — checks the `thumbnails` table first, falls back to loading from `source_images.path` on disk, writes the result back to `thumbnails` (lazy backfill), and returns bytes or `None`. `gallery.py` calls this instead of `get_thumbnail()`. The silent render-thumbnail exception at `gallery.py:1068` is replaced with a logged warning.

**Tech Stack:** Python 3.11+, SQLite (via `sqlite3`), Pillow (`Image`), `uv` as package manager. Tests run with `uv run pytest`. Gallery runs with `uv run streamlit run src/gallery.py -- --output-root D:/Temp/yogamann-output`. Shell: MINGW64/Git Bash.

---

## Pre-flight: Write PENDING-TASK.md

Before any agent starts, write `.claude/pm/PENDING-TASK.md` with the current branch, GH issues (#36, #33), approach, and checklist of tasks below. This must exist before the first tool call of any agent.

---

## Agent Split

Two agents run in parallel after PENDING-TASK.md is written:

| | Agent A | Agent B |
|---|---|---|
| **Task** | Tasks 1-5 — #36 code fix | Task 6 — #33 PM close |
| **Isolation** | `isolation: "worktree"` — branch `feature/gallery-missing-thumbnails` from `main` | None (PM-only, no code) |
| **Merges** | After user validates gallery visually | After user confirms content |

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `src/db.py` | Modify | Add `import logging`; add `get_or_create_thumbnail()` after `get_thumbnail()` |
| `src/gallery.py` | Modify | Add `import logging`; add `get_or_create_thumbnail` to import; update call site; fix silent except |
| `tests/test_db_thumbnail.py` | Create | Four unit tests for `get_or_create_thumbnail` |

---

## Task 1: Confirm root cause with DB investigation query

**Files:** None (investigation only — results posted to GH #36)

- [ ] **Step 1: Run the investigation query against the live DB**

  ```bash
  uv run python - <<'EOF'
  import sqlite3, sys
  conn = sqlite3.connect("D:/Temp/yogamann-output/yogamann.db")
  conn.row_factory = sqlite3.Row
  rows = conn.execute("""
      SELECT r.id,
             r.source_sha256,
             CASE WHEN t.sha256 IS NULL THEN 'MISSING' ELSE 'OK' END AS thumb_status
      FROM runs r
      LEFT JOIN thumbnails t ON t.sha256 = r.source_sha256
      WHERE r.id BETWEEN 957 AND 981
  """).fetchall()
  for r in rows:
      print(r["id"], r["thumb_status"], r["source_sha256"][:16])
  conn.close()
  EOF
  ```

  Expected: all 25 rows print `MISSING`. If any print `OK`, note which run IDs — those runs already have thumbnails and are not affected by this bug.

- [ ] **Step 2: Post investigation results as a comment on GH issue #36**

  Use `gh issue comment 36 --body "..."` with the query output pasted in. Example body:
  ```
  Root cause confirmed: all 25 runs 957-981 have source_sha256 values absent from
  the thumbnails table. source_images.path rows exist for all 25. Fix: add
  get_or_create_thumbnail() with lazy disk fallback.

  Query output:
  [paste output here]
  ```

---

## Task 2: Write failing tests for `get_or_create_thumbnail`

**Files:**
- Create: `tests/test_db_thumbnail.py`

- [ ] **Step 1: Create the test file**

  ```python
  # tests/test_db_thumbnail.py
  import pytest
  from pathlib import Path
  from PIL import Image
  from db import open_db, get_or_create_thumbnail


  def _make_jpeg(path: Path) -> None:
      img = Image.new("RGB", (200, 300), color=(128, 64, 32))
      img.save(path, format="JPEG")


  @pytest.fixture
  def conn(tmp_path):
      return open_db(tmp_path / "test.db")


  @pytest.fixture
  def conn_with_source(tmp_path):
      db = open_db(tmp_path / "test.db")
      img_path = tmp_path / "photo.jpg"
      _make_jpeg(img_path)
      db.execute("""
          INSERT INTO source_images (sha256, path, filename, added_at)
          VALUES ('deadbeef', ?, 'photo.jpg', '2026-01-01T00:00:00')
      """, (str(img_path),))
      db.commit()
      return db


  def test_fast_path_returns_existing_blob(conn_with_source):
      """When thumbnails table already has the sha256, return that blob directly."""
      blob = b"\xff\xd8\xff" + b"\x00" * 20  # fake JPEG header + padding
      conn_with_source.execute(
          "INSERT INTO thumbnails (sha256, data) VALUES ('deadbeef', ?)", (blob,)
      )
      conn_with_source.commit()

      result = get_or_create_thumbnail(conn_with_source, "deadbeef")

      assert result == blob


  def test_disk_fallback_returns_bytes_and_writes_to_db(conn_with_source):
      """When thumbnails table misses, load from disk, return bytes, and write back."""
      result = get_or_create_thumbnail(conn_with_source, "deadbeef")

      assert result is not None
      assert len(result) > 0

      # Verify it was written back to thumbnails table
      row = conn_with_source.execute(
          "SELECT data FROM thumbnails WHERE sha256='deadbeef'"
      ).fetchone()
      assert row is not None
      assert row["data"] == result


  def test_second_call_uses_db_not_disk(conn_with_source, tmp_path):
      """After first call populates thumbnails, second call uses the DB fast path."""
      first = get_or_create_thumbnail(conn_with_source, "deadbeef")
      # Delete the source file to prove second call does NOT hit disk
      (tmp_path / "photo.jpg").unlink()
      second = get_or_create_thumbnail(conn_with_source, "deadbeef")

      assert second == first


  def test_missing_sha256_returns_none(conn):
      """When sha256 is absent from both thumbnails and source_images, return None."""
      result = get_or_create_thumbnail(conn, "nosuchsha256")
      assert result is None


  def test_missing_file_on_disk_returns_none(conn):
      """When source_images has the sha256 but the file is gone, return None."""
      conn.execute("""
          INSERT INTO source_images (sha256, path, filename, added_at)
          VALUES ('deadbeef', '/nonexistent/missing.jpg', 'missing.jpg', '2026-01-01T00:00:00')
      """)
      conn.commit()

      result = get_or_create_thumbnail(conn, "deadbeef")
      assert result is None
  ```

- [ ] **Step 2: Run tests — expect ImportError or AttributeError (function does not exist yet)**

  ```bash
  uv run pytest tests/test_db_thumbnail.py -v
  ```

  Expected: all 5 tests fail with `ImportError: cannot import name 'get_or_create_thumbnail'`

---

## Task 3: Implement `get_or_create_thumbnail` in `db.py`

**Files:**
- Modify: `src/db.py`

- [ ] **Step 1: Add `import logging` to the imports block (line 17)**

  Current line 17:
  ```python
  import argparse, hashlib, io, json, sqlite3, sys
  ```
  Change to:
  ```python
  import argparse, hashlib, io, json, logging, sqlite3, sys
  ```

- [ ] **Step 2: Add `get_or_create_thumbnail` immediately after `get_thumbnail` (~line 445)**

  Find this block:
  ```python
  def get_thumbnail(conn: sqlite3.Connection, sha256: str) -> bytes | None:
      row = conn.execute("SELECT data FROM thumbnails WHERE sha256=?", (sha256,)).fetchone()
      return row["data"] if row else None
  ```

  Add immediately after it:
  ```python
  def get_or_create_thumbnail(conn: sqlite3.Connection, sha256: str) -> bytes | None:
      row = conn.execute("SELECT data FROM thumbnails WHERE sha256=?", (sha256,)).fetchone()
      if row:
          return row["data"]
      src = conn.execute("SELECT path FROM source_images WHERE sha256=?", (sha256,)).fetchone()
      if not src:
          logging.warning("get_or_create_thumbnail: no source_images row for sha256=%s", sha256)
          return None
      try:
          img = Image.open(src["path"]).convert("RGB")
          img.thumbnail((150, 150))
          buf = io.BytesIO()
          img.save(buf, format="JPEG", quality=80)
          data = buf.getvalue()
          conn.execute(
              "INSERT OR IGNORE INTO thumbnails (sha256, data) VALUES (?, ?)",
              (sha256, data),
          )
          conn.commit()
          return data
      except Exception as e:
          logging.warning("get_or_create_thumbnail failed for sha256=%s: %s", sha256, e)
          return None
  ```

- [ ] **Step 3: Run the tests — expect all 5 to pass**

  ```bash
  uv run pytest tests/test_db_thumbnail.py -v
  ```

  Expected output:
  ```
  tests/test_db_thumbnail.py::test_fast_path_returns_existing_blob PASSED
  tests/test_db_thumbnail.py::test_disk_fallback_returns_bytes_and_writes_to_db PASSED
  tests/test_db_thumbnail.py::test_second_call_uses_db_not_disk PASSED
  tests/test_db_thumbnail.py::test_missing_sha256_returns_none PASSED
  tests/test_db_thumbnail.py::test_missing_file_on_disk_returns_none PASSED
  5 passed
  ```

- [ ] **Step 4: Run the full test suite to check for regressions**

  ```bash
  uv run pytest -v
  ```

  Expected: all existing tests still pass.

- [ ] **Step 5: Commit**

  ```bash
  git add src/db.py tests/test_db_thumbnail.py
  git commit -m "feat: add get_or_create_thumbnail with lazy disk fallback (#36)"
  ```

---

## Task 4: Update `gallery.py`

**Files:**
- Modify: `src/gallery.py`

- [ ] **Step 1: Add `logging` to the imports block (line 15)**

  Current line 15:
  ```python
  import argparse, io, json, subprocess, sys, time
  ```
  Change to:
  ```python
  import argparse, io, json, logging, subprocess, sys, time
  ```

- [ ] **Step 2: Update the `db` import to include `get_or_create_thumbnail` (lines 24-26)**

  Current:
  ```python
  from db import (open_db, get_runs, get_runs_for_source, get_thumbnail, get_stats,
  ```
  Change to:
  ```python
  from db import (open_db, get_runs, get_runs_for_source, get_thumbnail, get_or_create_thumbnail, get_stats,
  ```

  (`get_thumbnail` stays — it may be used elsewhere; `get_or_create_thumbnail` is added alongside it.)

- [ ] **Step 3: Update the thumbnail map call site (~line 1058)**

  Find:
  ```python
  data = get_thumbnail(conn, vrow["source_sha256"])
  ```
  Change to:
  ```python
  data = get_or_create_thumbnail(conn, vrow["source_sha256"])
  ```

- [ ] **Step 4: Fix the silent render exception (~line 1068)**

  Find:
  ```python
          except Exception:
              pass
  ```
  (This is the `except` inside the `if rid not in render_map` block, under `rimg = Image.open(...)`)

  Change to:
  ```python
          except Exception as e:
              logging.warning("render thumbnail failed run %d: %s", rid, e)
  ```

- [ ] **Step 5: Verify the change with grep — confirm no remaining bare `except: pass` in the thumbnail loop**

  ```bash
  grep -n "except Exception" src/gallery.py
  ```

  Expected: the line now reads `except Exception as e:` with no bare `pass` following it.

- [ ] **Step 6: Commit**

  ```bash
  git add src/gallery.py
  git commit -m "fix: use get_or_create_thumbnail in gallery; log render failures (#36)"
  ```

---

## Task 5: Manual verification + push

- [ ] **Step 1: Start the Streamlit gallery**

  ```bash
  uv run streamlit run src/gallery.py -- --output-root D:/Temp/yogamann-output
  ```

- [ ] **Step 2: Navigate to the VLM comparison tab, page to runs 957-981**

  Confirm: source photo thumbnails are now visible for all 25 runs. Render thumbnails should also appear (output_png files confirmed present on disk).

- [ ] **Step 3: Reload the page and confirm thumbnails load instantly (DB fast path)**

  Second load should not hit disk — thumbnails are now in the `thumbnails` table. No perceptible slowdown vs. other pages.

- [ ] **Step 4: Check Python terminal for any `WARNING` lines**

  ```bash
  # Look at the Streamlit terminal output for lines matching:
  # WARNING:root:get_or_create_thumbnail failed...
  # WARNING:root:render thumbnail failed...
  ```

  If any warnings appear, note the sha256/run_id and investigate — they indicate either a moved/deleted source file or a PIL decode error.

- [ ] **Step 5: Hand off to user for visual confirmation**

  Stop here. The user validates the gallery visually before the branch is merged or the issue is closed. Do not merge, do not close issue #36.

---

## Task 6: Close issue #33 — Research PM (Agent B, parallel with Tasks 1-5)

**Files:** None (PM only — reads SYNTHESIS.md, posts to GH, closes issue)

- [ ] **Step 1: Read the SYNTHESIS.md content**

  File: `docs/research/issue-33-pose-pipeline-evaluation/SYNTHESIS.md`
  
  Read the full file to extract the content for the GH comment.

- [ ] **Step 2: Post findings to GH issue #33**

  ```bash
  gh issue comment 33 --body "$(cat docs/research/issue-33-pose-pipeline-evaluation/SYNTHESIS.md)"
  ```

- [ ] **Step 3: Close issue #33**

  ```bash
  gh issue close 33 --comment "Research complete. SYNTHESIS.md committed at docs/research/issue-33-pose-pipeline-evaluation/SYNTHESIS.md. No code changes — if a candidate warrants implementation, a new issue will be opened."
  ```

---

## Post-merge steps (after user validates #36)

1. Merge `feature/gallery-missing-thumbnails` → `main`
2. `git push origin main`
3. `gh issue close 36 --comment "Fixed: get_or_create_thumbnail added to db.py with lazy disk fallback. Confirmed runs 957-981 now show source thumbnails. Render exception now logged."`
4. Append completed work to `.claude/pm/PROGRESS.md`
5. Clear `.claude/pm/PENDING-TASK.md`
