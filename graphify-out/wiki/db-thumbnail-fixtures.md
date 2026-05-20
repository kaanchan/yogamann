# DB Thumbnail & Test Fixtures

**18 nodes · Community 8 · Cohesion 0.15**

`src/db.py` (connection + thumbnail layer) · `tests/test_db_thumbnail.py` · `tests/test_db_vlm.py`

---

## What it does

The lower half of the database module — connection management, thumbnail caching, and the test infrastructure that validates both. Separated from the high-level query layer by the graph because these functions form a tight cluster around a single responsibility: getting a database connection and getting a thumbnail blob.

## Key functions

| Function | Role |
|----------|------|
| `open_db(path)` | Opens (or creates) the SQLite database and runs schema migrations. Returns a connection. |
| `_get_conn()` | Module-level connection accessor — returns the cached connection or opens one. |
| `conn()` | Context manager for a DB connection (used in tests). |
| `conn_with_source(path)` | Like `conn()` but points to a specific DB file — used for test isolation. |
| `get_or_create_thumbnail(run_id, conn)` | Returns a JPEG thumbnail blob. Fast path: hit the `thumbnails` table. Slow path: load from disk, write back. |
| `_make_jpeg(img_path, size)` | Resizes and encodes an image as JPEG bytes. |

## Why `open_db()` is a god node

`open_db()` has 8 edges and high betweenness centrality — it connects Batch Orchestration, Contact Sheet Builder, Database Queries, and Single-Run Analyzer. Every subsystem that touches the DB goes through it. If schema migrations are ever needed, this is the single place to update them.

## Test fixtures

`tests/test_db_thumbnail.py` and `tests/test_db_vlm.py` use `conn_with_source()` to create throwaway in-memory or temp-file databases. The tests validate:
- Fast path returns existing blob without re-reading disk
- Slow path reads disk, caches, and returns identical bytes
- VLM annotation round-trip (write → read → verify)

## Connects to

- [Database — Run & Annotation Queries](database-queries.md) — shares `db.py`; this layer is called by the upper layer
- [Batch Orchestration](batch-orchestration.md) — `open_db()` called in `main()`
- [Review Gallery UI](review-gallery-ui.md) — `get_or_create_thumbnail()` called for gallery card images
- [Gallery Thumbnails Fix](gallery-thumbnails-fix.md) — the lazy disk fallback was the fix for missing thumbnails
