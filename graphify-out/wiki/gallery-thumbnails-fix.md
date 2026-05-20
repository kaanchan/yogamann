# Gallery Missing Thumbnails Fix

**3 nodes · Community 23 · Cohesion 1.00**

`docs/superpowers/plans/2026-05-15-gallery-missing-thumbnails.md` · `docs/superpowers/specs/2026-05-15-gallery-missing-thumbnails-design.md`

---

## What it is

The plan and spec for the fix that introduced `get_or_create_thumbnail()` — a lazy disk fallback for gallery cards whose thumbnails weren't in the DB.

## The problem

The gallery was showing blank cards for runs where the thumbnail had never been generated into the DB. This happened for runs ingested from `.metrics.json` sidecars before the thumbnail generation step ran.

## The fix

`get_or_create_thumbnail(run_id, conn)` now follows a two-step path:
1. **Fast path:** Check the `thumbnails` table — if the SHA-256 hash of the source image matches a stored blob, return it immediately
2. **Slow path:** If not found, load the source image from disk, generate the JPEG thumbnail, write it back to the DB, and return it

This means the gallery self-heals on first render — missing thumbnails are generated on demand rather than showing blank cards.

## Connects to

- [DB Thumbnail & Test Fixtures](db-thumbnail-fixtures.md) — `get_or_create_thumbnail()` lives there
- [Review Gallery UI](review-gallery-ui.md) — the gallery that was showing blank cards
