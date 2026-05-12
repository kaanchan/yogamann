"""
db.py — SQLite revision-history store for yogamann.

Schema
------
  source_images  one row per unique input photo (keyed by SHA-256)
  runs           one row per generation attempt
  thumbnails     150×150 JPEG blob per source image

DB location: {output_root}/yogamann.db

CLI (ingest existing .metrics.json files):
    python src/db.py --ingest --output-root D:/Temp/yogamann-output
"""
from __future__ import annotations

import argparse, hashlib, io, json, sqlite3, sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

# ── Schema ─────────────────────────────────────────────────────────────────────
_DDL = """
CREATE TABLE IF NOT EXISTS source_images (
    sha256    TEXT PRIMARY KEY,
    path      TEXT NOT NULL,
    filename  TEXT NOT NULL,
    width     INTEGER,
    height    INTEGER,
    added_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_sha256 TEXT    NOT NULL REFERENCES source_images(sha256),
    timestamp     TEXT    NOT NULL,
    profile       TEXT,
    seed          INTEGER,
    steps         INTEGER,
    guidance      REAL,
    cond_scale    REAL,
    width         INTEGER,
    height        INTEGER,
    output_png    TEXT    NOT NULL,
    prompt        TEXT,
    neg_prompt    TEXT,
    pose_s        REAL,
    gen_s         REAL,
    total_s       REAL,
    rating        TEXT,
    notes         TEXT,
    metrics_json  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS thumbnails (
    sha256  TEXT PRIMARY KEY REFERENCES source_images(sha256),
    data    BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_source ON runs(source_sha256);
CREATE INDEX IF NOT EXISTS idx_runs_rating ON runs(rating);
CREATE INDEX IF NOT EXISTS idx_runs_ts     ON runs(timestamp);
"""

# ── Connection ─────────────────────────────────────────────────────────────────
def open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_DDL)
    conn.commit()
    return conn

# ── Helpers ────────────────────────────────────────────────────────────────────
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def make_thumbnail(img: Image.Image, size: tuple[int, int] = (150, 150)) -> bytes:
    """Return JPEG bytes of a thumbnail, preserving aspect ratio."""
    thumb = img.copy()
    thumb.thumbnail(size, Image.LANCZOS)
    buf = io.BytesIO()
    thumb.convert("RGB").save(buf, format="JPEG", quality=75)
    return buf.getvalue()

# ── Ingest one .metrics.json ───────────────────────────────────────────────────
def ingest_json(conn: sqlite3.Connection, json_path: Path) -> int | None:
    """
    Read a .metrics.json sidecar and upsert into the DB.
    Returns the run id, or None if skipped.
    """
    try:
        m = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[skip] {json_path.name}: {e}", file=sys.stderr)
        return None

    input_photo = Path(m.get("input_photo", ""))
    output_png  = m.get("output_png", "")

    if not input_photo.exists():
        print(f"[skip] input not found: {input_photo}", file=sys.stderr)
        return None

    # Source image identity
    sha = m.get("source_sha256") or sha256_file(input_photo)
    try:
        img = ImageOps.exif_transpose(Image.open(input_photo)).convert("RGB")
        w, h = img.size
    except Exception:
        w = h = None
        img = None

    # Upsert source_images (update path if it moved)
    conn.execute("""
        INSERT INTO source_images (sha256, path, filename, width, height, added_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(sha256) DO UPDATE SET path=excluded.path
    """, (sha, str(input_photo), input_photo.name, w, h,
          m.get("timestamp", "")))

    # Thumbnail (only if not already stored)
    if img is not None:
        existing = conn.execute(
            "SELECT 1 FROM thumbnails WHERE sha256=?", (sha,)
        ).fetchone()
        if not existing:
            conn.execute(
                "INSERT OR IGNORE INTO thumbnails (sha256, data) VALUES (?, ?)",
                (sha, make_thumbnail(img)),
            )

    timing = m.get("timing", {})

    # Check for duplicate run (same output_png path)
    existing_run = conn.execute(
        "SELECT id, rating, notes FROM runs WHERE output_png=?", (output_png,)
    ).fetchone()

    if existing_run:
        # Preserve existing rating/notes; update metrics_json
        conn.execute("""
            UPDATE runs SET metrics_json=?, source_sha256=?, timestamp=?,
                profile=?, seed=?, steps=?, guidance=?, cond_scale=?,
                width=?, height=?, prompt=?, neg_prompt=?,
                pose_s=?, gen_s=?, total_s=?
            WHERE output_png=?
        """, (
            json.dumps(m),
            sha, m.get("timestamp", ""),
            m.get("profile"), m.get("seed"), m.get("steps"),
            m.get("guidance"), m.get("cond_scale"),
            m.get("width"), m.get("height"),
            m.get("prompt"), m.get("neg_prompt"),
            timing.get("pose_s"), timing.get("gen_s"), timing.get("total_s"),
            output_png,
        ))
        run_id = existing_run["id"]
    else:
        cur = conn.execute("""
            INSERT INTO runs (
                source_sha256, timestamp, profile, seed, steps, guidance,
                cond_scale, width, height, output_png, prompt, neg_prompt,
                pose_s, gen_s, total_s, rating, notes, metrics_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            sha, m.get("timestamp", ""),
            m.get("profile"), m.get("seed"), m.get("steps"),
            m.get("guidance"), m.get("cond_scale"),
            m.get("width"), m.get("height"),
            output_png,
            m.get("prompt"), m.get("neg_prompt"),
            timing.get("pose_s"), timing.get("gen_s"), timing.get("total_s"),
            m.get("rating"), m.get("notes"),
            json.dumps(m),
        ))
        run_id = cur.lastrowid

    conn.commit()
    return run_id

# ── Bulk ingest ────────────────────────────────────────────────────────────────
def ingest_all(conn: sqlite3.Connection, root: Path) -> tuple[int, int]:
    """Walk root for *.metrics.json and ingest each. Returns (ok, skipped)."""
    ok = skipped = 0
    for p in sorted(root.rglob("*.metrics.json")):
        run_id = ingest_json(conn, p)
        if run_id is not None:
            ok += 1
        else:
            skipped += 1
    return ok, skipped

# ── Rating write ───────────────────────────────────────────────────────────────
def save_rating(conn: sqlite3.Connection, run_id: int, rating: str | None, notes: str) -> None:
    conn.execute(
        "UPDATE runs SET rating=?, notes=? WHERE id=?",
        (rating or None, notes.strip() or None, run_id),
    )
    conn.commit()

# ── Queries ────────────────────────────────────────────────────────────────────
def get_runs(
    conn: sqlite3.Connection,
    *,
    rating_filter: list[str] | None = None,
    newest_first: bool = True,
    limit: int = 0,
) -> list[sqlite3.Row]:
    wheres: list[str] = []
    params: list[Any] = []

    if rating_filter:
        unrated = "Unrated" in rating_filter
        named   = [r for r in rating_filter if r != "Unrated"]
        if unrated and named:
            wheres.append(f"(rating IS NULL OR rating IN ({','.join('?'*len(named))}))")
            params.extend(named)
        elif unrated:
            wheres.append("rating IS NULL")
        else:
            wheres.append(f"rating IN ({','.join('?'*len(named))})")
            params.extend(named)

    sql = "SELECT r.*, s.path as source_path FROM runs r JOIN source_images s ON r.source_sha256=s.sha256"
    if wheres:
        sql += " WHERE " + " AND ".join(wheres)
    sql += f" ORDER BY r.timestamp {'DESC' if newest_first else 'ASC'}"
    if limit:
        sql += f" LIMIT {int(limit)}"

    return conn.execute(sql, params).fetchall()

def get_runs_for_source(conn: sqlite3.Connection, sha256: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM runs WHERE source_sha256=? ORDER BY timestamp DESC", (sha256,)
    ).fetchall()

def get_thumbnail(conn: sqlite3.Connection, sha256: str) -> bytes | None:
    row = conn.execute("SELECT data FROM thumbnails WHERE sha256=?", (sha256,)).fetchone()
    return row["data"] if row else None

def get_stats(conn: sqlite3.Connection) -> dict:
    total  = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    rated  = conn.execute("SELECT COUNT(*) FROM runs WHERE rating IS NOT NULL").fetchone()[0]
    gold   = conn.execute("SELECT COUNT(*) FROM runs WHERE rating='Gold'").fetchone()[0]
    sources = conn.execute("SELECT COUNT(*) FROM source_images").fetchone()[0]
    return {"total": total, "rated": rated, "gold": gold, "sources": sources}

# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="yogamann DB tool")
    ap.add_argument("--output-root", default="D:/Temp/yogamann-output")
    ap.add_argument("--ingest", action="store_true", help="Import all .metrics.json into DB")
    ap.add_argument("--stats",  action="store_true", help="Print DB stats")
    args = ap.parse_args()

    root   = Path(args.output_root)
    db_path = root / "yogamann.db"
    root.mkdir(parents=True, exist_ok=True)

    conn = open_db(db_path)
    print(f"DB: {db_path}")

    if args.ingest:
        print(f"Scanning {root} …")
        ok, skipped = ingest_all(conn, root)
        print(f"Ingested: {ok}   Skipped: {skipped}")

    if args.stats or not args.ingest:
        s = get_stats(conn)
        print(f"Sources: {s['sources']}  Runs: {s['total']}  "
              f"Rated: {s['rated']}  Gold: {s['gold']}")

    conn.close()
