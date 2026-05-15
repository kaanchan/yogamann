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
from datetime import datetime, timezone
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
    rating             TEXT,
    notes              TEXT,
    misaligned         TEXT,
    unwanted_features  TEXT,
    metrics_json       TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS thumbnails (
    sha256  TEXT PRIMARY KEY REFERENCES source_images(sha256),
    data    BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_source ON runs(source_sha256);
CREATE INDEX IF NOT EXISTS idx_runs_rating ON runs(rating);
CREATE INDEX IF NOT EXISTS idx_runs_ts     ON runs(timestamp);

CREATE TABLE IF NOT EXISTS users (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT UNIQUE NOT NULL,
    password  TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS annotations (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id             INTEGER NOT NULL REFERENCES runs(id),
    user_id            INTEGER NOT NULL REFERENCES users(id),
    rating             TEXT,
    notes              TEXT,
    misaligned         TEXT,
    unwanted_features  TEXT,
    observations       TEXT,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL,
    UNIQUE(run_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_ann_run    ON annotations(run_id);
CREATE INDEX IF NOT EXISTS idx_ann_user   ON annotations(user_id);
CREATE INDEX IF NOT EXISTS idx_ann_rating ON annotations(rating);

CREATE TABLE IF NOT EXISTS vlm_annotations (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            INTEGER NOT NULL REFERENCES runs(id),
    model_id          TEXT    NOT NULL,
    timestamp         TEXT    NOT NULL,
    rating            TEXT,
    misaligned        TEXT,
    unwanted_features TEXT,
    fail_patterns     TEXT,
    notes             TEXT,
    raw_output        TEXT,
    latency_s         REAL,
    UNIQUE(run_id, model_id)
);

CREATE INDEX IF NOT EXISTS idx_vlm_run   ON vlm_annotations(run_id);
CREATE INDEX IF NOT EXISTS idx_vlm_model ON vlm_annotations(model_id);
"""

# ── Connection ─────────────────────────────────────────────────────────────────
def open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_DDL)
    for col in ("misaligned TEXT", "unwanted_features TEXT", "observations TEXT"):
        try:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass
    # idempotent — ensures multi-user tables exist even on a cached pre-migration connection
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT UNIQUE NOT NULL,
            password  TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS annotations (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id             INTEGER NOT NULL REFERENCES runs(id),
            user_id            INTEGER NOT NULL REFERENCES users(id),
            rating             TEXT,
            notes              TEXT,
            misaligned         TEXT,
            unwanted_features  TEXT,
            observations       TEXT,
            created_at         TEXT NOT NULL,
            updated_at         TEXT NOT NULL,
            UNIQUE(run_id, user_id)
        );
        CREATE INDEX IF NOT EXISTS idx_ann_run    ON annotations(run_id);
        CREATE INDEX IF NOT EXISTS idx_ann_user   ON annotations(user_id);
        CREATE INDEX IF NOT EXISTS idx_ann_rating ON annotations(rating);
    """)
    # rename legacy option values
    conn.execute("""
        UPDATE runs SET unwanted_features = REPLACE(unwanted_features,
            'unspecified surfaces', 'textured/colored surfaces')
        WHERE unwanted_features LIKE '%unspecified surfaces%'
    """)
    # seed default user and migrate legacy run-level annotations
    conn.execute("INSERT OR IGNORE INTO users (name, password) VALUES ('satchid', 'satchid')")
    satchid_id = conn.execute("SELECT id FROM users WHERE name='satchid'").fetchone()[0]
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT OR IGNORE INTO annotations
            (run_id, user_id, rating, notes, misaligned, unwanted_features, observations, created_at, updated_at)
        SELECT r.id, ?, r.rating, r.notes, r.misaligned, r.unwanted_features, r.observations,
               COALESCE(r.timestamp, ?), ?
        FROM runs r
        WHERE r.rating IS NOT NULL
           OR (r.misaligned  IS NOT NULL AND r.misaligned  != '')
           OR (r.unwanted_features IS NOT NULL AND r.unwanted_features != '')
           OR (r.notes       IS NOT NULL AND r.notes       != '')
           OR (r.observations IS NOT NULL AND r.observations != '')
    """, (satchid_id, now, now))
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

# ── User helpers ───────────────────────────────────────────────────────────────
def get_users(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT id, name FROM users ORDER BY name").fetchall()

def get_or_create_user(conn: sqlite3.Connection, name: str, password: str) -> sqlite3.Row | None:
    """Return the user row if name+password match, else None."""
    row = conn.execute("SELECT * FROM users WHERE name=?", (name,)).fetchone()
    if row is None:
        conn.execute("INSERT INTO users (name, password) VALUES (?, ?)", (name, password))
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE name=?", (name,)).fetchone()
    return row if row["password"] == password else None

# ── Annotation write ────────────────────────────────────────────────────────────
def save_annotation(
    conn: sqlite3.Connection,
    run_id: int,
    user_id: int,
    rating: str | None,
    notes: str,
    misaligned: list[str] | None = None,
    unwanted_features: list[str] | None = None,
    observations: list[str] | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO annotations
            (run_id, user_id, rating, notes, misaligned, unwanted_features, observations, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, user_id) DO UPDATE SET
            rating=excluded.rating, notes=excluded.notes,
            misaligned=excluded.misaligned, unwanted_features=excluded.unwanted_features,
            observations=excluded.observations, updated_at=excluded.updated_at
    """, (
        run_id, user_id,
        rating or None,
        notes.strip() or None,
        ",".join(misaligned) if misaligned else None,
        ",".join(unwanted_features) if unwanted_features else None,
        ",".join(observations) if observations else None,
        now, now,
    ))
    conn.commit()

def get_annotation(conn: sqlite3.Connection, run_id: int, user_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM annotations WHERE run_id=? AND user_id=?", (run_id, user_id)
    ).fetchone()

def get_all_annotations(conn: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
    """All users' annotations for a run, joined with user name."""
    return conn.execute("""
        SELECT a.*, u.name as reviewer
        FROM annotations a JOIN users u ON a.user_id=u.id
        WHERE a.run_id=?
        ORDER BY u.name
    """, (run_id,)).fetchall()

# ── Legacy rating write (ingest path — writes to runs columns) ──────────────────
def save_rating(
    conn: sqlite3.Connection,
    run_id: int,
    rating: str | None,
    notes: str,
    misaligned: list[str] | None = None,
    unwanted_features: list[str] | None = None,
    observations: list[str] | None = None,
) -> None:
    conn.execute(
        "UPDATE runs SET rating=?, notes=?, misaligned=?, unwanted_features=?, observations=? WHERE id=?",
        (
            rating or None,
            notes.strip() or None,
            ",".join(misaligned) if misaligned else None,
            ",".join(unwanted_features) if unwanted_features else None,
            ",".join(observations) if observations else None,
            run_id,
        ),
    )
    conn.commit()

# ── Queries ────────────────────────────────────────────────────────────────────
# State filters reference the per-user annotation join (a.*); a.id IS NULL = unvisited
_STATE_SQL = {
    "Unvisited":            "a.id IS NULL",
    "Updated":              "a.id IS NOT NULL",
    "Rating unspecified":   "(a.rating IS NULL OR a.rating='')",
    "Unwanted unspecified": "(a.unwanted_features IS NULL OR a.unwanted_features='')",
    "Notes blank":          "(a.notes IS NULL OR a.notes='')",
}

def get_runs(
    conn: sqlite3.Connection,
    *,
    user_id: int = 1,
    rating_filter: list[str] | None = None,
    state_filter: list[str] | None = None,
    newest_first: bool = True,
    limit: int = 0,
) -> list[sqlite3.Row]:
    wheres: list[str] = []
    params: list[Any] = [user_id]   # first param is always the LEFT JOIN user_id

    if rating_filter:
        unrated = "Unrated" in rating_filter
        named   = [r for r in rating_filter if r != "Unrated"]
        if unrated and named:
            wheres.append(f"(a.rating IS NULL OR a.rating IN ({','.join('?'*len(named))}))")
            params.extend(named)
        elif unrated:
            wheres.append("a.rating IS NULL")
        else:
            wheres.append(f"a.rating IN ({','.join('?'*len(named))})")
            params.extend(named)

    for s in (state_filter or []):
        if s in _STATE_SQL:
            wheres.append(_STATE_SQL[s])

    sql = """
        SELECT r.id, r.source_sha256, r.timestamp, r.profile, r.seed, r.steps,
               r.guidance, r.cond_scale, r.width, r.height, r.output_png,
               r.prompt, r.neg_prompt, r.pose_s, r.gen_s, r.total_s, r.metrics_json,
               s.path as source_path,
               COALESCE(a.rating,            r.rating)            as rating,
               COALESCE(a.notes,             r.notes)             as notes,
               COALESCE(a.misaligned,        r.misaligned)        as misaligned,
               COALESCE(a.unwanted_features, r.unwanted_features) as unwanted_features,
               COALESCE(a.observations,      r.observations)      as observations,
               a.id as annotation_id
        FROM runs r
        JOIN source_images s ON r.source_sha256=s.sha256
        LEFT JOIN annotations a ON a.run_id=r.id AND a.user_id=?
    """
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

# ── VLM annotation helpers ─────────────────────────────────────────────────────
def save_vlm_annotation(
    conn: sqlite3.Connection,
    run_id: int,
    model_id: str,
    rating: str,
    misaligned: list[str] | None,
    unwanted_features: list[str] | None,
    fail_patterns: list[str] | None,
    notes: str,
    raw_output: str,
    latency_s: float,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO vlm_annotations
            (run_id, model_id, timestamp, rating, misaligned, unwanted_features,
             fail_patterns, notes, raw_output, latency_s)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, model_id) DO UPDATE SET
            timestamp=excluded.timestamp,
            rating=excluded.rating,
            misaligned=excluded.misaligned,
            unwanted_features=excluded.unwanted_features,
            fail_patterns=excluded.fail_patterns,
            notes=excluded.notes,
            raw_output=excluded.raw_output,
            latency_s=excluded.latency_s
    """, (
        run_id, model_id, now, rating,
        # VLM list columns stored as JSON (not CSV like annotations table) — supports values with commas
        json.dumps(misaligned) if misaligned is not None else None,
        json.dumps(unwanted_features) if unwanted_features is not None else None,
        json.dumps(fail_patterns) if fail_patterns is not None else None,
        notes, raw_output, latency_s,
    ))
    conn.commit()


def get_vlm_annotations(conn: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM vlm_annotations WHERE run_id=? ORDER BY model_id",
        (run_id,),
    ).fetchall()


def count_vlm_annotations(conn: sqlite3.Connection, model_id: str) -> int:
    """Return the number of vlm_annotations rows for a model — used to
    report 'already done, skipping' counts at the start of a batch."""
    row = conn.execute(
        "SELECT COUNT(*) FROM vlm_annotations WHERE model_id = ?",
        (model_id,),
    ).fetchone()
    return row[0] if row else 0


def get_unanalyzed_runs(
    conn: sqlite3.Connection,
    model_id: str,
    limit: int = 0,
) -> list[sqlite3.Row]:
    """Returns runs with no vlm_annotations entry for model_id.
    Each row exposes: id, output_png, source_path (from source_images.path).
    """
    q = """
        SELECT r.id, r.output_png, si.path as source_path
        FROM runs r
        JOIN source_images si ON r.source_sha256 = si.sha256
        WHERE NOT EXISTS (
            SELECT 1 FROM vlm_annotations va
            WHERE va.run_id = r.id AND va.model_id = ?
        )
        ORDER BY r.timestamp DESC
    """
    if limit:
        q += f" LIMIT {int(limit)}"
    return conn.execute(q, (model_id,)).fetchall()


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
