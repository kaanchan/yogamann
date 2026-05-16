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
    blob = b"\xff\xd8\xff" + b"\x00" * 20
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

    row = conn_with_source.execute(
        "SELECT data FROM thumbnails WHERE sha256='deadbeef'"
    ).fetchone()
    assert row is not None
    assert row["data"] == result


def test_second_call_uses_db_not_disk(conn_with_source, tmp_path):
    """After first call populates thumbnails, second call uses the DB fast path."""
    first = get_or_create_thumbnail(conn_with_source, "deadbeef")
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
