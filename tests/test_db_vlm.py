import json
import pytest
from pathlib import Path
from db import open_db, save_vlm_annotation, get_vlm_annotations, get_unanalyzed_runs


@pytest.fixture
def conn(tmp_path):
    return open_db(tmp_path / "test.db")


def _seed_run(conn):
    conn.execute("""
        INSERT INTO source_images (sha256, path, filename, added_at)
        VALUES ('abc123', '/tmp/photo.jpg', 'photo.jpg', '2026-01-01T00:00:00+00:00')
    """)
    conn.execute("""
        INSERT INTO runs (source_sha256, timestamp, output_png, metrics_json)
        VALUES ('abc123', '2026-01-01T00:00:00+00:00', '/tmp/render.png', '{}')
    """)
    conn.commit()
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def test_save_and_get_vlm_annotation(conn):
    run_id = _seed_run(conn)
    save_vlm_annotation(
        conn, run_id, "qwen2_5_vl_7b",
        rating="good",
        misaligned=["left arm"],
        unwanted_features=[],
        fail_patterns=[],
        notes="Pose matches well.",
        raw_output='{"rating":"good"}',
        latency_s=2.3,
    )
    rows = get_vlm_annotations(conn, run_id)
    assert len(rows) == 1
    assert rows[0]["rating"] == "good"
    assert json.loads(rows[0]["misaligned"]) == ["left arm"]
    assert rows[0]["latency_s"] == pytest.approx(2.3)


def test_upsert_replaces_existing(conn):
    run_id = _seed_run(conn)
    for rating in ("poor", "good"):
        save_vlm_annotation(
            conn, run_id, "qwen2_5_vl_7b",
            rating=rating, misaligned=[], unwanted_features=[],
            fail_patterns=[], notes="", raw_output="", latency_s=1.0,
        )
    rows = get_vlm_annotations(conn, run_id)
    assert len(rows) == 1
    assert rows[0]["rating"] == "good"


def test_two_models_two_rows(conn):
    run_id = _seed_run(conn)
    for model in ("qwen2_5_vl_7b", "minicpm_v_4_6"):
        save_vlm_annotation(
            conn, run_id, model,
            rating="acceptable", misaligned=[], unwanted_features=[],
            fail_patterns=[], notes="", raw_output="", latency_s=1.0,
        )
    rows = get_vlm_annotations(conn, run_id)
    assert len(rows) == 2
    assert {r["model_id"] for r in rows} == {"qwen2_5_vl_7b", "minicpm_v_4_6"}


def test_get_unanalyzed_runs_includes_unannotated(conn):
    run_id = _seed_run(conn)
    runs = get_unanalyzed_runs(conn, "qwen2_5_vl_7b")
    assert any(r["id"] == run_id for r in runs)


def test_get_unanalyzed_runs_excludes_annotated(conn):
    run_id = _seed_run(conn)
    save_vlm_annotation(
        conn, run_id, "qwen2_5_vl_7b",
        rating="good", misaligned=[], unwanted_features=[],
        fail_patterns=[], notes="", raw_output="", latency_s=1.0,
    )
    runs = get_unanalyzed_runs(conn, "qwen2_5_vl_7b")
    assert not any(r["id"] == run_id for r in runs)


def test_get_unanalyzed_runs_exposes_paths(conn):
    run_id = _seed_run(conn)
    runs = get_unanalyzed_runs(conn, "qwen2_5_vl_7b")
    row = next(r for r in runs if r["id"] == run_id)
    assert row["output_png"] == "/tmp/render.png"
    assert row["source_path"] == "/tmp/photo.jpg"
