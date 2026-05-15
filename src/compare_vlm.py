"""src/compare_vlm.py — multi-model VLM annotation orchestrator.

Model-major loop: load model -> annotate N images in micro-batches -> evict.
DB-driven resumability: skips (run_id, model_id) pairs already in vlm_annotations
unless --force. Each annotation auto-commits via save_vlm_annotation, so a
killed process resumes precisely from the last completed image.

Usage:
    # Annotate all runs with every model in vlm.yml (4-model comparison):
    python src/compare_vlm.py --limit 0 --output-root D:/Temp/yogamann-output

    # Annotate specific runs with one model:
    python src/compare_vlm.py --run-ids 981 980 --models qwen2_5_vl_7b

    # Force re-annotate even pairs already in DB:
    python src/compare_vlm.py --limit 50 --models qwen2_5_vl_7b --force

    # Larger micro-batch for memory headroom check:
    python src/compare_vlm.py --limit 200 --batch-size 50
"""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import torch
import yaml

from _batch_utils import chunked, format_summary
from db import (
    count_vlm_annotations,
    get_unanalyzed_runs,
    open_db,
    save_vlm_annotation,
)
from vlm_inference import annotate, evict_model, VLMSchemaError


def _fetch_run_candidates(conn, run_ids: list[int] | None, limit: int) -> list:
    """Return the candidate run set the user is filtering to.

    Per-model resumability filtering happens inside the model phase.
    """
    if run_ids:
        placeholders = ",".join("?" * len(run_ids))
        return conn.execute(
            f"""SELECT r.id, r.output_png, si.path as source_path
                FROM runs r JOIN source_images si ON r.source_sha256 = si.sha256
                WHERE r.id IN ({placeholders})""",
            run_ids,
        ).fetchall()
    q = """SELECT r.id, r.output_png, si.path as source_path
           FROM runs r JOIN source_images si ON r.source_sha256 = si.sha256
           ORDER BY r.timestamp DESC"""
    if limit:
        q += f" LIMIT {int(limit)}"
    return conn.execute(q).fetchall()


def _run_model_phase(
    conn,
    model_key: str,
    candidates: list,
    batch_size: int,
    force: bool,
) -> dict:
    """Annotate every candidate run with one model, in micro-batches.

    Returns {'ok': N, 'error': M, 'skipped': K, 'elapsed_s': T}.
    """
    if force:
        todo = candidates
        skipped = 0
    else:
        unanalyzed_ids = {
            r["id"] for r in get_unanalyzed_runs(conn, model_key)
        }
        todo = [r for r in candidates if r["id"] in unanalyzed_ids]
        skipped = len(candidates) - len(todo)

    print(
        f"\n=== {model_key} ===\n"
        f"  candidates : {len(candidates)}\n"
        f"  already done: {skipped}\n"
        f"  to process : {len(todo)}\n"
        f"  micro-batch: {batch_size}"
    )

    stats = {"ok": 0, "error": 0, "skipped": skipped, "elapsed_s": 0.0}
    if not todo:
        print(f"  [{model_key}] nothing to do.")
        return stats

    t_start = time.perf_counter()
    for batch_idx, batch in enumerate(chunked(todo, batch_size), start=1):
        for run in batch:
            try:
                result = annotate(
                    Path(run["source_path"]),
                    Path(run["output_png"]),
                    model_key=model_key,
                )
                save_vlm_annotation(
                    conn, run["id"], model_key,
                    rating=result["rating"],
                    misaligned=result["misaligned"],
                    unwanted_features=result["unwanted_features"],
                    fail_patterns=result["fail_patterns"],
                    notes=result["notes"],
                    raw_output=result["raw_output"],
                    latency_s=result["latency_s"],
                )
                stats["ok"] += 1
            except (FileNotFoundError, VLMSchemaError) as exc:
                print(f"  [skip] run {run['id']} {model_key}: {exc}")
                stats["error"] += 1
            except Exception as exc:
                print(f"  [error] run {run['id']} {model_key}: {exc}")
                traceback.print_exc()
                stats["error"] += 1

        # End of micro-batch — release transient memory + log progress
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        done = stats["ok"] + stats["error"]
        print(
            f"  [{model_key}] batch {batch_idx}: "
            f"{stats['ok']} ok, {stats['error']} err, "
            f"{len(todo) - done} remain"
        )

    stats["elapsed_s"] = time.perf_counter() - t_start
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare VLM pose analysis across models (model-major batch orchestrator)"
    )
    parser.add_argument("--run-ids", nargs="+", type=int, help="Specific run IDs")
    parser.add_argument("--limit", type=int, default=5,
                        help="N most recent runs (0=all). Default 5.")
    parser.add_argument("--output-root", default=r"D:\Temp\yogamann-output")
    parser.add_argument("--models", nargs="+",
                        help="Model keys (default: all enabled in vlm.yml)")
    parser.add_argument("--force", action="store_true",
                        help="Re-annotate even if (run_id, model_id) already in DB")
    parser.add_argument("--batch-size", type=int, default=25,
                        help="Images per micro-batch (default 25)")
    args = parser.parse_args()

    db_path = Path(args.output_root) / "yogamann.db"
    conn = open_db(db_path)

    vlm_cfg = yaml.safe_load(
        (Path(__file__).parent.parent / "profiles" / "vlm.yml")
        .read_text(encoding="utf-8")
    )
    model_keys = args.models or list(vlm_cfg["models"].keys())

    candidates = _fetch_run_candidates(conn, args.run_ids, args.limit)
    if not candidates:
        print("No candidate runs found.")
        return

    print(f"Models in run: {model_keys}")
    print(f"Candidate runs: {len(candidates)}")
    print(f"DB:             {db_path}")

    summary: dict[str, dict] = {}
    for model_key in model_keys:
        try:
            summary[model_key] = _run_model_phase(
                conn, model_key, candidates, args.batch_size, args.force
            )
        finally:
            evict_model(model_key)

    print("\n" + format_summary(summary))


if __name__ == "__main__":
    main()
