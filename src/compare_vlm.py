"""src/compare_vlm.py — multi-model VLM annotation orchestrator.

Model-major loop: load model -> annotate N images in micro-batches -> evict.
DB-driven resumability: skips (run_id, model_id) pairs already in vlm_annotations
unless --force. Each annotation auto-commits via save_vlm_annotation, so a
killed process resumes precisely from the last completed image.

Real-time metrics:
- Top-line batch ETA at start (sum of per-model phase ETAs from history)
- Per-phase header with historical avg + sample count
- Per-image one-liner: idx/total, run_id, elapsed, rating, steady avg, ETA
- Per-micro-batch summary (every K images): ok/err/remain counts
- Per-phase summary: total elapsed, steady-state s/img, % vs historical
- Final batch summary table

Usage:
    # Full 5-model batch on all 981 runs:
    python src/compare_vlm.py --limit 0 --output-root D:/Temp/yogamann-output

    # Subset of models, force re-annotate:
    python src/compare_vlm.py --models qwen2_5_vl_7b --limit 50 --force

    # Quieter (errors + summary only):
    python src/compare_vlm.py --limit 0 -q

    # More verbose (full tracebacks on errors):
    python src/compare_vlm.py --limit 0 -v
"""
from __future__ import annotations

import argparse
import time
import traceback
from datetime import datetime
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


def _historical_avg_latency(conn, model_id: str) -> tuple[float, int]:
    """Mean latency_s for this model from prior annotations.
    Returns (avg_seconds, sample_count). avg=0 if no samples yet."""
    row = conn.execute(
        "SELECT AVG(latency_s), COUNT(*) FROM vlm_annotations WHERE model_id = ?",
        (model_id,),
    ).fetchone()
    return (row[0] or 0.0, row[1] or 0)


def _fmt_eta(seconds: float) -> str:
    """Compact ETA — auto-choose seconds/minutes/hours."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.1f}h"


def _ts() -> str:
    """Brief HH:MM:SS timestamp for log lines."""
    return datetime.now().strftime("%H:%M:%S")


def _run_model_phase(
    conn,
    model_key: str,
    candidates: list,
    batch_size: int,
    force: bool,
    verbose: bool,
    quiet: bool,
) -> dict:
    """Annotate every candidate run with one model, in micro-batches.
    Returns {'ok': N, 'error': M, 'skipped': K, 'elapsed_s': T}.
    """
    if force:
        todo = candidates
        skipped = 0
    else:
        unanalyzed_ids = {r["id"] for r in get_unanalyzed_runs(conn, model_key)}
        todo = [r for r in candidates if r["id"] in unanalyzed_ids]
        skipped = len(candidates) - len(todo)

    hist_avg, n_samples = _historical_avg_latency(conn, model_key)
    phase_eta = hist_avg * len(todo) if hist_avg else 0.0

    if not quiet:
        print(f"\n=== {model_key} === [{_ts()}]")
        print(f"  candidates  : {len(candidates)}")
        print(f"  already done: {skipped}")
        print(f"  to process  : {len(todo)}")
        print(f"  micro-batch : {batch_size}")
        if hist_avg:
            print(
                f"  est. phase  : {_fmt_eta(phase_eta)}  "
                f"(avg {hist_avg:.1f}s/img from {n_samples} prior samples)"
            )
        else:
            print(f"  est. phase  : unknown (no prior samples for this model)")

    stats = {"ok": 0, "error": 0, "skipped": skipped, "elapsed_s": 0.0}
    if not todo:
        if not quiet:
            print(f"  [{model_key}] nothing to do.")
        return stats

    if not quiet:
        print(f"  loading {model_key}... (first call includes ~15-25s model load)")

    t_phase_start = time.perf_counter()
    load_plus_first = 0.0
    first_img = True

    for batch_idx, batch in enumerate(chunked(todo, batch_size), start=1):
        for run in batch:
            t_inf = time.perf_counter()
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
                elapsed_inf = time.perf_counter() - t_inf

                if first_img:
                    load_plus_first = elapsed_inf
                    first_img = False

                if not quiet:
                    done = stats["ok"] + stats["error"]
                    phase_elapsed = time.perf_counter() - t_phase_start
                    if done > 1:
                        steady_avg = (phase_elapsed - load_plus_first) / (done - 1)
                    else:
                        steady_avg = elapsed_inf
                    eta_remaining = steady_avg * (len(todo) - done)
                    rating = (result.get("rating") or "?")[:10]
                    marker = "L+1   " if done == 1 else f"avg={steady_avg:4.1f}s"
                    print(
                        f"  [{_ts()}] {model_key:<14} "
                        f"{done:>4}/{len(todo):<4} "
                        f"run={run['id']:<5} "
                        f"{elapsed_inf:5.1f}s "
                        f"rating={rating:<10} "
                        f"{marker} "
                        f"ETA={_fmt_eta(eta_remaining)}"
                    )
            except (FileNotFoundError, VLMSchemaError) as exc:
                stats["error"] += 1
                print(f"  [{_ts()}] {model_key:<14} [SKIP] run={run['id']}: {exc}")
            except Exception as exc:
                stats["error"] += 1
                print(
                    f"  [{_ts()}] {model_key:<14} [ERROR] run={run['id']}: "
                    f"{type(exc).__name__}: {exc}"
                )
                if verbose:
                    traceback.print_exc()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not quiet:
            done = stats["ok"] + stats["error"]
            print(
                f"  -- {model_key} batch {batch_idx} done: "
                f"{stats['ok']} ok / {stats['error']} err / "
                f"{len(todo) - done} remain --"
            )

    stats["elapsed_s"] = time.perf_counter() - t_phase_start
    if not quiet:
        n_inf = stats["ok"] + stats["error"]
        if n_inf > 1:
            steady = (stats["elapsed_s"] - load_plus_first) / (n_inf - 1)
        else:
            steady = stats["elapsed_s"]
        delta_str = ""
        if hist_avg:
            delta_pct = (steady - hist_avg) / hist_avg * 100
            delta_str = f" ({delta_pct:+.0f}% vs prior {hist_avg:.1f}s)"
        print(
            f"  [{_ts()}] {model_key} PHASE DONE: "
            f"{stats['ok']} ok, {stats['error']} err, "
            f"{_fmt_eta(stats['elapsed_s'])} total, "
            f"steady-state {steady:.1f}s/img{delta_str}"
        )

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
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true",
                       help="More detail (full tracebacks on errors)")
    group.add_argument("-q", "--quiet", action="store_true",
                       help="Only errors and final summary")
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

    # Top-level batch ETA — sum of per-model phase predictions.
    batch_eta_sec = 0.0
    for mk in model_keys:
        hist_avg, _ = _historical_avg_latency(conn, mk)
        if hist_avg:
            unanalyzed = {r["id"] for r in get_unanalyzed_runs(conn, mk)}
            n_todo = (
                len(candidates) if args.force
                else sum(1 for c in candidates if c["id"] in unanalyzed)
            )
            batch_eta_sec += hist_avg * n_todo

    print(f"[{_ts()}] Batch orchestrator starting")
    print(f"Models in run  : {model_keys}")
    print(f"Candidate runs : {len(candidates)}")
    print(f"DB             : {db_path}")
    print(f"Force re-ann.  : {args.force}")
    print(f"Verbosity      : {'quiet' if args.quiet else 'verbose' if args.verbose else 'normal'}")
    if batch_eta_sec:
        print(f"Total batch ETA: {_fmt_eta(batch_eta_sec)} (based on historical averages)")
    else:
        print(f"Total batch ETA: unknown (no historical samples)")

    summary: dict[str, dict] = {}
    t_total_start = time.perf_counter()
    for model_key in model_keys:
        try:
            summary[model_key] = _run_model_phase(
                conn, model_key, candidates, args.batch_size,
                args.force, args.verbose, args.quiet,
            )
        finally:
            evict_model(model_key)

    total_elapsed = time.perf_counter() - t_total_start
    print(f"\n[{_ts()}] BATCH COMPLETE in {_fmt_eta(total_elapsed)}")
    print(format_summary(summary))


if __name__ == "__main__":
    main()
