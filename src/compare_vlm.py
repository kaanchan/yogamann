"""src/compare_vlm.py — multi-model VLM annotation orchestrator.

Model-major loop: load model -> annotate N images in micro-batches -> evict.
DB-driven resumability: skips (run_id, model_id) pairs already in vlm_annotations
unless --force. Each annotation auto-commits via save_vlm_annotation, so a
killed process resumes precisely from the last completed image.

Real-time metrics:
- Top-line batch ETA at start (sum of per-model phase ETAs from history)
- Per-phase header with historical avg + sample count
- Per-image one-liner: idx/total, run_id, elapsed, rating, steady avg, ETA + GPU/CPU
- Per-micro-batch summary (every K images): ok/err/remain counts
- Per-phase summary: total elapsed, steady-state s/img, % vs historical
- Final batch summary table
- JSONL structured log: one record per image with latency + GPU + CPU metrics

Usage:
    # Full 5-model batch on all 981 runs:
    python src/compare_vlm.py --limit 0 --output-root D:/Temp/yogamann-output

    # With thermal guard (pause if GPU ≥ 80°C for 45s):
    python src/compare_vlm.py --limit 0 --gpu-temp-limit 80 --cooldown-secs 45

    # Subset of models, force re-annotate:
    python src/compare_vlm.py --models qwen2_5_vl_7b --limit 50 --force

    # Quieter (errors + summary only):
    python src/compare_vlm.py --limit 0 -q

    # More verbose (full tracebacks on errors):
    python src/compare_vlm.py --limit 0 -v
"""
from __future__ import annotations

import argparse
import json
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
from gpu_monitor import cooldown_if_hot, fmt_metrics, sample as gpu_sample
from vlm_inference import annotate, evict_model, VLMSchemaError
import batch_lock


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


def _count_completed(conn, model_key: str, candidate_ids: set[int]) -> int:
    """Completed annotations for this model within the active candidate set."""
    if not candidate_ids:
        return 0
    placeholders = ",".join("?" * len(candidate_ids))
    return conn.execute(
        f"SELECT COUNT(*) FROM vlm_annotations "
        f"WHERE model_id = ? AND run_id IN ({placeholders})",
        (model_key, *candidate_ids),
    ).fetchone()[0]


def _compute_slice_size(
    conn,
    model_key: str,
    total_candidates: int,
    time_budget_s: float,
    slice_pct: float,
    slice_count: int | None,
) -> int:
    """Images to run in one round for this model.

    Priority:
      1. --slice N  (hard override)
      2. hist_avg exists → floor(time_budget / avg)  (time-budget mode)
      3. fallback → ceil(total * slice_pct / 100)    (percentage mode)
    """
    if slice_count is not None:
        return max(1, slice_count)
    hist_avg, _ = _historical_avg_latency(conn, model_key)
    if hist_avg > 0:
        return max(1, int(time_budget_s / hist_avg))
    return max(1, int(total_candidates * slice_pct / 100))


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
    gpu_temp_limit: float,
    cooldown_secs: float,
    jsonl_fh,
    max_images: int | None = None,
) -> dict:
    """Annotate up to max_images candidate runs with one model, in micro-batches.
    Returns {'ok': N, 'error': M, 'skipped': K, 'elapsed_s': T}.
    max_images=None means run all remaining (model-major behaviour).
    """
    if force:
        todo_full = candidates
        skipped = 0
    else:
        unanalyzed_ids = {r["id"] for r in get_unanalyzed_runs(conn, model_key)}
        todo_full = [r for r in candidates if r["id"] in unanalyzed_ids]
        skipped = len(candidates) - len(todo_full)

    todo = todo_full[:max_images] if max_images is not None else todo_full

    hist_avg, n_samples = _historical_avg_latency(conn, model_key)
    phase_eta = hist_avg * len(todo) if hist_avg else 0.0

    if not quiet:
        print(f"\n=== {model_key} === [{_ts()}]")
        print(f"  candidates  : {len(candidates)}")
        print(f"  already done: {skipped}")
        if max_images is not None and len(todo_full) > len(todo):
            print(f"  remaining   : {len(todo_full)}  (slice capped at {len(todo)})")
        else:
            print(f"  to process  : {len(todo)}")
        print(f"  micro-batch : {batch_size}")
        if hist_avg:
            print(
                f"  est. slice  : {_fmt_eta(phase_eta)}  "
                f"(avg {hist_avg:.1f}s/img from {n_samples} prior samples)"
            )
        else:
            print(f"  est. slice  : unknown (no prior samples for this model)")

    stats = {"ok": 0, "error": 0, "skipped": skipped, "elapsed_s": 0.0}
    if not todo:
        if not quiet:
            print(f"  [{model_key}] nothing to do.")
        return stats

    if not quiet:
        print(f"  loading {model_key}... (first call includes ~15-25s model load)", flush=True)

    t_phase_start = time.perf_counter()
    load_plus_first = 0.0
    first_img = True

    for batch_idx, batch in enumerate(chunked(todo, batch_size), start=1):
        for run in batch:
            # Thermal guard — check before starting inference on each image.
            cooldown_if_hot(gpu_temp_limit, cooldown_secs, quiet=quiet)

            done_before = stats["ok"] + stats["error"]
            if not quiet:
                src_name = Path(run["source_path"]).name
                print(
                    f"  [{_ts()}] {model_key:<14} "
                    f"[{done_before+1:>4}/{len(todo):<4}] "
                    f"run={run['id']:<5} {src_name}  → inferring ...",
                    flush=True,
                )

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

                # Sample hardware metrics immediately after inference completes.
                hw = gpu_sample()

                if jsonl_fh is not None:
                    record = {
                        "ts": hw["ts"],
                        "model_id": model_key,
                        "run_id": run["id"],
                        "status": "ok",
                        "rating": result.get("rating"),
                        "latency_s": round(elapsed_inf, 3),
                        "gpu_temp_c": hw["gpu_temp_c"],
                        "gpu_util_pct": hw["gpu_util_pct"],
                        "gpu_mem_used_mb": hw["gpu_mem_used_mb"],
                        "gpu_power_w": hw["gpu_power_w"],
                        "cpu_util_pct": hw["cpu_util_pct"],
                    }
                    jsonl_fh.write(json.dumps(record) + "\n")
                    jsonl_fh.flush()

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
                        f"ETA={_fmt_eta(eta_remaining)} "
                        f"| {fmt_metrics(hw)}",
                        flush=True,
                    )
            except (FileNotFoundError, VLMSchemaError) as exc:
                stats["error"] += 1
                if jsonl_fh is not None:
                    hw = gpu_sample()
                    record = {
                        "ts": hw["ts"],
                        "model_id": model_key,
                        "run_id": run["id"],
                        "status": "skip",
                        "error": str(exc),
                        "latency_s": round(time.perf_counter() - t_inf, 3),
                        "gpu_temp_c": hw["gpu_temp_c"],
                        "gpu_util_pct": hw["gpu_util_pct"],
                        "gpu_mem_used_mb": hw["gpu_mem_used_mb"],
                        "gpu_power_w": hw["gpu_power_w"],
                        "cpu_util_pct": hw["cpu_util_pct"],
                    }
                    jsonl_fh.write(json.dumps(record) + "\n")
                    jsonl_fh.flush()
                print(f"  [{_ts()}] {model_key:<14} [SKIP] run={run['id']}: {exc}")
            except Exception as exc:
                stats["error"] += 1
                if jsonl_fh is not None:
                    hw = gpu_sample()
                    record = {
                        "ts": hw["ts"],
                        "model_id": model_key,
                        "run_id": run["id"],
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "latency_s": round(time.perf_counter() - t_inf, 3),
                        "gpu_temp_c": hw["gpu_temp_c"],
                        "gpu_util_pct": hw["gpu_util_pct"],
                        "gpu_mem_used_mb": hw["gpu_mem_used_mb"],
                        "gpu_power_w": hw["gpu_power_w"],
                        "cpu_util_pct": hw["cpu_util_pct"],
                    }
                    jsonl_fh.write(json.dumps(record) + "\n")
                    jsonl_fh.flush()
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
    parser.add_argument("--gpu-temp-limit", type=float, default=80.0,
                        help="Pause inference if GPU temp reaches this °C (default 80)")
    parser.add_argument("--cooldown-secs", type=float, default=30.0,
                        help="Seconds to sleep when GPU is over temp limit (default 30)")
    parser.add_argument("--metrics-log",
                        help="Path for per-image JSONL metrics log "
                             "(default: {output_root}/batch_metrics.jsonl)")
    # ── Scheduling ────────────────────────────────────────────────────────────
    parser.add_argument("--strategy", choices=["round-robin", "model-major"],
                        default="round-robin",
                        help="round-robin: rotate models every slice (default). "
                             "model-major: finish all images per model before next.")
    parser.add_argument("--time-budget", type=float, default=300.0,
                        metavar="SECS",
                        help="Seconds per model per round (round-robin only). "
                             "Slice size = floor(budget / hist_avg). Default 300 (5 min).")
    parser.add_argument("--slice-pct", type=float, default=10.0,
                        metavar="PCT",
                        help="Fallback slice size as %% of total when no history. Default 10.")
    parser.add_argument("--slice", type=int, default=None,
                        metavar="N",
                        help="Hard-override: fixed image count per model per round. "
                             "Overrides --time-budget and --slice-pct.")
    # ──────────────────────────────────────────────────────────────────────────
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true",
                       help="More detail (full tracebacks on errors)")
    group.add_argument("-q", "--quiet", action="store_true",
                       help="Only errors and final summary")
    args = parser.parse_args()

    db_path = Path(args.output_root) / "yogamann.db"
    conn = open_db(db_path)

    # Resolve model list before acquiring lock so we record it accurately.
    vlm_cfg = yaml.safe_load(
        (Path(__file__).parent.parent / "profiles" / "vlm.yml")
        .read_text(encoding="utf-8")
    )
    model_keys = args.models or list(vlm_cfg["models"].keys())

    metrics_log_path = Path(
        args.metrics_log or (Path(args.output_root) / "batch_metrics.jsonl")
    )

    # ── GPU-busy lock ──────────────────────────────────────────────────────────
    existing = batch_lock.check(args.output_root)
    if existing:
        print(
            f"[{_ts()}] ERROR: GPU is already in use by another process.\n"
            f"  PID={existing['pid']}  type={existing['job_type']}\n"
            f"  models={existing['model_ids']}\n"
            f"  started={existing['start_time']}\n"
            "Exiting. Kill that process first, or wait for it to finish."
        )
        return
    batch_lock.acquire(args.output_root, job_type="batch", model_ids=model_keys)
    # ──────────────────────────────────────────────────────────────────────────

    candidates = _fetch_run_candidates(conn, args.run_ids, args.limit)
    if not candidates:
        print("No candidate runs found.")
        batch_lock.release(args.output_root)
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
    print(f"Metrics log    : {metrics_log_path}")
    print(f"GPU temp limit : {args.gpu_temp_limit}°C  cooldown={args.cooldown_secs}s")
    print(f"Strategy       : {args.strategy}")
    if args.strategy == "round-robin":
        if args.slice is not None:
            print(f"Slice mode     : fixed {args.slice} imgs/model/round")
        else:
            print(f"Slice mode     : time-budget {args.time_budget:.0f}s/model  "
                  f"(fallback {args.slice_pct:.0f}% of {len(candidates)} = "
                  f"{max(1, int(len(candidates) * args.slice_pct / 100))} imgs)")
    print(f"Force re-ann.  : {args.force}")
    print(f"Verbosity      : {'quiet' if args.quiet else 'verbose' if args.verbose else 'normal'}")
    if batch_eta_sec:
        print(f"Total batch ETA: {_fmt_eta(batch_eta_sec)} (based on historical averages)")
    else:
        print(f"Total batch ETA: unknown (no historical samples)")

    candidate_ids = {r["id"] for r in candidates}
    summary: dict[str, dict] = {}
    t_total_start = time.perf_counter()

    def _accumulate(key: str, stats: dict) -> None:
        if key not in summary:
            summary[key] = {"ok": 0, "error": 0, "skipped": 0, "elapsed_s": 0.0}
        for k in ("ok", "error", "skipped", "elapsed_s"):
            summary[key][k] += stats[k]

    try:
        with open(metrics_log_path, "a", encoding="utf-8") as jsonl_fh:

            if args.strategy == "model-major":
                for model_key in model_keys:
                    try:
                        _accumulate(model_key, _run_model_phase(
                            conn, model_key, candidates, args.batch_size,
                            args.force, args.verbose, args.quiet,
                            args.gpu_temp_limit, args.cooldown_secs,
                            jsonl_fh,
                        ))
                    finally:
                        evict_model(model_key)
                        cooldown_if_hot(args.gpu_temp_limit, args.cooldown_secs,
                                        quiet=args.quiet)

            else:  # round-robin (default)
                round_num = 0
                while True:
                    round_num += 1

                    # Lag-first: sort by completions ASC within our candidate set.
                    counts = {mk: _count_completed(conn, mk, candidate_ids)
                              for mk in model_keys}
                    ordered = sorted(model_keys, key=lambda mk: counts[mk])

                    lag_str = "  ".join(
                        f"{mk.split('_')[0]}({counts[mk]})" for mk in ordered
                    )
                    print(f"\n{'═'*60}", flush=True)
                    print(f"  Round {round_num}  [{_ts()}]  lag-order: {lag_str}", flush=True)
                    print(f"{'═'*60}", flush=True)

                    any_work = False
                    for model_key in ordered:
                        slice_n = _compute_slice_size(
                            conn, model_key, len(candidates),
                            args.time_budget, args.slice_pct, args.slice,
                        )
                        hist_avg, _ = _historical_avg_latency(conn, model_key)
                        if hist_avg > 0 and args.slice is None:
                            slice_src = f"budget={args.time_budget:.0f}s ÷ avg={hist_avg:.1f}s"
                        elif args.slice is not None:
                            slice_src = "fixed override"
                        else:
                            slice_src = f"fallback {args.slice_pct:.0f}% (no history)"
                        print(f"  {model_key} → {slice_n} imgs  ({slice_src})", flush=True)

                        try:
                            stats = _run_model_phase(
                                conn, model_key, candidates, args.batch_size,
                                args.force, args.verbose, args.quiet,
                                args.gpu_temp_limit, args.cooldown_secs,
                                jsonl_fh, max_images=slice_n,
                            )
                            _accumulate(model_key, stats)
                            if stats["ok"] + stats["error"] > 0:
                                any_work = True
                        finally:
                            evict_model(model_key)
                            cooldown_if_hot(args.gpu_temp_limit, args.cooldown_secs,
                                            quiet=args.quiet)

                    if not any_work:
                        print(f"\n[{_ts()}] All models complete — nothing left to process.")
                        break

    finally:
        batch_lock.release(args.output_root)

    total_elapsed = time.perf_counter() - t_total_start
    print(f"\n[{_ts()}] BATCH COMPLETE in {_fmt_eta(total_elapsed)}")
    print(format_summary(summary))


if __name__ == "__main__":
    main()
