"""src/analyze.py — poll DB and annotate unanalyzed runs with active VLM model.

Usage:
    # Annotate all pending runs and exit:
    .venv\\Scripts\\python.exe src/analyze.py --once --output-root D:\\Temp\\yogamann-output

    # Run as persistent daemon (checks every 30 s):
    .venv\\Scripts\\python.exe src/analyze.py --poll-interval 30 --output-root D:\\Temp\\yogamann-output
"""
from __future__ import annotations

import argparse
import signal
import time
import traceback
from pathlib import Path

import yaml

from db import open_db, save_vlm_annotation, get_unanalyzed_runs, save_rating
from vlm_inference import annotate, VLMSchemaError

_RUNNING = True


def _handle_sigint(sig, frame):
    global _RUNNING
    print("\n[analyze] Ctrl-C received — shutting down after current run...")
    _RUNNING = False


def _process_run(conn, run, model_key: str, promote_to_runs: bool) -> None:
    run_id = run["id"]
    source = Path(run["source_path"])
    render = Path(run["output_png"])

    if not source.exists():
        print(f"[skip] run {run_id}: source photo missing: {source}")
        return
    if not render.exists():
        print(f"[skip] run {run_id}: render missing: {render}")
        return

    try:
        result = annotate(source, render, model_key=model_key)
        save_vlm_annotation(
            conn, run_id, model_key,
            rating=result["rating"],
            misaligned=result["misaligned"],
            unwanted_features=result["unwanted_features"],
            fail_patterns=result["fail_patterns"],
            notes=result["notes"],
            raw_output=result["raw_output"],
            latency_s=result["latency_s"],
        )
        if promote_to_runs:
            # VLM is the automated primary annotator for runs.rating;
            # human annotations live in the separate annotations table.
            save_rating(
                conn, run_id,
                result["rating"],
                result["notes"],
                misaligned=result["misaligned"],
                unwanted_features=result["unwanted_features"],
            )
        conn.commit()
        print(
            f"[ok] run {run_id} | {model_key} | {result['rating']}"
            f" | {result['latency_s']:.1f}s"
        )
    except MemoryError as exc:
        print(f"[oom] run {run_id}: {exc} — skipping")
    except VLMSchemaError as exc:
        print(f"[warn] run {run_id} schema error: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"[error] run {run_id}: {exc}")
        traceback.print_exc()


def main() -> None:
    global _RUNNING
    _RUNNING = True  # reset so main() can be called again in tests
    signal.signal(signal.SIGINT, _handle_sigint)

    parser = argparse.ArgumentParser(description="Annotate DB runs with active VLM model")
    parser.add_argument("--once", action="store_true", help="Process pending runs and exit")
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between polls")
    parser.add_argument("--output-root", default=r"D:\Temp\yogamann-output")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.exists():
        raise SystemExit(f"[analyze] --output-root does not exist: {output_root}")
    db_path = output_root / "yogamann.db"
    conn = open_db(db_path)

    vlm_cfg = yaml.safe_load(
        (Path(__file__).parent.parent / "profiles" / "vlm.yml").read_text(encoding="utf-8")
    )
    active_model = vlm_cfg["active_model"]

    mode = "once" if args.once else f"poll every {args.poll_interval}s"
    print(f"[analyze] active model : {active_model}")
    print(f"[analyze] mode         : {mode}")
    print(f"[analyze] db           : {db_path}")

    while _RUNNING:
        runs = get_unanalyzed_runs(conn, active_model)
        if runs:
            print(f"[analyze] {len(runs)} unanalyzed run(s)")
            for run in runs:
                if not _RUNNING:
                    break
                _process_run(conn, run, active_model, promote_to_runs=True)
        elif args.once:
            print("[analyze] No pending runs.")

        if args.once:
            break
        time.sleep(args.poll_interval)

    print("[analyze] Done.")


if __name__ == "__main__":
    main()
