"""src/compare_vlm.py — run N models over DB runs, print side-by-side table.

Usage:
    python src/compare_vlm.py --run-ids 1 2 3 --output-root D:\\Temp\\yogamann-output
    python src/compare_vlm.py --limit 5 --output-root D:\\Temp\\yogamann-output
    python src/compare_vlm.py --limit 3 --models qwen2_5_vl_7b minicpm_v_4_6
"""
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import yaml

from db import open_db, save_vlm_annotation
from vlm_inference import annotate, VLMSchemaError


def _fetch_runs(conn, run_ids: list[int] | None, limit: int) -> list:
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


def _print_comparison(run_id: int, source_name: str, results: list[dict]) -> None:
    if not results:
        return
    model_ids = [r["model_id"] for r in results]
    col_w = max(16, max(len(m) for m in model_ids) + 2)
    label_w = 22
    header = f"{'':>{label_w}}" + "".join(m.center(col_w) for m in model_ids)
    sep = "-" * len(header)
    print(f"\nRun {run_id}  {source_name}")
    print(sep)
    print(header)
    for field in ("rating", "misaligned", "unwanted_features", "fail_patterns", "notes"):
        row = f"  {field:<{label_w - 2}}"
        for r in results:
            val = r.get(field, "")
            if isinstance(val, list):
                val = repr(val)
            row += str(val)[: col_w - 2].center(col_w)
        print(row)
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare VLM pose analysis across models")
    parser.add_argument("--run-ids", nargs="+", type=int, help="Specific run IDs to analyze")
    parser.add_argument("--limit", type=int, default=5, help="N most recent runs (default 5)")
    parser.add_argument("--output-root", default=r"D:\Temp\yogamann-output")
    parser.add_argument("--models", nargs="+",
                        help="Model keys to use (default: all enabled in vlm.yml)")
    parser.add_argument("--force", action="store_true",
                        help="Re-annotate even if (run_id, model_id) already exists in DB")
    parser.add_argument("--batch-size", type=int, default=25,
                        help="Images per micro-batch within a model phase (default 25)")
    args = parser.parse_args()

    db_path = Path(args.output_root) / "yogamann.db"
    conn = open_db(db_path)

    vlm_cfg = yaml.safe_load(
        (Path(__file__).parent.parent / "profiles" / "vlm.yml").read_text(encoding="utf-8")
    )
    model_keys = args.models or list(vlm_cfg["models"].keys())

    runs = _fetch_runs(conn, args.run_ids, args.limit)
    if not runs:
        print("No runs found.")
        return

    for run in runs:
        run_id = run["id"]
        source_name = Path(run["source_path"]).name
        results: list[dict] = []

        for model_key in model_keys:
            try:
                result = annotate(
                    Path(run["source_path"]),
                    Path(run["output_png"]),
                    model_key=model_key,
                )
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
                conn.commit()
                results.append(result)
            except (FileNotFoundError, VLMSchemaError) as exc:
                print(f"[skip] run {run_id} {model_key}: {exc}")
                results.append({
                    "model_id": model_key,
                    "rating": "[skip]",
                    "misaligned": [],
                    "unwanted_features": [],
                    "fail_patterns": [],
                    "notes": str(exc),
                })
            except Exception as exc:  # noqa: BLE001
                print(f"[error] run {run_id} {model_key}: {exc}")
                traceback.print_exc()
                results.append({
                    "model_id": model_key,
                    "rating": "[error]",
                    "misaligned": [],
                    "unwanted_features": [],
                    "fail_patterns": [],
                    "notes": str(exc),
                })

        _print_comparison(run_id, source_name, results)


if __name__ == "__main__":
    main()
