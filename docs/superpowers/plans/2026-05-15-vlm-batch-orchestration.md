# VLM Batch Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `src/compare_vlm.py` from image-major to model-major loop with micro-batching, explicit GPU eviction between models, and DB-driven resumability — so a 1000-image × N-model batch can run on a 16 GB RTX 5080 Laptop, survive interruptions, and skip already-completed work.

**Architecture:** Outer loop iterates models (one model resident at a time); inner loop iterates images in micro-batches of K (default 25) with `torch.cuda.empty_cache()` between batches; each (run_id, model_id) annotation auto-commits to DB; resumption uses existing `get_unanalyzed_runs()` SQL filter to skip already-done pairs.

**Tech Stack:** Python 3.13.5, transformers==4.49.0 (pinned), bitsandbytes 4-bit, PyTorch 2.13 nightly, SQLite via `src/db.py`. No new dependencies.

---

## Hardware constraint (critical)

RTX 5080 Laptop, **15.92 GB VRAM**. Each VLM in bnb 4-bit peaks at ~7-9 GB during inference. **No two models fit simultaneously.** Eviction between models is mandatory, not optional.

## Files touched

| File | Action | Responsibility |
|---|---|---|
| `src/vlm_inference.py` | modify | Add public `evict_model()` + `evict_all()`; keep existing `_MODEL_CACHE`, `_TOKENIZER_CACHE` |
| `src/compare_vlm.py` | major refactor | Invert outer loop, add flags, add per-model phases with eviction, add summary reporting |
| `src/db.py` | minor | Add `count_vlm_annotations(conn, model_id)` helper (read-only stat) |
| `tests/test_vlm_helpers.py` | create | Unit tests for pure-Python pieces (chunking, summary aggregation, args parsing) |
| `docs/architecture-decisions.md` | modify | Fix `~16 GB` reference in ADR-003; add ADR-004 for orchestration |
| `docs/research/issue-32-transformers-version-pinning/PROMPT.md` | modify | Fix "≤24 GB" filter to "≤16 GB" |

**Out of scope (deferred to a separate change):**
- `src/analyze.py` (single-model batch, already model-major) — does not need this refactor
- Cross-model voting / agreement metrics — separate analysis, not orchestration
- Parallelism across GPUs — single-GPU workstation only

---

## Task 1: Add eviction helpers to vlm_inference.py

**Files:**
- Modify: `src/vlm_inference.py` (append two functions near the existing `_MODEL_CACHE` definition)

- [ ] **Step 1: Read the existing cache structure**

Look at `src/vlm_inference.py` near the top of the file to confirm the cache globals exist:
- `_MODEL_CACHE: dict[str, tuple]`
- `_TOKENIZER_CACHE: dict[str, object]`

- [ ] **Step 2: Add `evict_model` and `evict_all` after `_get_tokenizer` (around line ~100)**

Add this block after the `_get_tokenizer` function:

```python
# ── Eviction (public — callers manage GPU memory between model phases) ───────
def evict_model(model_key: str) -> None:
    """Drop a cached model and free its GPU memory.

    Safe to call when model_key was never loaded — that's a no-op.
    Always clears the tokenizer entry too (tokenizers are small but
    keeping them around obscures which models are 'loaded').
    """
    import gc
    if model_key in _MODEL_CACHE:
        model, _ = _MODEL_CACHE.pop(model_key)
        del model
    # Also drop tokenizers cached under the model's repo (best-effort).
    _TOKENIZER_CACHE.clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def evict_all() -> None:
    """Drop all cached models and tokenizers. Same semantics as
    evict_model called for every cached key."""
    for key in list(_MODEL_CACHE.keys()):
        evict_model(key)
```

- [ ] **Step 3: Verify nothing else imports `_MODEL_CACHE` or `_TOKENIZER_CACHE` from outside this module**

Run: `grep -rn "_MODEL_CACHE\|_TOKENIZER_CACHE" src/ tests/ 2>/dev/null`
Expected: only references are inside `src/vlm_inference.py` itself.

- [ ] **Step 4: Smoke test — load and evict a model, observe VRAM drop**

Run (note: do NOT bind the return value to locals — those references would block GC):
```bash
PYTHONPATH=src .venv/Scripts/python.exe -c "
import torch
from vlm_inference import _load_config, _load_model, evict_model
cfg = _load_config('qwen2_5_vl_7b')
print(f'Before:      {torch.cuda.memory_allocated()/1024**3:.2f} GB')
_load_model('qwen2_5_vl_7b', cfg)  # do NOT bind: only _MODEL_CACHE holds it
print(f'After load:  {torch.cuda.memory_allocated()/1024**3:.2f} GB')
evict_model('qwen2_5_vl_7b')
print(f'After evict: {torch.cuda.memory_allocated()/1024**3:.2f} GB')
" 2>&1 | tail -5
```

Working directory: `c:/Users/kaanchan/Projects/Yoga/yogamann`.

Expected output: "Before" 0.00 GB; "After load" ~5-6 GB; "After evict" 0.00 GB.

*In real callers (`compare_vlm.py`) the model is held only by `_MODEL_CACHE`, so eviction works without any caller-side cleanup. If you bind the return value to a local in a smoke test, you must also `del` that local before calling `evict_model` or the test will misleadingly show no memory drop.*

- [ ] **Step 5: Commit**

```bash
git add src/vlm_inference.py
git commit -m "feat: add evict_model + evict_all to vlm_inference (#32)

Public eviction helpers so callers can manage GPU memory between
multi-model batch phases. evict_model() drops the cached model,
clears tokenizer cache, runs gc.collect() and torch.cuda.empty_cache().
evict_all() loops over evict_model() for every cached key.

Needed for model-major orchestration on 16 GB VRAM (RTX 5080 Laptop)
where no two VLMs fit simultaneously."
```

---

## Task 2: Add DB helper for completion stats

**Files:**
- Modify: `src/db.py` (add one function near `get_unanalyzed_runs`)

- [ ] **Step 1: Locate `get_unanalyzed_runs` in `src/db.py`** (around line 500)

- [ ] **Step 2: Add `count_vlm_annotations` right above it**

```python
def count_vlm_annotations(conn: sqlite3.Connection, model_id: str) -> int:
    """Return the number of vlm_annotations rows for a model — used to
    report 'already done, skipping' counts at the start of a batch."""
    row = conn.execute(
        "SELECT COUNT(*) FROM vlm_annotations WHERE model_id = ?",
        (model_id,),
    ).fetchone()
    return row[0] if row else 0
```

- [ ] **Step 3: Smoke test the helper**

Run:
```bash
.venv/Scripts/python.exe -c "
from db import open_db, count_vlm_annotations
from pathlib import Path
conn = open_db(Path(r'D:/Temp/yogamann-output/yogamann.db'))
for m in ['qwen2_5_vl_7b', 'internvl2_5_8b', 'molmo_7b_d', 'minicpm_v_2_6']:
    print(f'{m}: {count_vlm_annotations(conn, m)} annotations done')
" 2>&1 | tail -10
```

Expected: integer counts per model. Could be 0 if no batch has run yet, or small numbers from interactive tests. No errors.

- [ ] **Step 4: Commit**

```bash
git add src/db.py
git commit -m "feat: add count_vlm_annotations helper to db (#32)

Small read-only helper used by the model-major batch orchestrator to
report 'already done, skipping N' at the start of each model phase."
```

---

## Task 3: Add chunking + summary helpers (TDD, pytest)

**Files:**
- Create: `src/_batch_utils.py` (pure Python; no torch/transformers imports)
- Create: `tests/test_batch_utils.py`
- Modify: `pyproject.toml` (add pytest pythonpath config)

- [ ] **Step 1: Configure pytest to find modules under `src/`**

The project uses unqualified imports (`from db import open_db`, `from vlm_inference import annotate`). For pytest to resolve these, add `src/` to `sys.path`.

Edit `pyproject.toml`, append at end:

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_batch_utils.py`:

```python
import pytest
from _batch_utils import chunked, format_summary


def test_chunked_exact_divisor():
    assert list(chunked([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]


def test_chunked_remainder():
    assert list(chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_chunked_empty():
    assert list(chunked([], 3)) == []


def test_chunked_size_larger_than_input():
    assert list(chunked([1, 2], 5)) == [[1, 2]]


def test_chunked_size_zero_raises():
    with pytest.raises(ValueError):
        list(chunked([1, 2], 0))


def test_format_summary_basic():
    stats = {
        "qwen2_5_vl_7b": {"ok": 95, "error": 3, "skipped": 2, "elapsed_s": 412.0},
        "internvl2_5_8b": {"ok": 90, "error": 8, "skipped": 2, "elapsed_s": 530.5},
    }
    out = format_summary(stats)
    assert "qwen2_5_vl_7b" in out
    assert "95" in out and "3" in out and "412" in out
    assert "internvl2_5_8b" in out
    # Header row exists
    assert "model" in out.lower()


def test_format_summary_empty():
    assert "no models" in format_summary({}).lower()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_batch_utils.py -v`
Expected: ImportError or ModuleNotFoundError on `_batch_utils`. All 7 tests fail.

- [ ] **Step 4: Create the implementation**

Create `src/_batch_utils.py`:

```python
"""src/_batch_utils.py — pure-Python helpers for VLM batch orchestration.

No torch/transformers imports here on purpose: keeps this module fast to
import for unit testing.
"""
from __future__ import annotations

from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


def chunked(items: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield successive lists of up to `size` items from `items`.
    Empty iterable yields nothing. Final chunk may be shorter."""
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def format_summary(stats: dict[str, dict]) -> str:
    """Format a per-model batch-run summary as a table string."""
    if not stats:
        return "No models processed."
    header = f"{'model':<24} {'ok':>6} {'err':>5} {'skip':>6} {'elapsed_s':>11}"
    sep = "-" * len(header)
    lines = [header, sep]
    total_ok = total_err = total_skip = 0
    total_elapsed = 0.0
    for model_id, s in stats.items():
        lines.append(
            f"{model_id:<24} {s['ok']:>6} {s['error']:>5} "
            f"{s['skipped']:>6} {s['elapsed_s']:>11.1f}"
        )
        total_ok += s["ok"]
        total_err += s["error"]
        total_skip += s["skipped"]
        total_elapsed += s["elapsed_s"]
    lines.append(sep)
    lines.append(
        f"{'TOTAL':<24} {total_ok:>6} {total_err:>5} "
        f"{total_skip:>6} {total_elapsed:>11.1f}"
    )
    return "\n".join(lines)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_batch_utils.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/_batch_utils.py tests/test_batch_utils.py pyproject.toml
git commit -m "feat: add _batch_utils with chunked + format_summary (#32)

Pure-Python helpers for the orchestrator. Pytest-covered.
- chunked(iterable, size) yields successive lists; raises on size<=0
- format_summary(stats_dict) renders per-model results as a fixed-width table
  with a TOTAL row.

Kept in a separate module with no torch/transformers imports so unit
tests run in <1s without GPU."
```

---

## Task 4: Add new CLI flags to compare_vlm.py (forward-compatible, before loop refactor)

**Files:**
- Modify: `src/compare_vlm.py` (only `argparse` block, no behavior change yet)

This step adds the flags but doesn't use them yet. Lets us land the parsing change cleanly before the bigger refactor.

- [ ] **Step 1: Read current `argparse` block**

Open `src/compare_vlm.py`, find the `parser = argparse.ArgumentParser(...)` block (around line 60).

- [ ] **Step 2: Add three new flags**

Replace the existing argparse setup with this. Only the three new `add_argument` lines (`--force`, `--batch-size`, plus the existing `--models` description tweak for the new semantics) are added:

```python
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
```

- [ ] **Step 3: Smoke test — argparse still works**

Run: `.venv/Scripts/python.exe src/compare_vlm.py --help 2>&1 | head -20`
Expected: help text shows the three new flags. No crash.

- [ ] **Step 4: Verify backwards compatibility — old invocation still works**

Run: `.venv/Scripts/python.exe src/compare_vlm.py --run-ids 981 --models qwen2_5_vl_7b --output-root D:/Temp/yogamann-output 2>&1 | tail -10`
Expected: Qwen runs as before, produces a rating. Same output as before this change.

- [ ] **Step 5: Commit**

```bash
git add src/compare_vlm.py
git commit -m "chore: add --force and --batch-size flags to compare_vlm (#32)

Parsing only; the existing image-major loop is unchanged in this commit.
Subsequent commit inverts the loop and consumes these flags."
```

---

## Task 5: Refactor compare_vlm.py to model-major + micro-batching + resumability

**Files:**
- Modify: `src/compare_vlm.py` (replace `main()` and `_fetch_runs`; remove `_print_comparison`)

This is the big change. Replaces the image-major loop entirely with a model-major orchestrator.

- [ ] **Step 1: Read the current `main()` and `_fetch_runs` to understand what's being replaced**

Look at `src/compare_vlm.py` lines 1-130. Note the two helpers `_fetch_runs` and `_print_comparison` and the loop structure.

- [ ] **Step 2: Replace the file with the new structure**

The new `src/compare_vlm.py` (full file — overwrite):

```python
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
```

- [ ] **Step 3: Verify imports resolve**

Run: `.venv/Scripts/python.exe -c "import compare_vlm" 2>&1`
Expected: no output (clean import). Working dir: `c:/Users/kaanchan/Projects/Yoga/yogamann`. Set `PYTHONPATH=src` if needed.

- [ ] **Step 4: Smoke test 1 — single model, single image (regression check)**

Run:
```bash
.venv/Scripts/python.exe src/compare_vlm.py --run-ids 981 --models qwen2_5_vl_7b --output-root D:/Temp/yogamann-output 2>&1 | tail -20
```

Expected:
- Header lines showing 1 candidate run, 1 model
- Model phase shows either "to process: 1" (if not yet annotated) or "already done: 1" (if previously annotated by Qwen)
- A micro-batch line "[qwen2_5_vl_7b] batch 1: 1 ok, 0 err, 0 remain" (or 0/0/0 if skipped)
- Summary table at the end with one row for `qwen2_5_vl_7b`

- [ ] **Step 5: Smoke test 2 — 4 models, 3 images (the real model-major test)**

Run:
```bash
.venv/Scripts/python.exe src/compare_vlm.py --limit 3 --output-root D:/Temp/yogamann-output 2>&1 | tail -50
```

Expected:
- Output shows **4 separate "=== model_key ===" phases**, one for each of qwen2_5_vl_7b, internvl2_5_8b, molmo_7b_d, minicpm_v_2_6
- Each phase prints its own batch progress line
- Between models, no OOM error
- Final summary table shows all 4 rows + TOTAL

If any model OOMs, debug by checking `nvidia-smi` between phases (should drop to <1 GB between models). Verify `evict_model` is actually being called (the `finally:` block).

- [ ] **Step 6: Smoke test 3 — resumability**

Run the same 4-model 3-image batch again immediately:

```bash
.venv/Scripts/python.exe src/compare_vlm.py --limit 3 --output-root D:/Temp/yogamann-output 2>&1 | tail -30
```

Expected:
- Each phase reports "already done: 3, to process: 0"
- No model is actually loaded (the phase exits early before annotate is called)
- Summary table: all rows show `0 ok, 0 err, 3 skipped`
- Total elapsed: under 5 seconds (no GPU work happens)

- [ ] **Step 7: Smoke test 4 — `--force` flag re-annotates**

```bash
.venv/Scripts/python.exe src/compare_vlm.py --limit 3 --models qwen2_5_vl_7b --force --output-root D:/Temp/yogamann-output 2>&1 | tail -15
```

Expected:
- "already done: 0, to process: 3" (force makes us ignore the DB filter)
- 3 successful annotations
- Summary: `3 ok, 0 err, 0 skipped`

- [ ] **Step 8: Smoke test 5 — interruption mid-batch**

Start a longer run, interrupt it with Ctrl+C after the first model's batch completes, restart:

```bash
# Terminal 1: start the run
.venv/Scripts/python.exe src/compare_vlm.py --limit 10 --output-root D:/Temp/yogamann-output
# Wait ~30s for the first batch to print "batch 1: ... ok, 0 err, 0 remain", then Ctrl+C

# Then re-run the same command
.venv/Scripts/python.exe src/compare_vlm.py --limit 10 --output-root D:/Temp/yogamann-output 2>&1 | tail -30
```

Expected on re-run:
- First model: "already done: 10" (Qwen finished before interrupt) or "already done: N, to process: 10-N"
- Other models: pick up from wherever they were
- No re-annotation of the same (run_id, model_id) pair

- [ ] **Step 9: Commit**

```bash
git add src/compare_vlm.py
git commit -m "feat: model-major batch orchestration with resumability (#32)

Refactor compare_vlm.py from image-major (outer=images) to model-major
(outer=models). Per-model phase loads weights once, processes all
candidate images in micro-batches of K (default 25), evicts at end
via vlm_inference.evict_model().

Resumability: each model phase filters candidates against
get_unanalyzed_runs() unless --force. Each save_vlm_annotation auto-
commits, so an interrupted run resumes precisely from the last image.

New flags: --models (filter), --force (re-annotate), --batch-size (K).

Required for 16 GB VRAM hardware where no two VLMs fit simultaneously
(see ADR-003). Removes _print_comparison (no longer useful with
model-major ordering — see Task 6 for replacement summary table)."
```

---

## Task 6: Fix VRAM references in existing docs

**Files:**
- Modify: `docs/architecture-decisions.md` (one paragraph in ADR-003)
- Modify: `docs/research/issue-32-transformers-version-pinning/PROMPT.md` (Q3 model filter)

- [ ] **Step 1: Find the wrong 16 GB reference in ADR-003**

Open `docs/architecture-decisions.md`. Find the MiniCPM-o "Possible paths" list. Look for "(a) Disable bnb 4-bit for MiniCPM-o (load in bf16 — ~16 GB VRAM)".

- [ ] **Step 2: Strike that path explicitly**

Replace that line with:

```
- (a) ~~Disable bnb 4-bit and load in bf16~~ — not viable on this hardware.
  An 8B model in bf16 needs ~16 GB just for weights, leaving zero room for
  KV cache on the 15.92 GB RTX 5080 Laptop. Reserved for a future workstation
  upgrade or running MiniCPM-o on CPU (very slow but possible).
```

- [ ] **Step 3: Find the 24 GB filter in PROMPT.md**

Open `docs/research/issue-32-transformers-version-pinning/PROMPT.md`. Find the line "Run locally on **≤24 GB VRAM**".

- [ ] **Step 4: Annotate the over-broad filter**

Replace with:

```
- Run locally on **≤16 GB VRAM** in 4-bit quantization (RTX 5080 Laptop
  budget — corrected from the original "≤24 GB" filter used in the
  first deep-research pass; revisit any Q3 model recommendations
  exceeding 16 GB peak with that constraint in mind)
```

- [ ] **Step 5: Commit**

```bash
git add docs/architecture-decisions.md docs/research/issue-32-transformers-version-pinning/PROMPT.md
git commit -m "docs: fix VRAM references — RTX 5080 Laptop is 16 GB not 24 GB (#32)

ADR-003 mentioned bf16 MiniCPM-o as a fallback at '~16 GB VRAM' —
that ties the hardware budget exactly without KV cache headroom, so
bf16 is not viable on this card. Marked the path as not viable on
current hardware with a note pointing to a future workstation upgrade.

PROMPT.md Q3 filter was '≤24 GB' which over-broadly admits models
that won't fit on this card. Updated to '≤16 GB' with a note that
any v5-only models surfaced by the first research pass need to be
re-checked against the tighter budget."
```

---

## Task 7: Add ADR-004 for orchestration architecture

**Files:**
- Modify: `docs/architecture-decisions.md` (insert ADR-004 at the top, above ADR-003)

- [ ] **Step 1: Open the file and find the top**

`docs/architecture-decisions.md` — the first ADR after the header is ADR-003 (newest first).

- [ ] **Step 2: Insert ADR-004 above ADR-003**

After the file header (line ~7) and before "## 2026-05-15 (eve, post-research)", insert:

```markdown
## 2026-05-15 (late eve, post-orchestration)

### ADR-004 — Model-major batch orchestration with DB-driven resumability

**Decision:** `src/compare_vlm.py` runs N images × M models in a model-major
loop: outer iterates models (one resident at a time), inner iterates images
in micro-batches of K (default 25). Each (run_id, model_id) annotation is
committed to `vlm_annotations` immediately via `save_vlm_annotation`.
Resumption uses `get_unanalyzed_runs(conn, model_id)` to skip already-done
pairs unless `--force` is passed.

**Why:** RTX 5080 Laptop has 15.92 GB VRAM. Each VLM in bnb 4-bit peaks at
~7-9 GB during inference (weights + KV cache + activations on a 2-image
prompt). No two models fit simultaneously. The previous image-major loop
with a never-evicting `_MODEL_CACHE` would OOM on the second image of any
multi-model run — it only worked in our testing because each test was a
fresh Python process.

Model-major also has the right cost shape: total cold-load cost is
**N_models × 1**, not **N_models × N_images**. For a 1000-image run that's
~2 minutes of swap overhead vs. **hours**.

**Why not image-major with LRU eviction:** Same swap cost as model-major
(must evict every model on every image anyway) with worse cache locality.
The only place image-major wins is when you need *all model responses for
image i before moving to image i+1* (e.g., cross-model voting). Our DB
records each (image, model) pair independently, so this isn't a constraint.

**Micro-batch boundary purpose:** Not for commit cadence (each save auto-
commits). The boundary exists to (a) call `torch.cuda.empty_cache()` to
release transient activation memory between iterations, and (b) emit a
progress line every K images. K=25 was picked as a balance between
progress visibility and overhead — adjust via `--batch-size` if a model
turns out to have unusual memory characteristics.

**Resumability invariant:** "If a row exists in `vlm_annotations` for
`(run_id, model_id)`, that pair is done." This is the only state the
orchestrator reads — there's no separate progress file, no journal, no
crash-recovery code. Kill the process at any point; on next invocation it
filters out the completed rows and continues. The `UNIQUE(run_id,
model_id)` constraint on the table makes this trivially correct.

**Failure handling:** Per-image `try/except` increments an error counter
and continues to the next image. Three exception classes:
- `FileNotFoundError` — source photo or render missing on disk
- `VLMSchemaError` — model emitted unparseable JSON twice (after retry)
- `Exception` — anything else (OOM, hardware fault, etc.) — full traceback
  is printed but the run continues.

Errors do NOT cause re-annotation on the next run — if a particular
`(run_id, model_id)` always fails, it will be retried every time. To
mark a pair as "permanently broken," insert a stub annotation manually
or use `--force` deliberately. Acceptable for our scale; revisit if we
see frequent persistent failures.

**Considerations for future choices:**

- The micro-batch boundary is also the right place to add per-batch
  metrics writes (mean latency, retry rate, etc.) if we want timeseries
  data later. Currently we only emit a progress print — adding metrics
  is a small change inside `_run_model_phase`.
- The `--force` flag is "ignore the DB filter, redo everything." A
  cheaper version would be "redo only failed annotations" — if we want
  that, add an `--errors-only` flag that filters to `(run_id, model_id)`
  pairs where the last annotation has an `[error]` rating. Easy follow-up.
- For multi-GPU workstations: this orchestrator is single-GPU. Adding
  GPU-parallel model phases (e.g., one model per GPU running concurrently)
  is a separate change in scope to a future ADR.

**Related:** #32 (transformers pinning + this orchestration),
[`src/compare_vlm.py`](../src/compare_vlm.py),
[`src/_batch_utils.py`](../src/_batch_utils.py),
[`src/vlm_inference.py`](../src/vlm_inference.py) (`evict_model`, `evict_all`)

---
```

- [ ] **Step 3: Verify the file still renders cleanly**

Open the file in a markdown previewer or just visually inspect — ADR-004 should appear above ADR-003, both above ADR-002 and ADR-001. Newest-first ordering is preserved.

- [ ] **Step 4: Commit**

```bash
git add docs/architecture-decisions.md
git commit -m "docs: add ADR-004 — model-major batch orchestration (#32)

Captures: why model-major over image-major on 16 GB, why micro-batches
are for cuda.empty_cache + progress not for commit cadence, the
'row exists -> done' resumability invariant, and the per-image
exception-handling contract. Cross-references compare_vlm.py and
_batch_utils.py."
```

---

## Task 8: Final integration smoke — full 4-model 10-image run

**Files:** none modified; this is an empirical validation gate.

- [ ] **Step 1: Clear any prior annotations for the test runs to make this a real cold run**

Optional — only if you want to see the full cold-load timing:

```bash
.venv/Scripts/python.exe -c "
from db import open_db
from pathlib import Path
conn = open_db(Path(r'D:/Temp/yogamann-output/yogamann.db'))
# Get the 10 most recent run_ids
rows = conn.execute('SELECT id FROM runs ORDER BY timestamp DESC LIMIT 10').fetchall()
ids = [r[0] for r in rows]
ph = ','.join(['?']*len(ids))
conn.execute(f'DELETE FROM vlm_annotations WHERE run_id IN ({ph})', ids)
conn.commit()
print(f'Cleared annotations for {len(ids)} runs')
"
```

Skip this step if you don't want to redo work — the resumability test in Task 5 covered the warm case.

- [ ] **Step 2: Run the full orchestration**

```bash
.venv/Scripts/python.exe src/compare_vlm.py --limit 10 --output-root D:/Temp/yogamann-output 2>&1 | tee /tmp/orchestration-test.log
```

(On Windows PowerShell, replace `tee /tmp/...` with `Tee-Object -FilePath C:\tmp\orchestration-test.log` or just omit and pipe to a file.)

Expected:
- 4 model phases run sequentially
- Each phase prints: header, per-batch progress, completion
- No OOM error
- Final summary shows 4 rows + TOTAL with ~40 successful annotations total (10 images × 4 models, minus any persistent errors)
- Total elapsed: roughly 10-15 minutes (cold loads + 40 inferences at ~8-15s each)

- [ ] **Step 3: Verify DB contents**

```bash
.venv/Scripts/python.exe -c "
from db import open_db, count_vlm_annotations
from pathlib import Path
conn = open_db(Path(r'D:/Temp/yogamann-output/yogamann.db'))
for m in ['qwen2_5_vl_7b', 'internvl2_5_8b', 'molmo_7b_d', 'minicpm_v_2_6']:
    print(f'{m}: {count_vlm_annotations(conn, m)} total annotations')
"
```

Expected: counts increased by ~10 each (the 10-image batch). If a model errored on all 10, that's a real bug — investigate before declaring success.

- [ ] **Step 4: Check VRAM stayed bounded during the run**

If you had `nvidia-smi` running in a separate terminal during the test, peak GPU memory should have been ~9 GB (one model at a time + KV cache). Between phases, memory should briefly drop to <1 GB before the next model loads. If you saw memory climb past 12 GB at any point, `evict_model` isn't doing its job and that needs investigation.

- [ ] **Step 5: Commit the test log (optional)**

If you want a record of the smoke run, save the log:

```bash
mkdir -p docs/superpowers/test-logs
cp /tmp/orchestration-test.log docs/superpowers/test-logs/2026-05-15-orchestration-smoke.log
git add docs/superpowers/test-logs/2026-05-15-orchestration-smoke.log
git commit -m "test: orchestration smoke run log — 4 models × 10 images (#32)"
```

Otherwise skip the commit; the empirical test was the deliverable.

---

## Task 9: Push branch and update PM files

**Files:**
- Modify: `.claude/pm/PROGRESS.md` (prepend new session entry)
- Modify: `.claude/pm/PENDING-TASK.md` (clear or update with MiniCPM-o next step)

- [ ] **Step 1: Push the branch**

```bash
git push 2>&1 | tail -3
```

Expected: branch pushed to `origin/feature/v4-downgrade-testing` (or wherever this work is landing). No errors.

- [ ] **Step 2: Append to PROGRESS.md**

Use `Edit` to prepend after the file header (do NOT read the file first — see CLAUDE.md PROGRESS.md rules):

Anchor: `# PROGRESS\n\n## 2026-05-15 (eve, post-research)`

Insert after the anchor:

```markdown

## 2026-05-15 (late eve) — VLM batch orchestration refactor (#32)

- Inverted `compare_vlm.py` loop from image-major to model-major. Each model phase: load → annotate K-image micro-batches → evict. Total cold-load cost is now N_models, not N_models × N_images.
- Added `evict_model(key)` + `evict_all()` to `vlm_inference.py` — public eviction so callers manage GPU memory between phases. Required for 16 GB VRAM hardware where no two VLMs fit simultaneously.
- DB-driven resumability: each model phase filters candidates against `get_unanalyzed_runs(conn, model_id)` to skip already-done pairs. `--force` overrides. Each `save_vlm_annotation` auto-commits, so kill-and-resume picks up precisely from the last completed image.
- New CLI flags on `compare_vlm.py`: `--models`, `--force`, `--batch-size`.
- New module `src/_batch_utils.py` with `chunked()` + `format_summary()`, pytest-covered.
- New DB helper `count_vlm_annotations(conn, model_id)` for per-model "done" counts in summary output.
- Architecture-decisions ADR-004 added. ADR-003 + PROMPT.md corrected: 15.92 GB VRAM, not 24 GB (RTX 5080 Laptop, not desktop class).
- Smoke tested: 4 models × 10 images runs end-to-end without OOM; resumability verified (re-run skips done pairs); `--force` verified to re-annotate.
```

- [ ] **Step 3: Update PENDING-TASK.md**

Use `Write` (after `Read`) to set the active task to MiniCPM-o investigation:

```markdown
# PENDING TASK

_Pre-MiniCPM-o orchestration refactor landed. Branch ready for user-led MiniCPM-o research._

## Active context for MiniCPM-o investigation (issue #32 follow-up)

**Working models on transformers==4.49.0:** Qwen2.5-VL, InternVL2.5-MPO, Molmo-7B-D, MiniCPM-V-2.6. Orchestrator can now run all 4 in one batch.

**Blocked model:** MiniCPM-o-2.6. Failure: `NotImplementedError: "normal_kernel_cpu" not implemented for 'Byte'` in `transformers/models/qwen2/modeling_qwen2.py:385`. Root cause: MiniCPM-o uses a Qwen2 audio-language backbone; transformers' native Qwen2 `_init_weights` calls `.normal_()` on bnb-quantized uint8 weights, which has no kernel implementation.

**Search queries to run (user-led):**
1. `"normal_kernel_cpu" "Byte" MiniCPM-o bitsandbytes 4-bit`
2. HF discussions: https://huggingface.co/openbmb/MiniCPM-o-2_6/discussions (filter "bitsandbytes" / "4bit" / "quantization")
3. OpenBMB official inference: https://github.com/OpenBMB/MiniCPM-o (requirements.txt, examples)
4. Fallback: search for `MiniCPM-o-2_6 bf16 inference` — bf16 path would sidestep bnb entirely

**Constraints when evaluating proposed solutions:**
- ~15 GB peak VRAM budget on RTX 5080 Laptop (bf16 8B weights = ~16 GB → not viable)
- Cannot change transformers==4.49.0 pin without breaking the other 4 models
- `bnb_4bit_skip_modules` would need exact module names from MiniCPM-o's architecture

## Last commit
- `<commit-hash>` — feat: model-major batch orchestration with resumability (#32)
```

- [ ] **Step 4: Commit PM updates**

```bash
git add .claude/pm/PROGRESS.md .claude/pm/PENDING-TASK.md
git commit -m "docs: append PROGRESS + update PENDING-TASK for MiniCPM-o handoff (#32)"
git push 2>&1 | tail -3
```

---

## Acceptance gate

Before declaring this plan complete, all of these must be true:

1. `compare_vlm.py --limit 3` (4 models) runs end-to-end without OOM ✅
2. Re-running the same command shows "already done" and exits in <5s ✅
3. `--force` re-annotates 3 images × 4 models successfully ✅
4. `nvidia-smi` shows VRAM drop to <1 GB between model phases ✅
5. `pytest tests/test_batch_utils.py` passes ✅
6. ADR-003, ADR-004, PROMPT.md all reference 15.92 / 16 GB (not 24 GB) ✅
7. PROGRESS.md has the new session entry; PENDING-TASK.md points to MiniCPM-o ✅
8. Branch pushed to origin ✅
