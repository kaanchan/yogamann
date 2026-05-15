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
