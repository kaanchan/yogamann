"""src/batch_lock.py — GPU-busy lock file for yogamann batch inference.

Prevents simultaneous terminal (compare_vlm.py) and gallery-spawned inference
jobs from both loading models onto the same GPU.

Lock file location: {output_root}/.batch_lock.json
Lock dict schema:
    {
        "pid":        int,    # owning process PID
        "job_type":   str,    # "batch" | "gallery"
        "model_ids":  list,   # models being processed
        "start_time": str,    # ISO-8601 UTC
    }
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _lock_path(output_root: str | Path) -> Path:
    return Path(output_root) / ".batch_lock.json"


def acquire(
    output_root: str | Path,
    job_type: str,
    model_ids: list[str],
    pid: int | None = None,
) -> Path:
    """Write the lock file and return its path.

    Args:
        output_root: Directory that contains yogamann.db.
        job_type:    "batch" for compare_vlm.py runs, "gallery" for Streamlit-spawned jobs.
        model_ids:   Model keys being processed.
        pid:         Owning PID; defaults to os.getpid().
    """
    payload = {
        "pid":        pid if pid is not None else os.getpid(),
        "job_type":   job_type,
        "model_ids":  model_ids,
        "start_time": datetime.now(timezone.utc).isoformat(),
    }
    path = _lock_path(output_root)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def release(output_root: str | Path) -> None:
    """Delete the lock file. Safe to call if it does not exist."""
    try:
        _lock_path(output_root).unlink()
    except FileNotFoundError:
        pass


def check(output_root: str | Path) -> dict[str, Any] | None:
    """Return the lock dict if a live process holds it, else None.

    Auto-deletes the lock file when the recorded PID is no longer alive
    (stale lock from a crash or hard kill).
    """
    path = _lock_path(output_root)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        _safe_unlink(path)
        return None

    pid = data.get("pid")
    if pid is None:
        return None

    if _pid_alive(pid):
        return data

    _safe_unlink(path)
    return None


def is_locked(output_root: str | Path) -> bool:
    """Convenience bool wrapper around check()."""
    return check(output_root) is not None


# ── Internal ───────────────────────────────────────────────────────────────────

def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _pid_alive(pid: int) -> bool:
    """Return True if the process with pid is currently running."""
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        pass

    import sys
    if sys.platform == "win32":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not handle:
            return False
        still_active = ctypes.windll.kernel32.WaitForSingleObject(handle, 0) != 0
        ctypes.windll.kernel32.CloseHandle(handle)
        return still_active
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
