# Batch Lock (Process Safety)

**14 nodes · Community 12 · Cohesion 0.21**

`src/batch_lock.py`

---

## What it does

A file-based mutex that prevents two concurrent batch inference runs from fighting over the GPU. When `compare_vlm.py` or `analyze.py` starts, it calls `acquire()` to write a lock file containing the current PID. Any subsequent process that calls `check()` or `is_locked()` will see the lock and abort.

The lock is self-healing: if the process that wrote the lock is no longer alive (crash, kill), `check()` auto-deletes the stale lock file.

## Key functions

| Function | Role |
|----------|------|
| `acquire(output_root)` | Writes the lock file to `output_root/.batch_lock.json`. Returns the lock file path. |
| `release(output_root)` | Deletes the lock file. Safe to call if the file doesn't exist. |
| `check(output_root)` | Returns the lock dict if a live process holds the lock, else `None`. Auto-deletes stale locks. |
| `is_locked(output_root)` | Convenience bool wrapper around `check()`. |
| `_pid_alive(pid)` | Returns `True` if a process with the given PID is currently running. |
| `_lock_path(output_root)` | Returns the canonical lock file path. |
| `_safe_unlink(path)` | Deletes a file, ignoring `FileNotFoundError`. |

## Why a file lock instead of a Python lock?

Multiple Python processes (different terminal sessions, the gallery spawning inference) need to coordinate. In-process locks don't work across process boundaries. A JSON file in the output directory is visible to all processes and survives crashes.

## Connects to

- [Batch Orchestration](batch-orchestration.md) — acquires lock at start, releases at end
- [Single-Run Analyzer](single-run-analyzer.md) — also checks the lock before starting
