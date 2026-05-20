# Windows Job Objects

**9 nodes · Community 15 · Cohesion 0.25**

`src/win_job.py`

---

## What it does

A thin Win32 wrapper that creates a Windows Job Object and assigns child processes to it. The job object is configured with `KILL_ON_JOB_CLOSE` — when the parent Python process exits (normally or via crash), all child processes in the job are automatically killed.

This prevents orphaned inference subprocesses from continuing to consume GPU memory after the parent script has exited.

## Key functions

| Function | Role |
|----------|------|
| `create_job(name)` | Creates (or opens) a named Windows Job Object with `KILL_ON_JOB_CLOSE`. Returns the job handle. |
| `assign_process(job, pid)` | Assigns the process with the given PID to the job object. Returns `True` on success. |

## ctypes structs

The module defines three `ctypes.Structure` subclasses that mirror the Win32 API:
- `_JOBOBJECT_BASIC_LIMIT_INFORMATION`
- `_IO_COUNTERS`
- `_JOBOBJECT_EXTENDED_LIMIT_INFORMATION`

These are needed to call `SetInformationJobObject` with the kill-on-close flag.

## When it's used

`compare_vlm.py` (and the gallery's inference subprocess spawner) assigns spawned inference workers to a job object so they're guaranteed to clean up even on hard exits. Particularly important on Windows where `Ctrl-C` can leave GPU memory occupied.

## Connects to

- [Batch Orchestration](batch-orchestration.md) — assigns inference worker PIDs to the job
- [Review Gallery UI](review-gallery-ui.md) — gallery-spawned inference subprocess is also assigned
