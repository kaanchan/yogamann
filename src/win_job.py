"""src/win_job.py — Windows Job Object wrapper for auto-kill of child processes.

When a Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE is assigned to child
processes, the OS kills them all automatically when the last handle to the job
closes — i.e., when the Streamlit server process itself exits.

All public functions are no-ops on non-Windows so this module is safe to import
unconditionally.
"""
from __future__ import annotations

import sys

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    import ctypes
    import ctypes.wintypes as _wt

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", _wt.LARGE_INTEGER),
            ("PerJobUserTimeLimit",     _wt.LARGE_INTEGER),
            ("LimitFlags",             _wt.DWORD),
            ("MinimumWorkingSetSize",  ctypes.c_size_t),
            ("MaximumWorkingSetSize",  ctypes.c_size_t),
            ("ActiveProcessLimit",     _wt.DWORD),
            ("Affinity",               ctypes.POINTER(ctypes.c_ulong)),
            ("PriorityClass",          _wt.DWORD),
            ("SchedulingClass",        _wt.DWORD),
        ]

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount",  ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount",   ctypes.c_uint64),
            ("WriteTransferCount",  ctypes.c_uint64),
            ("OtherTransferCount",  ctypes.c_uint64),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo",                _IO_COUNTERS),
            ("ProcessMemoryLimit",    ctypes.c_size_t),
            ("JobMemoryLimit",        ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed",     ctypes.c_size_t),
        ]

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _JobObjectExtendedLimitInformation   = 9
    _PROCESS_ALL_ACCESS                  = 0x1F0FFF
    _k32 = ctypes.windll.kernel32


def create_job(name: str = "yogamann-inference") -> object | None:
    """Create (or open) a named Windows Job Object with KILL_ON_JOB_CLOSE.

    Returns the HANDLE (kept alive by the caller), or None on non-Windows / failure.
    The caller must hold this handle for the Streamlit server's lifetime.
    """
    if not IS_WINDOWS:
        return None
    try:
        job = _k32.CreateJobObjectW(None, name)
        if not job:
            return None

        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        if not _k32.QueryInformationJobObject(
            job, _JobObjectExtendedLimitInformation,
            ctypes.byref(info), ctypes.sizeof(info), None,
        ):
            _k32.CloseHandle(job)
            return None

        info.BasicLimitInformation.LimitFlags |= _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

        if not _k32.SetInformationJobObject(
            job, _JobObjectExtendedLimitInformation,
            ctypes.byref(info), ctypes.sizeof(info),
        ):
            _k32.CloseHandle(job)
            return None

        return job
    except Exception:
        return None


def assign_process(job: object, pid: int) -> bool:
    """Assign the process identified by pid to the job object.

    Returns True on success, False on non-Windows or any failure.
    """
    if not IS_WINDOWS or job is None:
        return False
    try:
        proc = _k32.OpenProcess(_PROCESS_ALL_ACCESS, False, pid)
        if not proc:
            return False
        result = bool(_k32.AssignProcessToJobObject(job, proc))
        _k32.CloseHandle(proc)
        return result
    except Exception:
        return False
