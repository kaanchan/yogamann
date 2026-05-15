"""src/gpu_monitor.py — lightweight GPU + CPU sampler for long-running batch jobs.

Uses nvidia-smi subprocess (no pynvml install needed) and psutil for CPU.
All public functions are safe to call even if no NVIDIA GPU is present — they
return None fields rather than raising.
"""
from __future__ import annotations

import subprocess
import time
from datetime import datetime, timezone
from typing import Optional

import psutil

# Keep one primed psutil call so first real poll returns a valid delta.
psutil.cpu_percent(interval=None)


def sample() -> dict:
    """Return a snapshot of GPU and CPU metrics.

    Keys: ts, gpu_temp_c, gpu_util_pct, gpu_mem_used_mb, gpu_mem_total_mb,
          gpu_power_w, cpu_util_pct.
    GPU fields are None if nvidia-smi is unavailable.
    """
    gpu = _query_nvidia_smi()
    cpu = psutil.cpu_percent(interval=None)
    return {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gpu_temp_c":       gpu.get("temp"),
        "gpu_util_pct":     gpu.get("util"),
        "gpu_mem_used_mb":  gpu.get("mem_used"),
        "gpu_mem_total_mb": gpu.get("mem_total"),
        "gpu_power_w":      gpu.get("power"),
        "cpu_util_pct":     cpu,
    }


def cooldown_if_hot(threshold_c: float, sleep_s: float, quiet: bool = False) -> bool:
    """Block for sleep_s seconds if GPU temp is at or above threshold_c.

    Prints a countdown every 10 seconds. Returns True if a cooldown was applied.
    """
    metrics = sample()
    temp = metrics.get("gpu_temp_c")
    if temp is None or temp < threshold_c:
        return False

    if not quiet:
        print(
            f"  [gpu_monitor] GPU at {temp:.0f}°C ≥ limit {threshold_c:.0f}°C — "
            f"cooling down for {sleep_s:.0f}s …"
        )
    deadline = time.monotonic() + sleep_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if not quiet and remaining % 10 < 1:
            cur = _query_nvidia_smi().get("temp")
            print(f"  [gpu_monitor] {remaining:.0f}s remaining, GPU now {cur}°C")
        time.sleep(min(1.0, remaining))

    if not quiet:
        cur = _query_nvidia_smi().get("temp")
        print(f"  [gpu_monitor] cooldown done, GPU now {cur}°C")
    return True


def fmt_metrics(m: dict) -> str:
    """One-line human-readable summary for console output."""
    temp = f"{m['gpu_temp_c']:.0f}°C" if m["gpu_temp_c"] is not None else "n/a"
    util = f"{m['gpu_util_pct']:.0f}%" if m["gpu_util_pct"] is not None else "n/a"
    mem  = (
        f"{m['gpu_mem_used_mb']:.0f}/{m['gpu_mem_total_mb']:.0f}MB"
        if m["gpu_mem_used_mb"] is not None else "n/a"
    )
    cpu  = f"{m['cpu_util_pct']:.0f}%"
    return f"gpu={temp} util={util} vram={mem} cpu={cpu}"


# ── Internal ───────────────────────────────────────────────────────────────────

def _query_nvidia_smi() -> dict:
    """Run nvidia-smi once; parse temp/util/mem/power. Returns {} on failure."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return {}
        parts = [p.strip() for p in out.stdout.strip().split(",")]
        return {
            "temp":      float(parts[0]),
            "util":      float(parts[1]),
            "mem_used":  float(parts[2]),
            "mem_total": float(parts[3]),
            "power":     float(parts[4]) if parts[4] not in ("[N/A]", "N/A") else None,
        }
    except Exception:
        return {}
