# Progress Monitor Script

**9 nodes · Community 16 · Cohesion (unnamed)**

`scripts/monitor.ps1`

---

## What it does

A PowerShell terminal dashboard that shows live GPU and CPU stats while a batch inference run is in progress. Displays a refreshing single-screen view with GPU utilisation, VRAM usage, temperature, power draw, and CPU percentage — without needing nvitop installed.

## Key functions

| Function | Role |
|----------|------|
| `Write-Metric(label, value, unit)` | Renders one metric row with a label and value |
| `Get-Bar(value, max, width)` | Draws an ASCII progress bar |
| `Update-GpuTimer()` | Reads GPU stats via `nvidia-smi` and updates the dashboard |
| `Format-Dur(seconds)` | Formats an elapsed time as `HH:MM:SS` |
| `Move-Up(n)` | Moves the cursor up n lines (for in-place refresh) |
| `Get-ProcCpuPct(pid)` | Reads per-process CPU percentage |
| `Write-Sep()` | Draws a separator line |
| `Get-ScriptName()` | Returns the script's own filename for the title bar |

## Usage

Run in a separate terminal while `compare_vlm.py` or `analyze.py` is running:

```powershell
.\scripts\monitor.ps1
```

Refreshes every few seconds. Press `Ctrl-C` to stop.

## Connects to

- [Batch Orchestration](batch-orchestration.md) — the process being monitored
- [GPU Stability & Memory Management](gpu-stability.md) — the GPU metrics it surfaces
