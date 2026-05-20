# RTX 5080 Diagnostics

**4 nodes · Community 19**

`scripts/rtx5080-test.py`

---

## What it does

A one-off benchmark and smoke-test script written to validate the RTX 5080 after installation. Confirms the GPU is visible to CUDA, measures memory bandwidth and compute throughput, and verifies that yogamann's core imports load without errors.

## Key functions

| Function | Role |
|----------|------|
| `run_benchmark()` | Runs a matrix multiply benchmark to measure TFLOPS and memory bandwidth |
| `check_yogamann_imports()` | Imports `vlm_inference`, `db`, `gallery`, etc. and reports any failures |
| `main()` | Calls both and prints a summary report |

## Status

This is a diagnostic utility, not part of the production pipeline. Safe to run at any time as a health check after environment changes.
