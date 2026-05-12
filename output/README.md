# output/ — Local Development Output

**Last updated: 2026-05-12**

---

This directory holds files generated during local development and test runs.
It is **gitignored** (except this README) — nothing here is committed to the repository.

## What goes here

| File pattern | Written by | Represents |
|---|---|---|
| `*.png` | `src/make_mannequin.py` | Mannequin renders, pose masks, contact sheets for local test runs |
| `*.metrics.json` | `src/make_mannequin.py` | Per-render metadata (seed, steps, profile, timing) |
| `*.controlnet.json` | `src/make_mannequin.py` | Raw DWPose keypoint coordinates |
| `index.html` | `src/make_gallery.py` | Browser gallery built from the PNGs in this directory |

## How it is populated

**Single test run** (via `make.ps1` or `make`):
```powershell
.\make.ps1                  # processes input/yoga-pose-sample-4.jpg
.\make.ps1 -Target test-all # processes all input/ images
```

**Gallery rebuild** (reads existing PNGs, writes `index.html`):
```powershell
.\make.ps1 -Target gallery
```

## Notes

- For full library batch runs, output goes to `D:\Temp\yogamann-output` (set via
  `-OutputRoot` in `scripts/batch.ps1`) — not this directory.
- The `output/` directory is created automatically on first run if it doesn't exist.
- Contents can be safely deleted at any time; re-run the pipeline to regenerate.
