# Mannequin Generator

**10 nodes · Community 14 · Cohesion 0.31**

`src/make_mannequin.py`

---

## What it does

Batch-plans and drives `src/sd_make.py` to render 3D wooden mannequin poses from source images. Given a profile YAML and a directory of source photos, it expands parameter sweeps, assigns output filenames, and queues the render tasks. Acts as the batch planner layer above the actual Stable Diffusion render.

## Key functions

| Function | Role |
|----------|------|
| `build_tasks(profile, images)` | Expands profile parameters into a list of (source, params, output_stem) tasks |
| `_read_profile(path)` | Loads a profile YAML, guessing extension and format if omitted |
| `expand_value(val)` | Expands a parameter value — handles scalar, list, and `linspace(start, stop, n)` |
| `linspace(start, stop, n)` | Generates a linear sweep of n values between start and stop |
| `next_free_stem(dir, stem)` | Finds the next available output filename (adds `_v2`, `_v3` etc. if file exists) |
| `pad_version(n)` | Zero-pads version numbers for consistent sort order |
| `_detect_format(path)` | Detects whether a profile is YAML or JSON |

## Profiles

The render profiles live in `profiles/` and control Stable Diffusion parameters. The mannequin generator reads these to know what renders to produce:

- `yoga_asana.yml` — baseline SDXL parameters
- `yoga_asana_batch_1.yml` — random parameter sweep
- `yoga_asana_batch_2.yml` — linear parameter sweep
- `yoga_asana_cs_sweep.yml` — `cond_scale` sweep (ControlNet conditioning strength)
- `yoga_asana_strong.yml` — high `cond_scale: 2.0`

## Connects to

- [Pose Pipeline Research](pose-pipeline-research.md) — profile configs are part of the pipeline docs
- [ControlNet Strength Sweep](controlnet-strength-sweep.md) — the sweep profiles
