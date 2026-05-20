# ControlNet Strength Sweep

**3 nodes · Community 24 · Cohesion 0.67**

`profiles/yoga_asana_cs_sweep.yml` · `profiles/yoga_asana_strong.yml`

---

## What it is

Two render profiles used to tune `cond_scale` — the ControlNet conditioning strength that controls how closely the SDXL render follows the DWPose skeleton.

## The parameter

`cond_scale` (ControlNet conditioning scale) balances skeleton fidelity vs. image quality:
- **Too low** (< 0.8): render ignores the skeleton, pose is incorrect
- **Too high** (> 2.0): render over-constrains the skeleton, produces artifacts and stiff-looking renders
- **Sweet spot** (1.0–1.5): good pose fidelity with natural-looking output

## The two profiles

| Profile | `cond_scale` | Purpose |
|---------|-------------|---------|
| `yoga_asana_cs_sweep` | 1.0–2.0 linear sweep | Systematic exploration to find the optimal value per pose type |
| `yoga_asana_strong` | 2.0 fixed | Tests the upper bound — useful for poses that need strong skeleton guidance (e.g. handstands) |

## Connects to

- [Mannequin Generator](mannequin-generator.md) — `make_mannequin.py` reads these profiles
- [Pose Pipeline Research](pose-pipeline-research.md) — sweep results inform which profiles to use in production
