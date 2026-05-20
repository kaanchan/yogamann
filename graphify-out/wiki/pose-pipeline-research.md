# Pose Pipeline Research

**23 nodes · Community 3 · Cohesion 0.12**

`README.md` · `docs/observations/` · `docs/research/issue-33-pose-pipeline-evaluation/` · `.claude/pm/`

---

## What it is

The research and observation corpus for the overall pose pipeline. This community contains the project overview (README), pipeline observations, and the Issue #33 research materials that documented the current system's limitations and recommended a future upgrade path.

## Current pipeline (as documented in README)

```
Source photo → DWPose skeleton detection → ControlNet OpenPose SDXL → SDXL render → VLM comparison
```

1. **DWPose** extracts a 2D skeleton from the source yoga photo
2. **ControlNet + SDXL** renders a 3D wooden mannequin conditioned on that skeleton
3. **VLM** compares the render to the source photo and produces a structured rating

## Pipeline observations

Two systematic failure modes were documented in `docs/observations/pipeline-observations.md`:

- **OBS-001: Outdoor Complex Backgrounds** — backgrounds with trees, walls, and uneven lighting cause DWPose to misdetect skeleton points, producing distorted renders
- **OBS-002: Prone Back-Bends** — poses where the spine bends backward (wheel pose, camel pose) are not captured correctly — the skeleton flattens the arc

These observations are also stored machine-readable in `observations.json`.

## Reference targets

`docs/ideal-targets/` contains reference images of what a good mannequin render looks like — a photo of a standard wooden mannequin (seated pose, dark background) and an ink cell drawing of multiple mannequin poses. These were semantically linked to each other by the graph (same subject, different medium).

## Issue #33 research

The Issue #33 research corpus (prompt, ChatGPT deep research response, and SYNTHESIS) evaluated alternatives to the current pipeline:

- **Pose estimators evaluated:** DWPose (recommended), Sapiens (high accuracy, NC license), RTMPose (real-time), OpenPose (current baseline)
- **Renderers evaluated:** FLUX.1-dev + ControlNet Union (best quality), HSMR + Headless Blender (parametric, no diffusion artifacts), SMPL-X + Blender (similar), current SDXL

The SYNTHESIS document is the 9th most connected node in the entire graph.

## Connects to

- [Pose Estimator Alternatives](pose-estimator-alternatives.md) — the detailed evaluated alternatives
- [VLM Model Registry](vlm-model-registry.md) — research informed model selection
- [Architecture Decisions & ADRs](architecture-decisions.md) — Issue #33 is tracked in ADR community
