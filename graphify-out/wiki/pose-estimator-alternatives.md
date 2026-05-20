# Pose Estimator Alternatives

**15 nodes · Community 10 · Cohesion 0.17**

`docs/research/issue-33-pose-pipeline-evaluation/SYNTHESIS.md`

---

## What it is

The evaluated landscape of pose estimators and renderers from the Issue #33 deep research. These are not implemented in yogamann yet — they represent the recommended upgrade path from the current DWPose + SDXL baseline.

## Pose estimators

| Tool | Status | Notes |
|------|--------|-------|
| **DWPose** | Current + recommended | 133-keypoint whole-body. Already used in production via `src/dwpose_onnx.py`. Best balance of accuracy and speed for this use case. |
| **OpenPose** | Current baseline (2D) | The original skeleton format. Still the ControlNet conditioning format — DWPose output is converted to OpenPose format before SD conditioning. |
| **Sapiens (Meta)** | Evaluated, not adopted | Higher accuracy than DWPose but NC (non-commercial) license. Would require licensing for any commercial use. |
| **RTMPose (OpenMMLab)** | Evaluated, not adopted | Designed for real-time exercise tracking. Good accuracy, MIT license, but no whole-body finger/face keypoints. |
| **MediaPipe** | Superseded (v1 baseline) | 33 landmarks, no finger or face detail. Used in the v1 archival profile (`yoga_asana_v1_archival.yml`). |

## Renderers

| Tool | Status | Notes |
|------|--------|-------|
| **SDXL + ControlNet OpenPose** | Current baseline | Fast, already integrated. Produces diffusion artifacts that can confuse VLM. |
| **FLUX.1-dev + ControlNet Union** | Recommended next step | Better prompt adherence, higher fidelity. Requires more VRAM. |
| **HSMR + Headless Blender** | Recommended parametric path | Lifts 3D SMPL-X mesh from 2D photo, renders in Blender. No diffusion — deterministic, no artifacts. Higher setup cost. |
| **SMPL-X + Blender** | Evaluated | Similar to HSMR but requires manual rigging. Less automated. |

## Recommended future stack

The SYNTHESIS document recommends: **DWPose** (keep) → **HSMR + Blender** (replace SDXL) → **VLM evaluation** (keep).

The graph captures this as a hyperedge: *Recommended Pose Pipeline Stack: DWPose + HSMR/Blender + VLM Evaluation*.

## Connects to

- [Pose Pipeline Research](pose-pipeline-research.md) — the research corpus this was extracted from
- [VLM Inference Core](vlm-inference-core.md) — VLM evaluation is the downstream consumer
- [Contact Sheet Builder](contact-sheet-builder.md) — DWPose is used in `dwpose_onnx.py` there
