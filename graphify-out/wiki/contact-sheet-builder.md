# Contact Sheet Builder

**25 nodes · Community 2 · Cohesion 0.11**

`src/create_contact_sheet.py` · `src/dwpose_onnx.py` · `src/extract_pose.py`

---

## What it does

Builds side-by-side comparison images for manual review. Given a source yoga photo and a mannequin render, it produces either a 2-up (source | render) or 4-up (source | skeleton | render | overlay) layout. This community also owns the DWPose ONNX detector and the pose extraction utility.

## Layouts

| Function | Layout |
|----------|--------|
| `build_comparison_2()` | 2-up: source photo and mannequin render side by side |
| `build_comparison_4()` | 4-up: source · DWPose skeleton · render · overlay |

## DWPose ONNX integration

`src/dwpose_onnx.py` provides a drop-in replacement for `controlnet_aux.DWposeDetector` that runs the detector via ONNX runtime rather than PyTorch. Faster and avoids a PyTorch version dependency for inference-only use.

`_render()` in `dwpose_onnx.py` converts the rtmlib 133-keypoint output back to the OpenPose skeleton image format, replicating the visual output expected by downstream tools.

## Pose extraction utility

`src/extract_pose.py` is a standalone CLI tool: given an image path, it runs DWPose and writes:
1. A transparent PNG with the skeleton overlaid
2. A `.metrics.json` sidecar with keypoint coordinates

The sidecar is what `db.py`'s `ingest_json()` reads to create run records.

## Connects to

- [Database — Run & Annotation Queries](database-queries.md) — `_meta()` reads `.metrics.json` sidecars
- [Review Gallery UI](review-gallery-ui.md) — contact sheets are what gallery cards display
- [Pose Pipeline Research](pose-pipeline-research.md) — DWPose is the current recommended pose estimator
