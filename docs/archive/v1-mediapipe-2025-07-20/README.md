# Archive: v1 MediaPipe Baseline (2025-07-20)

This directory preserves the exact source code used to generate the v1 baseline
outputs, committed retroactively to match the original generation date.

## Infrastructure at time of generation

| Component | v1 (this archive) | Current (2026-05) |
|---|---|---|
| SD model | `runwayml/stable-diffusion-v1-5` (SD 1.5) | `stabilityai/stable-diffusion-xl-base-1.0` (SDXL) |
| ControlNet | `lllyasviel/sd-controlnet-openpose` (SD 1.5) | `xinsir/controlnet-openpose-sdxl-1.0` (SDXL) |
| Pose detection | **MediaPipe** `mp.solutions.pose` — 33 landmarks | **DWPose ONNX** (`controlnet_aux`) — 133 keypoints |
| Python | 3.10 (conda) | 3.13 (uv venv) |
| Env manager | conda (`environment.yml`) | uv (`pyproject.toml`) |

## Profile parameters

See `profiles/yoga_asana.yml` in this directory.
See also `profiles/yoga_asana_v1_archival.yml` in the repo root `profiles/`
for a full annotated comparison against current parameters.

## Reference outputs

Generated outputs from this codebase are preserved in:
`docs/reference-outputs/v1-baseline/` — samples 1, 3, 4, 5 (mannequins + contact sheets)
`docs/reference-outputs/batch-feature/` — sample 1 from the profile-batch feature branch

## Reproduction notes

To reproduce these outputs exactly would require:
1. The conda environment defined in `environment.yml`
2. The MediaPipe-based `src/extract_pose.py` in this directory
3. SD 1.5 weights: `runwayml/stable-diffusion-v1-5`
4. ControlNet weights: `lllyasviel/sd-controlnet-openpose`
5. Profile: `profiles/yoga_asana.yml` (steps=30, cond_scale=1.0, guidance=7.5)
6. seed=108 for sample-1 exactly; other samples used random seeds

Note: The visual character of v1 outputs reflects three compounding factors:
MediaPipe's 33-point skeleton geometry, SD 1.5's specific aesthetic, and the
original prompt/parameter values — all of which changed in the current pipeline.
