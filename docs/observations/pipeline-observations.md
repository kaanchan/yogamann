# Pipeline Observations

Living research log for model behaviour, pose detection accuracy, and rendering quality findings.
Use this to inform profile tuning, prompt engineering, and engine selection decisions.

Newest entries at top. Each entry: what was observed, likely root cause, and what to try.

---

## 2026-05-12

### OBS-002 — Prone back-bends not captured correctly by pose estimator

**Observed:** Asanas with prone back-bends (e.g. Bhujangasana / Cobra, Dhanurasana / Bow,
Ustrasana / Camel) produce outputs where the pose is wrong or the figure is distorted.

**Likely cause:** DWPose / OpenPose struggle with:
- Limb occlusion in prone orientation (arms hidden under torso)
- Unusual body plane (face-down vs standing/seated baseline the model was trained on)
- Keypoint mirroring or swapped left/right limbs

**What to try:**
- Compare DWPose vs MediaPipe skeleton output for this pose family — MediaPipe sometimes
  handles prone poses better due to full-body holistic model
- Test lower `cond_scale` (e.g. 1.0–1.2) so the diffusion model can self-correct a
  partially wrong skeleton rather than reproduce it faithfully
- Consider a dedicated profile (`yoga_asana_prone`) with adjusted conditioning
- Flag these asanas in the gallery for a separate review pass

---

### OBS-001 — Outdoor / complex backgrounds produce multiple artifacts

**Observed:** Source images photographed outdoors or against busy, texture-heavy backgrounds
(foliage, uneven lighting, patterned floors) produce outputs with:
- Multiple unwanted surface textures blending into the figure
- Incorrect limb positions
- Background detail bleeding into the rendered mannequin

**Likely cause:** ControlNet conditioning competes with the background signal during
diffusion; the model fills non-pose regions with textures drawn from the source image
context rather than neutral studio space. The pose skeleton may also be noisier when
extracted from a cluttered background.

**What to try:**
- Pre-process outdoor source images with a background removal / subject isolation step
  (e.g. `rembg`, SAM segmentation mask) before pose extraction
- Test higher `cond_scale` (e.g. 1.75–2.0) to force the model to prioritise the pose
  signal over background noise
- Add `outdoor background, natural background, grass, foliage, floor texture` to
  `neg_prompt` as a targeted suppression
- Flag outdoor-sourced images as a category in the gallery for separate evaluation

---

## How to use this file

- **Pose detection issues** → check OBS entries tagged "pose estimator"; consider switching
  to an alternative backbone or adjusting `cond_scale` downward
- **Rendering / diffusion issues** → check entries tagged "output quality"; adjust prompt,
  `neg_prompt`, `guidance`, or `cond_scale`
- **Data quality issues** → check entries about source image characteristics; consider
  pre-processing steps before the pipeline runs

When adding a new observation, use this structure:

```
### OBS-NNN — Short title

**Observed:** What you saw in the gallery output.
**Likely cause:** Hypothesis about where in the pipeline it broke.
**What to try:** Concrete next steps (profile changes, pre-processing, engine swap).
```
