# docs/ideal-targets/ — Visual Target References

**Last updated: 2026-05-12**

---

This directory contains reference images that represent the **visual goal** of the
yogamann pipeline — what a well-rendered wooden artist mannequin should look like.
These are not pipeline outputs; they are curated examples used to calibrate and
evaluate generation quality.

## Contents

| File | Format | Description |
|---|---|---|
| `standard-wooden-mannequin-01.avif` | AVIF | Articulated wooden artist mannequin, standard pose, clear wood grain texture |
| `standard-wooden-mannequin-02.jpg` | JPEG | Wooden mannequin, neutral studio lighting, ball-jointed detail visible |
| `wooden-mannequin-cell-drawing-01.jpg` | JPEG | Schematic/cell-drawing style rendering of wooden mannequin structure |

## Purpose

These images serve as:

1. **Prompt calibration** — used when writing or refining `profiles/*.yml` prompts to
   anchor the target aesthetic (wood grain, joint articulation, neutral backdrop)
2. **VLM evaluation grounding** — will be used as reference inputs to the
   VLM comparison harness (issue #29) so models have a concrete visual target
   to compare against pipeline outputs
3. **Human review reference** — quick visual check when manually rating renders
   in the Streamlit gallery (`.\make.ps1 -Target review`)

## How to update

Add new reference images directly to this directory and update this README.
These files **are committed to git** as they represent deliberate editorial choices,
not generated outputs. Keep file sizes reasonable (compress if over ~1 MB).
