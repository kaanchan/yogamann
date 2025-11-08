# yogamann
Image to mannequin generator

Converts photos of people into wooden artist mannequin representations using AI pose detection and Stable Diffusion.

## Quick Start

```bash
# Set up environment
conda env create -f environment.yml
conda activate yogamann

# Run on a single image
python src/make_mannequin.py input/photo.jpg

# Batch process a folder
python src/make_mannequin.py --folder input/
```

## Main Script

### `make_mannequin.py` - Driver Script

The main entry point that orchestrates the entire workflow.

**Features:**
- Batch processing: multiple photos and/or `--folder` mode
- YAML profile support for reusable configs
- Auto-increment filenames or `--overwrite`
- Optional contact sheets
- Dry-run mode for testing
- Fail-fast validation

**Usage:**
```bash
python src/make_mannequin.py photo.jpg [options]
python src/make_mannequin.py photo1.jpg photo2.jpg --steps 30
python src/make_mannequin.py --folder input/ --profile myprofile
```

**Options:**
- `--profile NAME` - Load config from `profiles/NAME.yml`
- `--steps N` - Number of inference steps (default: 30)
- `--seed N` - Random seed for reproducibility
- `--guidance F` - Guidance scale 4.0-15.0 (default: 7.5)
- `--cond-scale F` - ControlNet conditioning scale 0.5-2.0 (default: 1.0)
- `--sheet-scale F` - Contact sheet scale 0.5-2.0 (default: 1.0)
- `--prompt TEXT` - Custom prompt (default: "wooden artist mannequin, neutral backdrop")
- `--neg-prompt TEXT` - Negative prompt
- `--no-contact-sheets` - Skip comparison image generation
- `--overwrite` - Overwrite existing files instead of auto-increment
- `--dry-run` - Fast pipeline test without rendering

## Utility Scripts

### `extract_pose.py` - Pose Detection

Uses MediaPipe to detect 33 body landmarks and export:
1. Transparent PNG skeleton (ControlNet-ready)
2. JSON with pixel coordinates and metadata

**Called by:** `make_mannequin.py`

### `sd_make.py` - Stable Diffusion Renderer

Uses Stable Diffusion v1.5 + ControlNet OpenPose to render mannequin images.

**Models:**
- Base: `runwayml/stable-diffusion-v1-5`
- ControlNet: `lllyasviel/sd-controlnet-openpose`

**Called by:** `make_mannequin.py` via subprocess

### `create_contact_sheet.py` - Comparison Generator

Creates side-by-side comparison images:
1. **2-image layout:** Original | Mannequin (vertical stack)
2. **4-image layout:** Original | ControlNet mask | Mannequin | Depth map (2×2 grid)

**Called by:** `make_mannequin.py` (unless `--no-contact-sheets`)

## Workflow

```
Input Photo
    ↓
extract_pose.py → Skeleton PNG + JSON
    ↓
sd_make.py → Mannequin PNG
    ↓
create_contact_sheet.py → Comparison sheets
```

## Output Files

For input `photo.jpg`, generates:
- `output/photo-controlnet.png` - Pose skeleton mask
- `output/photo-controlnet.json` - Keypoint coordinates + metadata
- `output/photo-mannequin.png` - Generated mannequin
- `output/photo-comparison_2.png` - 2-image comparison
- `output/photo-comparison_4.png` - 4-image comparison (with depth map)

## Technology Stack

- **PyTorch 2.3.1** - Deep learning framework
- **Stable Diffusion v1.5** - Image generation
- **ControlNet OpenPose** - Pose-guided generation
- **MediaPipe** - Pose detection (33 landmarks)
- **OpenCV** - Image processing
- **Diffusers** - Hugging Face diffusion models
- **Pillow** - Image I/O and metadata
