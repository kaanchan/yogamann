# yogamann

Converts yoga pose photos into wooden artist mannequin renders using ControlNet-OpenPose + Stable Diffusion XL.

Pipeline: **DWPose** detects the skeleton → **xinsir/controlnet-openpose-sdxl-1.0** conditions SDXL generation → contact-sheet comparisons saved alongside outputs → results ingested into a SQLite review database.

<img src="docs/reference-outputs/batch-feature/yoga-pose-sample-1.png" width="320" alt="Sample mannequin render — yoga-pose-sample-1, profile-batch feature era (SD 1.5 + MediaPipe, 2025-07-20)">

*Sample output from the v1 pipeline (SD 1.5 + MediaPipe, July 2025). See [`docs/reference-outputs/`](docs/reference-outputs/) for the full baseline set and [`docs/archive/v1-mediapipe-2025-07-20/`](docs/archive/v1-mediapipe-2025-07-20/) for the source that produced them.*

---

## Setup

Requires Python 3.11+ and an NVIDIA GPU (driver ≥ 572, RTX 40/50 series; tested on RTX 5080 16 GB).  
PyTorch wheels bundle CUDA 13.0 — your system CUDA version doesn't matter.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download model weights once (≈10 GB, cached in `~/.cache/huggingface/`):

```powershell
.\make.ps1 -Target download
```

Set `$env:HF_TOKEN = 'hf_...'` before downloading for faster authenticated transfers ([get a free token](https://huggingface.co/settings/tokens)).

---

## make.ps1 — common workflows

`make.ps1` is the primary entrypoint for everyday development tasks.

```powershell
.\make.ps1 [-Target <target>] [-LogLevel <level>] [-OutputRoot <path>]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-Target` | `test` | Which workflow to run (see below) |
| `-LogLevel` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `-OutputRoot` | `D:\Temp\yogamann-output` | Root directory for DB and output files |

### Targets

| Target | What it does |
|--------|-------------|
| `download` | Pre-fetch all Hugging Face model weights (run once) |
| `test` | Generate mannequin for one sample image → open HTML gallery |
| `test-all` | Generate mannequins for all `input/` images → open HTML gallery |
| `diag` | Hardware + import diagnostics (GPU, CUDA, model availability) |
| `gallery` | Regenerate `output/index.html` from existing outputs → open browser |
| `open` | Open existing `output/index.html` without regenerating |
| `ingest` | Import all `.metrics.json` files in `OutputRoot` into `yogamann.db` |
| `review` | Launch the Streamlit review gallery (reads `yogamann.db`) |

### Sample commands

```powershell
.\make.ps1                             # quick smoke test, INFO logging
.\make.ps1 -LogLevel DEBUG             # verbose — shows VRAM and timing per step
.\make.ps1 -Target test-all            # all reference images
.\make.ps1 -Target ingest              # import new outputs into DB
.\make.ps1 -Target review              # open browser review UI
.\make.ps1 -Target review -OutputRoot D:\Temp\yogamann-output
```

---

## scripts/batch.ps1 — full library runs

`scripts/batch.ps1` processes an external data library, mirroring its folder structure into the output directory. Use this for bulk runs over the full yoga deck.

```powershell
.\scripts\batch.ps1 [-Source <path>] [-SourceRoot <path>] [-OutputRoot <path>]
            [-PipelineProfile <name>] [-LogLevel <level>]
            [-Limit <n>] [-DryRun] [-Monitor] [-Overwrite]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-Source` | `D:\Temp\yogamann-data\Yoga Deck - Kaushalam Ways` | File or folder to process (recursive) |
| `-SourceRoot` | Same as `-Source` default | Root used to compute relative output paths; keep this at the library root so subfolder runs still produce the full mirrored tree |
| `-OutputRoot` | `D:\Temp\yogamann-output` | Where outputs land |
| `-PipelineProfile` | `yoga_asana` | Profile name or path (no extension needed) |
| `-LogLevel` | `INFO` | Logging verbosity |
| `-Limit` | `0` (unlimited) | Stop after processing this many images |
| `-DryRun` | off | Print what would run without generating anything |
| `-Monitor` | off | Spawn `monitor.ps1` GPU dashboard in a new terminal |
| `-Overwrite` | off | Reprocess images that already have output (default: skip) |

### Sample commands

```powershell
# Dry run — see what would be processed without using the GPU
.\scripts\batch.ps1 -DryRun

# Process up to 5 images from the default library, with GPU monitor
.\scripts\batch.ps1 -Limit 5 -Monitor

# Process a single subfolder
.\scripts\batch.ps1 -Source "D:\Temp\yogamann-data\Yoga Deck - Kaushalam Ways\Asanas\Vajrasana"

# Force-reprocess with a different profile
.\scripts\batch.ps1 -Limit 10 -PipelineProfile yoga_asana_cs14 -Overwrite

# Full run, verbose logging
.\scripts\batch.ps1 -LogLevel DEBUG
```

> **Tip:** After a batch run, ingest and review with:
> ```powershell
> .\make.ps1 -Target ingest
> .\make.ps1 -Target review
> ```

---

## scripts/multi-batch.ps1 — unattended overnight batch queue

`scripts/multi-batch.ps1` queues multiple `scripts/batch.ps1` runs and ingests results automatically when done. Edit the `$batches` array in the script to define your folder queue, then start it before you sleep.

```powershell
.\scripts\multi-batch.ps1 [-OutputRoot <path>] [-PipelineProfile <name>] [-LogLevel <level>]
               [-Limit <n>] [-DryRun] [-Monitor] [-Overwrite]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-OutputRoot` | `D:\Temp\yogamann-output` | Where outputs and DB land |
| `-PipelineProfile` | `yoga_asana` | Profile applied to all batches |
| `-LogLevel` | `INFO` | Logging verbosity |
| `-Limit` | `0` (unlimited) | Per-batch image cap |
| `-DryRun` | off | Print plan without generating |
| `-Monitor` | off | Launch GPU dashboard once for the whole run |
| `-Overwrite` | off | Reprocess images that already have output |

Edit the queue in `scripts/multi-batch.ps1` directly:

```powershell
$batches = @(
    @{
        Source     = "D:\Temp\yogamann-data\Yoga Deck - Kaushalam Ways\Asanas"
        SourceRoot = "D:\Temp\yogamann-data\Yoga Deck - Kaushalam Ways"
    }
    @{
        Source     = "D:\Temp\yogamann-data\Yoga Deck - Kaushalam Ways\Pranayama"
        SourceRoot = "D:\Temp\yogamann-data\Yoga Deck - Kaushalam Ways"
    }
)
```

Sample commands:

```powershell
.\scripts\multi-batch.ps1 -DryRun                # preview what would run — no GPU used
.\scripts\multi-batch.ps1 -Monitor               # full overnight run with GPU dashboard
.\scripts\multi-batch.ps1 -Limit 10 -Monitor     # cap at 10 images per folder set (testing)
```

After all batches complete, `scripts/multi-batch.ps1` automatically runs ingest. Review results with:

```powershell
.\make.ps1 -Target review
```

---

## Profiles

Profiles control generation parameters. They live in `profiles/` as YAML files.  
Pass by name (no `.yml` extension) or full path.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `versions` | int | 1 | Variants to generate per source image |
| `random` | bool | `false` | `true` = sample ranges randomly; `false` = linear sweep |
| `steps` | int or `[lo, hi]` | 25 | Diffusion steps |
| `cond_scale` | float or `[lo, hi]` | 1.0 | ControlNet conditioning strength (higher = pose followed more strictly) |
| `guidance` | float or `[lo, hi]` | 5.0 | CFG guidance scale |
| `seed` | int or `[lo, hi]` or null | null | Fixed seed for reproducibility; `null` = random each run |
| `sheet_scale` | float | 1.0 | Scale factor for comparison contact sheets |
| `prompt` | string | `wooden artist mannequin…` | Positive prompt |
| `neg_prompt` | string | null | Negative prompt |

When a value is `[lo, hi]`, the profile expands to `versions` linearly-interpolated steps between `lo` and `hi`.

### Example — parameter sweep

```yaml
# profiles/yoga_asana_cs_sweep.yml
versions: 5
random: false
steps: 30
cond_scale: [0.8, 2.0]   # produces 0.8, 1.1, 1.4, 1.7, 2.0
guidance: 6.0
seed: 42                  # fixed — only cond_scale varies
prompt: wooden artist mannequin, yoga pose, studio grey backdrop
neg_prompt: blurry, low quality, text, watermark
```

Run it:

```powershell
.\scripts\batch.ps1 -Source input\yoga-pose-sample-4.jpg -PipelineProfile yoga_asana_cs_sweep
.\make.ps1 -Target ingest
.\make.ps1 -Target review
```

---

## Review gallery (Streamlit)

The Streamlit gallery shows input/output pairs side-by-side with rating buttons and full run history per image.

```powershell
.\make.ps1 -Target review
# or directly:
.\.venv\Scripts\streamlit.exe run src/gallery.py -- --output-root D:\Temp\yogamann-output
```

The gallery reads from `{OutputRoot}\yogamann.db`. Run `.\make.ps1 -Target ingest` first if you have existing outputs not yet in the database.

---

## Output folder layout

```
{OutputRoot}/
  {source-subfolder}/
    {stem}--v1.png                    mannequin render
    {stem}--v1-comparison_2.png       vertical stack  (original / mannequin)
    {stem}--v1-comparison_4.png       2×2 grid        (original / controlnet / mannequin / depth)
    {stem}-controlnet.png             OpenPose skeleton used as control image
    {stem}-controlnet.json            DWPose keypoints JSON
    {stem}--v1.metrics.json           per-run metrics (seed, steps, timing, cond_scale, …)
  yogamann.db                         SQLite review database (all runs + ratings)
```

---

## Advanced — Python CLIs

For scripting or CI use, the Python scripts can be called directly.

### make_mannequin.py — plan and execute a batch

```bash
# Single image
python src/make_mannequin.py input/yoga-pose-sample-4.jpg --profile yoga_asana

# Whole folder
python src/make_mannequin.py --folder input --profile yoga_asana

# Inspect planned tasks without running (dry run)
python src/make_mannequin.py --folder input --profile yoga_asana \
  --dump-worklist worklist.json --dump-only

# Override individual profile keys on the command line
python src/make_mannequin.py input/yoga-pose-sample-4.jpg --profile yoga_asana \
  cond_scale=1.4 seed=42
```

| Flag | Description |
|------|-------------|
| `photo` | One or more image paths (positional) |
| `--folder` | Process all images in this directory |
| `--output-dir` | Output directory (default: `output/`) |
| `--profile` | Profile name or path |
| `--dump-worklist <file>` | Write planned task list to JSON |
| `--dump-only` | Exit after writing worklist, no generation |
| `--log-level` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `key=value` | Override any profile key (trailing positional args) |

### sd_make.py — low-level generation worker

```bash
# Run a pre-built worklist
python src/sd_make.py --from-worklist worklist.json

# Single image (uncommon — prefer make_mannequin.py)
python src/sd_make.py photo.jpg mask.png output.png \
  --steps 30 --cond-scale 1.4 --guidance 6.0 --seed 42 \
  --prompt "wooden artist mannequin, studio grey backdrop"
```

`sd_make.py` loads the model once and processes all worklist tasks in sequence — this is why `make_mannequin.py` passes the entire batch as a single subprocess call.

### db.py — database management

```bash
# Import all .metrics.json files in the output root
python src/db.py --ingest --output-root D:/Temp/yogamann-output

# Show DB statistics
python src/db.py --stats --output-root D:/Temp/yogamann-output
```

Database lives at `{output_root}/yogamann.db`. Schema: `source_images` (SHA-256 identity), `runs` (all settings + timing + rating), `thumbnails` (150×150 JPEG blobs).
