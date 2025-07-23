"""
make_mannequin.py – planner / batch driver

New:
• versions / random / range handling
• single or multiple PHOTO args and/or --folder
• work‑list export  (--dump-worklist FILE.json)
• executes immediately unless --dump-only is given
"""

import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Overwriting tiny_vit_.* in registry",
    category=UserWarning,
    module=r"controlnet_aux\.segment_anything\.modeling\.tiny_vit_sam",
)

import argparse, json, os, random, sys, yaml, pathlib, math, itertools, re, subprocess
from typing import Dict, List
from PIL import Image
from extract_pose import make_controlnet_mask, save_keypoints
import create_contact_sheet as cs

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    
ROOT   = pathlib.Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"; 
OUTPUT.mkdir(parents=True, exist_ok=True)

# ── defaults ───────────────────────────────────────────────────────────
DEFAULTS = {
    "versions"    : 1,
    "random"      : False,
    "steps"       : 30,
    "seed"        : None,
    "sheet_scale" : 1.0,
    "cond_scale"  : 1.0,
    "guidance"    : 7.5,
    "prompt"      : "wooden artist mannequin, neutral backdrop",
    "neg_prompt"  : None,
}

# ── util helpers ────────────────────────────────────────────────────────
def linspace(a: float, b: float, n: int) -> List[float]:
    if n == 1: return [a]
    step = (b - a) / (n - 1)
    return [a + i*step for i in range(n)]


def expand_value(val, versions: int, rand: bool):
    "Return a list of length *versions* for given profile value."
    if isinstance(val, list) and len(val) == 2:   # range
        a, b = val
        if rand:
            return [random.uniform(a, b) for _ in range(versions)]
        return linspace(a, b, versions)
    # scalar – repeat
    return [val] * versions


def pad_version(idx: int, total_versions: int) -> str:
    width = len(str(total_versions))
    return f"v{idx:0{width}d}"


def next_free_stem(base: str, ext: str) -> str:
    """Return a stem that doesn't yet exist (pose.png, pose-1.png, ...)"""
    p = OUTPUT / f"{base}{ext}"
    if not p.exists():
        return base
    i = 1
    while (OUTPUT / f"{base}-{i}{ext}").exists():
        i += 1
    return f"{base}-{i}"


def build_tasks(photo: pathlib.Path, cfg: Dict) -> List[Dict]:
    """Returns a list of task dicts for this photo according to cfg."""
    versions  = cfg["versions"]
    rand_mode = cfg["random"]
    pad_fn = lambda i: pad_version(i, versions)

    # expand every numeric / seed field into per‑version list
    per_field = {k: expand_value(v, versions, rand_mode)
                 for k, v in cfg.items()
                 if k in {"steps", "cond_scale", "guidance", "seed"}}

    # detect zero variation
    if versions > 1 and all(len(set(lst)) == 1 for lst in per_field.values()):
        print(f"⚠ [{photo.name}] versions>1 but no varying params; generating v1 only")
        versions = 1

    stem_base = photo.stem
    ext       = ".png"
    stem_with_auto = next_free_stem(stem_base, f"--{pad_fn(1)}{ext}")

    tasks = []
    for i in range(1, versions+1):
        vtag = f"--{pad_fn(i)}"
        stem = stem_with_auto.replace(f"--{pad_fn(1)}", vtag)
        out_png   = OUTPUT / f"{stem}.png"
        mask_png  = OUTPUT / f"{stem_base}-controlnet.png"   # shared mask
        task_cfg  = cfg.copy()
        for k,lst in per_field.items():
            task_cfg[k] = lst[i-1]
        task_cfg["version"] = i
        task_cfg["version_tag"] = vtag

        tasks.append({
            "photo"     : str(photo),
            "mask_png"  : str(mask_png),
            "output_png": str(out_png),
            "cfg"       : task_cfg
        })
    return tasks


# ── CLI ────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("photo", nargs="*", help="one or more image files")
ap.add_argument("--folder", help="process every image in this folder")
ap.add_argument("--profile")
ap.add_argument("--dump-worklist", help="write tasks to JSON and exit")
ap.add_argument("--dump-only", action="store_true",
                help="create work‑list JSON but do NOT execute")
ap.add_argument("--overwrite", action="store_true")
args, unknown_cli = ap.parse_known_args()

# merge CLI overrides (key=value) into a dict
cli_overrides = {}
pat = re.compile(r"--(?P<key>[^=]+)=(?P<val>.+)")
for token in unknown_cli:
    m = pat.match(token)
    if m:
        cli_overrides[m.group("key").replace("-", "_")] = yaml.safe_load(m.group("val"))
    else:
        print(f"Warning: ignoring unknown arg '{token}'")

# load profile + CLI
cfg = DEFAULTS.copy()
if args.profile:
    prof_path = ROOT / "profiles" / f"{pathlib.Path(args.profile).stem}.yml"
    cfg.update(yaml.safe_load(prof_path.read_text()))
cfg.update(cli_overrides)

# gather photos
photos: List[pathlib.Path] = []
if args.folder:
    for p in pathlib.Path(args.folder).iterdir():
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            photos.append(p.resolve())
photos.extend(pathlib.Path(p).resolve() for p in args.photo)
if not photos:
    ap.error("Provide at least one PHOTO or --folder")

# build full work‑list
worklist: List[Dict] = []
for photo in photos:
    worklist.extend(build_tasks(photo, cfg))

if args.dump_worklist:
    pathlib.Path(args.dump_worklist).write_text(json.dumps(worklist, indent=2))
    print("✓ work‑list written:", args.dump_worklist)
    if args.dump_only:
        sys.exit(0)

# execute tasks sequentially via sd_make.py
for t in worklist:
    print(f"\n=== {pathlib.Path(t['photo']).name}{t['cfg']['version_tag']} ===")

    proc = subprocess.run(
        [sys.executable, str(ROOT / "src/sd_make.py"), "--from-worklist", "-"],
        input=json.dumps([t]),      # pass str, not bytes
        text=True,                  # so stdin expects str
        capture_output=True         # capture BOTH streams
    )
    # stream through so you see progress or errors
    sys.stdout.write(proc.stdout); sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        sys.exit(proc.returncode)
        
