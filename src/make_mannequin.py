"""
make_mannequin.py – batch planner / driver
2025‑07‑20  ·  smarter versions  ·  robust profile loader
"""

from __future__ import annotations
import argparse, json, logging, os, random, re, subprocess, sys
import pathlib, warnings
from typing import Dict, List

import yaml

log = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message=r"Overwriting tiny_vit_.* in registry",
    category=UserWarning,
    module=r"controlnet_aux\.segment_anything\.modeling\.tiny_vit_sam",
)

SRC_DIR = pathlib.Path(__file__).resolve().parent      # …/yogamann/src
ROOT    = SRC_DIR.parent                               # …/yogamann
OUTPUT_DIR: pathlib.Path = None  # set after arg parsing
PROFILES_DIR = ROOT / "profiles"

# ── defaults ───────────────────────────────────────────────────────────
DEFAULT_PROFILE: Dict = {
    "versions"   : 1,
    "random"     : False,
    "steps"      : 25,
    "cond_scale" : 1.0,
    "guidance"   : 5.0,
    "sheet_scale": 1.0,
    "seed"       : None,
    "prompt"     : "wooden artist mannequin, neutral backdrop, sharp focus",
    "neg_prompt" : None,
}

# ── profile loader ─────────────────────────────────────────────────────
def _detect_format(path: pathlib.Path) -> str:
    ext = path.suffix.lower()
    if ext in {".yml", ".yaml"}:
        return "yaml"
    if ext == ".json":
        return "json"
    raise ValueError  # handled by caller

def _read_profile(p: str) -> Dict:
    """Find & read a profile, guessing extension + format when omitted."""
    cand = pathlib.Path(p)

    # 1️⃣ exact path?
    if cand.exists():
        fmt = _detect_format(cand)
        txt = cand.read_text()
        return yaml.safe_load(txt) if fmt == "yaml" else json.loads(txt)

    # 2️⃣ search nearby with added extension
    for ext in (".yml", ".yaml", ".json"):
        guess = cand.with_suffix(ext)
        if guess.exists():
            return _read_profile(str(guess))

    # 3️⃣ look in a local “profiles/” folder
    for ext in (".yml", ".yaml", ".json"):
        guess = ROOT / "profiles" / f"{cand.stem}{ext}"
        if guess.exists():
            return _read_profile(str(guess))

    raise FileNotFoundError(f"Profile file not found: {p}")

# ── helpers (unchanged) ────────────────────────────────────────────────
def linspace(a: float, b: float, n: int) -> List[float]:
    if n == 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]

def expand_value(val, versions: int, rand: bool):
    if isinstance(val, list) and len(val) == 2:
        lo, hi = val
        return [random.uniform(lo, hi) for _ in range(versions)] if rand else linspace(lo, hi, versions)
    return [val] * versions

def pad_version(i: int, total: int) -> str:
    return f"v{i:0{len(str(total))}d}"

def next_free_stem(base: str, ext: str) -> str:
    p = OUTPUT_DIR / f"{base}{ext}"
    if not p.exists():
        return base
    j = 1
    while (OUTPUT_DIR / f"{base}-{j}{ext}").exists():
        j += 1
    return f"{base}-{j}"

def build_tasks(photo: pathlib.Path, cfg: Dict) -> List[Dict]:
    V   = cfg["versions"]
    rnd = cfg["random"]
    fields = {k: expand_value(v, V, rnd)
              for k, v in cfg.items()
              if k in {"steps", "cond_scale", "guidance", "seed"}}

    # collapse only if seed fixed & all params identical
    if V > 1 and cfg.get("seed") is not None and not rnd \
       and all(len(set(v)) == 1 for v in fields.values()):
        print(f"[warn] [{photo.name}] versions>1 but no varying params; generating v1 only")
        V = 1

    stem0 = next_free_stem(photo.stem + f"--{pad_version(1,V)}", ".png")
    tasks = []
    for i in range(1, V + 1):
        vtag  = f"--{pad_version(i,V)}"
        stem  = stem0.replace(f"--{pad_version(1,V)}", vtag)
        out_p = OUTPUT_DIR / f"{stem}.png"
        mask  = OUTPUT_DIR / f"{photo.stem}-controlnet.png"

        cfg_i = cfg.copy()
        for k, seq in fields.items():
            cfg_i[k] = seq[i-1]
        cfg_i.update({"version": i, "version_tag": vtag})

        tasks.append({
            "photo"     : str(photo),
            "mask_png"  : str(mask),
            "output_png"    : str(out_p),
            "cfg"       : cfg_i
        })
    return tasks

# ── CLI ────────────────────────────────────────────────────────────────
cli = argparse.ArgumentParser()
cli.add_argument("photo", nargs="*", help="image file(s) to process")
cli.add_argument("--folder", help="process every image in this folder")
cli.add_argument("--output-dir", dest="output_dir",
                 help="directory for output images (default: <repo>/out)")
cli.add_argument("--profile", help="profile name / path")
cli.add_argument("--dump-worklist")
cli.add_argument("--dump-only", action="store_true")
cli.add_argument("--log-level", default=os.environ.get("YOGAMANN_LOG_LEVEL", "INFO"),
                 choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                 help="logging verbosity (default: INFO)")
args, kv_overrides = cli.parse_known_args()

OUTPUT_DIR = pathlib.Path(args.output_dir) if args.output_dir else ROOT / "out"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
# propagate level to subprocesses via env
os.environ["YOGAMANN_LOG_LEVEL"] = args.log_level

# parse key=value overrides --------------------------------------------------
overrides = {}
pat = re.compile(r"--(?P<key>[^=]+)=(?P<val>.+)")
for token in kv_overrides:
    if m := pat.match(token):
        overrides[m.group("key").replace("-", "_")] = yaml.safe_load(m.group("val"))
    else:
        print(f"warning: ignored unknown arg '{token}'")

# load profile ---------------------------------------------------------------
cfg = DEFAULT_PROFILE.copy()
if args.profile:
    cfg.update(_read_profile(args.profile))
    cfg["profile"] = args.profile
cfg.update(overrides)

# collect photos -------------------------------------------------------------
photos: List[pathlib.Path] = []
if args.folder:
    photos += [p for p in pathlib.Path(args.folder).iterdir()
               if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
photos += [pathlib.Path(p) for p in args.photo]
if not photos:
    cli.error("provide at least one PHOTO or --folder")

# build & maybe dump work‑list ----------------------------------------------
worklist = [t for photo in photos for t in build_tasks(photo.resolve(), cfg)]
if args.dump_worklist:
    pathlib.Path(args.dump_worklist).write_text(json.dumps(worklist, indent=2))
    print("[OK] work‑list written:", args.dump_worklist)
    if args.dump_only:
        sys.exit(0)

# execute — single subprocess handles all tasks so CUDA warms up only once ---
log.info("Processing %d task(s) — profile: %s", len(worklist),
         args.profile or "default")
subprocess.run(
    [sys.executable, str(SRC_DIR / "sd_make.py"), "--from-worklist", "-"],
    input=json.dumps(worklist),
    text=True,
    check=True,
    env={**os.environ, "PYTHONUTF8": "1"},
)
