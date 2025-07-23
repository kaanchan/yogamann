"""
make_mannequin.py – driver script

Features
────────
✓ fail‑fast validation
✓ optional dry‑run
✓ auto‑increment filenames (or --overwrite)
✓ contact sheets toggle
✓ **batch mode**: multiple PHOTO args and/or --folder
"""

import argparse, json, os, subprocess, sys, yaml, pathlib
from PIL import Image
from extract_pose import make_controlnet_mask, save_keypoints
import create_contact_sheet as cs

ROOT   = pathlib.Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"; OUTPUT.mkdir(exist_ok=True)

# ── default config values ───────────────────────────────────────────────
DEFAULTS = {
    "steps": 30,
    "seed": None,
    "sheet_scale": 1.0,
    "cond_scale": 1.0,
    "guidance": 7.5,
    "prompt": "wooden artist mannequin, neutral backdrop",
    "neg_prompt": None,
}
RECREATE_MASK = bool(int(os.getenv("RECREATE_MASK", "0")))


# ── helpers ─────────────────────────────────────────────────────────────
def validate_cfg(c):
    assert c["steps"] > 0, "steps must be positive"
    assert 0.5 <= c["sheet_scale"] <= 2.0, "sheet_scale 0.5–2.0"
    assert 0.5 <= c["cond_scale"] <= 2.0, "cond_scale 0.5–2.0"
    assert 4.0 <= c["guidance"] <= 15.0, "guidance 4–15"
    assert c["prompt"].strip(), "prompt cannot be empty"


def load_cfg(profile, cli_dict):
    cfg = DEFAULTS.copy()
    if profile:
        yml = ROOT / "profiles" / f"{pathlib.Path(profile).stem}.yml"
        cfg.update(yaml.safe_load(yml.read_text()))
    cfg.update({k: v for k, v in cli_dict.items() if v is not None})
    validate_cfg(cfg)
    return cfg


def next_free(path: pathlib.Path, overwrite: bool) -> pathlib.Path:
    if overwrite or not path.exists():
        return path
    i = 1
    while True:
        p = path.with_name(f"{path.stem}-{i}{path.suffix}")
        if not p.exists():
            return p
        i += 1


# ── main per‑photo routine ─────────────────────────────────────────────
def process_photo(photo: pathlib.Path, cfg: dict,
                  make_sheets: bool, overwrite: bool,
                  profile_name: str | None, dry_run: bool):

    stem = photo.stem
    mask_png = OUTPUT / f"{stem}-controlnet.png"
    json_fp  = OUTPUT / f"{stem}-controlnet.json"

    man_png = next_free(OUTPUT / f"{stem}-mannequin.png", overwrite)
    stem_out = man_png.stem.replace("-mannequin", "")
    sheet2 = OUTPUT / f"{stem_out}-comparison_2.png"
    sheet4 = OUTPUT / f"{stem_out}-comparison_4.png"

    provenance = {"profile_name": profile_name or "none",
                  "profile_version": cfg.get("profile_version", "n/a"),
                  **cfg}

    # mask generation / reuse
    if RECREATE_MASK or overwrite or not mask_png.exists():
        pts = make_controlnet_mask(photo, mask_png)
        save_keypoints(pts, json_fp, photo)
        print("✓ pose mask generated")
    else:
        print("↳ pose mask reused")

    # mannequin render or dry‑run
    cmd = [sys.executable, str(ROOT / "src/sd_make.py"),
           str(photo), str(mask_png), str(man_png),
           "--steps", str(cfg['steps']),
           "--cond-scale", str(cfg['cond_scale']),
           "--guidance", str(cfg['guidance']),
           "--prompt", cfg['prompt'],
           "--meta", json.dumps(provenance)]
    if cfg["neg_prompt"]:
        cmd += ["--neg-prompt", cfg["neg_prompt"]]
    if cfg["seed"] is not None:
        cmd += ["--seed", str(cfg["seed"])]
    if dry_run:
        cmd += ["--dry-run"]

    subprocess.run(cmd, check=True)
    print("✓ mannequin pipeline OK" + (" (dry-run)" if dry_run else ""))

    if dry_run:
        return  # skip heavy post‑steps in smoke test mode

    # write provenance into JSON
    with open(json_fp, "r+") as f:
        data = json.load(f); data["config"] = provenance
        f.seek(0); json.dump(data, f, indent=2); f.truncate()

    # contact sheets
    if make_sheets:
        o, m = Image.open(photo), Image.open(man_png)
        c = Image.open(mask_png).convert("RGB")
        scale = cfg["sheet_scale"]
        cs.build_comparison_2(o, m, sheet2, cfg["seed"], scale)
        cs.build_comparison_4(o, c, m, sheet4, cfg["seed"], scale)
        print(f"✓ contact sheets saved ({sheet2.name}, {sheet4.name})")


# ── CLI parsing & batch dispatch ───────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("photo", nargs="*", help="one or more image files")
    ap.add_argument("--folder", help="process every image in this folder")
    ap.add_argument("--profile")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--sheet-scale", type=float)
    ap.add_argument("--cond-scale", type=float)
    ap.add_argument("--guidance", type=float)
    ap.add_argument("--prompt")
    ap.add_argument("--neg-prompt")
    ap.add_argument("--no-contact-sheets", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="fast pipeline smoke test, no render")
    args = ap.parse_args()

    # collect photos
    photos: list[pathlib.Path] = []

    if args.folder:
        for p in pathlib.Path(args.folder).iterdir():
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                photos.append(p.resolve())

    photos.extend(pathlib.Path(p).resolve() for p in args.photo)

    if not photos:
        ap.error("Provide at least one PHOTO path or --folder.")

    # build CLI dict for cfg merge (remove path keys)
    cli_dict = vars(args).copy()
    cli_dict.pop("photo", None)
    cli_dict.pop("folder", None)

    for p in photos:
        print(f"\n=== {p.name} ===")
        cfg = load_cfg(args.profile, cli_dict)
        process_photo(p, cfg,
                      make_sheets=not args.no_contact_sheets,
                      overwrite=args.overwrite,
                      profile_name=args.profile,
                      dry_run=args.dry_run)
