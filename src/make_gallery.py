"""
make_gallery.py — scan out/ for comparison PNGs and write out/index.html.

Usage
-----
    python src/make_gallery.py                   # uses out/ → out/index.html
    python src/make_gallery.py --out-dir out/run1

Opens directly in any browser via file:// — no server required.
Reads cfg/* metadata embedded in each PNG by sd_make.py.

Closes #3 — https://github.com/kaanchan/yogamann/issues/3
"""

from __future__ import annotations
import argparse
import pathlib
import html
from PIL import Image

# cfg keys shown per card (in display order)
_CFG_KEYS = ["steps", "guidance", "cond_scale", "seed", "version_tag"]

_CSS = """\
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #111; color: #ddd; font-family: system-ui, sans-serif; padding: 1.5rem; }
h1   { font-size: 1.2rem; font-weight: 500; margin-bottom: 1rem; color: #aaa; }

.grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
    gap: 1.2rem;
}

.card {
    background: #1e1e1e;
    border-radius: 8px;
    overflow: hidden;
    cursor: pointer;
    transition: transform .15s, box-shadow .15s;
}
.card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px #0009; }

.card img { width: 100%; display: block; }

.meta {
    padding: .6rem .8rem;
    font-size: .72rem;
    line-height: 1.6;
    color: #888;
}
.meta strong { color: #bbb; display: block; margin-bottom: .2rem; font-size: .8rem; }
.meta span   { margin-right: .8rem; white-space: nowrap; }

/* ── lightbox ── */
#lb {
    display: none;
    position: fixed; inset: 0;
    background: #000c;
    z-index: 999;
    align-items: center;
    justify-content: center;
    cursor: zoom-out;
}
#lb.open { display: flex; }
#lb img  { max-width: 94vw; max-height: 94vh; border-radius: 6px; }
"""

_JS = """\
const lb = document.getElementById('lb');
const lbImg = document.getElementById('lb-img');

document.querySelectorAll('.card').forEach(card => {
    card.addEventListener('click', () => {
        lbImg.src = card.querySelector('img').src;
        lb.classList.add('open');
    });
});
lb.addEventListener('click', () => lb.classList.remove('open'));
document.addEventListener('keydown', e => { if (e.key === 'Escape') lb.classList.remove('open'); });
"""


def _read_meta(path: pathlib.Path) -> dict[str, str]:
    try:
        img = Image.open(path)
        img.load()
        return {k: v for k, v in img.info.items() if isinstance(v, str)}
    except Exception:
        return {}


def _cfg_badges(meta: dict[str, str]) -> str:
    parts = []
    for k in _CFG_KEYS:
        val = meta.get(f"cfg/{k}") or meta.get(k)
        if val:
            parts.append(f'<span><b>{html.escape(k)}</b>={html.escape(val)}</span>')
    return "".join(parts)


def _card(img_path: pathlib.Path, out_dir: pathlib.Path) -> str:
    rel   = img_path.relative_to(out_dir).as_posix()
    meta  = _read_meta(img_path)
    stem  = img_path.stem                          # e.g. yoga-pose-sample-1--v1-comparison_2
    title = html.escape(stem)
    source = html.escape(meta.get("source", "").split("\\")[-1].split("/")[-1])
    generated = html.escape(meta.get("generated", ""))
    badges = _cfg_badges(meta)

    return f"""\
<div class="card">
  <img src="{rel}" loading="lazy" alt="{title}">
  <div class="meta">
    <strong>{source or title}</strong>
    {badges}
    {"<br><span>" + generated + "</span>" if generated else ""}
  </div>
</div>"""


def build_gallery(out_dir: pathlib.Path) -> pathlib.Path:
    patterns = ["*-comparison_2.png", "*-comparison_4.png"]
    images: list[pathlib.Path] = []
    for pat in patterns:
        images.extend(sorted(out_dir.glob(pat)))

    if not images:
        print(f"⚠ No comparison PNGs found in {out_dir}")
        return out_dir / "index.html"

    cards = "\n".join(_card(p, out_dir) for p in images)

    page = f"""\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>yogamann – batch gallery ({len(images)} images)</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>yogamann gallery &mdash; {len(images)} comparison sheets</h1>
  <div class="grid">
{cards}
  </div>

  <!-- lightbox -->
  <div id="lb"><img id="lb-img" src="" alt="full size"></div>

  <script>{_JS}</script>
</body>
</html>"""

    dest = out_dir / "index.html"
    dest.write_text(page, encoding="utf-8")
    print(f"✓ gallery written: {dest}  ({len(images)} images)")
    return dest


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Build browser gallery from comparison PNGs")
    cli.add_argument("--out-dir", default="out", help="folder containing *-comparison*.png files")
    args = cli.parse_args()

    out_dir = pathlib.Path(args.out_dir).resolve()
    if not out_dir.is_dir():
        cli.error(f"Directory not found: {out_dir}")

    build_gallery(out_dir)
