# Static Gallery Builder

**6 nodes · Community 18 · Cohesion 0.53**

`src/make_gallery.py`

---

## What it does

Scans the `output/` directory for comparison PNGs and writes a static `output/index.html` gallery. Unlike the Streamlit gallery (`src/gallery.py`), this produces a self-contained HTML file with no server required — open it directly in a browser.

## Key functions

| Function | Role |
|----------|------|
| `build_gallery(output_dir)` | Entry point — scans for PNGs, calls `_card()` for each, writes `index.html` |
| `_card(img_path, meta)` | Generates the HTML for one gallery card: image + metadata badges |
| `_cfg_badges(meta)` | Renders small HTML badges for the render profile config (guidance scale, steps, seed, etc.) |
| `_read_meta(img_path)` | Reads the `.metrics.json` sidecar for an image |

## When to use this vs `gallery.py`

| | `make_gallery.py` | `gallery.py` (Streamlit) |
|--|---|---|
| Output | Static HTML file | Live server |
| VLM annotations | No (render metadata only) | Yes |
| Human annotation | No | Yes |
| Use case | Quick browse of renders | Full annotation review |

## Connects to

- [Pose Pipeline Research](pose-pipeline-research.md) — documented in README
- [Contact Sheet Builder](contact-sheet-builder.md) — the PNGs it displays are built by `create_contact_sheet.py`
