"""
gallery.py — live review gallery for yogamann outputs (DB-backed).

Usage:
    streamlit run src/gallery.py
    streamlit run src/gallery.py -- --output-root D:/Temp/yogamann-output

Reads from {output_root}/yogamann.db.
Run  `python src/db.py --ingest`  to import existing .metrics.json files first.

Closes #21
"""
from __future__ import annotations

import argparse, io, sys, time
from pathlib import Path

import streamlit as st
from PIL import Image

# db.py lives in the same src/ directory
sys.path.insert(0, str(Path(__file__).parent))
from db import open_db, get_runs, get_runs_for_source, get_thumbnail, get_stats, save_rating

# ── Argument parsing ───────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--output-root", default="D:/Temp/yogamann-output")
_cli, _ = _parser.parse_known_args()

# ── Constants ──────────────────────────────────────────────────────────────────
RATINGS      = ["Bad", "OK", "Good", "Gold"]
RATING_EMOJI = {"Bad": "🔴", "OK": "🟡", "Good": "🟢", "Gold": "🏆"}
PAGE_SIZE    = 10

# ── Session state defaults ─────────────────────────────────────────────────────
def _init(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

_init("output_root",  _cli.output_root)
_init("page",         0)
_init("last_refresh", time.time())
_init("last_count",   0)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="yogamann gallery", layout="wide", page_icon="🧘")

# ── DB connection (cached per session) ────────────────────────────────────────
@st.cache_resource
def _get_conn(root: str):
    db_path = Path(root) / "yogamann.db"
    if not db_path.exists():
        return None
    return open_db(db_path)

conn = _get_conn(st.session_state.output_root)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🧘 yogamann")

    new_root = st.text_input("Output root", value=st.session_state.output_root)
    if new_root != st.session_state.output_root:
        st.session_state.output_root = new_root
        st.session_state.page = 0
        st.cache_resource.clear()
        st.rerun()

    st.divider()

    auto_refresh = st.toggle("Auto-refresh", value=True)
    interval     = st.slider("Interval (s)", 15, 120, 30, step=15, disabled=not auto_refresh)

    if auto_refresh and conn is not None:
        elapsed   = time.time() - st.session_state.last_refresh
        remaining = max(0, interval - int(elapsed))
        st.caption(f"↻ refresh in {remaining}s")

    if st.button("🔄 Refresh now"):
        st.cache_resource.clear()
        conn = _get_conn(st.session_state.output_root)
        st.session_state.last_refresh = time.time()
        st.rerun()

    st.divider()

    rating_filter = st.multiselect(
        "Filter by rating",
        options=RATINGS + ["Unrated"],
        default=[],
        placeholder="All",
    )
    sort_order = st.radio("Sort", ["Newest first", "Oldest first"])

    st.divider()

    if conn is not None:
        s = get_stats(conn)
        st.metric("Total runs",    s["total"])
        st.metric("Rated",         f"{s['rated']} / {s['total']}")
        st.metric("🏆 Gold",       s["gold"])
        st.metric("Source images", s["sources"])

# ── No DB yet ──────────────────────────────────────────────────────────────────
st.title("yogamann gallery")

if conn is None:
    db_path = Path(st.session_state.output_root) / "yogamann.db"
    st.warning(
        f"No database found at `{db_path}`.\n\n"
        "Run the ingest command to import existing outputs:\n"
        "```\npython src/db.py --ingest\n```\n"
        "Or run a batch job — new outputs are added automatically."
    )
    st.stop()

# ── Load runs ──────────────────────────────────────────────────────────────────
all_runs = get_runs(
    conn,
    rating_filter=rating_filter if rating_filter else None,
    newest_first=(sort_order == "Newest first"),
)

# Detect new outputs
new_count = len(all_runs) - st.session_state.last_count
st.session_state.last_count = len(all_runs)

# ── Status bar ────────────────────────────────────────────────────────────────
col_a, col_b = st.columns([3, 1])
with col_a:
    if new_count > 0:
        st.success(f"✨ {new_count} new output(s) detected")
    root_label = st.session_state.output_root
    st.caption(f"{len(all_runs)} runs shown  ·  `{root_label}`")

# ── Pagination ────────────────────────────────────────────────────────────────
total_pages = max(1, (len(all_runs) + PAGE_SIZE - 1) // PAGE_SIZE)
st.session_state.page = min(st.session_state.page, total_pages - 1)

if total_pages > 1:
    pcol1, pcol2, pcol3 = st.columns([1, 4, 1])
    with pcol1:
        if st.button("← Prev", disabled=st.session_state.page == 0):
            st.session_state.page -= 1
            st.rerun()
    with pcol2:
        st.caption(f"Page {st.session_state.page + 1} of {total_pages}")
    with pcol3:
        if st.button("Next →", disabled=st.session_state.page >= total_pages - 1):
            st.session_state.page += 1
            st.rerun()

page_runs = all_runs[
    st.session_state.page * PAGE_SIZE :
    st.session_state.page * PAGE_SIZE + PAGE_SIZE
]

# ── Helper: thumbnail from DB ──────────────────────────────────────────────────
def _thumb_image(sha256: str):
    data = get_thumbnail(conn, sha256)
    if data:
        return Image.open(io.BytesIO(data))
    return None

# ── Cards ──────────────────────────────────────────────────────────────────────
if not page_runs:
    st.info("No runs match the current filter.")
else:
    for row in page_runs:
        run_id      = row["id"]
        sha256      = row["source_sha256"]
        input_path  = Path(row["source_path"])
        output_path = Path(row["output_png"])
        timing_s    = row["total_s"] or 0
        cur_rating  = row["rating"] or ""
        cur_notes   = row["notes"]  or ""
        badge       = RATING_EMOJI.get(cur_rating, "⬜")
        folder      = input_path.parent.name

        with st.container(border=True):
            img_col, out_col, meta_col = st.columns([4, 4, 3])

            # ── Input ──────────────────────────────────────────────────────────
            with img_col:
                st.caption("**Input**")
                thumb = _thumb_image(sha256)
                if input_path.exists():
                    st.image(str(input_path), use_container_width=True)
                elif thumb:
                    st.image(thumb, caption="(thumbnail — original moved)", use_container_width=True)
                else:
                    st.warning(f"Not found: `{input_path.name}`")

            # ── Output ─────────────────────────────────────────────────────────
            with out_col:
                st.caption("**Output**")
                if output_path.exists():
                    st.image(str(output_path), use_container_width=True)
                else:
                    st.info("⏳ Generating…")

            # ── Metadata + rating ──────────────────────────────────────────────
            with meta_col:
                st.markdown(f"**{badge} {input_path.name}**")
                st.caption(f"📁 `{folder}`")
                st.caption(
                    f"`{row['width']}×{row['height']}`  "
                    f"·  ⏱ {timing_s:.0f}s"
                )
                st.caption(
                    f"seed `{row['seed']}`  "
                    f"·  cond `{row['cond_scale']}`  "
                    f"·  steps `{row['steps']}`"
                )
                st.caption(f"profile `{row['profile'] or '—'}`")
                st.caption(f"🕐 {row['timestamp']}")

                st.divider()

                key = f"run_{run_id}"

                rating_opts = [""] + RATINGS
                new_rating  = st.selectbox(
                    "Rating",
                    rating_opts,
                    index=rating_opts.index(cur_rating) if cur_rating in rating_opts else 0,
                    format_func=lambda r: f"{RATING_EMOJI.get(r,'⬜')} {r}" if r else "— unrated —",
                    key=f"r_{key}",
                )
                new_notes = st.text_area("Notes", value=cur_notes, height=80, key=f"n_{key}")

                if st.button("💾 Save", key=f"s_{key}"):
                    save_rating(conn, run_id, new_rating, new_notes)
                    # Also update the .metrics.json sidecar for portability
                    mj = output_path.with_suffix(".metrics.json")
                    if mj.exists():
                        try:
                            import json
                            data = json.loads(mj.read_text(encoding="utf-8"))
                            data["rating"] = new_rating or None
                            data["notes"]  = new_notes.strip() or None
                            mj.write_text(json.dumps(data, indent=2), encoding="utf-8")
                        except Exception:
                            pass
                    st.success("Saved ✓")
                    time.sleep(0.4)
                    st.rerun()

            # ── Run history (other generations of same source) ─────────────────
            history = get_runs_for_source(conn, sha256)
            if len(history) > 1:
                with st.expander(f"📋 {len(history)} runs for this source"):
                    for h in history:
                        h_badge  = RATING_EMOJI.get(h["rating"] or "", "⬜")
                        h_active = "← this run" if h["id"] == run_id else ""
                        st.caption(
                            f"{h_badge} `{h['timestamp']}`  "
                            f"cond `{h['cond_scale']}`  "
                            f"seed `{h['seed']}`  "
                            f"steps `{h['steps']}`  "
                            f"profile `{h['profile'] or '—'}`  "
                            f"{h['rating'] or 'unrated'}  {h_active}"
                        )

# ── Auto-refresh loop ──────────────────────────────────────────────────────────
if auto_refresh:
    elapsed = time.time() - st.session_state.last_refresh
    if elapsed >= interval:
        st.cache_resource.clear()
        st.session_state.last_refresh = time.time()
        st.rerun()
    else:
        time.sleep(1)
        st.rerun()
