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

import argparse, io, json, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from PIL import Image

# db.py lives in the same src/ directory
sys.path.insert(0, str(Path(__file__).parent))
from db import (open_db, get_runs, get_runs_for_source, get_thumbnail, get_stats,
                save_annotation, get_all_annotations, get_or_create_user, get_users)

# ── Load observations catalogue ────────────────────────────────────────────────
_OBS_PATH = Path(__file__).parent.parent / "docs" / "observations.json"
_OBS_DATA  = json.loads(_OBS_PATH.read_text(encoding="utf-8"))["observations"] if _OBS_PATH.exists() else []
_OBS_BY_ID = {o["id"]: o for o in _OBS_DATA}
_OBS_ID_TO_OPT = {"none": "none", **{o["id"]: f"{o['id']} · {o['title']}" for o in _OBS_DATA}}
_OBS_OPT_TO_ID = {v: k for k, v in _OBS_ID_TO_OPT.items()}
OBS_OPTIONS    = ["none"] + [f"{o['id']} · {o['title']}" for o in _OBS_DATA]

# ── Argument parsing ───────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--output-root", default="D:/Temp/yogamann-output")
_cli, _ = _parser.parse_known_args()

# ── Constants ──────────────────────────────────────────────────────────────────
RATINGS      = ["Bad", "OK", "Good", "Gold"]
RATING_EMOJI = {"Bad": "🔴", "OK": "🟡", "Good": "🟢", "Gold": "🏆"}
DEFAULT_PAGE_SIZE = 10

def _status(rating, misaligned, unwanted, notes, observations):
    filled = sum([bool(rating), bool(misaligned), bool(unwanted), bool(notes.strip()), bool(observations)])
    if filled == 0:   color = "#9e9e9e"
    elif filled <= 2: color = "#ffc107"
    elif filled  < 5: color = "#2196f3"
    else:             color = "#4caf50"
    return filled, color, f"{filled} / 5"

MISALIGNED_OPTIONS = [
    "none",
    "face", "neck", "upper chest", "abdomen", "shoulders",
    "upper arms", "lower arms", "hands", "hips",
    "upper legs", "lower legs", "feet",
]

UNWANTED_OPTIONS = [
    "none",
    "facial expression", "deformities", "props",
    "random artifacts", "textured/colored surfaces",
]

# ── Session state defaults ─────────────────────────────────────────────────────
def _init(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

_init("output_root",    _cli.output_root)
_init("page",           0)
_init("page_size",      DEFAULT_PAGE_SIZE)
_init("last_refresh",   time.time())
_init("last_count",     0)
_init("runs_snapshot",  None)
_init("snapshot_dirty", False)
_init("filter_state",   None)
_init("current_user",   None)   # dict: {id, name} once authenticated
_init("pending_new_user", None) # dict: {name, password} awaiting creation confirmation

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="yogamann gallery", layout="wide", page_icon="🧘")

# ── Version info (cached — one git call per server lifetime) ──────────────────
@st.cache_resource
def _version_info() -> str:
    try:
        repo = Path(__file__).parent.parent
        sha  = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL
        ).decode().strip()
        ts   = subprocess.check_output(
            ["git", "log", "-1", "--format=%ct"], cwd=repo, stderr=subprocess.DEVNULL
        ).decode().strip()
        commit_dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        now       = datetime.now(tz=timezone.utc)
        delta     = now - commit_dt
        if delta.days == 0:
            age = "today"
        elif delta.days == 1:
            age = "yesterday"
        else:
            age = f"{delta.days}d ago"
        return f"`{sha}` · {age}"
    except Exception:
        return "unknown"

# ── DB connection (cached per session) ────────────────────────────────────────
@st.cache_resource
def _get_conn(root: str):
    db_path = Path(root) / "yogamann.db"
    if not db_path.exists():
        return None
    return open_db(db_path)

conn = _get_conn(st.session_state.output_root)

# If the cached connection pre-dates the multi-user schema, bust the cache and reopen.
if conn is not None:
    try:
        conn.execute("SELECT 1 FROM users LIMIT 1")
    except Exception:
        st.cache_resource.clear()
        conn = _get_conn(st.session_state.output_root)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🧘 yogamann")
    st.caption(_version_info())

    # ── User selector ──────────────────────────────────────────────────────────
    cu = st.session_state.current_user
    if cu:
        st.caption(f"Reviewing as **{cu['name']}**")
        if st.button("Switch user", use_container_width=True):
            st.session_state.current_user = None
            st.rerun()
    else:
        pending = st.session_state.pending_new_user
        if pending:
            st.warning(f"**{pending['name']}** does not exist. Create it?")
            c1, c2 = st.columns(2)
            if c1.button("Create", use_container_width=True, type="primary"):
                conn.execute("INSERT INTO users (name, password) VALUES (?, ?)",
                             (pending["name"], pending["password"]))
                conn.commit()
                row = conn.execute("SELECT * FROM users WHERE name=?", (pending["name"],)).fetchone()
                st.session_state.current_user     = {"id": row["id"], "name": row["name"]}
                st.session_state.pending_new_user = None
                st.session_state.runs_snapshot    = None
                st.rerun()
            if c2.button("Cancel", use_container_width=True):
                st.session_state.pending_new_user = None
                st.rerun()
        else:
            with st.form("login_form", border=False):
                uname = st.text_input("Username")
                upass = st.text_input("Password", type="password")
                if st.form_submit_button("Sign in", use_container_width=True):
                    if not (conn and uname.strip()):
                        st.warning("Enter a username.")
                    else:
                        existing = conn.execute(
                            "SELECT * FROM users WHERE name=?", (uname.strip(),)
                        ).fetchone()
                        if existing is None:
                            st.session_state.pending_new_user = {"name": uname.strip(), "password": upass}
                            st.rerun()
                        elif existing["password"] == upass:
                            st.session_state.current_user = {"id": existing["id"], "name": existing["name"]}
                            st.session_state.runs_snapshot = None
                            st.rerun()
                        else:
                            st.error("Wrong password.")

    st.divider()

    new_root = st.text_input("Output root", value=st.session_state.output_root)
    if new_root != st.session_state.output_root:
        st.session_state.output_root = new_root
        st.session_state.page = 0
        st.cache_resource.clear()
        st.rerun()

    st.divider()

    auto_refresh = st.toggle("Auto-refresh", value=False)
    interval     = st.slider("Interval (s)", 15, 120, 30, step=15, disabled=not auto_refresh)

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
    state_filter = st.multiselect(
        "Filter by state",
        options=["Unvisited", "Updated", "Rating unspecified", "Unwanted unspecified", "Notes blank"],
        default=[],
        placeholder="All",
        help="Multiple selections combine with AND",
    )
    sort_order = st.radio("Sort", ["Newest first", "Oldest first"])

    if st.session_state.get("snapshot_dirty"):
        if st.button("🔄 Re-apply filters", use_container_width=True,
                     help="Items have changed since filters were last run — click to refresh"):
            st.session_state.filter_state   = None  # force re-query on next render
            st.session_state.snapshot_dirty = False

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

if not st.session_state.current_user:
    st.info("Sign in via the sidebar to start reviewing.")
    st.stop()

# ── Load runs (snapshot pattern) ──────────────────────────────────────────────
filters_active     = bool(rating_filter or state_filter)
current_filter_key = (tuple(rating_filter), tuple(state_filter), sort_order)
filter_changed     = st.session_state.filter_state != current_filter_key
need_refresh       = (
    st.session_state.runs_snapshot is None
    or filter_changed
    or (st.session_state.snapshot_dirty and not filters_active)
)

_uid = (st.session_state.current_user or {}).get("id", 1)

if need_refresh:
    st.session_state.runs_snapshot  = get_runs(
        conn,
        user_id=_uid,
        rating_filter=rating_filter if rating_filter else None,
        state_filter=state_filter if state_filter else None,
        newest_first=(sort_order == "Newest first"),
    )
    st.session_state.filter_state   = current_filter_key
    st.session_state.snapshot_dirty = False

all_runs  = st.session_state.runs_snapshot

# Detect new outputs
new_count = len(all_runs) - st.session_state.last_count
st.session_state.last_count = len(all_runs)

# ── Status bar ────────────────────────────────────────────────────────────────
if new_count > 0:
    st.success(f"✨ {new_count} new output(s) detected")
st.caption(f"{len(all_runs)} runs shown  ·  `{st.session_state.output_root}`")


# ── Pagination ────────────────────────────────────────────────────────────────
page_size   = st.session_state.page_size
total_pages = max(1, (len(all_runs) + page_size - 1) // page_size)
st.session_state.page = min(st.session_state.page, total_pages - 1)

pcol1, pcol2, pcol3, pcol4 = st.columns([1, 3, 2, 1])
with pcol1:
    if st.button("← Prev", disabled=st.session_state.page == 0):
        st.session_state.page -= 1
        st.rerun()
with pcol2:
    st.caption(f"Page {st.session_state.page + 1} of {total_pages}  ·  {len(all_runs)} runs")
with pcol3:
    new_page_size = st.selectbox(
        "Per page",
        options=[5, 10, 20, 50],
        index=[5, 10, 20, 50].index(page_size) if page_size in [5, 10, 20, 50] else 1,
        label_visibility="collapsed",
        key="page_size_select",
    )
    if new_page_size != st.session_state.page_size:
        st.session_state.page_size = new_page_size
        st.session_state.page = 0
        st.rerun()
with pcol4:
    if st.button("Next →", disabled=st.session_state.page >= total_pages - 1):
        st.session_state.page += 1
        st.rerun()

page_runs = all_runs[
    st.session_state.page * page_size :
    st.session_state.page * page_size + page_size
]

# ── Helper: thumbnail from DB ──────────────────────────────────────────────────
def _thumb_image(sha256: str):
    data = get_thumbnail(conn, sha256)
    if data:
        return Image.open(io.BytesIO(data))
    return None

# ── Auto-save factory ──────────────────────────────────────────────────────────
def _autosave(run_id: int, output_path: Path, key: str, user_id: int):
    def _cb():
        rating     = st.session_state.get(f"r_{key}") or ""
        misaligned = st.session_state.get(f"m_{key}") or []
        unwanted   = st.session_state.get(f"u_{key}") or []
        notes      = st.session_state.get(f"n_{key}") or ""
        obs_disp   = st.session_state.get(f"o_{key}") or []
        obs_ids    = [_OBS_OPT_TO_ID.get(d, d) for d in obs_disp]
        save_annotation(conn, run_id, user_id, rating, notes, misaligned, unwanted, obs_ids)
        mj = output_path.with_suffix(".metrics.json")
        if mj.exists():
            try:
                data = json.loads(mj.read_text(encoding="utf-8"))
                data["rating"]            = rating or None
                data["notes"]             = notes.strip() or None
                data["misaligned"]        = misaligned or None
                data["unwanted_features"] = unwanted or None
                data["observations"]      = obs_ids or None
                mj.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception:
                pass
        st.session_state.snapshot_dirty = True
        st.toast("Updated", icon="✅")
    return _cb

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
        cur_rating     = row["rating"] or ""
        cur_notes      = row["notes"]  or ""
        _row           = dict(row)
        cur_misaligned = [v for v in (_row.get("misaligned") or "").split(",") if v]
        cur_unwanted   = [v for v in (_row.get("unwanted_features") or "").split(",") if v]
        cur_obs_ids    = [v for v in (_row.get("observations") or "").split(",") if v]
        cur_obs_disp   = [_OBS_ID_TO_OPT.get(i, i) for i in cur_obs_ids]
        badge       = RATING_EMOJI.get(cur_rating, "⬜")
        folder      = input_path.parent.name
        _filled, _color, _label = _status(cur_rating, cur_misaligned, cur_unwanted, cur_notes, cur_obs_ids)

        st.markdown(f"""
<style>
div[data-testid="stVerticalBlockBorderWrapper"]:has(.card-{run_id}) {{
    border-color: {_color} !important;
    border-width: 1px !important;
    border-radius: 6px !important;
}}
</style>""", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(f'<div class="card-{run_id}"></div>', unsafe_allow_html=True)
            # ── Metadata bar ───────────────────────────────────────────────────
            meta_col, status_col = st.columns([9, 1])
            with meta_col:
                st.markdown(
                    f"{badge} **{input_path.name}** &nbsp;·&nbsp; "
                    f"📁 `{folder}` &nbsp;·&nbsp; "
                    f"🕐 `{row['timestamp']}` &nbsp;·&nbsp; "
                    f"⏱ {timing_s:.0f}s &nbsp;·&nbsp; "
                    f"`{row['width']}×{row['height']}`"
                )
                st.caption(
                    f"profile `{row['profile'] or '—'}` &nbsp;·&nbsp; "
                    f"seed `{row['seed']}` &nbsp;·&nbsp; "
                    f"cond `{row['cond_scale']}` &nbsp;·&nbsp; "
                    f"steps `{row['steps']}`"
                )
            with status_col:
                st.markdown(
                    f'<div style="text-align:right;color:{_color};font-size:0.75rem;font-weight:600;padding-top:4px">'
                    f'{_label}</div>',
                    unsafe_allow_html=True,
                )

            # ── Images ─────────────────────────────────────────────────────────
            img_col, out_col = st.columns(2)
            with img_col:
                st.caption("**Input**")
                thumb = _thumb_image(sha256)
                if input_path.exists():
                    st.image(str(input_path), use_container_width=True)
                elif thumb:
                    st.image(thumb, caption="(thumbnail — original moved)", use_container_width=True)
                else:
                    st.warning(f"Not found: `{input_path.name}`")
            with out_col:
                st.caption("**Output**")
                if output_path.exists():
                    st.image(str(output_path), use_container_width=True)
                else:
                    st.info("⏳ Generating…")

            # ── Controls ───────────────────────────────────────────────────────
            key      = f"run_{run_id}"
            autosave = _autosave(run_id, output_path, key, _uid)

            rating_opts = [""] + RATINGS
            st.radio(
                "Rating",
                options=rating_opts,
                index=rating_opts.index(cur_rating) if cur_rating in rating_opts else 0,
                format_func=lambda r: f"{RATING_EMOJI.get(r,'⬜')} {r}" if r else "—",
                horizontal=True,
                key=f"r_{key}",
                on_change=autosave,
            )

            fc1, fc2 = st.columns(2)
            with fc1:
                st.multiselect(
                    "Misaligned",
                    options=MISALIGNED_OPTIONS,
                    default=cur_misaligned,
                    placeholder="Select body parts...",
                    key=f"m_{key}",
                    on_change=autosave,
                )
            with fc2:
                st.multiselect(
                    "Unwanted features",
                    options=UNWANTED_OPTIONS,
                    default=cur_unwanted,
                    placeholder="Select issues...",
                    key=f"u_{key}",
                    on_change=autosave,
                )

            nc1, nc2 = st.columns(2)
            with nc1:
                st.text_area(
                    "Notes",
                    value=cur_notes,
                    height=100,
                    key=f"n_{key}",
                    placeholder="Add notes...",
                    on_change=autosave,
                )
            with nc2:
                new_obs = st.multiselect(
                    "Patterns",
                    options=OBS_OPTIONS,
                    default=cur_obs_disp,
                    placeholder="Tag a known pattern...",
                    key=f"o_{key}",
                    on_change=autosave,
                )
                if new_obs and new_obs != ["none"]:
                    selected_ids = [_OBS_OPT_TO_ID.get(d, d) for d in new_obs if d != "none"]
                    if selected_ids:
                        with st.expander("📋 Pattern notes", expanded=False):
                            for oid in selected_ids:
                                obs = _OBS_BY_ID.get(oid)
                                if obs:
                                    st.markdown(f"**{obs['id']} — {obs['title']}**")
                                    st.caption(f"*{obs['description']}*")
                                    st.caption(f"**Likely cause:** {obs['likely_cause']}")

            # ── Other reviewers ────────────────────────────────────────────────
            others = [a for a in get_all_annotations(conn, run_id) if a["user_id"] != _uid]
            if others:
                with st.expander(f"👥 {len(others)} other reviewer(s)", expanded=False):
                    for ann in others:
                        r_badge = RATING_EMOJI.get(ann["rating"] or "", "⬜")
                        st.markdown(f"**{ann['reviewer']}** — {r_badge} {ann['rating'] or '—'}")
                        if ann["misaligned"]:
                            st.caption(f"Misaligned: {ann['misaligned']}")
                        if ann["unwanted_features"]:
                            st.caption(f"Unwanted: {ann['unwanted_features']}")
                        if ann["observations"]:
                            obs_titles = ", ".join(
                                _OBS_ID_TO_OPT.get(i, i) for i in ann["observations"].split(",") if i
                            )
                            st.caption(f"Patterns: {obs_titles}")
                        if ann["notes"]:
                            st.caption(f"Notes: *{ann['notes']}*")
                        st.divider()

            # ── Run history ────────────────────────────────────────────────────
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

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    elapsed   = time.time() - st.session_state.last_refresh
    remaining = interval - elapsed
    if remaining <= 0:
        st.cache_resource.clear()
        st.session_state.last_refresh = time.time()
        st.rerun()
    else:
        time.sleep(remaining)
        st.rerun()
