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

import argparse, io, json, logging, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from PIL import Image

# db.py lives in the same src/ directory
sys.path.insert(0, str(Path(__file__).parent))
from db import (open_db, get_runs, get_runs_for_source, get_thumbnail, get_or_create_thumbnail, get_stats,
                save_annotation, get_all_annotations, get_or_create_user, get_users,
                get_vlm_annotations, get_vlm_comparison_page)
import batch_lock
import win_job

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

# ── VLM Compare helpers ────────────────────────────────────────────────────────
_VLM_BADGE = {
    "good":       ("🟢", "#2d6a4f", "#e8f5e9"),
    "acceptable": ("🟡", "#b45309", "#fff8e1"),
    "poor":       ("🔴", "#c0392b", "#fdecea"),
}
_HUMAN_BADGE = {
    "Bad":  ("🔴", "#c0392b", "#fdecea"),
    "OK":   ("🟡", "#b45309", "#fff8e1"),
    "Good": ("🟢", "#2d6a4f", "#e8f5e9"),
    "Gold": ("🏆", "#c9a84c", "#1a1a2e"),
}
_VLM_TO_HUMAN = {
    "good":       {"Good", "Gold"},
    "acceptable": {"OK"},
    "poor":       {"Bad"},
}
_MODEL_LABEL = {
    "qwen2_5_vl_7b":   "Qwen2.5-VL",
    "internvl2_5_8b":  "InternVL2.5",
    "minicpm_v_2_6":   "MiniCPM-V",
    "molmo_7b_d":      "Molmo-7B",
    "minicpm_o_2_6":   "MiniCPM-o",
}


def _vlm_badge_html(rating: str | None) -> str:
    if not rating:
        return '<span style="color:#666;font-style:italic">—</span>'
    icon, bg, fg = _VLM_BADGE.get(rating.lower(), ("❓", "#444", "#eee"))
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 9px;'
        f'border-radius:10px;font-size:0.78rem;font-weight:700;white-space:nowrap">'
        f'{icon} {rating}</span>'
    )


def _human_badge_html(rating: str | None) -> str:
    if not rating:
        return '<span style="color:#777;font-style:italic">—</span>'
    icon, bg, fg = _HUMAN_BADGE.get(rating, ("⬜", "#333", "#eee"))
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 9px;'
        f'border-radius:10px;font-size:0.78rem;font-weight:700;white-space:nowrap">'
        f'{icon} {rating}</span>'
    )


def _vlm_cell_style(vlm_r: str | None, ref_r: str | None,
                    agree_count: int, is_outlier: bool) -> str:
    parts = ["padding:7px 10px;vertical-align:top;border-bottom:1px solid #262640"]
    if vlm_r and ref_r and ref_r in _VLM_TO_HUMAN.get(vlm_r.lower(), set()):
        parts.append("border-left:3px solid #c9a84c;background:#1e1a0e")
    elif agree_count >= 3:
        parts.append("background:#0e1e16")
    elif is_outlier:
        parts.append("background:#1e0e0e")
    return ";".join(parts)


def _render_vlm_table(rows, active_models: list, thumb_map: dict, render_map: dict | None = None) -> str:
    import json as _json
    render_map = render_map or {}
    heads = "".join(
        f'<th style="padding:8px 10px;text-align:left;font-size:0.75rem;'
        f'color:#aaa;border-bottom:2px solid #333;white-space:nowrap">'
        f'{_MODEL_LABEL.get(m, m)}</th>'
        for m in active_models
    )
    html = (
        f'<table style="width:100%;border-collapse:collapse;'
        f'background:#12122a;border-radius:8px;overflow:hidden;font-family:monospace">'
        f'<thead><tr style="background:#0d0d1f">'
        f'<th style="padding:8px 10px;text-align:left;font-size:0.75rem;color:#aaa;'
        f'border-bottom:2px solid #333;width:100px">Source / Render</th>'
        f'<th style="padding:8px 10px;text-align:left;font-size:0.75rem;color:#aaa;'
        f'border-bottom:2px solid #333">Run</th>'
        f'<th style="padding:8px 10px;text-align:left;font-size:0.75rem;color:#c9a84c;'
        f'border-bottom:2px solid #333;border-right:2px solid #2a2a4a">⭐ Human</th>'
        f'{heads}'
        f'</tr></thead><tbody>'
    )

    _img_style = "max-width:90px;max-height:90px;object-fit:contain;border-radius:4px;display:block"
    _placeholder = '<div style="width:90px;height:60px;background:#2a2a4a;border-radius:4px"></div>'

    for i, row in enumerate(rows):
        run_id   = row["run_id"]
        ref_r    = row["ref_rating"]
        row_bg   = "#16163a" if i % 2 == 0 else "#12122a"

        vlm_ratings = {
            m: (row[f"{m}_rating"] or "").lower() or None
            for m in active_models
        }
        non_null = [r for r in vlm_ratings.values() if r]

        def flags(m):
            r = vlm_ratings[m]
            if not r:
                return 0, False
            ag = sum(1 for v in vlm_ratings.values() if v == r)
            return ag, (ag == 1 and len(non_null) >= 3)

        src_b64  = thumb_map.get(run_id)
        rnd_b64  = render_map.get(run_id)
        src_html = (f'<img src="data:image/jpeg;base64,{src_b64}" style="{_img_style}">'
                    if src_b64 else _placeholder)
        rnd_html = (f'<img src="data:image/jpeg;base64,{rnd_b64}" style="{_img_style};margin-top:4px">'
                    if rnd_b64 else "")
        img_html = f'{src_html}{rnd_html}'

        ts = (row["timestamp"] or "")[:10]

        vlm_cells = ""
        for m in active_models:
            r = vlm_ratings[m]
            notes_raw   = (row.get(f"{m}_notes") or "").strip()
            mis_raw     = row.get(f"{m}_misaligned") or "[]"
            unwanted_raw = row.get(f"{m}_unwanted") or "[]"
            ag, outlier = flags(m)
            style = _vlm_cell_style(r, ref_r, ag, outlier)

            def _parse_list(raw) -> list:
                try:
                    parsed = _json.loads(raw) if isinstance(raw, str) else raw
                    return parsed if isinstance(parsed, list) else []
                except Exception:
                    return []

            mis_list  = _parse_list(mis_raw)
            uw_list   = _parse_list(unwanted_raw)

            detail_parts = []
            if notes_raw:
                detail_parts.append(
                    f'<span style="color:#b0b0c8;font-size:0.7rem;line-height:1.5;'
                    f'display:block;white-space:pre-wrap">{notes_raw}</span>'
                )
            if mis_list:
                bullets = "".join(f'<li>{p}</li>' for p in mis_list)
                detail_parts.append(
                    f'<div style="margin-top:4px;color:#e07070;font-size:0.68rem">'
                    f'<strong>Misaligned:</strong><ul style="margin:2px 0 0 12px;padding:0">{bullets}</ul></div>'
                )
            if uw_list:
                bullets = "".join(f'<li>{p}</li>' for p in uw_list)
                detail_parts.append(
                    f'<div style="margin-top:4px;color:#e0b050;font-size:0.68rem">'
                    f'<strong>Artifacts:</strong><ul style="margin:2px 0 0 12px;padding:0">{bullets}</ul></div>'
                )

            if detail_parts:
                preview = notes_raw[:50] if notes_raw else (mis_list[0] if mis_list else "details")
                ellipsis = "…" if len(notes_raw) > 50 or len(detail_parts) > 1 else ""
                notes_html = (
                    f'<details style="margin-top:4px">'
                    f'<summary style="color:#7a7a9a;font-size:0.68rem;cursor:pointer;'
                    f'list-style:none;white-space:nowrap;overflow:hidden;'
                    f'text-overflow:ellipsis;max-width:200px">'
                    f'&#9656; {preview}{ellipsis}'
                    f'</summary>'
                    + "".join(detail_parts) +
                    f'</details>'
                )
            else:
                notes_html = '<span style="color:#444;font-size:0.68rem">no notes</span>'

            empty_cell = '<span style="color:#333;font-size:0.8rem">—</span>'
            vlm_cells += (
                f'<td style="{style}">{_vlm_badge_html(r) if r else empty_cell}'
                f'{notes_html}</td>'
            )

        html += (
            f'<tr style="background:{row_bg}">'
            f'<td style="padding:6px 10px;vertical-align:top;border-bottom:1px solid #262640">{img_html}</td>'
            f'<td style="padding:7px 10px;vertical-align:top;color:#7a7a9a;'
            f'font-size:0.75rem;border-bottom:1px solid #262640">#{run_id}'
            f'<br><span style="color:#555">{ts}</span></td>'
            f'<td style="padding:7px 10px;vertical-align:top;border-bottom:1px solid #262640;'
            f'border-right:2px solid #2a2a4a">{_human_badge_html(ref_r)}</td>'
            f'{vlm_cells}'
            f'</tr>'
        )
    return html + "</tbody></table>"


def _render_agreement_matrix(all_rows, active_models: list, ref_label: str) -> str:
    labels = active_models + [ref_label]

    def to_bucket(r):
        if not r:
            return None
        mapping = {"Bad": "poor", "OK": "acceptable", "Good": "good", "Gold": "good"}
        return mapping.get(r) or r.lower()

    def get_r(row, col):
        if col == ref_label:
            return to_bucket(row.get("ref_rating"))
        return to_bucket(row.get(f"{col}_rating"))

    matrix = {}
    for a in labels:
        for b in labels:
            if a == b:
                matrix[(a, b)] = 100.0
                continue
            pairs = [
                (get_r(r, a), get_r(r, b)) for r in all_rows
                if get_r(r, a) and get_r(r, b)
            ]
            matrix[(a, b)] = (
                round(100.0 * sum(x == y for x, y in pairs) / len(pairs), 1)
                if pairs else None
            )

    short = {m: _MODEL_LABEL.get(m, m) for m in active_models}
    short[ref_label] = f"⭐ {ref_label}"

    def cell_bg(pct):
        if pct is None:
            return "#1a1a2e"
        g = int(80 * pct / 100)
        r = int(80 * (1 - pct / 100))
        return f"rgb({20+r},{20+g},40)"

    head = "".join(
        f'<th style="padding:5px 8px;font-size:0.7rem;color:#9a9ab0;'
        f'text-align:center;white-space:nowrap">{short[l]}</th>'
        for l in labels
    )
    body = ""
    for a in labels:
        body += (
            f'<tr><td style="padding:5px 8px;font-size:0.72rem;color:#9a9ab0;'
            f'font-weight:600;white-space:nowrap">{short[a]}</td>'
        )
        for b in labels:
            pct = matrix.get((a, b))
            txt = f"{pct}%" if pct is not None else "—"
            body += (
                f'<td style="padding:5px 8px;background:{cell_bg(pct)};'
                f'text-align:center;font-size:0.75rem;color:#ddd;'
                f'border:1px solid #1a1a2e;border-radius:2px">{txt}</td>'
            )
        body += "</tr>"

    return (
        f'<table style="border-collapse:separate;border-spacing:3px;'
        f'background:#0d0d1f;padding:8px;border-radius:6px">'
        f'<thead><tr><th></th>{head}</tr></thead>'
        f'<tbody>{body}</tbody></table>'
    )


# ── VLM Generate — subprocess management ──────────────────────────────────────
import queue as _queue
import subprocess as _subprocess
import threading as _threading


def _launch_vlm_inference(model_key: str, run_id: int, output_root: str) -> None:
    """Spawn a compare_vlm.py subprocess for one (model, run) and store in session_state.

    Checks the GPU-busy lock before spawning. If locked, sets vlm_blocked_by_lock
    in session_state and returns without launching.
    """
    existing = batch_lock.check(output_root)
    if existing:
        st.session_state.vlm_blocked_by_lock = existing
        return

    cmd = [
        "python", "src/compare_vlm.py",
        "--models", model_key,
        "--run-ids", str(run_id),
        "--output-root", output_root,
        "--strategy", "model-major",
        "--batch-size", "1",
    ]
    proc = _subprocess.Popen(
        cmd,
        stdout=_subprocess.PIPE,
        stderr=_subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(Path(__file__).parent.parent),
    )
    # Record subprocess PID in the lock (not Streamlit's PID) so batch_lock.check()
    # auto-clears when the subprocess exits — no explicit release needed.
    batch_lock.acquire(output_root, job_type="gallery", model_ids=[model_key], pid=proc.pid)
    win_job.assign_process(_win_job_handle, proc.pid)

    q: _queue.Queue = _queue.Queue()

    def _drain(p, q):
        for line in iter(p.stdout.readline, ""):
            q.put(line.rstrip())
        p.stdout.close()

    _threading.Thread(target=_drain, args=(proc, q), daemon=True).start()
    st.session_state.vlm_active_job = {
        "model_key": model_key,
        "run_id":    run_id,
        "process":   proc,
        "queue":     q,
        "lines":     [],
        "status":    "running",
    }
    st.session_state.vlm_blocked_by_lock = None
    st.session_state.vlm_show_terminal = True


@st.dialog("VLM Inference Output", width="large")
def _inference_terminal_dialog() -> None:
    """Modal showing live stdout from the active inference subprocess."""
    job = st.session_state.get("vlm_active_job")
    if not job:
        st.info("No active job.")
        if st.button("Close"):
            st.session_state.vlm_show_terminal = False
            st.rerun()
        return

    # Drain any new lines from the reader thread
    q = job["queue"]
    while not q.empty():
        try:
            job["lines"].append(q.get_nowait())
        except _queue.Empty:
            break

    is_running = job["process"].poll() is None
    model_label = _MODEL_LABEL.get(job["model_key"], job["model_key"])

    if is_running:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px">'
            f'<span style="animation:spin 1s linear infinite;display:inline-block">⟳</span>'
            f'<b>{model_label}</b> — run #{job["run_id"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        rc = job["process"].returncode
        icon = "✅" if rc == 0 else "❌"
        st.markdown(f'{icon} **{model_label}** — run #{job["run_id"]} — '
                    f'{"complete" if rc == 0 else f"exited (code {rc})"}')
        job["status"] = "done" if rc == 0 else "error"

    # Terminal output
    output_text = "\n".join(job["lines"][-80:]) or "(waiting for output…)"
    st.code(output_text, language=None)

    col1, col2 = st.columns([1, 1])
    with col1:
        if is_running:
            if st.button("⛔ Abort", type="secondary", use_container_width=True):
                job["process"].terminate()
                job["status"] = "aborted"
                st.rerun()
        else:
            if st.button("Close & clear", use_container_width=True):
                st.session_state.vlm_active_job = None
                st.session_state.vlm_show_terminal = False
                st.rerun()
    with col2:
        if st.button("Close (keep job state)", use_container_width=True):
            st.session_state.vlm_show_terminal = False
            st.rerun()

    # Auto-refresh every 0.75 s while process is still running
    if is_running:
        time.sleep(0.75)
        st.rerun()


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
_init("vlm_page",          0)
_init("vlm_page_size",     25)
_init("vlm_disagree_only", False)
_init("vlm_active_job",       None)
_init("vlm_show_terminal",    False)
_init("vlm_blocked_by_lock",  None)
_init("vlm_stale_lock_toast", False)

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

# ── Windows Job Object (one handle per server lifetime — auto-kills children) ──
@st.cache_resource
def _get_win_job():
    return win_job.create_job("yogamann-inference")

_win_job_handle = _get_win_job()

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

    # ── VLM Compare controls ───────────────────────────────────────────────────
    st.divider()
    st.caption("**VLM Compare**")
    _KNOWN_MODELS = [
        "qwen2_5_vl_7b", "internvl2_5_8b", "minicpm_v_2_6",
        "molmo_7b_d", "minicpm_o_2_6",
    ]
    if conn is not None:
        _models_with_data = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT model_id FROM vlm_annotations ORDER BY model_id"
            ).fetchall()
        ]
    else:
        _models_with_data = []
    _model_opts = _models_with_data or _KNOWN_MODELS

    selected_models = st.multiselect(
        "VLM models to compare",
        options=_model_opts,
        default=_model_opts,
        format_func=lambda m: _MODEL_LABEL.get(m, m),
        key="vlm_model_select",
    )

    _all_users = get_users(conn) if conn else []
    _user_names = [u["name"] for u in _all_users]
    _default_ref = next(
        (u["name"] for u in _all_users if u["name"].lower() in ("satchit", "satchid")),
        _user_names[0] if _user_names else None,
    )
    ref_user_name = st.selectbox(
        "Reference reviewer",
        options=_user_names or ["—"],
        index=_user_names.index(_default_ref) if _default_ref in _user_names else 0,
        key="vlm_ref_user_select",
    )
    ref_user_id = next(
        (u["id"] for u in _all_users if u["name"] == ref_user_name), None
    )

    vlm_pg_size = st.selectbox(
        "Rows per page (VLM)",
        options=[10, 25, 50],
        index=[10, 25, 50].index(st.session_state.vlm_page_size)
        if st.session_state.vlm_page_size in [10, 25, 50] else 1,
        key="vlm_page_size_select",
    )
    if vlm_pg_size != st.session_state.vlm_page_size:
        st.session_state.vlm_page_size = vlm_pg_size
        st.session_state.vlm_page = 0

    disagree_only = st.checkbox(
        "Disagreements only",
        value=st.session_state.vlm_disagree_only,
        key="vlm_disagree_check",
    )
    if disagree_only != st.session_state.vlm_disagree_only:
        st.session_state.vlm_disagree_only = disagree_only
        st.session_state.vlm_page = 0

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

tab1, tab2 = st.tabs(["🖼 Gallery", "🤖 VLM Compare"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Gallery
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VLM Compare
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    import base64 as _b64

    # ── Dialog trigger — must be at top of tab so it fires on every rerun ──────
    if st.session_state.vlm_show_terminal:
        _inference_terminal_dialog()

    # ── Stale-lock auto-clear toast ────────────────────────────────────────────
    # If a previous gallery job left a lock file behind after the process died,
    # batch_lock.check() already cleaned it up. Notify the user once.
    if st.session_state.vlm_stale_lock_toast:
        st.toast("Stale GPU lock cleared — previous inference process is no longer running.", icon="🧹")
        st.session_state.vlm_stale_lock_toast = False

    # ── Blocked-by-lock error banner ───────────────────────────────────────────
    _lock_info = st.session_state.vlm_blocked_by_lock
    if _lock_info:
        _lock_pid  = _lock_info.get("pid", "?")
        _lock_type = _lock_info.get("job_type", "?")
        _lock_mods = ", ".join(_lock_info.get("model_ids", []))
        _lock_t    = (_lock_info.get("start_time") or "")[:19].replace("T", " ")
        st.error(
            f"**GPU is busy** — cannot start a new inference job.  \n"
            f"Running: `{_lock_type}` · models: `{_lock_mods}` · PID {_lock_pid} · started {_lock_t} UTC  \n"
            f"Wait for it to finish, or kill PID {_lock_pid} and refresh this page.",
            icon="🔒",
        )

    st.subheader("VLM Model Comparison")

    _active_models = selected_models if selected_models else _model_opts
    _pg            = st.session_state.vlm_page
    _pg_size       = st.session_state.vlm_page_size
    _disagree      = st.session_state.vlm_disagree_only
    _ref_label     = ref_user_name or "Reference"

    if not conn:
        st.info("No database found. Run the batch orchestrator first.")
    else:
        _raw_rows, vlm_total = get_vlm_comparison_page(
            conn,
            limit=_pg_size,
            offset=_pg * _pg_size,
            models=_active_models,
            disagree_only=_disagree,
            ref_user_id=ref_user_id,
        )
        vlm_rows = [dict(r) for r in _raw_rows]
        vlm_total_pages = max(1, (vlm_total + _pg_size - 1) // _pg_size)
        if _pg >= vlm_total_pages:
            st.session_state.vlm_page = 0
            _pg = 0

        # ── Thumbnail maps (source from DB, render from disk) ─────────────────
        thumb_map: dict[int, str] = {}
        render_map: dict[int, str] = {}
        for vrow in vlm_rows:
            rid = vrow["run_id"]
            if rid not in thumb_map:
                data = get_or_create_thumbnail(conn, vrow["source_sha256"])
                if data:
                    thumb_map[rid] = _b64.b64encode(data).decode()
            if rid not in render_map and vrow.get("output_png"):
                try:
                    rimg = Image.open(vrow["output_png"]).convert("RGB")
                    rimg.thumbnail((150, 150))
                    buf = io.BytesIO()
                    rimg.save(buf, format="JPEG", quality=70)
                    render_map[rid] = _b64.b64encode(buf.getvalue()).decode()
                except Exception as e:
                    logging.warning("render thumbnail failed run %d: %s", rid, e)

        # ── Pagination bar ─────────────────────────────────────────────────────
        vc1, vc2, vc3, vc4 = st.columns([1, 4, 3, 1])
        with vc1:
            if st.button("← Prev", key="vlm_prev",
                         disabled=st.session_state.vlm_page == 0):
                st.session_state.vlm_page -= 1
                st.rerun()
        with vc2:
            st.caption(
                f"Page {st.session_state.vlm_page + 1} of {vlm_total_pages}"
                f"  ·  **{vlm_total}** annotated runs"
                + ("  ·  disagreements only" if _disagree else "")
            )
        with vc3:
            st.caption(f"Reference: **{_ref_label}** &nbsp;·&nbsp; "
                       f"Models: {len(_active_models)}")
        with vc4:
            if st.button("Next →", key="vlm_next",
                         disabled=st.session_state.vlm_page >= vlm_total_pages - 1):
                st.session_state.vlm_page += 1
                st.rerun()

        if not vlm_rows:
            st.info(
                "No VLM annotations found yet. Run the batch orchestrator:\n"
                "```\npython src/compare_vlm.py --limit 0 --output-root D:/Temp/yogamann-output\n```"
            )
        else:
            # ── Legend ────────────────────────────────────────────────────────
            st.markdown(
                '<div style="display:flex;gap:12px;align-items:center;'
                'padding:6px 0;font-size:0.75rem;color:#aaa">'
                '<span>Cell highlight:</span>'
                '<span style="padding:2px 8px;border-left:3px solid #c9a84c;'
                'background:#1e1a0e;border-radius:3px">⭐ matches you</span>'
                '<span style="padding:2px 8px;background:#0e1e16;border-radius:3px">'
                '🟢 3+ models agree</span>'
                '<span style="padding:2px 8px;background:#1e0e0e;border-radius:3px">'
                '🔴 outlier</span>'
                '</div>',
                unsafe_allow_html=True,
            )

            # ── Comparison table ───────────────────────────────────────────────
            table_html = _render_vlm_table(vlm_rows, _active_models, thumb_map, render_map)
            st.markdown(table_html, unsafe_allow_html=True)

            # ── Full notes expanders ───────────────────────────────────────────
            st.markdown("---")
            any_notes = False
            for vrow in vlm_rows:
                notes_by_model = {
                    m: (vrow.get(f"{m}_notes") or "").strip()
                    for m in _active_models
                }
                if any(notes_by_model.values()):
                    any_notes = True
                    with st.expander(f"Run #{vrow['run_id']} — full notes", expanded=False):
                        for m, notes in notes_by_model.items():
                            if notes:
                                rating = vrow.get(f"{m}_rating") or "—"
                                st.markdown(
                                    f"**{_MODEL_LABEL.get(m, m)}** "
                                    f"({_vlm_badge_html(rating)}): {notes}",
                                    unsafe_allow_html=True,
                                )
            if not any_notes:
                st.caption("No notes for any run on this page.")

            # ── Generate panel ─────────────────────────────────────────────────
            st.markdown("---")
            _empty_cells = [
                (row, m)
                for row in vlm_rows
                for m in _active_models
                if not (row.get(f"{m}_rating") or "").strip()
            ]
            with st.expander(
                f"⚡ Generate missing annotations  ({len(_empty_cells)} unprocessed on this page)",
                expanded=bool(_empty_cells),
            ):
                _job = st.session_state.vlm_active_job
                _job_running = _job and _job["process"].poll() is None

                if _job_running:
                    jm = _MODEL_LABEL.get(_job["model_key"], _job["model_key"])
                    st.warning(
                        f"**{jm}** is running on run #{_job['run_id']} — "
                        f"only one inference at a time.",
                        icon="⟳",
                    )
                    if st.button("📺 Show terminal output", key="vlm_show_term_btn"):
                        st.session_state.vlm_show_terminal = True
                        st.rerun()
                    # Warn the user before navigating away or closing the tab —
                    # the subprocess is tied to this Streamlit server process via
                    # the Windows Job Object, so closing the browser won't kill it,
                    # but a page reload loses the terminal output handle.
                    import streamlit.components.v1 as _components
                    _components.html(
                        """<script>
                        window.onbeforeunload = function(e) {
                            var msg = "A VLM inference job is running. Closing this tab will lose terminal output (the subprocess will continue).";
                            e.returnValue = msg;
                            return msg;
                        };
                        </script>""",
                        height=0,
                    )

                if not _empty_cells:
                    st.caption("All cells on this page are annotated.")
                else:
                    _hdr = st.columns([2, 3, 2])
                    _hdr[0].caption("**Run**")
                    _hdr[1].caption("**Model**")
                    _hdr[2].caption("**Action**")
                    for _row, _m in _empty_cells:
                        _c1, _c2, _c3 = st.columns([2, 3, 2])
                        _c1.caption(f"#{_row['run_id']}")
                        _c2.caption(_MODEL_LABEL.get(_m, _m))
                        _btn_key = f"gen_{_row['run_id']}_{_m}"
                        if _job_running:
                            _c3.button(
                                "⟳ Running…", key=_btn_key,
                                disabled=True, use_container_width=True,
                            )
                        else:
                            if _c3.button(
                                "⚡ Generate", key=_btn_key,
                                use_container_width=True, type="primary",
                            ):
                                _launch_vlm_inference(
                                    _m, _row["run_id"],
                                    st.session_state.output_root,
                                )
                                st.rerun()

            # ── Agreement matrix ───────────────────────────────────────────────
            st.markdown("---")
            with st.expander("Pairwise agreement matrix", expanded=False):
                all_vlm_rows, _ = get_vlm_comparison_page(
                    conn, limit=500, offset=0,
                    models=_active_models, disagree_only=False,
                    ref_user_id=ref_user_id,
                )
                matrix_html = _render_agreement_matrix(
                    [dict(r) for r in all_vlm_rows], _active_models, _ref_label
                )
                st.markdown(matrix_html, unsafe_allow_html=True)
                st.caption(f"Based on up to 500 most recent annotated runs.")
