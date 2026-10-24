from __future__ import annotations

import streamlit as st

# 🛠️ MUST BE THE VERY FIRST STREAMLIT COMMAND
st.set_page_config(page_title="LangGraph Blog Writer", layout="wide")

import json
import os
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, List, Iterator, Tuple
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv  # 🛠️ ADDED THIS

# 🛠️ ADDED THIS: Load .env file so os.getenv() can find DATABASE_URL
load_dotenv()

# =========================================================
# 1. BRIDGE SECRETS
# =========================================================
try:
    for key in ("GROQ_API_KEY", "TAVILY_API_KEY", "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN", "DATABASE_URL"):
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass

# =========================================================
# 2. DATABASE SETUP (Supabase)
# =========================================================
DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    """Connect to PostgreSQL (Supabase)."""
    if not DATABASE_URL:
        return None
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor, sslmode='require')

def init_db():
    """Create the table if it doesn't exist."""
    if not DATABASE_URL:
        return
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS blogs (
                        id SERIAL PRIMARY KEY,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
            conn.commit()
    except Exception as e:
        st.error(f"Database connection failed: {e}")

# Initialize DB on load
init_db()

# =========================================================
# 3. IMPORT BACKEND
# =========================================================
try:
    from bwa_backend import app
except Exception as e:
    st.error(f"❌ Failed to load backend. Error: {e}")
    st.stop()

# -----------------------------
# Helpers
# -----------------------------
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    return re.sub(r"\s+", "_", s).strip("_") or "blog"

def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))
        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()

def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    try:
        config = {"max_concurrency": 1} 
        for step in graph_app.stream(inputs, config=config, stream_mode="updates"):
            yield ("updates", step)
        out = graph_app.invoke(inputs, config=config)
        yield ("final", out)
    except Exception as e:
        yield ("error", str(e))

def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            inner = next(iter(step_payload.values()))
            current_state.update(inner)
        else:
            current_state.update(step_payload)
    return current_state

# -----------------------------
# Database Helpers
# -----------------------------
def save_blog_to_db(title: str, content: str):
    if not DATABASE_URL: return
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO blogs (title, content) VALUES (%s, %s)", (title, content))
            conn.commit()
    except Exception as e:
        st.error(f"Failed to save to database: {e}")

def get_past_blogs_from_db() -> List[Dict]:
    if not DATABASE_URL: return []
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, title, created_at FROM blogs ORDER BY created_at DESC LIMIT 50")
                return cur.fetchall()
    except Exception:
        return []

def get_blog_content_from_db(blog_id: int) -> str:
    if not DATABASE_URL: return ""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT content FROM blogs WHERE id = %s", (blog_id,))
                row = cur.fetchone()
                return row["content"] if row else ""
    except Exception:
        return ""

# -----------------------------
# Markdown renderer
# -----------------------------
_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")

def _resolve_image_path(src: str) -> Path:
    src = src.strip().lstrip("./")
    return Path.cwd() / src

def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last : m.start()]
        if before:
            parts.append(("md", before))
        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()

    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]
        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = payload.split("|||", 1)
        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    parts[i + 1] = ("md", "\n".join(nxt.splitlines()[1:]))

        if src.startswith("http"):
            st.image(src, caption=caption or (alt or None), use_container_width=True)
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists():
                try:
                    st.image(str(img_path), caption=caption or (alt or None), use_container_width=True)
                except Exception:
                    st.warning(f"Could not display image `{src}`.")
            else:
                st.warning(f"Image not found: `{src}`")
        i += 1

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Blog Writing Agent")

with st.sidebar:
    st.header("Generate New Blog")
    topic = st.text_area("Topic", height=120)
    as_of = st.date_input("As-of date", value=date.today())
    run_btn = st.button("🚀 Generate Blog", type="primary")

    st.divider()
    st.subheader("Past blogs (Database)")

    if not DATABASE_URL:
        st.warning("⚠️ DATABASE_URL not set in .env. History will not save.")
    else:
        past_blogs = get_past_blogs_from_db()
        if not past_blogs:
            st.caption("No saved blogs found in database.")
            selected_blog_id = None
        else:
            options = {f"{b['title']} ({b['created_at'].strftime('%Y-%m-%d %H:%M')})": b['id'] for b in past_blogs}
            selected_label = st.radio("Select a blog to load", options=list(options.keys()), label_visibility="collapsed")
            selected_blog_id = options.get(selected_label)

            if st.button("📂 Load selected blog"):
                if selected_blog_id:
                    md_text = get_blog_content_from_db(selected_blog_id)
                    st.session_state["last_out"] = {
                        "plan": None, "evidence": [], "image_specs": [], "final": md_text
                    }
                    st.rerun()

if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs(
    ["🧩 Plan", "🔎 Evidence", "📝 Markdown Preview", "🖼️ Images", "🧾 Logs"]
)

logs: List[str] = []

if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    inputs: Dict[str, Any] = {
        "topic": topic.strip(), "mode": "", "needs_research": False, "queries": [],
        "evidence": [], "plan": None, "as_of": as_of.isoformat(), "recency_days": 7,
        "sections": [], "merged_md": "", "md_with_placeholders": "", "image_specs": [], "final": "",
    }

    status = st.status("Running graph…", expanded=True)
    progress_area = st.empty()
    current_state: Dict[str, Any] = {}
    last_node = None

    for kind, payload in try_stream(app, inputs):
        if kind == "updates":
            node_name = None
            if isinstance(payload, dict) and len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
                node_name = next(iter(payload.keys()))
            if node_name and node_name != last_node:
                status.write(f"➡️ Node: `{node_name}`")
                last_node = node_name

            current_state = extract_latest_state(current_state, payload)
            summary = {
                "mode": current_state.get("mode"),
                "tasks": len((current_state.get("plan") or {}).get("tasks", [])) if isinstance(current_state.get("plan"), dict) else None,
                "images": len(current_state.get("image_specs", []) or []),
                "sections_done": len(current_state.get("sections", []) or []),
            }
            progress_area.json(summary)
            logs.append(f"[{kind}] {json.dumps(payload, default=str)[:500]}")

        elif kind == "final":
            st.session_state["last_out"] = payload
            status.update(label="✅ Done", state="complete", expanded=False)
            logs.append("[final] received final state")
            
            # 🛠️ SAVE TO DATABASE HERE
            final_md = payload.get("final")
            plan_obj = payload.get("plan")
            if final_md and plan_obj and DATABASE_URL:
                blog_title = plan_obj.blog_title if hasattr(plan_obj, "blog_title") else "Untitled Blog"
                save_blog_to_db(blog_title, final_md)
                st.success("Blog saved to Supabase database!")
            
        elif kind == "error":
            st.error(f"❌ Graph execution failed: {payload}")
            status.update(label="❌ Failed", state="error")
            logs.append(f"[error] {payload}")

out = st.session_state.get("last_out")
if out:
    with tab_plan:
        st.subheader("Plan")
        plan_obj = out.get("plan")
        if not plan_obj:
            st.info("No plan found.")
        else:
            plan_dict = plan_obj.model_dump() if hasattr(plan_obj, "model_dump") else plan_obj
            st.write("**Title:**", plan_dict.get("blog_title"))
            tasks = plan_dict.get("tasks", [])
            if tasks:
                df = pd.DataFrame([{"id": t.get("id"), "title": t.get("title"), "words": t.get("target_words")} for t in tasks]).sort_values("id")
                st.dataframe(df, use_container_width=True, hide_index=True)

    with tab_evidence:
        st.subheader("Evidence")
        evidence = out.get("evidence") or []
        if not evidence:
            st.info("No evidence returned.")
        else:
            rows = []
            for e in evidence:
                e_dict = e.model_dump() if hasattr(e, "model_dump") else e
                rows.append({"title": e_dict.get("title"), "url": e_dict.get("url")})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab_preview:
        st.subheader("Markdown Preview")
        final_md = out.get("final") or ""
        if not final_md:
            st.warning("No final markdown found.")
        else:
            render_markdown_with_local_images(final_md)

            plan_obj = out.get("plan")
            blog_title = plan_obj.blog_title if hasattr(plan_obj, "blog_title") else "blog"
            md_filename = f"{safe_slug(blog_title)}.md"
            
            st.download_button("⬇️ Download Markdown", data=final_md.encode("utf-8"), file_name=md_filename, mime="text/markdown")
            bundle = bundle_zip(final_md, md_filename, Path("images"))
            st.download_button("📦 Download Bundle", data=bundle, file_name=f"{safe_slug(blog_title)}_bundle.zip", mime="application/zip")

    with tab_images:
        st.subheader("Images")
        images_dir = Path("images")
        if not images_dir.exists():
            st.info("No images generated.")
        else:
            files = [p for p in images_dir.iterdir() if p.is_file() and not p.name.startswith(".")]
            for p in sorted(files):
                try:
                    st.image(str(p), caption=p.name, use_container_width=True)
                except Exception as e:
                    st.warning(f"Failed to display {p.name}: {e}")

    with tab_logs:
        st.subheader("Logs")
        st.text_area("Event log", value="\n\n".join(logs[-50:]), height=400)
else:
    st.info("Enter a topic and click **Generate Blog**.")