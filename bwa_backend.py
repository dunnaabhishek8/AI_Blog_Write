from __future__ import annotations

import operator
import os
import re
import time
import base64
import json
import requests
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# Import specific exceptions for error handling
from groq import BadRequestError, RateLimitError

load_dotenv()

# ============================================================
# Blog Writer
# ============================================================

# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=2, max_length=6) # Relaxed to 2 to prevent validation errors
    target_words: int = Field(..., description="Target words (120–550).")
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False

class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]

class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None

class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)

class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)

class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model. Do NOT include any text, words, or letters in this prompt.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"

class GlobalImagePlan(BaseModel):
    images: List[ImageSpec] = Field(default_factory=list)

class State(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    as_of: str
    recency_days: int
    sections: Annotated[List[tuple[int, str]], operator.add]
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    final: str

# -----------------------------
# 2) LLM Configuration
# -----------------------------
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    max_tokens=2048,
    max_retries=2,
)

# -----------------------------
# 3) Router
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.
Decide whether web research is needed BEFORE planning.
Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.
If needs_research=true: Output 3–10 high-signal, scoped queries."""

def router_node(state: State) -> dict:
    decider = llm.with_structured_output(RouterDecision, method="function_calling")
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ]
    )

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (Tavily)
# -----------------------------
def _tavily_search(query: str, max_results: int = 3) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        out: List[dict] = []
        for r in results or []:
            out.append({
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            })
        return out
    except Exception:
        return []

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None

RESEARCH_SYSTEM = """You are a research synthesizer. Given raw web search results, produce EvidenceItem objects.
Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets short. Deduplicate by URL."""

def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:5]
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=3))

    if not raw:
        return {"evidence": []}

    extractor = llm.with_structured_output(EvidencePack, method="function_calling")
    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(content=f"As-of date: {state['as_of']}\nRecency days: {state['recency_days']}\n\nRaw results:\n{str(raw)[:4000]}"),
        ]
    )

    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    evidence = list(dedup.values())

    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]

    return {"evidence": evidence}

# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer. Produce a highly actionable outline for a blog post.
Requirements:
- 5–7 tasks, each with goal + 3–6 bullets + target_words.
- CRITICAL: You MUST provide at least 3 bullets for EVERY task. Do not use fewer than 3.
- Tags are flexible; do not force a fixed taxonomy.
- closed_book: evergreen. hybrid: use evidence for up-to-date examples.
- open_book: weekly/news roundup (set blog_kind="news_roundup")."""

def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan, method="function_calling")
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])
    forced_kind = "news_roundup" if mode == "open_book" else None

    try:
        plan = planner.invoke(
            [
                SystemMessage(content=ORCH_SYSTEM),
                HumanMessage(content=f"Topic: {state['topic']}\nMode: {mode}\nAs-of: {state['as_of']}\n\nEvidence:\n{str([e.model_dump() for e in evidence][:3])[:2000]}"),
            ]
        )
        if forced_kind:
            plan.blog_kind = "news_roundup"
        return {"plan": plan}
    except Exception as e:
        print(f"Plan generation failed: {e}")
        # Fallback plan just in case the AI completely fails to generate JSON
        fallback_plan = Plan(
            blog_title=f"Blog on {state['topic']}",
            audience="General Audience",
            tone="Informative",
            blog_kind="explainer",
            tasks=[Task(id=1, title="Overview", goal="Provide an overview.", bullets=["Point 1", "Point 2", "Point 3"], target_words=250)]
        )
        return {"plan": fallback_plan}

# -----------------------------
# 6) Fanout & Worker
# -----------------------------
def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]

# 🛠️ UPDATED: Added strict spelling and proofreading instructions
WORKER_SYSTEM = """You are a senior technical writer. Write ONE section in Markdown.
Constraints:
- Cover ALL bullets in order.
- Target words ±15%.
- Output only section markdown starting with "## <Section Title>".
- CRITICAL: Proofread your work. Ensure perfect spelling, grammar, and punctuation. Double-check the spelling of character names, technical terms, and cultural words before writing them."""

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(f"- {e.title} | {e.url}" for e in evidence[:3])

    max_retries = 3
    base_delay = 10
    
    for attempt in range(max_retries):
        try:
            section_md = llm.invoke(
                [
                    SystemMessage(content=WORKER_SYSTEM),
                    HumanMessage(
                        content=(
                            f"Blog title: {plan.blog_title}\nTone: {plan.tone}\n"
                            f"Section title: {task.title}\nGoal: {task.goal}\n"
                            f"Bullets:{bullets_text}\n\n"
                            f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                        )
                    ),
                ]
            ).content.strip()
            return {"sections": [(task.id, section_md)]}
            
        except RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = base_delay * (2 ** attempt)
                match = re.search(r"try again in (\d+\.?\d*)s", str(e))
                if match:
                    wait_time = float(match.group(1)) + 1
                print(f"Rate limit hit for Task {task.id}. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                return {"sections": [(task.id, f"## {task.title}\n\n*Error: Rate limit exceeded.*")]}
        except Exception as e:
            return {"sections": [(task.id, f"## {task.title}\n\n*Error: {e}*")]}

# ============================================================
# 8) ReducerWithImages
# ============================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}

# 🛠️ UPDATED: Banned text in images and added cultural accuracy instructions
DECIDE_IMAGES_SYSTEM = """You are an expert technical editor. Decide if images/diagrams are needed.
Rules:
- Max 3 images total.
- CRITICAL: Do NOT include any text, words, letters, or labels in the image prompt. AI image generators cannot spell correctly, so focus purely on visual descriptions (e.g., "A warrior with a bow", not "A poster saying Rama").
- For mythological or cultural topics, describe the setting, attire, and weapons accurately (e.g., "ancient Indian architecture, traditional clothing, divine atmosphere").
- Avoid decorative images; prefer impactful visual representations."""

def decide_images(state: State) -> dict:
    planner = llm.with_structured_output(GlobalImagePlan, method="function_calling")
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    try:
        image_plan = planner.invoke(
            [
                SystemMessage(content=DECIDE_IMAGES_SYSTEM),
                HumanMessage(content=f"Blog kind: {plan.blog_kind}\nTopic: {state['topic']}\n\nText:\n{merged_md[:2000]}"),
            ]
        )
        return {"image_specs": [img.model_dump() for img in image_plan.images]}
    except Exception as e:
        # If the AI fails to output a valid tool call, just skip images
        print(f"Image generation skipped due to error: {e}")
        return {"image_specs": []}

def insert_placeholders(state: State) -> dict:
    md = state["merged_md"]
    specs = state.get("image_specs", [])
    
    if not specs: 
        return {"md_with_placeholders": md}
    
    for i, spec in enumerate(specs):
        placeholder = f"[[IMAGE_{i + 1}]]"
        spec["placeholder"] = placeholder
        md += f"\n\n{placeholder}\n"
        
    return {"md_with_placeholders": md}

# ✅ FIXED: Cloudflare Image Generation with Base64 Decoding
def _cloudflare_generate_image_bytes(prompt: str) -> bytes:
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not account_id or not api_token:
        raise RuntimeError("Cloudflare credentials not set.")

    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/black-forest-labs/flux-1-schnell"
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
    payload = {"prompt": prompt}

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    
    try:
        data = response.json()
        if data.get("success") and "result" in data and "image" in data["result"]:
            return base64.b64decode(data["result"]["image"])
    except json.JSONDecodeError:
        pass
        
    return response.content

def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    return re.sub(r"\s+", "_", s).strip("_") or "blog"

def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    specs = state.get("image_specs", [])

    if not specs:
        Path(f"{_safe_slug(plan.blog_title)}.md").write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = Path("images").resolve()
    images_dir.mkdir(exist_ok=True)

    for spec in specs:
        placeholder = spec["placeholder"]
        out_path = images_dir / spec["filename"]

        if not out_path.exists():
            try:
                img_bytes = _cloudflare_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                md = md.replace(placeholder, f"> **[IMAGE FAILED]** {e}")
                continue

        img_md = f"![{spec['alt']}](images/{spec['filename']})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    Path(f"{_safe_slug(plan.blog_title)}.md").write_text(md, encoding="utf-8")
    return {"final": md}

# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("insert_placeholders", insert_placeholders)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)

reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "insert_placeholders")
reducer_graph.add_edge("insert_placeholders", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()