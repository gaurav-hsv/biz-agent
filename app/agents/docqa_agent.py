# app/agents/docqa_agent.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import os
import time
import math
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError, APITimeoutError

load_dotenv()

# ---- Config ----
PG_DSN = os.getenv("DATABASE_URL") or os.getenv("PG_DSN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not PG_DSN:
    raise RuntimeError("DATABASE_URL/PG_DSN is not set")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

# ---- Models / dims ----
EMBED_MODEL = "text-embedding-3-small"  # 1536-d
CHAT_MODEL = "gpt-4o-mini"

# Distance/score: using IP => score ≈ 1 + cosine in [0,2]
HARD_ACCEPT = 1.60   # cosine >= 0.60
SOFT_ACCEPT = 1.50   # cosine >= 0.50

# Retrieval knobs
FIRST_STAGE_K = 12   # fetch more to allow reranking/diversity
MAX_CTX = 5          # passages we feed to the LLM
MMR_K = 5            # keep this many after rerank (<= MAX_CTX ideally)
MMR_LAMBDA = 0.7     # trade-off relevance vs diversity (0..1)

# Timeouts/retries
DB_STATEMENT_TIMEOUT_MS = 8000
OAI_MAX_RETRIES = 3
OAI_RETRY_BASE = 0.8  # seconds

_client = OpenAI(api_key=OPENAI_API_KEY)


# ---------- utilities ----------
def _retry_oai(func, *args, **kwargs):
    """Simple exponential backoff for OpenAI calls."""
    for attempt in range(1, OAI_MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except (RateLimitError, APIError, APITimeoutError) as e:
            if attempt == OAI_MAX_RETRIES:
                raise
            time.sleep(OAI_RETRY_BASE * (2 ** (attempt - 1)))
        except Exception:
            # non-retryable
            raise


def _embed(text: str) -> List[float]:
    r = _retry_oai(_client.embeddings.create, model=EMBED_MODEL, input=text)
    return r.data[0].embedding


def _embeds_texts(texts: List[str]) -> List[List[float]]:
    r = _retry_oai(_client.embeddings.create, model=EMBED_MODEL, input=texts)
    return [d.embedding for d in r.data]


def _cosine(u: List[float], v: List[float]) -> float:
    # robust cosine
    if not u or not v:
        return 0.0
    num = 0.0
    du = 0.0
    dv = 0.0
    for a, b in zip(u, v):
        num += a * b
        du += a * a
        dv += b * b
    if du == 0.0 or dv == 0.0:
        return 0.0
    return num / math.sqrt(du * dv)


def _format_citation(hit: Dict[str, Any]) -> str:
    sec = (hit.get("section") or hit.get("heading") or "Section").strip()
    pg = hit.get("page")
    base = f'{hit["file"]} — {sec}'
    return f'[{base} p.{pg}]' if pg else f'[{base}]'


# ---------- DB retrieval ----------
def _search_pg(query: str, k: int = FIRST_STAGE_K) -> List[Dict[str, Any]]:
    q_emb = _embed(query)

    # read-only conn with server-side timeout
    conn = psycopg2.connect(PG_DSN)
    try:
        conn.autocommit = True
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # set local statement timeout
            cur.execute(f"SET LOCAL statement_timeout TO {DB_STATEMENT_TIMEOUT_MS};")
            cur.execute(
                """
                SELECT d.file_name AS file,
                       c.section,
                       c.page,
                       c.text,
                       1 - (c.embedding <#> %s::vector) AS score
                FROM doc_chunks c
                JOIN doc_documents d ON d.id = c.document_id
                ORDER BY c.embedding <#> %s::vector
                LIMIT %s
                """,
                (q_emb, q_emb, k),
            )
            rows = cur.fetchall()
            return [
                {
                    "file": r["file"],
                    "section": r.get("section"),
                    "page": r.get("page"),
                    "text": r["text"],
                    "score": float(r["score"]),
                }
                for r in rows
            ]
    finally:
        conn.close()


# ---------- MMR rerank (diversify top-N) ----------
def _mmr_rerank(query: str, hits: List[Dict[str, Any]], k: int = MMR_K, lam: float = MMR_LAMBDA) -> List[Dict[str, Any]]:
    """
    Re-embed the candidate chunk texts to compute query & inter-chunk cosine,
    select k with Maximal Marginal Relevance.
    """
    if not hits:
        return []

    texts = [h["text"] for h in hits]
    try:
        q_emb = _embed(query)
        c_embs = _embeds_texts(texts)
    except Exception:
        # if embeddings fail, return top-k by original score
        return hits[:k]

    selected: List[int] = []
    candidate_ids = list(range(len(hits)))

    # first pick: highest cosine to query
    qs = [(_cosine(q_emb, c_embs[i]), i) for i in candidate_ids]
    qs.sort(reverse=True)
    if not qs:
        return hits[:k]
    selected.append(qs[0][1])
    candidate_ids.remove(qs[0][1])

    while len(selected) < min(k, len(hits)):
        best_id = None
        best_score = -1e9
        for i in candidate_ids:
            sim_to_query = _cosine(q_emb, c_embs[i])
            sim_to_selected = max(_cosine(c_embs[i], c_embs[j]) for j in selected) if selected else 0.0
            mmr = lam * sim_to_query - (1 - lam) * sim_to_selected
            if mmr > best_score:
                best_score = mmr
                best_id = i
        if best_id is None:
            break
        selected.append(best_id)
        candidate_ids.remove(best_id)

    return [hits[i] for i in selected]


_SYSTEM = (
    "You are a precise microsoft business app partner guide assistant that must answer only from the provided passages.\n"
    "Rules:\n"
    "- If the passages don’t contain the answer, say so explicitly.\n"
    "- Keep answers concise (max ~6 lines). Use bullet points only if steps are asked.\n"
)


def _build_context(hits: List[Dict[str, Any]]) -> str:
    parts = []
    for i, h in enumerate(hits[:MAX_CTX], 1):
        cite = _format_citation(h)
        parts.append(f"### Passage {i}\nSource: {cite}\n\n{h['text']}")
    return "\n\n".join(parts)


def _synthesize_answer(question: str, hits: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """
    Returns final_text, citations_list
    """
    if not hits:
        return "I don’t have enough policy context to answer that yet.", []

    ctx = _build_context(hits)

    # up to 3 unique (file, section) citations
    cites: List[str] = []
    seen: set[tuple[str, str]] = set()
    for h in hits:
        key = (h["file"], (h.get("section") or "").strip())
        if key in seen:
            continue
        seen.add(key)
        cites.append(_format_citation(h))
        if len(cites) >= 3:
            break

    msg_user = (
        f"User question:\n{question}\n\n"
        "Answer using the passages below. If the passages do not contain the answer, say so.\n\n"
        f"{ctx}"
    )

    resp = _retry_oai(
        _client.chat.completions.create,
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": msg_user},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        text = "I couldn’t generate a response from the provided passages."

    # Append citations at end (one line)
    return text, []


def docqa_turn(question: str, session: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entrypoint: retrieve → rerank → synthesize → update session with final_answer & last_docs.
    """
    try:
        hits = _search_pg(question, k=FIRST_STAGE_K)
    except Exception as e:
        session["final_answer"] = {"answer_text": "Sorry, I couldn’t query the document index right now.", "sources": []}
        session["last_docs"] = []
        session["last_docs_meta"] = {"error": str(e)}
        session["last_path"] = "doc_qa"
        return session

    # confidence gating on the raw top1
    top = hits[0]["score"] if hits else 0.0
    session["last_docs_meta"] = {
        "top_scores": [round(h["score"], 3) for h in hits[:5]],
        "first_stage_k": FIRST_STAGE_K,
    }

    # if top < SOFT_ACCEPT:
    #     ans = "I can look that up—do you mean timelines, POE, or eligibility/policies?"
    #     session["final_answer"] = {"answer_text": ans, "sources": []}
    #     session["last_docs"] = hits
    #     session["last_path"] = "doc_qa"
    #     return session

    # second-stage MMR rerank for diversity
    reranked = _mmr_rerank(question, hits, k=min(MMR_K, MAX_CTX))
    ctx_hits = reranked if reranked else hits[:MAX_CTX]

    answer_text, _ = _synthesize_answer(question, ctx_hits)
    session["final_answer"] = {"answer_text": answer_text, "sources": ctx_hits[:3]}
    session["last_docs"] = ctx_hits
    session["last_path"] = "doc_qa"
    return session
