# app/agents/router.py
from __future__ import annotations
import json
from typing import Any, Dict, Literal, List, Optional
from langchain_openai import ChatOpenAI

Route = Literal["incentive_lookup", "doc_qa"]

# -------------------------
# Conversation summarizers
# -------------------------
def _tail_messages(session: Optional[Dict[str, Any]], n: int = 6) -> List[Dict[str, str]]:
    msgs = (session or {}).get("messages") or []
    tail = msgs[-n:]
    out: List[Dict[str, str]] = []
    for m in tail:
        role = (m.get("role") or "user").strip()
        text = (m.get("text") or "").strip().replace("\n", " ")
        if len(text) > 280:
            text = text[:277] + "..."
        out.append({
            "role": role,
            "text": text,
            "field": (m.get("field_name") or "")[:64],
        })
    return out

def _summarize_session(session: Optional[Dict[str, Any]]) -> str:
    s = session or {}
    last_path = s.get("last_path") or ""
    last_intent = ((s.get("last_intent") or {}).get("intent") or {})
    topic = last_intent.get("topic") or ""
    picked = s.get("picked_set") or []
    rf = s.get("required_fields") or {}
    filled = [k for k, v in rf.items() if isinstance(v, list) and len(v) == 1]
    followup = (s.get("followup") or {}).get("field_name") or ""
    have_docs = bool(s.get("last_docs"))
    have_rows = bool(s.get("last_result"))
    tail = _tail_messages(s, n=6)

    summary = {
        "last_path": last_path,                 # "doc_qa" | "incentive_lookup" | ""
        "intent_topic": topic,                  # e.g., "incentive_lookup_by_market"
        "picked_set": picked,                   # order chosen from intent rule
        "filled_fields": filled,                # already known selectors
        "pending_followup_field": followup,     # expecting this next
        "have_docs_context": have_docs,         # doc_qa ran previously
        "have_table_rows": have_rows,           # incentive filter ran previously
        "recent_messages": tail,                # last few chat turns
    }
    return json.dumps(summary, ensure_ascii=False)

# -------------------------
# LLM-only Router (context-aware)
# -------------------------
_ROUTER_SYSTEM = """You are a deterministic ROUTER. Decide which single route best answers the NEW_USER_MESSAGE,
using both the new text and the CONVERSATION_CONTEXT.

ROUTES
- "incentive_lookup": engagements/workshops and business terms (funding, incentives, payouts, rates/%, caps, eligibility/qualification, segments, Market A|B|C, CSP transaction; pre/post sales when asking about money/eligibility).
- "doc_qa": guides/process/MCI policies, POE/deliverables/artifacts/templates/evidence, where-to-submit, timelines, SLA/TAT, approvals/exceptions.

CONTEXT RULES
- If last_path is "doc_qa" AND the new message is a short continuation (e.g., "POE", "template", "where to submit"), keep doc_qa.
- If there is a pending_followup_field, prefer the route implied by that field (usually incentive_lookup).
- If both themes appear, apply:
  • "requirements/required" + (poe/deliverables/documentation/submit/evidence/template) ⇒ doc_qa.
  • otherwise choose the dominant theme.
- Be decisive. Do NOT ask clarifying questions.

OUTPUT (STRICT JSON ONLY)
{"route":"incentive_lookup"|"doc_qa","why":"<≤1 sentence>","confidence":0.0-1.0}

EXAMPLES
Q: "What POE items are required for pre-sales workshops?"
A: {"route":"doc_qa","why":"POE items + required = documentation/deliverables","confidence":0.91}

Q: "What’s the payout rate for the Envisioning workshop in Market B?"
A: {"route":"incentive_lookup","why":"Asks payout rate for a workshop by market","confidence":0.94}

Q: "Are we eligible for assessment funding?"
A: {"route":"incentive_lookup","why":"Eligibility for funding","confidence":0.90}

Q: "Where do I submit the POE template?"
A: {"route":"doc_qa","why":"Submission + POE template = process/docs","confidence":0.93}

Q: "presales workshop requirements"
A: {"route":"doc_qa","why":"'requirements' here implies deliverables/process, not payout","confidence":0.70}

Q: "cap and % for SME segment"
A: {"route":"incentive_lookup","why":"Cap and percent are payout terms","confidence":0.88}

Q: "POE"
A: {"route":"doc_qa","why":"Single-word POE implies documentation context","confidence":0.85}
"""

def _llm_route(user_message: str, session: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    payload = {
        "NEW_USER_MESSAGE": user_message,
        "CONVERSATION_CONTEXT": _summarize_session(session)
    }
    resp = llm.invoke([
        {"role": "system", "content": _ROUTER_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
    ]).content

    # Strict parse; salvage JSON if wrapped
    try:
        data = json.loads(resp)
    except Exception:
        start, end = resp.find("{"), resp.rfind("}")
        data = json.loads(resp[start:end+1]) if start != -1 and end != -1 else {}

    route = data.get("route")
    why = data.get("why", "")
    conf_raw = data.get("confidence", 0.0)
    try:
        conf = float(conf_raw)
    except Exception:
        conf = 0.0

    if route not in ("incentive_lookup", "doc_qa"):
        route = "incentive_lookup"
        why = why or "Defaulted due to invalid route"
        conf = conf or 0.0

    return {"route": route, "by": "llm", "scores": {"confidence": conf, "why": why}}

# -------------------------
# Public API
# -------------------------
def route_message(user_message: str, session: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Decide route for this turn (LLM-only, conversation-aware).
    Returns:
      {
        "route": "incentive_lookup" | "doc_qa",
        "by": "llm",
        "scores": {"confidence": float, "why": str}
      }
    """
    return _llm_route(user_message, session)
