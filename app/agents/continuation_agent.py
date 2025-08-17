# app/agents/continuation_agent.py
from __future__ import annotations
import json
import re
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI


SYSTEM = """You are a continuation detector for a conversational assistant.

Task:
Decide if the NEW_USER_MESSAGE is a follow-up to the existing session context
(i.e., it can be answered using the already-computed LAST_RESULT without
re-running intent detection or collecting new required fields).

Guidelines:
- Return true when the message asks for clarifications, details, or next steps
  about the engagement(s)/data in LAST_RESULT (e.g., qualifications, modules,
  goals, activity requirements, rates, markets as defined in the rows, etc.).
- Return true when the message references "this/that workshop", "the above",
  "the engagement", or otherwise clearly relies on prior context shown.
- Return false when the message introduces a new topic or different filters
  (e.g., a different workload/incentive type/name), or asks about a different
  task outside the scope of the current result.
- If LAST_RESULT is empty or missing, return false.

Output STRICT JSON only:
{"is_continuation": true|false}
No extra text.
"""


def _last_messages_brief(messages: List[Dict[str, Any]], k: int = 8) -> List[Dict[str, str]]:
    """Take the last k messages and trim to minimal fields for the LLM."""
    brief = []
    for m in (messages or [])[-k:]:
        role = m.get("role")
        text = (m.get("text") or "")[:1000]  # keep prompt small
        field = m.get("field_name")
        brief.append({"role": role, "text": text, "field_name": field})
    return brief


def _result_names_summary(rows: List[Dict[str, Any]], cap: int = 5) -> List[str]:
    """Extract up to 'cap' engagement names from last_result for quick reference."""
    out = []
    for r in rows or []:
        name = r.get("name")
        if isinstance(name, str) and name.strip():
            out.append(name.strip())
        if len(out) >= cap:
            break
    return out


# simple, cheap heuristic to short-circuit obvious cases
_RESET_PATTERNS = [
    r"\bnew (request|topic|question)\b",
    r"\bstart over\b",
    r"\breset\b",
    r"\b(another|different) (engagement|workload|topic|customer)\b",
    r"\bchange (workload|incentive|topic)\b",
    r"\bcompare\b",
    r"\bvs\.?\b",
    r"\binstead\b",
]
_FOLLOWUP_HINTS = [
    "more detail", "details", "qualification", "qualifications",
    "activity requirement", "requirements", "modules", "module", "goal",
    "goals", "acv", "msx", "tpid", "proof of execution", "market",
    "market band", "rate", "earning", "hours", "duration", "scope",
    "deliverable", "deliverables", "summary", "summarize", "what about",
    "this workshop", "that workshop", "the workshop", "the engagement",
    "above", "those", "it", "can you", "how much", "what is", "what are"
]


def _quick_heuristic(user_message: str, session: Dict[str, Any]) -> bool | None:
    """
    Return True/False if clearly determinable, else None to defer to LLM.
    """
    msg = (user_message or "").strip().lower()
    last_result = session.get("last_result") or []

    # If no prior result, cannot be a continuation
    if not last_result:
        return False

    # Hard "new" indicators
    for pat in _RESET_PATTERNS:
        if re.search(pat, msg):
            return False

    # Strong follow-up hints
    if any(h in msg for h in _FOLLOWUP_HINTS):
        return True

    # Inconclusive → let LLM decide
    return None


def detect_continuation(user_message: str, session: Dict[str, Any]) -> Dict[str, bool]:
    """
    Decide whether this turn is a continuation of the current thread,
    which can be answered from session['last_result'] context.

    Returns: {"is_continuation": bool}
    """
    # 1) quick heuristic
    h = _quick_heuristic(user_message, session)
    if h is not None:
        return {"is_continuation": bool(h)}

    # 2) LLM classification
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    context = {
        "last_intent_topic": (session.get("last_intent") or {}).get("topic"),
        "last_result_names": _result_names_summary(session.get("last_result") or []),
        "messages_tail": _last_messages_brief(session.get("messages") or [], k=8),
        "new_user_message": user_message or "",
    }

    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)}
    ]

    try:
        resp = llm.invoke(msgs).content
        data = json.loads(resp)
        is_cont = bool(data.get("is_continuation"))
        return {"is_continuation": is_cont}
    except Exception:
        # Fallback: if we reached here, default to False (be safe and re-run the full flow)
        return {"is_continuation": False}
