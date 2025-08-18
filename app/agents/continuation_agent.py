# app/agents/continuation_agent.py
from __future__ import annotations
import json
import re
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI


SYSTEM = """You are a continuation detector for a conversational assistant.

Task:
Given the NEW_USER_MESSAGE and the current session’s LAST_RESULT (rows already computed),
decide if the new message is a follow-up that can be answered from LAST_RESULT without
re-running intent detection or collecting new required fields.

Hard rules (evaluate in order):
1) If LAST_RESULT is empty or missing → return false.
2) If NEW_USER_MESSAGE explicitly mentions a different engagement name than those in LAST_RESULT
   (e.g., says "CRM ..." while LAST_RESULT is about "ERP ...") → return false.
3) If NEW_USER_MESSAGE introduces a different workload/incentive type/name not present in LAST_RESULT → return false.
4) If NEW_USER_MESSAGE requests comparison to another engagement not in LAST_RESULT → return false.
5) If NEW_USER_MESSAGE clearly refers anaphorically to the current result
   (“this/that workshop”, “the above”, “that engagement”, “same workshop”) → return true.
6) If NEW_USER_MESSAGE asks for clarifications/details/next steps that are attributes of rows in LAST_RESULT
   (e.g., goals, activity requirements, partner/customer qualifications, markets/rates/amounts/caps, proof of execution, min hours) → return true.
7) If NEW_USER_MESSAGE asks for values that require changing selectors beyond what LAST_RESULT covers
   (e.g., switching to a different engagement name or different workload not in LAST_RESULT) → return false.
8) When ambiguous, prefer false unless the message clearly relies on the prior result.

Normalization guidance:
- Normalize case, spacing, punctuation, dashes, quotes, and &→“and”.
- Treat synonyms/aliases that map to the SAME canonical engagement in LAST_RESULT as the same.
- If the message names a different canonical engagement than any in LAST_RESULT (even if similar
  like “CRM Envisioning Workshop” vs “ERP Envisioning Workshop”) → it is NOT a continuation.

Output STRICT JSON only:
{"is_continuation": true|false}
No extra text, no explanations.
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
_ANAPHORA_HINTS = [
    "this workshop", "that workshop", "the workshop", "the engagement", "above", "this", "that", "those", "it"
]

# Detail terms are allowed ONLY if they don't conflict with a family switch
_DETAIL_HINTS = [
    "more detail", "details", "qualification", "qualifications",
    "activity requirement", "requirements", "modules", "module", "goal", "goals",
    "acv", "msx", "tpid", "proof of execution", "market", "market band",
    "rate", "earning", "hours", "duration", "scope", "deliverable", "deliverables",
    "summary", "summarize", "what about", "how much", "what is", "what are", "can you"
]


def _clean_text(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("&", " and ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Minimal engagement "family" synonyms so we can detect switches like ERP <-> CRM
_NAME_FAMILY_SYNS = {
    "erp": [
        "erp", "finance", "fscm", "supply chain", "business central", "bc",
        "project operations", "commerce", "human resources", "hr"
    ],
    "crm": [
        "crm", "customer engagement", "sales", "customer service",
        "field service", "contact center", "customer insights", "ci"
    ],
}

def _mentions_family(text: str, family: str) -> bool:
    t = _clean_text(text)
    for kw in _NAME_FAMILY_SYNS.get(family, []):
        if _clean_text(kw) in t:
            return True
    return False

def _family_in_names(family: str, names: list[str]) -> bool:
    big = _clean_text(" || ".join(n for n in (names or []) if isinstance(n, str)))
    for kw in _NAME_FAMILY_SYNS.get(family, []):
        if _clean_text(kw) in big:
            return True
    return False

def _mentions_different_family(msg: str, last_names: list[str]) -> bool:
    """True if msg clearly names a different family than what exists in last_names."""
    for fam in _NAME_FAMILY_SYNS.keys():
        if _mentions_family(msg, fam) and not _family_in_names(fam, last_names):
            return True
    return False


def _quick_heuristic(user_message: str, session: Dict[str, Any]) -> bool | None:
    """
    Return True/False if clearly determinable, else None to defer to LLM.
    """
    msg = (user_message or "")
    msg_clean = _clean_text(msg)
    last_result = session.get("last_result") or []
    last_names = _result_names_summary(last_result)

    # If no prior result, cannot be a continuation
    if not last_result:
        return False

    # Hard "new" indicators
    for pat in _RESET_PATTERNS:
        if re.search(pat, msg_clean):
            return False

    # If the message explicitly mentions a different engagement family than LAST_RESULT, it's NOT a continuation
    if _mentions_different_family(msg, last_names):
        return False

    # Strong anaphora → continuation
    if any(h in msg_clean for h in _ANAPHORA_HINTS):
        return True

    # Detail terms allowed only when no family conflict detected above
    if any(h in msg_clean for h in _DETAIL_HINTS):
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
