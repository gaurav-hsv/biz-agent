# app/agents/followup_llm.py
from __future__ import annotations

import json
import re
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI

# Canonical enums (internal only; never surface snake_case to users)
INCENTIVE_TYPES = ["pre_sales", "csp_transaction"]
SEGMENTS = ["enterprise", "smec"]

SYSTEM = """You generate ONE short question to collect a SINGLE missing field.

Hard rules:
- Ask ONLY for the field named in FIELD_NAME.
- ≤ 15 words, exactly ONE sentence, must end with a question mark.
- No explanations, no multi-sentence text, no bullets.
- No promises like "I can …" or "I'll …".
- If OPTIONS are provided, you MAY append a very brief "Options: a, b, c" inline (no bullets), but keep it concise.

Re-ask behavior:
- If ATTEMPT_COUNT > 1, REPHRASE the question (do not reuse the same wording).
- If ATTEMPT_COUNT > 2, add a micro-nudge like "to match the catalog" within the same sentence (still ≤ 15 words).

Output STRICT JSON: {"question":"<string>"} — nothing else.
"""

# ---------------- humanization helpers ----------------
_INCENTIVE_DISPLAY = {
    "pre_sales": "pre-sales",
    "csp_transaction": "CSP transaction",
}
_SEGMENT_DISPLAY = {
    "enterprise": "Enterprise",
    "smec": "SMEC",
}

# As a final safety net, replace any stray canonical tokens in model output
_DECANON_REPLACEMENTS = [
    (r"\bpre_sales\b", "pre-sales"),
    (r"\bcsp_transaction\b", "CSP transaction"),
    (r"\bsmec\b", "SMEC"),
]

def _humanize_value(field_name: str, v: str) -> str:
    if not isinstance(v, str):
        return str(v)
    val = v.strip()
    if field_name == "incentive_type":
        return _INCENTIVE_DISPLAY.get(val, val.replace("_", " "))
    if field_name == "segment":
        return _SEGMENT_DISPLAY.get(val, val.capitalize() if val.islower() else val)
    # For other fields (workload, name, country), keep as-is
    return val

def _humanize_list(field_name: str, vals: List[str]) -> List[str]:
    return [_humanize_value(field_name, x) for x in (vals or []) if isinstance(x, str) and x.strip()]

def _decanonicalize_text(txt: str) -> str:
    out = txt
    for pat, repl in _DECANON_REPLACEMENTS:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    # Also convert any remaining snake_case-ish token heuristically
    # (only when it's a single token like foo_bar)
    def _snake_to_words(m):
        token = m.group(0)
        # don't touch things that already got mapped above
        if token.lower() in {"pre_sales", "csp_transaction"}:
            return token
        return token.replace("_", " ")
    out = re.sub(r"\b[a-z]+_[a-z_]+\b", _snake_to_words, out)
    return out

# ---------------- core prompt helpers ----------------
def _build_hints(field_name: str, session: Dict[str, Any], options: Optional[List[str]]) -> Dict[str, Any]:
    hints: Dict[str, Any] = {}

    # Provide HUMAN-FRIENDLY allowed values (never snake_case)
    if field_name == "incentive_type":
        hints["allowed_values"] = _humanize_list(field_name, INCENTIVE_TYPES)
    elif field_name == "segment":
        hints["allowed_values"] = _humanize_list(field_name, SEGMENTS)

    # pass candidates (values only) in humanized form
    cands = (session.get("candidates") or {}).get(field_name) or []
    cand_vals = [c.get("value") for c in cands if isinstance(c, dict) and c.get("value")]
    if cand_vals:
        hints["candidate_values"] = _humanize_list(field_name, cand_vals[:5])

    # UI-provided options (e.g., from fuzzy matches) — humanized
    if options:
        hints["options"] = _humanize_list(field_name, options[:5])

    # current partial value if any — humanized
    cur = (session.get("required_fields") or {}).get(field_name)
    if isinstance(cur, list) and cur:
        hints["current_value"] = _humanize_list(field_name, cur)

    return hints

def _postprocess(q: Optional[str], last_question_text: Optional[str]) -> str:
    """Ensure final formatting, humanize any canonical tokens, and avoid trivial repetition."""
    if not isinstance(q, str) or not q.strip():
        return ""
    txt = q.strip()
    # Remove accidental surrounding quotes/brackets
    txt = txt.strip(" \n\r\t\"'`")

    # De-canonicalize any snake_case that slipped through
    txt = _decanonicalize_text(txt)

    # Enforce one sentence with question mark
    if not txt.endswith("?"):
        txt += "?"

    # Kill "I can" / "I'll" phrasing if any
    txt = re.sub(r"^\s*i\s+can\s+", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"^\s*i(?:'|’)ll\s+", "", txt, flags=re.IGNORECASE)

    # Avoid repeating exact previous question
    if last_question_text and txt.strip().lower() == last_question_text.strip().lower():
        if txt.lower().startswith("can you "):
            txt = re.sub(r"^can you ", "Could you ", txt, flags=re.IGNORECASE)
        else:
            txt = "Could you " + txt[0].lower() + txt[1:]
        if not txt.endswith("?"):
            txt += "?"

    # Word cap (best-effort).
    words = txt.split()
    if len(words) > 15:
        core = [w for w in words if w.lower() not in {"please", "kindly"}]
        txt = " ".join(core)
        if not txt.endswith("?"):
            txt += "?"

    return txt

def generate_followup_question(field_name: str,
                               intent: Dict[str, Any],
                               session: Dict[str, Any],
                               *,
                               attempt_count: int = 1,
                               last_question_text: Optional[str] = None,
                               options: Optional[List[str]] = None) -> str:
    """
    field_name: e.g., "incentive_type", "workload", "country", "segment", "name"
    intent:    s["last_intent"] (same object you already store)
    session:   full session dict (to give context like required_fields/candidates)
    attempt_count: 1 for first ask, increment on re-asks to trigger rephrasing/micro-nudge
    last_question_text: previous question we asked for this field (to avoid repetition)
    options: optional explicit suggestions to embed briefly
    Returns:   the question string; falls back to a minimal prompt if parsing fails.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    ctx = {
        "FIELD_NAME": field_name,
        "INTENT_TOPIC": (intent or {}).get("topic"),
        "ATTEMPT_COUNT": attempt_count,
        "LAST_QUESTION_TEXT": (last_question_text or ""),
        "HINTS": _build_hints(field_name, session, options),
    }

    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "JSON_CONTEXT=\n" + json.dumps(ctx, ensure_ascii=False)},
    ]

    try:
        resp = llm.invoke(msgs).content
        data = json.loads(resp)  # expect {"question": "..."}
        q = _postprocess((data or {}).get("question"), last_question_text)
        if q:
            return q
    except Exception:
        pass

    # ultra-minimal fallback if LLM returns garbage — humanized field name if possible
    fallback_field = _humanize_value(field_name, field_name)
    fallback = f"Please provide {fallback_field}?"
    return _postprocess(fallback, last_question_text)
