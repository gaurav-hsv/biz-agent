# app/agents/final_answer_agent.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI


SYSTEM = """You are a Final Answer generator for BizApps incentives.

Rules:
- Use ONLY the provided FILTER_RESULT_ROWS. Do NOT invent or infer beyond those rows.
- If a detail is not present in FILTER_RESULT_ROWS, omit it.
- Write in short, plain language, suitable for a partner/customer.
- Recommendations must be USER-GUIDANCE QUESTIONS phrased for the user to ask the agent, starting with "Can you ...".
- Do NOT offer to perform tasks (avoid "I can ...").

Output STRICT JSON with EXACT keys:
{
  "answer_text": "<plain short answer for the user>",
  "recommendations": ["<user-guidance question>", "<user-guidance question>", "<user-guidance question>", ...]
}
Requirements:
- "recommendations" MUST include at least 3 items.
- No extra text or keys.
"""

USER_TEMPLATE = """ORIGINAL_USER_MESSAGE:
{original_user_message}

REQUIRED_FIELDS (fully resolved):
{required_fields_json}

FILTER_RESULT_ROWS (DB rows; use ONLY these):
{filter_result_json}

Instructions to generate output:
- If FILTER_RESULT_ROWS is empty:
  - "answer_text": briefly state that no matching engagement was found based on the provided filters.
  - "recommendations": at least 3 short questions that help the user refine inputs (e.g., confirm workload variant, try a related workload, specify incentive type).
- If there is exactly 1 row:
  - "answer_text": concise summary that mentions the engagement name and the most relevant highlights present in the row (e.g., goal, key activity requirements). Do not add anything not present in the row.
  - "recommendations": at least 3 short, concrete questions grounded in the row fields (e.g., confirm customer qualifications like ACV/stage; choose which module to emphasize; confirm market band if definitions are present).
- If multiple rows:
  - "answer_text": brief comparison of the top 1–3 relevant matches (name + one-liner why each fits). Use only details present in rows.
  - "recommendations": at least 3 short questions to help the user choose or proceed.

Style:
- Keep "answer_text" to a few sentences or short bullets.
- "recommendations" must be short questions (no promises to draft emails, documents, or do tasks).
"""


def _safe_json(o: Any) -> str:
    return json.dumps(o, ensure_ascii=False, indent=2)


def _fallback_answer(original_user_message: str,
                     required_fields: Dict[str, Any],
                     filter_result: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Deterministic, safe fallback with recommendations phrased as user questions."""
    if not filter_result:
        return {
            "answer_text": "No matching engagement was found based on the current filters.",
            "recommendations": [
                "Can you clarify the exact workload name or try a related workload?",
                "Can you confirm which incentive type you want to use?",
                "Can you share any part of the engagement name you have in mind?"
            ]
        }
    if len(filter_result) == 1:
        row = filter_result[0]
        name = row.get("name") or "this engagement"
        recs: List[str] = []
        if row.get("customer_qualification"):
            recs.append("Can you confirm the opportunity meets the customer qualifications (e.g., ACV and stage)?")
        if any(k in row for k in ("market_a_definition", "market_b_definition", "market_c_definition")):
            recs.append("Can you tell me which market band applies to your customer (A, B, or C)?")
        if row.get("activity_requirement"):
            recs.append("Can you tell me which module you want to emphasize first: Assess, Art of the Possible, or Build the Plan?")
        while len(recs) < 3:
            recs.append("Can you confirm if you want a concise outline to brief your customer on this workshop?")
        return {
            "answer_text": f"{name} appears to fit your inputs based on the catalog entry.",
            "recommendations": recs
        }
    names = [r.get("name") for r in filter_result[:3] if r.get("name")]
    joined = "; ".join(names) if names else "multiple options"
    return {
        "answer_text": f"I found {joined} that align with your inputs.",
        "recommendations": [
            "Can you tell me which option best maps to your immediate goal?",
            "Can you confirm whether you prefer a workload-specific focus or a broader envisioning approach?",
            "Can you confirm any prerequisites listed (e.g., opportunity stage or size) to narrow the choices?"
        ]
    }


# ---- post-processing to enforce phrasing ----

_I_CAN_PREFIX = re.compile(r"^\s*i\s+can\s+", re.IGNORECASE)

def _flip_pronouns(text: str) -> str:
    # Basic swaps so questions read naturally from user → agent
    text = re.sub(r"\byour\b", "my", text, flags=re.IGNORECASE)
    text = re.sub(r"\byours\b", "mine", text, flags=re.IGNORECASE)
    text = re.sub(r"\byou\b", "me", text, flags=re.IGNORECASE)
    return text

def _to_question(text: str) -> str:
    t = (text or "").strip().rstrip(".")
    if not t:
        return ""
    # Convert leading "I can ..." to "Can you ..."
    if _I_CAN_PREFIX.match(t):
        t = _I_CAN_PREFIX.sub("", t)
        t = _flip_pronouns(t)
        t = f"Can you {t}"
    # Ensure it's a question
    if not t.endswith("?"):
        t = t + "?"
    # Normalize start
    if not re.match(r"^(Can you|Would you|Do you)\b", t, flags=re.IGNORECASE):
        # Default to "Can you ..."
        # Keep the original content but make it a question
        if not t.lower().startswith("can you "):
            # lowercase first char of content part to blend naturally
            core = t
            if len(core) > 1:
                core = core[0].lower() + core[1:]
            t = "Can you " + core
    return t

def _normalize_recommendations(recs: List[str]) -> List[str]:
    out: List[str] = []
    for r in recs or []:
        if not isinstance(r, str):
            continue
        q = _to_question(r)
        if q:
            out.append(q.strip())
    # de-dup (case-insensitive) while preserving order
    seen = set()
    uniq = []
    for q in out:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(q)
    # ensure minimum 3 by padding with generic helpful questions
    PAD = [
        "Can you confirm the exact workload or a close variant you want to target?",
        "Can you confirm which incentive type applies here?",
        "Can you tell me if any specific engagement name should be considered?"
    ]
    i = 0
    while len(uniq) < 3 and i < len(PAD):
        if PAD[i].lower() not in {x.lower() for x in uniq}:
            uniq.append(PAD[i])
        i += 1
    return uniq[:5]  # cap for UI sanity


def generate_final_answer(original_user_message: str,
                          required_fields: Dict[str, Any],
                          filter_result: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns:
      {
        "answer_text": str,
        "recommendations": List[str]  # >= 3 items, phrased as user questions
      }
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    msgs = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                original_user_message=original_user_message,
                required_fields_json=_safe_json(required_fields or {}),
                filter_result_json=_safe_json(filter_result or []),
            ),
        },
    ]

    try:
        resp = llm.invoke(msgs).content
        data = json.loads(resp)

        answer = (data or {}).get("answer_text")
        recs = (data or {}).get("recommendations")

        if not isinstance(answer, str) or not answer.strip():
            fb = _fallback_answer(original_user_message, required_fields, filter_result)
            fb["recommendations"] = _normalize_recommendations(fb.get("recommendations") or [])
            return fb

        # normalize recommendations → user questions
        if not isinstance(recs, list) or not recs:
            fb = _fallback_answer(original_user_message, required_fields, filter_result)
            recs = fb.get("recommendations") or []
        norm_recs = _normalize_recommendations(recs)

        return {
            "answer_text": answer.strip(),
            "recommendations": norm_recs
        }
    except Exception:
        fb = _fallback_answer(original_user_message, required_fields, filter_result)
        fb["recommendations"] = _normalize_recommendations(fb.get("recommendations") or [])
        return fb
