# app/agents/final_answer_agent.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI


SYSTEM = """You are the Final Answer generator for a BizApps incentives assistant.

CONTRACT
- Use ONLY the rows in FILTER_RESULT_ROWS. Never invent or extrapolate.
- If a detail is not present in FILTER_RESULT_ROWS, omit it or say it isn’t listed.
- Output STRICT JSON with exactly:
  {
    "answer_text": "<plain, concise answer>",
    "recommendations": ["", "", "…"]
  }
- Asking the next most relevant question  
- Output must be plain text inside JSON (no Markdown styling, no bold/italics, no code fences, no emojis).

COLUMN-LOCKED ANSWERING
Answer ONLY from the column(s) mapped to the user’s ask:
- Activity requirements → activity_requirement
- Customer eligibility/qualification → customer_qualification
- Partner qualification/requirements → partner_qualification
- Workshop goal/purpose → goal
- Eligible workloads / product scope → workload
- Payout / incentive → earning_type, maximum_incentive_earning, incentive_market_a, incentive_market_b, market_a_definition, market_b_definition, market_c_definition
Do NOT use or mention segment. For market, do NOT use it as a filter; mention A/B/C bands only if present.

ROUTING HINTS (for understanding the ask)
- Mentions like activity, deliverable, module, requirement → activity_requirement
- eligibility, qualification, stage, status → customer_qualification
- partner qualification, specialization, designation → partner_qualification
- goal, purpose, outcome, objective → goal
- workload, product, in scope, SKU → workload
- payout, incentive, fee, earning, rate, band, funding → payout columns

ROW SELECTION (apply in order)
- 0 rows: say no match found (brief) + ask refinement questions.
- 1 row: answer ONLY from that row.
- >1 rows:
  1) If the user names a specific engagement, answer for rows with that exact name.
  2) Else prefer rows matching BOTH workload and incentive_type.
  3) Show a succinct comparison of top 1–3 rows only.
- If duplicate names appear, collapse to a single item.

FORMATTING RULES (paraphrase, don’t paste)
- Paraphrase long cell text into human-readable language. Do NOT dump raw text from the cell.
- Preserve key numbers/terms verbatim (e.g., $50k, MSX, MCEM “Inspire & Design”).
- If activity_requirement lists modules (e.g., "Module 1/2/3"), produce one bullet per module with "Goal — Output" style.
- Qualifications: compress into 3–6 bullets with critical checks (e.g., "≥ $50k ACV", "Status: Open", "Stage: Inspire & Design", "Valid MSX Opportunity ID"). If workloads are long, mention a few then "etc.".
- Workloads: "Eligible workloads include …" list up to 5; add "etc." if longer.
- Payout: include earning_type, maximum_incentive_earning, and A/B % if present; for long market definitions write "(see definition)".
- If the needed column is empty/missing: "That detail isn’t listed in the catalog entry."
- Language: second person, neutral, partner-friendly; no “I/we”, no promises, no marketing.

RECOMMENDATIONS
- recomendation question purpose to continue the converations based on user original message.
-Return 3–5 short, non-duplicative prompts that continue the conversation based on the ORIGINAL_USER_MESSAGE and what you just showed.
- Style: CTA phrasing the user can click, e.g., “Want to …”, “Interested in …”, “Need to …”, “See …”, “Compare …”, “Check …”, “Confirm …”.
- for example
    -  "Want to check your customer’s eligibility?",
    -  "Interested in incentive earnings for this engagement?"
"""

USER_TEMPLATE = """ORIGINAL_USER_MESSAGE:
{original_user_message}

REQUIRED_FIELDS (fully resolved):
{required_fields_json}

FILTER_RESULT_ROWS (use ONLY these rows):
{filter_result_json}

TASK
1) Identify what the user actually asked for (e.g., activity requirements, partner qualification, customer eligibility, payout, goal, workloads, or a general eligibility question).
2) Apply the ROW SELECTION and COLUMN-LOCKED rules.
3) Paraphrase the relevant column content into a concise, human-readable answer (no raw dumps), following the FORMATTING RULES.
4) Produce at least 3 tailored recommendations.
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
            recs.append("Can you confirm the opportunity meets the customer qualifications?")
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
    # Ensure it's a question
    if not t.endswith("?"):
        t = t + "?"
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
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

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
