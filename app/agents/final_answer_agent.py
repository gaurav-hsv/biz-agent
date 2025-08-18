# app/agents/final_answer_agent.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, List
import math

from langchain_openai import ChatOpenAI

def _one(v):
    if isinstance(v, list) and v: return v[0]
    return v

def _to_float(v):
    if v is None: return None
    if isinstance(v, (int, float)): return float(v)
    s = str(v).strip().lower().replace(",", "")
    # support 10k / 100k / 1m style
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*([kKmM]?)", s)
    if not m:
        # last fallback: digits only
        try: return float(re.sub(r"[^0-9.\-]", "", s))
        except: return None
    num = float(m.group(1))
    suf = m.group(2)
    if suf in ("k","K"): num *= 1_000
    elif suf in ("m","M"): num *= 1_000_000
    return float(num)

def _norm_country(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def _country_in_def(country, definition_text):
    if not country or not definition_text: return False
    c = _norm_country(country)
    # split by comma/“and”/semicolon/slash
    tokens = [t.strip() for t in re.split(r",|;|/|\band\b", definition_text) if t.strip()]
    for t in tokens:
        if _norm_country(t) == c:
            return True
    # also allow word-boundary substring match as a safety net
    return re.search(rf"\b{re.escape(country)}\b", definition_text, flags=re.IGNORECASE) is not None

def _pick_band(row, country):
    if _country_in_def(country, row.get("market_a_definition")): return "A"
    if _country_in_def(country, row.get("market_b_definition")): return "B"
    return "C"

def _compute_presales_payout(row, country, acv, hours):
    """Returns a dict with band, inputs, candidates and final payout (or can_compute=False)."""
    band = _pick_band(row, country)
    key = band.lower()

    percent = row.get(f"incentive_market_{key}")         # e.g., 7.5
    hourly = row.get(f"workshop_rate_hourly_{key}")      # e.g., 163
    cap    = row.get("maximum_incentive_earning")        # e.g., 6000

    acv_f   = _to_float(acv)
    hours_f = _to_float(hours)

    candidates = []

    if percent is not None and acv_f is not None:
        candidates.append(("percent_of_acv", float(percent) / 100.0 * acv_f))
    if hourly is not None and hours_f is not None:
        candidates.append(("hours_x_rate", float(hourly) * hours_f))
    if cap is not None:
        candidates.append(("cap", float(cap)))

    if not candidates:
        return {"can_compute": False, "reason": "Missing inputs/fields", "band": band}

    winner = min(candidates, key=lambda kv: kv[1])
    out = {
        "can_compute": True,
        "band": band,
        "inputs": {
            "country": country,
            "acv": acv_f,
            "hours": hours_f,
            "percent": percent,
            "hourly_rate": hourly,
            "cap": cap,
        },
        "candidates": {k: v for k, v in candidates},
        "payout": winner[1],
        "limiter": winner[0],
    }
    return out

def precompute_calcs(required_fields, rows):
    """Build PRECOMPUTED_CALC for the LLM. One entry per row."""
    country = _one((required_fields or {}).get("country"))
    acv     = _one((required_fields or {}).get("acv"))
    hours   = _one((required_fields or {}).get("hours"))

    out = []
    for r in (rows or []):
        calc = _compute_presales_payout(r, country, acv, hours)
        out.append({
            "name": r.get("name"),
            "band": calc.get("band"),
            "can_compute": calc.get("can_compute", False),
            "inputs": calc.get("inputs"),
            "candidates": calc.get("candidates"),
            "payout": calc.get("payout"),
            "limiter": calc.get("limiter"),
        })
    return out


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

CALCULATIONS
- If PRECOMPUTED_CALC is present, USE THOSE NUMBERS AS-IS (do not recompute).
- Each entry includes: band (A/B/C), candidates used, limiter, and payout (if computable).
- If none are computable, say the calculation cannot be completed from the catalog and name the missing inputs.
- When computable, state the band used and a one-line breakdown of the compared terms (percent-of-ACV, hours×rate, cap), then the final payout = minimum.

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

PRECOMPUTED_CALC (use these numbers as-is; do not recompute):
{precomputed_calc_json}

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
        "recommendations": List[str]
      }
    """
    # Precompute deterministic payout math so the LLM only narrates
    precomp = precompute_calcs(required_fields or {}, filter_result or [])

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    msgs = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                original_user_message=original_user_message,
                required_fields_json=_safe_json(required_fields or {}),
                filter_result_json=_safe_json(filter_result or []),
                precomputed_calc_json=_safe_json(precomp or []),
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

        if not isinstance(recs, list) or not recs:
            fb = _fallback_answer(original_user_message, required_fields, filter_result)
            recs = fb.get("recommendations") or []
        norm_recs = _normalize_recommendations(recs)

        return {"answer_text": answer.strip(), "recommendations": norm_recs}
    except Exception:
        fb = _fallback_answer(original_user_message, required_fields, filter_result)
        fb["recommendations"] = _normalize_recommendations(fb.get("recommendations") or [])
        return fb