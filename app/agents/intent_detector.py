import json
from langchain_openai import ChatOpenAI
from app.intent_catalog import INTENTS

SYSTEM = """
You are an intent detector for BizApps incentives.

OUTPUT: Return STRICT JSON only:
{"topic":"<one-of-intents>"}

RULES:
1) Pick EXACTLY ONE topic from INTENTS.
2) Prefer CALC intents ONLY when the user explicitly asks to compute/estimate/payout math.
3) Apply the following PRIORITY order with keyword/semantics:

   A. calc_presales_workshop_payout
      Trigger if any of: ("calculate"|"calc"|"compute"|"estimate"|"how much will we earn")
      AND mentions ("workshop"|"immersion"|hours|acv|rate).
   B. calc_presales_briefing_payout
      Trigger if: explicit calc/estimate AND mentions ("briefing"|"envisioning") without hours/acv.

   C. earning_amount
      Trigger if asking payout/rate/amount/cap/percentage ("how much", "rate", "market A/B/C", "cap", "maximum"),
      with NO explicit math request.

   D. activity_requirement
      Trigger if asking activities, requirements, scope, deliverables, min hours ("what are the requirements", "activities", "scope").

   E. partner_qualification
      Trigger if eligibility of the PARTNER is asked/implied:
      mentions "we/us/our company/partner/designation/specialization/solution partner".
   
   F. customer_qualification
      Trigger if eligibility of the CUSTOMER is asked/implied:
      mentions "customer/client/tenant/end customer" or “is my customer eligible”.

   G. recommend_engagement
      Fallback when user asks which program/what to do/recommendations without clear ask for payout/eligibility/requirements.

4) Disambiguation:
   - If both partner and customer eligibility are referenced, prefer partner_qualification.
   - If both requirements and payout are referenced, prefer payout (earning_amount) unless explicit math → choose calc_*.
   - If both calc_* are referenced, pick the one whose artifact is named (workshop vs briefing); if neither named and hours/acv present → workshop; else → briefing.

5) Calc gating (STRICT):
   - Choose calc_* only if: explicit intent to "calculate/compute/estimate" OR numeric inputs (hours/acv) are mentioned.
   - Otherwise NEVER pick calc_*.

Return ONLY the JSON object. No extra text.
"""

INTENT_NAMES = { item["topic"] for item in INTENTS }

def detect_intent(user_text: str) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    msgs = [
        {"role":"system","content": SYSTEM + "\nINTENTS:\n" + json.dumps([i["topic"] for i in INTENTS], ensure_ascii=False)},
        {"role":"user","content": user_text},
    ]
    resp = llm.invoke(msgs).content
    try:
        data = json.loads(resp)  # expect {"topic": "..."}
        topic = (data or {}).get("topic")
        if topic in INTENT_NAMES:
            # return the full object from catalog
            full = next(x for x in INTENTS if x["topic"] == topic)
            return {"topic": topic, "intent": full}
    except Exception:
        pass
    # fallback
    return {"topic": "unknown", "intent": None}