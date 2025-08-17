import json
from langchain_openai import ChatOpenAI
from app.intent_catalog import INTENTS

SYSTEM = """You are an intent detector for BizApps incentives.
Pick exactly one topic name from INTENTS and return STRICT JSON:
{"topic":"<one-of-intents>"}
No extra text."""

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