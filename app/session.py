# app/session.py
import os
import redis
import json
import datetime as dt
import decimal
import uuid
from typing import Any, Dict

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB   = int(os.getenv("REDIS_DB", 0))

# Global redis client
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True
)

def _json_default(o: Any):
    """JSON encoder for non-serializable types we may get from DB results."""
    if isinstance(o, (dt.datetime, dt.date)):
        return o.isoformat()
    if isinstance(o, decimal.Decimal):
        return float(o)  # switch to str(o) if you need full precision
    if isinstance(o, uuid.UUID):
        return str(o)
    return str(o)  # fallback

def _ensure_schema_defaults(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backfill missing keys so older sessions don't break when we add fields.
    """
    state.setdefault("session_id", None)
    state.setdefault("original_user_message", "")
    state.setdefault("required_fields", {})           # { key: [values] | None }
    state.setdefault("messages", [])                  # list of {role, text, field_name?}
    state.setdefault("intent_topics", [])

    # New fields for multi-intent + dialog control
    state.setdefault("current_intent_topic", None)    # active intent topic
    state.setdefault("pending_field", None)           # which field we're currently asking for
    state.setdefault("asked_log", {})                 # { field: {count, last_msg_index, last_question_text} }

    # Results + final answer caching
    state.setdefault("last_result", [])               # list of DB rows for the active intent
    state.setdefault("final_answer", None)            # {answer_text, recommendations} or None

    return state

def get_session(session_id: str) -> dict | None:
    """Fetch session state from Redis. Returns None if not found."""
    data = redis_client.get(session_id)
    if not data:
        return None
    try:
        state = json.loads(data)
    except Exception:
        return None
    return _ensure_schema_defaults(state)

def save_session(session_id: str, state: dict):
    """Save session state to Redis (30min expiry)."""
    redis_client.set(session_id, json.dumps(state, default=_json_default), ex=1800)

def create_session(session_id: str, user_message: str) -> dict:
    """Create a new session object with default schema."""
    state = {
        "session_id": session_id,
        "original_user_message": user_message,
        "required_fields": {},     # { key: [values] }
        "messages": [],            # list of {role, text, field_name?}
        "intent_topics": [],

        # Active intent management
        "current_intent_topic": None,
        "pending_field": None,
        "asked_log": {},

        # Results + final answer
        "last_result": [],
        "final_answer": None,
    }
    return _ensure_schema_defaults(state)

def add_message(session_id: str, role: str, text: str, field_name: str | None = None):
    """Append one message turn to session."""
    state = get_session(session_id)
    if not state:
        raise ValueError(f"Session {session_id} not found in Redis")

    state["messages"].append({
        "role": role,
        "text": text,
        "field_name": field_name
    })
    save_session(session_id, state)
