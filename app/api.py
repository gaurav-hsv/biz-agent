# app/api.py

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uuid

from app.agents.final_answer_agent import generate_final_answer
from app.agents.continuation_agent import detect_continuation  # continuation check
from app.session import get_session, create_session, save_session, add_message
from app.graph import app_graph
from app.agents.field_validator_v1 import field_validator_v1
from app.agents.followup_llm import generate_followup_question
from app.agents.field_value_resolver import resolve_field_from_message
from app.agents.db_filter_service import filter_incentives

router = APIRouter()

# ---------------- models ----------------
class TurnInput(BaseModel):
    session_id: Optional[str] = None
    user_message: str
    input_type: str  # "text" | "followup" | "follow_up"
    current_field: Optional[str] = None          # BE-friendly
    current_field_name: Optional[str] = None     # FE-friendly

# ---------------- helpers ----------------
def _list_or_none(v):
    if v is None:
        return None
    if isinstance(v, list):
        return v
    return [v]

def _merge_required_fields(old: Dict[str, Optional[List[str]]],
                           new: Dict[str, Optional[List[str]]]) -> Dict[str, Optional[List[str]]]:
    out: Dict[str, Optional[List[str]]] = {**(old or {})}
    for k, v in (new or {}).items():
        vv = _list_or_none(v)
        if vv is not None and len(vv) > 0:
            out[k] = vv
        else:
            out.setdefault(k, None)
    return out

def _trailing_from_rule(req_expr: List[str]) -> List[str]:
    return [e.strip() for e in (req_expr[1:] if isinstance(req_expr, list) else []) if isinstance(e, str) and e.strip()]

def _is_bad(v) -> bool:
    return (v is None) or (isinstance(v, list) and len(v) != 1)

def _next_missing_field(session: Dict[str, Any], req_expr: List[str]) -> Optional[str]:
    order = list(session.get("picked_set", [])) + _trailing_from_rule(req_expr)
    rf = session.get("required_fields") or {}
    for f in order:
        if _is_bad(rf.get(f)):
            return f
    for f in rf.keys():
        if _is_bad(rf.get(f)):
            return f
    return None

def _run_db_filter_and_save(session_id: str, s: Dict[str, Any], *, limit: int = 25, offset: int = 0):
    """Run DB filter by current required_fields and store rows in session['last_result']."""
    rf: Dict[str, Optional[List[str]]] = s.get("required_fields") or {}
    res = filter_incentives(rf, limit=limit, offset=offset)
    s["last_result"] = res.rows or []
    save_session(session_id, s)

def _run_final_answer_and_save(session_id: str, s: Dict[str, Any], override_message: Optional[str] = None):
    """
    Generate final answer from (message, required_fields, last_result). Save to session.
    Only append the answer_text to messages. DO NOT append recommendations.
    Also scrub any previously-added recommendation messages from history.
    """
    s = get_session(session_id) or s
    final = generate_final_answer(
        original_user_message=(
            override_message if override_message is not None
            else (s.get("original_user_message") or "")
        ),
        required_fields=s.get("required_fields") or {},
        filter_result=s.get("last_result") or [],
    )
    s["final_answer"] = final
    save_session(session_id, s)

    # 1) Append only the human-facing answer to messages.
    answer = final.get("answer_text")
    if isinstance(answer, str) and answer.strip():
        add_message(session_id, "assistant", answer.strip(), field_name=None)

    # 2) Ensure recommendations are NOT in chat history.
    #    If any were added by older runs, remove them now.
    recs = [r.strip() for r in (final.get("recommendations") or []) if isinstance(r, str) and r.strip()]
    if recs:
        s = get_session(session_id) or s
        msgs = s.get("messages") or []
        rec_set = set(recs)
        filtered = [
            m for m in msgs
            if not (
                m.get("role") == "assistant"
                and isinstance(m.get("text"), str)
                and m["text"].strip() in rec_set
            )
        ]
        if len(filtered) != len(msgs):
            s["messages"] = filtered
            save_session(session_id, s)


def _make_api_response(s: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert full session -> FE expected minimal payload:
    {
      session_id, response_type, text?, followup?, recommendations?
    }
    """
    out: Dict[str, Any] = {"session_id": s.get("session_id")}
    fu = s.get("followup") or {}
    final = s.get("final_answer") or {}

    if fu and fu.get("question") and fu.get("field_name"):
        out["response_type"] = "follow_up"
        out["followup"] = {
            "question": fu["question"],
            "field_name": fu["field_name"],
        }
        return out

    text = final.get("answer_text")
    recs = final.get("recommendations") or []
    if text or recs:
        out["response_type"] = "final_answer"
        if text:
            out["text"] = text
        if recs:
            out["recommendations"] = recs
        return out

    # fallback (rare)
    out["response_type"] = "follow_up"
    return out

def _is_followup_turn(inp: TurnInput) -> bool:
    t = (inp.input_type or "").lower().replace("-", "_")
    return t in ("followup", "follow_up")

def _pick_field_name(inp: TurnInput) -> Optional[str]:
    return (inp.current_field or inp.current_field_name or None)

# ---------------- route ----------------
@router.post("/message")
def turn(inp: TurnInput):
    # ensure session
    session_id = inp.session_id
    s = get_session(session_id) if session_id else None
    if not s:
        session_id = str(uuid.uuid4())
        s = create_session(session_id, inp.user_message)
        save_session(session_id, s)

    # ---- FOLLOWUP TURN ------------------------------------------------------
    if _is_followup_turn(inp):
        add_message(session_id, "user", inp.user_message, _pick_field_name(inp))
        s = get_session(session_id) or s

        field = (_pick_field_name(inp) or "").strip()
        if not field:
            s = get_session(session_id) or s
            return _make_api_response(s)

        # resolve the single field
        res = resolve_field_from_message(field, inp.user_message)
        value = res.get("value")

        if value is not None:
            rf: Dict[str, Optional[List[str]]] = s.get("required_fields") or {}
            rf[field] = [value]
            s["required_fields"] = rf
            save_session(session_id, s)
            s = get_session(session_id) or s
        else:
            q = generate_followup_question(field_name=field, intent=s.get("last_intent", {}), session=s)
            s["followup"] = {"question": q, "field_name": field, "options": None}
            save_session(session_id, s)
            add_message(session_id, "assistant", q, field)
            s = get_session(session_id) or s
            return _make_api_response(s)

        # after filling current field, check missing
        intent_obj = (s.get("last_intent") or {}).get("intent") or {}
        req_expr = intent_obj.get("required_fields") or []
        missing = _next_missing_field(s, req_expr)

        if not missing:
            s["followup"] = None
            save_session(session_id, s)
            s = get_session(session_id) or s
            _run_db_filter_and_save(session_id, s)
            _run_final_answer_and_save(session_id, s)
            s = get_session(session_id) or s
            return _make_api_response(s)

        # ask next missing
        options = None
        if missing in (s.get("candidates") or {}):
            opts = [c.get("value") for c in (s["candidates"].get(missing) or []) if c.get("value")]
            options = opts or None

        q = generate_followup_question(field_name=missing, intent=s.get("last_intent", {}), session=s)
        s["followup"] = {"question": q, "field_name": missing, "options": options}
        save_session(session_id, s)
        add_message(session_id, "assistant", q, missing)
        s = get_session(session_id) or s
        return _make_api_response(s)

    # ---- TEXT TURN ----------------------------------------------------------
    add_message(session_id, "user", inp.user_message, _pick_field_name(inp))
    s = get_session(session_id) or s

    # Continuation: answer from existing last_result without re-deriving intent
    if s.get("last_result"):
        cont = detect_continuation(inp.user_message, s)
        if cont.get("is_continuation"):
            _run_final_answer_and_save(session_id, s, override_message=inp.user_message)
            s = get_session(session_id) or s
            return _make_api_response(s)

    # Fresh intent detection pipeline
    result = app_graph.invoke({"session_id": session_id, "text": inp.user_message})

    s = get_session(session_id) or s
    s.setdefault("intent_topics", [])
    if result.get("intent"):
        s["last_intent"] = result["intent"]
        topic = result["intent"].get("topic")
        if topic and (not s["intent_topics"] or s["intent_topics"][-1] != topic):
            s["intent_topics"].append(topic)
        save_session(session_id, s)
        s = get_session(session_id) or s

    intent_obj = (s.get("last_intent") or {}).get("intent") or {}
    req_expr = intent_obj.get("required_fields") or []

    field_result = field_validator_v1(
        user_message=inp.user_message,
        required_fields=req_expr
    )

    new_rfo = field_result.get("required_fields_object", {})
    old_rfo = s.get("required_fields") or {}
    merged_rfo = _merge_required_fields(old_rfo, new_rfo)

    s["picked_set"] = field_result.get("picked_set", [])
    s["required_fields"] = {k: _list_or_none(v) for k, v in merged_rfo.items()}
    s["candidates"] = field_result.get("candidates", {})
    save_session(session_id, s)
    s = get_session(session_id) or s

    required_keys = list(s.get("picked_set", [])) + _trailing_from_rule(req_expr)
    is_complete = required_keys and all(
        isinstance(s["required_fields"].get(k), list) and len(s["required_fields"].get(k, [])) == 1
        for k in required_keys
    )

    if is_complete:
        s["followup"] = None
        save_session(session_id, s)
        s = get_session(session_id) or s
        _run_db_filter_and_save(session_id, s)
        _run_final_answer_and_save(session_id, s)
        s = get_session(session_id) or s
        return _make_api_response(s)

    # Ask next missing
    missing = _next_missing_field(s, req_expr)
    options = None
    if missing in (s.get("candidates") or {}):
        opts = [c.get("value") for c in (s["candidates"].get(missing) or []) if c.get("value")]
        options = opts or None

    q = generate_followup_question(field_name=missing, intent=s.get("last_intent", {}), session=s)
    s["followup"] = {"question": q, "field_name": missing, "options": options}
    save_session(session_id, s)
    add_message(session_id, "assistant", q, missing)

    s = get_session(session_id) or s
    return _make_api_response(s)
