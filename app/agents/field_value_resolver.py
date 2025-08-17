# app/agents/field_value_resolver.py
from __future__ import annotations
"""
Single-field resolver.

Given (field_name, user_message) it returns a canonical value (or None) and optional candidates.
- workload: synonyms + partial + fuzzy over canonical list (workloads.json + your DB, if you merge upstream)
- incentive_type: regex/keyword canonicalization -> {'pre_sales','post_sales','csp_transaction'}
- segment: regex/keyword canonicalization -> {'enterprise','smec'}
- market: LLM extracts ONE country; we validate against ISO-3166 using pycountry (no local JSON)

Return shape:
{
  "field_name": "<input field>",
  "value": "<canonical or None>",
  "candidates": [ { "value": str, "score": int }, ... ]  # empty for market
}
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import json
import re

from rapidfuzz import process, fuzz
from langchain_openai import ChatOpenAI
import pycountry


# ---------- paths (JSONs live in app/) ----------
_APP_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_WL_SYNS_PATH = os.environ.get("WORKLOAD_SYNS_PATH", os.path.join(_APP_DIR, "workload_synonyms.json"))
_WL_LIST_PATH = os.environ.get("WORKLOADS_PATH",     os.path.join(_APP_DIR, "workloads.json"))

# ---------- static canonical options ----------
INCENTIVE_TYPES = ["pre_sales", "post_sales", "csp_transaction"]
SEGMENTS = ["enterprise", "smec"]


# ---------- utils ----------
def _clean(s: Optional[str]) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return (
        s.replace("–", "-")
         .replace("—", "-")
         .replace("’", "'")
         .replace("“", '"')
         .replace("”", '"')
    )


def _tokens(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", _clean(s))


def _safe_load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


# load once
_WL_SYNS: Dict[str, List[str]] = _safe_load_json(_WL_SYNS_PATH, {})
_WL_LIST: List[str] = _safe_load_json(_WL_LIST_PATH, [])


# ---------- matching primitives ----------
def _synonym_hits(
    msg: str,
    mapping: Dict[str, List[str]],
    exact: int = 100,
    contains: int = 96,
    token_subset: int = 93
) -> List[Tuple[str, int]]:
    """
    Generate (canonical, score) pairs using a synonyms map.
    - exact clean match -> exact
    - contains (either direction) -> contains
    - token subset (either direction) -> token_subset
    """
    if not msg or not mapping:
        return []
    m_clean = _clean(msg)
    m_tok = set(_tokens(msg))
    out: List[Tuple[str, int]] = []

    for canon, alts in mapping.items():
        c_clean = _clean(canon)
        if m_clean == c_clean:
            out.append((canon, exact))
            continue
        for a in alts or []:
            a_clean = _clean(a)
            if not a_clean:
                continue
            if m_clean == a_clean:
                out.append((canon, exact))
                break
            if a_clean in m_clean or m_clean in a_clean:
                out.append((canon, contains))
                break
            a_tok = set(_tokens(a))
            if a_tok and (a_tok.issubset(m_tok) or m_tok.issubset(a_tok)):
                out.append((canon, token_subset))
                break
    return out


def _fuzzy_hits(
    msg: str,
    choices: List[str],
    limit: int = 8,
    cutoff: int = 70
) -> List[Tuple[str, int]]:
    """
    Combine token_set_ratio and partial_ratio. Keep the best score per original choice.
    """
    if not msg or not choices:
        return []
    cln = _clean(msg)
    cleaned_choices = [(_clean(c) or "") for c in choices]
    back = {(_clean(c) or ""): c for c in choices}

    ts = process.extract(cln, cleaned_choices, scorer=fuzz.token_set_ratio, limit=limit, score_cutoff=cutoff)
    pr = process.extract(cln, cleaned_choices, scorer=fuzz.partial_ratio,   limit=limit, score_cutoff=cutoff)

    best: Dict[str, int] = {}
    for ch, sc, _ in ts + pr:
        orig = back.get(ch)
        if orig:
            best[orig] = max(best.get(orig, 0), int(sc))
    return list(best.items())


def _rank_unique(pairs: List[Tuple[str, int]], top: int = 5) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Tuple[str, int]] = []
    for val, sc in sorted(pairs, key=lambda x: x[1], reverse=True):
        if val not in seen:
            seen.add(val)
            out.append((val, sc))
    return [{"value": v, "score": sc} for v, sc in out[:top]]


# ---------- canonicalizers ----------
def _map_incentive_type(text: Optional[str]) -> Optional[str]:
    t = _clean(text)
    if not t:
        return None
    if re.search(r"\b(pre[\s\-]?sales?|presales?)\b", t):
        return "pre_sales"
    if re.search(r"\b(post[\s\-]?sales?|postsales?)\b", t):
        return "post_sales"
    if "csp" in t:
        return "csp_transaction"
    if t in INCENTIVE_TYPES:
        return t
    return None


def _map_segment(text: Optional[str]) -> Optional[str]:
    t = _clean(text)
    if not t:
        return None
    if re.search(r"\b(ent|enterprise|large)\b", t):
        return "enterprise"
    if re.search(r"\b(smb|sme|smec|mid|midsize|small|medium)\b", t):
        return "smec"
    if t in SEGMENTS:
        return t
    return None


def _resolve_workload(text: str) -> Dict[str, Any]:
    syn_hits = _synonym_hits(text, _WL_SYNS or {})
    fuzzy = _fuzzy_hits(text, _WL_LIST or [])
    cands = _rank_unique(syn_hits + fuzzy, top=5)
    value = cands[0]["value"] if cands and cands[0]["score"] >= 85 else None
    return {"value": value, "candidates": cands}


# ---------- market (country) via LLM + ISO validation ----------
_MARKET_SYSTEM = (
    'Extract exactly one sovereign country (ISO 3166-1 English short name). '
    'If none is present, return {"country": null}. '
    'Output STRICT JSON only: {"country": "<Name or ISO code>" | null}'
)

def _extract_country_with_llm(user_message: str) -> Optional[str]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    msgs = [
        {"role": "system", "content": _MARKET_SYSTEM},
        {"role": "user", "content": user_message or ""}
    ]
    try:
        resp = llm.invoke(msgs).content
        data = json.loads(resp)
        c = data.get("country")
        return c.strip() if isinstance(c, str) and c.strip() else None
    except Exception:
        return None


def _validate_country_iso(value: Optional[str]) -> Optional[str]:
    """
    Accepts ISO name or code; returns canonical English short name (e.g., 'United Kingdom') or None.
    Rejects non-sovereign/regions (e.g., 'England', 'EMEA', 'EU').
    """
    if not value:
        return None
    v = value.strip()

    # pycountry lookup (handles names and common variants/codes)
    try:
        m = pycountry.countries.lookup(v)
        # Return standardized name
        return getattr(m, "name", None)
    except (LookupError, KeyError, AttributeError):
        pass

    # Explicit code checks
    if re.fullmatch(r"[A-Za-z]{2}", v):
        m = pycountry.countries.get(alpha_2=v.upper())
        return m.name if m else None
    if re.fullmatch(r"[A-Za-z]{3}", v):
        m = pycountry.countries.get(alpha_3=v.upper())
        return m.name if m else None

    return None


# ---------- public API ----------
def resolve_field_from_message(field_name: str, user_message: str) -> Dict[str, Any]:
    """
    Resolve a SINGLE field from a fresh user message.

    Args:
      field_name: one of {"workload","incentive_type","segment","market"}
      user_message: raw user text

    Returns:
      {
        "field_name": "<input field>",
        "value": "<canonical or None>",
        "candidates": [ { "value": str, "score": int }, ... ]  # empty for market
      }
    """
    f = (field_name or "").strip().lower()
    msg = user_message or ""

    if f == "workload":
        res = _resolve_workload(msg)
        return {"field_name": field_name, "value": res["value"], "candidates": res["candidates"]}

    if f == "incentive_type":
        v = _map_incentive_type(msg)
        # Provide static candidates if unresolved, helpful for UI
        cands = (
            [{"value": v, "score": 100}]
            if v
            else [{"value": x, "score": 92} for x in INCENTIVE_TYPES]
        )
        return {"field_name": field_name, "value": v, "candidates": cands}

    if f == "segment":
        v = _map_segment(msg)
        cands = (
            [{"value": v, "score": 100}]
            if v
            else [{"value": x, "score": 92} for x in SEGMENTS]
        )
        return {"field_name": field_name, "value": v, "candidates": cands}

    if f == "market":
        # LLM extraction → strict ISO validation
        candidate = _extract_country_with_llm(msg)
        valid = _validate_country_iso(candidate)
        return {"field_name": field_name, "value": valid, "candidates": []}

    # Unknown field → None
    return {"field_name": field_name, "value": None, "candidates": []}
