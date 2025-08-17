# app/agents/field_validator_v1.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import re, os, json
from rapidfuzz import process, fuzz

# If you already have these elsewhere, keep imports and delete the inline defs.
from app.db import qall  # SELECT helper that returns list[dict]

# ---------- config / paths ----------
# This file is inside app/agents/, JSONs live in app/
_APP_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_ENG_SYNS_PATH = os.environ.get("ENGAGEMENT_NAME_SYNS_PATH", os.path.join(_APP_DIR, "engagement_name_synonyms.json"))
_WL_SYNS_PATH  = os.environ.get("WORKLOAD_SYNS_PATH", os.path.join(_APP_DIR, "workload_synonyms.json"))
_WL_LIST_PATH  = os.environ.get("WORKLOADS_PATH",      os.path.join(_APP_DIR, "workloads.json"))

# ---------- canonical options ----------
INCENTIVE_TYPES = ["pre_sales", "post_sales", "csp_transaction"]
SEGMENTS = ["enterprise", "smec"]

# ---------- small cleaners / mappers ----------
def _clean(s: Optional[str]) -> Optional[str]:
    if not isinstance(s, str):
        return None
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    # normalize dashes/quotes
    s = s.replace("–", "-").replace("—", "-").replace("’", "'").replace("“", '"').replace("”", '"')
    return s

def _tokens(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", _clean(s) or "")

def _map_incentive_type(text: Optional[str]) -> Optional[str]:
    t = _clean(text)
    if not t:
        return None
    if re.search(r"\b(pre[\s\-]?sales?|presales?)\b", t): return "pre_sales"
    if re.search(r"\b(post[\s\-]?sales?|postsales?)\b", t): return "post_sales"
    if "csp" in t: return "csp_transaction"
    if t in INCENTIVE_TYPES: return t
    return None

def _map_segment(text: Optional[str]) -> Optional[str]:
    t = _clean(text)
    if not t:
        return None
    if re.search(r"\b(ent|enterprise|large)\b", t): return "enterprise"
    if re.search(r"\b(smb|sme|smec|mid|midsize|small|medium)\b", t): return "smec"
    if t in SEGMENTS: return t
    return None

# ---------- load JSON synonyms / lists ----------
_ENG_SYNS: Dict[str, List[str]] | None = None
_WL_SYNS:  Dict[str, List[str]] | None = None
_WL_LIST:  List[str] | None = None

def _safe_load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _load_syns_and_lists():
    global _ENG_SYNS, _WL_SYNS, _WL_LIST
    if _ENG_SYNS is None:
        _ENG_SYNS = _safe_load_json(_ENG_SYNS_PATH, {})
    if _WL_SYNS is None:
        _WL_SYNS = _safe_load_json(_WL_SYNS_PATH, {})
    if _WL_LIST is None:
        _WL_LIST = _safe_load_json(_WL_LIST_PATH, [])
    return _ENG_SYNS, _WL_SYNS, _WL_LIST

# ---------- catalog (names/workloads from DB) ----------
_catalog: Dict[str, List[str]] = {"names": [], "workloads": []}

def _load_catalog() -> Dict[str, List[str]]:
    """
    Fetch distinct canonical values from incentives table (cached)
    + extend workloads with static list file if provided.
    """
    _load_syns_and_lists()

    if not _catalog["names"]:
        rows = qall("SELECT DISTINCT name FROM incentives WHERE name IS NOT NULL ORDER BY 1;")
        _catalog["names"] = [r["name"] for r in rows if r.get("name")]

    if not _catalog["workloads"]:
        rows = qall("SELECT DISTINCT workload FROM incentives WHERE workload IS NOT NULL ORDER BY 1;")
        db_vals = [r["workload"] for r in rows if r.get("workload")]
        # merge with static list (keeps DB as source of truth but enriches)
        merged = set(db_vals)
        for w in (_WL_LIST or []):
            if isinstance(w, str) and w.strip():
                merged.add(w.strip())
        _catalog["workloads"] = sorted(merged)

    return _catalog

# ---------- matching helpers (synonyms + partial + fuzzy) ----------
def _synonym_candidates(msg: str, mapping: Dict[str, List[str]],
                        exact_score: int = 100, contains_score: int = 96, token_subset_score: int = 93
                        ) -> List[Tuple[str, int]]:
    """
    Try synonyms first:
    - exact clean match => exact_score
    - message contains synonym (or vice versa) => contains_score
    - all tokens of synonym are subset of msg tokens (or vice versa) => token_subset_score
    Returns list of (canonical, score)
    """
    if not msg or not mapping:
        return []
    m_clean = _clean(msg) or ""
    m_tokens = set(_tokens(msg))
    out: List[Tuple[str, int]] = []

    for canon, alts in mapping.items():
        c_clean = _clean(canon) or ""
        # canonical full exact
        if m_clean == c_clean:
            out.append((canon, exact_score))
            continue

        for a in alts or []:
            a_clean = _clean(a) or ""
            if not a_clean:
                continue
            if m_clean == a_clean:
                out.append((canon, exact_score))
                break
            # contains either direction to handle prefixes/suffixes
            if a_clean in m_clean or m_clean in a_clean:
                out.append((canon, contains_score))
                break
            # token subset check (robust partial)
            a_tokens = set(_tokens(a))
            if a_tokens and (a_tokens.issubset(m_tokens) or m_tokens.issubset(a_tokens)):
                out.append((canon, token_subset_score))
                break
    return out

def _fuzzy_candidates(msg: str, choices: List[str],
                      limit: int = 8,
                      token_set_accept: int = 88,
                      partial_accept: int = 86,
                      cutoff: int = 70) -> List[Tuple[str, int]]:
    """
    Use two RapidFuzz scorers and take the better score:
    - token_set_ratio (handles reordering)
    - partial_ratio (handles substring/partial matches)
    """
    if not msg or not choices:
        return []
    cln = _clean(msg) or ""

    # token_set
    ts = process.extract(
        cln, [(_clean(c) or "") for c in choices],
        scorer=fuzz.token_set_ratio, limit=limit, score_cutoff=cutoff
    )
    # partial
    pr = process.extract(
        cln, [(_clean(c) or "") for c in choices],
        scorer=fuzz.partial_ratio, limit=limit, score_cutoff=cutoff
    )

    # Build maps: cleaned -> original
    cleaned_to_orig = {(_clean(c) or ""): c for c in choices}

    # keep max score per original
    best: Dict[str, int] = {}
    for ch, score, _ in ts + pr:
        orig = cleaned_to_orig.get(ch)
        if not orig:
            continue
        best[orig] = max(best.get(orig, 0), int(score))

    # Accept candidates with decent scores; let ranker sort
    out = [(orig, sc) for orig, sc in best.items()]
    return out

def _rank_and_unique(pairs: List[Tuple[str, int]], top: int = 5) -> List[Dict[str, Any]]:
    seen = set()
    dedup: List[Tuple[str, int]] = []
    for val, score in sorted(pairs, key=lambda x: x[1], reverse=True):
        if val not in seen:
            seen.add(val)
            dedup.append((val, score))
    return [{"value": v, "score": s} for v, s in dedup[:top]]

def _resolve_with_syns_and_fuzzy(msg: str,
                                 synonyms: Dict[str, List[str]],
                                 canonical_choices: List[str],
                                 accept_if_score_ge: int) -> Dict[str, Any]:
    """
    Combined resolver:
      1) synonyms (exact / contains / token-subset)
      2) fuzzy (token_set + partial)
    Returns {value: str|None, candidates: [{value, score}...]}
    """
    syn_hits = _synonym_candidates(msg, synonyms)
    fuzzy_hits = _fuzzy_candidates(msg, canonical_choices)

    all_pairs = syn_hits + fuzzy_hits
    cands = _rank_and_unique(all_pairs, top=5)

    value = None
    if cands and cands[0]["score"] >= accept_if_score_ge:
        value = cands[0]["value"]

    return {"value": value, "candidates": cands}

# ---------- extractors from free text ----------
def _extract_name(msg: str) -> Dict[str, Any]:
    """Try synonyms + partial + fuzzy against engagement names."""
    eng_syns, _, _ = _load_syns_and_lists()
    cat = _load_catalog()
    # Prefer DB names for canonical list (catalog["names"])
    # If DB is empty, synonyms keys serve as a fallback pool.
    choices = cat["names"] if cat["names"] else list(eng_syns.keys())
    res = _resolve_with_syns_and_fuzzy(
        msg, eng_syns or {}, choices, accept_if_score_ge=88
    )
    # Ensure 'score' for top when value picked (optional)
    if res["value"] and (not res["candidates"] or res["candidates"][0]["value"] != res["value"]):
        res["candidates"] = [{"value": res["value"], "score": 100}] + res["candidates"]
    return res

def _extract_workload(msg: str) -> Dict[str, Any]:
    """Try synonyms + partial + fuzzy against workloads."""
    _, wl_syns, wl_list = _load_syns_and_lists()
    cat = _load_catalog()
    # canonical choices: union of DB and static file (already merged in _load_catalog)
    choices = cat["workloads"] if cat["workloads"] else (wl_list or [])
    res = _resolve_with_syns_and_fuzzy(
        msg, wl_syns or {}, choices, accept_if_score_ge=85
    )
    if res["value"] and (not res["candidates"] or res["candidates"][0]["value"] != res["value"]):
        res["candidates"] = [{"value": res["value"], "score": 100}] + res["candidates"]
    return res

def _extract_incentive_type(msg: str) -> Optional[str]:
    return _map_incentive_type(msg)

def _extract_segment(msg: str) -> Optional[str]:
    return _map_segment(msg)

def _extract_country(_: str) -> Optional[str]:
    # first message: don't guess; UI/LLM must ask explicitly
    return None

# ---------- parse required_fields expressions ----------
def _parse_branch(expr: str) -> List[List[str]]:
    expr = expr.strip()
    if "|" in expr:
        alts = [a.strip() for a in expr.split("|")]
        out: List[List[str]] = []
        for a in alts:
            a = a.strip()
            if a.startswith("(") and a.endswith(")"):
                out.append([x.strip() for x in a[1:-1].split(",") if x.strip()])
            elif "," in a:  # split comma branch without parens
                out.append([x.strip() for x in a.split(",") if x.strip()])
            else:
                out.append([a])
        return out
    # single branch (no '|')
    if expr.startswith("(") and expr.endswith(")"):
        return [[x.strip() for x in expr[1:-1].split(",") if x.strip()]]
    if "," in expr:
        return [[x.strip() for x in expr.split(",") if x.strip()]]
    return [[expr]]

# ---------- util ----------
def _first_missing_or_ambiguous(order: List[str],
                                bag: Dict[str, Optional[List[str]]]) -> List[str]:
    """Return fields that are None OR lists with len != 1, in order."""
    bad: List[str] = []
    for f in order:
        v = bag.get(f)
        if v is None or (isinstance(v, list) and len(v) != 1):
            bad.append(f)
    return bad

# ---------- MAIN ----------
def field_validator_v1(user_message: str, required_fields: List[str]) -> Dict[str, Any]:
    """
    Input:
      user_message: raw text
      required_fields: e.g. ["name | (workload,incentive_type)", "country", "segment"]

    Output (NO question text here):
      {
        "picked_set": ["name"] | ["workload","incentive_type"],
        "required_fields_object": { field: None | [values...] },
        "missing_or_ambiguous": [field, ...],
        "complete": bool,
        "candidates": { "name":[{value,score}], "workload":[{value,score}] }
      }
    """
    msg = user_message or ""

    # 1) extract signals (now synonym + partial + fuzzy aware)
    name_res = _extract_name(msg)             # {'value'|None,'candidates':[...] }
    workload_res = _extract_workload(msg)     # {'value'|None,'candidates':[...] }
    inc_type = _extract_incentive_type(msg)   # canonical or None
    segment = _extract_segment(msg)           # canonical or None
    country = _extract_country(msg)           # None in v1

    name_val = [name_res["value"]] if name_res.get("value") else None
    workload_val = [workload_res["value"]] if workload_res.get("value") else None

    # 2) parse required expressions
    if not required_fields:
        return {
            "picked_set": [],
            "required_fields_object": {},
            "missing_or_ambiguous": [],
            "complete": True,
            "candidates": {"name": [], "workload": []}
        }

    first_expr = required_fields[0]
    branches = _parse_branch(first_expr)      # e.g. [["name"], ["workload","incentive_type"]]
    trailing = [e.strip() for e in required_fields[1:] if e.strip()]  # e.g. ["country","segment"]

    # 3) choose branch deterministically (UPDATED default)
    has_name_branch = any(b == ["name"] for b in branches)
    has_wk_inc_branch = any(set(b) == {"workload", "incentive_type"} and len(b) == 2 for b in branches)

    if has_name_branch and has_wk_inc_branch:
        if name_val:
            picked = ["name"]
        elif (workload_val and inc_type):
            picked = ["workload", "incentive_type"]
        else:
            # NEW RULE: when no decisive field is detected, default to workload+incentive_type
            picked = ["workload", "incentive_type"]
    else:
        # Original behavior when special pair not present
        if has_name_branch and name_val:
            picked = ["name"]
        elif has_wk_inc_branch and (workload_val and inc_type):
            picked = ["workload", "incentive_type"]
        else:
            picked = list(branches[0])

    # 4) build required_fields_object (list-or-None discipline)
    rfo: Dict[str, Optional[List[str]]] = {}
    if "name" in picked:
        rfo["name"] = name_val
    if "workload" in picked:
        rfo["workload"] = workload_val
    if "incentive_type" in picked:
        rfo["incentive_type"] = [inc_type] if inc_type else None

    for f in trailing:
        if f == "country":
            rfo["country"] = [country] if country else None
        elif f == "segment":
            rfo["segment"] = [segment] if segment else None
        else:
            rfo[f] = None

    # 5) completion + candidates
    order = picked + trailing
    missing_or_ambiguous = _first_missing_or_ambiguous(order, rfo)
    complete = len(missing_or_ambiguous) == 0

    candidates = {
        "name": [] if rfo.get("name") else name_res.get("candidates", []),
        "workload": [] if rfo.get("workload") else workload_res.get("candidates", []),
    }

    return {
        "picked_set": picked,
        "required_fields_object": rfo,
        "missing_or_ambiguous": missing_or_ambiguous,
        "complete": complete,
        "candidates": candidates
    }
