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
INCENTIVE_TYPES = ["pre_sales", "csp_transaction"]
SEGMENTS = ["enterprise", "smec"]

_INC_TYPE_SYNS = {
    "pre_sales": [
        "funded", "funded engagement", "funded program", "funded offer",
        "workshop", "immersion", "immersion workshop", "briefing",
        "envisioning", "assessment", "readiness assessment",
        "discovery", "scoping session", "poc", "proof of concept", "pilot"
    ],
    "csp_transaction": [
        "csp", "cloud solution provider", "csp transaction",
        "csp billed", "csp revenue", "csp sale"
    ],
}

# ---------- numeric value extractors (simple value only) ----------
_SUFFIX_MULT = {
    "k": 1_000,
    "m": 1_000_000,
    "b": 1_000_000_000,
    "bn": 1_000_000_000,
    "l": 100_000,
    "lac": 100_000,
    "lakh": 100_000,
    "cr": 10_000_000,
    "crore": 10_000_000,
}

def _normalize_number_str(x: float) -> str:
    # stringify without scientific notation; drop trailing .0
    if int(x) == x:
        return str(int(x))
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"
def _extract_acv_value(msg: str) -> Optional[str]:
    """
    Parse ACV from free text.
    - Accepts plain numbers, 1,20,000 style, decimals, and k/m/b/lac/lakh/cr/crore suffixes.
    - Ignores currency (always treat as USD downstream).
    - If multiple numbers: prefer one near ACV-ish keywords else pick the largest.
    - Returns normalized numeric string (e.g., "120000") or None.
    """
    if not msg:
        return None
    text = msg

    rx = re.compile(
        r"(?i)(?:[$₹€£]\s*)?"
        r"(?P<num>\d{1,3}(?:[,\s]?\d{2,3})+|\d+(?:\.\d+)?)"
        r"\s*(?P<suf>k|m|bn|b|l|lac|lakh|cr|crore)?"
        r"(?:\s*(usd|inr|eur|gbp))?"
    )

    hits: List[Tuple[float, int]] = []
    for m in rx.finditer(text):
        raw = m.group("num") or ""
        suf = (m.group("suf") or "").lower()
        n = re.sub(r"[,\s]", "", raw)
        try:
            val = float(n)
        except Exception:
            continue
        if suf in _SUFFIX_MULT:
            val *= _SUFFIX_MULT[suf]
        hits.append((val, m.start()))

    if not hits:
        return None

    ctx_rx = re.compile(r"(?i)\b(acv|annual|contract|deal|oppty|opportunity|value|revenue)\b")
    ctx_pos = [m.start() for m in ctx_rx.finditer(text)]
    if ctx_pos:
        def dist(p: Tuple[float, int]) -> int:
            _, pos = p
            return min(abs(pos - cp) for cp in ctx_pos)
        hits.sort(key=lambda p: (dist(p), -p[0]))
        chosen = hits[0][0]
    else:
        chosen = max(hits, key=lambda p: p[0])[0]

    return _normalize_number_str(chosen)

# Accepts: 10h, 8 hr, 7.5 hours, and also bare "10"
_HOURS_RX = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours)\b")

def _extract_hours_value(msg: str) -> Optional[str]:
    if not msg:
        return None
    last = None
    for m in _HOURS_RX.finditer(msg):
        last = m.group(1)
    if last is not None:
        f = float(last)
        return str(int(f)) if abs(f - round(f)) < 1e-9 else str(f)
    m2 = re.search(r"(?<!\d)(\d{1,4}(?:\.\d+)?)(?!\d)", msg)
    if not m2:
        return None
    f = float(m2.group(1))
    return str(int(f)) if abs(f - round(f)) < 1e-9 else str(f)


# ---------- small cleaners / mappers ----------
def _clean(s: Optional[str]) -> Optional[str]:
    if not isinstance(s, str):
        return None
    s = s.strip().lower()
    # normalize common variants
    s = s.replace("&", " and ")
    s = re.sub(r"\s+", " ", s)
    s = s.replace("–", "-").replace("—", "-").replace("’", "'").replace("“", '"').replace("”", '"')
    return s


def _tokens(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", _clean(s) or "")

def _map_incentive_type(text: Optional[str]) -> Optional[str]:
    """
    Detect incentive_type as either 'pre_sales' or 'csp_transaction'.
    """
    t = _clean(text)
    if not t:
        return None

    # 1) synonym/contains/subset check
    syn_hits = _synonym_candidates(t, _INC_TYPE_SYNS,
                                   exact_score=100, contains_score=98, token_subset_score=95)
    if syn_hits:
        syn_hits.sort(key=lambda x: x[1], reverse=True)
        top_val, top_score = syn_hits[0]
        if top_score >= 95:
            return top_val

    # 2) regex fallback
    if re.search(r"\b(pre[\s\-]?sales?|presales?)\b", t):
        return "pre_sales"
    if "csp" in t:
        return "csp_transaction"

    # 3) canonical direct
    if t in INCENTIVE_TYPES:
        return t

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
                      cutoff: int = 60) -> List[Tuple[str, int]]:
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
    eng_syns, _, _ = _load_syns_and_lists()
    cat = _load_catalog()

    # Union of DB canonicals and synonym keys to ensure coverage
    choices = list(cat["names"] or [])
    if eng_syns:
        for canon in eng_syns.keys():
            if canon not in choices:
                choices.append(canon)

    # Resolve with synonyms + fuzzy. Loosen accept a bit for names.
    res = _resolve_with_syns_and_fuzzy(
        msg, eng_syns or {}, choices, accept_if_score_ge=80
    )

    # Ensure top candidate present if value chosen
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
    name_res = _extract_name(msg)    
    workload_res = _extract_workload(msg)     
    inc_type = _extract_incentive_type(msg)  
    segment = _extract_segment(msg)           
    country = _extract_country(msg)         
    acv_val = _extract_acv_value(msg)      
    hours_val = _extract_hours_value(msg)

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
    if "acv" in picked:
        rfo["acv"] = [acv_val] if acv_val is not None else None
    if "hours" in picked:
        rfo["hours"] = [hours_val] if hours_val is not None else None

    for f in trailing:
        if f == "country":
            rfo["country"] = [country] if country else None
        elif f == "segment":
            rfo["segment"] = [segment] if segment else None
        elif f == "acv":
            rfo["acv"] = [acv_val] if acv_val is not None else None
        elif f == "hours":
            rfo["hours"] = [hours_val] if hours_val is not None else None
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
