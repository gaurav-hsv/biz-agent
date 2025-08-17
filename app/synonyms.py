import re
from .db import qall
PUNCT = re.compile(r"[^a-z0-9 ]+")

def clean(s: str|None):
    if not s: return None
    s = PUNCT.sub(" ", s.lower().strip())
    return re.sub(r"\s+", " ", s)

def canon_from_db(kind: str, text: str|None):
    if not text: return None
    t = clean(text)
    rows = qall("SELECT phrase, canonical FROM synonyms WHERE kind=%s", (kind,))
    mapping = { (r["phrase"].lower()): r["canonical"] for r in rows }
    return mapping.get(t, text)

def canon_workload(s: str|None):
    v = clean(s) or ""
    if any(k in v for k in ["d365","dynamics","business applications","business apps","biz apps"]):
        return "Business Applications" if "business" in v or "dynamics" in v else "D365"
    return canon_from_db("workload", s) or s

def canon_incentive_type(s: str|None):
    v = clean(s) or ""
    if "csp" in v: return "csp_transaction"
    if "pre" in v and "sale" in v: return "pre_sales"
    if "post" in v and "sale" in v: return "post_sales"
    return canon_from_db("incentive_type", s) or s

def canon_bool(s: str|None):
    if s is None: return None
    v = clean(s)
    if v in {"yes","y","true"}: return True
    if v in {"no","n","false"}: return False
    return None
