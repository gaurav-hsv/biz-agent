from typing import Dict, Any, List, Optional
import re

def norm(s: Optional[str]) -> Optional[str]:
    return s.strip().lower() if isinstance(s, str) else None

def extract_market(t: str) -> Optional[str]:
    tl = t.lower()
    if re.search(r"\bmarket\s*[abc]\b", tl):  # "market a"
        return tl.split("market")[-1].strip()[:1].upper()
    if any(w in tl for w in ["us","uk","france","germany","europe","canada","australia","japan","sweden","switzerland","new zealand","ireland","netherlands","denmark","norway","finland","luxembourg","iceland"]):
        return "A"
    if any(w in tl for w in ["uae","saudi","qatar","bahrain","singapore","hong kong","south africa","mexico","brazil","china","chile","poland","czech","israel","korea","taiwan","malaysia","thailand","indonesia","colombia","philippines","portugal","greece","turkey","oman","kuwait","latvia","lithuania","estonia","slovakia","slovenia","uruguay","jamaica","puerto rico"]):
        return "B"
    if any(w in tl for w in ["other","rest of world","row","others","anywhere"]):
        return "C"
    return None

def extract_segment(t: str) -> Optional[str]:
    tl = t.lower()
    if any(k in tl for k in ["enterprise","ent","large"]): return "enterprise"
    if any(k in tl for k in ["smb","sme","smec","mid","small","midsize"]): return "smec"
    return None

def extract_cpor(t: str) -> Optional[bool]:
    tl = t.lower()
    if re.search(r"\b(cpor|claiming partner)\b", tl):
        if any(k in tl for k in ["yes","have","available","present","true"]): return True
        if any(k in tl for k in ["no","not","absent","false"]): return False
    return None


def extract_workload(t: str) -> Optional[str]:
    tl = t.lower()
    if any(k in tl for k in ["crm","customer engagement","sales","service","field service","contact center"]):
        return "D365 Customer Engagement"
    if any(k in tl for k in ["finance","supply chain","f&scm","scm","fno","f&o"]):
        return "D365 Finance & Supply Chain"
    if any(k in tl for k in ["business central","bc"]):
        return "Business Central"
    if "csp" in tl: return None
    if "d365" in tl or "dynamics" in tl: return "D365"
    return None


def extract_incentive_type(t: str) -> Optional[str]:
    tl = t.lower()
    if "pre" in tl and "sale" in tl: return "pre_sales"
    if "post" in tl and "sale" in tl: return "post_sales"
    if "csp" in tl: return "csp_transaction"
    return None