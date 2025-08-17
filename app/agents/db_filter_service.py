# app/agents/db_filter_service.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from app.db import qall  # your existing SELECT helper

# ---- ONLY these fields are allowed to filter ----
ALLOWED_FILTER_FIELDS = {"name", "workload", "incentive_type"}

# Map session fields -> DB columns (restricted to allowed)
FIELD_TO_COLUMN: Dict[str, str] = {
    "name": "name",
    "workload": "workload",
    "incentive_type": "incentive_type",
}

# We return everything with SELECT * (simple + resilient to schema drift)
TABLE_NAME = "incentives"


def _listify(v: Any) -> Optional[List[str]]:
    """Ensure a non-empty list[str]; otherwise None."""
    if v is None:
        return None
    if isinstance(v, list):
        vals = [str(x).strip() for x in v if str(x).strip()]
        return vals if vals else None
    s = str(v).strip()
    return [s] if s else None


def _prepare_filters(required_fields: Dict[str, Any]) -> Tuple[List[str], List[Any], Dict[str, List[str]], List[str]]:
    """
    Build WHERE parts & params.
    - AND across fields
    - OR within a field
    - Case-insensitive exact equality for {name, incentive_type}
    - Case-insensitive SUBSTRING match for {workload}  <-- IMPORTANT
    """
    where_parts: List[str] = []
    params: List[Any] = []
    applied: Dict[str, List[str]] = {}
    skipped: List[str] = []

    for field, col in FIELD_TO_COLUMN.items():
        vals = _listify(required_fields.get(field))
        if not vals:
            continue

        if field == "workload":
            # substring OR across provided values (Postgres)
            # (workload ILIKE %s OR workload ILIKE %s ...)
            patterns = [f"%{v}%" for v in vals]
            clause = "(" + " OR ".join([f"{col} ILIKE %s" for _ in patterns]) + ")"
            where_parts.append(clause)
            params.extend(patterns)
        else:
            # exact, case-insensitive: LOWER(col) IN (LOWER(%s), ...)
            placeholders = ", ".join(["LOWER(%s)"] * len(vals))
            clause = f"LOWER({col}) IN ({placeholders})"
            where_parts.append(clause)
            params.extend(vals)

        applied[field] = vals

    # anything outside allowed is skipped (not used in WHERE)
    for k in required_fields.keys():
        if k not in ALLOWED_FILTER_FIELDS:
            skipped.append(k)

    return where_parts, params, applied, skipped


def _build_sql(where_parts: List[str], order_by: Optional[str], limit: int, offset: int) -> str:
    # order only by safe, guaranteed columns to avoid UndefinedColumn
    safe_order = order_by if order_by in {"name", "workload", "incentive_type"} else "name"
    sql = f"SELECT * FROM {TABLE_NAME}"
    if where_parts:
        sql += " WHERE " + " AND ".join(where_parts)
    sql += f" ORDER BY {safe_order}"
    sql += " LIMIT %s OFFSET %s"
    return sql


@dataclass
class FilterResult:
    rows: List[Dict[str, Any]]
    applied_filters: Dict[str, List[str]]
    skipped_fields: List[str]
    limit: int
    offset: int
    count: int


def filter_incentives(required_fields: Dict[str, Any],
                      *,
                      limit: int = 25,
                      offset: int = 0,
                      order_by: Optional[str] = "name") -> FilterResult:
    """
    Filter incentives by ONLY {name, workload, incentive_type}.
    - workload: case-insensitive SUBSTRING (ILIKE) with OR across multiple values
    - name, incentive_type: case-insensitive exact equality with IN (...)
    - AND across different fields
    - Ignores/marks skipped any other fields (segment, country/market, etc.)
    """
    where_parts, params, applied_filters, skipped_fields = _prepare_filters(required_fields or {})

    sql = _build_sql(where_parts, order_by, limit, offset)
    params = params + [limit, offset]

    # NOTE: adapt this if your qall signature expects *params
    rows = qall(sql, params)

    return FilterResult(
        rows=rows or [],
        applied_filters=applied_filters,
        skipped_fields=skipped_fields,
        limit=limit,
        offset=offset,
        count=len(rows or []),
    )
