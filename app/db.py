from psycopg_pool import ConnectionPool
from .config import DATABASE_URL

pool = ConnectionPool(DATABASE_URL, max_size=10, open=True) if DATABASE_URL else None

def qall(sql: str, params=()):
    if not pool: return []
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params); cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

def qone(sql: str, params=()):
    rows = qall(sql, params)
    return rows[0] if rows else None
