import psycopg
from pgvector.psycopg import register_vector

from app.config import settings


def get_conn() -> psycopg.Connection:
    conn = psycopg.connect(settings.database_url, autocommit=True)
    register_vector(conn)
    return conn


def ensure_pages_table(conn: psycopg.Connection) -> None:
    """Idempotent, so databases created in Phase 1 can be upgraded without recreating the volume."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS pages ("
        " document_id INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,"
        " page INT NOT NULL, text TEXT NOT NULL, PRIMARY KEY (document_id, page))"
    )
