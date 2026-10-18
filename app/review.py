"""Review one contract: run all 40 clause checks and return what was found, with the exact character range of each
quote so a page can highlight it. Same retrieval and extraction as the evaluation (evals/run_extraction.py).

Results are cached in Postgres per (contract, prompt version), so a contract costs about one cent the first time and
nothing after that. That also bounds the cost of a public demo: only ingested contracts can be reviewed.
"""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np

from app.clauses import load_clauses, query_text
from app.config import settings
from app.db import get_conn
from app.embeddings import get_embeddings
from app.extract import PROMPT_VERSION, build_chain, extract_clause, passage_ranges
from app.retrieval import keyword_query, retrieve

CHUNKER = "clause"
PRICE_IN_PER_M = 0.15  # USD per million input tokens (gpt-4o-mini list price when written; check current pricing)
PRICE_OUT_PER_M = 0.60

CACHE_SQL = """
CREATE TABLE IF NOT EXISTS reviews (
    doc_name       TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    result         JSONB NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (doc_name, prompt_version)
)
"""


@lru_cache(maxsize=1)
def _query_vectors() -> dict[str, tuple[str, np.ndarray]]:
    """Search text and its embedding for each clause type. Computed once per process (40 embeddings)."""
    clauses = load_clauses()
    texts = {c: query_text(c, info) for c, info in clauses.items()}
    vecs = get_embeddings().embed_documents(list(texts.values()))
    return {c: (texts[c], np.array(v, dtype=np.float32)) for c, v in zip(texts, vecs)}


def get_cached(doc_name: str) -> dict | None:
    with get_conn() as conn:
        conn.execute(CACHE_SQL)
        row = conn.execute(
            "SELECT result FROM reviews WHERE doc_name = %s AND prompt_version = %s", (doc_name, PROMPT_VERSION)
        ).fetchone()
    return row[0] if row else None


def review_contract(doc_name: str, workers: int = 8, use_cache: bool = True) -> dict:
    """Raises KeyError for an unknown contract. Returns {"clauses": [...], "found", "seconds", "cost_usd", ...}."""
    if use_cache and (cached := get_cached(doc_name)):
        return {**cached, "cached": True}

    clauses = load_clauses()
    qv = _query_vectors()
    with get_conn() as conn:
        row = conn.execute("SELECT id, text FROM contracts WHERE doc_name = %s", (doc_name,)).fetchone()
        if not row:
            raise KeyError(doc_name)
        cid, text = row
        tsq = {c: keyword_query(conn, CHUNKER, qv[c][0]) for c in clauses}

    chain = build_chain()
    local = threading.local()
    started = time.time()

    def check(cat: str) -> dict:
        if not hasattr(local, "conn"):
            local.conn = get_conn()
        qtext, vec = qv[cat]
        docs = retrieve(local.conn, cid, CHUNKER, vec, settings.top_k, tsq=tsq[cat], hybrid=True, query=qtext)
        ranges = passage_ranges(docs, text, cat, head=True)
        res = extract_clause(chain, text, cat, clauses[cat], ranges)
        return {"category": cat, **res}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(check, clauses))

    tin = sum(r.pop("input_tokens", 0) for r in results)
    tout = sum(r.pop("output_tokens", 0) for r in results)
    out = {
        "doc_name": doc_name,
        "prompt_version": PROMPT_VERSION,
        "clauses": [
            {k: r.get(k) for k in ("category", "present", "quote", "start", "end", "grounded", "match")}
            for r in results
        ],
        "found": sum(1 for r in results if r["present"]),
        "total": len(results),
        "seconds": round(time.time() - started, 1),
        "cost_usd": round(tin / 1e6 * PRICE_IN_PER_M + tout / 1e6 * PRICE_OUT_PER_M, 4),
        "cached": False,
    }
    with get_conn() as conn:
        conn.execute(CACHE_SQL)
        conn.execute(
            "INSERT INTO reviews (doc_name, prompt_version, result) VALUES (%s, %s, %s) "
            "ON CONFLICT (doc_name, prompt_version) DO UPDATE SET result = EXCLUDED.result, created_at = now()",
            (doc_name, PROMPT_VERSION, json.dumps(out)),
        )
    return out
