"""Find the passages of ONE contract that answer a question.

Contract review always asks "what does THIS contract say", so search is scoped to a contract (and to a chunking
strategy, so strategies can be compared). Phase 2 techniques are switches; the defaults reproduce Phase 1.

  hybrid   dense (embedding) search + keyword search, merged with Reciprocal Rank Fusion
  rerank   re-order the top `candidates` with a cross-encoder ("bge")

Keyword search is Postgres full-text ranking (ts_rank_cd) over the question's rarer words, not true BM25.
"""
from functools import lru_cache

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.config import settings
from app.db import get_conn
from app.embeddings import get_embeddings

COLUMNS = "c.id, c.chunk_index, c.char_start, c.char_end, c.heading, c.content"

DENSE_SQL = f"""
SELECT {COLUMNS}, 1 - (c.embedding <=> %(q)s) AS score
FROM chunks c
WHERE c.contract_id = %(contract_id)s AND c.chunker = %(chunker)s
ORDER BY c.embedding <=> %(q)s
LIMIT %(n)s
"""

KEYWORD_SQL = f"""
SELECT {COLUMNS}, ts_rank_cd(c.tsv, q.query) AS score
FROM chunks c, (SELECT to_tsquery('simple', %(tsq)s) AS query) q
WHERE c.contract_id = %(contract_id)s AND c.chunker = %(chunker)s AND c.tsv @@ q.query
ORDER BY score DESC
LIMIT %(n)s
"""

STEM_SQL = "SELECT unnest(tsvector_to_array(to_tsvector('english', %s)))"
MAX_COMMON = 0.10  # a term in more than 10% of chunks matches everything and ranks nothing
MAX_TERMS = 25
RRF_K = 60


def _to_documents(rows, contract_id: int) -> list[Document]:
    return [
        Document(
            page_content=content,
            metadata={"chunk_id": cid, "chunk_index": idx, "char_start": s, "char_end": e,
                      "heading": heading, "score": float(score), "contract_id": contract_id},
        )
        for cid, idx, s, e, heading, content, score in rows
    ]


def search_contract(conn, query_vec, contract_id: int, chunker: str, k: int) -> list[Document]:
    """Exact nearest-neighbour search inside one contract (about 50-100 chunks, so no index is needed;
    an HNSW index would also post-filter by contract_id and could return too few rows)."""
    with conn.transaction():
        conn.execute("SELECT set_config('enable_indexscan', 'off', true)")
        rows = conn.execute(
            DENSE_SQL, {"q": query_vec, "contract_id": contract_id, "chunker": chunker, "n": k}
        ).fetchall()
    return _to_documents(rows, contract_id)


@lru_cache(maxsize=4)
def _doc_freq(chunker: str) -> tuple[dict[str, int], int]:
    """lexeme -> number of chunks containing it, for one chunker. Computed once per process."""
    with get_conn() as conn:
        n = conn.execute("SELECT count(*) FROM chunks WHERE chunker = %s", (chunker,)).fetchone()[0]
        inner = "SELECT tsv FROM chunks WHERE chunker = '%s'" % chunker.replace("'", "''")
        rows = conn.execute("SELECT word, ndoc FROM ts_stat(%s)", (inner,)).fetchall()
    return {w: nd for w, nd in rows}, n


def keyword_query(conn, chunker: str, text: str) -> str | None:
    """OR-query over the text's rarest stemmed terms, for to_tsquery('simple', ...). Terms are quoted, so safe."""
    df, n = _doc_freq(chunker)
    stems = [r[0] for r in conn.execute(STEM_SQL, (text,)).fetchall()]
    kept = sorted({s for s in stems if df.get(s, 0) and df[s] <= MAX_COMMON * n}, key=lambda s: df[s])[:MAX_TERMS]
    return " | ".join("'" + s.replace("'", "''") + "'" for s in kept) or None


def keyword_search(conn, tsq: str, contract_id: int, chunker: str, k: int) -> list[Document]:
    rows = conn.execute(
        KEYWORD_SQL, {"tsq": tsq, "contract_id": contract_id, "chunker": chunker, "n": k}
    ).fetchall()
    return _to_documents(rows, contract_id)


def rrf_fuse(*rankings: list[Document], k: int = RRF_K) -> list[Document]:
    """Reciprocal Rank Fusion: score = sum over lists of 1 / (k + rank). Needs no score calibration."""
    scores: dict[int, float] = {}
    docs: dict[int, Document] = {}
    for ranking in rankings:
        for rank, d in enumerate(ranking, start=1):
            cid = d.metadata["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            docs.setdefault(cid, d)
    fused = []
    for cid in sorted(scores, key=lambda c: -scores[c]):
        d = docs[cid].model_copy(deep=True)
        d.metadata["score"] = scores[cid]
        fused.append(d)
    return fused


def retrieve(conn, contract_id: int, chunker: str, query_vec, k: int, *, tsq: str | None = None,
             hybrid: bool = False, rerank: str = "none", query: str = "", candidates: int = 20) -> list[Document]:
    """Dense (+ keyword) candidates, optional rerank, then the top k."""
    pool = max(candidates, k) if (hybrid or rerank != "none") else k
    docs = search_contract(conn, query_vec, contract_id, chunker, pool)
    if hybrid and tsq:
        docs = rrf_fuse(docs, keyword_search(conn, tsq, contract_id, chunker, pool))
    if rerank != "none":
        from app.rerank import get_reranker

        docs = get_reranker(rerank).rerank(query, docs[:candidates])
    return docs[:k]


def contract_id_for(conn, doc_name: str) -> int | None:
    row = conn.execute("SELECT id FROM contracts WHERE doc_name = %s", (doc_name,)).fetchone()
    return row[0] if row else None


class ContractRetriever(BaseRetriever):
    doc_name: str
    k: int = settings.top_k
    chunker: str = "clause"
    hybrid: bool = False
    rerank: str = "none"
    candidates: int = 20

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        q = np.array(get_embeddings().embed_query(query), dtype=np.float32)
        with get_conn() as conn:
            cid = contract_id_for(conn, self.doc_name)
            if cid is None:
                return []
            tsq = keyword_query(conn, self.chunker, query) if self.hybrid else None
            docs = retrieve(conn, cid, self.chunker, q, self.k, tsq=tsq, hybrid=self.hybrid,
                            rerank=self.rerank, query=query, candidates=self.candidates)
        for d in docs:
            d.metadata["doc_name"] = self.doc_name
        return docs
