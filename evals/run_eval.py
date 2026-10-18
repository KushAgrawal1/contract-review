"""Does retrieval find the clause a lawyer marked? Scored against CUAD's expert labels. No LLM calls, so it is free
apart from 40 query embeddings (one per clause type).

  python -m evals.run_eval --name p1_fixed  --chunker fixed  --k 5
  python -m evals.run_eval --name p1_clause --chunker clause --k 5
  python -m evals.run_eval --name smoke --chunker clause --limit 20

Phase 2 switches (each run is one row in the ablation table):
  --expand           search with an LLM-written example of the clause wording (run scripts/make_clause_queries.py first)
  --hybrid           add keyword search, merged with Reciprocal Rank Fusion
  --rerank bge       re-order the top --candidates with a cross-encoder (pip install -r requirements-rerank.txt)

For every (contract, clause type) that the experts marked, we search that contract with the clause-type question
and ask whether the top-k chunks cover the marked text.

  clause_hit@k   share of (contract, clause) pairs where at least one top-k chunk covers a marked span
  span_recall@k  share of individual marked spans covered
  mrr            mean reciprocal rank of the first covering chunk

A chunk "covers" a span when it contains at least half of the span, or at least 100 characters of it (very long
clauses cannot fit in one chunk). "Document Name" is skipped: it is the title, not a clause.
"""
import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.config import settings
from app.db import get_conn
from app.embeddings import get_embeddings
from app.retrieval import keyword_query, retrieve

LABELS_PATH = Path("data/cuad/labels.jsonl")
EXAMPLES_PATH = Path("data/cuad/clause_examples.json")
CATEGORIES_PATH = Path("data/cuad/categories.json")
RESULTS_DIR = Path("evals/results")
SKIP = {"Document Name"}
MIN_CHARS = 100


def covers(chunk_start: int, chunk_end: int, span: dict) -> bool:
    overlap = min(chunk_end, span["end"]) - max(chunk_start, span["start"])
    return overlap > 0 and (overlap >= 0.5 * (span["end"] - span["start"]) or overlap >= MIN_CHARS)


def query_text(category: str, info: dict, example: str | None = None) -> str:
    base = f"{category}: {info['question']}"
    return f"{base}\nExample wording: {example}" if example else base


def score_pair(docs, spans: list[dict]) -> dict:
    covered = [any(covers(d.metadata["char_start"], d.metadata["char_end"], sp) for d in docs) for sp in spans]
    rank = next(
        (r for r, d in enumerate(docs, 1) if any(covers(d.metadata["char_start"], d.metadata["char_end"], sp) for sp in spans)),
        None,
    )
    return {"hit": rank is not None, "rank": rank, "spans": len(spans), "spans_covered": sum(covered)}


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    total_spans = sum(r["spans"] for r in rows)
    pct = lambda a, b: round(100 * a / b, 1) if b else None
    cats = {c: {"n": len(v), "clause_hit_pct": pct(sum(x["hit"] for x in v), len(v))} for c, v in sorted(by_cat.items())}
    return {
        "pairs_scored": n,
        "clause_hit_pct": pct(sum(r["hit"] for r in rows), n),
        "span_recall_pct": pct(sum(r["spans_covered"] for r in rows), total_spans),
        "mrr": round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 3) if n else None,
        "macro_clause_hit_pct": round(statistics.mean(c["clause_hit_pct"] for c in cats.values()), 1) if cats else None,
        "by_category": cats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--chunker", default="clause")
    parser.add_argument("--k", type=int, default=settings.top_k)
    parser.add_argument("--limit", type=int, default=None, help="only the first N contracts")
    parser.add_argument("--random", action="store_true", help="chance baseline: k random chunks per query")
    parser.add_argument("--expand", action="store_true", help="search with LLM-written example clause wording")
    parser.add_argument("--hybrid", action="store_true", help="dense + keyword search merged with RRF")
    parser.add_argument("--rerank", choices=["none", "bge"], default="none")
    parser.add_argument("--candidates", type=int, default=20, help="candidate pool before rerank / cut to k")
    args = parser.parse_args()

    if not LABELS_PATH.exists():
        sys.exit("Run first: python scripts/prepare_cuad.py")
    categories = json.loads(CATEGORIES_PATH.read_text(encoding="utf-8"))
    examples = {}
    if args.expand:
        if not EXAMPLES_PATH.exists():
            sys.exit("Run first: python -m scripts.make_clause_queries")
        examples = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
    labels = [json.loads(l) for l in LABELS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    started, rows = time.time(), []
    rng = random.Random(0)
    with get_conn() as conn:
        ids = dict(conn.execute("SELECT doc_name, id FROM contracts").fetchall())
        if args.limit:
            keep = set(sorted(ids)[: args.limit])
            ids = {k: v for k, v in ids.items() if k in keep}
        if not ids:
            sys.exit("No contracts in the database. Run: python -m app.ingest")
        n_chunks = conn.execute("SELECT count(*) FROM chunks WHERE chunker = %s", (args.chunker,)).fetchone()[0]

        qvec, qtext, qtsq = {}, {}, {}
        if not args.random:
            for cat in {l["category"] for l in labels} - SKIP:
                qtext[cat] = query_text(cat, categories[cat], examples.get(cat))
                qvec[cat] = np.array(get_embeddings().embed_query(qtext[cat]), dtype=np.float32)
                qtsq[cat] = keyword_query(conn, args.chunker, qtext[cat]) if args.hybrid else None

        todo = [l for l in labels if l["category"] not in SKIP and l["doc_name"] in ids]
        for i, lab in enumerate(todo, start=1):
            if i % 200 == 0 or i == len(todo):
                print(f"\r{i}/{len(todo)} lookups, {time.time() - started:.0f}s elapsed", end="", flush=True)
            cid = ids[lab["doc_name"]]
            if args.random:
                spans = conn.execute(
                    "SELECT char_start, char_end FROM chunks WHERE contract_id = %s AND chunker = %s", (cid, args.chunker)
                ).fetchall()
                picked = rng.sample(spans, min(args.k, len(spans)))
                docs = [SimpleNamespace(metadata={"char_start": a, "char_end": b}) for a, b in picked]
            else:
                cat = lab["category"]
                docs = retrieve(conn, cid, args.chunker, qvec[cat], args.k, tsq=qtsq[cat], hybrid=args.hybrid,
                                rerank=args.rerank, query=qtext[cat], candidates=args.candidates)
            rows.append({"doc_name": lab["doc_name"], "category": lab["category"], **score_pair(docs, lab["spans"])})

    print()
    summary = summarise(rows)
    summary["config"] = {
        "run_name": args.name, "chunker": args.chunker, "k": args.k, "random_baseline": args.random,
        "expand": args.expand, "hybrid": args.hybrid, "rerank": args.rerank, "candidates": args.candidates,
        "embedding_model": settings.embedding_model, "contracts": len(ids), "chunks_in_db_for_chunker": n_chunks,
        "seconds": round(time.time() - started, 1), "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{args.name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (RESULTS_DIR / f"{args.name}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    head = {k: v for k, v in summary.items() if k != "by_category"}
    print(json.dumps(head, indent=2))
    worst = sorted(summary["by_category"].items(), key=lambda kv: kv[1]["clause_hit_pct"])[:8]
    print("\nhardest clause types:")
    for cat, v in worst:
        print(f"  {v['clause_hit_pct']:5.1f}%  {cat} (n={v['n']})")


if __name__ == "__main__":
    main()
