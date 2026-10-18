"""Load CUAD contracts into Postgres, chunked and embedded.

  python -m app.ingest --chunker both            # fixed and clause-aware chunks (about 60k chunks)
  python -m app.ingest --chunker clause --limit 5   # cheap smoke test

Re-running skips contracts that already have chunks for that chunker. Embeddings are computed before the
transaction, so a failure never leaves a half-loaded contract.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from app.chunking import CHUNKERS
from app.db import get_conn
from app.embeddings import get_embeddings

CONTRACTS_PATH = Path("data/cuad/contracts.jsonl")


def load_contracts(limit: int | None = None) -> list[dict]:
    rows = [json.loads(l) for l in CONTRACTS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:limit] if limit else rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunker", choices=[*CHUNKERS, "both"], default="clause")
    parser.add_argument("--limit", type=int, default=None, help="only the first N contracts")
    args = parser.parse_args()
    if not CONTRACTS_PATH.exists():
        raise SystemExit("Run first: python scripts/prepare_cuad.py")
    chunkers = list(CHUNKERS) if args.chunker == "both" else [args.chunker]

    contracts = load_contracts(args.limit)
    started, added, skipped, failed = time.time(), 0, 0, 0
    with get_conn() as conn:
        for n, c in enumerate(contracts, start=1):
            label = f"[{n}/{len(contracts)}] {c['doc_name'][:60]}"
            try:
                row = conn.execute("SELECT id FROM contracts WHERE doc_name = %s", (c["doc_name"],)).fetchone()
                contract_id = row[0] if row else conn.execute(
                    "INSERT INTO contracts (doc_name, n_chars, text) VALUES (%s, %s, %s) RETURNING id",
                    (c["doc_name"], len(c["text"]), c["text"]),
                ).fetchone()[0]
                for name in chunkers:
                    if conn.execute(
                        "SELECT 1 FROM chunks WHERE contract_id = %s AND chunker = %s LIMIT 1", (contract_id, name)
                    ).fetchone():
                        skipped += 1
                        continue
                    chunks = CHUNKERS[name](c["text"])
                    vectors = get_embeddings().embed_documents([ch.content for ch in chunks])
                    rows = [
                        (contract_id, name, i, ch.start, ch.end, ch.heading, ch.content, np.array(v, dtype=np.float32))
                        for i, (ch, v) in enumerate(zip(chunks, vectors))
                    ]
                    with conn.transaction(), conn.cursor() as cur:
                        cur.executemany(
                            "INSERT INTO chunks (contract_id, chunker, chunk_index, char_start, char_end, heading, "
                            "content, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                            rows,
                        )
                    added += len(rows)
                print(f"{label}: ok ({time.time() - started:.0f}s elapsed)")
            except Exception as exc:  # one bad contract should not stop a long ingest
                failed += 1
                print(f"{label}: FAILED - {type(exc).__name__}: {exc}")
                if conn.closed or conn.broken:
                    print("database connection lost - re-run the same command to resume.")
                    break
    print(f"done in {time.time() - started:.0f}s: {added} chunks added, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
