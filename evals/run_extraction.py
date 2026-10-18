"""Score LLM clause extraction against CUAD's expert labels.

  python -m evals.run_extraction --name p3_dev --limit 20           # tuning set: read its failures, improve the prompt
  python -m evals.run_extraction --name p3_100 --skip 20 --limit 100   # held-out: contracts you never tuned on (about $1)
  python -m evals.run_extraction --name p3_full                     # all 510 contracts, 20,400 clause checks
  python -m evals.run_extraction --name p3_nohead --no-head         # ablation: do not add the contract opening

For every (contract, clause type) the system says present / not present and quotes the clause. Gold = the experts'
marked spans.
  precision / recall / F1   presence detection per clause type (micro over all pairs, and macro over clause types)
  span_match_rate           of the correctly detected clauses, how often the quote overlaps an expert span
                            (overlap >= half of the shorter of the two)
  grounded_rate             how often a quote is found in the contract (exact, or at least 90% of it); the rest are not
  exact_quote_rate          how often the quote is found word for word
  cost                      tokens and dollars per contract, at the prices set below (check current pricing)

Resumable: finished pairs are kept in evals/results/<name>.jsonl, so re-running the same command continues.
"""
import argparse
import json
import statistics
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.config import settings
from app.db import get_conn
from app.embeddings import get_embeddings
from app.extract import PROMPT_VERSION, build_chain, extract_clause, passage_ranges
from app.retrieval import keyword_query, retrieve
from evals.run_eval import CATEGORIES_PATH, EXAMPLES_PATH, LABELS_PATH, RESULTS_DIR, SKIP, query_text

PRICE_IN_PER_M = 0.15  # USD per million input tokens (gpt-4o-mini list price when this was written)
PRICE_OUT_PER_M = 0.60  # USD per million output tokens. Check current pricing before quoting costs.


def overlaps(span: tuple[int, int], gold: list[dict]) -> bool:
    s, e = span
    for g in gold:
        ov = min(e, g["end"]) - max(s, g["start"])
        if ov > 0 and ov >= 0.5 * min(e - s, g["end"] - g["start"]):
            return True
    return False


def summarise(rows: list[dict], n_contracts: int) -> dict:
    pct = lambda a, b: round(100 * a / b, 1) if b else None
    by_cat = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "match": 0})
    for r in rows:
        c = by_cat[r["category"]]
        gold, pred = r["gold_present"], r["present"]
        c["tp" if gold and pred else "fp" if pred else "fn" if gold else "tn"] += 1
        if gold and pred and r.get("span_match"):
            c["match"] += 1

    def prf(tp, fp, fn):
        p, rec = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0
        return p, rec, (2 * p * rec / (p + rec) if p + rec else 0.0)

    tp, fp, fn = (sum(c[k] for c in by_cat.values()) for k in ("tp", "fp", "fn"))
    p, rec, f1 = prf(tp, fp, fn)
    per_cat = {}
    for name, c in sorted(by_cat.items()):
        cp, cr, cf = prf(c["tp"], c["fp"], c["fn"])
        per_cat[name] = {"n_gold": c["tp"] + c["fn"], "precision": round(100 * cp, 1), "recall": round(100 * cr, 1),
                         "f1": round(100 * cf, 1), "span_match_pct": pct(c["match"], c["tp"])}
    present = [r for r in rows if r["present"]]
    tin = sum(r.get("input_tokens", 0) for r in rows)
    tout = sum(r.get("output_tokens", 0) for r in rows)
    cost = tin / 1e6 * PRICE_IN_PER_M + tout / 1e6 * PRICE_OUT_PER_M
    return {
        "pairs_scored": len(rows),
        "contracts": n_contracts,
        "precision_pct": round(100 * p, 1),
        "recall_pct": round(100 * rec, 1),
        "f1_pct": round(100 * f1, 1),
        "macro_f1_pct": round(statistics.mean(v["f1"] for v in per_cat.values() if v["n_gold"]), 1),
        "span_match_rate_pct": pct(sum(c["match"] for c in by_cat.values()), tp),
        "grounded_rate_pct": pct(sum(1 for r in present if r.get("grounded")), len(present)),
        "exact_quote_rate_pct": pct(sum(1 for r in present if r.get("match") == "exact"), len(present)),
        "input_tokens": tin,
        "output_tokens": tout,
        "cost_usd_total": round(cost, 3),
        "cost_usd_per_contract": round(cost / n_contracts, 4) if n_contracts else None,
        "by_category": per_cat,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--limit", type=int, default=None, help="only N contracts (after --skip)")
    parser.add_argument("--skip", type=int, default=0, help="skip the first N contracts (keep them as a tuning set)")
    parser.add_argument("--workers", type=int, default=8, help="parallel LLM calls")
    parser.add_argument("--no-head", action="store_true", help="do not add the contract opening for Parties and dates")
    parser.add_argument("--chunker", default="clause")
    parser.add_argument("--k", type=int, default=settings.top_k)
    args = parser.parse_args()

    if not settings.openai_api_key:
        sys.exit("OPENAI_API_KEY is not set (see .env.example)")
    for p in (LABELS_PATH, CATEGORIES_PATH, EXAMPLES_PATH):
        if not p.exists():
            sys.exit(f"{p} not found. Run: python scripts/prepare_cuad.py and python -m scripts.make_clause_queries")
    categories = json.loads(CATEGORIES_PATH.read_text(encoding="utf-8"))
    examples = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
    cats = [c for c in categories if c not in SKIP]
    gold = defaultdict(list)
    for line in LABELS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            gold[(r["doc_name"], r["category"])] = r["spans"]

    out_path = RESULTS_DIR / f"{args.name}.jsonl"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    done = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["doc_name"], r["category"])] = r

    with get_conn() as conn:
        rows_db = conn.execute("SELECT doc_name, id, text FROM contracts ORDER BY doc_name").fetchall()
        rows_db = rows_db[args.skip:]
        if args.limit:
            rows_db = rows_db[: args.limit]
        if not rows_db:
            sys.exit("No contracts in the database. Run: python -m app.ingest")
        contracts = {n: (cid, text) for n, cid, text in rows_db}
        qtext = {c: query_text(c, categories[c], examples.get(c)) for c in cats}
        qvec = {c: np.array(get_embeddings().embed_query(qtext[c]), dtype=np.float32) for c in cats}
        qtsq = {c: keyword_query(conn, args.chunker, qtext[c]) for c in cats}

    todo = [(n, c) for n in contracts for c in cats if (n, c) not in done]
    print(f"{len(contracts)} contracts x {len(cats)} clause types = {len(contracts) * len(cats)} checks; "
          f"{len(done)} already done, {len(todo)} to run")
    chain = build_chain()
    local = threading.local()
    lock = threading.Lock()
    counter = {"n": 0}
    started = time.time()
    out_file = out_path.open("a", encoding="utf-8")

    def work(item):
        name, cat = item
        if not hasattr(local, "conn"):
            local.conn = get_conn()
        cid, text = contracts[name]
        try:
            docs = retrieve(local.conn, cid, args.chunker, qvec[cat], args.k, tsq=qtsq[cat], hybrid=True,
                            query=qtext[cat])
            ranges = passage_ranges(docs, text, cat, head=not args.no_head)
            res = extract_clause(chain, text, cat, categories[cat], ranges)
        except Exception as exc:  # keep going; failed pairs are retried on the next run
            print(f"\n{name[:40]} / {cat}: {type(exc).__name__}: {str(exc)[:120]}")
            return
        g = gold.get((name, cat), [])
        row = {"doc_name": name, "category": cat, "gold_present": bool(g), **res}
        if g and res["present"] and res["start"] is not None:
            row["span_match"] = overlaps((res["start"], res["end"]), g)
        with lock:
            out_file.write(json.dumps(row) + "\n")
            out_file.flush()
            counter["n"] += 1
            if counter["n"] % 50 == 0 or counter["n"] == len(todo):
                print(f"\r{counter['n']}/{len(todo)} checks, {time.time() - started:.0f}s elapsed", end="", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, todo))
    out_file.close()
    print()

    rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["doc_name"] in contracts]
    if not rows:
        sys.exit("No clause checks succeeded (see the errors above). Fix the cause and re-run the same command.")
    summary = summarise(rows, len(contracts))
    summary["config"] = {
        "run_name": args.name, "chunker": args.chunker, "k": args.k, "head_context": not args.no_head, "skip": args.skip, "prompt_version": PROMPT_VERSION,
        "retrieval": "expand+hybrid", "llm_model": settings.llm_model, "embedding_model": settings.embedding_model,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (RESULTS_DIR / f"{args.name}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("by_category", "config")}, indent=2))
    ranked = sorted(((c, v) for c, v in summary["by_category"].items() if v["n_gold"]), key=lambda kv: kv[1]["f1"])
    print("\nweakest clause types (F1):")
    for cat, v in ranked[:8]:
        print(f"  F1 {v['f1']:5.1f}  P {v['precision']:5.1f}  R {v['recall']:5.1f}  {cat} (gold in {v['n_gold']} contracts)")


if __name__ == "__main__":
    main()
