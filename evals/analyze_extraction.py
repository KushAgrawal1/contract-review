"""Read the failures of an extraction run yourself. Numbers hide whether the system or the labels are wrong.

  python -m evals.analyze_extraction p3_100                 # 4 examples of each failure type
  python -m evals.analyze_extraction p3_100 --show 8
  python -m evals.analyze_extraction p3_100 --category Insurance

  ungrounded      quote is not in the contract word for word (paraphrase, trimmed, or made up)
  false_positive  the LLM says present, the experts marked nothing (maybe a real clause the experts skipped)
  missed          the experts marked it, the LLM said absent
  wrong_span      detected, but the quote does not overlap the expert span
"""
import argparse
import json
import random
import sys
from collections import Counter

from app.db import get_conn
from app.extract import _squashed
from evals.run_eval import LABELS_PATH, RESULTS_DIR


def diagnose(quote: str, text: str) -> str:
    """Where does a quote that is 'not in the contract' stop matching? Shows the quote vs the contract at that point."""
    squashed, pos = _squashed(text)
    q = "".join(quote.split()).lower()
    lo, hi = 0, len(q)
    while lo < hi:  # longest prefix of the quote that exists in the contract
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if q[:mid] in squashed else (lo, mid - 1)
    if lo == 0:
        return "no part of the quote is in the contract (likely invented or from the wrong place)"
    i = squashed.find(q[:lo])
    end = i + lo
    return (f"first {lo} of {len(q)} characters match; then quote says {q[lo:lo + 40]!r} "
            f"but contract says {squashed[end:end + 40]!r}")


def one_line(text, n):
    return " ".join((text or "").split())[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--show", type=int, default=4)
    parser.add_argument("--category", default=None)
    args = parser.parse_args()

    path = RESULTS_DIR / f"{args.name}.jsonl"
    if not path.exists():
        sys.exit(f"{path} not found. Run evals.run_extraction first.")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.category:
        rows = [r for r in rows if r["category"] == args.category]
    gold = {}
    for line in LABELS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            gold[(r["doc_name"], r["category"])] = r["spans"]

    groups = {
        "ungrounded": [r for r in rows if r["present"] and not r.get("grounded")],
        "false_positive": [r for r in rows if r["present"] and not r["gold_present"]],
        "missed": [r for r in rows if not r["present"] and r["gold_present"]],
        "wrong_span": [r for r in rows if r["gold_present"] and r["present"] and r.get("grounded") and not r.get("span_match")],
    }
    print(f"{len(rows)} checks in '{args.name}'")
    for key, pool in groups.items():
        print(f"{len(pool):5d}  {key}")
    for key, pool in groups.items():
        if not pool:
            continue
        print(f"\n=== {key}: most common clause types: " + ", ".join(f"{c} ({n})" for c, n in Counter(r['category'] for r in pool).most_common(4)))
        for r in random.Random(0).sample(pool, min(args.show, len(pool))):
            print(f"\n  [{r['category']}] {r['doc_name'][:50]}")
            print(f"    LLM quote : {one_line(r.get('quote'), 260) or '(none)'}")
            g = gold.get((r["doc_name"], r["category"]), [])
            if key == "ungrounded" and r.get("quote"):
                with get_conn() as conn:
                    row = conn.execute("SELECT text FROM contracts WHERE doc_name = %s", (r["doc_name"],)).fetchone()
                if row:
                    print(f"    why       : {diagnose(r['quote'], row[0])}")
            if g and key != "ungrounded":
                print(f"    experts   : {one_line(g[0]['text'], 260)}")


if __name__ == "__main__":
    main()
