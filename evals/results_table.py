"""Print the retrieval ablation table (Phase 1 and 2) from the summary files in evals/results/, ready to paste into the
README, so the numbers in the README are copied from the runs and never typed by hand.

  python -m evals.results_table
"""

import json

from evals.run_eval import RESULTS_DIR

RUNS = [
    ("p1_random", "random chunks"),
    ("p1_fixed", "fixed 1000-character chunks"),
    ("p1_clause", "clause-aware chunks"),
    ("p2a_hybrid", "+ keyword search (RRF)"),
    ("p2b_expand", "+ example clause wording as the query"),
    ("p2c_both", "+ both"),
]


def main() -> None:
    print("| Run | Change | clause_hit@5 | span_recall@5 | mrr | macro_clause_hit |")
    print("|-----|--------|--------------|---------------|-----|------------------|")
    for name, label in RUNS:
        path = RESULTS_DIR / f"{name}.summary.json"
        if not path.exists():
            print(f"| {name} | {label} | missing | | | |")
            continue
        s = json.loads(path.read_text(encoding="utf-8"))
        print(
            f"| {name} | {label} | {s['clause_hit_pct']} | {s['span_recall_pct']} | {s['mrr']} | {s['macro_clause_hit_pct']} |"
        )


if __name__ == "__main__":
    main()
