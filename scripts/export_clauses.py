"""Write app/clauses.json from the CUAD files, so the app (and Docker image) do not need the CUAD download.

  python -m scripts.export_clauses
"""
import json
import sys

from app.clauses import CUAD_DIR, JSON_PATH


def main() -> None:
    cats_path, ex_path = CUAD_DIR / "categories.json", CUAD_DIR / "clause_examples.json"
    for p in (cats_path, ex_path):
        if not p.exists():
            sys.exit(f"{p} not found. Run: python scripts/prepare_cuad.py and python -m scripts.make_clause_queries")
    cats = json.loads(cats_path.read_text(encoding="utf-8"))
    examples = json.loads(ex_path.read_text(encoding="utf-8"))
    data = {c: {**info, "example": examples.get(c)} for c, info in cats.items() if c != "Document Name"}
    JSON_PATH.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(data)} clause types to {JSON_PATH}")


if __name__ == "__main__":
    main()
