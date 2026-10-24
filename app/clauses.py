"""The 40 clause types the app reviews: CUAD's name, question and answer format, plus an LLM-written example wording.

Read from app/clauses.json (committed, so the deployed app does not need the CUAD download). Create it once with
`python -m scripts.export_clauses`. Falls back to data/cuad/ if that file is missing.
"""

import json
from functools import lru_cache
from pathlib import Path

APP_DIR = Path(__file__).parent
JSON_PATH = APP_DIR / "clauses.json"
CUAD_DIR = APP_DIR.parent / "data" / "cuad"
SKIP = {"Document Name"}  # the title, not a clause


@lru_cache(maxsize=1)
def load_clauses() -> dict[str, dict]:
    """{category: {"question", "answer_format", "example" (may be None)}}"""
    if JSON_PATH.exists():
        data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    else:
        cats_path, ex_path = CUAD_DIR / "categories.json", CUAD_DIR / "clause_examples.json"
        if not cats_path.exists():
            raise FileNotFoundError(
                "No clause list found. Run: python scripts/prepare_cuad.py, python -m scripts.make_clause_queries, "
                "then python -m scripts.export_clauses"
            )
        cats = json.loads(cats_path.read_text(encoding="utf-8"))
        examples = json.loads(ex_path.read_text(encoding="utf-8")) if ex_path.exists() else {}
        data = {c: {**info, "example": examples.get(c)} for c, info in cats.items()}
    return {c: v for c, v in data.items() if c not in SKIP}


def query_text(category: str, info: dict) -> str:
    """Same search text the evaluation used: the clause question plus an example of how such a clause is worded."""
    base = f"{category}: {info['question']}"
    return f"{base}\nExample wording: {info['example']}" if info.get("example") else base
