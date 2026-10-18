"""Query expansion for clause search: ask the LLM for a realistic example of how each clause type is worded.

  python scripts/make_clause_queries.py        # 41 cheap LLM calls -> data/cuad/clause_examples.json

Contracts say "Neither party shall assign this Agreement without prior written consent", not "Anti-Assignment:
does a party need consent to assign?". Searching with an example of the real wording matches the real text better.
The LLM only sees the clause name and CUAD's one-line description, never any contract or label.
Use it in the eval with --expand.
"""
import json
from pathlib import Path

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.llm import get_chat_model

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a commercial contracts lawyer. Write a short, realistic example of how a clause of the "
            "given type is typically worded in a commercial contract (2 or 3 sentences, generic party names "
            "like 'Company' and 'Distributor'). Use the usual legal phrasing and key terms. "
            'Reply with JSON only: {{"example": "<the clause wording>"}}',
        ),
        ("human", "Clause type: {category}\nWhat it covers: {question}"),
    ]
)


def main() -> None:
    categories = json.loads(Path("data/cuad/categories.json").read_text(encoding="utf-8"))
    chain = PROMPT | get_chat_model(model_kwargs={"response_format": {"type": "json_object"}}) | JsonOutputParser()
    out = {}
    for name, info in categories.items():
        if name == "Document Name":
            continue
        out[name] = chain.invoke({"category": name, "question": info["question"]})["example"].strip()
        print(f"{name}: {out[name][:100]}")
    Path("data/cuad/clause_examples.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {len(out)} examples to data/cuad/clause_examples.json")


if __name__ == "__main__":
    main()
