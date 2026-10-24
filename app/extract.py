"""Phase 3: turn a clause search into a structured answer.

For one (contract, clause type) pair we retrieve the best passages, show the LLM the exact contract text, and ask:
is this clause present, and if so which words of the contract are it? The LLM must copy those words verbatim, so
every answer can be checked against the contract text. A quote that cannot be found in the contract is a
hallucination and is flagged (grounded = False).
"""

import difflib
import json
import re
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate

from app.llm import get_chat_model

PROMPT_VERSION = "v4"  # v1 first; v2 stricter on related clause types + ignore blank fields; v3 adds per-clause "does not count" notes; v4 drops the two Affiliate License notes (they cut recall to 0)
HEAD_CHARS = 3000
# Parties and dates sit at the start of a contract, where retrieval is weak, so the opening is added as context.
HEAD_CATEGORIES = {"Parties", "Agreement Date", "Effective Date"}

# Clause types the model kept confusing with a neighbour on the 20 tuning contracts. Each note says what does NOT count.
# Written after reading tuning failures only, so the held-out run is what shows whether they generalise.
NOT_THIS = {
    "Notice Period To Terminate Renewal": "Only a notice period to stop an agreement from renewing (non-renewal notice). Notice to terminate early, for convenience or for breach is NOT this clause.",
    "Non-Transferable License": "Only a license (or rights to use IP) that the licensee cannot transfer. A general ban on assigning the whole agreement is Anti-Assignment, NOT this clause.",
    "Change Of Control": "Only a clause triggered by a change in who owns or controls a party (merger, acquisition, sale of the business) that gives a termination right, consent right or other effect. A plain ban on assignment is NOT this clause.",
    "Price Restrictions": "Only a limit on how much prices may change (price cap, freeze, or restriction on raising prices). Ordinary price lists and fee schedules are NOT this clause.",
    "Post-Termination Services": "Only a duty to keep providing services, support, transition help or payments AFTER termination or expiry. Non-competes and survival clauses are NOT this clause.",
    "Irrevocable Or Perpetual License": "Only a LICENSE that is irrevocable or perpetual. An assignment of ownership is NOT this clause.",
}

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a commercial contracts lawyer reviewing excerpts of ONE contract. "
            "Decide whether the excerpts contain a clause of the requested type. "
            "Answer present=true ONLY if an excerpt actually contains a clause of exactly that type. A clause of a "
            "related but different type does not count, and neither does a passage that merely mentions the topic. "
            "Ignore blank form fields, placeholders and template text such as '____' or '<date>'. "
            "If it is present, copy the relevant words EXACTLY as written in ONE excerpt (verbatim, no edits, no "
            "paraphrase, never join separate passages with '...', at most about 600 characters, the most relevant "
            "sentence or sentences). "
            "If the excerpts do not clearly contain it, answer present=false. Never guess or infer. "
            'Reply with JSON only: {{"present": true or false, "quote": "<verbatim text>" or null}}',
        ),
        (
            "human",
            "Clause type: {category}\nWhat to look for: {question}\n{note}Expected answer format: {answer_format}\n\n"
            "Excerpts:\n{excerpts}",
        ),
    ]
)


def build_chain(llm=None):
    llm = llm or get_chat_model(model_kwargs={"response_format": {"type": "json_object"}})
    return PROMPT | llm


def excerpts_text(contract_text: str, ranges: list[tuple[int, int]]) -> str:
    """Raw contract text for each range (not the chunk text, which carries an added heading line)."""
    return "\n\n".join(f"[{i}]\n{contract_text[s:e].strip()}" for i, (s, e) in enumerate(ranges, 1))


def passage_ranges(docs, contract_text: str, category: str, head: bool) -> list[tuple[int, int]]:
    ranges = [(d.metadata["char_start"], d.metadata["char_end"]) for d in docs]
    if head and category in HEAD_CATEGORIES:
        ranges = [(0, min(HEAD_CHARS, len(contract_text)))] + [
            r for r in ranges if r[0] >= HEAD_CHARS
        ]
    return ranges


@lru_cache(maxsize=64)
def _squashed(text: str) -> tuple[str, list[int]]:
    """Contract text without whitespace, lower-cased, plus each kept character's position in the original."""
    chars, pos = [], []
    for i, c in enumerate(text):
        if not c.isspace():
            low = c.lower()
            chars.append(low if len(low) == 1 else c)
            pos.append(i)
    return "".join(chars), pos


def _find_squashed(segment: str, squashed: str, pos: list[int], start_at: int = 0):
    needle = "".join(segment.split()).lower()
    if len(needle) < 3:
        return None
    i = squashed.find(needle, start_at)
    if i == -1:
        return None
    return pos[i], pos[i + len(needle) - 1] + 1, i + len(needle)


def find_quote(quote: str | None, contract_text: str) -> tuple[int, int] | None:
    """Character range of `quote` in the contract, or None if it is not there.

    Whitespace and capitalisation are ignored, because PDF text often has glued words ("THEUNAUTHORIZED") that a
    model naturally writes with spaces. A quote that uses "..." to skip words is matched piece by piece, in order.
    """
    if not quote or not quote.strip():
        return None
    squashed, pos = _squashed(contract_text)
    quote = (
        quote.strip().strip("\"'").rstrip(".,;:")
    )  # the model often adds a full stop where the contract has a comma
    segments = [seg for seg in re.split(r"\.\.\.|\u2026", quote) if len("".join(seg.split())) >= 3]
    if not segments:
        return None
    start = end = None
    cursor = 0
    for seg in segments:
        hit = _find_squashed(seg, squashed, pos, cursor)
        if hit is None:
            return None
        if start is None:
            start = hit[0]
        end, cursor = hit[1], hit[2]
    return start, end


FUZZY_MIN = 0.90  # share of the quote's characters that must be found, in order, in the contract


def find_quote_fuzzy(quote: str | None, contract_text: str) -> tuple[int, int, float] | None:
    """Fallback for quotes that are almost verbatim: the model skipped a page header or a "(as defined in ...)"
    aside, or fixed a typo. Finds the quote's neighbourhood with a few short anchors, then measures how much of
    the quote matches there. Returns (start, end, share_matched) when the share is at least FUZZY_MIN."""
    if not quote:
        return None
    q = "".join(quote.split()).lower().rstrip(".,;:")
    if len(q) < 60:
        return None
    squashed, pos = _squashed(contract_text)
    best = None
    for off in sorted({0, len(q) // 3, 2 * len(q) // 3, len(q) - 20}):
        anchor = q[off : off + 20]
        i = squashed.find(anchor)
        if i == -1:
            continue
        lo = max(0, i - off - 20)
        window = squashed[lo : lo + int(len(q) * 1.5) + 60]
        blocks = [
            b
            for b in difflib.SequenceMatcher(None, q, window, autojunk=False).get_matching_blocks()
            if b.size >= 4
        ]
        if not blocks:
            continue
        share = sum(b.size for b in blocks) / len(q)
        if best is None or share > best[0]:
            best = (share, pos[lo + blocks[0].b], pos[lo + blocks[-1].b + blocks[-1].size - 1] + 1)
    if best and best[0] >= FUZZY_MIN:
        return best[1], best[2], round(best[0], 3)
    return None


def parse_answer(content: str) -> dict:
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {"present": False, "quote": None, "parse_error": True}
    return {
        "present": bool(data.get("present")),
        "quote": data.get("quote") if data.get("present") else None,
    }


def extract_clause(
    chain, contract_text: str, category: str, info: dict, ranges: list[tuple[int, int]]
) -> dict:
    msg = chain.invoke(
        {
            "category": category,
            "question": info["question"],
            "answer_format": info["answer_format"],
            "note": f"Important: {NOT_THIS[category]}\n" if category in NOT_THIS else "",
            "excerpts": excerpts_text(contract_text, ranges),
        }
    )
    out = parse_answer(getattr(msg, "content", msg))
    span, match = None, None
    if out["present"]:
        span = find_quote(out["quote"], contract_text)
        match = "exact" if span else None
        if span is None and (fuzzy := find_quote_fuzzy(out["quote"], contract_text)):
            span, match = fuzzy[:2], "fuzzy"
    out["start"], out["end"] = span if span else (None, None)
    out["grounded"] = span is not None if out["present"] else None
    out["match"] = (
        match  # "exact" = found word for word; "fuzzy" = at least 90% found, see find_quote_fuzzy
    )
    usage = getattr(msg, "usage_metadata", None) or {}
    out["input_tokens"] = usage.get("input_tokens", 0)
    out["output_tokens"] = usage.get("output_tokens", 0)
    return out
