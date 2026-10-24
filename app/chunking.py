"""Two ways to cut a contract into chunks. Both return character ranges of the original text, so a chunk can
be checked against the expert-marked clause spans exactly.

fixed   LangChain RecursiveCharacterTextSplitter, 1000 characters, 150 overlap (the usual default, our baseline)
clause  follows the contract's own structure: a new chunk starts at a numbered section / ARTICLE / ALL-CAPS
        heading, small sections are kept whole, long paragraphs are split with the same LangChain splitter. Each chunk carries its section heading.
"""

import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    start: int  # char_start, inclusive
    end: int  # char_end, exclusive
    heading: str | None
    content: str  # text that gets embedded (heading line + the passage)


def _split(text: str, start: int, stop: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """LangChain's RecursiveCharacterTextSplitter on text[start:stop], returned as character ranges of `text`.

    add_start_index gives each piece's offset. We verify every piece really is text[s:e], because the eval
    compares chunk ranges with the experts' marked spans and a wrong offset would silently corrupt the scores.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, add_start_index=True
    )
    out = []
    for doc in splitter.create_documents([text[start:stop]]):
        s = start + doc.metadata["start_index"]
        e = s + len(doc.page_content)
        if text[s:e] != doc.page_content:
            raise ValueError(f"splitter offset mismatch at {s}")
        out.append((s, e))
    return out


def fixed_chunks(text: str, size: int = 1000, overlap: int = 150) -> list[Chunk]:
    return [Chunk(s, e, None, text[s:e]) for s, e in _split(text, 0, len(text), size, overlap)]


PARAGRAPH = re.compile(r"\S[\s\S]*?(?=\n[ \t]*\n|\Z)")
SECTION_START = re.compile(
    r"^\s*(?:"
    r"(?:ARTICLE|Article|SECTION|Section)\s+[\dIVXLivxl]+\b"  # ARTICLE 5 / Section 2
    r"|\d{1,2}(?:\.\d{1,2}){0,3}[.)]?\s+\S"  # 1. / 1.1 / 12.3.4
    r")"
)


def _is_section_start(par: str) -> bool:
    first = par.split("\n", 1)[0].strip()
    if SECTION_START.match(first):
        return True
    letters = [c for c in first if c.isalpha()]
    return (
        3 <= len(letters) and len(first) <= 80 and first.upper() == first
    )  # ALL-CAPS heading line


def _with_heading(heading: str | None, body: str) -> str:
    body = body.strip()
    if not heading or body.startswith(heading[:60]):  # the passage already opens with its heading
        return body
    return f"[{heading}]\n{body}"


def clause_chunks(text: str, max_chars: int = 1800, min_chars: int = 300) -> list[Chunk]:
    paragraphs = [(m.start(), m.end()) for m in PARAGRAPH.finditer(text)]
    chunks: list[Chunk] = []
    heading: str | None = None
    cur_start = cur_end = None
    cur_heading: str | None = None

    def flush() -> None:
        nonlocal cur_start, cur_end
        if cur_start is not None and text[cur_start:cur_end].strip():
            chunks.append(
                Chunk(
                    cur_start,
                    cur_end,
                    cur_heading,
                    _with_heading(cur_heading, text[cur_start:cur_end]),
                )
            )
        cur_start = cur_end = None

    for s, e in paragraphs:
        par = text[s:e]
        starts_section = _is_section_start(par)
        if starts_section:
            heading = " ".join(par.split("\n", 1)[0].split())[:100]
        length = e - s
        if length > max_chars:  # one huge paragraph: close what we have, then split it
            flush()
            for ws, we in _split(text, s, e, max_chars, overlap=100):
                chunks.append(Chunk(ws, we, heading, _with_heading(heading, text[ws:we])))
            continue
        too_big = cur_start is not None and (e - cur_start) > max_chars
        new_section = (
            starts_section and cur_start is not None and (cur_end - cur_start) >= min_chars
        )
        if too_big or new_section:
            flush()
        if cur_start is None:
            cur_start, cur_heading = s, heading
        cur_end = e
    flush()
    return chunks


CHUNKERS = {"fixed": fixed_chunks, "clause": clause_chunks}
