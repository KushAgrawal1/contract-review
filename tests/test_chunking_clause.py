import pytest

from app.chunking import _is_section_start, clause_chunks

CONTRACT = (
    "ARTICLE 1 DEFINITIONS\n\n" + "Alpha beta gamma. " * 30
    + "\n\n2. Term\n\n" + "The term is five years. " * 20
    + "\n\n3. Fees\n\nFees are due monthly."
)


@pytest.mark.parametrize(
    "line",
    ["ARTICLE 5", "Section 2 Payment", "1. Definitions", "12.3.4 Notices", "TERMINATION"],
)
def test_headings_are_recognised(line):
    assert _is_section_start(line)


@pytest.mark.parametrize("line", ["This is a normal sentence.", "AB", "2025 was a good year"])
def test_ordinary_lines_are_not_headings(line):
    assert not _is_section_start(line)


def test_new_sections_start_new_chunks_with_their_heading():
    chunks = clause_chunks(CONTRACT)
    assert [c.heading for c in chunks] == ["ARTICLE 1 DEFINITIONS", "2. Term", "3. Fees"]


def test_chunk_ranges_point_into_the_original_text_in_order():
    chunks = clause_chunks(CONTRACT)
    assert all(0 <= c.start < c.end <= len(CONTRACT) for c in chunks)
    assert all(a.end <= b.start for a, b in zip(chunks, chunks[1:]))


def test_embedded_content_carries_the_heading():
    chunks = clause_chunks(CONTRACT)
    assert all(c.heading in c.content for c in chunks)


def test_long_paragraph_is_split_and_keeps_its_heading():
    text = "1. Big\n\n" + "word " * 1200
    chunks = clause_chunks(text, max_chars=1800)
    assert len(chunks) > 2
    assert all(c.end - c.start <= 1800 for c in chunks)
    assert {c.heading for c in chunks} == {"1. Big"}


def test_empty_text_gives_no_chunks():
    assert clause_chunks("") == []
