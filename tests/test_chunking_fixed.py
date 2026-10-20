from app.chunking import CHUNKERS, fixed_chunks

TEXT = "word " * 600


def test_every_chunk_is_exactly_the_text_it_points_to():
    chunks = fixed_chunks(TEXT)
    assert chunks
    assert all(TEXT[c.start:c.end] == c.content for c in chunks)


def test_chunks_respect_the_size_limit():
    assert max(c.end - c.start for c in fixed_chunks(TEXT, size=1000, overlap=150)) <= 1000


def test_neighbouring_chunks_overlap():
    first, second = fixed_chunks(TEXT, size=1000, overlap=150)[:2]
    assert second.start < first.end


def test_whole_text_is_covered():
    chunks = fixed_chunks(TEXT)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(TEXT.rstrip())


def test_fixed_chunks_have_no_heading():
    assert all(c.heading is None for c in fixed_chunks(TEXT))


def test_empty_text_gives_no_chunks():
    assert fixed_chunks("") == []


def test_both_strategies_are_registered():
    assert set(CHUNKERS) == {"fixed", "clause"}
