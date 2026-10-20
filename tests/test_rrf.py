from langchain_core.documents import Document

from app.retrieval import RRF_K, rrf_fuse


def doc(chunk_id: int) -> Document:
    return Document(page_content=f"chunk {chunk_id}", metadata={"chunk_id": chunk_id})


def ids(docs):
    return [d.metadata["chunk_id"] for d in docs]


def test_single_ranking_keeps_its_order():
    assert ids(rrf_fuse([doc(1), doc(2), doc(3)])) == [1, 2, 3]


def test_chunk_found_by_both_searches_beats_chunks_found_by_one():
    dense = [doc(1), doc(2), doc(3)]
    keyword = [doc(4), doc(2), doc(5)]
    assert ids(rrf_fuse(dense, keyword))[0] == 2


def test_score_is_sum_of_reciprocal_ranks():
    fused = rrf_fuse([doc(1), doc(2)], [doc(2), doc(1)])
    expected = 1 / (RRF_K + 1) + 1 / (RRF_K + 2)
    assert all(abs(d.metadata["score"] - expected) < 1e-12 for d in fused)


def test_each_chunk_appears_once():
    fused = rrf_fuse([doc(1), doc(2)], [doc(2), doc(3)])
    assert sorted(ids(fused)) == [1, 2, 3]


def test_empty_rankings_give_empty_result():
    assert rrf_fuse() == []
    assert rrf_fuse([], []) == []


def test_input_documents_are_not_modified():
    first = doc(1)
    rrf_fuse([first])
    assert "score" not in first.metadata
