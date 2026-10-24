from app.clauses import load_clauses, query_text
from app.extract import HEAD_CATEGORIES, NOT_THIS


def test_forty_clause_types_are_loaded():
    assert len(load_clauses()) == 40


def test_title_row_is_not_treated_as_a_clause():
    assert "Document Name" not in load_clauses()


def test_every_clause_has_a_question_and_an_answer_format_field():
    # answer_format may be blank: CUAD leaves it empty for "Affiliate License-Licensee"
    for name, info in load_clauses().items():
        assert info["question"].strip(), name
        assert isinstance(info["answer_format"], str), name


def test_clause_names_used_in_extraction_rules_exist():
    # a typo in HEAD_CATEGORIES or NOT_THIS would silently switch that rule off
    known = set(load_clauses())
    assert HEAD_CATEGORIES <= known
    assert set(NOT_THIS) <= known


def test_query_text_without_example():
    assert (
        query_text("Governing Law", {"question": "Which law?", "example": None})
        == "Governing Law: Which law?"
    )


def test_query_text_with_example():
    text = query_text("Governing Law", {"question": "Which law?", "example": "laws of England"})
    assert text == "Governing Law: Which law?\nExample wording: laws of England"
