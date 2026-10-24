"""find_quote is the hallucination check: a quote the model gives must be found in the contract text."""

from app.extract import find_quote

CONTRACT = (
    "This Agreement is governed by the laws of THEUNAUTHORIZED State.\n\n"
    "Either party may terminate on 30 days notice, except during the Initial Term. "
    "Fees are payable monthly."
)


def text_of(quote: str) -> str:
    start, end = find_quote(quote, CONTRACT)
    return CONTRACT[start:end]


def test_exact_quote_is_found_at_the_right_place():
    assert (
        text_of("Either party may terminate on 30 days notice")
        == "Either party may terminate on 30 days notice"
    )


def test_whitespace_and_case_are_ignored():
    assert (
        text_of("governed by the laws of the unauthorized state")
        == "governed by the laws of THEUNAUTHORIZED State"
    )


def test_trailing_full_stop_added_by_the_model_is_ignored():
    assert (
        text_of("Either party may terminate on 30 days notice.")
        == "Either party may terminate on 30 days notice"
    )


def test_surrounding_quote_marks_are_ignored():
    assert text_of('"Fees are payable monthly."') == "Fees are payable monthly"


def test_ellipsis_skips_words_but_keeps_order():
    assert text_of("Either party may terminate ... Fees are payable monthly") == (
        "Either party may terminate on 30 days notice, except during the Initial Term. Fees are payable monthly"
    )


def test_ellipsis_pieces_out_of_order_do_not_match():
    assert find_quote("Fees are payable monthly ... Either party may terminate", CONTRACT) is None


def test_text_that_is_not_in_the_contract_is_not_found():
    assert find_quote("The supplier shall indemnify the customer", CONTRACT) is None


def test_empty_or_tiny_quotes_are_not_found():
    for quote in (None, "", "   ", "ab"):
        assert find_quote(quote, CONTRACT) is None
