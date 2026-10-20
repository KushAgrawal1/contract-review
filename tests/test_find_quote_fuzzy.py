"""find_quote_fuzzy accepts quotes that are almost, but not exactly, in the contract."""
from app.extract import FUZZY_MIN, find_quote_fuzzy

SENTENCE = (
    "The Supplier shall provide the Services in accordance with the specifications set out in Schedule 1 "
    "and shall use reasonable skill and care at all times during the term of this Agreement."
)
CONTRACT = f"Intro text. {SENTENCE} More text after."


def test_quote_with_an_inserted_aside_still_matches():
    quote = SENTENCE.replace("Schedule 1", "Schedule 1 (as defined above)")
    start, end, share = find_quote_fuzzy(quote, CONTRACT)
    assert share >= FUZZY_MIN
    assert CONTRACT[start:end].startswith("The Supplier shall provide")
    assert CONTRACT[start:end].endswith("this Agreement")


def test_exact_quote_scores_full_match():
    assert find_quote_fuzzy(SENTENCE, CONTRACT)[2] == 1.0


def test_short_quotes_are_not_fuzzy_matched():
    assert find_quote_fuzzy("The Supplier shall provide the Services", CONTRACT) is None


def test_unrelated_text_is_rejected():
    unrelated = "The licensee may not sublicense, assign or transfer any rights granted under this licence to third parties."
    assert find_quote_fuzzy(unrelated, CONTRACT) is None


def test_missing_quote_is_rejected():
    assert find_quote_fuzzy(None, CONTRACT) is None
    assert find_quote_fuzzy("", CONTRACT) is None
