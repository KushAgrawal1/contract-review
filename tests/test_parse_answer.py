from app.extract import parse_answer


def test_present_clause_keeps_its_quote():
    assert parse_answer('{"present": true, "quote": "Fees are payable monthly."}') == {
        "present": True,
        "quote": "Fees are payable monthly.",
    }


def test_absent_clause_drops_any_quote():
    assert parse_answer('{"present": false, "quote": "stray text"}') == {
        "present": False,
        "quote": None,
    }


def test_invalid_json_is_flagged_not_raised():
    assert parse_answer("not json") == {"present": False, "quote": None, "parse_error": True}


def test_none_content_is_flagged_not_raised():
    assert parse_answer(None)["parse_error"] is True
