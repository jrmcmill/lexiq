from src.ui_helpers import average_relevance, coerce_distance, format_distance


def test_coerce_distance_handles_missing_and_bad_values():
    assert coerce_distance(None) is None
    assert coerce_distance("bad") is None
    assert coerce_distance(float("nan")) is None
    assert coerce_distance(0.25) == 0.25


def test_format_distance_uses_n_a_for_missing_values():
    assert format_distance(None) == "n/a"
    assert format_distance("bad") == "n/a"
    assert format_distance(0.1234) == "0.123"


def test_average_relevance_ignores_none_distances():
    items = [
        {"distance": 0.2},
        {"distance": None},
        {"distance": "bad"},
        {"distance": 0.4},
    ]
    score = average_relevance(items)
    assert score == ((1 - 0.2) + (1 - 0.4)) / 2


def test_average_relevance_returns_none_when_no_numeric_distances():
    assert average_relevance([{"distance": None}, {"distance": "bad"}]) is None