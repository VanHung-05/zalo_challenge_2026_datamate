import pytest
from src.datamate.formatter import (
    format_number,
    format_percent,
    format_date,
    format_boolean,
    format_string,
    format_list,
    format_table,
    format_submission_item,
    round_half_up,
)
import pandas as pd


def test_format_number():
    assert format_number(4820000) == 4820000
    assert format_number("4.820.000") == 4820000
    assert format_number("4,820,000") == 4820000
    assert format_number("4820000 đ") == 4820000
    assert format_number(18.6) == 18.6
    assert format_number(18.0) == 18
    assert format_number(pd.Series([100])) == 100


def test_format_percent():
    assert format_percent(12.4) == 12.4
    assert format_percent("12,4%") == 12.4
    assert format_percent(5.27) == 5.27
    # Values are taken as percent units: 0.0527 means 0.05 %, it is not rescaled to 5.27
    assert format_percent(0.0527) == 0.05
    assert format_percent(-12.445) == -12.45
    assert format_percent(-0.001) == 0.0


def test_round_half_up_and_money():
    assert round_half_up(2.675, 2) == 2.68
    assert round_half_up(0.5) == 1 and round_half_up(1.5) == 2 and round_half_up(2.5) == 3
    assert format_number(156347.5, is_money=True) == 156348
    assert format_number(3.3333, is_money=True) == 3
    assert format_number(3.3333) == 3.33
    assert format_number(2.0000000001) == 2
    with pytest.raises(ValueError):
        format_number(True)


def test_format_date():
    assert format_date("2026-08-15") == "2026-08-15"
    assert format_date("15/08/2026") == "2026-08-15"
    assert format_date("2025-09") == "2025-09"
    assert format_date("09/2025") == "2025-09"
    assert format_date("2025-09-30", granularity="month") == "2025-09"
    assert format_date(pd.Timestamp("2026-02-04 10:30")) == "2026-02-04"
    with pytest.raises(ValueError):
        format_date("2025-09", granularity="day")


def test_format_boolean():
    assert format_boolean(True) is True
    assert format_boolean("có") is True
    assert format_boolean("đúng") is True
    assert format_boolean("không") is False
    assert format_boolean(False) is False


def test_format_list():
    assert format_list(["A", "B"]) == ["A", "B"]
    assert format_list("Single") == ["Single"]
    assert format_list(pd.Series([1, 2, 3])) == [1, 2, 3]
    assert format_list(["SP0001", "0123", 16.0]) == ["SP0001", "0123", 16]
    with pytest.raises(ValueError):
        format_list(["A", None, "B"])


def test_format_table():
    df = pd.DataFrame({"khach_hang": ["An"], "so_tien_no": [949333.5]})
    assert format_table(df, is_money=True) == {"columns": ["khach_hang", "so_tien_no"], "rows": [["An", 949334]]}


def test_format_submission_item():
    item = format_submission_item("q1", 4820000, "number")
    assert item == {"id": "q1", "type": "number", "answer": 4820000}

    item_abstain = format_submission_item("q2", None, "abstain")
    assert item_abstain == {"id": "q2", "type": "abstain"}

    item_unanswerable = format_submission_item("q3", "UNANSWERABLE", "unanswerable")
    assert item_unanswerable["type"] == "unanswerable"
    assert isinstance(item_unanswerable["reason"], str)

    assert format_submission_item("q4", "2026-02-14", "date", date_granularity="month")["answer"] == "2026-02"
    assert format_submission_item("q5", "abc", "number") == {"id": "q5", "type": "abstain"}
    assert format_submission_item("q6", pd.DataFrame({"a": [1]}), "number")["type"] == "abstain"
