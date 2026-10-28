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
    # Ratio converted to percent
    assert format_percent(0.0527) == 5.27
    assert format_percent(5.27) == 5.27


def test_format_date():
    assert format_date("2026-08-15") == "2026-08-15"
    assert format_date("15/08/2026") == "2026-08-15"
    assert format_date("2025-09") == "2025-09"
    assert format_date("09/2025") == "2025-09"


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


def test_format_submission_item():
    item = format_submission_item("q1", 4820000, "number")
    assert item == {"id": "q1", "type": "number", "answer": 4820000}

    item_abstain = format_submission_item("q2", None, "abstain")
    assert item_abstain == {"id": "q2", "type": "abstain"}

    item_unanswerable = format_submission_item("q3", "UNANSWERABLE", "unanswerable")
    assert item_unanswerable["type"] == "unanswerable"
