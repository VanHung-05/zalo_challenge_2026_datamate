import os

import pandas as pd
import pytest

from src.datamate.rules import Rulebook, holiday_dates, load_calendar, lunar_to_solar, validate_calendar

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _rulebook(split):
    path = os.path.join(ROOT, split, "rulebook.md")
    if not os.path.exists(path):
        pytest.skip(f"{path} not available")
    return Rulebook(path)


def test_params_read_from_split_rulebook():
    rb = _rulebook("train")
    assert rb.params["revenue_excludes_returned_whole_order"].found
    assert rb.params["revenue_subtracts_order_discount"].found and rb.value("revenue_subtracts_order_discount") is False
    assert rb.value("shifts") == {"morning": ("06:00", "13:59"), "afternoon": ("14:00", "21:59")}
    assert rb.params["tet_window"].found and rb.value("tet_window")["start_lunar_day"] == 23
    assert rb.params["tie_break"].found


def test_rules_missing_in_split_are_uncertain_not_assumed():
    train, public = _rulebook("train"), _rulebook("public_test")
    # "Khách sộp" is only defined in the public-test rulebook
    assert "big_spender_min_orders" in train.uncertain_params()
    assert public.params["big_spender_min_orders"].found and public.value("big_spender_min_orders") == 3
    assert "Sửa xe" in public.industry_sections() and "Sửa xe" not in train.industry_sections()


def test_missing_rulebook_everything_uncertain():
    rb = Rulebook(None)
    assert rb.raw_text == "" and set(rb.uncertain_params()) == set(rb.params)


def test_calendar_known_lunar_new_years():
    cal = load_calendar(os.path.join(ROOT, "train"))
    if cal is None:
        pytest.skip("calendar.csv not available")
    assert validate_calendar(cal) == []
    assert [lunar_to_solar(cal, y, 1, 1) for y in (2024, 2025, 2026, 2027)] == [
        "2024-02-10", "2025-01-29", "2026-02-17", "2027-02-06"]
    # Tết 2026 window: 23 tháng Chạp (lunar 2025) .. mùng 10 tháng Giêng (lunar 2026)
    assert lunar_to_solar(cal, 2025, 12, 23) == "2026-02-10"
    assert lunar_to_solar(cal, 2026, 1, 10) == "2026-02-26"
    assert "2025-10-06" in holiday_dates(cal, "Trung thu")


def test_validate_calendar_detects_bad_weekday():
    cal = pd.DataFrame({"date": ["2026-01-05", "2026-01-06"], "weekday": [2, 3], "lunar_day": [17, 18],
                        "lunar_month": [11, 11], "lunar_year": [2025, 2025], "lunar_leap": [False, False],
                        "holiday": [None, None]})
    assert any("weekday" in p for p in validate_calendar(cal))
