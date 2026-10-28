import pytest
import pandas as pd
from src.datamate.time_resolver import resolve_time, TimeRange


def test_explicit_exact_date():
    tr = resolve_time("xem giúp ngày 25/09/2025 cửa hàng bán kẹo", "2026-08-24")
    assert tr is not None
    assert tr.start_date == "2025-09-25"
    assert tr.end_date == "2025-09-25"
    assert tr.is_exact_day is True


def test_hom_qua():
    tr = resolve_time("Cho tôi biết số đơn hôm qua.", "2025-09-24")
    assert tr is not None
    assert tr.start_date == "2025-09-23"
    assert tr.end_date == "2025-09-23"


def test_hom_nay():
    tr = resolve_time("Doanh thu hôm nay là bao nhiêu?", "2025-09-24")
    assert tr is not None
    assert tr.start_date == "2025-09-24"
    assert tr.end_date == "2025-09-24"


def test_thang_nay():
    tr = resolve_time("Tháng này bán chạy nhất mặt hàng gì?", "2026-05-24")
    assert tr is not None
    assert tr.start_date == "2026-05-01"
    assert tr.end_date == "2026-05-24"


def test_thang_truoc():
    tr = resolve_time("Tháng trước doanh thu bao nhiêu?", "2026-05-24")
    assert tr is not None
    assert tr.start_date == "2026-04-01"
    assert tr.end_date == "2026-04-30"


def test_tuan_nay():
    # 2026-05-24 was a Sunday
    tr = resolve_time("Tuần này doanh thu bao nhiêu?", "2026-05-24")
    assert tr is not None
    assert tr.start_date == "2026-05-18"  # Monday
    assert tr.end_date == "2026-05-24"


def test_tuan_roi():
    # 2026-03-04 was a Wednesday
    tr = resolve_time("Tuần rồi bán được bao nhiêu ly?", "2026-03-04")
    assert tr is not None
    assert tr.start_date == "2026-02-23"  # Monday of prev week
    assert tr.end_date == "2026-03-01"    # Sunday of prev week


def test_quy_4_2025():
    tr = resolve_time("số đơn hàng quý 4/2025 là bao nhiêu", "2026-08-24")
    assert tr is not None
    assert tr.start_date == "2025-10-01"
    assert tr.end_date == "2025-12-31"


def test_thang_11_2025():
    tr = resolve_time("doanh thu tháng 11/2025 là bao nhiêu", "2026-08-24")
    assert tr is not None
    assert tr.start_date == "2025-11-01"
    assert tr.end_date == "2025-11-30"


def test_n_days_recent():
    tr = resolve_time("7 ngày gần nhất bán được bao nhiêu", "2026-08-24")
    assert tr is not None
    assert tr.start_date == "2026-08-18"
    assert tr.end_date == "2026-08-24"


def test_tet_window():
    cal_df = pd.read_csv("train/calendar.csv")
    tr = resolve_time("Tết 2025 bán được bao nhiêu tiền", "2026-08-24", calendar_df=cal_df)
    assert tr is not None
    # Tet 2025: 23 thang Chap nam Giap Thin / 2024 to mung 10 thang Gieng nam At Ty / 2025
    assert tr.start_date == "2025-01-22"
    assert tr.end_date == "2025-02-07"
