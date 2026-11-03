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


# ── TimeSpec (rulebook §2) ───────────────────────────────────────────────────
from datetime import date

from src.datamate.time_resolver import resolve_time_spec

CAL = pd.read_csv("train/calendar.csv")


def _p(p):
    return (p.start.isoformat(), p.end.isoformat()) if p else None


def test_regression_tet_this_year_vs_last_year():
    # The old resolver answered "Năm nay"
    s = resolve_time_spec("Cho hỏi doanh thu dịp Tết năm nay thay đổi bao nhiêu % so với dịp Tết năm ngoái?", "2026-06-28", CAL)
    assert _p(s.primary) == ("2026-02-10", "2026-02-26")
    assert _p(s.compare) == ("2025-01-22", "2025-02-07")
    assert s.confident


def test_regression_month_to_date_range():
    # The old resolver answered only today
    s = resolve_time_spec("Từ đầu tháng này đến hết hôm nay, doanh thu hơn hay kém cùng kỳ tháng trước mấy %?", "2026-09-07", CAL)
    assert _p(s.primary) == ("2026-09-01", "2026-09-07")
    assert _p(s.compare) == ("2026-08-01", "2026-08-07")


def test_regression_telex_month():
    # The old resolver returned None for 'thangs'
    s = resolve_time_spec("so cùng kỳ năm trước, doanh thu thangs 5/2026 tawng/giamr mấy %", "2026-07-28", CAL)
    assert _p(s.primary) == ("2026-05-01", "2026-05-31")
    assert _p(s.compare) == ("2025-05-01", "2025-05-31")


def test_this_month_vs_last_month_same_day_count_capped():
    s = resolve_time_spec("doanh thu tháng này so với tháng trước tăng bao nhiêu %", "2026-03-31", CAL)
    assert _p(s.primary) == ("2026-03-01", "2026-03-31")
    assert _p(s.compare) == ("2026-02-01", "2026-02-28")


def test_month_vs_previous_month_full():
    s = resolve_time_spec("Tháng 2/2025 doanh thu thay đổi bao nhiêu % so với tháng liền trước?", "2026-09-14", CAL)
    assert _p(s.primary) == ("2025-02-01", "2025-02-28")
    assert _p(s.compare) == ("2025-01-01", "2025-01-31")
    s = resolve_time_spec("so vs cung thang nam truoc, dthu thang 4 nam 2026 thay doi may %", "2026-06-16", CAL)
    assert _p(s.compare) == ("2025-04-01", "2025-04-30")


def test_month_without_year_after_today_is_last_year():
    assert _p(resolve_time_spec("tháng 8 doanh thu bao nhiêu", "2026-05-10", CAL).primary) == ("2025-08-01", "2025-08-31")
    assert _p(resolve_time_spec("tháng 5 doanh thu bao nhiêu", "2026-05-10", CAL).primary) == ("2026-05-01", "2026-05-31")
    assert _p(resolve_time_spec("quý 3 có bao nhiêu đơn", "2026-05-10", CAL).primary) == ("2025-07-01", "2025-09-30")


def test_ytd_and_last_n_days():
    assert _p(resolve_time_spec("Doanh thu từ đầu năm tới hết hnay là bao nhiêu?", "2025-04-13", CAL).primary) == ("2025-01-01", "2025-04-13")
    assert _p(resolve_time_spec("Từ 1/1 năm nay đến hết hôm nay đã bán đc bao nhiêu đơn?", "2025-07-18", CAL).primary) == ("2025-01-01", "2025-07-18")
    assert _p(resolve_time_spec("dthu 30 ngay gan nhat", "2026-03-01", CAL).primary) == ("2026-01-31", "2026-03-01")


def test_tet_vs_days_before_tet():
    s = resolve_time_spec("Dịp Tết âm lịch 2026 có doanh thu hơn hay kém 17 ngày trước Tết bao nhiêu %?", "2026-09-14", CAL)
    assert _p(s.primary) == ("2026-02-10", "2026-02-26")
    assert _p(s.compare) == ("2026-01-24", "2026-02-09")
    assert s.confident


def test_price_change_windows():
    s = resolve_time_spec("pin panasonic aa đổi giá từ 22.000đ lên 23.000đ vào ngày 02/06/2026. 30 ngày kể từ ngày đó "
                          "bán được nhìu hay ít hơn 30 ngày liền trước bao nhiêu %", "2026-08-05", CAL)
    assert _p(s.primary) == ("2026-06-02", "2026-07-01")
    assert _p(s.compare) == ("2026-05-03", "2026-06-01")
    s = resolve_time_spec("Kể từ khi Prospan siro ho chai 100ml lên giá 97.500 đồng/chai, 14 ngày đầu bán đc nhiều hay ít hơn "
                          "14 ngày liền trước đó bao nhiêu phần trăm?", "2026-07-01", CAL)
    assert not s.confident and s.primary is None  # the date of the increase must come from the data


def test_ordered_month_lists():
    s = resolve_time_spec("cho tôi dthu của tháng 1/2026, tháng 2/2026 vaf tháng 3/2026, ghi lần luwowtj theo thứ tự đó", "2026-09-20", CAL)
    assert [p.start.month for p in s.periods] == [1, 2, 3]
    s = resolve_time_spec("dthu 3 thang lien nhau ket thuc o thang 12/2025 la bnhiu moi thang", "2026-01-20", CAL)
    assert [(p.start.year, p.start.month) for p in s.periods] == [(2025, 10), (2025, 11), (2025, 12)]
    s = resolve_time_spec("Xu hướng doanh thu 3 tháng 9/2025 → 10/2025 → 11/2025: có phải tháng 10/2025 cao hơn tháng 9/2025 ạ?", "2026-09-20", CAL)
    assert [p.start.month for p in s.periods] == [9, 10, 11] and s.compare is None
    s = resolve_time_spec("Từ tháng 8/2025 tới tháng 10/2025, tháng nào cửa hàng bán được nhiều tiền nhất?", "2026-07-09", CAL)
    assert len(s.periods) == 3 and s.granularity == "month"
    s = resolve_time_spec("từ thangs 9/2025 qua tháng 10/2025 r tháng 11/2025, doanh thu có tăng liên tục hok", "2026-07-09", CAL)
    assert [p.start.month for p in s.periods] == [9, 10, 11]


def test_point_in_time_and_shift():
    s = resolve_time_spec("Tính đến hết ngày 19/08/2026, có bao nhiêu khách còn nợ cửa hàng?", "2026-08-19", CAL)
    assert s.point_in_time == date(2026, 8, 19) and s.primary is None
    s = resolve_time_spec("Cuối ngày hôm nay Bánh croissant còn mấy hộp chẵn, mấy cái lẻ?", "2026-09-14", CAL)
    assert s.point_in_time == date(2026, 9, 14)
    s = resolve_time_spec("Trong tháng 5/2025, doanh thu ca chiều hơn hay kém ca sáng bao nhiêu phần trăm vậy?", "2026-08-05", CAL)
    assert (s.shift, s.compare_shift) == ("afternoon", "morning")
    assert _p(s.compare) == ("2025-05-01", "2025-05-31")
    s = resolve_time_spec("Theo tồn kho ngày 25/05/2026, có mấy lô hàng còn tồn sẽ hết hạn từ nay đến hết ngày 22/09/2026?", "2026-05-25", CAL)
    assert s.point_in_time == date(2026, 5, 25) and _p(s.primary) == ("2026-05-25", "2026-09-22")


def test_next_n_days_and_weekday():
    assert _p(resolve_time_spec("hàng nào hết hạn trong 90 ngày tới", "2026-05-10", CAL).primary) == ("2026-05-10", "2026-08-08")
    assert resolve_time_spec("Trung bình mỗi ngày thứ bảy của tháng 3/2026 bán được bao nhiêu", "2026-05-10", CAL).weekday == 6
    assert resolve_time_spec("ghi theo thứ tự tháng 1/2026", "2026-05-10", CAL).weekday is None


def test_trend_question_is_confident():
    s = resolve_time_spec("Từ tháng 2/2026 đến tháng 4/2026, doanh thu tháng sau có luôn cao hơn tháng trước không?", "2026-07-01", CAL)
    assert s.confident and len(s.periods) == 3


def test_unresolvable_is_not_confident():
    assert not resolve_time_spec("doanh thu ngày lương tháng 5/2026", "2026-07-01", CAL).confident
    assert resolve_time("doanh thu ngày lương tháng 5/2026", "2026-07-01", CAL) is None
