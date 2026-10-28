"""Time resolver for DataMate.
Extracts and resolves natural language Vietnamese time expressions into exact (start_date, end_date).
Handles both absolute dates (DD/MM/YYYY, tháng M/YYYY, quý Q/YYYY) and relative dates (hôm qua, tháng này, tuần rồi).
"""

import calendar as pycalendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import re
from typing import Any, Dict, Optional, Tuple
import pandas as pd


@dataclass
class TimeRange:
    start_date: str          # YYYY-MM-DD
    end_date: str            # YYYY-MM-DD
    is_exact_day: bool = False
    is_month: bool = False
    is_quarter: bool = False
    description: str = ""


def get_month_range(year: int, month: int) -> Tuple[str, str]:
    """Return (first_day, last_day) of month in YYYY-MM-DD."""
    last_day = pycalendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{last_day:02d}"


def get_quarter_range(year: int, quarter: int) -> Tuple[str, str]:
    """Return (first_day, last_day) of quarter."""
    start_month = (quarter - 1) * 3 + 1
    end_month = start_month + 2
    last_day = pycalendar.monthrange(year, end_month)[1]
    return f"{year:04d}-{start_month:02d}-01", f"{year:04d}-{end_month:02d}-{last_day:02d}"


def get_tet_window(year: int, calendar_df: Optional[pd.DataFrame]) -> Optional[Tuple[str, str]]:
    """Tết Nguyên Đán Y: From 23 tháng Chạp năm Y-1 âm lịch to mùng 10 tháng Giêng năm Y âm lịch."""
    if calendar_df is None or calendar_df.empty:
        return None
    try:
        # Find start: lunar_day=23, lunar_month=12, lunar_year=year-1
        start_row = calendar_df[
            (calendar_df["lunar_day"] == 23) &
            (calendar_df["lunar_month"] == 12) &
            (calendar_df["lunar_year"] == year - 1)
        ]
        # Find end: lunar_day=10, lunar_month=1, lunar_year=year
        end_row = calendar_df[
            (calendar_df["lunar_day"] == 10) &
            (calendar_df["lunar_month"] == 1) &
            (calendar_df["lunar_year"] == year)
        ]
        if not start_row.empty and not end_row.empty:
            s_date = str(start_row["date"].iloc[0]).split()[0]
            e_date = str(end_row["date"].iloc[0]).split()[0]
            return s_date, e_date
    except Exception:
        pass
    return None


def resolve_time(
    question: str,
    as_of_date_str: str,
    calendar_df: Optional[pd.DataFrame] = None
) -> Optional[TimeRange]:
    """Analyze question text and resolve to concrete TimeRange."""
    if not as_of_date_str:
        return None

    try:
        as_of = datetime.strptime(as_of_date_str[:10], "%Y-%m-%d").date()
    except Exception:
        return None

    q = question.lower()

    # 1. Check for explicit exact date: DD/MM/YYYY or DD-MM-YYYY
    m_exact = re.search(r"(?:ngày\s+)?(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})", q)
    if m_exact:
        d, m, y = int(m_exact.group(1)), int(m_exact.group(2)), int(m_exact.group(3))
        d_str = f"{y:04d}-{m:02d}-{d:02d}"
        return TimeRange(d_str, d_str, is_exact_day=True, description=f"ngày {d_str}")

    # 2. Check for "hôm qua" / "hqua" / "hôm kia"
    if re.search(r"\b(?:hôm qua|hqua|hom qua)\b", q):
        yest = as_of - timedelta(days=1)
        yest_str = yest.strftime("%Y-%m-%d")
        return TimeRange(yest_str, yest_str, is_exact_day=True, description="hôm qua")

    if re.search(r"\b(?:hôm kia|hom kia)\b", q):
        day_before = as_of - timedelta(days=2)
        db_str = day_before.strftime("%Y-%m-%d")
        return TimeRange(db_str, db_str, is_exact_day=True, description="hôm kia")

    # 3. Check for "hôm nay"
    if re.search(r"\b(?:hôm nay|hom nay)\b", q):
        as_str = as_of.strftime("%Y-%m-%d")
        return TimeRange(as_str, as_str, is_exact_day=True, description="hôm nay")

    # 4. Check for N ngày gần nhất / qua / vừa qua
    m_ndays = re.search(r"(\d+)\s+ngày\s+(?:gần nhất|vừa qua|qua|trở lại đây)", q)
    if m_ndays:
        n = int(m_ndays.group(1))
        start_d = as_of - timedelta(days=n - 1)
        return TimeRange(start_d.strftime("%Y-%m-%d"), as_of.strftime("%Y-%m-%d"), description=f"{n} ngày gần nhất")

    # 5. Check for "Tết YYYY"
    m_tet = re.search(r"tết\s+(?:nguyên đán\s+)?(?:năm\s+)?(\d{4})", q)
    if m_tet:
        tet_year = int(m_tet.group(1))
        tet_win = get_tet_window(tet_year, calendar_df)
        if tet_win:
            return TimeRange(tet_win[0], tet_win[1], description=f"Tết Nguyên Đán {tet_year}")

    # 6. Check for quý Q/YYYY or quý Q năm YYYY
    m_quarter = re.search(r"(?:quý|quy)\s+([1-4])\s*(?:/|năm\s+)?(\d{4})", q)
    if m_quarter:
        q_num = int(m_quarter.group(1))
        q_year = int(m_quarter.group(2))
        s_date, e_date = get_quarter_range(q_year, q_num)
        # Cap at as_of_date if current quarter
        if q_year == as_of.year and s_date <= as_of_date_str <= e_date:
            e_date = as_of_date_str
        return TimeRange(s_date, e_date, is_quarter=True, description=f"Quý {q_num}/{q_year}")

    # Check for "quý này"
    if re.search(r"(?:quý này|quy nay)", q):
        current_quarter = (as_of.month - 1) // 3 + 1
        s_date, _ = get_quarter_range(as_of.year, current_quarter)
        return TimeRange(s_date, as_of_date_str, is_quarter=True, description=f"Quý này ({current_quarter}/{as_of.year})")

    # Check for "quý trước" / "quý rồi"
    if re.search(r"(?:quý trước|quý rồi|quy truoc|quy roi)", q):
        current_quarter = (as_of.month - 1) // 3 + 1
        if current_quarter == 1:
            prev_q, prev_y = 4, as_of.year - 1
        else:
            prev_q, prev_y = current_quarter - 1, as_of.year
        s_date, e_date = get_quarter_range(prev_y, prev_q)
        return TimeRange(s_date, e_date, is_quarter=True, description=f"Quý trước ({prev_q}/{prev_y})")

    # 7. Check for tháng MM/YYYY or tháng MM năm YYYY or thg MM/YYYY
    m_month = re.search(r"(?:tháng|thang|thg)\s+(\d{1,2})\s*(?:/|năm\s+)(\d{4})", q)
    if m_month:
        m_num = int(m_month.group(1))
        m_year = int(m_month.group(2))
        s_date, e_date = get_month_range(m_year, m_num)
        # Cap at as_of_date if this is the current active month
        if m_year == as_of.year and m_num == as_of.month:
            e_date = min(e_date, as_of_date_str)
        return TimeRange(s_date, e_date, is_month=True, description=f"Tháng {m_num}/{m_year}")

    # 8. Check for "tháng này" / "thang nay"
    if re.search(r"\b(?:tháng này|thang nay)\b", q):
        s_date, _ = get_month_range(as_of.year, as_of.month)
        return TimeRange(s_date, as_of_date_str, is_month=True, description=f"Tháng này ({as_of.month}/{as_of.year})")

    # 9. Check for "tháng trước" / "tháng rồi"
    if re.search(r"\b(?:tháng trước|tháng rồi|thang truoc|thang roi)\b", q):
        if as_of.month == 1:
            p_month, p_year = 12, as_of.year - 1
        else:
            p_month, p_year = as_of.month - 1, as_of.year
        s_date, e_date = get_month_range(p_year, p_month)
        return TimeRange(s_date, e_date, is_month=True, description=f"Tháng trước ({p_month}/{p_year})")

    # 10. Check for "tuần này" (Monday to as_of_date)
    if re.search(r"\b(?:tuần này|tuan nay)\b", q):
        # as_of.weekday(): 0 is Monday
        monday = as_of - timedelta(days=as_of.weekday())
        return TimeRange(monday.strftime("%Y-%m-%d"), as_of_date_str, description="Tuần này")

    # 11. Check for "tuần rồi" / "tuần trước" (Monday to Sunday of previous week)
    if re.search(r"\b(?:tuần rồi|tuần trước|tuan roi|tuan truoc)\b", q):
        monday_this = as_of - timedelta(days=as_of.weekday())
        monday_prev = monday_this - timedelta(days=7)
        sunday_prev = monday_prev + timedelta(days=6)
        return TimeRange(monday_prev.strftime("%Y-%m-%d"), sunday_prev.strftime("%Y-%m-%d"), description="Tuần trước")

    # 12. Check for "từ đầu năm" / "năm nay"
    if re.search(r"\b(?:từ đầu năm|tu dau nam|năm nay|nam nay)\b", q):
        s_date = f"{as_of.year:04d}-01-01"
        return TimeRange(s_date, as_of_date_str, description=f"Năm nay ({as_of.year})")

    # 13. Check for "năm YYYY"
    m_year = re.search(r"\bnăm\s+(\d{4})\b", q)
    if m_year:
        yr = int(m_year.group(1))
        s_date = f"{yr:04d}-01-01"
        e_date = f"{yr:04d}-12-31"
        if yr == as_of.year:
            e_date = min(e_date, as_of_date_str)
        return TimeRange(s_date, e_date, description=f"Năm {yr}")

    return None
