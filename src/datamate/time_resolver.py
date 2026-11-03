"""Deterministic time resolver for DataMate (plan §4.3.2).
Turns Vietnamese time expressions into explicit inclusive date periods, following rulebook §2:
the asked period (primary), the comparison period (compare), ordered sub-periods (periods), a point in time
for "còn lại" questions, shift (ca sáng/chiều) and weekday restrictions.
All patterns run on the accent-stripped, teencode-expanded question (text_norm.normalize_question).
When an expression is seen but cannot be resolved unambiguously, `confident` is False: callers must then
not pass the periods to the LLM as facts.
"""

import calendar as pycalendar
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, List, Optional, Tuple

import pandas as pd

from src.datamate.text_norm import normalize_question


@dataclass
class Period:
    start: date
    end: date            # inclusive
    label: str = ""

    def n_days(self) -> int:
        return (self.end - self.start).days + 1

    def __str__(self) -> str:
        return f"{self.start.isoformat()}..{self.end.isoformat()}" + (f" ({self.label})" if self.label else "")


@dataclass
class TimeSpec:
    primary: Optional[Period]
    compare: Optional[Period] = None
    periods: List[Period] = field(default_factory=list)
    point_in_time: Optional[date] = None
    shift: Optional[str] = None             # 'morning' 06:00-13:59 | 'afternoon' 14:00-21:59
    compare_shift: Optional[str] = None     # baseline shift when two shifts are compared (rulebook: ca sáng)
    weekday: Optional[int] = None           # 1 = Thứ Hai ... 7 = Chủ nhật
    granularity: str = "day"                # 'day' | 'month'
    confident: bool = True
    matched: List[str] = field(default_factory=list)


@dataclass
class TimeRange:
    """Legacy single-range result (kept for agent.py / sql_agent.py)."""
    start_date: str
    end_date: str
    is_exact_day: bool = False
    is_month: bool = False
    is_quarter: bool = False
    description: str = ""


# ── Calendar helpers ─────────────────────────────────────────────────────────

def month_period(year: int, month: int, label: str = "") -> Period:
    last = pycalendar.monthrange(year, month)[1]
    return Period(date(year, month, 1), date(year, month, last), label or f"tháng {month}/{year}")


def quarter_period(year: int, quarter: int) -> Period:
    m0 = (quarter - 1) * 3 + 1
    last = pycalendar.monthrange(year, m0 + 2)[1]
    return Period(date(year, m0, 1), date(year, m0 + 2, last), f"quý {quarter}/{year}")


def year_period(year: int) -> Period:
    return Period(date(year, 1, 1), date(year, 12, 31), f"năm {year}")


def prev_month(year: int, month: int) -> Tuple[int, int]:
    return (year - 1, 12) if month == 1 else (year, month - 1)


def shift_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:  # 29/02
        return d.replace(year=d.year + years, day=28)


def month_to_date(as_of: date) -> Period:
    return Period(as_of.replace(day=1), as_of, "tháng này")


def same_days_prev_month(as_of: date) -> Period:
    """Rulebook §2: tháng này vs tháng trước = day 1 .. same day number of the previous month (capped at month end)."""
    y, m = prev_month(as_of.year, as_of.month)
    last = pycalendar.monthrange(y, m)[1]
    return Period(date(y, m, 1), date(y, m, min(as_of.day, last)), "cùng kỳ tháng trước")


def _safe_date(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _lunar_to_solar(cal: pd.DataFrame, lday: int, lmonth: int, lyear: int) -> Optional[date]:
    rows = cal[(cal["lunar_day"] == lday) & (cal["lunar_month"] == lmonth) & (cal["lunar_year"] == lyear)]
    if "lunar_leap" in rows.columns:
        non_leap = rows[~rows["lunar_leap"].astype(str).str.lower().isin(["true", "1"])]
        rows = non_leap if not non_leap.empty else rows
    if rows.empty:
        return None
    return datetime.strptime(str(rows["date"].iloc[0])[:10], "%Y-%m-%d").date()


def tet_period(year: int, calendar_df: Optional[pd.DataFrame]) -> Optional[Period]:
    """Tết Y: solar date of 23 tháng Chạp (lunar Y-1) .. mùng 10 tháng Giêng (lunar Y)."""
    if calendar_df is None or calendar_df.empty:
        return None
    try:
        start = _lunar_to_solar(calendar_df, 23, 12, year - 1)
        end = _lunar_to_solar(calendar_df, 10, 1, year)
    except Exception:
        return None
    if start is None or end is None:
        return None
    return Period(start, end, f"Tết {year}")


def get_tet_window(year: int, calendar_df: Optional[pd.DataFrame]) -> Optional[Tuple[str, str]]:
    p = tet_period(year, calendar_df)
    return (p.start.isoformat(), p.end.isoformat()) if p else None


def previous_period(p: Period, kind: str) -> Period:
    """'Kỳ trước' of a period: previous month / quarter / week / year, otherwise the same number of days just before."""
    if kind == "month" and p.start.day == 1:
        y, m = prev_month(p.start.year, p.start.month)
        return month_period(y, m)
    if kind == "quarter":
        q = (p.start.month - 1) // 3 + 1
        return quarter_period(p.start.year - 1, 4) if q == 1 else quarter_period(p.start.year, q - 1)
    if kind == "year":
        return year_period(p.start.year - 1)
    n = p.n_days()
    return Period(p.start - timedelta(days=n), p.start - timedelta(days=1), "kỳ trước")


# ── Pattern pieces (accent-stripped text) ────────────────────────────────────

_DMY = r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})"
_SEP_YEAR = r"\s*(?:/|-|nam)\s*(\d{4})"
_WEEKDAYS = {"hai": 1, "2": 1, "ba": 2, "3": 2, "tu": 3, "4": 3, "nam": 4, "5": 4, "sau": 5, "6": 5, "bay": 6, "7": 6}
_GROWTH_RE = re.compile(r"tang truong|thay doi|tang hay giam|tang/giam|hon hay kem|chenh|tang giam|giam hay tang|nhieu hay it hon")


@dataclass
class _Anchor:
    start: int           # char span in the ascii text
    end: int
    period: Period
    kind: str            # day | month | quarter | year | week | range | mtd | ytd | lastn | tet
    relative: str = ""   # 'prev' for 'tháng trước' etc. (resolved against another anchor when comparing)


class _Resolver:
    def __init__(self, a: str, as_of: date, cal: Optional[pd.DataFrame]):
        self.a = a
        self.as_of = as_of
        self.cal = cal
        self.spec = TimeSpec(primary=None)
        self.anchors: List[_Anchor] = []
        self.taken = [False] * len(a)

    # helpers
    def _free(self, s: int, e: int) -> bool:
        return not any(self.taken[s:e])

    def _take(self, s: int, e: int) -> None:
        for i in range(s, e):
            self.taken[i] = True

    def _add(self, m: "re.Match", period: Period, kind: str, relative: str = "", group: int = 0) -> bool:
        s, e = m.span(group)
        if not self._free(s, e):
            return False
        self._take(s, e)
        self.anchors.append(_Anchor(s, e, period, kind, relative))
        self.spec.matched.append(m.group(group))
        return True

    def _unsure(self, what: str) -> None:
        self.spec.confident = False
        self.spec.matched.append(f"?{what}")

    def _year_for_month(self, month: int, day: int = 1) -> int:
        """Rulebook §2: month / quarter / day without a year belongs to this year unless that is after today."""
        y = self.as_of.year
        d = _safe_date(y, month, min(day, pycalendar.monthrange(y, month)[1]))
        return y - 1 if d and d > self.as_of else y

    def _parse_day(self, dd: str, mm: str, yy: Optional[str]) -> Optional[date]:
        d, m = int(dd), int(mm)
        if not (1 <= m <= 12 and 1 <= d <= 31):
            return None
        y = int(yy) if yy else self._year_for_month(m, d)
        return _safe_date(y, m, d)

    def _date_or_relative(self, s: str) -> Optional[date]:
        s = s.strip()
        if re.fullmatch(r"(?:hom|bua) nay|nay|bay gio", s):
            return self.as_of
        if re.fullmatch(r"(?:hom|bua) qua", s):
            return self.as_of - timedelta(days=1)
        m = re.fullmatch(r"(?:ngay\s+)?(\d{1,2})[/.-](\d{1,2})(?:[/.-](\d{4}))?", s)
        if m:
            return self._parse_day(m.group(1), m.group(2), m.group(3))
        return None

    # passes, most specific first
    def run(self) -> TimeSpec:
        a = self.a
        # 'doanh thu tháng sau có luôn cao hơn tháng trước' compares consecutive months of the asked range
        for m in re.finditer(r"\b(?:thang|tuan|quy|nam)\s+sau\b[^?]*?\b(?:thang|tuan|quy|nam)\s+(?:lien\s+)?truoc\b", a):
            self._take(*m.span())
        self._shift_weekday_granularity()
        self._price_change()
        self._tet()
        self._months_lists()
        self._ranges()
        self._point_in_time()
        self._relative_days()
        self._months()
        self._quarters()
        self._years()
        self._weeks()
        self._unresolved_markers()
        self._assign()
        return self.spec

    def _shift_weekday_granularity(self) -> None:
        a, spec = self.a, self.spec
        morning = re.search(r"\bca sang\b", a)
        afternoon = re.search(r"\bca chieu\b", a)
        if morning and afternoon:
            spec.shift, spec.compare_shift = "afternoon", "morning"
            spec.matched.append("ca chiều vs ca sáng")
        elif morning:
            spec.shift = "morning"
            spec.matched.append("ca sáng")
        elif afternoon:
            spec.shift = "afternoon"
            spec.matched.append("ca chiều")
        if re.search(r"\b(?:thang|ky) nao\b", a):
            spec.granularity = "month"
        # 'thứ tư' (Wednesday) and 'thứ tự' (order) are identical without accents: require a day context
        m = re.search(r"\b(?:ngay|cac|moi|vao|nhung)\s+(?:ngay\s+)?thu\s+(hai|ba|tu|nam|sau|bay|[2-7])\b", a)
        if m and not re.search(r"theo\s+(?:dung\s+)?thu tu|thu tu\s+(?:do|sau|nhu|tang|giam)", a[m.start():m.end() + 8]):
            spec.weekday = _WEEKDAYS[m.group(1)]
            spec.matched.append(m.group(0))
        elif re.search(r"\b(?:ngay|cac|moi|vao|nhung)\s+chu nhat\b", a):
            spec.weekday = 7
            spec.matched.append("chủ nhật")

    def _price_change(self) -> None:
        """'So N ngày kể từ ngày D (ngày tăng giá) với N ngày ngay trước': [D, D+N-1] vs [D-N, D-1]."""
        a = self.a
        m = re.search(r"(\d+)\s+ngay\s+(?:ke tu|tinh tu|bat dau tu|dau(?:\s+tien)?)", a)
        if not m:
            return
        n = int(m.group(1))
        rest = a[m.end():m.end() + 30]
        md = re.match(r"\s*(?:ngay\s+)?" + _DMY, rest)
        d = None
        if md:
            d = _safe_date(int(md.group(3)), int(md.group(2)), int(md.group(1)))
        else:
            # Date given elsewhere: 'vào ngày D', 'từ ngày D, X tăng giá', 'ngày D X tăng giá'
            dates = list(re.finditer(r"(?:vao|tu)?\s*ngay\s+" + _DMY, a))
            if len(dates) == 1:
                g = dates[0]
                d = _safe_date(int(g.group(3)), int(g.group(2)), int(g.group(1)))
        if not re.search(r"truoc", a[m.end():]):
            return
        for g in re.finditer(r"(?:(?:vao|tu|ke tu)\s+)?ngay\s+" + _DMY, a):
            self._take(*g.span())
        self._take(*m.span())
        if d is None or n <= 0:
            self._unsure("ngày tăng giá lấy từ dữ liệu")
            self.spec.matched.append(f"{n} ngày kể từ ngày tăng giá vs {n} ngày trước")
            self.price_change_days = n
            return
        primary = Period(d, d + timedelta(days=n - 1), f"{n} ngày kể từ {d.isoformat()}")
        compare = Period(d - timedelta(days=n), d - timedelta(days=1), f"{n} ngày ngay trước {d.isoformat()}")
        self.spec.primary, self.spec.compare = primary, compare
        self.spec.matched.append(primary.label)
        self.fixed = True

    def _tet(self) -> None:
        a = self.a
        # 'N ngày trước Tết' / 'so với trước Tết': same number of days right before 23 tháng Chạp
        mb = re.search(r"(?:(\d+)\s+ngay\s+)?(?:ngay\s+|khoang\s+)?truoc\s+tet\b", a)
        if mb:
            self._take(*mb.span())
        found: List[Tuple["re.Match", Optional[int]]] = []
        for m in re.finditer(r"\btet\b(?:\s+(?:am lich|nguyen dan|co truyen))?(?:\s+(?:nam\s+)?(\d{4}|nay|ngoai|truoc|roi))?", a):
            if not self._free(*m.span()):
                continue
            word = m.group(1)
            if word and word.isdigit():
                year = int(word)
            elif word in ("ngoai", "truoc", "roi"):
                year = self.as_of.year - 1
            elif word == "nay":
                year = self.as_of.year
            else:
                year = None
            found.append((m, year))
        if not found:
            return
        if self.cal is None or self.cal.empty:
            self._unsure("Tết (thiếu calendar.csv)")
            return
        for m, year in found:
            if year is None:
                # 'dịp Tết' without a year: the latest Tết that has started by today
                year = self.as_of.year
                p = tet_period(year, self.cal)
                if p is None or p.start > self.as_of:
                    year -= 1
            p = tet_period(year, self.cal)
            if p is None:
                self._unsure(f"Tết {year} ngoài calendar")
                self._take(*m.span())
                continue
            if p.start > self.as_of:
                self._unsure(f"Tết {year} chưa tới")
            self._add(m, p, "tet")
        tets = [x for x in self.anchors if x.kind == "tet"]
        if mb and tets:
            base = tets[0].period
            n = int(mb.group(1)) if mb.group(1) else base.n_days()
            before = Period(base.start - timedelta(days=n), base.start - timedelta(days=1), f"{n} ngày trước Tết")
            self.before_tet = before
            self.spec.matched.append(before.label)

    def _ranges(self) -> None:
        a = self.a
        # từ đầu tháng (này) đến hết hôm nay; từ đầu năm (tới nay)
        for m in re.finditer(r"(?:tinh\s+)?tu\s+dau\s+thang(?:\s+nay)?(?:\s+(?:den|toi)(?:\s+het)?\s+(?:(?:hom|bua)\s+)?nay)?", a):
            self._add(m, month_to_date(self.as_of), "mtd")
        for m in re.finditer(r"(?:tinh\s+)?tu\s+(?:dau\s+nam|(?:ngay\s+)?0?1/0?1)(?![/\d])(?:\s+(?:nay|nam nay))?(?:\s+(?:den|toi)(?:\s+het)?\s+(?:(?:hom|bua)\s+)?nay)?", a):
            self._add(m, Period(date(self.as_of.year, 1, 1), self.as_of, "từ đầu năm"), "ytd")
        # từ (ngày) X đến (hết) (ngày) Y — dates or 'hôm nay'
        day = r"(?:ngay\s+)?\d{1,2}[/.-]\d{1,2}(?:[/.-]\d{4})?|(?:hom|bua) (?:nay|qua)"
        for m in re.finditer(rf"(?:tu|giua)\s+({day}|nay|bay gio)\s+(?:den|toi|toi|-|–|va)\s+(?:het\s+)?(?:ngay\s+)?({day}|nay)", a):
            s, e = self._date_or_relative(m.group(1)), self._date_or_relative(m.group(2))
            if s and e and s <= e:
                self._add(m, Period(s, e, f"từ {s.isoformat()} đến {e.isoformat()}"), "range")
            else:
                self._unsure(m.group(0))
                self._take(*m.span())
        # từ tháng A(/Y) đến/tới/qua tháng B(/Y): whole months, also listed one by one
        for m in re.finditer(r"tu\s+thang\s+(\d{1,2})(?:" + _SEP_YEAR + r")?\s+(?:den|toi|qua)\s+(?:het\s+)?thang\s+(\d{1,2})(?:" + _SEP_YEAR + r")?", a):
            m1, y1, m2, y2 = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
            if not (1 <= m1 <= 12 and 1 <= m2 <= 12):
                continue
            y2i = int(y2) if y2 else (int(y1) if y1 and m2 >= m1 else None) or self._year_for_month(m2)
            y1i = int(y1) if y1 else (y2i if m1 <= m2 else y2i - 1)
            months = self._month_seq(y1i, m1, y2i, m2)
            if not months:
                continue
            p = Period(months[0].start, months[-1].end, f"tháng {m1}/{y1i}–{m2}/{y2i}")
            if self._add(m, p, "range"):
                self.month_list = months
        # N tháng liên nhau / liên tiếp kết thúc ở tháng M/Y
        m = re.search(r"(\d+)\s+thang\s+(?:lien nhau|lien tiep|gan nhau)[^.?]*?(?:ket thuc|den|toi)\s+(?:o\s+|vao\s+)?thang\s+(\d{1,2})" + _SEP_YEAR, a)
        if m:
            n, mm, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 1 <= mm <= 12 and 1 <= n <= 24:
                months: List[Period] = []
                y, mo = yy, mm
                for _ in range(n):
                    months.insert(0, month_period(y, mo))
                    y, mo = prev_month(y, mo)
                if self._add(m, Period(months[0].start, months[-1].end, f"{n} tháng đến {mm}/{yy}"), "range"):
                    self.month_list = months

    def _month_seq(self, y1: int, m1: int, y2: int, m2: int) -> List[Period]:
        out: List[Period] = []
        y, m = y1, m1
        while (y, m) <= (y2, m2) and len(out) < 36:
            out.append(month_period(y, m))
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        return out

    def _point_in_time(self) -> None:
        a = self.a
        # 'theo tồn kho ngày D', 'tồn kho (lúc) cuối ngày D': state at the end of D
        for m in re.finditer(r"(?:theo\s+)?ton kho\s+(?:luc\s+|vao\s+)?(?:cuoi\s+)?ngay\s+" + _DMY, a):
            d = _safe_date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            if d and self._free(*m.span()):
                self.spec.point_in_time = d
                self.spec.matched.append(m.group(0))
                self._take(*m.span())
        for m in re.finditer(r"(?:(?:tinh\s+)?(?:den|toi)\s+)?(?:het|cuoi)\s+ngay\s+(?:hom nay|" + _DMY + r")|(?:tinh\s+)?(?:den|toi)\s+(?:het\s+)?(?:ngay\s+)?" + _DMY, a):
            if not self._free(*m.span()):
                continue
            if m.group(1):
                d = _safe_date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            elif m.group(4):
                d = _safe_date(int(m.group(6)), int(m.group(5)), int(m.group(4)))
            else:
                d = self.as_of
            if d:
                self.spec.point_in_time = d
                self.spec.matched.append(m.group(0))
                self._take(*m.span())
        for m in re.finditer(r"(?:(?:tinh\s+)?(?:den|toi)\s+)?het\s+(?:hom|bua)\s+nay|cuoi\s+ngay(?:\s+hom\s+nay)?|(?:tinh\s+)?den\s+(?:hom|bua)\s+nay|hom nay[^.?]{0,40}luc cuoi ngay", a):
            if not self._free(*m.span()):
                continue
            self.spec.point_in_time = self.as_of
            self.spec.matched.append(m.group(0))
            self._take(*m.span())

    def _relative_days(self) -> None:
        a = self.a
        for m in re.finditer(r"(?:trong\s+)?(\d+)\s+ngay\s+(?:gan nhat|gan day|vua qua|qua|tro lai day|vua roi|lien nhau gan nhat)", a):
            n = int(m.group(1))
            self._add(m, Period(self.as_of - timedelta(days=n - 1), self.as_of, f"{n} ngày gần nhất"), "lastn")
        for m in re.finditer(r"(?:trong\s+)?(\d+)\s+ngay\s+toi", a):
            n = int(m.group(1))
            # Rulebook (nhà thuốc): 'hết hạn trong N ngày tới' = from today D to the end of D + N
            self._add(m, Period(self.as_of, self.as_of + timedelta(days=n), f"{n} ngày tới"), "range")
        for m in re.finditer(r"(\d+)\s+(tuan|thang)\s+(?:gan nhat|gan day|vua qua|qua)", a):
            if self._free(*m.span()):
                self._unsure(m.group(0))
                self._take(*m.span())
        for m in re.finditer(r"\b(?:hom|bua)\s+nay\b", a):
            self._add(m, Period(self.as_of, self.as_of, "hôm nay"), "day")
        for m in re.finditer(r"\b(?:hom|bua)\s+qua\b", a):
            d = self.as_of - timedelta(days=1)
            self._add(m, Period(d, d, "hôm qua"), "day")
        for m in re.finditer(r"\bhom\s+kia\b", a):
            d = self.as_of - timedelta(days=2)
            self._add(m, Period(d, d, "hôm kia"), "day")
        for m in re.finditer(r"(?:ngay\s+)?" + _DMY, a):
            d = _safe_date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            if d:
                self._add(m, Period(d, d, d.isoformat()), "day")
        for m in re.finditer(r"ngay\s+(\d{1,2})[/.-](\d{1,2})(?![\d/.-])", a):
            d = self._parse_day(m.group(1), m.group(2), None)
            if d:
                self._add(m, Period(d, d, d.isoformat()), "day")

    def _months_lists(self) -> None:
        """'(3 tháng) 4/2026, 5/2026, 6/2026' or 'tháng 11/2025 qua tháng 12/2025 rồi tháng 1/2026': ordered months."""
        a = self.a
        m = re.search(r"thang\s+(\d{1,2}/\d{4}(?:\s*(?:,|va|roi|r|qua|->|→|den|sang)\s*(?:thang\s+)?\d{1,2}/\d{4})+)", a)
        if not m or not self._free(*m.span()):
            return
        months = []
        for mm, yy in re.findall(r"(\d{1,2})/(\d{4})", m.group(1)):
            if 1 <= int(mm) <= 12:
                months.append(month_period(int(yy), int(mm)))
        if len(months) == 2 and re.search(r"\btu\s+$", a[max(0, m.start() - 4):m.start()]) and not re.search(r",|\bva\b", m.group(1)):
            return  # 'từ tháng A qua tháng B' is a range of months, handled by _ranges
        if len(months) >= 2:
            self.month_list = months
            lo, hi = min(p.start for p in months), max(p.end for p in months)
            self._add(m, Period(lo, hi, "các tháng " + ", ".join(p.label for p in months)), "range")

    def _months(self) -> None:
        a = self.a
        for m in re.finditer(r"\bthang\s+(\d{1,2})(?:" + _SEP_YEAR + r")?(?![\d/])", a):
            mm = int(m.group(1))
            if not 1 <= mm <= 12:
                continue
            yy = int(m.group(2)) if m.group(2) else self._year_for_month(mm)
            self._add(m, month_period(yy, mm), "month")
        for m in re.finditer(r"\bthang\s+nay\b", a):
            self._add(m, month_to_date(self.as_of), "mtd")
        for m in re.finditer(r"\bthang\s+(?:lien\s+)?(?:truoc|roi)(?:\s+do)?\b|\bthang\s+lien\s+ke\s+truoc\b", a):
            y, mo = prev_month(self.as_of.year, self.as_of.month)
            self._add(m, month_period(y, mo, "tháng trước"), "month", relative="prev")

    def _quarters(self) -> None:
        a = self.a
        for m in re.finditer(r"\bquy\s+([1-4])(?:" + _SEP_YEAR + r")?(?![\d/])", a):
            q = int(m.group(1))
            if m.group(2):
                yy = int(m.group(2))
            else:
                yy = self.as_of.year
                if quarter_period(yy, q).start > self.as_of:
                    yy -= 1
            self._add(m, quarter_period(yy, q), "quarter")
        q_now = (self.as_of.month - 1) // 3 + 1
        for m in re.finditer(r"\bquy\s+nay\b", a):
            p = quarter_period(self.as_of.year, q_now)
            self._add(m, Period(p.start, self.as_of, "quý này"), "quarter")
        for m in re.finditer(r"\bquy\s+(?:truoc|roi)\b", a):
            p = quarter_period(self.as_of.year - 1, 4) if q_now == 1 else quarter_period(self.as_of.year, q_now - 1)
            self._add(m, p, "quarter", relative="prev")

    def _years(self) -> None:
        a = self.a
        for m in re.finditer(r"\bnam\s+(\d{4})\b", a):
            self._add(m, year_period(int(m.group(1))), "year")
        for m in re.finditer(r"\bnam\s+nay\b", a):
            self._add(m, Period(date(self.as_of.year, 1, 1), self.as_of, "năm nay"), "ytd")
        for m in re.finditer(r"\bnam\s+(?:ngoai|truoc|roi|qua)\b", a):
            self._add(m, year_period(self.as_of.year - 1), "year", relative="prev")

    def _weeks(self) -> None:
        a = self.a
        monday = self.as_of - timedelta(days=self.as_of.weekday())
        for m in re.finditer(r"\btuan\s+nay\b", a):
            self._add(m, Period(monday, self.as_of, "tuần này"), "week")
        for m in re.finditer(r"\btuan\s+(?:truoc|roi|vua roi|lien truoc)\b", a):
            self._add(m, Period(monday - timedelta(days=7), monday - timedelta(days=1), "tuần trước"), "week", relative="prev")
        for m in re.finditer(r"\btuan\s+qua\b", a):
            if self._free(*m.span()):
                # Not defined by the rulebook (previous week or last 7 days?)
                self._unsure("tuần qua")
                self._add(m, Period(monday - timedelta(days=7), monday - timedelta(days=1), "tuần qua"), "week", relative="prev")

    def _unresolved_markers(self) -> None:
        """Expressions with no deterministic meaning in the rulebook."""
        a = self.a
        for pat in (r"ngay luong", r"dau thang(?!\s+nay)", r"cuoi thang", r"giua thang", r"dip le", r"trung thu", r"cuoi tuan",
                    r"gio vang", r"mua (?:he|mua|dong|thu)", r"(?:thang|nam|tuan)\s+sau\b", r"ngay mai"):
            m = re.search(pat, a)
            if m and self._free(*m.span()):
                self._unsure(m.group(0))

    # combination
    def _assign(self) -> None:
        spec, a = self.spec, self.a
        if getattr(self, "fixed", False):
            return
        anchors = sorted(self.anchors, key=lambda x: x.start)
        if getattr(self, "month_list", None):
            spec.periods = list(self.month_list)
            # 'có phải tháng 10 cao hơn tháng 9 ...' re-mentions months of the list: they are not new periods
            listed = {(p.start, p.end) for p in spec.periods}
            anchors = [x for x in anchors if (x.period.start, x.period.end) not in listed]
            if len(spec.periods) == 2 and _GROWTH_RE.search(a):
                # 'từ tháng 8 qua tháng 9 doanh thu tăng bao nhiêu %': later month vs earlier month
                spec.primary, spec.compare = spec.periods[1], spec.periods[0]
                return
            if len(anchors) == 1 or all(x.kind == "range" for x in anchors):
                spec.primary = anchors[0].period if anchors else Period(spec.periods[0].start, spec.periods[-1].end, "các tháng")
                return
            if not anchors:
                spec.primary = Period(min(p.start for p in spec.periods), max(p.end for p in spec.periods), "các tháng")
                return
        if not anchors:
            return
        compare_idx = None
        # The anchor right after 'so (sánh) với / vs / với / hơn hay kém / so' is the comparison period
        for i, anc in enumerate(anchors):
            before = a[max(0, anc.start - 22):anc.start]
            if re.search(r"(?:\bso(?:\s+sanh)?(?:\s+voi)?|\bvoi|\bvs|hon hay kem|hon|kem|chenh(?:\s+lech)?(?:\s+voi)?)\s+(?:(?:cung ky|dip|ky|khoang|giai doan)\s+)?$", before) and len(anchors) > 1:
                compare_idx = i
        if compare_idx is None and len(anchors) == 2 and anchors[1].relative == "prev" and anchors[0].relative != "prev":
            compare_idx = 1
        if compare_idx is None and len(anchors) == 2 and anchors[0].relative == "prev" and anchors[1].relative != "prev" and re.search(r"\bso\b|\bvoi\b", a):
            compare_idx = 0
        primary_anchors: List[_Anchor] = []
        for i, x in enumerate(anchors):
            if i != compare_idx and not any((y.period.start, y.period.end) == (x.period.start, x.period.end) for y in primary_anchors):
                primary_anchors.append(x)
        cmp_anchor = anchors[compare_idx] if compare_idx is not None else None

        # Primary: several explicit periods without comparison -> their span (ordered list kept in `periods`)
        if len(primary_anchors) == 1:
            prim = primary_anchors[0]
            spec.primary = prim.period
        else:
            kinds = {x.kind for x in primary_anchors}
            spec.primary = Period(min(x.period.start for x in primary_anchors), max(x.period.end for x in primary_anchors), "nhiều kỳ")
            if not spec.periods and kinds <= {"month"}:
                spec.periods = [x.period for x in primary_anchors]
            elif not spec.periods:
                # Different granularities mentioned together: the intent is not clear
                nested = all(any(o is not x and o.period.start <= x.period.start and x.period.end <= o.period.end
                                 for o in primary_anchors) or x.period.n_days() == max(y.period.n_days() for y in primary_anchors)
                             for x in primary_anchors)
                if nested:
                    widest = max(primary_anchors, key=lambda x: x.period.n_days())
                    narrow = [x for x in primary_anchors if x is not widest]
                    spec.primary = min(narrow, key=lambda x: x.period.n_days()).period if narrow else widest.period
                    spec.matched.append("?nested periods")
                spec.confident = False
            prim = primary_anchors[0]

        if cmp_anchor is not None:
            spec.compare = self._resolve_compare(prim, cmp_anchor)
        cung_ky_year = re.search(r"cung (?:ky|thang|quy|tuan|thoi diem)\s+(?:cua\s+|nay\s+)?nam\s+(?:ngoai|truoc|roi|qua)", a)
        cung_ky_month = re.search(r"cung ky\s+(?:cua\s+)?thang\s+(?:truoc|roi|lien truoc)", a)
        if cung_ky_year and spec.primary:
            p = spec.primary
            spec.compare = Period(shift_years(p.start, -1), shift_years(p.end, -1), "cùng kỳ năm trước")
        elif cung_ky_month and spec.primary:
            spec.compare = self._resolve_compare(prim, _Anchor(0, 0, month_period(*prev_month(self.as_of.year, self.as_of.month)), "month", "prev"))
        if getattr(self, "before_tet", None) is not None and spec.primary is not None:
            spec.compare = self.before_tet
        if spec.compare is None and spec.primary is not None and spec.compare_shift is None and _GROWTH_RE.search(a) \
                and not spec.periods and re.search(r"%|phan tram|bao nhieu|may|nhieu", a):
            # Rulebook §1: percent change / difference is versus the previous period (kỳ trước)
            spec.compare = self._previous_of(prim)
            spec.matched.append("kỳ trước (mặc định)")
        if spec.compare_shift and spec.compare is None and spec.primary is not None:
            spec.compare = Period(spec.primary.start, spec.primary.end, "ca sáng")
        if spec.primary and spec.point_in_time and spec.primary.start == spec.primary.end == spec.point_in_time:
            spec.primary = None  # 'hết hôm nay' state question: no flow period

    def _previous_of(self, prim: _Anchor) -> Period:
        if prim.kind == "mtd":
            return same_days_prev_month(self.as_of)
        if prim.kind == "tet":
            p = tet_period(prim.period.start.year + 1, self.cal) if self.cal is not None else None
            yr = int(prim.period.label.split()[-1]) if prim.period.label.split()[-1].isdigit() else None
            prev = tet_period(yr - 1, self.cal) if yr else None
            if prev:
                return prev
        if prim.kind == "ytd":
            return Period(shift_years(prim.period.start, -1), shift_years(prim.period.end, -1), "cùng kỳ năm trước")
        kind = {"month": "month", "quarter": "quarter", "year": "year"}.get(prim.kind, "days")
        if prim.kind == "week":
            return Period(prim.period.start - timedelta(days=7), prim.period.start - timedelta(days=1), "tuần trước đó")
        return previous_period(prim.period, kind)

    def _resolve_compare(self, prim: _Anchor, cmp_anchor: _Anchor) -> Period:
        # 'tháng trước / tháng liền trước / tháng trước đó' is relative to the asked period, not to today
        if cmp_anchor.relative == "prev" and cmp_anchor.kind == prim.kind or (cmp_anchor.relative == "prev" and prim.kind == "mtd"):
            return self._previous_of(prim)
        if cmp_anchor.relative == "prev" and prim.kind == "tet":
            return self._previous_of(prim)
        return cmp_anchor.period


def resolve_time_spec(text_or_raw_question: str, as_of: Any, calendar_df: Optional[pd.DataFrame] = None) -> TimeSpec:
    """Resolve all time expressions of a question (raw or already normalized)."""
    if isinstance(as_of, str):
        as_of = datetime.strptime(as_of[:10], "%Y-%m-%d").date()
    elif isinstance(as_of, datetime):
        as_of = as_of.date()
    _, ascii_text = normalize_question(text_or_raw_question)
    # 'thang 5 / 2026' -> 'thang 5/2026'; en dash ranges
    ascii_text = re.sub(r"(\d)\s*/\s*(\d)", r"\1/\2", ascii_text)
    return _Resolver(ascii_text, as_of, calendar_df).run()


def resolve_time(question: str, as_of_date_str: str, calendar_df: Optional[pd.DataFrame] = None) -> Optional[TimeRange]:
    """Legacy wrapper: the primary period as a TimeRange, or None when absent or not confidently resolved."""
    if not as_of_date_str:
        return None
    try:
        spec = resolve_time_spec(question, as_of_date_str, calendar_df)
    except ValueError:
        return None
    p = spec.primary
    if p is None or not spec.confident:
        return None
    label = p.label or ""
    return TimeRange(
        start_date=p.start.isoformat(),
        end_date=p.end.isoformat(),
        is_exact_day=p.start == p.end,
        is_month=label.startswith("tháng"),
        is_quarter=label.startswith("quý"),
        description=label,
    )
