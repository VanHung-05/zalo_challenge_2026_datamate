"""Rulebook loader and extractor for DataMate (plan §4.2).
Dynamically reads rulebook.md from the active data split to avoid hard-coding.
The full text goes into the LLM prompt; the key defaults are also parsed into RuleParams for the
semantic layer. A parameter whose sentence is not found in the split's rulebook is marked
uncertain (never silently assumed): questions depending on it lose confidence (plan §4.2).
"""

import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class RuleEntry:
    section: str      # "1. Đơn hàng và doanh thu" / "6. Theo ngành > Cà phê"
    key: str          # left column of the rule table
    text: str         # right column


@dataclass
class RuleParam:
    value: Any
    found: bool        # True: matched a sentence of this split's rulebook; False: fallback default
    source: str = ""   # the rule text it was read from


# Fallbacks are the train rulebook's defaults; they are only used with found=False
PARAM_SPECS: List[Tuple[str, Any, str, str]] = [
    # name, fallback, rule-key regex, value regex (searched in the rule text; '' = presence of the key is enough)
    ("revenue_excludes_cancelled", True, r"đơn tính vào doanh thu", r"không\s*huỷ|không\s*hủy"),
    ("revenue_excludes_returned_whole_order", True, r"đơn tính vào doanh thu", r"trả một phần.*bỏ\s*\**cả đơn"),
    ("revenue_includes_delivering", True, r"đơn tính vào doanh thu", r"đang giao vẫn tính"),
    ("revenue_subtracts_order_discount", False, r"doanh thu một đơn", r"\**không\**\s*trừ giảm giá trên cả đơn"),
    ("revenue_includes_vat", False, r"doanh thu một đơn", r"\**không\**\s*cộng thuế"),
    ("revenue_includes_shipping", False, r"doanh thu một đơn", r"\**không\**\s*gồm phí giao"),
    ("order_period_by", "order_date", r"đơn thuộc kỳ nào", r"theo\s*\**ngày đặt"),
    ("cancel_rate_denominator", "all_orders", r"tỉ lệ đơn huỷ|tỉ lệ đơn hủy", r"mọi đơn"),
    ("change_vs_previous_period", True, r"phần trăm thay đổi", r"kỳ trước"),
    ("week_start", "monday", r"^tuần$", r"thứ hai"),
    ("best_seller_metric", "quantity_base_unit", r"bán chạy", r"số lượng quy về đơn vị cơ bản"),
    ("top_staff_metric", "revenue", r"nhân viên bán giỏi", r"theo\s*\**doanh thu"),
    ("big_spender_min_orders", 3, r"khách sộp", r"từ\s*(\d+)\s*đơn"),
    ("loyal_customer_metric", "purchase_days", r"khách trung thành", r"nhiều ngày có mua"),
    ("favourite_item_metric", "order_count", r"được ưa chuộng", r"nhiều đơn nhất"),
    ("peak_hour_bucket", "1h_start_hour", r"giờ cao điểm", r"một giờ tròn"),
    ("tie_break", "code_asc", r"", r"bằng nhau.*mã tăng dần|theo\s*\**mã\**\s*tăng dần"),
    ("money_rounding", "half_up", r"giá trị trung bình một đơn", r"nửa đồng làm tròn lên"),
    ("cost_method", "wavg_lot", r"^giá vốn$", r"bình quân gia quyền"),
    ("debt_includes_returned", True, r"đơn được tính vào nợ", r"trả hàng vẫn tính"),
    ("debt_age_from", "order_date", r"tuổi nợ", r"từ\s*\**ngày đặt"),
]

SHIFT_RE = re.compile(r"ca sáng\s*(\d{1,2}):(\d{2})\s*[–-]\s*(\d{1,2}):(\d{2}).*?ca chiều\s*(\d{1,2}):(\d{2})\s*[–-]\s*(\d{1,2}):(\d{2})")
TET_RE = re.compile(r"(\d{1,2})\s*tháng\s*Chạp.*?mùng\s*(\d{1,2})\s*tháng\s*Giêng", re.IGNORECASE | re.DOTALL)


def _clean_cell(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("**", "").replace("`", "")).strip()


class Rulebook:
    def __init__(self, rulebook_path: Optional[str] = None):
        self.path = rulebook_path
        self.raw_text = ""
        self.sections: Dict[str, str] = {}
        self.entries: List[RuleEntry] = []
        self.params: Dict[str, RuleParam] = {}
        if rulebook_path and os.path.exists(rulebook_path):
            try:
                with open(rulebook_path, "r", encoding="utf-8") as f:
                    self.raw_text = unicodedata.normalize("NFC", f.read())
                self._parse_sections()
                self._parse_tables()
            except Exception as e:
                print(f"[Rules Warning] Failed to read {rulebook_path}: {e}")
        self._parse_params()

    def _parse_sections(self):
        current_section = "intro"
        current_lines = []
        for line in self.raw_text.splitlines():
            if line.startswith("## "):
                if current_lines:
                    self.sections[current_section] = "\n".join(current_lines).strip()
                current_section = line[3:].strip()
                current_lines = [line]
            else:
                current_lines.append(line)
        if current_lines:
            self.sections[current_section] = "\n".join(current_lines).strip()

    def _parse_tables(self):
        """Every 2-column markdown table row `| key | text |` -> RuleEntry (headers and separators skipped)."""
        section = sub = ""
        for line in self.raw_text.splitlines():
            if line.startswith("## "):
                section, sub = line[3:].strip(), ""
                continue
            if line.startswith("### "):
                sub = line[4:].strip()
                continue
            if not line.startswith("|"):
                continue
            cells = [c for c in line.strip().strip("|").split("|")]
            if len(cells) < 2 or set("".join(cells).strip()) <= set("-: "):
                continue
            key, text = _clean_cell(cells[0]), _clean_cell("|".join(cells[1:]))
            if key.lower() in ("luật", "cách nói", "cách ghi", "từ", "type"):
                continue
            self.entries.append(RuleEntry(f"{section} > {sub}" if sub else section, key, text))

    def find(self, key_pattern: str) -> List[RuleEntry]:
        return [e for e in self.entries if re.search(key_pattern, e.key, re.IGNORECASE)]

    def _parse_params(self):
        # Rules written as bullet paragraphs (e.g. §9 "Bằng nhau") are searched in the whole text
        paragraphs = [RuleEntry("text", "", _clean_cell(p)) for p in re.split(r"\n\s*\n|\n- ", self.raw_text) if p.strip()]
        for name, fallback, key_re, value_re in PARAM_SPECS:
            candidates = self.find(key_re) if key_re else self.entries + paragraphs
            param = RuleParam(fallback, False)
            for e in candidates:
                m = re.search(value_re, e.text, re.IGNORECASE)
                if m:
                    value = int(m.group(1)) if m.groups() and isinstance(fallback, int) and not isinstance(fallback, bool) else fallback
                    param = RuleParam(value, True, e.text)
                    break
            self.params[name] = param

        shift = RuleParam({"morning": ("06:00", "13:59"), "afternoon": ("14:00", "21:59")}, False)
        for e in self.find(r"ca sáng"):
            m = SHIFT_RE.search(e.text)
            if m:
                g = m.groups()
                shift = RuleParam({"morning": (f"{int(g[0]):02d}:{g[1]}", f"{int(g[2]):02d}:{g[3]}"),
                                   "afternoon": (f"{int(g[4]):02d}:{g[5]}", f"{int(g[6]):02d}:{g[7]}")}, True, e.text)
        self.params["shifts"] = shift

        tet = RuleParam({"start_lunar_day": 23, "start_lunar_month": 12, "end_lunar_day": 10, "end_lunar_month": 1}, False)
        for e in self.find(r"tết"):
            m = TET_RE.search(e.text)
            if m:
                tet = RuleParam({"start_lunar_day": int(m.group(1)), "start_lunar_month": 12,
                                 "end_lunar_day": int(m.group(2)), "end_lunar_month": 1}, True, e.text)
        self.params["tet_window"] = tet

    def value(self, name: str) -> Any:
        return self.params[name].value

    def uncertain_params(self) -> List[str]:
        return [k for k, p in self.params.items() if not p.found]

    def industry_sections(self) -> Dict[str, str]:
        """'### <ngành>' blocks of section 6 (industry rules override the general ones)."""
        out: Dict[str, str] = {}
        for e in self.entries:
            if " > " in e.section:
                ind = e.section.split(" > ", 1)[1]
                out[ind] = out.get(ind, "") + f"- {e.key}: {e.text}\n"
        return out

    def to_json(self) -> Dict[str, Any]:
        return {"path": self.path, "params": {k: asdict(v) for k, v in self.params.items()},
                "uncertain": self.uncertain_params(), "n_entries": len(self.entries)}

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=1, default=str)


_RULEBOOK_CACHE: Dict[str, Rulebook] = {}


def find_rulebook_path(start_dir: str, max_up: int = 2) -> Optional[str]:
    """Locate rulebook.md of the split: <split>/rulebook.md.
    Accepts the split dir itself or a business dir (<split>/data/<business_id>), searching upwards.
    """
    cur = os.path.abspath(start_dir)
    for _ in range(max_up + 1):
        candidate = os.path.join(cur, "rulebook.md")
        if os.path.exists(candidate):
            return candidate
        cur = os.path.dirname(cur)
    return None


def get_rulebook(data_dir: str) -> Rulebook:
    rb_path = find_rulebook_path(data_dir)
    key = rb_path or ""
    if key not in _RULEBOOK_CACHE:
        if rb_path is None:
            print(f"[Rules Warning] rulebook.md not found from {data_dir}")
        _RULEBOOK_CACHE[key] = Rulebook(rb_path)
    return _RULEBOOK_CACHE[key]


# ── calendar.csv ──────────────────────────────────────────────────────────────

CALENDAR_COLUMNS = ["date", "weekday", "lunar_day", "lunar_month", "lunar_year", "lunar_leap", "holiday"]


def load_calendar(split_dir: str) -> Optional[pd.DataFrame]:
    """Read <split>/calendar.csv and validate it; returns None (with a warning) if unusable."""
    path = os.path.join(split_dir, "calendar.csv")
    if not os.path.exists(path):
        return None
    try:
        cal = pd.read_csv(path)
    except Exception as e:
        print(f"[Calendar Warning] cannot read {path}: {e}")
        return None
    problems = validate_calendar(cal)
    if problems:
        print(f"[Calendar Warning] {path}: {'; '.join(problems)}")
    return cal


def validate_calendar(cal: pd.DataFrame) -> List[str]:
    """Structural checks: required columns, one row per consecutive solar day, weekday consistent
    with the date, and each lunar new year (mùng 1 tháng Giêng) present exactly once per lunar year."""
    problems = [f"missing column {c}" for c in CALENDAR_COLUMNS if c not in cal.columns]
    if problems:
        return problems
    dates = pd.to_datetime(cal["date"], errors="coerce")
    if dates.isna().any():
        problems.append(f"{int(dates.isna().sum())} unparsable dates")
        return problems
    if not (dates.diff().dropna() == pd.Timedelta(days=1)).all():
        problems.append("dates are not consecutive days")
    if not ((dates.dt.day_of_week + 1) == cal["weekday"]).all():
        problems.append("weekday does not match date (expected 1 = Monday)")
    new_years = cal[(cal["lunar_day"] == 1) & (cal["lunar_month"] == 1) & (~cal["lunar_leap"].astype(bool))]
    if new_years["lunar_year"].duplicated().any():
        problems.append("a lunar year has several new-year days")
    return problems


def lunar_to_solar(cal: pd.DataFrame, lunar_year: int, lunar_month: int, lunar_day: int) -> Optional[str]:
    """Solar 'YYYY-MM-DD' of a (non-leap) lunar date, or None if outside the calendar."""
    rows = cal[(cal["lunar_year"] == lunar_year) & (cal["lunar_month"] == lunar_month)
               & (cal["lunar_day"] == lunar_day) & (~cal["lunar_leap"].astype(bool))]
    return str(rows["date"].iloc[0])[:10] if len(rows) else None


def holiday_dates(cal: pd.DataFrame, name: str) -> List[str]:
    """Solar dates whose `holiday` column equals `name` (e.g. 'Trung thu', 'Tết')."""
    hol = cal["holiday"].astype("string").str.strip()
    return [str(d)[:10] for d in cal.loc[hol == name, "date"]]
