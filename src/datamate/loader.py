"""Data loader for DataMate.
Loads CSV, XLSX (including multi-sheet, offset headers, monthly multi-files), and SQLite data.
Cleans Vietnamese dates, number formats, and filters summary rows.
"""

import glob
import os
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import openpyxl
import pandas as pd


@dataclass
class BusinessData:
    business_id: str
    data_dir: str
    readme: str = ""
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    unit_conversion: Optional[pd.DataFrame] = None
    calendar: Optional[pd.DataFrame] = None
    value_index: Optional[Any] = None
    date_range: Tuple[Optional[str], Optional[str]] = (None, None)
    metadata: Dict[str, Any] = field(default_factory=dict)


def normalize_vietnamese_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    return text.strip()


EXCEL_EPOCH = datetime(1899, 12, 30)


def _format_datetime(dt: datetime, keep_time: bool) -> str:
    """'YYYY-MM-DD', or 'YYYY-MM-DD HH:MM:SS' when keep_time and the time is not midnight."""
    if keep_time and (dt.hour or dt.minute or dt.second):
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return dt.strftime("%Y-%m-%d")


def _excel_serial_to_datetime(num: float) -> Optional[datetime]:
    """Excel serial day number (e.g. 45717 = 2025-03-01) -> datetime, only for plausible years (1954-2119)."""
    if 20000 <= num <= 80000:
        return EXCEL_EPOCH + pd.Timedelta(days=float(num)).to_pytimedelta()
    return None


def parse_vietnamese_date(val: Any, default_year: Optional[int] = None, keep_time: bool = False) -> Optional[str]:
    """Parse date from various formats into 'YYYY-MM-DD' (or 'YYYY-MM-DD HH:MM:SS' if keep_time)."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (datetime, pd.Timestamp)):
        return _format_datetime(val, keep_time)
    if isinstance(val, date):
        return val.strftime("%Y-%m-%d")
    if isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(val, bool):
        dt = _excel_serial_to_datetime(float(val))
        # Not a plausible serial (e.g. "Số ngày nợ" = 30): keep the number as is
        return _format_datetime(dt, keep_time) if dt else val

    s = str(val).strip()
    if not s:
        return None

    # Excel serial stored as text: "45748" or "45717.0"
    if re.match(r"^\d{5}(\.\d+)?$", s):
        dt = _excel_serial_to_datetime(float(s))
        if dt:
            return _format_datetime(dt, keep_time)

    # Already YYYY-MM-DD, optionally followed by a time
    m_iso = re.match(r"^(\d{4}-\d{2}-\d{2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?", s)
    if m_iso:
        if keep_time and m_iso.group(2):
            dt = datetime.strptime(m_iso.group(1), "%Y-%m-%d").replace(
                hour=int(m_iso.group(2)), minute=int(m_iso.group(3)), second=int(m_iso.group(4) or 0)
            )
            return _format_datetime(dt, keep_time)
        return m_iso.group(1)

    # Vietnamese format: "1 thg 6 2025" or "15 tháng 1, 2025" or "31 thg 12, 2024"
    m_vn = re.search(r"(\d{1,2})\s+(?:thg|tháng)\s+(\d{1,2})(?:[,\s]+(\d{4}))?", s, re.IGNORECASE)
    if m_vn:
        d = int(m_vn.group(1))
        m = int(m_vn.group(2))
        y = int(m_vn.group(3)) if m_vn.group(3) else default_year
        if y:
            return f"{y:04d}-{m:02d}-{d:02d}"

    # dd/mm/yyyy or dd-mm-yyyy or dd.mm.yyyy, optionally followed by HH:MM[:SS]
    m_dmy = re.match(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?", s)
    if m_dmy:
        d, m, y = int(m_dmy.group(1)), int(m_dmy.group(2)), int(m_dmy.group(3))
        if keep_time and m_dmy.group(4):
            try:
                dt = datetime(y, m, d, int(m_dmy.group(4)), int(m_dmy.group(5)), int(m_dmy.group(6) or 0))
                return _format_datetime(dt, keep_time)
            except ValueError:
                pass
        return f"{y:04d}-{m:02d}-{d:02d}"

    # dd/mm without year
    m_dm = re.match(r"^(\d{1,2})[/.-](\d{1,2})$", s)
    if m_dm and default_year:
        d, m = int(m_dm.group(1)), int(m_dm.group(2))
        return f"{default_year:04d}-{m:02d}-{d:02d}"

    # Fallback to pd.to_datetime
    try:
        ts = pd.to_datetime(s, dayfirst=True)
        return _format_datetime(ts, keep_time)
    except Exception:
        return s


def parse_vietnamese_number(val: Any) -> Any:
    """Parse string number into int or float if possible."""
    if val is None or pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return val

    s = str(val).strip()
    if not s:
        return np.nan

    # Remove currency suffixes first ("250kđ", "1,25tr đ")
    s = re.sub(r"\s*(đồng|đ|vnđ|vnd|\$)\s*$", "", s, flags=re.IGNORECASE).strip()

    # Magnitude suffix: "250k" = 250 000, "1,25tr" / "1.25 triệu" = 1 250 000. With a magnitude
    # suffix the separator is a decimal mark ("1,002tr" = 1 002 000)
    # Shorthand "1tr090" = 1 090 000, "1tr5" = 1 500 000, "2k5" = 2 500 (digits after the unit are its fraction)
    m_short = re.match(r"^(-?\d+)\s*(tr|k)\s*(\d{1,3})$", s, flags=re.IGNORECASE)
    if m_short:
        unit = 1_000_000 if m_short.group(2).lower() == "tr" else 1_000
        frac = int(m_short.group(3).ljust(3, "0")) * unit // 1000
        whole = int(m_short.group(1))
        return whole * unit + (frac if whole >= 0 else -frac)

    multiplier = 1.0
    m_mag = re.match(r"^(-?\d+(?:[.,]\d+)?)\s*(k|nghìn|ngàn|tr|triệu|trieu)$", s, flags=re.IGNORECASE)
    if m_mag:
        multiplier = 1000.0 if m_mag.group(2).lower() in ("k", "nghìn", "ngàn") else 1_000_000.0
        s = m_mag.group(1).replace(",", ".")
    s = re.sub(r"^\s*-\s*", "-", s)  # negative sign

    # Clean dots and commas
    if m_mag:
        pass
    elif re.match(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$", s):
        # dot thousand, comma decimal e.g. 4.820.000 or 4.820.000,50
        s = s.replace(".", "").replace(",", ".")
    elif re.match(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$", s):
        # comma thousand, dot decimal e.g. 4,820,000 or 4,820,000.50
        s = s.replace(",", "")
    elif "," in s and "." not in s:
        # e.g. 12,4 -> 12.4
        s = s.replace(",", ".")

    try:
        num = round(float(s) * multiplier, 6)  # 1.002 * 1e6 -> 1002000.0, not 1001999.9999
        if num.is_integer():
            return int(num)
        return num
    except ValueError:
        return val


# A text cell that is a number written the Vietnamese way: "1.250.000", "1,250,000", "12,5", "250k", "1,25tr", "45.000đ"
NUMBER_TEXT_RE = re.compile(
    r"^-?\s*((\d{1,3}([.,]\d{3})+([.,]\d+)?|\d+([.,]\d+)?)\s*(k|nghìn|ngàn|tr|triệu|trieu)?|\d+\s*(tr|k)\s*\d{1,3})\s*(đồng|đ|vnđ|vnd)?$",
    re.IGNORECASE,
)


def looks_numeric_column(s: pd.Series, min_share: float = 0.98) -> bool:
    """Decide from the values (not the column name) whether a text column holds numbers.
    Codes and phone numbers with leading zeros ('0912...', '0012') are not numbers."""
    vals = s.dropna()
    if vals.empty:
        return False
    texts = [v for v in vals.tolist() if isinstance(v, str)]
    if not texts:
        return False
    uniq = list(dict.fromkeys(t.strip() for t in texts))[:5000]
    if any(re.match(r"^0\d", u) for u in uniq):
        return False
    hits = sum(1 for u in uniq if NUMBER_TEXT_RE.match(u))
    return hits >= max(1, min_share * len(uniq))


def is_date_column_name(col_name: str) -> bool:
    norm = col_name.lower().strip()
    date_keywords = [
        "ngày", "ngay", "date", "thời gian", "thoigian", "time", "hạn dùng", "han_dung",
        "hạn sử dụng", "han_su_dung", "hansudung", "hsd", "kỳ", "ky"
    ]
    return any(k in norm for k in date_keywords)


def is_numeric_column_name(col_name: str) -> bool:
    norm = col_name.lower().strip()
    num_keywords = [
        "giá", "gia", "tiền", "tien", "số lượng", "so_luong", "sl", "quy đổi",
        "quy_doi", "nợ", "no", "chi phí", "chi_phi", "doanh thu", "doanh_thu",
        "thu", "trả", "tra", "tổng", "tong", "thành tiền", "thanh_tien", "chiết khấu"
    ]
    return any(k in norm for k in num_keywords)


EXCEL_ERROR_VALUES = {"#VALUE!", "#N/A", "#REF!", "#DIV/0!", "#NAME?", "#NUM!", "#NULL!"}

DATE_TEXT_RE = re.compile(
    r"^\s*(\d{1,2}[/.-]\d{1,2}[/.-]\d{4}|\d{4}-\d{1,2}-\d{1,2}|\d{1,2}\s+(?:thg|tháng)\s+\d{1,2},?\s+\d{4})"
    r"(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?\s*$", re.IGNORECASE)
DMY_RE = re.compile(r"^\s*(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})")


def looks_date_column(s: pd.Series, min_share: float = 0.95) -> bool:
    """Values, not the column name, decide: 'Sinh nhật', 'Hạn thanh toán', 'TaoLuc', 'Giờ bán' hold dates."""
    texts = [v.strip() for v in s.dropna().tolist() if isinstance(v, str) and v.strip()]
    if len(texts) < 3:
        return False
    uniq = list(dict.fromkeys(texts))[:5000]
    return sum(1 for u in uniq if DATE_TEXT_RE.match(u)) >= min_share * len(uniq)


def month_first_column(s: pd.Series) -> bool:
    """True when a d/m/y-looking column is really m/d/y: some second field > 12 and no first field > 12.
    Decided on the whole column (plan §4.1), never per value."""
    first_gt12 = second_gt12 = False
    for v in pd.unique(s.dropna()):
        m = DMY_RE.match(str(v))
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            first_gt12 |= a > 12
            second_gt12 |= b > 12
    return second_gt12 and not first_gt12


def _swap_day_month(v: Any) -> Any:
    if isinstance(v, str):
        return DMY_RE.sub(lambda m: f"{m.group(2)}/{m.group(1)}/{m.group(3)}", v, count=1)
    return v


def _map_unique(s: pd.Series, fn) -> pd.Series:
    """s.apply(fn), evaluated once per distinct value (keyed by type too: 1 and '1' differ)."""
    cache: Dict[Tuple[type, Any], Any] = {}
    out = []
    for v in s.tolist():
        try:
            key = (type(v), v)
            if key not in cache:
                cache[key] = fn(v)
            out.append(cache[key])
        except TypeError:  # unhashable cell
            out.append(fn(v))
    return pd.Series(out, index=s.index)


SUMMARY_LABEL_RE = r"(tổng|tong|total|cộng|cong|sum)(\s*(cộng|cong|số|so|tiền|tien|kết|ket|chung|cả|ca|all))*\s*[:.]?"


def fill_down_grouped_rows(df: pd.DataFrame, min_share: float = 0.05) -> Tuple[pd.DataFrame, List[str]]:
    """Rulebook §8: 'Ô để trống ở các dòng sau của cùng một nhóm dòng: giống ô đó ở dòng đầu nhóm'.
    Detected from the layout, not from column names: a set of >= 2 columns that are empty together
    on exactly the same rows (>= min_share of the rows, never the first row) is a group header
    (order id, date, state, customer...). Those cells are copied down from the row that opens the
    group. Columns that are also empty elsewhere (e.g. walk-in customers on the first row) are only
    filled on the group-continuation rows. Returns (df, filled columns)."""
    if len(df) < 3:
        return df, []
    na = df.isna()
    counts = na.sum()
    cand = [c for c in df.columns if 0 < counts[c] < len(df)]
    if len(cand) < 2:
        return df, []
    # Continuation rows = the most common exact null-pattern among rows with nulls in `cand`
    patterns = na[cand].apply(lambda r: tuple(r.values), axis=1)
    patterns = patterns[patterns.map(any)]
    if patterns.empty:
        return df, []
    top, n_top = patterns.value_counts().index[0], int(patterns.value_counts().iloc[0])
    group_cols = [c for c, is_na in zip(cand, top) if is_na]
    if len(group_cols) < 2 or n_top < min_share * len(df):
        return df, []
    cont = na[group_cols].all(axis=1)
    if cont.iloc[0]:
        return df, []
    # Columns whose own null mask equals the continuation mask, or contains it
    filled = [c for c in group_cols if (na[c] & cont).sum() == cont.sum()]
    if len(filled) < 2:
        return df, []
    df = df.copy()
    group_id = (~cont).cumsum()
    for c in filled:
        # The opening row's own value, even when empty (ffill would copy the previous order's customer
        # into a walk-in order)
        opener = df[c].where(~cont).groupby(group_id).transform("first")
        df.loc[cont, c] = opener[cont]
    return df, filled


def clean_dataframe(df: pd.DataFrame, default_year: Optional[int] = None, keep_time: bool = False) -> pd.DataFrame:
    """Clean column names, filter summary rows, normalize dates and numbers (fast vectorized)."""
    if df.empty:
        return df

    # Strip and normalize column names
    clean_cols = []
    for c in df.columns:
        c_str = str(c).strip()
        c_str = unicodedata.normalize("NFC", c_str)
        clean_cols.append(c_str)
    df.columns = clean_cols

    # Excel error cells exported as text ('#VALUE!', '#N/A' in a CSV) are not data
    df = df.replace(list(EXCEL_ERROR_VALUES), np.nan)

    # Vectorized drop of nearly-all-null rows
    min_non_null = max(1, min(2, len(df.columns)))
    df = df.dropna(thresh=min_non_null).reset_index(drop=True)
    if df.empty:
        return df

    # Drop header rows repeated inside the data (cells equal to their own column name)
    header_hits = pd.Series(0, index=df.index)
    for c in df.columns:
        header_hits += (df[c].astype(str).str.strip() == c).astype(int)
    is_repeated_header = header_hits >= max(2, (len(df.columns) + 1) // 2)
    if is_repeated_header.any():
        df = df[~is_repeated_header].reset_index(drop=True)

    # Summary/total rows ("TỔNG", "Tổng cộng:", "TOTAL"). The whole cell must be the label, or start
    # with "tổng cộng"/"total" on a sparse row: a prefix test would drop real rows such as customer
    # "Tống Hoàng Linh" written without accents ("Tong Hoang Linh") or a customer named "Công".
    check_cols = df.columns[:min(4, len(df.columns))]
    is_summary = pd.Series(False, index=df.index)
    sparse = df.notna().sum(axis=1) <= max(1, int(len(df.columns) * 0.6))
    for c in check_cols:
        s_str = df[c].astype(str).str.strip().str.lower()
        is_summary |= s_str.str.fullmatch(SUMMARY_LABEL_RE)
        is_summary |= s_str.str.match(r"^(tổng cộng|tong cong|total)\b") & sparse

    if is_summary.any():
        df = df[~is_summary].reset_index(drop=True)

    # Group rows whose order cells are written only on the first line (rulebook §8)
    df, _filled = fill_down_grouped_rows(df)

    # Process date and numeric columns (parsed once per distinct value: dates repeat a lot)
    for col in df.columns:
        if is_date_column_name(col) or looks_date_column(df[col]):
            try:
                src = df[col]
                if month_first_column(src):
                    src = _map_unique(src, _swap_day_month)
                df[col] = _map_unique(src, lambda x: parse_vietnamese_date(x, default_year, keep_time))
            except Exception:
                pass
        elif looks_numeric_column(df[col]):
            # e.g. "Amount" = '1,000,000' or "Tiền bán" = '1,002tr': the values decide, whatever the name
            try:
                df[col] = _map_unique(df[col], parse_vietnamese_number)
            except Exception:
                pass
        elif is_numeric_column_name(col):
            try:
                # Quick sample check before applying to entire column
                sample = df[col].dropna().head(20)
                if not sample.empty:
                    converted_sample = sample.apply(parse_vietnamese_number)
                    num_ratio = sum(isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v) for v in converted_sample) / len(sample)
                    if num_ratio >= 0.7:
                        df[col] = _map_unique(df[col], parse_vietnamese_number)
            except Exception:
                pass

    return df


def _calamine_cell(v: Any) -> Any:
    # Same values as openpyxl(data_only=True): empty -> None, integral floats -> int
    if v == "":
        return None
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def read_workbook_raw(filepath: str) -> Dict[str, List[list]]:
    """All sheets of a workbook as raw cell rows, read once.
    python-calamine is ~10x faster than openpyxl on 300k-row files and returns identical values;
    openpyxl (read_only) is the fallback."""
    try:
        from python_calamine import CalamineWorkbook
        wb = CalamineWorkbook.from_path(filepath)
        return {
            name: [[_calamine_cell(v) for v in row] for row in wb.get_sheet_by_name(name).to_python(skip_empty_area=False)]
            for name in wb.sheet_names
        }
    except ImportError:
        pass
    wb = openpyxl.load_workbook(filepath, data_only=True, read_only=True)
    try:
        return {name: [list(r) for r in wb[name].iter_rows(values_only=True)] for name in wb.sheetnames}
    finally:
        wb.close()


def load_excel_sheet(
    filepath: str,
    sheet_name: Optional[str] = None,
    default_year: Optional[int] = None,
    keep_time: bool = False,
    raw_rows: Optional[List[list]] = None
) -> pd.DataFrame:
    """Load an Excel sheet, detecting offset headers and trailing totals.
    raw_rows: the sheet's cells already read by read_workbook_raw (avoids re-opening the file)."""
    try:
        if raw_rows is None:
            sheets = read_workbook_raw(filepath)
            raw_rows = sheets[sheet_name] if sheet_name else next(iter(sheets.values()), [])
        all_rows = raw_rows
        if not all_rows:
            return pd.DataFrame()

        # Find the header row (search first 10 rows for max non-null strings)
        best_header_idx = 0
        max_valid_cols = 0
        for i in range(min(10, len(all_rows))):
            row = all_rows[i]
            valid_cols = [c for c in row if c is not None and str(c).strip() != ""]
            # Exclude rows that look like report titles (e.g. single long string)
            if len(valid_cols) > max_valid_cols:
                max_valid_cols = len(valid_cols)
                best_header_idx = i

        headers = [str(c).strip() if c is not None else f"col_{idx}" for idx, c in enumerate(all_rows[best_header_idx])]
        # Discard empty trailing column headers if all values are None
        data_rows = all_rows[best_header_idx + 1 :]
        if not data_rows:
            return pd.DataFrame(columns=headers)

        # Pad rows to length of headers
        num_cols = len(headers)
        padded_rows = []
        for r in data_rows:
            r_list = list(r)
            if len(r_list) < num_cols:
                r_list.extend([None] * (num_cols - len(r_list)))
            padded_rows.append(r_list[:num_cols])

        df = pd.DataFrame(padded_rows, columns=headers)
        return clean_dataframe(df, default_year, keep_time)
    except Exception as e:
        # Fallback to standard pandas read_excel
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            return clean_dataframe(df, default_year, keep_time)
        except Exception:
            return pd.DataFrame()


def load_csv_file(filepath: str, default_year: Optional[int] = None, keep_time: bool = False) -> pd.DataFrame:
    """Load CSV with automatic encoding and separator detection."""
    encodings = ["utf-8-sig", "utf-8", "cp1258", "latin1"]
    for enc in encodings:
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(filepath, encoding=enc, sep=sep, low_memory=False)
                if df.shape[1] > 1:
                    return clean_dataframe(df, default_year, keep_time)
            except Exception:
                continue
    # Fallback default
    try:
        df = pd.read_csv(filepath, low_memory=False)
        return clean_dataframe(df, default_year, keep_time)
    except Exception:
        return pd.DataFrame()


def load_sqlite_file(filepath: str, keep_time: bool = False) -> Dict[str, pd.DataFrame]:
    """Load all tables from SQLite file into DataFrames."""
    tables = {}
    conn = sqlite3.connect(filepath)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]
    for t_name in table_names:
        try:
            df = pd.read_sql_query(f'SELECT * FROM "{t_name}"', conn)
            tables[t_name] = clean_dataframe(df, keep_time=keep_time)
        except Exception:
            pass
    conn.close()
    return tables


def extract_year_month_from_filename(filename: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract month and year from names like BanHang_T1_2025.xlsx or BanHang_Thang05_2026.xlsx."""
    m = re.search(r"(?:T|Thang|tháng)_?(\d{1,2})_(\d{4})", filename, re.IGNORECASE)
    if m:
        return int(m.group(2)), int(m.group(1))
    m_year = re.search(r"(\d{4})", filename)
    if m_year:
        return int(m_year.group(1)), None
    return None, None


# Suffixes that split one list into several files: _1, _2026, _T5_2025, _Thang05_2026, _TongHop_2025, _Sheet1
# "_T10-2025", "_T7" (month sheet without year), "_Thang05_2026", "_1", "_Sheet2"
SPLIT_SUFFIX_RE = re.compile(r"(?:[_\- ](?:\d{1,4}|T\d{1,2}(?:[_\-]\d{4})?|Thang\d{1,2}(?:[_\-]\d{4})?|TongHop[_\-]\d{4}|Sheet\d+))+$", re.IGNORECASE)


# A period token anywhere in the name ends the file family: "BanHang_TongHop_2025_BanHang" (sheet of a
# multi-sheet yearly file) and "BanHang_TongHop_2026" (single-sheet file) are both family "BanHang"
PERIOD_TOKEN_RE = re.compile(r"[_\- ](?:TongHop[_\-])?(?:T\d{1,2}[_\-]|Thang\d{1,2}[_\-])?(?:19|20)\d{2}(?=$|[_\- ])", re.IGNORECASE)


def table_stem(name: str) -> str:
    head = PERIOD_TOKEN_RE.split(name, maxsplit=1)[0]
    return SPLIT_SUFFIX_RE.sub("", head) or head or name


def merge_split_tables(tables: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    """Union tables with the same stem and the same set of columns (order may differ).
    The union is stored under the stem name (or '<stem>_all' if the stem is itself a part) and
    the parts are listed in the returned {union name: [parts]} so SQL mode can hide them."""
    groups: Dict[Tuple[str, Tuple[str, ...]], List[str]] = {}
    for name, df in tables.items():
        groups.setdefault((table_stem(name), tuple(sorted(map(str, df.columns)))), []).append(name)
    merged: Dict[str, List[str]] = {}
    for (stem, _cols), parts in groups.items():
        if len(parts) < 2:
            continue
        parts = sorted(parts, key=_natural_key)
        cols = list(tables[parts[0]].columns)
        union = pd.concat([tables[p][cols] for p in parts], ignore_index=True)
        target = stem if stem not in parts else f"{stem}_all"
        while target in tables and target not in parts:
            target += "_all"
        tables[target] = union
        merged[target] = parts
    return merged


def _natural_key(name: str) -> List[Any]:
    """'X_2' < 'X_10'; 'BanHang_T2_2025' < 'BanHang_T10_2025' < 'BanHang_T1_2026' (year before month)."""
    m = re.search(r"T(?:hang)?(\d{1,2})[_\-](\d{4})$", name, re.IGNORECASE)
    if m:
        return [name[: m.start()], int(m.group(2)), int(m.group(1))]
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


_BUSINESS_CACHE: Dict[Tuple[str, bool], BusinessData] = {}


def load_business_data(
    business_dir: str,
    calendar_df: Optional[pd.DataFrame] = None,
    use_cache: bool = True,
    keep_time: bool = False
) -> BusinessData:
    """Load and unify all data files for a specific business.
    keep_time=True keeps 'HH:MM:SS' in datetime columns (needed for hour/shift questions in SQL mode).
    """
    abs_dir = os.path.abspath(business_dir)
    biz_id = os.path.basename(abs_dir)
    cache_key = (abs_dir, keep_time)

    if use_cache and cache_key in _BUSINESS_CACHE:
        return _BUSINESS_CACHE[cache_key]

    bdata = BusinessData(business_id=biz_id, data_dir=abs_dir, calendar=calendar_df)

    # 1. Read README.md if present
    readme_path = os.path.join(abs_dir, "README.md")
    if os.path.exists(readme_path):
        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                bdata.readme = f.read()
        except Exception:
            pass

    # 2. Gather all data files
    all_files = sorted(os.listdir(abs_dir))
    csv_files = [f for f in all_files if f.endswith(".csv")]
    xlsx_files = [f for f in all_files if f.endswith(".xlsx")]
    sqlite_files = [f for f in all_files if f.endswith(".sqlite") or f.endswith(".db")]

    other_tables: Dict[str, pd.DataFrame] = {}

    # Check for SQLite first
    for sq in sqlite_files:
        sq_path = os.path.join(abs_dir, sq)
        sq_tables = load_sqlite_file(sq_path, keep_time=keep_time)
        other_tables.update(sq_tables)

    # Process XLSX files
    for xf in xlsx_files:
        x_path = os.path.join(abs_dir, xf)
        year, month = extract_year_month_from_filename(xf)
        base_name = os.path.splitext(xf)[0]

        try:
            raw_sheets: Dict[Optional[str], Optional[List[list]]] = dict(read_workbook_raw(x_path))
        except Exception:
            raw_sheets = {None: None}

        sheet_frames: List[Tuple[Optional[str], pd.DataFrame]] = []
        for s_name, raw_rows in raw_sheets.items():
            # Rulebook §8: a date without year takes the year of the sheet holding the row
            sheet_year = extract_year_month_from_filename(s_name)[0] if s_name else None
            df = load_excel_sheet(x_path, sheet_name=s_name, default_year=sheet_year or year, keep_time=keep_time,
                                  raw_rows=raw_rows)
            if df.empty:
                continue
            sheet_frames.append((s_name, df))

        # Several sheets with identical columns (e.g. one sheet per month) -> one table named after the file;
        # the "sheet" column keeps the sheet name (it may carry the month or the branch)
        if len(sheet_frames) > 1 and all(list(df.columns) == list(sheet_frames[0][1].columns) for _, df in sheet_frames):
            merged = pd.concat([df.assign(sheet=s_name) for s_name, df in sheet_frames], ignore_index=True)
            sheet_frames = [(None, merged)]

        for s_name, df in sheet_frames:
            # Check if this is unit conversion table
            norm_s = (s_name or "").lower()
            norm_base = base_name.lower()
            if any(k in norm_s or k in norm_base for k in ["quydoi", "quy_doi", "quy đổi", "banggia", "bang_gia"]):
                bdata.unit_conversion = df

            table_key = f"{base_name}_{s_name}" if s_name and len(sheet_frames) > 1 else base_name
            other_tables[table_key] = df

    # Process CSV files
    for cf in csv_files:
        c_path = os.path.join(abs_dir, cf)
        base_name = os.path.splitext(cf)[0]
        year, month = extract_year_month_from_filename(cf)
        df = load_csv_file(c_path, default_year=year, keep_time=keep_time)
        if df.empty:
            continue

        if any(k in base_name.lower() for k in ["quydoi", "quy_doi", "quy đổi"]):
            bdata.unit_conversion = df

        other_tables[base_name] = df

    # A list split into several files ("_1", "_2", "_T5_2025", "_2026", "_TongHop_2025") -> one table.
    # Rows are concatenated as they are: duplicates are NOT dropped here, because rulebook §8 only
    # merges identical lines of the same order (handled by the semantic layer), never two records of
    # a table not kept by order.
    merged_into = merge_split_tables(other_tables)
    if merged_into:
        bdata.metadata["merged_into"] = merged_into

    bdata.tables = other_tables

    # Detect date range across tables
    min_date = None
    max_date = None
    for t_name, df in other_tables.items():
        for col in df.columns:
            if is_date_column_name(col):
                date_vals = df[col].dropna().astype(str)
                valid_dates = [d for d in date_vals if re.match(r"^\d{4}-\d{2}-\d{2}$", d)]
                if valid_dates:
                    cur_min = min(valid_dates)
                    cur_max = max(valid_dates)
                    min_date = cur_min if min_date is None or cur_min < min_date else min_date
                    max_date = cur_max if max_date is None or cur_max > max_date else max_date

    bdata.date_range = (min_date, max_date)

    if use_cache:
        _BUSINESS_CACHE[cache_key] = bdata

    return bdata
