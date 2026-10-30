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

    # Check for unit suffix like "k", "K"
    multiplier = 1.0
    if re.search(r"\d+\s*[kK]$", s):
        multiplier = 1000.0
        s = re.sub(r"[kK]$", "", s).strip()

    # Remove currency / quantity suffixes: đ, vnđ, vnd, ly, chai, lon, kg, m, cái, bộ
    s = re.sub(r"(đ|vnđ|vnd|\$)\s*$", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^\s*-\s*", "-", s)  # negative sign

    # Clean dots and commas
    if re.match(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$", s):
        # dot thousand, comma decimal e.g. 4.820.000 or 4.820.000,50
        s = s.replace(".", "").replace(",", ".")
    elif re.match(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$", s):
        # comma thousand, dot decimal e.g. 4,820,000 or 4,820,000.50
        s = s.replace(",", "")
    elif "," in s and "." not in s:
        # e.g. 12,4 -> 12.4
        s = s.replace(",", ".")

    try:
        num = float(s) * multiplier
        if num.is_integer():
            return int(num)
        return num
    except ValueError:
        return val


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

    # Vectorized filter for summary/total rows ("TỔNG", "TOTAL", etc.)
    summary_keywords = ("tổng", "tong", "total", "cộng", "cong")
    check_cols = df.columns[:min(4, len(df.columns))]
    is_summary = pd.Series(False, index=df.index)
    for c in check_cols:
        s_str = df[c].astype(str).str.strip().str.lower()
        is_summary |= s_str.str.startswith(summary_keywords)

    if is_summary.any():
        df = df[~is_summary].reset_index(drop=True)

    # Process date and numeric columns
    for col in df.columns:
        if is_date_column_name(col):
            try:
                df[col] = df[col].apply(lambda x: parse_vietnamese_date(x, default_year, keep_time))
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
                        df[col] = df[col].apply(parse_vietnamese_number)
            except Exception:
                pass

    return df


def _rows_to_dataframe(all_rows: List[tuple], default_year: Optional[int] = None, keep_time: bool = False) -> pd.DataFrame:
    """Raw sheet rows -> cleaned DataFrame, detecting offset headers and trailing totals."""
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


def load_excel_file(
    filepath: str,
    default_year: Optional[int] = None,
    keep_time: bool = False
) -> List[Tuple[Optional[str], pd.DataFrame]]:
    """Load every sheet of a workbook as (sheet_name, DataFrame), opening the file once.
    openpyxl read-only mode scans every sheet's XML on open, so opening per sheet is quadratic in sheet count.
    Rulebook §8: a date without year takes the year of the sheet holding the row, else the file's year.
    """
    try:
        wb = openpyxl.load_workbook(filepath, data_only=True, read_only=True)
    except Exception:
        # Fallback to standard pandas read_excel (first sheet)
        try:
            return [(None, clean_dataframe(pd.read_excel(filepath), default_year, keep_time))]
        except Exception:
            return []

    sheets: List[Tuple[Optional[str], pd.DataFrame]] = []
    try:
        for s_name in wb.sheetnames:
            sheet_year = extract_year_month_from_filename(s_name)[0] or default_year
            try:
                df = _rows_to_dataframe(list(wb[s_name].iter_rows(values_only=True)), sheet_year, keep_time)
            except Exception:
                try:
                    df = clean_dataframe(pd.read_excel(filepath, sheet_name=s_name), sheet_year, keep_time)
                except Exception:
                    df = pd.DataFrame()
            sheets.append((s_name, df))
    finally:
        wb.close()
    return sheets


def load_excel_sheet(
    filepath: str,
    sheet_name: Optional[str] = None,
    default_year: Optional[int] = None,
    keep_time: bool = False
) -> pd.DataFrame:
    """Load one Excel sheet (first sheet if sheet_name is None). Use load_excel_file to read all sheets."""
    try:
        wb = openpyxl.load_workbook(filepath, data_only=True, read_only=True)
        try:
            ws = wb[sheet_name if sheet_name else wb.sheetnames[0]]
            all_rows = list(ws.iter_rows(values_only=True))
        finally:
            wb.close()
        return _rows_to_dataframe(all_rows, default_year, keep_time)
    except Exception:
        # Fallback to standard pandas read_excel
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name or 0)
            return clean_dataframe(df, default_year, keep_time)
        except Exception:
            return pd.DataFrame()


def load_csv_file(filepath: str, default_year: Optional[int] = None, keep_time: bool = False) -> pd.DataFrame:
    """Load CSV with automatic encoding and separator detection."""
    encodings = ["utf-8-sig", "utf-8", "cp1258", "latin1"]
    for enc in encodings:
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(filepath, encoding=enc, sep=sep)
                if df.shape[1] > 1:
                    return clean_dataframe(df, default_year, keep_time)
            except Exception:
                continue
    # Fallback default
    try:
        df = pd.read_csv(filepath)
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


# Suffix that splits one table over files: _T1_2025, _Thang01_2026, _1, _2, _2026
SPLIT_FILE_SUFFIX_RE = re.compile(r"(?:_(?:T|Thang)\d{1,2})?(?:_\d+)?$", re.IGNORECASE)


def split_file_stem(base_name: str) -> str:
    """BanHang_T1_2025 -> BanHang, So_chi_tiet_ban_hang_1 -> So_chi_tiet_ban_hang, DonHang_ChiTiet -> DonHang_ChiTiet."""
    return SPLIT_FILE_SUFFIX_RE.sub("", base_name) or base_name


def merge_split_files(tables: Dict[str, pd.DataFrame], parts: List[Tuple[str, str, Optional[str], bool]]) -> Dict[str, List[str]]:
    """Concatenate, in place, tables split over files that share a split stem and have identical columns.
    parts: (table_key, file name, sheet name, whole_file) per loaded table.
    The split unit is a whole file (So_chi_tiet_ban_hang_1 + _2026) or one sheet of it
    (sheet BanHang of BanHang_TongHop_2025 with 4 other sheets + BanHang_TongHop_2026 holding only BanHang).
    Returns {merged table: [source file names]}.
    Rows are kept as is (no de-duplication): identical line items in different files are distinct sales.
    """
    buckets: Dict[Tuple[str, Tuple[str, ...]], List[Tuple[str, str, Optional[str], bool]]] = {}
    for key, fname, sheet, whole_file in parts:
        stem = split_file_stem(os.path.splitext(fname)[0])
        buckets.setdefault((stem, tuple(tables[key].columns)), []).append((key, fname, sheet, whole_file))

    merged_into: Dict[str, List[str]] = {}
    for (stem, _), members in buckets.items():
        if len({fname for _, fname, _, _ in members}) < 2:
            continue
        sheets = {sheet for _, _, sheet, _ in members}
        if all(whole for _, _, _, whole in members):
            target = stem
        elif len(sheets) == 1 and None not in sheets:
            target = f"{stem}_{sheets.pop()}"
        else:
            target = f"{stem}_gop"
        member_keys = [key for key, _, _, _ in members]
        if target in tables and target not in member_keys:
            target = f"{target}_gop"
        merged = pd.concat([tables[k] for k in member_keys], ignore_index=True)
        for k in member_keys:
            del tables[k]
        tables[target] = merged
        merged_into[target] = [fname for _, fname, _, _ in members]
    return merged_into


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
    # (table_key, file name, sheet name, whole_file) of every table: input of the split-file merge
    table_parts: List[Tuple[str, str, Optional[str], bool]] = []

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

        sheet_frames = [(s, df) for s, df in load_excel_file(x_path, default_year=year, keep_time=keep_time) if not df.empty]

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
            table_parts.append((table_key, xf, s_name, len(sheet_frames) == 1))

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
        table_parts.append((base_name, cf, None, True))

    # One table split over several files (BanHang_T1_2025 + BanHang_T2_2025, So_chi_tiet_ban_hang_1 + _2026)
    # -> one table named after the shared stem; the slices are dropped so no query sees only part of the data
    bdata.metadata["merged_into"] = merge_split_files(other_tables, table_parts)

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
