"""DuckDB layer for DataMate (SQL mode).
Loads a business's DataFrames into an in-memory DuckDB database with ASCII snake_case names,
profiles the tables into a schema text for the prompt, and executes LLM-generated SELECT
queries read-only with a real (interruptible) timeout.
"""

import decimal
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
from unidecode import unidecode

from src.datamate.loader import BusinessData


ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ISO_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?$")
# "1 ly", "5 chai", "2,5 kg", "12" -> (number, unit)
NUMBER_UNIT_RE = re.compile(r"^\s*(-?\d+(?:[.,]\d+)?)\s*([^\d\s][^\d]*)?$")

MAX_DISTINCT_LISTED = 25
MAX_SAMPLE_CHARS = 40

_RESERVED_KEYWORDS: Optional[set] = None


def _reserved_keywords() -> set:
    global _RESERVED_KEYWORDS
    if _RESERVED_KEYWORDS is None:
        con = duckdb.connect()
        rows = con.execute(
            "SELECT keyword_name FROM duckdb_keywords() WHERE keyword_category IN ('reserved', 'type_function')"
        ).fetchall()
        con.close()
        _RESERVED_KEYWORDS = {r[0].lower() for r in rows}
    return _RESERVED_KEYWORDS


def to_sql_identifier(name: Any, used: set) -> str:
    """Original (Vietnamese) name -> unique ASCII snake_case identifier, e.g. 'Ngày đặt hàng' -> 'ngay_dat_hang'."""
    ident = unidecode(str(name))
    ident = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", ident).lower()  # ThoiGianLap -> thoi_gian_lap
    ident = re.sub(r"[^a-z0-9]+", "_", ident).strip("_") or "col"
    if ident[0].isdigit():
        ident = f"c_{ident}"
    if ident in _reserved_keywords():
        ident = f"{ident}_"
    base, i = ident, 2
    while ident in used:
        ident = f"{base}_{i}"
        i += 1
    used.add(ident)
    return ident


@dataclass
class ColumnInfo:
    sql_name: str
    original: str
    sql_type: str = ""
    n_null: int = 0
    n_distinct: int = 0
    distinct_values: List[Any] = field(default_factory=list)  # filled when n_distinct is small
    samples: List[Any] = field(default_factory=list)
    min_val: Any = None
    max_val: Any = None
    note: str = ""


@dataclass
class TableInfo:
    sql_name: str
    original: str
    n_rows: int
    columns: List[ColumnInfo] = field(default_factory=list)
    hidden: bool = False


def _prepare_column(s: pd.Series) -> Tuple[List[Tuple[str, pd.Series, str]], str]:
    """Decide the DuckDB type of one column.
    Returns ([(suffix, series, cast_type)], note) where suffix '' is the column itself and other
    suffixes are derived columns. cast_type is a DuckDB type to CAST to, or '' to keep the inferred one.
    Conversions are only applied when every non-null value fits, so no information is dropped.
    """
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return [("", s, "")], ""

    non_null = s.dropna()
    if non_null.empty:
        return [("", s.astype("string"), "VARCHAR")], ""

    if all(isinstance(v, (bool, np.bool_)) for v in non_null):
        return [("", s.astype("boolean"), "")], ""
    if all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) for v in non_null):
        return [("", pd.to_numeric(s), "")], ""

    as_str = non_null.astype(str).str.strip()
    if as_str.str.match(ISO_DATETIME_RE).all():
        parsed = pd.to_datetime(s.astype("string").str.strip(), errors="coerce", format="ISO8601")
        cast = "DATE" if as_str.str.match(ISO_DATE_RE).all() else "TIMESTAMP"
        return [("", parsed, cast)], ""

    # Leading zeros (phone numbers, codes like '0012') would be lost as numbers
    has_leading_zero = as_str.str.match(r"^0\d").any()
    numeric = pd.to_numeric(as_str, errors="coerce")
    if numeric.notna().all() and not has_leading_zero:
        return [("", pd.to_numeric(s.astype("string").str.strip(), errors="coerce"), "")], ""

    m = as_str.str.extract(NUMBER_UNIT_RE)
    units = m[1].dropna().str.strip()
    # Quantities repeat a few units ('1 ly', '2 ly'); addresses ('334 Quang Trung') do not
    is_quantity = units.nunique() <= MAX_DISTINCT_LISTED and units.nunique() * 3 <= len(as_str)
    if m[0].notna().all() and not units.empty and is_quantity and not has_leading_zero:
        number = pd.to_numeric(m[0].str.replace(",", ".", regex=False), errors="coerce").reindex(s.index)
        unit = m[1].str.strip().reindex(s.index)
        text = s.where(s.isna(), s.astype(str))
        note = "dạng 'số + đơn vị'; đã tách sẵn cột {name}_so (số) và {name}_don_vi (đơn vị, NULL nếu không ghi)"
        return [("", text, "VARCHAR"), ("_so", number, ""), ("_don_vi", unit, "VARCHAR")], note

    return [("", s.where(s.isna(), s.astype(str)), "VARCHAR")], ""


def _short(v: Any) -> str:
    text = str(v)
    return text if len(text) <= MAX_SAMPLE_CHARS else text[: MAX_SAMPLE_CHARS - 3] + "..."


def _fmt_stat(v: Any) -> str:
    if isinstance(v, (float, np.floating)) and not float(v).is_integer():
        return f"{float(v):.6g}"
    if isinstance(v, (int, float, np.integer, np.floating)):
        return str(int(v))
    return str(v)


class BusinessDB:
    """In-memory DuckDB database of one business."""

    def __init__(self, business_data: BusinessData, calendar_df: Optional[pd.DataFrame] = None, threads: int = 1):
        self.business_id = business_data.business_id
        self.conn = duckdb.connect(database=":memory:")
        # threads=1 keeps float aggregation order (and unordered row order) reproducible between runs
        self.conn.execute(f"SET threads = {int(threads)}").fetchall()
        self.tables: Dict[str, TableInfo] = {}

        merged_into: Dict[str, List[str]] = business_data.metadata.get("merged_into", {})
        hidden_originals = {k for keys in merged_into.values() for k in keys}

        used_tables: set = set()
        self.sql_name_of: Dict[str, str] = {}
        for original, df in business_data.tables.items():
            sql_name = to_sql_identifier(original, used_tables)
            self.sql_name_of[original] = sql_name
            self._load_table(sql_name, original, df, hidden=original in hidden_originals)
        self.merged_into = {
            self.sql_name_of[target]: [self.sql_name_of[k] for k in keys if k in self.sql_name_of]
            for target, keys in merged_into.items() if target in self.sql_name_of
        }

        if calendar_df is not None and not calendar_df.empty:
            sql_name = to_sql_identifier("lich", used_tables)
            self._load_table(sql_name, "calendar.csv", calendar_df, hidden=False)
            self.calendar_table = sql_name
        else:
            self.calendar_table = None

        # Generated SQL must not read or write files, attach databases, or change settings
        self.conn.execute("SET enable_external_access = false").fetchall()
        self.conn.execute("SET lock_configuration = true").fetchall()
        self.schema_text = self._build_schema_text()

    def _load_table(self, sql_name: str, original: str, df: pd.DataFrame, hidden: bool) -> None:
        used_cols: set = set()
        out = {}
        casts = []
        col_infos: List[ColumnInfo] = []
        for col in df.columns:
            base = to_sql_identifier(col, used_cols)
            parts, note = _prepare_column(df[col].reset_index(drop=True))
            for suffix, series, cast in parts:
                name = base if not suffix else to_sql_identifier(base + suffix, used_cols)
                out[name] = series.reset_index(drop=True)
                casts.append(f'CAST("{name}" AS {cast}) AS "{name}"' if cast else f'"{name}"')
                info = ColumnInfo(sql_name=name, original=str(col) if not suffix else "")
                if not suffix and note:
                    info.note = note.format(name=base)
                col_infos.append(info)

        frame = pd.DataFrame(out)
        self.conn.register("_tmp_frame", frame)
        self.conn.execute(f'CREATE TABLE "{sql_name}" AS SELECT {", ".join(casts)} FROM _tmp_frame').fetchall()
        self.conn.unregister("_tmp_frame")

        table = TableInfo(sql_name=sql_name, original=original, n_rows=len(frame), columns=col_infos, hidden=hidden)
        self._profile_table(table)
        self.tables[sql_name] = table

    def _profile_table(self, table: TableInfo) -> None:
        types = dict(self.conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ?", [table.sql_name]
        ).fetchall())
        for info in table.columns:
            info.sql_type = types.get(info.sql_name, "")
            q = f'"{info.sql_name}"'
            n_null, n_distinct, mn, mx = self.conn.execute(
                f'SELECT COUNT(*) - COUNT({q}), COUNT(DISTINCT {q}), MIN({q}), MAX({q}) FROM "{table.sql_name}"'
            ).fetchall()[0]
            info.n_null, info.n_distinct = n_null, n_distinct
            if info.sql_type == "VARCHAR":
                if n_distinct <= MAX_DISTINCT_LISTED:
                    info.distinct_values = [r[0] for r in self.conn.execute(
                        f'SELECT {q} FROM "{table.sql_name}" WHERE {q} IS NOT NULL GROUP BY {q} ORDER BY COUNT(*) DESC, {q}'
                    ).fetchall()]
                else:
                    info.samples = [r[0] for r in self.conn.execute(
                        f'SELECT DISTINCT {q} FROM "{table.sql_name}" WHERE {q} IS NOT NULL ORDER BY {q} LIMIT 3'
                    ).fetchall()]
            elif info.sql_type == "BOOLEAN":
                pass
            else:
                info.min_val, info.max_val = mn, mx

    def _build_schema_text(self) -> str:
        lines: List[str] = []
        for table in self.tables.values():
            if table.hidden:
                continue
            header = f"TABLE {table.sql_name}  -- {table.n_rows:,} dòng".replace(",", ".")
            if table.original != table.sql_name:
                header += f'; nguồn: "{table.original}"'
            lines.append(header)
            empty_cols = []
            for c in table.columns:
                if table.n_rows and c.n_null == table.n_rows:
                    empty_cols.append(c.sql_name)
                    continue
                desc = []
                if c.original and c.original != c.sql_name:
                    desc.append(f'cột gốc "{c.original}"')
                if c.note:
                    desc.append(c.note)
                if c.distinct_values:
                    desc.append("giá trị: " + ", ".join(repr(_short(v)) for v in c.distinct_values))
                elif c.samples:
                    desc.append(f"{c.n_distinct:,} giá trị khác nhau".replace(",", ".") + ", vd: " + ", ".join(repr(_short(v)) for v in c.samples))
                elif c.min_val is not None:
                    desc.append(f"từ {_fmt_stat(c.min_val)} đến {_fmt_stat(c.max_val)}")
                if c.n_null:
                    desc.append(f"{c.n_null:,} NULL".replace(",", "."))
                lines.append(f"  {c.sql_name} {c.sql_type}" + (f"  -- {'; '.join(desc)}" if desc else ""))
            if empty_cols:
                lines.append(f"  (cột luôn rỗng, bỏ qua: {', '.join(empty_cols)})")
            if table.original == "calendar.csv":
                lines.append("  (lịch dương–âm dùng chung; weekday: 1 = Thứ Hai ... 7 = Chủ nhật; lunar_*: ngày/tháng/năm âm lịch)")
            lines.append("")

        # Quan hệ gợi ý (JOIN hints) giữa các bảng
        join_hints = []
        visible_tables = [t for t in self.tables.values() if not t.hidden]
        candidate_keys = ["ma_hoa_don", "ma_don_hang", "ma_nhap_hang", "ma_hang", "ma_sp", "ma_khach_hang", "ma_kh", "ma_ncc"]
        for key in candidate_keys:
            tables_with_key = []
            for t in visible_tables:
                col_names = {c.sql_name for c in t.columns}
                if key in col_names:
                    tables_with_key.append((t.sql_name, key))
            if len(tables_with_key) >= 2:
                names = [t[0] for t in tables_with_key]
                join_hints.append(f"  - Các bảng có thể JOIN qua cột '{key}': {', '.join(names)}")

        if join_hints:
            lines.append("GỢI Ý QUAN HỆ GIỮA CÁC BẢNG (JOIN HINTS):")
            lines.extend(join_hints)
            lines.append("")

        for target, parts in self.merged_into.items():
            if parts:
                lines.append(f"Các bảng {', '.join(parts)} là các phần đã gộp sẵn vào bảng {target}: dùng {target}.")
        return "\n".join(lines).strip()

    def execute(self, sql: str, timeout: float = 30, max_rows: int = 10000) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Run one read-only SELECT. Returns (DataFrame, None) or (None, error message)."""
        try:
            statements = self.conn.extract_statements(sql)
        except duckdb.Error as e:
            return None, f"Lỗi cú pháp SQL: {e}"
        if len(statements) != 1:
            return None, f"Chỉ được viết đúng 1 câu lệnh SELECT (nhận {len(statements)} câu lệnh)"
        if statements[0].type != duckdb.StatementType.SELECT:
            return None, f"Chỉ được dùng câu lệnh SELECT (nhận {statements[0].type.name})"

        cur = self.conn.cursor()
        timer = threading.Timer(timeout, cur.interrupt)
        timer.start()
        try:
            cur.execute(sql)
            columns = [d[0] for d in cur.description]
            rows = cur.fetchmany(max_rows + 1)
        except duckdb.InterruptException:
            return None, f"Truy vấn chạy quá {timeout} giây, hãy viết truy vấn đơn giản hơn"
        except duckdb.Error as e:
            return None, f"Lỗi khi chạy SQL: {type(e).__name__}: {e}"
        finally:
            timer.cancel()
            cur.close()
        if len(rows) > max_rows:
            return None, f"Kết quả có hơn {max_rows} dòng; câu hỏi cần kết quả đã tổng hợp, hãy GROUP BY / LIMIT"
        rows = [[float(v) if isinstance(v, decimal.Decimal) else v for v in r] for r in rows]
        return pd.DataFrame(rows, columns=columns), None

    def close(self) -> None:
        self.conn.close()
