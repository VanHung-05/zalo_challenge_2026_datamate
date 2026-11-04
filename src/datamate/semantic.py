"""Semantic layer (plan §4.4): the rulebook's definitions, executed by code.

From the Business Card's sales model two DuckDB tables are materialized once per business (fast to query):
  sl_lines   one row per product line, with canonical columns and the order-level flags
  sl_orders  one row per order: date, state, customer, revenue, valid (counts for revenue)
The rulebook defaults live in these views once (fill-down of order cells and duplicate lines
§8; cancelled / returned orders §1, a partial return drops the whole order; order date = date of
the sale lines, not of later return lines), so the fast path and the LLM (which may query the
views) share one definition. Functions raise CapabilityMissing when the data cannot answer and
SemanticUncertain when the answer would depend on a guess (e.g. an unmapped order state).
"""

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.datamate.business_card import BusinessCard, ascii_key
from src.datamate.db import BusinessDB
from src.datamate.time_resolver import Period


class CapabilityMissing(Exception):
    """The data does not record what the question needs -> unanswerable candidate (with reason)."""


class SemanticUncertain(Exception):
    """The answer would depend on an unverified assumption -> abstain."""


class OutOfCoverage(Exception):
    """The period lies entirely outside the dates covered by the data -> unanswerable (rulebook §7)."""


@dataclass
class SemResult:
    value: Any
    money: bool = False
    evidence: Dict[str, Any] = field(default_factory=dict)


# Rulebook §3: "Dòng Phí phục vụ không phải mặt hàng" (still counted in revenue)
NON_ITEM_RE = r"phi (phuc|dich) vu"
WALK_IN_RE = r"^(khach le|khach vang lai|vang lai|le|khach|kl|k l|khach l|vang lai le)$"  # walk-in labels, incl. "kl"
SHIFT_HOURS = {"morning": (6, 13), "afternoon": (14, 21)}


# Name with a trailing phone number ("La Sơn Lan (0793720707)", "Đàm Hà Duy - 0932 811 434") -> name only
PHONE_SUFFIX_SQL = r"regexp_replace(customer_name, '[\s\-–(]*\(?0[0-9 .]{8,}\)?\s*$', '')"


def ascii_sql(expr: str) -> str:
    """SQL for an accent-free lowercase form (DuckDB strip_accents keeps 'đ', so map it first)."""
    return f"strip_accents(replace(lower({expr}), 'đ', 'd'))"


def norm_sql(expr: str) -> str:
    """SQL twin of business_card.ascii_key(): accent-free, lowercase, non-alphanumerics -> one space."""
    return f"trim(regexp_replace({ascii_sql(expr)}, '[^a-z0-9]+', ' ', 'g'))"


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _lit(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, (int, float)):
        return repr(v)
    return "'" + str(v).replace("'", "''") + "'"


class SemanticLayer:
    def __init__(self, db: BusinessDB, card: BusinessCard, shifts: Optional[Dict[str, Tuple[str, str]]] = None):
        self.db = db
        self.card = card
        self.sm = card.sales
        self.ok = False
        self.problems: List[str] = []
        self.has_order_id = "order_id" in self.sm.cols
        self.has_status = "status" in self.sm.cols
        self.unknown_states: List[str] = [k for k, v in self.sm.status_map.items() if v is None]
        if shifts:
            self.shift_hours = {k: (int(a[:2]), int(b[:2])) for k, (a, b) in shifts.items()}
        else:
            self.shift_hours = dict(SHIFT_HOURS)
        try:
            self._build_views()
            self.ok = True
        except Exception as e:  # the layer is optional: the LLM path still works without it
            self.problems.append(f"views not built: {type(e).__name__}: {e}")

    # ── views ────────────────────────────────────────────────────────────────
    def _col(self, role: str, alias: Dict[str, str], cast: Optional[str] = None) -> str:
        if role not in self.sm.cols:
            return "NULL"
        t, c = self.sm.cols[role]
        expr = f"{alias[t]}.{_q(c)}"
        # '1 hộp' quantities were split by db.py into <col>_so (number) / <col>_don_vi (unit)
        if role in ("quantity", "quantity_base") and self._sql_type(t, c) == "VARCHAR" and self._has_col(t, c + "_so"):
            expr = f"{alias[t]}.{_q(c + '_so')}"
        return f"TRY_CAST({expr} AS {cast})" if cast else expr

    def _sql_type(self, table: str, col: str) -> str:
        for ci in self.db.tables[table].columns:
            if ci.sql_name == col:
                return ci.sql_type
        return ""

    def _has_col(self, table: str, col: str) -> bool:
        return any(ci.sql_name == col for ci in self.db.tables[table].columns)

    def _build_views(self) -> None:
        sm = self.sm
        base = sm.line_table or sm.header_table
        if not base:
            raise CapabilityMissing("no sales table")
        alias = {base: "L"}
        joins = []
        if sm.line_table and sm.header_table and sm.join:
            alias[sm.header_table] = "H"
            joins.append(f"LEFT JOIN {_q(sm.header_table)} H ON CAST(L.{_q(sm.join[0])} AS VARCHAR) = CAST(H.{_q(sm.join[1])} AS VARCHAR)")
        for i, (dim, (key_role, key_col)) in enumerate(sm.dims.items()):
            a = f"D{i}"
            alias[dim] = a
            kt, kc = sm.cols[key_role]
            if key_role.endswith("_name"):  # joined on the normalized name (shops without codes)
                dkey, skey = norm_sql(f"CAST({_q(key_col)} AS VARCHAR)"), norm_sql(f"CAST({alias[kt]}.{_q(kc)} AS VARCHAR)")
                on = f"{skey} = {a}._key"
            else:
                dkey = f"CAST({_q(key_col)} AS VARCHAR)"
                on = f"CAST({alias[kt]}.{_q(kc)} AS VARCHAR) = {a}._key"
            joins.append(f"LEFT JOIN (SELECT *, {dkey} AS _key FROM {_q(dim)} QUALIFY ROW_NUMBER() OVER (PARTITION BY {dkey} ORDER BY rowid) = 1) {a} ON {on}")

        # §8: identical lines of the same order are one export error -> keep one (only when orders exist)
        src = _q(base)
        if self.has_order_id and sm.line_table:
            cols = ", ".join(_q(c.sql_name) for c in self.db.tables[base].columns)
            src = f"(SELECT *, rowid AS _row FROM {_q(base)} QUALIFY ROW_NUMBER() OVER (PARTITION BY {cols} ORDER BY rowid) = 1)"
        else:
            src = f"(SELECT *, rowid AS _row FROM {_q(base)})"

        date_expr = self._col("order_date", alias, "TIMESTAMP")
        if "order_time" in sm.cols:
            t_expr = self._col("order_time", alias)
            # time column may be a TIME/'08:15' text or a full creation TIMESTAMP: keep its time of day
            t_of_day = f"COALESCE(TRY_CAST({t_expr} AS TIME), TRY_CAST(TRY_CAST({t_expr} AS TIMESTAMP) AS TIME))"
            date_expr = f"COALESCE(CAST({date_expr} AS DATE) + {t_of_day}, {date_expr})"  # DATE + TIME -> TIMESTAMP
        order_id = f"CAST({self._col('order_id', alias)} AS VARCHAR)" if self.has_order_id else "'R' || CAST(L._row AS VARCHAR)"
        qty = self._col("quantity", alias, "DOUBLE")
        qty_base = self._col("quantity_base", alias, "DOUBLE") if "quantity_base" in sm.cols else qty
        self.qty_base_source = "column" if "quantity_base" in sm.cols else "quantity"
        if "quantity_base" not in sm.cols and "quantity" in sm.cols and self._build_conversion():
            # qty × factor of (product, line unit); units without a conversion row are base units (rulebook §10)
            use_code = self.conversion_key == "product_code"
            pkey = (f"CAST({self._col('product_code', alias)} AS VARCHAR)" if use_code
                    else norm_sql(f"CAST({self._col('product_name', alias)} AS VARCHAR)"))
            unit_src = self._line_unit_expr(alias)
            joins.append(f"LEFT JOIN sl_conversion CV ON CV.pkey = {pkey} AND CV.ukey = {norm_sql(unit_src)}")
            qty_base = f"{qty} * COALESCE(CV.factor, 1)"
            self.qty_base_source = f"conversion ({self.conversion_origin})"
        if sm.grain == "order":
            amount = self._col("order_goods_amount", alias, "DOUBLE") if "order_goods_amount" in sm.cols else self._col("order_total", alias, "DOUBLE")
        else:
            amount = self._col("line_amount", alias, "DOUBLE")
        text = lambda r: f"NULLIF(TRIM(CAST({self._col(r, alias)} AS VARCHAR)), '')"  # noqa: E731

        raw_sql = f"""
            SELECT L._row AS _row, {order_id} AS order_id, {date_expr} AS order_ts,
                   {text('product_code')} AS product_code, {text('product_name')} AS product_name,
                   {text('product_group')} AS product_group, {text('unit')} AS unit,
                   {qty} AS qty, {qty_base} AS qty_base, {self._col('unit_price', alias, 'DOUBLE')} AS unit_price,
                   {amount} AS line_amount, {self._col('cost_per_unit', alias, 'DOUBLE')} AS cost_per_unit,
                   {self._col('return_qty', alias, 'DOUBLE')} AS return_qty,
                   {text('status')} AS status_raw, {text('customer_code')} AS customer_code,
                   {text('customer_name')} AS customer_name, {text('customer_group')} AS customer_group,
                   {text('payment_method')} AS payment_method, {text('sales_channel')} AS sales_channel,
                   {text('staff')} AS staff
            FROM {src} L {' '.join(joins)}"""
        con = self.db.conn
        con.execute(f"CREATE OR REPLACE TABLE sl_lines_raw AS {raw_sql}").fetchall()

        state_cases = " ".join(f"WHEN {_lit(k)} THEN {_lit(v or 'unknown')}" for k, v in sm.status_map.items())
        state = f"CASE status_raw {state_cases} ELSE 'unknown' END" if self.has_status and state_cases else "'completed'"
        first = lambda c: f"arg_min({c}, _row) FILTER (WHERE {c} IS NOT NULL)"  # noqa: E731 - fill-down (§8)
        con.execute(f"""
            CREATE OR REPLACE TABLE sl_orders AS
            WITH o AS (
                SELECT order_id,
                       COALESCE(MIN(order_ts) FILTER (WHERE COALESCE(qty, 0) >= 0 AND COALESCE(line_amount, 0) >= 0), MIN(order_ts)) AS order_ts,
                       {first('status_raw')} AS status_raw, {first('customer_code')} AS customer_code,
                       {first('customer_name')} AS customer_name, {first('customer_group')} AS customer_group,
                       {first('payment_method')} AS payment_method, {first('sales_channel')} AS sales_channel,
                       {first('staff')} AS staff,
                       BOOL_OR(qty < 0 OR line_amount < 0) AS has_negative_line,
                       BOOL_OR(return_qty > 0) AS has_return_qty,
                       SUM(line_amount) AS amount, COUNT(*) AS n_lines
                FROM sl_lines_raw GROUP BY order_id
            )
            SELECT *, CAST(order_ts AS DATE) AS order_date,
                   CASE WHEN status_raw IS NULL AND {str(self.has_status).upper()} THEN 'unknown' ELSE {state} END AS state_from_status,
                   CASE
                     WHEN ({state}) = 'cancelled' THEN 'cancelled'
                     WHEN ({state}) = 'returned' OR COALESCE(has_negative_line, FALSE) OR COALESCE(has_return_qty, FALSE) THEN 'returned'
                     WHEN status_raw IS NULL AND {str(self.has_status).upper()} THEN 'unknown'
                     ELSE ({state}) END AS state
            FROM o""").fetchall()
        con.execute("""
            CREATE OR REPLACE TABLE sl_lines AS
            SELECT l._row, l.order_id, o.order_ts, o.order_date, l.product_code, l.product_name, l.product_group, l.unit,
                   l.qty, l.qty_base, l.unit_price, l.line_amount, l.cost_per_unit, l.return_qty,
                   o.state, o.customer_code, o.customer_name, o.customer_group, o.payment_method, o.sales_channel, o.staff
            FROM sl_lines_raw l JOIN sl_orders o USING (order_id)""").fetchall()

    def _line_unit_expr(self, alias: Dict[str, str]) -> str:
        """Unit of a sales line: the unit column, else the unit written in the quantity text ('3 chai')."""
        parts = []
        if "unit" in self.sm.cols:
            parts.append(f"CAST({self._col('unit', alias)} AS VARCHAR)")
        t, c = self.sm.cols["quantity"]
        if self._has_col(t, c + "_don_vi"):
            parts.append(f"CAST({alias[t]}.{_q(c + '_don_vi')} AS VARCHAR)")
        return f"COALESCE({', '.join(parts)})" if parts else "NULL"

    def _build_conversion(self) -> bool:
        """TABLE sl_conversion(pkey, ukey, factor) from the shop's conversion table, else from the
        README ('- Coca Cola 330ml: 1 thùng = 24 lon'). Keys are normalized names (or product codes)."""
        rows: List[Tuple[str, str, float]] = []
        conv = self.sm.conversion
        self.conversion_key, self.conversion_origin = "product_name", ""
        if conv:
            t = conv["table"]
            data = self.db.conn.execute(
                f'SELECT CAST({_q(conv["key"])} AS VARCHAR), CAST({_q(conv["unit"])} AS VARCHAR), TRY_CAST({_q(conv["factor"])} AS DOUBLE) '
                f'FROM {_q(t)}').fetchall()
            use_code = conv["key_role"] == "product_code"
            rows = [((k if use_code else ascii_key(k)), ascii_key(u), f) for k, u, f in data if k and u and f]
            self.conversion_key = "product_code" if use_code else "product_name"
            self.conversion_origin = f"table {t}"
        elif self.card.readme_conversions:
            rows = [(ascii_key(p), ascii_key(u), f) for p, u, f, _b in self.card.readme_conversions]
            self.conversion_origin = "README"
        if not rows:
            return False
        # one factor per (product, unit): conflicting rows make the conversion unusable for that pair
        uniq: Dict[Tuple[str, str], set] = {}
        for k, u, f in rows:
            uniq.setdefault((k, u), set()).add(f)
        clean = [(k, u, next(iter(fs))) for (k, u), fs in uniq.items() if len(fs) == 1]
        con = self.db.conn
        con.execute("CREATE OR REPLACE TABLE sl_conversion (pkey VARCHAR, ukey VARCHAR, factor DOUBLE)").fetchall()
        con.executemany("INSERT INTO sl_conversion VALUES (?, ?, ?)", clean)
        return True

    def view_description(self) -> str:
        """Text for the LLM prompt (the views can be queried like any table)."""
        if not self.ok:
            return ""
        return (
            "VIEW sl_orders (mỗi dòng một đơn, ĐÃ áp rulebook): order_id, order_ts, order_date, state "
            "('completed'|'delivering'|'cancelled'|'returned'|'unknown'; đơn có trả hàng dù một phần là 'returned'), "
            "customer_code, customer_name, customer_group, payment_method, sales_channel, staff, amount (tổng thành tiền dòng), n_lines.\n"
            "VIEW sl_lines (mỗi dòng một dòng hàng, đã gộp dòng trùng cùng đơn và kế thừa ô trống của đơn): order_id, order_ts, order_date, "
            "product_code, product_name, product_group, unit, qty, qty_base (quy về đơn vị cơ bản), unit_price, line_amount, cost_per_unit, "
            "state, customer_code, customer_name, customer_group, payment_method, sales_channel, staff.\n"
            "Doanh thu = SUM(line_amount) của sl_lines (hoặc SUM(amount) của sl_orders) WHERE state IN ('completed','delivering')."
        )

    # ── helpers ──────────────────────────────────────────────────────────────
    def _require(self, *roles: str) -> None:
        if not self.ok:
            raise SemanticUncertain("; ".join(self.problems) or "semantic layer unavailable")
        missing = [r for r in roles if r not in self.sm.cols]
        if missing:
            raise CapabilityMissing(f"dữ liệu không có {', '.join(missing)}")

    def _period_where(self, period: Optional[Period], as_of: Optional[date], shift: Optional[str] = None,
                      prefix: str = "") -> str:
        conds = []
        if period is not None:
            start, end = period.start, period.end
            if as_of is not None and end > as_of:
                end = as_of  # records after today have not happened yet (rulebook intro)
            self._check_coverage(start, end)
            conds.append(f"{prefix}order_date BETWEEN DATE '{start}' AND DATE '{end}'")
        if shift:
            if not self._one("SELECT COUNT(*) FROM sl_orders WHERE CAST(order_ts AS TIME) <> TIME '00:00:00'"):
                raise CapabilityMissing("dữ liệu không ghi giờ tạo đơn nên không chia được ca")
            lo, hi = self.shift_hours[shift]
            conds.append(f"HOUR({prefix}order_ts) BETWEEN {lo} AND {hi}")
        return " AND ".join(conds) or "TRUE"

    def _check_coverage(self, start: date, end: date) -> None:
        cmin, cmax = self.card.coverage
        if cmin and cmax and (str(end) < cmin or str(start) > cmax):
            raise OutOfCoverage(f"kỳ {start}..{end} nằm ngoài khoảng dữ liệu {cmin}..{cmax}")

    def _guard_states(self, where: str, table: str = "sl_orders") -> None:
        """Refuse when an order in scope has a state we could not map (no guessing)."""
        n = self.db.conn.execute(f"SELECT COUNT(*) FROM {table} WHERE state = 'unknown' AND {where}").fetchall()[0][0]
        if n:
            raise SemanticUncertain(f"{n} đơn có trạng thái chưa xác định ({self.unknown_states[:3]})")

    def _one(self, sql: str) -> Any:
        return self.db.conn.execute(sql).fetchall()[0][0]

    def _filters(self, filters: Optional[Dict[str, Any]]) -> str:
        """filters: {column of sl_lines: value or [values]}. Text values match every spelling of the
        same name (rulebook §8: 'các tên cùng chỉ một thứ ... là cùng một thứ'): accent-, case-,
        space- and punctuation-insensitive, the same normalization as ascii_key()."""
        if not filters:
            return "TRUE"
        conds = []
        for col, val in filters.items():
            # A filter on a column the sales lines do not fill would silently give 0
            if not self._one(f"SELECT COUNT({col}) FROM sl_lines"):
                raise SemanticUncertain(f"cột {col} không có dữ liệu trong sl_lines (chưa nối được bảng danh mục?)")
            vals = val if isinstance(val, (list, tuple, set)) else [val]
            if col in ("product_code", "customer_code"):
                conds.append(f"{col} IN ({', '.join(_lit(v) for v in vals)})")
            else:
                conds.append(f"{norm_sql(col)} IN ({', '.join(_lit(ascii_key(v)) for v in vals)})")
        return " AND ".join(conds)

    # ── metrics ──────────────────────────────────────────────────────────────
    def revenue(self, period: Optional[Period], as_of: Optional[date] = None, filters: Optional[Dict[str, Any]] = None,
                shift: Optional[str] = None) -> SemResult:
        self._require("line_amount" if self.sm.grain == "line" else ("order_goods_amount" if "order_goods_amount" in self.sm.cols else "order_total"))
        where = self._period_where(period, as_of, shift)
        flt = self._filters(filters)
        self._guard_states(f"{where} AND order_id IN (SELECT order_id FROM sl_lines WHERE {flt})")
        if self.sm.grain == "order" and "order_goods_amount" not in self.sm.cols:
            raise SemanticUncertain("doanh thu chỉ có tổng tiền đơn (có thể gồm phí giao, giảm giá đơn)")
        sql = f"SELECT COALESCE(SUM(line_amount), 0) FROM sl_lines WHERE state IN ('completed','delivering') AND {where} AND {flt}"
        return SemResult(self._one(sql), money=True, evidence={"sql": sql})

    def order_count(self, period: Optional[Period], as_of: Optional[date] = None, states: Sequence[str] = ("completed", "delivering"),
                    filters: Optional[Dict[str, Any]] = None, shift: Optional[str] = None) -> SemResult:
        if not self.has_order_id:
            raise CapabilityMissing("dữ liệu không ghi mã đơn")
        where = self._period_where(period, as_of, shift)
        flt = self._filters(filters)
        self._guard_states(where)
        sql = (f"SELECT COUNT(DISTINCT order_id) FROM sl_lines WHERE state IN ({', '.join(_lit(s) for s in states)}) "
               f"AND {where} AND {flt}")
        return SemResult(self._one(sql), evidence={"sql": sql})

    def cancel_rate(self, period: Optional[Period], as_of: Optional[date] = None) -> SemResult:
        """% cancelled among ALL orders placed in the period, whatever their state (rulebook §1)."""
        if not self.has_order_id or not self.has_status:
            raise CapabilityMissing("dữ liệu không ghi trạng thái/mã đơn")
        if "cancelled" not in self.sm.status_map.values():
            raise CapabilityMissing("dữ liệu không có trạng thái huỷ")
        where = self._period_where(period, as_of)
        self._guard_states(where)
        sql = (f"SELECT 100.0 * COUNT(*) FILTER (WHERE state = 'cancelled') / NULLIF(COUNT(*), 0) FROM sl_orders WHERE {where}")
        v = self._one(sql)
        if v is None:
            raise SemanticUncertain("không có đơn nào trong kỳ")
        return SemResult(v, evidence={"sql": sql})

    def avg_order_value(self, period: Optional[Period], as_of: Optional[date] = None) -> SemResult:
        rev = self.revenue(period, as_of).value
        n = self.order_count(period, as_of).value
        if not n:
            raise SemanticUncertain("không có đơn hợp lệ trong kỳ")
        return SemResult(rev / n, money=True, evidence={"revenue": rev, "orders": n})

    def quantity(self, period: Optional[Period], as_of: Optional[date] = None, filters: Optional[Dict[str, Any]] = None,
                 base: bool = True) -> SemResult:
        self._require("quantity_base" if base and "quantity_base" in self.sm.cols else "quantity")
        where = self._period_where(period, as_of)
        flt = self._filters(filters)
        self._guard_states(f"{where} AND order_id IN (SELECT order_id FROM sl_lines WHERE {flt})")
        col = "qty_base" if base else "qty"
        sql = f"SELECT COALESCE(SUM({col}), 0) FROM sl_lines WHERE state IN ('completed','delivering') AND {where} AND {flt}"
        return SemResult(self._one(sql), evidence={"sql": sql})

    def period_compare(self, metric: str, p1: Period, p2: Period, as_of: Optional[date] = None, **kw: Any) -> SemResult:
        """% change of the asked period p1 against the comparison period p2: (p1 - p2) / p2 * 100 (rulebook §1)."""
        fn = getattr(self, metric)
        v1, v2 = fn(p1, as_of, **kw).value, fn(p2, as_of, **kw).value
        if not v2:
            raise SemanticUncertain("kỳ so sánh bằng 0")
        return SemResult((v1 - v2) * 100.0 / v2, evidence={"current": v1, "previous": v2})

    # ── rankings ─────────────────────────────────────────────────────────────
    def _entity_key(self, by: str) -> Tuple[str, str, str]:
        """(key expr, display expr, extra where) for ranking `by` in sl_lines."""
        if by == "product":
            if "product_code" not in self.sm.cols:
                self._require("product_name")
            # Without codes, spellings of one name are one product (rulebook §8)
            key = "product_code" if "product_code" in self.sm.cols else norm_sql("product_name")
            item_text = ascii_sql("COALESCE(product_name, '') || ' ' || COALESCE(product_group, '')")
            excl = f"NOT regexp_matches({item_text}, '{NON_ITEM_RE}')"
            return key, "product_name" if "product_name" in self.sm.cols else "product_code", excl
        if by == "customer":
            if "customer_code" in self.sm.cols:
                return "customer_code", "customer_name" if "customer_name" in self.sm.cols else "customer_code", "customer_code IS NOT NULL"
            self._require("customer_name")
            # Shop without codes: names that differ only in spelling are one customer (rulebook §3)
            key = norm_sql(PHONE_SUFFIX_SQL)
            walk_in = f"NOT regexp_matches(trim({key}), '{WALK_IN_RE}')"
            return key, "customer_name", f"customer_name IS NOT NULL AND {walk_in}"
        if by == "staff":
            self._require("staff")
            return "staff", "staff", "staff IS NOT NULL"
        if by == "day":
            return "CAST(order_date AS VARCHAR)", "CAST(order_date AS VARCHAR)", "TRUE"
        if by == "month":
            return "strftime(order_date, '%Y-%m')", "strftime(order_date, '%Y-%m')", "TRUE"
        if by == "hour":
            return "CAST(HOUR(order_ts) AS VARCHAR)", "HOUR(order_ts)", "TRUE"
        raise SemanticUncertain(f"unknown ranking dimension {by}")

    METRIC_SQL = {
        "revenue": "SUM(line_amount)",
        "quantity": "SUM(qty_base)",
        "order_count": "COUNT(DISTINCT order_id)",
        "purchase_days": "COUNT(DISTINCT order_date)",
    }

    def rank(self, by: str, metric: str, period: Optional[Period], as_of: Optional[date] = None, k: Optional[int] = 1,
             filters: Optional[Dict[str, Any]] = None, ascending: bool = False, min_orders: Optional[int] = None) -> SemResult:
        """Top-k `by` by `metric`, ties broken by code (else name) ascending (rulebook §9)."""
        key, display, extra = self._entity_key(by)
        where = self._period_where(period, as_of)
        flt = self._filters(filters)
        self._guard_states(where)
        m = self.METRIC_SQL[metric] if metric != "aov" else "SUM(line_amount) / COUNT(DISTINCT order_id)"
        having = f"HAVING COUNT(DISTINCT order_id) >= {int(min_orders)}" if min_orders else ""
        order = "ASC" if ascending else "DESC"
        # Display the most frequent spelling of the entity (any spelling in the data is accepted)
        sql = f"""
            SELECT {key} AS k, mode({display}) AS name, {m} AS v
            FROM sl_lines WHERE state IN ('completed','delivering') AND {where} AND {flt} AND {extra}
            GROUP BY 1 {having} ORDER BY v {order}, k ASC {f'LIMIT {int(k) + 1}' if k else ''}"""
        rows = self.db.conn.execute(sql).fetchall()
        if not rows:
            raise SemanticUncertain("không có đối tượng nào trong kỳ")
        if k and len(rows) > k and rows[k - 1][2] == rows[k][2]:
            # Organizer excludes boundary ties; if we see one our mapping is probably wrong
            raise SemanticUncertain("hoà ở biên top-k")
        top = rows[:k] if k else rows
        return SemResult([r[1] for r in top], evidence={"sql": sql, "values": [r[2] for r in top], "keys": [r[0] for r in top]})

    def count_entities(self, by: str, metric: str, op: str, threshold: float, period: Optional[Period],
                       as_of: Optional[date] = None) -> SemResult:
        """How many `by` have metric `op` threshold (e.g. customers with >= 5 orders, days with revenue >= X)."""
        if by == "order":
            if not self.has_order_id:
                raise CapabilityMissing("dữ liệu không ghi mã đơn")
            key, extra = "order_id", "TRUE"
        else:
            key, _display, extra = self._entity_key(by)
        where = self._period_where(period, as_of)
        self._guard_states(where)
        m = self.METRIC_SQL[metric]
        sql = f"""SELECT COUNT(*) FROM (SELECT {key} AS k, {m} AS v FROM sl_lines
                  WHERE state IN ('completed','delivering') AND {where} AND {extra} GROUP BY 1) WHERE v {op} {threshold!r}"""
        return SemResult(self._one(sql), evidence={"sql": sql})

    def distinct_count(self, by: str, period: Optional[Period], as_of: Optional[date] = None) -> SemResult:
        key, _d, extra = self._entity_key(by)
        where = self._period_where(period, as_of)
        self._guard_states(where)
        sql = f"SELECT COUNT(DISTINCT {key}) FROM sl_lines WHERE state IN ('completed','delivering') AND {where} AND {extra}"
        return SemResult(self._one(sql), evidence={"sql": sql})

    def share_of_total(self, filters: Dict[str, Any], period: Optional[Period], as_of: Optional[date] = None) -> SemResult:
        """% of the shop's revenue (whole shop, walk-ins included) that matches `filters`."""
        part = self.revenue(period, as_of, filters).value
        total = self.revenue(period, as_of).value
        if not total:
            raise SemanticUncertain("doanh thu kỳ bằng 0")
        return SemResult(part * 100.0 / total, evidence={"part": part, "total": total})

    # ── entity resolution ────────────────────────────────────────────────────
    def resolve_value(self, column: str, phrase: str) -> List[str]:
        """Values of an sl_lines text column whose accent-free normalized form equals / contains the phrase."""
        key = ascii_key(phrase)
        rows = self.db.conn.execute(
            f"SELECT DISTINCT {column} FROM sl_lines WHERE {column} IS NOT NULL").fetchall()
        vals = [r[0] for r in rows]
        exact = [v for v in vals if ascii_key(v) == key]
        if exact:
            return exact
        return [v for v in vals if re.search(rf"\b{re.escape(key)}\b", ascii_key(v))]
