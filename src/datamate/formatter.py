"""Answer formatter for DataMate.
Ensures outputs strictly adhere to rulebook §9 format requirements.
Rounding happens once, here, with Decimal half-up ("nửa đồng làm tròn lên", rulebook §3),
so float noise from SQL/pandas never reaches the submission.
"""

import math
import re
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


def clean_number_str(val_str: str) -> Optional[float]:
    """Parse string with potential Vietnamese number formatting into float."""
    s = val_str.strip()
    # Remove currency / percent suffix
    s = re.sub(r"(đ|vnđ|vnd|%|\$)\s*$", "", s, flags=re.IGNORECASE).strip()
    # Handle Vietnamese notation like "4.820.000" or "4,820,000"
    if re.match(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$", s):
        # dot as thousand, comma as decimal
        s = s.replace(".", "").replace(",", ".")
    elif re.match(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$", s):
        # comma as thousand, dot as decimal
        s = s.replace(",", "")
    elif "," in s and "." not in s:
        # e.g. "12,4" -> "12.4"
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def round_half_up(val: Any, digits: int = 0) -> Union[int, float]:
    """Half-up rounding on the decimal representation (2.675 -> 2.68, 0.5 -> 1, -0.5 -> -1)."""
    d = Decimal(repr(float(val))) if not isinstance(val, Decimal) else val
    q = Decimal(1).scaleb(-digits)
    r = d.quantize(q, rounding=ROUND_HALF_UP)
    return int(r) if digits == 0 else float(r)


def _scalar(val: Any, what: str) -> Any:
    if isinstance(val, (pd.Series, np.ndarray, list, tuple)):
        if len(val) == 1:
            return val[0]
        raise ValueError(f"Cannot format a sequence of length {len(val)} as a single {what}")
    return val


def _to_float(val: Any, what: str) -> float:
    if isinstance(val, (bool, np.bool_)):
        raise ValueError(f"Boolean is not a {what}")
    if isinstance(val, str):
        parsed = clean_number_str(val)
        if parsed is None:
            raise ValueError(f"Cannot parse '{val}' as {what}")
        val = parsed
    if val is None or pd.isna(val):
        raise ValueError(f"Cannot format NaN as {what}")
    num = float(val)
    if not math.isfinite(num):
        raise ValueError(f"Cannot format {num} as {what}")
    return num


def format_number(val: Any, is_money: bool = False) -> Union[int, float]:
    """Money (and averages of money) -> integer VND, half-up; integral values -> int;
    anything else -> 2 decimals, half-up."""
    num = _to_float(_scalar(val, "number"), "number")
    if is_money:
        return round_half_up(num, 0)
    if abs(num - round(num)) < 1e-9:
        return int(round(num))
    out = round_half_up(num, 2)
    return int(out) if float(out).is_integer() else out


def format_percent(val: Any) -> float:
    """The value must already be in percent units (12.4 for 12,4 %); rounded to 2 decimals.
    A ratio like 0.124 is NOT rescaled: 0.124 % is a valid answer, so guessing would be wrong
    on real small percentages. decision.py flags suspicious magnitudes instead."""
    if isinstance(val, str):
        val = val.replace("%", "").strip()
    num = _to_float(_scalar(val, "percent"), "percent")
    out = round_half_up(num, 2)
    return 0.0 if out == 0 else out  # no "-0.0"


def format_date(val: Any, granularity: Optional[str] = None) -> str:
    """'YYYY-MM-DD', or 'YYYY-MM' when granularity == 'month' (questions about a month)."""
    if isinstance(val, (pd.Timestamp, datetime, date)):
        s = val.strftime("%Y-%m-%d")
    else:
        s = str(val).strip()
        m_iso = re.match(r"^(\d{4})-(\d{1,2})(?:-(\d{1,2}))?(?:[ T].*)?$", s)
        m_dmy = re.match(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})$", s)
        m_my = re.match(r"^(\d{1,2})[/.-](\d{4})$", s)
        if m_iso:
            y, mo, d = int(m_iso.group(1)), int(m_iso.group(2)), m_iso.group(3)
            s = f"{y:04d}-{mo:02d}" + (f"-{int(d):02d}" if d else "")
        elif m_dmy:
            s = f"{int(m_dmy.group(3)):04d}-{int(m_dmy.group(2)):02d}-{int(m_dmy.group(1)):02d}"
        elif m_my:
            s = f"{int(m_my.group(2)):04d}-{int(m_my.group(1)):02d}"
        else:
            try:
                s = pd.to_datetime(s, dayfirst=True).strftime("%Y-%m-%d")
            except Exception:
                raise ValueError(f"Cannot parse '{val}' as date")
    if granularity == "month":
        return s[:7]
    if granularity == "day" and len(s) != 10:
        raise ValueError(f"Day-level date expected, got '{s}'")
    return s


def format_boolean(val: Any) -> bool:
    """Format boolean output (True / False)."""
    val = _scalar(val, "boolean")
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)) and val in (0, 1):
        return bool(val)
    s = str(val).strip().lower()
    if s in ("true", "1", "có", "đúng", "yes"):
        return True
    if s in ("false", "0", "không", "sai", "no"):
        return False
    raise ValueError(f"Cannot parse '{val}' as boolean")


def format_string(val: Any) -> str:
    """Names are copied verbatim from the data (only surrounding whitespace is stripped)."""
    val = _scalar(val, "string")
    if val is None or (not isinstance(val, str) and pd.isna(val)):
        raise ValueError("Cannot format NaN as string")
    s = str(val).strip()
    if not s:
        raise ValueError("Empty string answer")
    return s


def _format_cell(item: Any, is_money: bool = False) -> Any:
    if isinstance(item, (bool, np.bool_)):
        return bool(item)
    if isinstance(item, (int, np.integer)):
        return int(item)
    if isinstance(item, (float, np.floating, Decimal)):
        return format_number(item, is_money)
    if isinstance(item, (datetime, date, pd.Timestamp)):
        return format_date(item)
    s = str(item).strip()
    # Numeric text ("12", "3.5") becomes a number, but codes with leading zeros stay text
    if re.fullmatch(r"-?(?:0|[1-9]\d*)(?:\.\d+)?", s):
        return format_number(float(s), is_money)
    return s


def format_list(val: Any, elem_type: Optional[str] = None, is_money: bool = False) -> List[Any]:
    """Format list output: must be a list, even for single element. Missing elements are an error,
    not silently dropped (dropping would shift every later position)."""
    if isinstance(val, (pd.Series, np.ndarray)):
        raw_list = val.tolist()
    elif isinstance(val, (list, tuple)):
        raw_list = list(val)
    elif isinstance(val, set):
        raise ValueError("A set has no order; lists must be ordered")
    else:
        raw_list = [val]
    out = []
    for item in raw_list:
        if item is None or (not isinstance(item, (str, list, dict)) and pd.isna(item)):
            raise ValueError("List contains a missing element")
        out.append(_format_cell(item, is_money))
    if not out:
        raise ValueError("Empty list answer")
    return out


def format_table(val: Any, is_money: bool = False) -> Dict[str, Any]:
    """Format table output: {"columns": [...], "rows": [[...], ...]}."""
    if isinstance(val, pd.DataFrame):
        cols = [str(c).strip() for c in val.columns]
        rows = [[None if (not isinstance(x, str) and pd.isna(x)) else _format_cell(x, is_money) for x in row]
                for row in val.itertuples(index=False, name=None)]
    elif isinstance(val, dict) and "columns" in val and "rows" in val:
        cols = [str(c).strip() for c in val["columns"]]
        rows = [[_format_cell(x, is_money) if x is not None else None for x in row] for row in val["rows"]]
    else:
        raise ValueError(f"Cannot format {type(val)} as table")
    if not rows or any(len(r) != len(cols) for r in rows):
        raise ValueError("Table is empty or has ragged rows")
    return {"columns": cols, "rows": rows}


def format_submission_item(
    question_id: str,
    raw_result: Any,
    expected_type: str,
    reason: Optional[str] = None,
    is_money: bool = False,
    date_granularity: Optional[str] = None,
) -> Dict[str, Any]:
    """Format final prediction item for submission.jsonl.
    Handles 'abstain' and 'unanswerable' as special types. A value that cannot be put in the
    exact shape of its type becomes abstain (wrong format is scored -1, abstain 0).
    """
    # isinstance guard: `DataFrame == "ABSTAIN"` is element-wise and raises in a boolean context
    is_str = isinstance(raw_result, str)
    if expected_type == "abstain" or raw_result is None or (is_str and raw_result == "ABSTAIN"):
        return {"id": question_id, "type": "abstain"}

    if expected_type == "unanswerable" or (is_str and raw_result == "UNANSWERABLE"):
        return {
            "id": question_id,
            "type": "unanswerable",
            "reason": str(reason) if reason else "Không đủ thông tin trong dữ liệu để trả lời câu hỏi"
        }

    try:
        if expected_type == "number":
            ans = format_number(raw_result, is_money)
        elif expected_type == "percent":
            ans = format_percent(raw_result)
        elif expected_type == "date":
            ans = format_date(_scalar(raw_result, "date"), date_granularity)
        elif expected_type == "boolean":
            ans = format_boolean(raw_result)
        elif expected_type == "string":
            ans = format_string(raw_result)
        elif expected_type == "list":
            ans = format_list(raw_result, is_money=is_money)
        elif expected_type == "table":
            ans = format_table(raw_result, is_money)
        else:
            return {"id": question_id, "type": "abstain"}
        return {"id": question_id, "type": expected_type, "answer": ans}
    except Exception:
        # If formatting fails, abstain to protect PA score from wrong format penalty (-1)
        return {"id": question_id, "type": "abstain"}
