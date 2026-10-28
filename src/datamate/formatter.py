"""Answer formatter for DataMate.
Ensures outputs strictly adhere to rulebook §9 format requirements.
"""

import math
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


def clean_number_str(val_str: str) -> Optional[float]:
    """Parse string with potential Vietnamese number formatting into float."""
    s = val_str.strip()
    # Remove currency / percent suffix
    s = re.sub(r"(đ|vnđ|vnd|%|\$)\s*$", "", s, flags=re.IGNORECASE).strip()
    # Handle Vietnamese notation like "4.820.000" or "4,820,000"
    if re.match(r"^\d{1,3}(\.\d{3})+(,\d+)?$", s):
        # dot as thousand, comma as decimal
        s = s.replace(".", "").replace(",", ".")
    elif re.match(r"^\d{1,3}(,\d{3})+(\.\d+)?$", s):
        # comma as thousand, dot as decimal
        s = s.replace(",", "")
    elif "," in s and "." not in s:
        # e.g. "12,4" -> "12.4"
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def format_number(val: Any, is_money: bool = False) -> Union[int, float]:
    """Format numeric output:
    - If money or integer count: return int
    - If average of money: return int
    - If other float: round to 2 decimals (or integer if is_integer)
    """
    if isinstance(val, (pd.Series, np.ndarray, list)):
        if len(val) == 1:
            val = val[0]
        else:
            raise ValueError(f"Cannot format list/series of length {len(val)} as single number")

    if pd.isna(val):
        raise ValueError("Cannot format NaN as number")

    if isinstance(val, str):
        parsed = clean_number_str(val)
        if parsed is not None:
            val = parsed
        else:
            raise ValueError(f"Cannot parse '{val}' as number")

    num = float(val)
    if is_money or num.is_integer():
        return int(round(num))
    return round(num, 2)


def format_percent(val: Any) -> float:
    """Format percent output:
    - Must be percentage value (e.g. 12.4, NOT 0.124)
    - If value <= 1.0 and user returned decimal ratio like 0.124, convert to 12.4
    - Round to 2 decimals
    """
    if isinstance(val, (pd.Series, np.ndarray, list)):
        if len(val) == 1:
            val = val[0]
        else:
            raise ValueError(f"Cannot format list/series as single percent")

    if pd.isna(val):
        raise ValueError("Cannot format NaN as percent")

    if isinstance(val, str):
        cleaned = val.replace("%", "").strip()
        parsed = clean_number_str(cleaned)
        if parsed is not None:
            val = parsed
        else:
            raise ValueError(f"Cannot parse '{val}' as percent")

    num = float(val)
    # If returned as ratio 0.0527 instead of 5.27, convert
    # Note: If ratio is between 0 and 1, check if question asked for percent
    if 0.0 < abs(num) < 1.0:
        num = num * 100.0

    return round(num, 2)


def format_date(val: Any) -> str:
    """Format date output as YYYY-MM-DD or YYYY-MM."""
    if isinstance(val, (pd.Timestamp, datetime, date)):
        return val.strftime("%Y-%m-%d")
    s = str(val).strip()
    # Check if YYYY-MM-DD or YYYY-MM
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s) or re.match(r"^\d{4}-\d{2}$", s):
        return s
    # Try parsing common Vietnamese date formats: dd/mm/yyyy
    m = re.match(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})$", s)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{year:04d}-{month:02d}-{day:02d}"
    # Try monthly mm/yyyy
    m = re.match(r"^(\d{1,2})[/.-](\d{4})$", s)
    if m:
        month, year = int(m.group(1)), int(m.group(2))
        return f"{year:04d}-{month:02d}"
    # Fallback to pandas to_datetime
    try:
        ts = pd.to_datetime(s)
        return ts.strftime("%Y-%m-%d")
    except Exception:
        return s


def format_boolean(val: Any) -> bool:
    """Format boolean output (True / False)."""
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("true", "1", "có", "đúng", "yes"):
        return True
    if s in ("false", "0", "không", "sai", "no"):
        return False
    return bool(val)


def format_string(val: Any) -> str:
    """Format string output preserving original Vietnamese accents."""
    if pd.isna(val):
        return ""
    return str(val).strip()


def format_list(val: Any, elem_type: Optional[str] = None) -> List[Any]:
    """Format list output: must be a list, even for single element."""
    if isinstance(val, (pd.Series, np.ndarray)):
        raw_list = val.tolist()
    elif isinstance(val, list):
        raw_list = val
    elif isinstance(val, (set, tuple)):
        raw_list = list(val)
    else:
        raw_list = [val]

    formatted = []
    for item in raw_list:
        if pd.isna(item):
            continue
        if isinstance(item, (int, np.integer)):
            formatted.append(int(item))
        elif isinstance(item, (float, np.floating)):
            if float(item).is_integer():
                formatted.append(int(item))
            else:
                formatted.append(round(float(item), 2))
        elif isinstance(item, (datetime, date, pd.Timestamp)):
            formatted.append(format_date(item))
        elif isinstance(item, bool):
            formatted.append(item)
        else:
            # String or other
            s = str(item).strip()
            # Try to see if it's actually an integer
            try:
                num = float(s)
                if num.is_integer():
                    formatted.append(int(num))
                else:
                    formatted.append(round(num, 2))
            except ValueError:
                formatted.append(s)
    return formatted


def format_table(val: Any) -> Dict[str, Any]:
    """Format table output: {"columns": [...], "rows": [[...], ...]}."""
    if isinstance(val, pd.DataFrame):
        cols = [str(c).strip() for c in val.columns]
        rows = []
        for _, row in val.iterrows():
            formatted_row = []
            for item in row:
                if pd.isna(item):
                    formatted_row.append(None)
                elif isinstance(item, (int, np.integer)):
                    formatted_row.append(int(item))
                elif isinstance(item, (float, np.floating)):
                    if float(item).is_integer():
                        formatted_row.append(int(item))
                    else:
                        formatted_row.append(round(float(item), 2))
                else:
                    formatted_row.append(str(item).strip())
            rows.append(formatted_row)
        return {"columns": cols, "rows": rows}
    elif isinstance(val, dict) and "columns" in val and "rows" in val:
        return val
    raise ValueError(f"Cannot format {type(val)} as table")


def format_submission_item(
    question_id: str,
    raw_result: Any,
    expected_type: str,
    reason: Optional[str] = None
) -> Dict[str, Any]:
    """Format final prediction item for submission.jsonl.
    Handles 'abstain' and 'unanswerable' as special types.
    """
    # isinstance guard: `DataFrame == "ABSTAIN"` is element-wise and raises in a boolean context
    is_str = isinstance(raw_result, str)
    if expected_type == "abstain" or raw_result is None or (is_str and raw_result == "ABSTAIN"):
        return {"id": question_id, "type": "abstain"}

    if expected_type == "unanswerable" or (is_str and raw_result == "UNANSWERABLE"):
        return {
            "id": question_id,
            "type": "unanswerable",
            "reason": reason or "Không đủ thông tin trong dữ liệu để trả lời câu hỏi"
        }

    try:
        if expected_type == "number":
            ans = format_number(raw_result)
        elif expected_type == "percent":
            ans = format_percent(raw_result)
        elif expected_type == "date":
            ans = format_date(raw_result)
        elif expected_type == "boolean":
            ans = format_boolean(raw_result)
        elif expected_type == "string":
            ans = format_string(raw_result)
        elif expected_type == "list":
            ans = format_list(raw_result)
        elif expected_type == "table":
            ans = format_table(raw_result)
        else:
            ans = raw_result

        return {"id": question_id, "type": expected_type, "answer": ans}
    except Exception as e:
        # If formatting fails, abstain to protect PA score from wrong format penalty (-1)
        return {"id": question_id, "type": "abstain"}
