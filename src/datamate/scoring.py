"""Answer matching per docs/evaluation.md (the organizer's rules).
Shared by scripts/evaluate.py (offline PA) and the answer voter (decision.py), so that two
candidate answers are "the same" exactly when the organizer would score them the same.
"""

import math
import re
import unicodedata
from typing import Any, Dict, List, Optional

# Only floating-point noise is forgiven ("Chỉ bỏ qua sai số dấu phẩy động", rulebook §9):
# absolute 1e-6 plus a relative 1e-12, so 1 VND off on 89,526,400 is still wrong
FLOAT_EPS = 1e-6
FLOAT_REL_EPS = 1e-12


def normalize_string(s: Any) -> str:
    """NFC -> lowercase -> every char that is not a letter/digit/whitespace becomes a space -> collapse.
    Diacritics are kept. '_' is not a letter or digit, so it becomes a space too."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s)).lower()
    s = re.sub(r"[^\w\s]|_", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _flat_aliases(aliases: Any) -> List[Any]:
    if aliases is None:
        return []
    if not isinstance(aliases, list):
        return [aliases]
    out: List[Any] = []
    for a in aliases:
        out.extend(a if isinstance(a, list) else [a])
    return out


def match_string(predicted: Any, reference: Any, aliases: Any = None) -> bool:
    if not isinstance(predicted, str):
        return False
    p = normalize_string(predicted)
    return p == normalize_string(reference) or any(p == normalize_string(a) for a in _flat_aliases(aliases))


def match_number(predicted: Any, reference: Any) -> bool:
    if not _is_number(predicted) or not _is_number(reference):
        return False
    return abs(float(predicted) - float(reference)) <= FLOAT_EPS + FLOAT_REL_EPS * abs(float(reference))


def match_percent(predicted: Any, reference: Any) -> bool:
    if not _is_number(predicted) or not _is_number(reference):
        return False
    return abs(round(float(predicted), 2) - round(float(reference), 2)) <= FLOAT_EPS


def match_date(predicted: Any, reference: Any) -> bool:
    return isinstance(predicted, str) and isinstance(reference, str) and predicted.strip() == reference.strip()


def match_boolean(predicted: Any, reference: Any) -> bool:
    return isinstance(predicted, bool) and isinstance(reference, bool) and predicted is reference


def match_element(predicted: Any, reference: Any, aliases: Any = None) -> bool:
    """List/table cell: names with the string rule, numbers with the number rule."""
    if isinstance(reference, bool):
        return match_boolean(predicted, reference)
    if _is_number(reference):
        return match_number(predicted, reference)
    if isinstance(reference, str):
        return match_string(predicted, reference, aliases)
    return predicted == reference


def _alias_at(aliases: Any, i: int) -> Any:
    if isinstance(aliases, list) and i < len(aliases):
        return aliases[i]
    return None


def match_list(predicted: Any, reference: Any, aliases: Any = None) -> bool:
    if not isinstance(predicted, list) or not isinstance(reference, list) or len(predicted) != len(reference):
        return False
    return all(match_element(p, r, _alias_at(aliases, i)) for i, (p, r) in enumerate(zip(predicted, reference)))


def _align_columns(pred_cols: List[Any], ref_cols: List[Any]) -> Optional[List[int]]:
    """Index of the predicted column for each reference column: by normalized name, else by position."""
    if len(pred_cols) != len(ref_cols):
        return None
    pred_norm = [normalize_string(c) for c in pred_cols]
    ref_norm = [normalize_string(c) for c in ref_cols]
    if sorted(pred_norm) == sorted(ref_norm) and len(set(ref_norm)) == len(ref_norm):
        return [pred_norm.index(c) for c in ref_norm]
    return list(range(len(ref_cols)))


def match_table(predicted: Any, reference: Any, aliases: Any = None) -> bool:
    """Columns by normalized name (else position); rows compared as a multiset."""
    if not isinstance(predicted, dict) or not isinstance(reference, dict):
        return False
    try:
        pred_cols, pred_rows = list(predicted["columns"]), list(predicted["rows"])
        ref_cols, ref_rows = list(reference["columns"]), list(reference["rows"])
    except (KeyError, TypeError):
        return False
    order = _align_columns(pred_cols, ref_cols)
    if order is None or len(pred_rows) != len(ref_rows):
        return False

    used = set()
    for pred_row in pred_rows:
        if not isinstance(pred_row, list) or len(pred_row) != len(ref_cols):
            return False
        cells = [pred_row[j] for j in order]
        for idx, ref_row in enumerate(ref_rows):
            if idx in used:
                continue
            row_alias = _alias_at(aliases, idx)
            if all(match_element(cells[c], ref_row[c], row_alias if c == 0 else None) for c in range(len(ref_cols))):
                used.add(idx)
                break
        else:
            return False
    return True


def match_answer(predicted: Dict[str, Any], reference: Dict[str, Any]) -> bool:
    """True if `predicted` scores +1 against `reference`. Abstain is never a match."""
    p_type, r_type = predicted.get("type"), reference.get("type")
    if p_type == "abstain" or p_type != r_type:
        return False
    if r_type == "unanswerable":
        return True
    pa, ra, aliases = predicted.get("answer"), reference.get("answer"), reference.get("alias")
    matcher = {
        "number": match_number, "percent": match_percent, "date": match_date, "boolean": match_boolean,
    }.get(r_type)
    if matcher:
        return matcher(pa, ra)
    if r_type == "string":
        return match_string(pa, ra, aliases)
    if r_type == "list":
        return match_list(pa, ra, aliases)
    if r_type == "table":
        return match_table(pa, ra, aliases)
    return False


def same_answer(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Two candidates agree if either, taken as the reference, would score the other as correct."""
    return match_answer(a, b) or match_answer(b, a)


def score(predicted: Optional[Dict[str, Any]], reference: Dict[str, Any]) -> int:
    """+1 correct, 0 abstain / missing, -1 wrong."""
    if predicted is None or predicted.get("type") == "abstain":
        return 0
    return 1 if match_answer(predicted, reference) else -1
