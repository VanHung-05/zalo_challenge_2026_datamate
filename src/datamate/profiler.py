"""Data profiler for DataMate.
Generates schema cards, builds value indices for entity linking, and identifies join keys.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import re
import unicodedata
import pandas as pd
from rapidfuzz import process, fuzz


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    null_pct: float
    n_unique: int
    sample_values: List[Any] = field(default_factory=list)
    min_val: Optional[Any] = None
    max_val: Optional[Any] = None


@dataclass
class SchemaCard:
    table_name: str
    row_count: int
    columns: List[ColumnProfile]
    join_keys: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"### Bảng `{self.table_name}` ({self.row_count:,} dòng)"]
        lines.append("| Cột | Kiểu | % Null | Unique | Giá trị mẫu |")
        lines.append("|---|---|---|---|---|")
        for col in self.columns:
            samples_str = ", ".join([f'"{str(v)}"' for v in col.sample_values[:4]])
            lines.append(f"| `{col.name}` | {col.dtype} | {col.null_pct:.1f}% | {col.n_unique:,} | {samples_str} |")
        if self.join_keys:
            lines.append(f"**Khóa nối tiềm năng**: {', '.join([f'`{k}`' for k in self.join_keys])}")
        return "\n".join(lines)


@dataclass
class ValueIndex:
    """Index of unique string values across columns for entity linking."""
    entries: List[Tuple[str, str, str]] = field(default_factory=list)  # (normalized_val, original_val, table_col)

    def search(self, query: str, top_k: int = 5, score_cutoff: float = 60.0) -> List[Dict[str, Any]]:
        """Search query against value index using fuzzy token sort ratio."""
        if not self.entries or not query:
            return []
        
        q_norm = unicodedata.normalize("NFC", query).lower().strip()
        choices = [e[0] for e in self.entries]
        
        matches = process.extract(
            q_norm,
            choices,
            scorer=fuzz.token_sort_ratio,
            limit=top_k,
            score_cutoff=score_cutoff
        )
        
        results = []
        seen = set()
        for matched_text, score, idx in matches:
            norm_val, orig_val, table_col = self.entries[idx]
            key = (orig_val, table_col)
            if key in seen:
                continue
            seen.add(key)
            tbl, col = table_col.split(".", 1)
            results.append({
                "query": query,
                "matched_value": orig_val,
                "table": tbl,
                "column": col,
                "score": round(score / 100.0, 3)
            })
        return results


def is_potential_join_key(col_name: str) -> bool:
    c = col_name.lower().strip()
    return any(k in c for k in ["mã", "ma_", "code", "_id", "id", "stt", "số phiếu", "so_phieu"])


def profile_table(table_name: str, df: pd.DataFrame) -> SchemaCard:
    """Generate a SchemaCard for a DataFrame."""
    col_profiles = []
    join_keys = []
    n_rows = len(df)

    for col in df.columns:
        s = df[col]
        null_count = s.isna().sum()
        null_pct = (null_count / n_rows * 100.0) if n_rows > 0 else 0.0
        n_unique = s.nunique(dropna=True)

        non_null_s = s.dropna()
        samples = non_null_s.head(5).tolist()

        min_val = None
        max_val = None
        if pd.api.types.is_numeric_dtype(s) and not non_null_s.empty:
            min_val = non_null_s.min()
            max_val = non_null_s.max()

        dtype_str = str(s.dtype)
        if "datetime" in dtype_str:
            dtype_str = "datetime"
        elif "int" in dtype_str:
            dtype_str = "int"
        elif "float" in dtype_str:
            dtype_str = "float"
        elif "object" in dtype_str:
            dtype_str = "string"

        cp = ColumnProfile(
            name=str(col),
            dtype=dtype_str,
            null_pct=null_pct,
            n_unique=n_unique,
            sample_values=samples,
            min_val=min_val,
            max_val=max_val
        )
        col_profiles.append(cp)

        if is_potential_join_key(str(col)):
            join_keys.append(str(col))

    return SchemaCard(
        table_name=table_name,
        row_count=n_rows,
        columns=col_profiles,
        join_keys=join_keys
    )


def build_value_index(tables: Dict[str, pd.DataFrame]) -> ValueIndex:
    """Build value index across text columns (products, categories, customers, etc.)."""
    entries = []
    seen_entries = set()
    candidate_keywords = ["tên", "ten", "sản phẩm", "san_pham", "khách", "khach", "nhóm", "nhom", "mặt hàng", "loại", "thợ", "tho", "nhân viên", "nhan_vien", "nội dung"]

    # If unified BanHang exists, skip the individual monthly slices
    has_unified = "BanHang" in tables

    for tbl_name, df in tables.items():
        if has_unified and tbl_name != "BanHang" and re.search(r"^(?:BanHang|DonHang|So_chi_tiet_ban_hang)_T", tbl_name, re.IGNORECASE):
            continue

        for col in df.columns:
            col_lower = str(col).lower()
            if any(k in col_lower for k in candidate_keywords) and not is_potential_join_key(str(col)):
                try:
                    unique_vals = df[col].dropna().unique()
                    for val in unique_vals:
                        v_str = str(val).strip()
                        if not v_str or len(v_str) > 80:
                            continue
                        v_norm = unicodedata.normalize("NFC", v_str).lower()
                        if v_norm not in seen_entries:
                            seen_entries.add(v_norm)
                            entries.append((v_norm, v_str, f"{tbl_name}.{col}"))
                except Exception:
                    continue

    return ValueIndex(entries=entries)


def profile_business(tables: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, SchemaCard], ValueIndex]:
    """Profile all tables of a business and return schema cards and value index."""
    cards = {}
    for tbl_name, df in tables.items():
        cards[tbl_name] = profile_table(tbl_name, df)
    v_index = build_value_index(tables)
    return cards, v_index
